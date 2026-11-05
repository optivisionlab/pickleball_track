"""UCF-101 feature extraction v2 (stage 1): every clip is decoded once and encoded with

    VideoMAE x2 backbones : ssl  = videomae-base (common/pretrained)       (K400, no labels)
                            k400 = MCG-NJU/videomae-base-finetuned-kinetics (K400 labels)
        32 frames over the whole clip, run as 2 x 16-frame clips (the length the
        checkpoints were trained with) -> 16 temporal tokens; per token the global mean
        plus a 2x2 spatial grid -> [16, 5, 768].
    CLIP image ViT-B/16   : one image embedding per temporal token (frame 2t) -> [16, 512]
    BEATs x2              : base = iter3+ AS2M pretrained, ft = iter3+ fine-tuned on AS2M
                            (predictor removed, hidden states) -> 32 pooled tokens.

Views (VIEWS below): full frame, its flip, left/center/right square crops, and the full
frame sampled half a stride later. Views are train-time augmentation and are averaged at
test time.

Output (memmap .npy, resumable): <out>/video_{ssl,k400}.npy [V,N,16,5,768] f16,
clip_b16.npy [V,N,16,512] f16, audio_{base,ft}.npy [N,32,768] f16, pad.npy [V,N,16] bool,
has_audio.npy [N] bool, done.npy [N] bool, meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as cfg  # noqa: E402
import ucf101_common as uc  # noqa: E402

bc = uc.bc
VIEWS = ["full", "flip", "crop_left", "crop_center", "crop_right", "full_shift"]
NUM_FRAMES, TOKENS, SIZE, GRID = 32, 16, 224, 2
AUDIO_TOKENS = 32
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
BACKBONES = {"ssl": cfg.VIDEOMAE, "k400": "MCG-NJU/videomae-base-finetuned-kinetics"}
BEATS = {"base": cfg.BEATS_CHECKPOINT,
         "ft": cfg.BEATS_FT_CHECKPOINT}
CLIP_IMAGE = "openai/clip-vit-base-patch16"


def frame_sets(total: int):
    """Two sets of 32 indices: uniform over the clip, and the same shifted by half a stride."""
    base = np.linspace(0, max(total - 1, 0), NUM_FRAMES)
    stride = (total - 1) / max(NUM_FRAMES - 1, 1) if total > 1 else 0
    shift = np.minimum(base + stride / 2, max(total - 1, 0))
    return np.stack([base, shift]).round().astype(np.int64)


def read_native(path: str):
    """uint8 [2, 32, H, W, 3] at native resolution for both frame sets, plus validity [2, 32]."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sets = frame_sets(total)
    wanted = sorted(set(sets.ravel().tolist()))
    got, pos, last = {}, -1, None
    for target in wanted:
        while pos < target and cap.grab():
            pos += 1
        if pos == target:
            ok, frame = cap.retrieve()
            if ok:
                last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                got[target] = last
    cap.release()
    if last is None:
        return np.zeros((2, NUM_FRAMES, 240, 320, 3), np.uint8), np.zeros((2, NUM_FRAMES), bool)
    h, w = last.shape[:2]
    frames = np.zeros((2, NUM_FRAMES, h, w, 3), np.uint8)
    valid = np.zeros((2, NUM_FRAMES), bool)
    for s in range(2):
        prev = None
        for i, t in enumerate(sets[s]):
            f = got.get(int(t))
            if f is not None and f.shape[:2] == (h, w):
                prev, valid[s, i] = f, True
            frames[s, i] = prev if prev is not None else last
    return frames, valid


class DecodeDataset(Dataset):
    def __init__(self, rows, indices):
        self.rows, self.indices = rows, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        index = int(self.indices[i])
        path = self.rows[index].path
        frames, valid = read_native(path)
        return index, torch.from_numpy(frames), torch.from_numpy(valid), torch.from_numpy(bc.read_audio(path))


def make_views(frames: torch.Tensor) -> torch.Tensor:
    """frames uint8 [2,32,H,W,3] (GPU) -> float [V,32,3,224,224] in [0,1]."""
    x = frames.permute(0, 1, 4, 2, 3).float().div_(255.0)  # [2,32,3,H,W]
    h, w = x.shape[-2:]
    side = min(h, w)

    def resize(t):
        return F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)

    full = resize(x[0])
    top = (h - side) // 2
    crops = [resize(x[0][..., top:top + side, left:left + side])
             for left in (0, (w - side) // 2, w - side)]
    return torch.stack([full, full.flip(-1), *crops, resize(x[1])])


def load_beats_safe(path: str, device):
    for extra in (cfg.BEATS_ROOT, os.path.join(cfg.BEATS_ROOT, "beats")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    from beats.BEATs import BEATs, BEATsConfig

    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # tensors only, no code
    model = BEATs(BEATsConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model.predictor = None  # hidden states instead of AudioSet logits
    return model.to(device).eval().requires_grad_(False)


def videomae_tokens(model, pixels: torch.Tensor, chunk: int) -> torch.Tensor:
    """pixels [M,32,3,224,224] normalised -> [M,16,5,768]: 2 x 16-frame clips, mean + 2x2 grid."""
    m = pixels.shape[0]
    clips = pixels.reshape(m * 2, 16, *pixels.shape[2:])
    outs = []
    for s in range(0, clips.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = model(pixel_values=clips[s:s + chunk]).last_hidden_state
        outs.append(tok.float())
    tok = torch.cat(outs)  # [2M, 8*196, 768]
    hidden = tok.shape[-1]
    side = int(round((tok.shape[1] // 8) ** 0.5))
    grid = tok.reshape(m, TOKENS, side, side, hidden)
    pooled = F.adaptive_avg_pool2d(grid.permute(0, 1, 4, 2, 3).reshape(-1, hidden, side, side), GRID)
    pooled = pooled.reshape(m, TOKENS, hidden, GRID * GRID).permute(0, 1, 3, 2)
    return torch.cat([grid.mean((2, 3)).unsqueeze(2), pooled], dim=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=cfg.FEATURES)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--chunk", type=int, default=8, help="16-frame clips per VideoMAE forward")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N clips")
    args = p.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    rows, classes, splits, audible = uc.load_dataset(cfg.VIDEO_ROOT, cfg.SPLIT_DIR, cfg.ANALYSIS_CSV)
    n, v = len(rows), len(VIEWS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def mm(name, shape, dtype):
        path = out / name
        if path.is_file():
            arr = np.load(path, mmap_mode="r+")
            if arr.shape == shape:
                return arr
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    video = {k: mm(f"video_{k}.npy", (v, n, TOKENS, 1 + GRID * GRID, 768), np.float16) for k in BACKBONES}
    clip_img = mm("clip_b16.npy", (v, n, TOKENS, 512), np.float16)
    audio = {k: mm(f"audio_{k}.npy", (n, AUDIO_TOKENS, 768), np.float16) for k in BEATS}
    pad = mm("pad.npy", (v, n, TOKENS), np.bool_)
    has_audio = mm("has_audio.npy", (n,), np.bool_)
    done = mm("done.npy", (n,), np.bool_)
    meta = {"video_ids": [r.video_id for r in rows], "labels": [r.label for r in rows],
            "games": [r.game_id for r in rows], "classes": classes, "views": VIEWS,
            "backbones": BACKBONES, "beats": BEATS, "clip_image": CLIP_IMAGE, "complete": False}
    todo = np.nonzero(~np.asarray(done))[0]
    if args.limit:
        todo = todo[todo < args.limit]
    print(f"[FEATURES v2] {n} clips, {len(todo)} to do -> {out}", flush=True)
    if len(todo):
        from transformers import VideoMAEModel
        models = {k: VideoMAEModel.from_pretrained(name).to(device).eval().requires_grad_(False)
                  for k, name in BACKBONES.items()}
        for m in models.values():
            if m.config.num_frames != 16:
                raise RuntimeError("expected 16-frame VideoMAE checkpoints")
        beats = {k: load_beats_safe(path, device) for k, path in BEATS.items()}
        from transformers import CLIPVisionModelWithProjection
        clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_IMAGE).to(device).eval()
        vm_mean = torch.tensor(bc.IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        vm_std = torch.tensor(bc.IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
        cl_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        cl_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

        loader = DataLoader(DecodeDataset(rows, todo), batch_size=1, num_workers=args.workers,
                            collate_fn=lambda b: b[0], prefetch_factor=8)
        start, count = time.time(), 0
        with torch.inference_mode():
            for index, frames, valid, wave in tqdm(loader, total=len(todo), desc="extract v2", mininterval=30):
                views = make_views(frames.to(device, non_blocking=True))  # [V,32,3,224,224]
                for key, model in models.items():
                    tok = videomae_tokens(model, (views - vm_mean) / vm_std, args.chunk)
                    video[key][:, index] = tok.half().cpu().numpy()
                imgs = views[:, ::2].reshape(-1, 3, SIZE, SIZE)  # frame 2t per token
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb = clip_model(pixel_values=(imgs - cl_mean) / cl_std).image_embeds
                clip_img[:, index] = F.normalize(emb.float(), dim=-1).reshape(v, TOKENS, -1).half().cpu().numpy()
                tok_valid = valid.reshape(2, TOKENS, 2).any(-1).numpy()
                pad[:, index] = ~tok_valid[[0, 0, 0, 0, 0, 1]]
                length = wave.numel()
                if length >= bc.AUDIO_SR // 4:
                    wave = wave.to(device)
                    chunk = 30 * bc.AUDIO_SR
                    parts = [wave[s:s + chunk] for s in range(0, length, chunk) if length - s >= bc.AUDIO_SR // 4]
                    for key, model in beats.items():
                        feats = torch.cat([model.extract_features(q[None])[0][0] for q in parts])
                        pooled, ok = bc.pool_tokens(feats, torch.ones(len(feats), dtype=torch.bool,
                                                                      device=device), AUDIO_TOKENS)
                        audio[key][index] = pooled.half().cpu().numpy()
                    has_audio[index] = True
                done[index] = True
                count += 1
                if count % 500 == 0:
                    for arr in (*video.values(), clip_img, *audio.values(), pad, has_audio, done):
                        arr.flush()
                    rate = count / (time.time() - start)
                    print(f"[FEATURES v2] {count}/{len(todo)} clips, {rate:.2f} clips/s", flush=True)
    for arr in (*video.values(), clip_img, *audio.values(), pad, has_audio, done):
        arr.flush()
    meta["complete"] = bool(np.asarray(done).all())
    with open(out / "meta.json", "w") as h:
        json.dump(meta, h)
    print(f"[FEATURES v2] complete={meta['complete']} clips with audio={int(np.asarray(has_audio).sum())}")


if __name__ == "__main__":
    main()
