"""NSVA v2 feature extraction: every clip is decoded once and encoded with

    VideoMAE x2 : ssl  = MCG-NJU/videomae-base                    (self-supervised only)
                  k400 = MCG-NJU/videomae-base-finetuned-kinetics (Kinetics labels)
        32 frames run as 2 x 16-frame clips (the checkpoints' length) -> 16 temporal tokens;
        per token the global mean plus a 2x2 spatial grid -> [16, 5, 768].
    CLIP image ViT-B/16 : one embedding per temporal token -> [16, 512].
    BEATs x2 : base = iter3+ AS2M pretrained, ft = fine-tuned on AS2M -> 32 pooled tokens.

Views (nsva_common.VIEWS):
    0-4  full frame, its flip, left/center/right square crops   (32 frames over the clip)
    5    full frame, sampled half a stride later
    6-7  full frame and its flip, 32 frames over the SECOND HALF of the clip only
         (twice the temporal density where the shot outcome happens)

Output (memmap .npy, resumable via done.npy) in --out:
    video_{ssl,k400}.npy [V,N,16,5,768] f16, clip_b16.npy [V,N,16,512] f16,
    audio_{base,ft}.npy [N,32,768] f16, pad.npy [V,N,16] bool, has_audio.npy [N],
    done.npy [N], meta.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsva_common as nc  # noqa: E402

bc = nc.bc
NUM_FRAMES, TOKENS, SIZE, GRID, SHORT_SIDE = 32, 16, 224, 2, 256
AUDIO_TOKENS = 32
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def frame_sets(total: int) -> np.ndarray:
    """[3, 32] frame indices: whole clip, whole clip shifted half a stride, second half."""
    last = max(total - 1, 0)
    base = np.linspace(0, last, NUM_FRAMES)
    stride = last / max(NUM_FRAMES - 1, 1)
    shift = np.minimum(base + stride / 2, last)
    tail = np.linspace(last / 2, last, NUM_FRAMES)
    return np.stack([base, shift, tail]).round().astype(np.int64)


def read_sets(path: str):
    """uint8 [3, 32, 256, W, 3] (short side 256) plus validity [3, 32]."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sets = frame_sets(total)
    wanted = set(sets.ravel().tolist())
    got, pos, size = {}, -1, None
    for target in sorted(wanted):
        while pos < target and cap.grab():
            pos += 1
        if pos != target:
            break
        ok, frame = cap.retrieve()
        if not ok:
            continue
        if size is None:
            h, w = frame.shape[:2]
            size = (int(round(w * SHORT_SIDE / min(h, w))), int(round(h * SHORT_SIDE / min(h, w))))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        got[target] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    if not got:
        return np.zeros((3, NUM_FRAMES, SHORT_SIDE, 455, 3), np.uint8), np.zeros((3, NUM_FRAMES), bool)
    any_frame = next(iter(got.values()))
    frames = np.zeros((3, NUM_FRAMES, *any_frame.shape), np.uint8)
    valid = np.zeros((3, NUM_FRAMES), bool)
    for s in range(3):
        prev = None
        for i, t in enumerate(sets[s]):
            f = got.get(int(t))
            if f is not None:
                prev, valid[s, i] = f, True
            frames[s, i] = prev if prev is not None else any_frame
    return frames, valid


class DecodeDataset(Dataset):
    def __init__(self, rows, members, indices):
        self.rows, self.members, self.indices = rows, members, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        index = int(self.indices[i])
        with nc.ZipClip(self.members, self.rows[index].path) as path:
            frames, valid = read_sets(path)
            wave = bc.read_audio(path)
        return index, torch.from_numpy(frames), torch.from_numpy(valid), torch.from_numpy(wave)


def make_views(frames: torch.Tensor) -> torch.Tensor:
    """frames uint8 [3,32,H,W,3] (GPU) -> float [8,32,3,224,224] in [0,1]."""
    x = frames.permute(0, 1, 4, 2, 3).float().div_(255.0)  # [3,32,3,H,W]
    h, w = x.shape[-2:]
    side = min(h, w)

    def resize(t):
        return F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)

    full = resize(x[0])
    top = (h - side) // 2
    crops = [resize(x[0][..., top:top + side, left:left + side]) for left in (0, (w - side) // 2, w - side)]
    tail = resize(x[2])
    return torch.stack([full, full.flip(-1), *crops, resize(x[1]), tail, tail.flip(-1)])


def load_beats(path: str, device):
    for extra in (nc.BEATS_ROOT, os.path.join(nc.BEATS_ROOT, "beats")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    from beats.BEATs import BEATs, BEATsConfig

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model.predictor = None  # hidden states, not AudioSet logits
    return model.to(device).eval().requires_grad_(False)


def videomae_tokens(model, pixels: torch.Tensor, chunk: int) -> torch.Tensor:
    """pixels [M,32,3,224,224] normalised -> [M,16,5,768]: 2 x 16-frame clips, mean + 2x2 grid."""
    m = pixels.shape[0]
    clips = pixels.reshape(m * 2, 16, *pixels.shape[2:])
    outs = []
    for s in range(0, clips.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(model(pixel_values=clips[s:s + chunk]).last_hidden_state.float())
    tok = torch.cat(outs)  # [2M, 8*196, 768]
    hidden = tok.shape[-1]
    side = int(round((tok.shape[1] // 8) ** 0.5))
    grid = tok.reshape(m, TOKENS, side, side, hidden)
    pooled = F.adaptive_avg_pool2d(grid.permute(0, 1, 4, 2, 3).reshape(-1, hidden, side, side), GRID)
    pooled = pooled.reshape(m, TOKENS, hidden, GRID * GRID).permute(0, 1, 3, 2)
    return torch.cat([grid.mean((2, 3)).unsqueeze(2), pooled], dim=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=nc.FEATURES)
    p.add_argument("--chunk", type=int, default=8, help="16-frame clips per VideoMAE forward")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N clips")
    args = p.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    rows, members = nc.load_rows()
    n, v = len(rows), len(nc.VIEWS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def mm(name, shape, dtype):
        path = out / name
        if path.is_file():
            arr = np.load(path, mmap_mode="r+")
            if arr.shape == shape:
                return arr
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    video = {k: mm(f"video_{k}.npy", (v, n, TOKENS, 1 + GRID * GRID, 768), np.float16) for k in nc.BACKBONES}
    clip_img = mm("clip_b16.npy", (v, n, TOKENS, 512), np.float16)
    audio = {k: mm(f"audio_{k}.npy", (n, AUDIO_TOKENS, 768), np.float16) for k in nc.BEATS}
    pad = mm("pad.npy", (v, n, TOKENS), np.bool_)
    has_audio = mm("has_audio.npy", (n,), np.bool_)
    done = mm("done.npy", (n,), np.bool_)
    meta = {"video_ids": [r.video_id for r in rows], "files": [r.path for r in rows],
            "labels": [r.label for r in rows], "games": [r.game_id for r in rows],
            "views": nc.VIEWS, "backbones": nc.BACKBONES, "beats": nc.BEATS,
            "clip_image": nc.CLIP_IMAGE, "complete": False}
    todo = np.nonzero(~np.asarray(done))[0]
    if args.limit:
        todo = todo[todo < args.limit]
    print(f"[FEATURES] {n} clips, {len(todo)} to do -> {out}", flush=True)
    if len(todo):
        from transformers import CLIPVisionModelWithProjection, VideoMAEModel

        models = {k: VideoMAEModel.from_pretrained(name).to(device).eval().requires_grad_(False)
                  for k, name in nc.BACKBONES.items()}
        for m in models.values():
            if m.config.num_frames != 16:
                raise RuntimeError("expected 16-frame VideoMAE checkpoints")
        beats = {k: load_beats(path, device) for k, path in nc.BEATS.items()}
        clip_model = CLIPVisionModelWithProjection.from_pretrained(nc.CLIP_IMAGE).to(device).eval()
        vm_mean = torch.tensor(bc.IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        vm_std = torch.tensor(bc.IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
        cl_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        cl_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        set_of_view = [0, 0, 0, 0, 0, 1, 2, 2]

        loader = DataLoader(DecodeDataset(rows, members, todo), batch_size=1, num_workers=args.workers,
                            collate_fn=lambda b: b[0], prefetch_factor=4)
        start, count = time.time(), 0
        with torch.inference_mode():
            for index, frames, valid, wave in tqdm(loader, total=len(todo), desc="extract", mininterval=60):
                views = make_views(frames.to(device, non_blocking=True))
                for key, model in models.items():
                    tok = videomae_tokens(model, (views - vm_mean) / vm_std, args.chunk)
                    video[key][:, index] = tok.half().cpu().numpy()
                imgs = views[:, ::2].reshape(-1, 3, SIZE, SIZE)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb = clip_model(pixel_values=(imgs - cl_mean) / cl_std).image_embeds
                clip_img[:, index] = F.normalize(emb.float(), dim=-1).reshape(v, TOKENS, -1).half().cpu().numpy()
                tok_valid = valid.reshape(3, TOKENS, 2).any(-1).numpy()
                pad[:, index] = ~tok_valid[set_of_view]
                length = wave.numel()
                if length >= bc.AUDIO_SR // 4:
                    wave = wave.to(device)
                    step = 30 * bc.AUDIO_SR
                    parts = [wave[s:s + step] for s in range(0, length, step) if length - s >= bc.AUDIO_SR // 4]
                    for key, model in beats.items():
                        feats = torch.cat([model.extract_features(q[None])[0][0] for q in parts])
                        pooled, _ = bc.pool_tokens(feats, torch.ones(len(feats), dtype=torch.bool,
                                                                     device=device), AUDIO_TOKENS)
                        audio[key][index] = pooled.half().cpu().numpy()
                    has_audio[index] = True
                done[index] = True
                count += 1
                if count % 500 == 0:
                    for arr in (*video.values(), clip_img, *audio.values(), pad, has_audio, done):
                        arr.flush()
                    print(f"[FEATURES] {count}/{len(todo)} clips, {count / (time.time() - start):.2f} clips/s",
                          flush=True)
    for arr in (*video.values(), clip_img, *audio.values(), pad, has_audio, done):
        arr.flush()
    meta["complete"] = bool(np.asarray(done).all())
    with open(out / "meta.json", "w") as h:
        json.dump(meta, h)
    print(f"[FEATURES] complete={meta['complete']} clips with audio={int(np.asarray(has_audio).sum())}")


if __name__ == "__main__":
    main()
