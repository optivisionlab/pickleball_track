"""Football feature extraction: every clip is decoded once and encoded with

    VideoMAE k400 (MCG-NJU/videomae-base-finetuned-kinetics): 32 frames run as 2 x 16-frame
        clips -> 16 temporal tokens; per token the global mean plus a 2x2 spatial grid -> [16, 5, 768]
    VideoMAE ssl  (MCG-NJU/videomae-base, self-supervised only): global mean only -> [16, 768]
        (the grid is stored for k400 only: the disk has ~20 GB free)
    CLIP image ViT-B/16 : one embedding per temporal token -> [16, 512]
    BEATs base / ft     : 15 s of audio -> 32 pooled tokens [32, 768]

Views (fb_common.VIEWS), all resized (squashed) to 224 x 224:
    0 full        32 frames over the whole 15 s clip
    1 full_flip   its horizontal flip
    2 focus       32 frames over [3 s, 11 s] (shot at 5 s + outcome), 2x temporal density
    3 focus_flip  its horizontal flip

Output (memmap .npy, resumable via done.npy) in --out:
    video_k400.npy [4,N,16,5,768] f16, video_ssl.npy [4,N,16,768] f16, clip_b16.npy [4,N,16,512] f16,
    audio_{base,ft}.npy [N,32,768] f16, pad.npy [4,N,16] bool, has_audio.npy [N], done.npy [N], meta.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb_common as fc  # noqa: E402

bc = fc.bc
NUM_FRAMES, TOKENS, SIZE, GRID, SHORT_SIDE = 32, 16, 224, 2, 256
AUDIO_TOKENS = 32
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def frame_sets(total: int, fps: float) -> np.ndarray:
    """[2, 32] frame indices: whole clip, focus window."""
    last = max(total - 1, 0)
    full = np.linspace(0, last, NUM_FRAMES)
    a, b = (min(s * fps, last) for s in fc.FOCUS_SECONDS)
    focus = np.linspace(a, b, NUM_FRAMES)
    return np.stack([full, focus]).round().astype(np.int64)


def read_sets(path: str):
    """uint8 [2, 32, 224, 224, 3] plus validity [2, 32]."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sets = frame_sets(total, fps)
    got, pos = {}, -1
    for target in sorted(set(sets.ravel().tolist())):
        while pos < target and cap.grab():
            pos += 1
        if pos != target:
            break
        ok, frame = cap.retrieve()
        if ok:
            frame = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            got[target] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    frames = np.zeros((2, NUM_FRAMES, SIZE, SIZE, 3), np.uint8)
    valid = np.zeros((2, NUM_FRAMES), bool)
    if not got:
        return frames, valid
    any_frame = next(iter(got.values()))
    for s in range(2):
        prev = None
        for i, t in enumerate(sets[s]):
            f = got.get(int(t))
            if f is not None:
                prev, valid[s, i] = f, True
            frames[s, i] = prev if prev is not None else any_frame
    return frames, valid


class DecodeDataset(Dataset):
    def __init__(self, clips, indices):
        self.clips, self.indices = clips, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        index = int(self.indices[i])
        frames, valid = read_sets(self.clips[index].path)
        wave = bc.read_audio(self.clips[index].path)
        return index, torch.from_numpy(frames), torch.from_numpy(valid), torch.from_numpy(wave)


def make_views(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [B,2,32,224,224,3] (GPU) -> float [B,4,32,3,224,224] in [0,1]."""
    x = frames.permute(0, 1, 2, 5, 3, 4).float().div_(255.0)
    return torch.stack([x[:, 0], x[:, 0].flip(-1), x[:, 1], x[:, 1].flip(-1)], 1)


def load_beats(path: str, device):
    for extra in (fc.BEATS_ROOT, os.path.join(fc.BEATS_ROOT, "beats")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    from beats.BEATs import BEATs, BEATsConfig

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model.predictor = None
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


def collate(batch):
    idx, frames, valid, waves = zip(*batch)
    return list(idx), torch.stack(frames), torch.stack(valid), list(waves)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=fc.FEATURES)
    p.add_argument("--batch", type=int, default=4, help="clips per GPU step")
    p.add_argument("--chunk", type=int, default=32, help="16-frame clips per VideoMAE forward")
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N clips")
    args = p.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    clips = fc.load_clips()
    n, v = len(clips), len(fc.VIEWS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def mm(name, shape, dtype):
        path = out / name
        if path.is_file():
            arr = np.load(path, mmap_mode="r+")
            if arr.shape == shape:
                return arr
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    vk = mm("video_k400.npy", (v, n, TOKENS, 1 + GRID * GRID, 768), np.float16)
    vs = mm("video_ssl.npy", (v, n, TOKENS, 768), np.float16)
    clip_img = mm("clip_b16.npy", (v, n, TOKENS, 512), np.float16)
    audio = {k: mm(f"audio_{k}.npy", (n, AUDIO_TOKENS, 768), np.float16) for k in fc.BEATS}
    pad = mm("pad.npy", (v, n, TOKENS), np.bool_)
    has_audio = mm("has_audio.npy", (n,), np.bool_)
    done = mm("done.npy", (n,), np.bool_)
    arrays = (vk, vs, clip_img, *audio.values(), pad, has_audio, done)
    meta = {"paths": [c.info["path"] for c in clips], "labels": [c.label for c in clips],
            "splits": [c.split for c in clips], "games": [c.game_idx for c in clips],
            "label_names": list(fc.LABELS), "views": fc.VIEWS, "focus_seconds": fc.FOCUS_SECONDS,
            "backbones": fc.BACKBONES, "beats": fc.BEATS, "clip_image": fc.CLIP_IMAGE, "complete": False}
    todo = np.nonzero(~np.asarray(done))[0]
    if args.limit:
        todo = todo[todo < args.limit]
    print(f"[FEATURES] {n} clips, {len(todo)} to do -> {out}", flush=True)
    if len(todo):
        from transformers import CLIPVisionModelWithProjection, VideoMAEModel

        models = {k: VideoMAEModel.from_pretrained(name).to(device).eval().requires_grad_(False)
                  for k, name in fc.BACKBONES.items()}
        beats = {k: load_beats(path, device) for k, path in fc.BEATS.items()}
        clip_model = CLIPVisionModelWithProjection.from_pretrained(fc.CLIP_IMAGE).to(device).eval()
        vm_mean = torch.tensor(bc.IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        vm_std = torch.tensor(bc.IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
        cl_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        cl_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        set_of_view = [0, 0, 1, 1]

        loader = DataLoader(DecodeDataset(clips, todo), batch_size=args.batch, num_workers=args.workers,
                            collate_fn=collate, prefetch_factor=4, persistent_workers=False)
        start, count, last_flush = time.time(), 0, 0
        with torch.inference_mode():
            for index, frames, valid, waves in loader:
                b = len(index)
                views = make_views(frames.to(device, non_blocking=True)).reshape(b * v, NUM_FRAMES, 3, SIZE, SIZE)
                normed = (views - vm_mean) / vm_std
                tk = videomae_tokens(models["k400"], normed, args.chunk).reshape(b, v, TOKENS, -1, 768)
                ts = videomae_tokens(models["ssl"], normed, args.chunk)[:, :, 0].reshape(b, v, TOKENS, 768)
                imgs = views[:, ::2].reshape(-1, 3, SIZE, SIZE)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb = clip_model(pixel_values=(imgs - cl_mean) / cl_std).image_embeds
                emb = F.normalize(emb.float(), dim=-1).reshape(b, v, TOKENS, -1)
                tok_valid = valid.reshape(b, 2, TOKENS, 2).any(-1).numpy()[:, set_of_view]
                tk, ts, emb = (t.half().cpu().numpy() for t in (tk, ts, emb))
                for j, i in enumerate(index):  # memmap writes clip by clip (indices are sorted)
                    vk[:, i], vs[:, i], clip_img[:, i] = tk[j], ts[j], emb[j]
                    pad[:, i] = ~tok_valid[j]
                    wave = waves[j]
                    if wave.numel() >= bc.AUDIO_SR // 4:
                        wave = wave.to(device)
                        for key, model in beats.items():
                            feats = model.extract_features(wave[None])[0][0]
                            pooled, _ = bc.pool_tokens(feats, torch.ones(len(feats), dtype=torch.bool,
                                                                         device=device), AUDIO_TOKENS)
                            audio[key][i] = pooled.half().cpu().numpy()
                        has_audio[i] = True
                    done[i] = True
                count += b
                if count - last_flush >= 500:
                    last_flush = count
                    for arr in arrays:
                        arr.flush()
                    print(f"[FEATURES] {count}/{len(todo)} clips, {count / (time.time() - start):.2f} clips/s",
                          flush=True)
    for arr in arrays:
        arr.flush()
    meta["complete"] = bool(np.asarray(done).all())
    with open(out / "meta.json", "w") as h:
        json.dump(meta, h)
    print(f"[FEATURES] complete={meta['complete']} clips with audio={int(np.asarray(has_audio).sum())}")


if __name__ == "__main__":
    main()
