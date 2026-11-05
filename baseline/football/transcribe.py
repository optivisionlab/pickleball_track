"""Commentary transcripts for the v4 / v5 text stream (replaces the fixed prompt).

Whisper (large-v3-turbo, language detected per clip) transcribes each clip's own audio track,
an input of the clip, never its label. The transcript is then encoded into the CLIP ViT-B/32
text space that v4 / v5 use:
    --encoder clip   CLIP text encoder (English tokenizer)
    --encoder mclip  sentence-transformers/clip-ViT-B-32-multilingual-v1 (multilingual
                     DistilBERT distilled into the same CLIP text space), for non-English commentary

Output in --out (the feature folder):
    transcripts.json         {clip path: {"lang": .., "text": ..}}
    text_{encoder}.npy       [N, 512] f16, L2-normalised
    has_text.npy             [N] bool, False for empty / no-audio clips (prompt is used instead)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb_common as fc  # noqa: E402

bc = fc.bc
MCLIP = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


class AudioDataset(Dataset):
    def __init__(self, clips, todo):
        self.clips, self.todo = clips, todo

    def __len__(self):
        return len(self.todo)

    def __getitem__(self, i):
        index = self.todo[i]
        return index, bc.read_audio(self.clips[index].path)[: 30 * bc.AUDIO_SR]


def encode_mclip(texts, device, batch_size=256):
    """Multilingual CLIP text space without sentence-transformers: DistilBERT mean pool + Dense."""
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MCLIP)
    model = AutoModel.from_pretrained(MCLIP).to(device).eval()
    try:
        dense = load_file(hf_hub_download(MCLIP, "2_Dense/model.safetensors"))
    except Exception:
        dense = torch.load(hf_hub_download(MCLIP, "2_Dense/pytorch_model.bin"), map_location="cpu")
    w = dense["linear.weight"].to(device)
    b = dense.get("linear.bias")
    out = np.zeros((len(texts), 512), np.float32)
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            enc = tok(list(texts[s:s + batch_size]), padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"][..., None].float()
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
            e = pooled @ w.T + (b.to(device) if b is not None else 0)
            out[s:s + batch_size] = F.normalize(e.float(), dim=-1).cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=fc.FEATURES)
    p.add_argument("--model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl  # one transcriber at a time: a second instance waits, then finds everything cached
    lock = open(out / ".transcribe.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    clips = fc.load_clips()
    keys = [c.info["path"] for c in clips]

    cache = out / "transcripts.json"
    texts = json.load(open(cache)) if cache.is_file() else {}
    todo = [i for i, k in enumerate(keys) if k not in texts]
    print(f"[ASR] {len(clips)} clips, {len(todo)} to transcribe with {args.model}", flush=True)
    if todo:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(args.model)
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.float16).to(device).eval()
        loader = DataLoader(AudioDataset(clips, todo), batch_size=args.batch_size, num_workers=args.workers,
                            collate_fn=lambda b: list(zip(*b)))
        with torch.inference_mode():
            for step, (idx, waves) in enumerate(loader):
                keep = [k for k, w in enumerate(waves) if len(w) >= bc.AUDIO_SR // 2]
                for k in set(range(len(idx))) - set(keep):
                    texts[keys[idx[k]]] = {"lang": "", "text": ""}
                if keep:
                    feats = processor([waves[k] for k in keep], sampling_rate=bc.AUDIO_SR,
                                      return_tensors="pt").input_features.to(device, torch.float16)
                    lang_ids = model.detect_language(feats)
                    langs = [re.sub(r"[<|>]", "", t) for t in processor.tokenizer.batch_decode(lang_ids[:, None])]
                    ids = model.generate(feats, task="transcribe", max_new_tokens=160)
                    clean = processor.batch_decode(ids, skip_special_tokens=True)
                    for k, lang, t in zip(keep, langs, clean):
                        texts[keys[idx[k]]] = {"lang": lang, "text": t.strip()}
                if step % 20 == 0:
                    json.dump(texts, open(cache, "w"), ensure_ascii=False)
                    print(f"[ASR] {len(texts)}/{len(keys)}", flush=True)
        json.dump(texts, open(cache, "w"), ensure_ascii=False)
        del model
        torch.cuda.empty_cache()

    all_text = [texts.get(k, {}).get("text", "") for k in keys]
    has_text = np.array([len(t.split()) >= 2 for t in all_text])
    np.save(out / "has_text.npy", has_text)
    np.save(out / "text_clip.npy", fc.encode_texts(all_text, device).astype(np.float16))
    np.save(out / "text_mclip.npy", encode_mclip(all_text, device).astype(np.float16))
    langs = {}
    for k in keys:
        lang = texts.get(k, {}).get("lang", "")
        langs[lang] = langs.get(lang, 0) + 1
    print(f"[ASR] clips with text: {int(has_text.sum())}/{len(keys)}; languages: "
          f"{dict(sorted(langs.items(), key=lambda kv: -kv[1])[:10])}")
    for t in all_text[:5]:
        print("   ", t[:150])


if __name__ == "__main__":
    main()
