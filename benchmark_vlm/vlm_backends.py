"""Model adapters for eval_vlm.py: one class per model family, same scoring for all.

Every backend turns a clip into next-token logits after the multiple-choice
prompt; eval_vlm.py then reads the probabilities of the letters A/B/C.

    load_clip(path)      CPU, runs in DataLoader workers: decode + sample frames
    logits(items)        GPU: next-token logits [n, vocab] for a list of decoded clips
    letter_ids           for "A", "B", "C": the token ids of each letter's spellings

Backends
    qwen     Qwen3-VL, Qwen3.5            qwen-vl-utils, native transformers
    hf       InternVL3.5-HF, Gemma 4      transformers video processors
    keye     Keye-VL-1.5                  keye-vl-utils, trust_remote_code
    minicpm  MiniCPM-V 4.0 / 4.5          trust_remote_code chat(); logits from an lm_head hook

Frame sampling: --num-frames frames uniformly over the clip, each resized to at most
--max-pixels (aspect ratio kept). InternVL and Gemma then resize to their own input
size. MiniCPM-V 4.5 can pack --pack consecutive frames into one group (3D resampler),
so it sees pack x num_frames frames for the token cost of num_frames.
"""
import glob
import inspect
import math
import os

import numpy as np
import torch

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LETTERS = ["A", "B", "C"]


def _tf_major():
    import transformers
    return int(transformers.__version__.split(".")[0])


def bf16_kwargs():
    """transformers 5 renamed torch_dtype -> dtype; remote-code models run on 4.x (env vlm-tf4)."""
    return {"dtype": torch.bfloat16} if _tf_major() >= 5 else {"torch_dtype": torch.bfloat16}


def detect_backend(model_id):
    m = model_id.lower()
    if "minicpm" in m:
        return "minicpm"
    if "keye" in m:
        return "keye"
    if "qwen" in m:
        return "qwen"
    return "hf"


def pick_attn(attn):
    if attn != "auto":
        return attn
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def letter_token_ids(tok):
    """For each letter, every single-token spelling of it: "A", "▁A" / "ĠA" (with a leading
    space). Tokenizers differ: SentencePiece (MiniCPM-V 4) encodes "A" as "▁A" while the model
    answers with the bare "A". eval_vlm.py adds up the probabilities of all spellings."""
    unk = getattr(tok, "unk_token_id", None)
    groups = []
    for L in LETTERS:
        ids = set()
        for text in (L, " " + L):
            t = tok.encode(text, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        for piece in (L, "\u2581" + L, "\u0120" + L):
            i = tok.convert_tokens_to_ids(piece)
            if isinstance(i, int) and i >= 0 and i != unk:
                ids.add(i)
        assert ids, f"letter {L} has no single-token spelling"
        groups.append(sorted(ids))
    flat = [i for g in groups for i in g]
    assert len(flat) == len(set(flat)), f"letter tokens overlap: {groups}"
    return groups


# ---------------------------------------------------------------- frame decoding
def frame_indices(total, n):
    return np.linspace(0, max(total - 1, 0), num=n).round().astype(int)


def fit_pixels(h, w, max_pixels, multiple=28):
    """Size with h*w <= max_pixels, aspect kept, sides rounded down to `multiple`."""
    scale = min(1.0, math.sqrt(max_pixels / (h * w)))
    nh = max(multiple, int(h * scale) // multiple * multiple)
    nw = max(multiple, int(w * scale) // multiple * multiple)
    return nh, nw


def read_frames(path, n, max_pixels):
    """uint8 [n, H, W, 3] sampled uniformly over a video file or a frame folder, plus metadata."""
    import cv2
    if os.path.isdir(path):
        files = sorted(f for f in glob.glob(os.path.join(path, "*")) if f.lower().endswith(IMG_EXT))
        if not files:
            raise FileNotFoundError(f"no frames in {path}")
        idx = frame_indices(len(files), n)
        frames = np.stack([cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2RGB) for i in idx])
        fps, total = 2.0, len(files)          # real fps of a folder is unknown
    else:
        import decord
        vr = decord.VideoReader(path, num_threads=1)
        total, fps = len(vr), float(vr.get_avg_fps())
        idx = frame_indices(total, n)
        frames = vr.get_batch(list(idx)).asnumpy()
    h, w = frames.shape[1:3]
    nh, nw = fit_pixels(h, w, max_pixels)
    if (nh, nw) != (h, w):
        frames = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames])
    meta = {"fps": fps, "total_num_frames": int(total), "frames_indices": [int(i) for i in idx],
            "duration": total / fps if fps else None}
    return frames, meta


# ---------------------------------------------------------------- remote-code compatibility
def patch_remote_model_for_tf5(cls):
    """Remote-code models written for transformers 4.x skip post_init() in __init__;
    transformers 5 needs it (it sets all_tied_weights_keys used by from_pretrained)."""
    if _tf_major() < 5 or getattr(cls, "_tf5_patched", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        if not hasattr(self, "all_tied_weights_keys"):
            self.post_init()
    cls.__init__ = __init__
    cls._tf5_patched = True


# ---------------------------------------------------------------- base
class Backend:
    batchable = True     # False: logits() is always called with one clip

    def __init__(self, args, prompt):
        self.args, self.prompt = args, prompt

    def load_model(self):
        raise NotImplementedError

    def load_clip(self, path):
        raise NotImplementedError

    def logits(self, items):
        raise NotImplementedError


# ---------------------------------------------------------------- Qwen3-VL / Qwen3.5
class QwenBackend(Backend):
    """Same path as the guide's eval_vlm.py: qwen-vl-utils video entry."""

    def _messages(self, path):
        a = self.args
        if os.path.isdir(path):
            files = sorted(f for f in glob.glob(os.path.join(path, "*")) if f.lower().endswith(IMG_EXT))
            if not files:
                raise FileNotFoundError(f"no frames in {path}")
            idx = list(frame_indices(len(files), min(a.num_frames, len(files))))
            if len(idx) % 2:                     # qwen-vl-utils needs an even number of frames
                idx.append(idx[-1])
            video = {"type": "video", "video": ["file://" + os.path.abspath(files[i]) for i in idx],
                     "sample_fps": a.frame_fps, "max_pixels": a.max_pixels}
        else:
            video = {"type": "video", "video": "file://" + os.path.abspath(path),
                     "nframes": a.num_frames, "max_pixels": a.max_pixels}
        return [{"role": "user", "content": [video, {"type": "text", "text": self.prompt}]}]

    def load_model(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        attn = pick_attn(self.args.attn)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.args.model, **bf16_kwargs(), device_map=self.args.device, attn_implementation=attn).eval()
        self.processor = AutoProcessor.from_pretrained(self.args.model)
        self.processor.tokenizer.padding_side = "left"
        self.letter_ids = letter_token_ids(self.processor.tokenizer)
        # enable_thinking=False: Qwen3.5 answers directly; Qwen3-VL Instruct ignores the flag
        self.text = self.processor.apply_chat_template(
            self._messages("clip.mp4"), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return attn

    def load_clip(self, path):
        from qwen_vl_utils import process_vision_info
        _, videos, _ = process_vision_info(self._messages(path), image_patch_size=16,
                                           return_video_kwargs=True, return_video_metadata=True)
        video, meta = videos[0]
        return {"video": video, "meta": meta}

    def logits(self, items):
        inputs = self.processor(text=[self.text] * len(items), videos=[it["video"] for it in items],
                                video_metadata=[it["meta"] for it in items], do_resize=False,
                                do_sample_frames=False, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        return self.model(**inputs, logits_to_keep=1).logits[:, -1].float()


# ---------------------------------------------------------------- InternVL3.5-HF, Gemma 4
class HFVideoBackend(Backend):
    """Models whose transformers processor takes videos directly."""

    def load_model(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        attn = pick_attn(self.args.attn)
        kw = dict(**bf16_kwargs(), device_map=self.args.device)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.args.model, attn_implementation=attn, **kw).eval()
        except ValueError as e:
            # e.g. Gemma 4: its audio tower has no sdpa; let transformers choose per submodule
            if "attention implementation" not in str(e):
                raise
            print(f"{attn} not supported by every submodule; using the default attention", flush=True)
            self.model = AutoModelForImageTextToText.from_pretrained(self.args.model, **kw).eval()
            attn = "default"
        self.processor = AutoProcessor.from_pretrained(self.args.model)
        self.processor.tokenizer.padding_side = "left"
        self.letter_ids = letter_token_ids(self.processor.tokenizer)
        self.video_kwargs = {}
        if self.model.config.model_type == "internvl":
            # the HF video processor config says 384x384, but the vision tower takes 448x448 tiles
            # (same as the image processor); 384 breaks pixel_shuffle
            size = self.processor.image_processor.size
            self.video_kwargs["size"] = {"height": size["height"], "width": size["width"]}
        self._frames_per_video = None
        inner = getattr(self.model, "model", None)
        if self.model.config.model_type == "gemma4" and inner is not None:
            # transformers 5.20 bug: Gemma4Model.forward takes num_frames_per_video but does not
            # pass it to get_video_features, which then raises. Inject it for the current batch.
            orig = inner.get_video_features

            def get_video_features(pixel_values_videos, video_position_ids=None, **kw):
                if kw.get("num_frames_per_video") is None:
                    kw["num_frames_per_video"] = self._frames_per_video
                return orig(pixel_values_videos, video_position_ids, **kw)
            inner.get_video_features = get_video_features
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": self.prompt}]}]
        self.text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return attn

    def load_clip(self, path):
        frames, meta = read_frames(path, self.args.num_frames, self.args.max_pixels)
        return {"video": frames, "meta": meta}

    def logits(self, items):
        inputs = self.processor(text=[self.text] * len(items), videos=[it["video"] for it in items],
                                video_metadata=[it["meta"] for it in items], do_sample_frames=False,
                                padding=True, return_tensors="pt", **self.video_kwargs)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        self._frames_per_video = inputs.get("num_frames_per_video")
        return self.model(**inputs, logits_to_keep=1).logits[:, -1].float()


# ---------------------------------------------------------------- Keye-VL-1.5
class KeyeBackend(Backend):
    """Keye-VL-1.5 (remote code). '/no_think' asks for a direct answer."""

    def _messages(self, path):
        a = self.args
        video = {"type": "video", "video": "file://" + os.path.abspath(path),
                 "nframes": a.num_frames, "max_pixels": a.max_pixels}
        return [{"role": "user", "content": [video, {"type": "text", "text": self.prompt + "/no_think"}]}]

    def load_model(self):
        from transformers import AutoModel, AutoProcessor
        attn = pick_attn(self.args.attn)
        self.model = AutoModel.from_pretrained(
            self.args.model, **bf16_kwargs(), trust_remote_code=True,
            attn_implementation=attn).eval().to(self.args.device)
        self.processor = AutoProcessor.from_pretrained(self.args.model, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        self.letter_ids = letter_token_ids(self.processor.tokenizer)
        self.text = self.processor.apply_chat_template(
            self._messages("clip.mp4"), tokenize=False, add_generation_prompt=True)
        self.text += self.args.answer_prefix
        return attn

    def load_clip(self, path):
        from keye_vl_utils import process_vision_info
        _, videos, kwargs = process_vision_info(self._messages(path))
        return {"video": videos[0], "kwargs": kwargs}

    def logits(self, items):
        kwargs = {}
        for k, v in items[0]["kwargs"].items():   # per-video lists (e.g. fps) are concatenated
            kwargs[k] = sum((it["kwargs"][k] for it in items), []) if isinstance(v, list) else v
        inputs = self.processor(text=[self.text] * len(items), videos=[it["video"] for it in items],
                                padding=True, return_tensors="pt", **kwargs).to(self.model.device)
        return self.model(**inputs).logits[:, -1].float()


# ---------------------------------------------------------------- MiniCPM-V 4.0 / 4.5
class MiniCPMBackend(Backend):
    """MiniCPM-V (remote code): frames as images through model.chat(); the first lm_head
    call (the prompt pass) gives the next-token logits."""
    batchable = False
    TIME_SCALE = 0.1      # temporal ids are timestamps in 0.1 s steps (MiniCPM-V 4.5)

    def load_model(self):
        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        attn = pick_attn(self.args.attn)
        config = AutoConfig.from_pretrained(self.args.model, trust_remote_code=True)
        cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], self.args.model)
        patch_remote_model_for_tf5(cls)
        self.model = cls.from_pretrained(
            self.args.model, config=config, attn_implementation=attn,
            **bf16_kwargs()).eval().to(self.args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model, trust_remote_code=True)
        self.letter_ids = letter_token_ids(self.tokenizer)
        self.has_temporal = "temporal_ids" in inspect.signature(self.model.chat).parameters or \
            "temporal_ids" in inspect.getsource(type(self.model).chat)
        self._captured = None

        def hook(_, __, out):
            if self._captured is None:
                self._captured = out[:, -1].float()
        self.model.llm.get_output_embeddings().register_forward_hook(hook)
        return attn

    def load_clip(self, path):
        from PIL import Image
        pack = self.args.pack
        frames, meta = read_frames(path, self.args.num_frames * pack, self.args.max_pixels)
        out = {"frames": [Image.fromarray(f) for f in frames]}
        if pack > 1 or self.args.model.lower().rstrip("/").endswith("4_5"):
            ts = np.array(meta["frames_indices"]) / meta["fps"]
            ids = np.round(ts / self.TIME_SCALE).astype(int)
            out["temporal_ids"] = [ids[i:i + pack].tolist() for i in range(0, len(ids), pack)]
        return out

    def logits(self, items):
        it = items[0]
        kwargs = dict(use_image_id=False, max_slice_nums=1)
        if self.has_temporal and "temporal_ids" in it:
            kwargs["temporal_ids"] = it["temporal_ids"]
        msgs = [{"role": "user", "content": it["frames"] + [self.prompt]}]
        self._captured = None
        self.model.chat(msgs=msgs, tokenizer=self.tokenizer, sampling=False, max_new_tokens=1,
                        num_beams=1, repetition_penalty=1.0, enable_thinking=False, **kwargs)
        assert self._captured is not None, "lm_head hook did not fire"
        return self._captured[:1]


BACKENDS = {"qwen": QwenBackend, "hf": HFVideoBackend, "keye": KeyeBackend, "minicpm": MiniCPMBackend}
