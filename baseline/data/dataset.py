import random
import wave
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

try:
    torchaudio.set_audio_backend("sox_io")
except Exception:
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass


class PickleballDataset(Dataset):

    def __init__(self, samples, num_frames=60, image_size=224,
                 audio_sr=16000, audio_duration=2.0, train=True):
        self.samples = samples
        self.train = train
        self.num_frames = num_frames
        self.image_size = image_size
        self.target_sr = audio_sr
        self.max_audio_len = int(audio_duration * audio_sr)
        self.resamplers = {}

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def sample_indices(self, total_frames):
        if total_frames < self.num_frames:
            return 0
        if self.train:
            return random.randint(0, total_frames - self.num_frames)
        else:
            return (total_frames - self.num_frames) // 2

    def load_frames(self, image_paths):
        total = len(image_paths)
        start = self.sample_indices(total)
        image_paths = image_paths[start:start + self.num_frames]
        if len(image_paths) < self.num_frames:
            pad = self.num_frames - len(image_paths)
            image_paths = image_paths + [image_paths[-1]] * pad
        frame_mask = torch.ones(self.num_frames, dtype=torch.bool)
        frames = []
        for path in image_paths:  
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f"Cannot read {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(img)
        video = torch.stack(frames)
        video = (video - self.mean) / self.std
        return video.contiguous(), frame_mask

    def crop_audio(self, waveform):
        length = waveform.shape[0]
        if length < self.max_audio_len:
            return F.pad(waveform, (0, self.max_audio_len - length))
        if self.train:
            start = random.randint(0, length - self.max_audio_len)
        else:
            start = (length - self.max_audio_len) // 2

        return waveform[start:start + self.max_audio_len]

    def _resample_numpy(self, waveform, sr):
        if sr == self.target_sr:
            return waveform.astype(np.float32)

        target_len = int(len(waveform) * self.target_sr / sr)
        if target_len <= 1:
            return np.zeros(self.max_audio_len, dtype=np.float32)

        src_idx = np.arange(len(waveform), dtype=np.float32)
        dst_idx = np.linspace(0, len(waveform) - 1, target_len, dtype=np.float32)
        resampled = np.interp(dst_idx, src_idx, waveform)
        return resampled.astype(np.float32)

    def load_audio(self, audio_path):
        audio_path = str(audio_path)
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.numpy().astype(np.float32)
        except Exception:
            with wave.open(audio_path, "rb") as wav_file:
                sr = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                frames = wav_file.readframes(n_frames)
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                if wav_file.getnchannels() > 1:
                    audio = audio.reshape(-1, wav_file.getnchannels()).mean(axis=1)
                waveform = audio / 32768.0

        waveform = self._resample_numpy(waveform, sr)
        waveform = self.crop_audio(torch.from_numpy(waveform))
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
        return waveform.contiguous()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video, frame_mask = self.load_frames(sample["frames"])
        audio = self.load_audio(sample["audio"])
        return {
            "video": video,
            "frame_mask": frame_mask,
            "audio": audio,
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }