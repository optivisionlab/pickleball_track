import torch
import torch.nn.functional as F


def collate_fn(batch):
    videos = torch.stack([x["video"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    frame_masks = torch.stack([x["frame_mask"] for x in batch])
    audios = [x["audio"] for x in batch]
    lengths = [a.shape[0] for a in audios]
    max_len = max(lengths)
    padded = []
    masks = []

    for a in audios:
        pad_len = max_len - a.shape[0]
        padded_audio = F.pad(a, (0, pad_len))
        mask = torch.zeros(max_len, dtype=torch.bool)
        if pad_len > 0:
            mask[a.shape[0]:] = True
        padded.append(padded_audio)
        masks.append(mask)

    return {
        "video": videos,
        "frame_mask": frame_masks,
        "audio": torch.stack(padded),
        "audio_mask": torch.stack(masks),
        "label": labels
    }