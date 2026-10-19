import torch


def collate_fn(batch):
    videos = torch.stack([x["video"] for x in batch])
    frame_masks = torch.stack([x["frames_mask"] for x in batch])
    audios = torch.stack([x["audio"] for x in batch])
    audio_masks = torch.stack([x["audio_mask"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    return {
        "video": videos,
        "frame_mask": frame_masks,
        "audio": audios,
        "audio_mask": audio_masks,
        "label": labels
    }
    