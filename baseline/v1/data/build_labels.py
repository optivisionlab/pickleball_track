import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


LABEL_MAP = {
    "trung": 1,
    "truot": 0,
}

def build_index(root_dir):
    root = Path(root_dir)
    samples = []
    for label_name, label in LABEL_MAP.items():
        label_dir = root / label_name
        print("label_dir: ", label_dir)
        if not label_dir.exists():
            continue
        for hand_dir in sorted(label_dir.iterdir()):
            # print("hand_dir: ", hand_dir)
            if not hand_dir.is_dir():
                continue
            for player_dir in sorted(hand_dir.iterdir()):
                # print("player_dir: ", player_dir)
                if not player_dir.is_dir():
                    continue
                image_dir = player_dir / "shot1" / "images"
                audio_dir = player_dir / "shot1" / "audios"
                if not image_dir.exists() or not audio_dir.exists():
                    continue
                frame_groups = defaultdict(list)
                for img_path in image_dir.glob("*.*"):
                    if img_path.suffix.lower() not in {".jpg"}:
                        continue
                    try:
                        video_name = img_path.stem.rsplit("_", 1)[0]
                        print(video_name)
                    except Exception:
                        continue
                    frame_groups[video_name].append(img_path)
                if len(frame_groups.keys()) > 1:
                    print("len_group: ", len(frame_groups.keys()))
                for video_name, frames in frame_groups.items():
                    frames = sorted(frames)
                    audio_path = audio_dir / f"{video_name}.wav"
                    if not audio_path.exists():
                        print(f"[Warning] Missing audio: {audio_path}")
                        continue
                    samples.append(
                        {
                            "video_name": video_name,
                            "player": player_dir.name,
                            "hand": hand_dir.name,
                            "label_name": label_name,
                            "label": label,
                            "frames": [str(f) for f in frames],
                            "audio": str(audio_path),
                            "num_frames": len(frames),
                        }
                    )
    return samples



if __name__ == "__main__":

    root_dir = "/data/Cuong/pickleball/data/video_capcut"
    samples = build_index(root_dir)
    save_path = "index.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            samples,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved {len(samples)} samples to {save_path}")