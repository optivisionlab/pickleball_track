import os
import glob
from collections import defaultdict

LABEL_DIR = "/data/pickleball/data/persons"
BALL_CLASS = 3
MAX_VALID_GAP = 5


def has_ball(label_path):
    """
    Kiểm tra file txt có bbox của bóng hay không.
    """
    if not os.path.exists(label_path):
        return False

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])

            if cls == BALL_CLASS:
                return True

    return False


# ---------------------------------------------------
# Gom các frame theo video
# ---------------------------------------------------
video_frames = defaultdict(list)

for path in glob.glob(os.path.join(LABEL_DIR, "*.txt")):

    name = os.path.splitext(os.path.basename(path))[0]

    # video_000123
    video_name, frame_idx = name.rsplit("_", 1)

    video_frames[video_name].append(
        (
            int(frame_idx),
            path
        )
    )


global_max_gap = 0

print("=" * 60)

for video_name in sorted(video_frames.keys()):

    frames = sorted(video_frames[video_name])

    max_gap_video = 0

    current_gap = 0

    started = False

    for _, path in frames:

        if has_ball(path):

            if started:

                if 0 < current_gap <= MAX_VALID_GAP:
                    max_gap_video = max(max_gap_video, current_gap)

            current_gap = 0
            started = True

        else:

            if started:
                current_gap += 1

    print(f"{video_name:<40} max_gap = {max_gap_video}")

    global_max_gap = max(global_max_gap, max_gap_video)

print("=" * 60)
print(f"Largest valid gap (<= {MAX_VALID_GAP}) = {global_max_gap}")