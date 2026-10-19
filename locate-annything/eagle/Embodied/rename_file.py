import os
from pathlib import Path

ROOT = Path("/data/Cuong/pickleball/data/shot1")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

num_images = 0
num_labels = 0

for dirpath, dirnames, _ in os.walk(ROOT):
    current = Path(dirpath)

    # Xóa images
    if current.name == "images":
        for file in current.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                file.unlink()
                num_images += 1

    # Xóa labels
    elif current.name == "labels_balls":
        for file in current.iterdir():
            if file.is_file() and file.suffix.lower() == ".txt":
                file.unlink()
                num_labels += 1
    elif current.name == "labels_persons":
        for file in current.iterdir():
            if file.is_file() and file.suffix.lower() == ".txt":
                file.unlink()
                num_labels += 1
    elif current.name == "labels_paddles":
        for file in current.iterdir():
            if file.is_file() and file.suffix.lower() == ".txt":
                file.unlink()
                num_labels += 1

print(f"Deleted {num_images} images")
print(f"Deleted {num_labels} labels")
print("Done.")