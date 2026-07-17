import os
import glob
import cv2

# ====================== CONFIG ======================
IMAGE_DIR = "/data/pickleball/data/images"
LABEL_DIR = "/data/pickleball/data/labels/combined_labels"

VIDEO_NAME = "15_11_2025_Cuong_Trung_PhaiTay"
OUTPUT_VIDEO = "result.mp4"

FPS = 30

# COCO 17 keypoints
SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]
# ====================================================

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, f"{VIDEO_NAME}_*.jpg"))
)

if len(image_paths) == 0:
    raise ValueError("Không tìm thấy ảnh.")

# đọc ảnh đầu tiên để lấy kích thước
first = cv2.imread(image_paths[0])
h, w = first.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

for image_path in image_paths:

    image = cv2.imread(image_path)

    label_path = os.path.join(
        LABEL_DIR,
        os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    )

    if os.path.exists(label_path):

        with open(label_path) as f:

            for line in f:

                items = line.strip().split()

                if len(items) < 5:
                    continue

                cls = items[0]

                xc, yc, bw, bh = map(float, items[1:5])

                xmin = int((xc - bw / 2) * w)
                ymin = int((yc - bh / 2) * h)
                xmax = int((xc + bw / 2) * w)
                ymax = int((yc + bh / 2) * h)

                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w - 1, xmax)
                ymax = min(h - 1, ymax)

                # bbox
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    cls,
                    (xmin, max(20, ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # pose
                if len(items) > 5:

                    values = list(map(float, items[5:]))

                    pts = []

                    for i in range(0, len(values), 3):

                        if i + 2 >= len(values):
                            break

                        x, y, v = values[i:i + 3]

                        px = int(x * w)
                        py = int(y * h)

                        pts.append((px, py))

                        cv2.circle(image, (px, py), 4, (0, 0, 255), -1)

                    for p1, p2 in SKELETON:
                        if p1 < len(pts) and p2 < len(pts):
                            cv2.line(image, pts[p1], pts[p2], (255, 0, 0), 2)

    writer.write(image)

writer.release()

print("Saved:", OUTPUT_VIDEO)