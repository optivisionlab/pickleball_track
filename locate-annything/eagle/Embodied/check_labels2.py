import os
import glob
from collections import defaultdict

import cv2

ROOT = "/data/Cuong/pickleball/data/shot1"
LABEL_FOLDER = "labels"
SAVE_DIR = "/data/Cuong/pickleball/data/image_check3"

os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAME = {
    0: "paddle",
    1: "ball",
    2: "person",
}

COLOR = {
    0: (0, 255, 0),      # paddle
    1: (0, 255, 255),    # ball
    2: (0, 0, 255),      # person
}


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    return x1, y1, x2, y2


saved = 0

image_dirs = sorted(
    glob.glob(
        os.path.join(ROOT, "**", "images"),
        recursive=True,
    )
)

for image_dir in image_dirs:

    folder = os.path.dirname(image_dir)
    label_dir = os.path.join(folder, LABEL_FOLDER)

    if not os.path.isdir(label_dir):
        continue

    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    # gom theo video
    videos = defaultdict(list)

    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]

        if "_" not in name:
            continue

        video_name = name.rsplit("_", 1)[0]
        videos[video_name].append(img)

    # mỗi video lấy đúng 1 frame đầu tiên đủ 3 object
    for video_name, img_list in videos.items():

        found = False

        for img_path in img_list:

            txt_path = os.path.join(
                label_dir,
                os.path.basename(img_path).replace(".jpg", ".txt"),
            )

            if not os.path.exists(txt_path):
                continue

            with open(txt_path) as f:
                lines = [x.strip() for x in f if x.strip()]

            has_paddle = False
            has_ball = False
            has_person = False

            objects = []

            for line in lines:

                nums = list(map(float, line.split()))
                cls = int(nums[0])

                objects.append(nums)

                if cls == 0:
                    has_paddle = True

                elif cls == 1:
                    has_ball = True

                elif cls == 2:
                    # person phải có bbox + 17 keypoint
                    if len(nums) >= 56:
                        has_person = True

            if not (has_paddle and has_ball and has_person):
                continue

            img = cv2.imread(img_path)

            if img is None:
                continue

            H, W = img.shape[:2]

            # vẽ
            for obj in objects:

                cls = int(obj[0])

                xc, yc, bw, bh = obj[1:5]

                x1, y1, x2, y2 = yolo_to_xyxy(
                    xc, yc, bw, bh, W, H
                )

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    COLOR[cls],
                    2,
                )

                cv2.putText(
                    img,
                    CLASS_NAME[cls],
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    COLOR[cls],
                    2,
                )

                # person -> vẽ keypoint
                if cls == 2:

                    kps = obj[5:]

                    for i in range(17):

                        x = kps[i * 3]
                        y = kps[i * 3 + 1]
                        v = kps[i * 3 + 2]

                        if v <= 0:
                            continue

                        px = int(x * W)
                        py = int(y * H)

                        cv2.circle(
                            img,
                            (px, py),
                            3,
                            (255, 0, 255),
                            -1,
                        )

            save_path = os.path.join(
                SAVE_DIR,
                video_name + ".jpg",
            )

            cv2.imwrite(save_path, img)

            print(save_path)

            saved += 1
            found = True
            break

        if found:
            continue

print("=" * 50)
print("Saved:", saved)
print("Output:", SAVE_DIR)
print("=" * 50)