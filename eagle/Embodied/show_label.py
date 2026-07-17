import cv2
import os

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


def rotate_bbox_back(xc, yc, bw, bh):
    """
    Label được tạo trên ảnh xoay 90° ngược chiều kim đồng hồ (CCW),
    chuyển về ảnh gốc.
    """

    xc_old = xc
    yc_old = yc

    # Đổi lại hệ tọa độ
    xc = 1.0 - yc_old
    yc = xc_old

    # width và height đổi chỗ
    bw, bh = bh, bw

    return xc, yc, bw, bh


if __name__ == "__main__":

    image_path = "/data/pickleball/data/images/15_11_2025_Cuong_Trung_PhaiTay_000326.jpg"
    txt_path = "/data/pickleball/data/labels/balls/15_11_2025_Cuong_Trung_PhaiTay_000327.txt"

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
        exit()

    if not os.path.exists(txt_path):
        print(f"Label file not found: {txt_path}")
        exit()

    h, w = image.shape[:2]

    with open(txt_path, "r") as f:

        for line in f:

            items = line.strip().split()

            if len(items) < 5:
                continue

            cls = int(items[0])

            x_center, y_center, bw, bh = map(float, items[1:5])

            ###################################################
            # FIX: label được predict trên ảnh xoay 90°
            ###################################################
            if cls == 1:
                x_center, y_center, bw, bh = rotate_bbox_back(
                    x_center,
                    y_center,
                    bw,
                    bh
                )

            xmin = int((x_center - bw / 2) * w)
            ymin = int((y_center - bh / 2) * h)
            xmax = int((x_center + bw / 2) * w)
            ymax = int((y_center + bh / 2) * h)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w - 1, xmax)
            ymax = min(h - 1, ymax)

            # Draw bbox
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                str(cls),
                (xmin, max(20, ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            ###################################################
            # Pose
            ###################################################
            if len(items) > 5:

                keypoints = list(map(float, items[5:]))

                pts = []

                for i in range(0, len(keypoints), 3):

                    if i + 2 >= len(keypoints):
                        break

                    x, y, v = keypoints[i:i + 3]

                    px = int(x * w)
                    py = int(y * h)

                    pts.append((px, py))

                    if v > 0:
                        cv2.circle(image, (px, py), 4, (0, 0, 255), -1)

                for p1, p2 in SKELETON:
                    if p1 < len(pts) and p2 < len(pts):
                        cv2.line(image, pts[p1], pts[p2], (255, 0, 0), 2)

    cv2.imwrite("image.jpg", image)
    print("Saved to image.jpg")