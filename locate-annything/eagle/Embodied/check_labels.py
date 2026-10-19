import cv2
import glob 
import os

image_dir = "/data/pickleball/data/images"
label_dir = "/data/pickleball/data/combine_label"
label_ball = "/data/pickleball/data/balls"
label_paddle_check = "/data/pickleball/data/paddles_kalman_check"
list_video_file = "/data/pickleball/pickleball_track/locate-annything/eagle/Embodied/list_video.txt"
out_dir = '/data/pickleball/data/image_check'
out_dir2 = '/data/pickleball/data/image_check2'
os.makedirs(out_dir2, exist_ok = True)
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

def check1():
    with open(list_video_file, "r") as f:
        lines = f.read().strip().split("\n")
        for video_path in lines:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            list_images = glob.glob(os.path.join(image_dir, f"{video_name}_*.jpg"))
            for image_path in list_images:
                label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
                with open(label_path, 'r') as f:
                    data = f.read().strip().split("\n")
                if len(data) < 3:
                    continue
                else:
                    image = cv2.imread(image_path)
                    image_name = os.path.basename(image_path)
                    h, w = image.shape[:2]
                    for line in data:
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
                    out_image = os.path.join(out_dir, image_name)
                    cv2.imwrite(out_image, image)
                    break
                
def check2():
    with open(list_video_file, "r") as f:
        lines = f.read().strip().split("\n")
        for video_path in lines:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(video_name)
            image_path = glob.glob(os.path.join(image_dir, f"{video_name}_*.jpg"))[0]
            label_path = os.path.join(label_ball, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
            print(label_path)
            with open(label_path, 'r') as f:
                data = f.read().strip().split("\n")
            image = cv2.imread(image_path)
            image_name = os.path.basename(image_path)
            h, w = image.shape[:2]
            for line in data:
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
            out_image = os.path.join(out_dir2, image_name)
            cv2.imwrite(out_image, image)

def check3():
    label_dir = "/data/Cuong/pickleball/data/ball_tracked"
    image_dir = "/data/Cuong/pickleball/data/image_ball_check"
    save_dir = "/data/Cuong/pickleball/data/image_check4"
    os.makedirs(save_dir, exist_ok=True)
    list_txt = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    for txt_path in list_txt:
        img_path = os.path.join(image_dir, os.path.splitext(os.path.basename(txt_path))[0] + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                xmin = int((xc - bw / 2) * w)
                ymin = int((yc - bh / 2) * h)
                xmax = int((xc + bw / 2) * w)
                ymax = int((yc + bh / 2) * h)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img, str(cls), (xmin, max(ymin - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        print(save_path)
        cv2.imwrite(save_path, img)

    print(f"Done. Saved to {save_dir}")
    
if __name__ == "__main__":
    check3()
