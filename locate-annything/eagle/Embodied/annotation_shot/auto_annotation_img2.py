import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from load_model import worker
from pathlib import Path
import glob
import cv2
import re


class_map = {"black pickleball paddle in the player's hand": 0, "small yellow ball used for playing pickleball": 1}

if __name__ == "__main__":
    # ====== CONFIG ======
    shot = "/data/Cuong/pickleball/data/video_capcut"
    prompt = "small yellow ball used for playing pickleball"
    padding_ratio = 0.1

    list_images = sorted(
        glob.glob(os.path.join(shot, "**/*.jpg"), recursive=True)
    )
    print("Num of Image: ", len(list_images))
    start = 0
    for image_idx, image_path in enumerate(list_images[start:]):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_dir_path = Path(image_path).parent.parent / "labels_ball"
        os.makedirs(label_dir_path, exist_ok = True)
        number_object = 0
        frame = cv2.imread(image_path)
        h_ori, w_ori, d = frame.shape
        txt_name = f"{image_name}.txt"
        txt_path = os.path.join(label_dir_path, txt_name)

        #resize image and convert to PIL image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = 448 / max(h_ori, w_ori)
        new_w = int(round(w_ori * scale))
        new_h = int(round(h_ori * scale))
        image_rs = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img_pil = Image.fromarray(image_rs)
        
        #inference
        lines = []
        answer = worker.ground_multi(img_pil, prompt)["answer"]
        obj_pattern = r"<ref>(.*?)</ref>(.*?)(?=<ref>|<\|im_end\|>|$)"
        
        #save labels
        for label, content in re.findall(obj_pattern, answer, flags=re.DOTALL):
            cls = class_map[label]
            for x1, y1, x2, y2 in re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", content ):
                x1 = int(x1) / 1000
                y1 = int(y1) / 1000
                x2 = int(x2) / 1000
                y2 = int(y2) / 1000
                bw = x2 -x1
                bh = y2 - y1
                x1 = max(0, x1 - bw * padding_ratio / 2)
                y1 = max(0, y1 - bh * padding_ratio / 2)
                x2 = min(1, x2 + bw * padding_ratio / 2)
                y2 = min(1, y2 + bh * padding_ratio / 2)
                xcenter = (x1 + x2) / 2.0
                ycenter = (y1 + y2) / 2.0
                width = (x2 - x1)
                height = (y2 - y1)
                lines.append(
                    f"{cls} "
                    f"{xcenter:.6f} "
                    f"{ycenter:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )
                number_object += 1
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[INFO] Saved: {txt_name} | objects={number_object} | answer: {answer}")
    print("\n[DONE] Finished extracting images and annotations.")
