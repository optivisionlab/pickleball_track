import os
import glob
import cv2
import numpy as np
from utils import *
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


if __name__ == "__main__":
    # ====== CONFIG ======
    image_dir = "/data/pickleball/data/ball/images"
    annotation_dir = "/data/pickleball/data/ball/annotations"
    label_to_predict = "yellow ball"
    model_path = "/data/pickleball/sam3.pt"

    os.makedirs(image_dir, exist_ok = True)
    os.makedirs(annotation_dir, exist_ok = True)
    # ====== LOAD MODEL ======
    print("[INFO] Loading SAM3 model...")
    processor = Sam3Processor(build_sam3_image_model(checkpoint_path=model_path))

    list_video = sorted(
        glob.glob("/data/raw/*.mp4", recursive=True)
        + glob.glob("/data/raw/*.MOV", recursive=True)
    )
    video_start = 0
    for video_idx, video_path in enumerate(list_video[video_start:]):
        print(f"\n[INFO] Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_idx = 0
        frame_start = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            # if video_start == 41 and frame_idx - 1 <= frame_start:
            #     continue
            frame_idx += 1
            height, width = frame.shape[:2]
            depth = frame.shape[2] if len(frame.shape) == 3 else 1

            image_name = f"{video_name}_{frame_idx:06d}.jpg"
            xml_name = f"{video_name}_{frame_idx:06d}.xml"
            image_path = os.path.join(image_dir, image_name)
            xml_path = os.path.join(annotation_dir, xml_name)
            cv2.imwrite(image_path, frame)
            # Chạy SAM3 inference
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            state = processor.set_image(image_pil)
            results = processor.set_text_prompt(state=state, prompt=label_to_predict)

            boxes = results["boxes"].cpu().numpy()
            objects = []
            for box in boxes:
                x_min, y_min, x_max, y_max = max(0, int(round(box[0]))- 10), max(0, int(round(box[1])) - 10), min(width - 1, int(round(box[2])) + 10), min(height - 1, int(round(box[3])) + 10)
                objects.append({
                    "name": "ball",
                    "xmin": x_min,
                    "ymin": y_min,
                    "xmax": x_max,
                    "ymax": y_max
                })
            #Lưu XML
            create_voc_xml(
                image_path=image_name,
                width=width,
                height=height,
                depth=depth,
                objects=objects,
                xml_path=xml_path
            )

            print(f"[INFO] Saved: {xml_name} | objects={len(objects)}")

        cap.release()

    print("\n[DONE] Finished extracting images and annotations.")