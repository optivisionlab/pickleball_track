import os
import re
import glob
import cv2
from collections import defaultdict
import xml.etree.ElementTree as ET


def read_objects(root):
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bndbox = obj.find("bndbox")
        bbox = {
            "xmin": int(bndbox.findtext("xmin")),
            "ymin": int(bndbox.findtext("ymin")),
            "xmax": int(bndbox.findtext("xmax")),
            "ymax": int(bndbox.findtext("ymax")),
        }
        keypoints = []
        kp_root = obj.find("keypoints")
        if kp_root is not None:
            for kp in kp_root.findall("keypoint"):
                keypoints.append({
                    "name": kp.findtext("name"),
                    "x": float(kp.findtext("x")),
                    "y": float(kp.findtext("y")),
                    "v": int(kp.findtext("v")),
                })
        objects.append({
            "name": name,
            "bbox": bbox,
            "keypoints": keypoints
        })
    return objects

if __name__ == "__main__":
    image_dir = "/data/pickleball/data/person/images"
    annotation_no_ball = "/data/pickleball/data/processed/no_ball/*.xml"
    annotation_ball = "/data/pickleball/data/processed/ball/*.xml"
    output_dir = "/data/videos"
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r"(.+)_(\d+)\.jpg$", re.IGNORECASE)
    video_frames = defaultdict(list)
    list_xml = glob.glob(annotation_no_ball) + glob.glob(annotation_ball)
    for xml_path in list_xml:
        xml_name = os.path.basename(xml_path)
        image_name = os.path.splitext(xml_name)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        m = pattern.match(image_name)
        if m:
            video_name = m.group(1)
            frame_idx = int(m.group(2))
            video_frames[video_name].append((frame_idx, image_path, xml_path))

    fps = 25
    for video_name, frames in video_frames.items():
        frames.sort(key=lambda x: x[0])

        first_image = cv2.imread(frames[0][1])
        if first_image is None:
            continue
        h, w = first_image.shape[:2]
        out_path = os.path.join(output_dir, video_name + ".mp4")
        writer = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w, h))
        for _, image_path, xml_path in frames:
            image = cv2.imread(image_path)
            if image is None:
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = read_objects(root)
            for obj in objects:
                if obj["name"] == "person":
                    cls = 0
                    color = (255, 0, 0)
                elif obj["name"] == "paddle":
                    cls = 1
                    color = (0, 255, 0)
                else:
                    cls = 2
                    color = (0, 0, 255)
                bbox = obj["bbox"]
                xmin = bbox["xmin"]
                ymin = bbox["ymin"]
                xmax = bbox["xmax"]
                ymax = bbox["ymax"]
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image, str(cls), (xmin, max(20, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                for kp in obj["keypoints"]:
                    print("len kp: ", len(obj["keypoints"]))
                    if kp["v"] == 0:
                        continue
                    x = int(kp["x"])
                    y = int(kp["y"])
                    cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
            writer.write(image)
        print(f"Done {video_name}")
        writer.release()