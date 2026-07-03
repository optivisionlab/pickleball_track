from ultralytics import YOLO
import xml.etree.ElementTree as ET
import os
import glob
import cv2


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def create_voc_xml(image_path, width, height, depth, objects, xml_path):

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = os.path.basename(
        os.path.dirname(image_path)
    )

    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    ET.SubElement(annotation, "path").text = image_path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "YOLO-Pose"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    for obj in objects:

        obj_tag = ET.SubElement(annotation, "object")

        ET.SubElement(obj_tag, "name").text = obj["name"]
        ET.SubElement(obj_tag, "pose").text = "Unspecified"
        ET.SubElement(obj_tag, "truncated").text = "0"
        ET.SubElement(obj_tag, "difficult").text = "0"

        xmin, ymin, xmax, ymax = obj["bbox"]

        bndbox = ET.SubElement(obj_tag, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(round(xmin - 20)))
        ET.SubElement(bndbox, "ymin").text = str(int(round(ymin - 20)))
        ET.SubElement(bndbox, "xmax").text = str(int(round(xmax + 20)))
        ET.SubElement(bndbox, "ymax").text = str(int(round(ymax + 20)))

        kp_root = ET.SubElement(obj_tag, "keypoints")

        for idx, (x, y, v) in enumerate(obj["keypoints"]):

            kp = ET.SubElement(
                kp_root,
                "keypoint",
                id=str(idx),
                name=COCO_KEYPOINT_NAMES[idx]
            )

            ET.SubElement(kp, "x").text = f"{x:.2f}"
            ET.SubElement(kp, "y").text = f"{y:.2f}"
            ET.SubElement(kp, "v").text = str(int(v))

    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    image_dir = "/data/pickleball/data/person/images"
    annotation_dir = "/data/pickleball/data/person/annotations"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    print("Loading model...")
    model = YOLO("yolo11s-pose.pt")
    video_list = sorted(glob.glob("/data/raw/*.mp4")+ glob.glob("/data/raw/*.MOV"))
    for video_path in video_list:
        video_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]
        results = model(
            source=video_path,
            stream=True,
            batch=32,
            device=0,
            conf=0.9,
            verbose=False,
        )
        for frame_idx, result in enumerate(results):
            image = result.orig_img
            h, w, d = image.shape
            image_name = f"{video_name}_{frame_idx:06d}.jpg"
            image_path = os.path.join(image_dir, image_name)
            # cv2.imwrite(image_path, image)
            objects = []
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                keypoints = result.keypoints.data.cpu().numpy()
                for box, kps in zip(boxes, keypoints):
                    objects.append(
                        {
                            "name": "person",
                            "bbox": box.tolist(),
                            "keypoints": kps.tolist(),
                        }
                    )

            xml_path = os.path.join(annotation_dir, f"{video_name}_{frame_idx:06d}.xml")

            create_voc_xml(
                image_path=image_path,
                width=w,
                height=h,
                depth=d,
                objects=objects,
                xml_path=xml_path,
            )
        print(f"Đang xử lý {video_path}")

