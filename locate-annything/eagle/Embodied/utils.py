import xml.etree.ElementTree as ET
import os
import copy
import re
import subprocess
import json


class_map = {"black pickleball paddle in the player's hand": 0, "the small yellow ball used for playing pickleball": 1}


def get_video_rotation(video_path):
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        info = json.loads(result.stdout)
        for stream in info.get("streams", []):
            side_data = stream.get("side_data_list", [])
            for item in side_data:
                if "rotation" in item:
                    rot = int(item["rotation"])
                    return (rot + 360) % 360
            tags = stream.get("tags", {})
            if "rotate" in tags:
                rot = int(tags["rotate"])
                return (rot + 360) % 360
    except Exception as e:
        print(f"[DEBUG] Error getting rotation: {e}")
        pass
    return 0


def parse_answer( answer, original_size):
    w_ori, h_ori = original_size
    objects = []
    obj_pattern = r"<ref>(.*?)</ref>(.*?)(?=<ref>|<\|im_end\|>|$)"
    for label, content in re.findall(obj_pattern, answer, flags=re.DOTALL):
        for x1, y1, x2, y2 in re.findall(
            r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
            content
        ):
            x1 = int(x1) / 1000 * w_ori
            y1 = int(y1) / 1000 * h_ori
            x2 = int(x2) / 1000 * w_ori
            y2 = int(y2) / 1000 * h_ori
            objects.append({
                "name": label,
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2
            })
    return objects

def create_txt_file(txt_path, objects, img_size):
    w, h = img_size
    with open(txt_path, "a", encoding="utf8") as f:
        for obj in objects:
            cls = class_map[obj["name"]]

            xmin = obj["xmin"]
            ymin = obj["ymin"]
            xmax = obj["xmax"]
            ymax = obj["ymax"]
            
            xcenter = (xmin + xmax) / 2.0
            ycenter = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin

            xcenter_nor = xcenter / w
            ycenter_nor = ycenter / h
            width_nor = width / w
            height_nor = height / h
            
            f.write(
                f"{cls} "
                f"{xcenter_nor:.6f} "
                f"{ycenter_nor:.6f} "
                f"{width_nor:.6f} "
                f"{height_nor:.6f}\n"
            )
        
def create_voc_xml(image_path, width, height, depth, objects, xml_path):
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(os.path.dirname(image_path))

    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(image_path)

    path = ET.SubElement(annotation, "path")
    path.text = image_path

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    w = ET.SubElement(size, "width")
    w.text = str(width)
    h = ET.SubElement(size, "height")
    h.text = str(height)
    d = ET.SubElement(size, "depth")
    d.text = str(depth)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for obj in objects:
        obj_tag = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj_tag, "name")
        name.text = obj["name"]

        pose = ET.SubElement(obj_tag, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj_tag, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj_tag, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj_tag, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(obj["xmin"]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(obj["ymin"]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(obj["xmax"]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(obj["ymax"]))

    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def get_objects(root):
    return [copy.deepcopy(obj) for obj in root.findall("object")]
