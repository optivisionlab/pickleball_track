import os
import glob
import re
import xml.etree.ElementTree as ET


paddle_dir = "/data/Cuong/pickleball/data/video_yt/labels/labels_paddle"
ball_dir = "/data/Cuong/pickleball/data/video_yt/labels/labels_ball"
person_xml_dir = "/data/Cuong/pickleball/data/video_yt/labels/labels_person"

output_labels_dir = "/data/Cuong/pickleball/data/video_yt/labels/combine_label"
os.makedirs(output_labels_dir, exist_ok=True)


def convert_xml_to_yolo_pose(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.findtext("width"))
    img_h = float(size.findtext("height"))

    dw = 1.0 / img_w
    dh = 1.0 / img_h

    yolo_lines = []

    for obj in root.findall("object"):

        if obj.findtext("name") != "person":
            continue

        bbox = obj.find("bndbox")

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        xc = (xmin + xmax) / 2 * dw
        yc = (ymin + ymax) / 2 * dh
        bw = (xmax - xmin) * dw
        bh = (ymax - ymin) * dh

        line = f"2 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

        kp_root = obj.find("keypoints")

        if kp_root is not None:

            for kp in kp_root.findall("keypoint"):

                kx = float(kp.findtext("x")) * dw
                ky = float(kp.findtext("y")) * dh
                kv = int(kp.findtext("v"))

                line += f" {kx:.6f} {ky:.6f} {kv}"

        yolo_lines.append(line)

    return yolo_lines

paddle_files = sorted(glob.glob(os.path.join(paddle_dir, "*.txt")))

print(f"Total paddle labels: {len(paddle_files)}")

for p_path in paddle_files:

    filename = os.path.basename(p_path)

    combined_content = []
    with open(p_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            parts[0] = "0"
            combined_content.append(" ".join(parts))
    ball_path = os.path.join(ball_dir, filename)
    if os.path.exists(ball_path):
        with open(ball_path) as f:
            for line in f:
                parts = line.strip().split()
                parts[0] = "1"
                if len(parts) < 5:
                    continue
                combined_content.append(" ".join(parts))
    xml_path = os.path.join(
        person_xml_dir,
        filename.replace(".txt", ".xml"),
    )
    # if os.path.exists(xml_path):
    #     combined_content.extend(
    #         convert_xml_to_yolo_pose(xml_path)
    #     )
    output_path = os.path.join(
        output_labels_dir,
        filename,
    )
    with open(output_path, "w") as f:
        f.write("\n".join(combined_content))
    print(f"Saved: {filename}")
print("\nDone!")
print(f"Combined labels saved to:\n{output_labels_dir}")