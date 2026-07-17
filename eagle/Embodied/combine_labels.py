import os
import glob
import re
import xml.etree.ElementTree as ET

# ===========================
# Paths
# ===========================
paddle_dir = "/data/pickleball/data/labels/paddles"
ball_dir = "/data/pickleball/data/labels/balls"
person_xml_dir = "/data/pickleball/data/person/annotations"

output_labels_dir = "/data/pickleball/data/labels/combined_labels"
os.makedirs(output_labels_dir, exist_ok=True)


# ===========================
# Ball filename (+1 frame)
# ===========================
def get_ball_filename(paddle_txt_name):
    match = re.search(r"(.+)_(\d+)\.txt$", paddle_txt_name)

    if match:
        base, frame_idx = match.groups()
        return f"{base}_{int(frame_idx)+1:06d}.txt"

    return paddle_txt_name


# ===========================
# Rotate Ball bbox back
# ===========================
def rotate_bbox_back(xc, yc, bw, bh):
    """
    Ball label được tạo trên ảnh xoay 90° CCW.
    Chuyển lại về ảnh gốc.
    """

    xc_old = xc
    yc_old = yc

    xc = 1.0 - yc_old
    yc = xc_old

    bw, bh = bh, bw

    return xc, yc, bw, bh


# ===========================
# XML -> YOLO Pose
# ===========================
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


# ===========================
# Main
# ===========================
paddle_files = sorted(glob.glob(os.path.join(paddle_dir, "*.txt")))

print(f"Total paddle labels: {len(paddle_files)}")

for p_path in paddle_files:

    filename = os.path.basename(p_path)

    combined_content = []

    ##################################################
    # Paddle (Class 0)
    ##################################################
    with open(p_path) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) == 0:
                continue

            parts[0] = "0"

            combined_content.append(" ".join(parts))

    ##################################################
    # Ball (Class 1)
    ##################################################
    ball_filename = get_ball_filename(filename)
    ball_path = os.path.join(ball_dir, ball_filename)

    if os.path.exists(ball_path):

        with open(ball_path) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])

                xc, yc, bw, bh = rotate_bbox_back(
                    xc,
                    yc,
                    bw,
                    bh,
                )

                parts[0] = "1"
                parts[1] = f"{xc:.6f}"
                parts[2] = f"{yc:.6f}"
                parts[3] = f"{bw:.6f}"
                parts[4] = f"{bh:.6f}"

                combined_content.append(" ".join(parts))

    ##################################################
    # Person (Class 2 + Pose)
    ##################################################
    xml_path = os.path.join(
        person_xml_dir,
        filename.replace(".txt", ".xml"),
    )

    if os.path.exists(xml_path):

        combined_content.extend(
            convert_xml_to_yolo_pose(xml_path)
        )

    ##################################################
    # Save
    ##################################################
    output_path = os.path.join(
        output_labels_dir,
        filename,
    )

    with open(output_path, "w") as f:
        f.write("\n".join(combined_content))

    print(f"Saved: {filename}")

print("\nDone!")
print(f"Combined labels saved to:\n{output_labels_dir}")