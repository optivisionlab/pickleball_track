import os
import xml.etree.ElementTree as ET
from utils import *


if __name__ == "__main__":
    dir_person_annotations = "/data/pickleball/data/person/annotations"
    dir_paddle_annotations = "/data/pickleball/data/paddle/annotations"
    dir_ball_annotations = "/data/pickleball/data/ball/annotations"
    out_data_dir_with_ball = "/data/pickleball/data/processed/ball"
    out_data_dir_no_ball = "/data/pickleball/data/processed/no_ball"

    os.makedirs(out_data_dir_with_ball, exist_ok=True)
    os.makedirs(out_data_dir_no_ball, exist_ok=True)

    for xml_file in os.listdir(dir_paddle_annotations):
        xml_person = os.path.join(dir_person_annotations, xml_file)
        xml_paddle = os.path.join(dir_paddle_annotations, xml_file)
        xml_ball = os.path.join(dir_ball_annotations, xml_file)
        if not (os.path.exists(xml_person)and os.path.exists(xml_paddle)and os.path.exists(xml_ball)):
            continue

        # đọc xml
        tree_person = ET.parse(xml_person)
        root_person = tree_person.getroot()
        tree_paddle = ET.parse(xml_paddle)
        root_paddle = tree_paddle.getroot()
        tree_ball = ET.parse(xml_ball)
        root_ball = tree_ball.getroot()
        objects_person = get_objects(root_person)
        objects_paddle = get_objects(root_paddle)
        objects_ball = get_objects(root_ball)
        if len(objects_person) < 1 or len(objects_paddle) != 1:
            continue
        
        # tạo xml mới từ file person
        new_root = copy.deepcopy(root_person)
        # xóa toàn bộ object cũ
        for obj in new_root.findall("object"):
            new_root.remove(obj)
        # thêm object
        for obj in objects_person:
            new_root.append(obj)
        for obj in objects_paddle:
            new_root.append(obj)
        for obj in objects_ball:
            new_root.append(obj)
        new_tree = ET.ElementTree(new_root)

        # phân loại thư mục lưu
        if len(objects_ball) == 0:
            save_path = os.path.join(out_data_dir_no_ball, xml_file)
        else:
            save_path = os.path.join(out_data_dir_with_ball, xml_file)
        if not os.path.exists(save_path):
            new_tree.write(save_path, encoding="utf-8", xml_declaration=True)

print("Done!")