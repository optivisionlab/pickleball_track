import xml.etree.ElementTree as ET
import os
import copy


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
