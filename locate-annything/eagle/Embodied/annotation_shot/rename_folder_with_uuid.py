import os
import re
import json
import uuid
import unicodedata
import glob
import shutil
from pathlib import Path

root_path = "/data/Cuong/pickleball/data/video_capcut"


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = name.split("_")[0]
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


person_dirs = []
for dirpath, dirnames, filenames in os.walk(root_path):
    if {"shot1"}.issubset(set(dirnames)):
        person_dirs.append(dirpath)

print(f"Found {len(person_dirs)} person folders.")


person_uuid = {}
folder_mapping = {}
for person in person_dirs:
    key = normalize_name(os.path.basename(person))
    if key not in person_uuid:
        person_uuid[key] = str(uuid.uuid4())
    folder_mapping[str(person)] = person_uuid[key]


video_mapping = {}

for person in person_dirs:
    uid = folder_mapping[person]
    shot_dir = os.path.join(person, 'shot1')
    video_dir = os.path.join(shot_dir, "videos")
    videos = sorted(glob.glob(f"{video_dir}/*.mp4"))
    print("Videos: ", len(videos))
    for idx, video in enumerate(videos, start=1):
        new_name = f"{uid}_{idx:06d}.mp4"
        new_path = os.path.join(video_dir, new_name)
        video_mapping[str(video)] = str(new_path)
        os.rename(video, new_path)
        print(f"Video: {video} -> {new_path}")

person_dirs.sort(key=lambda x: len(str(x)), reverse=True)
rename_folder_mapping = {}

for person in person_dirs:
    uid = folder_mapping[str(person)]
    new_path = Path(person).parent / uid
    if new_path.exists():
        print(f"Skip folder (already exists): {new_path}")
        continue
    os.rename(person, new_path)
    rename_folder_mapping[str(person)] = str(new_path)
    print(f"Folder: {person} -> {new_path}")



mapping = {
    "person_uuid": person_uuid,
    "folder_mapping": folder_mapping,
    "rename_folder_mapping": rename_folder_mapping,
    "video_mapping": video_mapping,
}

mapping_path = os.path.join(root_path, "uuid_mapping.json")

with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=4, ensure_ascii=False)

print("\n=======================================")
print("Completed!")
print(f"UUID mapping saved to:\n{mapping_path}")
print("=======================================")