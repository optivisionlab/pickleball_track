import glob
import subprocess
from pathlib import Path
from tqdm import tqdm
import shutil
import os


shot_dir = Path("/data/Cuong/pickleball/data/video_capcut")

list_video = sorted(list(shot_dir.rglob("*.mp4")))
print(len(list_video))
start = 0
for idx, video_path in enumerate(tqdm(list_video[start:])):
    video_name = video_path.stem
    shot1_dir = video_path.parent.parent
    image_dir = shot1_dir / "images"
    audio_dir = shot1_dir / "audios"
    if os.path.exists(image_dir) and idx == 0:
        shutil.rmtree(image_dir)
    if os.path.exists(audio_dir) and idx == 0:
        shutil.rmtree(audio_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{video_name}.wav"
    cmd_frames = [
        "ffmpeg",
        "-y",  
        "-i", str(video_path),
        str(image_dir / f"{video_name}_%06d.jpg")
    ]
    subprocess.run(
        cmd_frames,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    cmd_audio = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(audio_path),
    ]
    subprocess.run(
        cmd_audio,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )