from ultralytics import YOLO
from pathlib import Path
import os
import glob


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


if __name__ == "__main__":

    # ============================================================
    # CONFIG
    # ============================================================

    video_shot = "/data/Cuong/pickleball/data/video_capcut"

    model = YOLO(os.path.join(os.environ.get("PB_ARTIFACTS", "/data/Cuong/pickleball/artifacts"), "weights", "yolo11s-pose.pt"))

    # ============================================================
    # TÌM TẤT CẢ VIDEO
    # ============================================================

    list_video = sorted(
        glob.glob(
            os.path.join(
                video_shot,
                "**",
                "*.mp4"
            ),
            recursive=True
        )
        +
        glob.glob(
            os.path.join(
                video_shot,
                "**",
                "*.mov"
            ),
            recursive=True
        )
        +
        glob.glob(
            os.path.join(
                video_shot,
                "**",
                "*.MOV"
            ),
            recursive=True
        )
    )

    print("Len video:", len(list_video))

    # ============================================================
    # PROCESS TỪNG VIDEO
    # ============================================================

    for video_path in list_video:

        video_path = Path(video_path)

        # --------------------------------------------------------
        # Tên video không có extension
        #
        # abc.mp4 -> abc
        # --------------------------------------------------------

        video_name = video_path.stem

        # --------------------------------------------------------
        # Folder chứa video
        #
        # .../xxx/videos/video.mp4
        #
        # => .../xxx/videos
        # --------------------------------------------------------

        videos_dir = video_path.parent

        # --------------------------------------------------------
        # labels_persons cùng cấp với folder videos
        #
        # .../xxx/videos/
        # .../xxx/labels_persons/
        # --------------------------------------------------------

        label_dir_path = videos_dir.parent / "labels_persons"

        label_dir_path.mkdir(
            parents=True,
            exist_ok=True
        )

        print("\n" + "=" * 80)
        print("Processing:", video_path)
        print("Labels:", label_dir_path)

        # ========================================================
        # YOLO INFERENCE
        # ========================================================

        results = model(
            source=str(video_path),
            stream=True,
            batch=16,
            device=0,
            conf=0.8,
            verbose=False,
        )

        # ========================================================
        # PROCESS FRAME
        # ========================================================

        for frame_idx, result in enumerate(
            results,
            start=1
        ):

            image = result.orig_img

            h, w = image.shape[:2]

            lines = []

            txt_name = (
                f"{video_name}_"
                f"{frame_idx:06d}.txt"
            )

            txt_path = label_dir_path / txt_name

            # ====================================================
            # DETECTION
            # ====================================================

            if (
                result.boxes is not None
                and result.keypoints is not None
            ):

                boxes = (
                    result.boxes
                    .xywh
                    .cpu()
                    .numpy()
                )

                keypoints = (
                    result.keypoints
                    .data
                    .cpu()
                    .numpy()
                )

                # ------------------------------------------------
                # Mỗi person
                # ------------------------------------------------

                for box, kpts in zip(
                    boxes,
                    keypoints
                ):

                    x, y, bw, bh = box

                    # ------------------------------------------------
                    # YOLO Pose format
                    #
                    # class x_center y_center width height
                    # ------------------------------------------------

                    line = [
                        2,
                        x / w,
                        y / h,
                        bw / w,
                        bh / h,
                    ]

                    # ------------------------------------------------
                    # 17 COCO KEYPOINTS
                    # ------------------------------------------------

                    for (
                        kx,
                        ky,
                        conf
                    ) in kpts:

                        vis = (
                            2
                            if conf > 0.5
                            else 0
                        )

                        line.extend([
                            kx / w,
                            ky / h,
                            vis,
                        ])

                    # ------------------------------------------------
                    # Convert line -> string
                    # ------------------------------------------------

                    lines.append(
                        " ".join(
                            (
                                f"{v:.6f}"
                                if isinstance(
                                    v,
                                    float
                                )
                                else str(v)
                            )
                            for v in line
                        )
                    )

            # ====================================================
            # SAVE LABEL
            #
            # Không detect person:
            # vẫn tạo file .txt rỗng
            # ====================================================

            with open(
                txt_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(lines)
                )

        print(
            f"Finished: {video_path}"
        )

    print("\n" + "=" * 80)
    print("DONE")