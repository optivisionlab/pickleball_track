import os
import glob
import shutil

# ============================================================
# CONFIG
# ============================================================

root_dir = "/data/Cuong/pickleball/data/video_capcut"


# ============================================================
# TÌM TẤT CẢ labels_paddles
# ============================================================

list_paddle_dirs = glob.glob(
    os.path.join(root_dir, "**", "labels_paddles"),
    recursive=True
)

print("Found:", len(list_paddle_dirs), "label folders")


# ============================================================
# ĐỌC LABEL
# ============================================================

def read_txt(path, new_class=None):
    """
    Đọc file YOLO txt.

    Return:
        list[str]: các dòng label hợp lệ.

    File không tồn tại / rỗng -> []
    """

    if not os.path.isfile(path):
        return []

    lines = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Bỏ dòng rỗng
            if not line:
                continue

            parts = line.split()
            parts = parts
            # YOLO bbox phải có ít nhất:
            # class x_center y_center width height
            if len(parts) < 5:
                continue

            # Kiểm tra class
            try:
                int(parts[0])
            except ValueError:
                continue

            # Kiểm tra các tọa độ
            try:
                coords = [float(x) for x in parts[1:5]]
            except ValueError:
                continue

            # Nếu muốn đổi class
            if new_class is not None:
                parts[0] = str(new_class)

            lines.append(" ".join(parts))

    return lines


# ============================================================
# XỬ LÝ
# ============================================================

for paddle_dir in list_paddle_dirs:

    parent_dir = os.path.dirname(paddle_dir)

    ball_dir = os.path.join(parent_dir, "labels_ball")
    person_dir = os.path.join(parent_dir, "labels_persons")
    output_dir = os.path.join(root_dir, "labels")

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Lấy danh sách tất cả filename
    # --------------------------------------------------------

    paddle_files = {
        os.path.basename(f)
        for f in glob.glob(
            os.path.join(paddle_dir, "*.txt")
        )
    }

    ball_files = {
        os.path.basename(f)
        for f in glob.glob(
            os.path.join(ball_dir, "*.txt")
        )
    }

    person_files = {
        os.path.basename(f)
        for f in glob.glob(
            os.path.join(person_dir, "*.txt")
        )
    }

    # ========================================================
    # FILE PHẢI TỒN TẠI Ở CẢ 3 FOLDER
    # ========================================================

    candidate_files = (
        paddle_files
        & ball_files
        & person_files
    )

    # ========================================================
    # COUNTER
    # ========================================================

    valid_count = 0
    skipped_missing = 0
    skipped_empty = 0
    skipped_invalid = 0

    # ========================================================
    # XỬ LÝ TỪNG FRAME
    # ========================================================

    for name in sorted(candidate_files):

        paddle_path = os.path.join(
            paddle_dir, name
        )

        ball_path = os.path.join(
            ball_dir, name
        )

        person_path = os.path.join(
            person_dir, name
        )

        # ----------------------------------------------------
        # Đọc 3 loại label
        # ----------------------------------------------------

        paddle_labels = read_txt(
            paddle_path,
            new_class=0
        )
        
        ball_labels = read_txt(
            ball_path,
            new_class=1
        )

        person_labels = read_txt(
            person_path,
            new_class=2
        )

        # ====================================================
        # PHẢI CÓ ĐỦ CẢ 3 LOẠI
        # ====================================================

        if not paddle_labels:
            skipped_empty += 1
            continue

        if not ball_labels:
            skipped_empty += 1
            continue

        if not person_labels:
            skipped_empty += 1
            continue

        # ====================================================
        # MERGE
        # ====================================================

        merged = (
            paddle_labels
            + ball_labels
        )

        # ----------------------------------------------------
        # Ghi file
        # ----------------------------------------------------

        output_path = os.path.join(
            output_dir,
            name
        )

        with open(output_path, "w") as f:
            f.write("\n".join(merged))

        valid_count += 1

    # ========================================================
    # THỐNG KÊ FILE BỊ THIẾU 1 TRONG 3
    # ========================================================

    all_files = (
        paddle_files
        | ball_files
        | person_files
    )

    missing_files = all_files - candidate_files

    skipped_missing = len(missing_files)

    # ========================================================
    # IN THỐNG KÊ
    # ========================================================

    print("\n" + "=" * 70)
    print(f"Processing: {parent_dir}")
    print("=" * 70)

    print(f"Paddle files       : {len(paddle_files)}")
    print(f"Ball files         : {len(ball_files)}")
    print(f"Person files       : {len(person_files)}")
    print(f"Candidate files    : {len(candidate_files)}")
    print(f"Valid merged files : {valid_count}")
    print(f"Missing label file : {skipped_missing}")
    print(f"Empty/invalid      : {skipped_empty}")
    print(f"Output             : {output_dir}")


print("\nFinished!")
