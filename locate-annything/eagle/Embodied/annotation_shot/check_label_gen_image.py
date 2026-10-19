import os
import glob
import shutil

# ============================================================
# CONFIG
# ============================================================

root_dir = "/data/Cuong/pickleball/data/video_capcut"
output_dir = os.path.join(root_dir, "images")

# Các định dạng ảnh cần move
image_extensions = {".jpg"}

# Tạo folder images
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# TÌM TẤT CẢ FOLDER images_map Ở MỌI CẤP
# ============================================================

images_map_dirs = glob.glob(
    os.path.join(root_dir, "**", "images_map"),
    recursive=True
)

print(f"Found {len(images_map_dirs)} images_map folders")


# ============================================================
# MOVE ẢNH
# ============================================================

total_moved = 0
total_skipped = 0
total_duplicate = 0

for images_map_dir in images_map_dirs:

    # Không xử lý images_map nếu nó chính là output
    if os.path.abspath(images_map_dir) == os.path.abspath(output_dir):
        continue

    print(f"\nProcessing: {images_map_dir}")

    for filename in os.listdir(images_map_dir):

        source_path = os.path.join(
            images_map_dir,
            filename
        )

        # Bỏ qua folder/subfolder
        if not os.path.isfile(source_path):
            continue

        # Kiểm tra extension
        ext = os.path.splitext(filename)[1]

        if ext not in image_extensions:
            total_skipped += 1
            continue

        destination_path = os.path.join(
            output_dir,
            filename
        )

        # ----------------------------------------------------
        # XỬ LÝ TRÙNG TÊN
        # ----------------------------------------------------

        if os.path.exists(destination_path):

            total_duplicate += 1

            print(
                f"[DUPLICATE] {filename}"
            )

            continue

        # ----------------------------------------------------
        # MOVE
        # ----------------------------------------------------

        shutil.move(
            source_path,
            destination_path
        )

        total_moved += 1

        print(
            f"[MOVE] {filename}"
        )


# ============================================================
# THỐNG KÊ
# ============================================================

print("\n" + "=" * 70)
print("FINISHED")
print("=" * 70)

print(f"images_map folders : {len(images_map_dirs)}")
print(f"Images moved       : {total_moved}")
print(f"Skipped            : {total_skipped}")
print(f"Duplicate          : {total_duplicate}")

print(f"\nOutput folder:")
print(output_dir)