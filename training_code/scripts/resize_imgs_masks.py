import os
import cv2

# ==========================
# Configuration
# ==========================
IMAGE_DIR = r"Z:\Devendra\ASPHALT\TRAIN_MIX\IMAGES"
MASK_DIR = r"Z:\Devendra\ASPHALT\TRAIN_MIX\MASKS"

TARGET_WIDTH = 419
TARGET_HEIGHT = 1024
# ==========================


def resize_folder(folder, is_mask=False):
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    resized_count = 0
    skipped_count = 0

    for filename in os.listdir(folder):
        if not filename.lower().endswith(valid_ext):
            continue

        filepath = os.path.join(folder, filename)

        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not read: {filepath}")
            continue

        h, w = img.shape[:2]

        # Skip if already the target size
        if w == TARGET_WIDTH and h == TARGET_HEIGHT:
            skipped_count += 1
            continue

        resized = cv2.resize(
            img,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=interpolation,
        )

        cv2.imwrite(filepath, resized)
        resized_count += 1
        print(f"Resized: {filename} ({w}x{h} -> {TARGET_WIDTH}x{TARGET_HEIGHT})")

    print(f"\nFolder: {folder}")
    print(f"Resized: {resized_count}")
    print(f"Skipped: {skipped_count}")


print("Resizing images...")
resize_folder(IMAGE_DIR, is_mask=False)

print("\nResizing masks...")
resize_folder(MASK_DIR, is_mask=True)

print("\nDone!")