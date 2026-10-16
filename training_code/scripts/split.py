
import random
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_FOLDER = Path(r"C:\Users\Admin\Downloads\Dataset\IMAGES")
MASK_FOLDER = Path(r"C:\Users\Admin\Downloads\Dataset\MASKS")

OUTPUT_FOLDER = Path(r"C:\Users\Admin\Downloads\Dataset\SPLIT")

TRAIN_RATIO = 0.90
RANDOM_SEED = 42

# True = move original files
# False = copy files (recommended)
MOVE_FILES = False

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

TRAIN_IMAGES = OUTPUT_FOLDER / "TRAIN" / "IMAGES"
TRAIN_MASKS = OUTPUT_FOLDER / "TRAIN" / "MASKS"

VAL_IMAGES = OUTPUT_FOLDER / "VAL" / "IMAGES"
VAL_MASKS = OUTPUT_FOLDER / "VAL" / "MASKS"

for folder in [
    TRAIN_IMAGES,
    TRAIN_MASKS,
    VAL_IMAGES,
    VAL_MASKS
]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# GET MATCHING IMAGES AND MASKS
# ============================================================

def get_files(folder):
    files = {}

    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.stem.lower()

            if key in files:
                raise ValueError(
                    f"Duplicate filename stem: {path.stem}"
                )

            files[key] = path

    return files


images = get_files(IMAGE_FOLDER)
masks = get_files(MASK_FOLDER)

matched_keys = sorted(set(images) & set(masks))

missing_masks = sorted(set(images) - set(masks))
missing_images = sorted(set(masks) - set(images))

print("=" * 55)
print("DATASET INFORMATION")
print("=" * 55)

print(f"Total images       : {len(images)}")
print(f"Total masks        : {len(masks)}")
print(f"Matched pairs      : {len(matched_keys)}")
print(f"Images without mask: {len(missing_masks)}")
print(f"Masks without image: {len(missing_images)}")

if missing_masks:
    print("\nImages without masks:")
    for key in missing_masks:
        print(" ", images[key].name)

if missing_images:
    print("\nMasks without images:")
    for key in missing_images:
        print(" ", masks[key].name)

if not matched_keys:
    raise RuntimeError("No matching image-mask pairs found!")

# ============================================================
# SHUFFLE AND SPLIT
# ============================================================

rng = random.Random(RANDOM_SEED)
rng.shuffle(matched_keys)

train_count = int(len(matched_keys) * TRAIN_RATIO)

train_keys = matched_keys[:train_count]
val_keys = matched_keys[train_count:]

print("\n" + "=" * 55)
print("SPLIT INFORMATION")
print("=" * 55)

print(f"Train pairs: {len(train_keys)}")
print(f"Val pairs  : {len(val_keys)}")

# ============================================================
# COPY / MOVE FILES
# ============================================================

def transfer_files(keys, image_dest, mask_dest):

    operation = shutil.move if MOVE_FILES else shutil.copy2

    for key in keys:
        image_path = images[key]
        mask_path = masks[key]

        operation(
            str(image_path),
            str(image_dest / image_path.name)
        )

        operation(
            str(mask_path),
            str(mask_dest / mask_path.name)
        )


transfer_files(train_keys, TRAIN_IMAGES, TRAIN_MASKS)
transfer_files(val_keys, VAL_IMAGES, VAL_MASKS)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("DATASET SPLIT COMPLETED")
print("=" * 55)

print(f"Train images: {len(train_keys)}")
print(f"Train masks : {len(train_keys)}")
print(f"Val images  : {len(val_keys)}")
print(f"Val masks   : {len(val_keys)}")

print(f"\nOutput saved at: {OUTPUT_FOLDER}")
