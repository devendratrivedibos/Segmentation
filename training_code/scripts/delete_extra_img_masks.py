import os
import cv2
import numpy as np

# ==========================
# Configuration
# ==========================
IMAGE_DIR = r"Z:\Devendra\ASPHALT\TRAIN_MIX\IMAGES"
MASK_DIR = r"Z:\Devendra\ASPHALT\TRAIN_MIX\MASKS"

# Delete if mask contains ONLY these class combinations
IMGS_TO_REMOVE = {
    (0,),      # Background only
    (0, 1),    # Background + Alligator only
    (0, 5),  # Background + Patches
}

COLOR_MAP = {
    (0, 0, 0): (0, "Background"),
    (255, 0, 0): (1, "Alligator"),
    (0, 0, 255): (2, "Transverse Crack"),
    (0, 255, 0): (3, "Longitudinal Crack"),
    (139, 69, 19): (4, "Pothole"),
    (255, 165, 0): (5, "Patches"),
    (255, 0, 255): (6, "Multiple Crack"),
    (0, 255, 255): (7, "Spalling"),
    (0, 128, 0): (8, "Corner Break"),
    (255, 100, 203): (9, "Sealed Joint Transverse"),
    (199, 21, 133): (10, "Sealed Joint Longitudinal"),
    (128, 0, 128): (11, "Punchout"),
    (112, 102, 255): (12, "Popout"),
    (255, 255, 255): (13, "Unclassified"),
    (255, 215, 0): (14, "Cracking"),
}

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

deleted = 0
kept = 0

for mask_name in os.listdir(MASK_DIR):

    if not mask_name.lower().endswith(VALID_EXT):
        continue

    mask_path = os.path.join(MASK_DIR, mask_name)

    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Could not read {mask_name}")
        continue

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    present_classes = set()

    for color, (class_id, _) in COLOR_MAP.items():
        if np.any(np.all(mask == color, axis=2)):
            present_classes.add(class_id)

    present_classes = tuple(sorted(present_classes))

    if present_classes in IMGS_TO_REMOVE:
        # Delete mask
        os.remove(mask_path)
        # Delete corresponding image (any supported extension)
        base = os.path.splitext(mask_name)[0]
        for ext in VALID_EXT:
            image_path = os.path.join(IMAGE_DIR, base + ext)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted Image : {os.path.basename(image_path)}")
                break
        print(f"Deleted Mask  : {mask_name}")
        print(f"Classes       : {present_classes}\n")
        deleted += 1

    else:
        kept += 1

print("=" * 40)
print(f"Deleted pairs : {deleted}")
print(f"Kept          : {kept}")
print("=" * 40)