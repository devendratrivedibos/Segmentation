import os
import cv2
import shutil
from multiprocessing import Pool, cpu_count

# --- Paths ---
BASE_DIR = r"Z:\Devendra\ASPHALT\TRAIN_MIX"

img_dir = r"Z:\Devendra\ASPHALT\IMAGES"
mask_dir = r"Z:\Devendra\ASPHALT\MASKS_OLD"

copy_mask_dir = os.path.join(BASE_DIR, "MASKS")
copy_image_dir = os.path.join(BASE_DIR, "IMAGES")

os.makedirs(copy_mask_dir, exist_ok=True)
os.makedirs(copy_image_dir, exist_ok=True)

# --- Color map (RGB) → Class ID ---
COLOR_MAP = {
    (0, 0, 0): 0,         # Background
    (255, 0, 0): 1,       # Alligator
    (0, 0, 255): 2,       # Transverse Crack
    (0, 255, 0): 3,       # Longitudinal Crack
    (139, 69, 19): 4,     # Pothole
    (255, 165, 0): 5,     # Patches
    (255, 0, 255): 6,     # Multiple Crack
    (0, 255, 255): 7,     # Spalling
    (0, 128, 0): 8,       # Corner Break
    (255, 100, 203): 9,   # Sealed Joint - T
    (199, 21, 133): 10,   # Sealed Joint - L
    (128, 0, 128): 11,    # Punchout
    (112, 102, 255): 12,  # Popout
    (255, 255, 255): 13,  # Unclassified
    (255, 215, 0): 14,    # Cracking
}

COLOR_TO_ID = COLOR_MAP

COPY_SETS = [
    {0, 1},
    {0, 4},
    {0, 5},
    {0, 4, 5},
    {0, 1, 4},
    {0, 1, 5},
    {0, 1, 4, 5},
]


def find_image(mask_name):
    """Find corresponding image with same filename."""
    base_name = os.path.splitext(mask_name)[0]

    for ext in [".png", ".jpg", ".jpeg"]:
        img_path = os.path.join(img_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path

    return None


def process_mask(mask_name):
    if not mask_name.lower().endswith(".png"):
        return None

    mask_path = os.path.join(mask_dir, mask_name)

    mask = cv2.imread(mask_path)
    if mask is None:
        return None

    # Convert BGR -> RGB
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Find unique RGB colors
    unique_colors = {tuple(c) for c in mask_rgb.reshape(-1, 3)}

    # Convert colors to class IDs
    unique_ids = {
        COLOR_TO_ID[c]
        for c in unique_colors
        if c in COLOR_TO_ID
    }

    # Copy only selected combinations
    if unique_ids in COPY_SETS:

        img_path = find_image(mask_name)

        try:
            shutil.copy2(mask_path, os.path.join(copy_mask_dir, mask_name))

            if img_path:
                shutil.copy2(
                    img_path,
                    os.path.join(copy_image_dir, os.path.basename(img_path))
                )

            return f"Copied: {mask_name} -> {sorted(unique_ids)}"

        except Exception as e:
            return f"Error: {mask_name}: {e}"

    return None


if __name__ == "__main__":

    all_masks = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    ]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_mask, all_masks)

    copied = [r for r in results if r]

    print(f"\n✅ Done. Copied {len(copied)} mask-image pairs.\n")

    for r in copied:
        print(r)