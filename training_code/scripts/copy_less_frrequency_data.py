import cv2
import os
import numpy as np
import shutil

# Paths
mask_folder = r"Z:\Devendra\ASPHALT\TRAIN_MIX\MASKS"
output_mask_folder = r"Z:\Devendra\ASPHALT\TRAIN_MIX\MASKS"

image_folder = r"Z:\Devendra\ASPHALT\TRAIN_MIX\IMAGES"
output_image_folder = r"Z:\Devendra\ASPHALT\TRAIN_MIX\IMAGES"

os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# --- Target classes (IDs) ---
target_classes = {2,3}

# --- Color map (RGB) → (ID, Name) ---
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
matching_images = []

# Colors corresponding to target classes (RGB)
target_colors = [color for color, (cls_id, cls_name) in COLOR_MAP.items() if cls_id in target_classes]


def find_image_file(image_folder, base_name):
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = os.path.join(image_folder, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


# --- Find matching masks ---
for filename in os.listdir(mask_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        mask_path = os.path.join(mask_folder, filename)

        # Read mask in RGB
        mask = cv2.imread(mask_path)
        if mask is None:
            print(f"⚠ Could not read mask: {mask_path}")
            continue

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Check for target colors
        if any(np.any(np.all(mask == color, axis=-1)) for color in target_colors):
            matching_images.append(filename)

            name, _ = os.path.splitext(filename)

            mask_src = os.path.join(mask_folder, filename)
            image_src = find_image_file(image_folder, name)

            if image_src is None:
                print(f"⚠ Image for {filename} not found, skipping.")
                continue

            _, mask_ext = os.path.splitext(filename)
            _, image_ext = os.path.splitext(image_src)

            # Copy mask duplicates
            for i in range(1, 3):
                shutil.copy2(mask_src, os.path.join(output_mask_folder, f"{name}_copy{i}{mask_ext}"))
                shutil.copy2(image_src, os.path.join(output_image_folder, f"{name}_copy{i}{image_ext}"))

            print(f"✔ Copied duplicates for: {filename}")
