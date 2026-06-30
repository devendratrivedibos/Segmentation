import os
import cv2
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# -------------------------------
# Folder path
# -------------------------------
folders = [
# r"Z:\Devendra\ASPHALT\ASPHALT_ACCEPTED\COMBINED_SPLITTED\TRAIN\MASKS_TRAIL",
    r"Z:\Devendra\ASPHALT\NEWTRAINING\Project\SegmentationClass"
]

image_files = []
for folder in folders:
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append((folder, f))

total_images = len(image_files)

# -------------------------------
# Color map (RGB) → (ID, Name)
# -------------------------------
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

# reverse lookup for faster name access
ID_TO_NAME = {v[0]: v[1] for v in COLOR_MAP.values()}

# minimum pixel size for object to be counted
MIN_PIXELS = 20


# -------------------------------
# Worker function
# -------------------------------
def process_image(COLOR_MAP, item):

    folder, filename = item
    path = os.path.join(folder, filename)

    image = cv2.imread(path)

    if image is None:
        return {
            "path": path,
            "error": True
        }

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_class_ids = set()
    instance_counts = defaultdict(int)

    # find unique colors in image
    pixels = image.reshape(-1, 3)
    unique_colors = set(map(tuple, pixels))

    unknown_colors = set()

    for color in unique_colors:

        if color not in COLOR_MAP:
            unknown_colors.add(color)
            continue

        cid, cname = COLOR_MAP[color]

        # mark image contains this class
        image_class_ids.add(cid)

        # binary mask
        mask = np.all(image == color, axis=-1).astype(np.uint8)

        if np.sum(mask) == 0:
            continue

        # connected components
        num_labels, labels = cv2.connectedComponents(mask)

        # skip label 0 (background)
        for label in range(1, num_labels):

            area = np.sum(labels == label)

            if area >= MIN_PIXELS:
                instance_counts[cid] += 1

    return {
        "path": path,
        "error": False,
        "image_class_ids": image_class_ids,
        "instance_counts": instance_counts,
        "unknown_colors": unknown_colors
    }


# -------------------------------
# Main multiprocessing
# -------------------------------
if __name__ == "__main__":

    print(f"\n🔍 Processing {total_images} images using {cpu_count()} CPUs...\n")

    results = []

    with Pool(processes=cpu_count()) as pool:

        for res in tqdm(
                pool.imap_unordered(
                    partial(process_image, COLOR_MAP),
                    image_files),
                total=total_images,
                desc="Processing images"):

            results.append(res)

    # -------------------------------
    # Aggregate Results
    # -------------------------------

    image_count_per_class = defaultdict(int)
    instance_count_per_class = defaultdict(int)

    invalid_color_files = {}
    unclassified_files = []

    for res in results:

        if res["error"]:
            print(f"⚠️ Could not read: {res['path']}")
            continue

        # image count
        for cid in res["image_class_ids"]:
            image_count_per_class[cid] += 1

        # individual object count
        for cid, count in res["instance_counts"].items():
            instance_count_per_class[cid] += count

        # unknown colors
        if res["unknown_colors"]:
            invalid_color_files[res["path"]] = res["unknown_colors"]

        # unclassified
        if 1 in res["image_class_ids"]:
            unclassified_files.append(res["path"])

    # -------------------------------
    # Unknown color report
    # -------------------------------

    if invalid_color_files:
        print("\n⚠️ Files with unknown colors:\n")
        for fname, colors in invalid_color_files.items():
            print(f"{fname}")
            print(f"Unknown colors: {sorted(colors)}\n")
    else:
        print("\n✅ All images contain only valid colors")

    if len(unclassified_files) > 0:
        print("\n📂 Images containing UNCLASSIFIED (ID=13):")
        # for f in unclassified_files:
        #     print(os.path.basename(f))

    print("\n===============================================")
    print(" CLASS DISTRIBUTION (Image count vs Object count)")
    print("===============================================\n")

    all_class_ids = sorted(set(image_count_per_class.keys()) | set(instance_count_per_class.keys()))

    for cid in all_class_ids:

        cname = ID_TO_NAME[cid]

        img_count = image_count_per_class.get(cid, 0)
        obj_count = instance_count_per_class.get(cid, 0)

        print(
            f"ID-{cid:2d} ({cname:25s}) : "
            f"{img_count:6d} images    "
            f"{obj_count:6d} objects")