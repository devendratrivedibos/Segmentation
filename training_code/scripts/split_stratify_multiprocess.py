import os
import shutil
import time
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== CONFIG =====================
images_dir = r"Z:\Devendra\ASPHALT\NEWTRAINING\IMAGES"
masks_dir = r"Z:\Devendra\ASPHALT\NEWTRAINING\MASKS"
output_dir = r"Z:\Devendra\ASPHALT\NEWTRAINING\SPLIT"

test_size = 0.01
val_size = 0.20
random_state = 21
max_workers = 8

# --- Color map (RGB) → ID ---
COLOR_MAP = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 0, 255): 2,
    (0, 255, 0): 3,
    (139, 69, 19): 4,
    (255, 165, 0): 5,
    (255, 0, 255): 6,
    (0, 255, 255): 7,
    (0, 128, 0): 8,
    (255, 100, 203): 9,
    (199, 21, 133): 10,
    (128, 0, 128): 11,
    (112, 102, 255): 12,
    (255, 255, 255): 13,
    (255, 215, 0): 14,
}

NUM_CLASSES = len(COLOR_MAP)

# ===================== LUT =====================
lut = np.full((256, 256, 256), -1, dtype=np.int16)
for rgb, cls_id in COLOR_MAP.items():
    lut[rgb] = cls_id

# ===================== FUNCTIONS =====================
def get_label_vector(mask_path):
    """Convert mask → multilabel vector"""
    mask = np.array(Image.open(mask_path).convert("RGB"))
    mask_cls = lut[mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]]
    unique_cls = np.unique(mask_cls)
    label_vec = np.zeros(NUM_CLASSES, dtype=int)
    for cls in unique_cls:
        if cls >= 0:
            label_vec[cls] = 1
    return label_vec


def process_mask_file(f):
    """Process one mask file: return (filename, label_vector)"""
    if not f.endswith(".png"):
        return None
    mask_path = os.path.join(masks_dir, f)
    label_vec = get_label_vector(mask_path)
    base = os.path.splitext(f)[0]
    img_name = None
    if os.path.exists(os.path.join(images_dir, base + ".jpg")):
        img_name = base + ".jpg"
    elif os.path.exists(os.path.join(images_dir, base + ".png")):
        img_name = base + ".png"
    if img_name:
        return img_name, label_vec
    return None


def safe_link_or_copy(src, dst):
    """Try to create a hard link; fall back to fast copy if needed"""
    try:
        if os.path.exists(dst):
            return
        os.link(src, dst)
    except Exception:
        shutil.copy(src, dst)


def copy_pair(f, split):
    """Copy or link image & mask pair"""
    base = os.path.splitext(f)[0]
    img_src = os.path.join(images_dir, f)
    mask_src = os.path.join(masks_dir, base + ".png")

    img_out = os.path.join(output_dir, split, "IMAGES", f)
    mask_out = os.path.join(output_dir, split, "MASKS", base + ".png")

    if os.path.exists(img_src) and os.path.exists(mask_src):
        safe_link_or_copy(img_src, img_out)
        safe_link_or_copy(mask_src, mask_out)


def count_classes_in_dir(mask_dir):
    """Count images containing each class"""
    counts = Counter()
    for f in os.listdir(mask_dir):
        if not f.endswith(".png"):
            continue
        mask = np.array(Image.open(os.path.join(mask_dir, f)).convert("RGB"))
        mask_cls = lut[mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]]
        unique_cls = np.unique(mask_cls)
        for cls in unique_cls:
            if cls >= 0:
                counts[cls] += 1
    return counts


# ===================== MAIN =====================
if __name__ == "__main__":
    print("Extracting labels from masks (parallel)...")
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

    filenames, labels = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_mask_file, f): f for f in mask_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing masks"):
            result = future.result()
            if result is not None:
                img_name, label_vec = result
                filenames.append(img_name)
                labels.append(label_vec)

    if len(filenames) == 0:
        raise ValueError("No matching images and masks found. Check your directories!")

    labels = np.array(labels)

    # -------- STRATIFIED SPLIT --------
    print("Performing stratified split...")
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(msss1.split(filenames, labels))

    val_ratio = val_size / (1 - test_size)
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(msss2.split([filenames[i] for i in train_val_idx],
                                          labels[train_val_idx]))

    train_files = [filenames[i] for i in train_val_idx[train_idx]]
    val_files = [filenames[i] for i in train_val_idx[val_idx]]
    test_files = [filenames[i] for i in test_idx]
    splits = {"TRAIN": train_files, "VAL": val_files, "TEST": test_files}

    # -------- COPY OR LINK FILES --------
    print("Creating dataset folders and linking/copying files...")
    for split, files in splits.items():
        img_out_dir = os.path.join(output_dir, split, "IMAGES")
        mask_out_dir = os.path.join(output_dir, split, "MASKS")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(lambda f: copy_pair(f, split), files),
                total=len(files),
                desc=f"Copy {split}"
            ))
        elapsed = time.time() - start_time
        print(f"{split} done in {elapsed:.2f} sec ({len(files)} files)")

    # -------- DISTRIBUTION CHECK --------
    print("\nClass Distribution (per split):")
    split_counts = defaultdict(Counter)
    for split in splits.keys():
        mask_dir = os.path.join(output_dir, split, "MASKS")
        split_counts[split] = count_classes_in_dir(mask_dir)

        print(f"\n{split}:")
        total_images = len(os.listdir(mask_dir))
        print(f"  Total images: {total_images}")
        for cls_id in range(NUM_CLASSES):
            count = split_counts[split].get(cls_id, 0)
            pct = (count / total_images) * 100 if total_images > 0 else 0
            print(f"  Class {cls_id}: {count} images ({pct:.2f}%)")

    print("\n✅ Done.")
