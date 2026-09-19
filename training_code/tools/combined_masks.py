import cv2
import numpy as np
from pathlib import Path




MASK1_FOLDER = Path(
    r"C:\Sidhesh\Distress\Asphalt\Extra Training Data\MANGAWAN-UPBORDER_2025-12-22_11-59-50\alligator_predictions"
    # r"\NAGAUR-JODHPUR_2026-05-02_12-13-17"
    # r"\SECTION-1-pred-alligator"
)

MASK2_FOLDER = Path(
    r"C:\Sidhesh\Distress\Asphalt\Extra Training Data\MANGAWAN-UPBORDER_2025-12-22_11-59-50\pred"
    # r"\NAGAUR-JODHPUR_2026-05-02_12-13-17"
    # r"\SECTION-1-pred"
)

OUTPUT_FOLDER = Path(
    r"C:\Sidhesh\Distress\Asphalt\Extra Training Data\MANGAWAN-UPBORDER_2025-12-22_11-59-50\combined_masks"
    # r"\NAGAUR-JODHPUR_2026-05-02_12-13-17"
    # r"\SECTION-1-combined"
)


# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# =========================================================
# SUPPORTED IMAGE EXTENSIONS
# =========================================================

extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


# =========================================================
# GET MASK FILES
# =========================================================

mask1_files = {
    p.stem: p
    for p in MASK1_FOLDER.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
}

mask2_files = {
    p.stem: p
    for p in MASK2_FOLDER.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
}


print("=" * 60)
print("MASK INFORMATION")
print("=" * 60)

print(f"Mask 1 (Alligator) : {len(mask1_files)}")
print(f"Mask 2 (Other)     : {len(mask2_files)}")

print("=" * 60)


# =========================================================
# FIND COMMON FILES
# =========================================================

common_files = set(mask1_files.keys()) & set(mask2_files.keys())

print(f"Common masks       : {len(common_files)}")
print()


# =========================================================
# PROCESS
# =========================================================

combined_count = 0
mask1_only_count = 0
mask2_only_count = 0
error_count = 0


for filename in sorted(common_files):

    mask1_path = mask1_files[filename]
    mask2_path = mask2_files[filename]

    # -----------------------------------------------------
    # Read masks
    # -----------------------------------------------------

    mask1 = cv2.imread(
        str(mask1_path),
        cv2.IMREAD_COLOR
    )

    mask2 = cv2.imread(
        str(mask2_path),
        cv2.IMREAD_COLOR
    )

    if mask1 is None:
        print(f"ERROR reading Mask 1: {mask1_path}")
        error_count += 1
        continue

    if mask2 is None:
        print(f"ERROR reading Mask 2: {mask2_path}")
        error_count += 1
        continue

    # -----------------------------------------------------
    # Check dimensions
    # -----------------------------------------------------

    if mask1.shape != mask2.shape:

        print(
            f"SIZE MISMATCH: {filename}\n"
            f"  Mask 1: {mask1.shape}\n"
            f"  Mask 2: {mask2.shape}"
        )

        error_count += 1
        continue

    # -----------------------------------------------------
    # MASK 1 = ALLIGATOR
    #
    # Anything that is NOT black is considered
    # Alligator.
    # -----------------------------------------------------

    mask1_background = np.all(
        mask1 == [0, 0, 0],
        axis=2
    )

    # -----------------------------------------------------
    # Start with Mask 1
    #
    # This protects Alligator.
    # -----------------------------------------------------

    combined = mask1.copy()

    # -----------------------------------------------------
    # Add Mask 2 only where Mask 1 is background
    # -----------------------------------------------------

    combined[mask1_background] = mask2[mask1_background]

    # -----------------------------------------------------
    # Save
    #
    # Keep same filename as Mask 1
    # -----------------------------------------------------

    output_path = OUTPUT_FOLDER / mask1_path.name

    cv2.imwrite(
        str(output_path),
        combined
    )

    combined_count += 1

    print(f"COMBINED: {filename}")


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Mask 1 files       : {len(mask1_files)}")
print(f"Mask 2 files       : {len(mask2_files)}")
print(f"Common files       : {len(common_files)}")
print(f"Combined           : {combined_count}")
print(f"Errors             : {error_count}")
print(f"Output folder      : {OUTPUT_FOLDER}")

print("=" * 60)