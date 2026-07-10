from pathlib import Path
from PIL import Image
import numpy as np

def combine_all_masks(folder1, folder2, output_folder):
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {'.png', '.jpg', '.jpeg'}
    filenames1 = {f.name for f in folder1_path.iterdir() if f.suffix.lower() in valid_exts}
    filenames2 = {f.name for f in folder2_path.iterdir() if f.suffix.lower() in valid_exts}
    all_filenames = filenames1.union(filenames2)

    for name in all_filenames:
        mask1_path = folder1_path / name
        mask2_path = folder2_path / name

        arr1 = arr2 = None
        if mask1_path.exists():
            arr1 = np.array(Image.open(mask1_path).convert('RGB'))
        if mask2_path.exists():
            arr2 = np.array(Image.open(mask2_path).convert('RGB'))

        if arr1 is None and arr2 is None:
            continue

        if arr1 is None:
            combined = arr2
        elif arr2 is None:
            combined = arr1
        else:
            # Start with mask1
            combined = arr1.copy()
            mask2_non_black = np.any(arr2 != [0, 0, 0], axis=-1)
            mask1_black = np.all(combined == [0, 0, 0], axis=-1)
            # Fill in mask2 pixels only where mask1 is black
            combined[mask2_non_black & mask1_black] = arr2[mask2_non_black & mask1_black]

        # Always save as PNG
        save_name = Path(name).stem + ".png"
        Image.fromarray(combined.astype(np.uint8)).save(output_path / save_name)

    return f'Combined masks saved to {output_folder}'


combine_all_masks(
                r'Z:\Devendra\ASPHALT\Asphalt_GoldenSet_Test\EXCEPT_ALLIGATOR',
                r'Z:\Devendra\ASPHALT\Asphalt_GoldenSet_Test\ONLY_ALLIGATOR',
                r'Z:\Devendra\ASPHALT\Asphalt_GoldenSet_Test\ALL_MASKS'
                  )