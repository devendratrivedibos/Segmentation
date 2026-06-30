import cv2
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

image_folder = r"Z:\Devendra\ASPHALT\NEWTRAINING\IMAGES"

# List all image files
image_files = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

def process_image(img_path):
    try:
        # Read image in BGR (OpenCV default)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to read image")

        # Convert BGR → RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Compute per-channel mean and variance (efficient)
        mean = img.mean(axis=(0, 1))
        var = img.var(axis=(0, 1))
        pixel_count = img.shape[0] * img.shape[1]

        return mean * pixel_count, var * pixel_count, pixel_count

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0


if __name__ == '__main__':
    total_mean = np.zeros(3, dtype=np.float64)
    total_var = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    num_workers = max(1, cpu_count() // 2)

    with Pool(processes=num_workers) as pool:
        for mean_sum, var_sum, count in tqdm(pool.imap_unordered(process_image, image_files, chunksize=8), total=len(image_files)):
            total_mean += mean_sum
            total_var += var_sum
            total_pixels += count

    if total_pixels > 0:
        mean = total_mean / total_pixels
        std = np.sqrt(total_var / total_pixels)
        print("\n✅ Mean (RGB):", mean)
        print("✅ Std (RGB):", std)
    else:
        print("❌ No valid images processed.")
