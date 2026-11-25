import os

# ==========================
# CONFIGURATION
# ==========================
IMAGE_FOLDER = r"C:\Users\HP\Downloads\IMAGES"
MASK_FOLDER = r"C:\Users\HP\Downloads\MASKS"
# ==========================

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MASK_EXTENSION = ".png"

# Read image files
image_files = {}
for f in os.listdir(IMAGE_FOLDER):
    if f.lower().endswith(IMAGE_EXTENSIONS):
        image_files[os.path.splitext(f)[0]] = f

# Read mask files
mask_files = {}
for f in os.listdir(MASK_FOLDER):
    if f.lower().endswith(MASK_EXTENSION):
        mask_files[os.path.splitext(f)[0]] = f

image_names = set(image_files.keys())
mask_names = set(mask_files.keys())

# Find unmatched files
images_to_delete = image_names - mask_names
masks_to_delete = mask_names - image_names

# Delete unmatched images
for name in images_to_delete:
    os.remove(os.path.join(IMAGE_FOLDER, image_files[name]))
    print(f"Deleted Image: {image_files[name]}")

# Delete unmatched masks
for name in masks_to_delete:
    os.remove(os.path.join(MASK_FOLDER, mask_files[name]))
    print(f"Deleted Mask: {mask_files[name]}")

# Final counts
final_images = len([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(IMAGE_EXTENSIONS)
])

final_masks = len([
    f for f in os.listdir(MASK_FOLDER)
    if f.lower().endswith(MASK_EXTENSION)
])

print("\n========== SUMMARY ==========")
print(f"Images deleted : {len(images_to_delete)}")
print(f"Masks deleted  : {len(masks_to_delete)}")
print(f"Final Images   : {final_images}")
print(f"Final Masks    : {final_masks}")

if final_images == final_masks:
    print("✅ Images and masks are perfectly matched.")
else:
    print("❌ Counts are still different. Please check for duplicate image names with different extensions.")