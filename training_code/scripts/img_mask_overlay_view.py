import os
import pdb
import cv2
import shutil
import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import re
import random
from collections import OrderedDict
import queue
from concurrent.futures import ThreadPoolExecutor
import gc

# --- CONFIG ---
root_dir = r"D:/"
image_dir = r"Z:\Devendra\ASPHALT\Asphalt_GoldenSet_Test\IMAGES"
mask_dir = r"Z:\Devendra\ASPHALT\TRAININGNEW\MASKS"
pcams_dir = os.path.join(root_dir, 'pcams')
pcams_dir =  "Z:\Devendra\CONCRETE\ACCEPTED_MASKS"
start_number = 0

accepted_img_dir = os.path.join(root_dir, "ACCEPTED_IMAGES")
accepted_mask_dir = os.path.join(root_dir, "ACCEPTED_MASKS")
rework_img_dir = os.path.join(root_dir, "REWORK_IMAGES")
rework_mask_dir = os.path.join(root_dir, "REWORK_MASKS")
csv_log = os.path.join(root_dir, "rework_log.csv")
os.makedirs(accepted_img_dir, exist_ok=True)

os.makedirs(accepted_mask_dir, exist_ok=True)
os.makedirs(rework_img_dir, exist_ok=True)
os.makedirs(rework_mask_dir, exist_ok=True)

# --- Load file names (same as before) ---
images = sorted([f for f in os.listdir(image_dir)])
masks = sorted([f for f in os.listdir(mask_dir)])

image_stems = {os.path.splitext(f)[0]: f for f in images}
mask_stems = {os.path.splitext(f)[0]: f for f in masks}

common_keys = sorted(set(image_stems.keys()) & set(mask_stems.keys()))
images = [image_stems[k] for k in common_keys]
masks = [mask_stems[k] for k in common_keys]

assert len(images) == len(masks), "Images and masks count mismatch!"
print(len(images), "image-mask pairs found.")
combined = list(zip(images, masks))
random.shuffle(combined)
images, masks = zip(*combined)
images, masks = list(images), list(masks)


def find_start_index(files, number):
    for i, f in enumerate(files):
        if re.search(rf"{number}", f):
            return i
    return 0


start_index = find_start_index(images, start_number)


class LRUCache:
    """Simple LRU cache for images with memory management"""

    def __init__(self, maxsize=20):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                # Remove least recently used
                oldest = self.cache.popitem(last=False)
                del oldest  # Help garbage collection
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        gc.collect()


class ImageMaskViewerOptimized:
    def __init__(self, images, masks, image_dir, mask_dir, pcams_dir, start_index=0, cache_size=20, prefetch_count=3):
        self.images = images
        self.masks = masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.pcams_dir = pcams_dir
        self.index = start_index
        self.comment = ""

        # Caching and prefetching
        self.cache = LRUCache(cache_size)
        self.prefetch_count = prefetch_count
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.prefetch_queue = queue.Queue()

        # Pre-build pcams lookup for faster access
        self.pcams_lookup = self._build_pcams_lookup()

        # UI Setup (same as before)
        self.fig, self.axs = plt.subplots(1, 4, figsize=(30, 10))
        plt.subplots_adjust(bottom=0.25)
        pos = self.axs[3].get_position()
        self.axs[3].set_position([pos.x0 - 0.05, pos.y0 - 0.05, pos.width * 1.6, pos.height * 1.6])

        # Buttons
        axprev = plt.axes([0.15, 0.15, 0.1, 0.07])
        axnext = plt.axes([0.30, 0.15, 0.1, 0.07])
        axaccept = plt.axes([0.55, 0.15, 0.12, 0.07])
        axrework = plt.axes([0.70, 0.15, 0.12, 0.07])
        # Textbox for comment (existing)
        axbox = plt.axes([0.15, 0.05, 0.5, 0.05])
        # 🔹 New textbox for number-based jump
        axnum = plt.axes([0.70, 0.05, 0.2, 0.05])

        self.bnext = Button(axnext, 'Next')
        self.bprev = Button(axprev, 'Previous')
        self.baccept = Button(axaccept, 'Accept')
        self.brework = Button(axrework, 'Rework')
        self.textbox = TextBox(axbox, "Comment: ")
        self.num_box = TextBox(axnum, "Go to #: ")

        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)
        self.baccept.on_clicked(self.accept)
        self.brework.on_clicked(self.rework)
        self.textbox.on_submit(self.update_text)
        self.num_box.on_submit(self.go_to_number)

        # Load initial images and start prefetching
        self._prefetch_around_index(self.index)
        self.update_display()
        plt.show()

    def _build_pcams_lookup(self):
        """Pre-build lookup dictionary for pcams files"""
        lookup = {}
        if os.path.exists(self.pcams_dir):
            for f in os.listdir(self.pcams_dir):
                if "LL" in f:
                    numbers = re.findall(r"(\d+)", f)
                    if numbers:
                        number = str(int(numbers[-1]))  # Remove leading zeros
                        lookup[number] = f
        return lookup

    def get_number_from_name(self, filename):
        matches = re.findall(r"(\d+)", filename)
        if matches:
            return str(int(matches[-1]))
        return None

    def go_to_number(self, text):
        """Jump to image that contains this number in its filename"""
        number = text.strip()
        if not number.isdigit():
            print(f"Invalid number: {number}")
            return

        for i, fname in enumerate(self.images):
            if re.search(rf"{number}", fname):
                self.index = i
                self.update_display()
                print(f"Jumped to image {fname}")
                return

        print(f"No image found with number {number}")

    def load_images_sync(self, idx):
        """Synchronously load images for given index"""
        if idx < 0 or idx >= len(self.images):
            return None

        # Check cache first
        cached = self.cache.get(idx)
        if cached is not None:
            return cached

        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load main image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.flip(mask, 0)
        # mask = cv2.resize(mask, (4183, 10217), interpolation=cv2.INTER_NEAREST)
        if len(mask.shape) == 2:
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        else:
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)

        # Load pcams using lookup
        pcams_img = None
        number = self.get_number_from_name(self.images[idx])
        if number:
            adjusted_number = str(int(number))  #+1
            if adjusted_number in self.pcams_lookup:
                pcams_path = os.path.join(self.pcams_dir, self.pcams_lookup[adjusted_number])
                pcams_img = cv2.imread(pcams_path)
                if pcams_img is not None:
                    pcams_img = cv2.cvtColor(pcams_img, cv2.COLOR_BGR2RGB)

        result = (img, mask_colored, overlay, pcams_img)

        # Cache the result
        self.cache.put(idx, result)

        return result

    def _prefetch_around_index(self, center_idx):
        """Prefetch images around the current index"""
        indices_to_prefetch = []

        # Prefetch next few images
        for i in range(1, self.prefetch_count + 1):
            next_idx = center_idx + i
            if next_idx < len(self.images) and self.cache.get(next_idx) is None:
                indices_to_prefetch.append(next_idx)

        # Prefetch previous few images
        for i in range(1, self.prefetch_count + 1):
            prev_idx = center_idx - i
            if prev_idx >= 0 and self.cache.get(prev_idx) is None:
                indices_to_prefetch.append(prev_idx)

        # Submit prefetch tasks
        for idx in indices_to_prefetch:
            self.executor.submit(self._prefetch_single, idx)

    def _prefetch_single(self, idx):
        """Prefetch a single image in background"""
        try:
            self.load_images_sync(idx)
        except Exception as e:
            print(f"Prefetch failed for index {idx}: {e}")

    def update_display(self):
        if not self.images:
            self.fig.suptitle("No more images left!", fontsize=14, color="red")
            for ax in self.axs:
                ax.clear()
                ax.axis("off")
            self.fig.canvas.draw()
            return

        # Load current images (from cache if available)
        result = self.load_images_sync(self.index)
        if result is None:
            return

        img, mask, overlay, pcams_img = result

        # Clear and display
        for ax in self.axs:
            ax.clear()
            ax.axis("off")

        self.axs[0].imshow(img)
        self.axs[0].set_title("Image")
        self.axs[1].imshow(mask)
        self.axs[1].set_title("Mask")
        self.axs[2].imshow(overlay)
        self.axs[2].set_title("Overlay")

        if pcams_img is not None:
            self.axs[3].imshow(pcams_img)
            self.axs[3].set_title("Pcams (LL)")
        else:
            self.axs[3].set_title("Pcams Missing")

        self.fig.suptitle(f"Image {self.index + 1}/{len(self.images)}: {self.images[self.index]}")
        self.fig.canvas.draw()

        # Trigger prefetching for nearby images
        self._prefetch_around_index(self.index)

    def update_text(self, text):
        self.comment = text

    def move_files(self, target_img_dir, target_mask_dir, log_comment=False):
        img_name = self.images[self.index]
        mask_name = self.masks[self.index]

        src_img = os.path.join(self.image_dir, img_name)
        src_mask = os.path.join(self.mask_dir, mask_name)
        dst_img = os.path.join(target_img_dir, img_name)
        dst_mask = os.path.join(target_mask_dir, mask_name)

        shutil.move(src_img, dst_img)
        shutil.move(src_mask, dst_mask)

        comment = self.comment + ("- Devendra Rejected" if log_comment else " - Devendra Accepted")
        # if log_comment:
        with open(csv_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([img_name, comment])
        print(f"Logged: {img_name} | {comment}")

        # Remove from cache and lists
        if self.index in self.cache.cache:
            del self.cache.cache[self.index]

        del self.images[self.index]
        del self.masks[self.index]

        # Adjust cache indices (shift down indices > current)
        new_cache = OrderedDict()
        for idx, data in self.cache.cache.items():
            if idx < self.index:
                new_cache[idx] = data
            elif idx > self.index:
                new_cache[idx - 1] = data
        self.cache.cache = new_cache

        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

        self.comment = ""
        self.textbox.set_val("")
        self.update_display()

    def accept(self, event):
        if self.images:
            self.move_files(accepted_img_dir, accepted_mask_dir, log_comment=False)

    def rework(self, event):
        if self.images:
            self.move_files(rework_img_dir, rework_mask_dir, log_comment=True)

    def next(self, event):
        if self.images:
            self.index = (self.index + 1) % len(self.images)
            self.update_display()

    def prev(self, event):
        if self.images:
            self.index = (self.index - 1) % len(self.images)
            self.update_display()

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'cache'):
            self.cache.clear()


# --- Run Optimized Viewer ---
viewer = ImageMaskViewerOptimized(
    images, masks, image_dir, mask_dir, pcams_dir,
    start_index=start_index,
    cache_size=15,  # Cache 15 images (adjust based on RAM)
    prefetch_count=3  # Prefetch 3 images ahead/behind
)
