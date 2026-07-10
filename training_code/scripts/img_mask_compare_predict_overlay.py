import os
import pdb
import cv2
import matplotlib
matplotlib.use("TkAgg")   # or "TkAgg"
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import re
import random
from collections import OrderedDict
import gc
import csv
from concurrent.futures import ThreadPoolExecutor

# --- CONFIG ---
start_number = 0  # <<< starting image number
root_dir = r"D:/"

pcams_dir = r"Z:/Devendra/CONCRETE/COMBINED_SPLITTED/TEST/IMAGES"
image_dirs = [
    rf"Z:\Devendra\ASPHALT\SPLIT\TRAIN\FOLDER_3\IMAGES",
]

orig_mask_dirs = [
    rf"Z:\Devendra\ASPHALT\SPLIT\TRAIN\FOLDER_3\MASKS"
]

old_pred_mask_dirs = [
    rf"Z:\Devendra\ASPHALT\SPLIT\TRAIN\FOLDER_3\MASKS"
]

new_pred_mask_dirs = [
    rf"Z:\Devendra\ASPHALT\SPLIT\TRAIN\FOLDER_3\MASKS"
]

# --- Output dirs ---)
accepted_img_dir = os.path.join(root_dir, "ACCEPTED_IMAGES")
accepted_mask_dir = os.path.join(root_dir, "ACCEPTED_MASKS")
rework_img_dir = os.path.join(root_dir, "REWORK_IMAGES")
rework_mask_dir = os.path.join(root_dir, "REWORK_MASKS")
csv_log = os.path.join(root_dir, "review_log.csv")

# --- Ensure output directories exist ---
for d in [accepted_img_dir, accepted_mask_dir, rework_img_dir, rework_mask_dir]:
    os.makedirs(d, exist_ok=True)


# --- Load file names ---
def load_files(dirs, exts):
    files = []
    for d in dirs:
        print("Checking folder:", d)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(exts):
                    files.append(os.path.join(d, f))
        else:
            print("Path not found:", d)

    print("Loaded", len(files), "files")
    return sorted(files)

images = load_files(image_dirs, ('.png', '.jpg', '.jpeg'))
orig_masks = load_files(orig_mask_dirs, ('.png', '.jpg', '.jpeg'))
new_pred_masks = load_files(new_pred_mask_dirs, ('.png', '.jpg', '.jpeg'))
old_pred_masks = load_files(old_pred_mask_dirs, ('.png', '.jpg', '.jpeg'))

# --- Strip extensions for matching ---
image_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in images}
orig_mask_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in orig_masks}
new_pred_mask_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in new_pred_masks}
old_pred_mask_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in old_pred_masks}

# --- Keep only intersection across all three ---
common_keys = sorted(set(image_stems) & set(orig_mask_stems) & set(new_pred_mask_stems) & set(old_pred_mask_stems))
images = [image_stems[k] for k in common_keys]
orig_masks = [orig_mask_stems[k] for k in common_keys]
new_pred_masks = [new_pred_mask_stems[k] for k in common_keys]
old_pred_masks = [old_pred_mask_stems[k] for k in common_keys]

print(f"Kept {len(images)} triplets")
assert len(images) == len(orig_masks) == len(new_pred_masks)

# --- Shuffle together ---
combined = list(zip(images, orig_masks, new_pred_masks, old_pred_masks))
random.shuffle(combined)
images, orig_masks, new_pred_masks, old_pred_masks  = zip(*combined)
images, orig_masks, new_pred_masks, old_pred_masks = list(images), list(orig_masks),list(new_pred_masks), list(old_pred_masks)


def find_start_index(files, number):
    for i, f in enumerate(files):
        if re.search(rf"{number}", f):
            return i
    return 0


# --- LRU Cache ---
class LRUCache:
    def __init__(self, maxsize=20):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        gc.collect()


# --- Viewer ---
class ImageMaskViewerOptimized:
    def __init__(self, images, orig_masks, new_pred_masks, old_pred_masks, pcams_dir,
                 start_index=0, cache_size=20, prefetch_count=3):
        self.images = images
        self.orig_masks = orig_masks
        self.new_pred_masks = new_pred_masks
        self.old_pred_masks = old_pred_masks
        self.pcams_dir = pcams_dir
        self.index = start_index
        self.comment = ""

        self.cache = LRUCache(cache_size)
        self.prefetch_count = prefetch_count
        self.executor = ThreadPoolExecutor(max_workers=2)

        # --- 5 panels now ---
        self.fig, self.axs = plt.subplots(1, 6, figsize=(35, 10))
        plt.subplots_adjust(bottom=0.25)
        pos = self.axs[5].get_position()
        self.axs[5].set_position([pos.x0, pos.y0 - 0.05, pos.width * 1.6, pos.height * 1.6])

        # Buttons
        axprev = plt.axes([0.15, 0.15, 0.1, 0.07])
        axnext = plt.axes([0.30, 0.15, 0.1, 0.07])
        axaccept = plt.axes([0.55, 0.15, 0.12, 0.07])
        axrework = plt.axes([0.70, 0.15, 0.12, 0.07])
        axbox = plt.axes([0.15, 0.05, 0.5, 0.05])
        axnum = plt.axes([0.70, 0.05, 0.2, 0.05])

        self.bprev = Button(axprev, 'Previous')
        self.bnext = Button(axnext, 'Next')
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

        self.update_display()
        plt.show()

    def go_to_number(self, text):
        number = text.strip()
        if not number.isdigit():
            print(f"Invalid number: {number}")
            return
        for i, fname in enumerate(self.images):
            if re.search(rf"{number}", fname):
                self.index = i
                self.update_display()
                return

    def get_number_from_name(self, filename):
        matches = re.findall(r"(\d+)", filename)
        if matches:
            return str(int(matches[-1]))
        return None

    def load_images_sync(self, idx):
        if idx < 0 or idx >= len(self.images):
            return None

        cached = self.cache.get(idx)
        if cached is not None:
            return cached

        img_path = self.images[idx]
        orig_mask_path = self.orig_masks[idx]
        new_pred_mask_path = self.new_pred_masks[idx]
        old_pred_mask_path = self.old_pred_masks[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)
        orig_mask = cv2.imread(orig_mask_path, cv2.IMREAD_UNCHANGED)
        orig_mask = cv2.flip(orig_mask, 0)
        if len(orig_mask.shape) == 2:
            orig_mask_colored = cv2.applyColorMap(orig_mask, cv2.COLORMAP_JET)
        else:
            orig_mask_colored = cv2.cvtColor(orig_mask, cv2.COLOR_BGR2RGB)


        old_mask = cv2.imread(old_pred_mask_path, cv2.IMREAD_UNCHANGED)
        old_mask = cv2.flip(old_mask, 0)
        if len(old_mask.shape) == 2:
            old_mask = cv2.applyColorMap(old_mask, cv2.COLORMAP_JET)
        else:
            old_mask = cv2.cvtColor(old_mask, cv2.COLOR_BGR2RGB)

        pred_mask = cv2.imread(new_pred_mask_path, cv2.IMREAD_UNCHANGED)
        pred_mask = cv2.flip(pred_mask, 0)
        if len(pred_mask.shape) == 2:
            pred_mask_colored = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
        else:
            pred_mask_colored = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB)
        pred_mask_colored = cv2.resize(pred_mask_colored, (419, 1024), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(img, 0.5, old_mask, 0.5, 0)

        # --- Load pcams (LL) ---
        pcams_img = None
        number = self.get_number_from_name(os.path.basename(img_path))
        number = int(number) + 1 if number else None
        if number and os.path.exists(self.pcams_dir):
            for f in os.listdir(self.pcams_dir):
                if "LL" in f and re.search(rf"{number}", f):
                    pcams_img = cv2.imread(os.path.join(self.pcams_dir, f))
                    pcams_img = cv2.cvtColor(pcams_img, cv2.COLOR_BGR2RGB)
                    break

        result = (img, orig_mask_colored, old_mask, pred_mask_colored, overlay, pcams_img)
        self.cache.put(idx, result)
        return result

    def update_display(self):
        if not self.images:
            self.fig.suptitle("No more images!", fontsize=14, color="red")
            for ax in self.axs:
                ax.clear()
                ax.axis("off")
            self.fig.canvas.draw()
            return

        result = self.load_images_sync(self.index)
        if result is None:
            return

        img, orig_mask,old_mask, pred_mask, overlay, pcams_img = result

        for ax in self.axs:
            ax.clear()
            ax.axis("off")

        self.axs[0].imshow(img);
        self.axs[0].set_title("Image")
        self.axs[1].imshow(orig_mask);
        self.axs[1].set_title("Original Mask")
        self.axs[2].imshow(old_mask);
        self.axs[2].set_title("OLD")
        self.axs[3].imshow(pred_mask);
        self.axs[3].set_title("NEW")
        self.axs[4].imshow(overlay)
        self.axs[4].set_title("Overlay")
        if pcams_img is not None:
            self.axs[5].imshow(pcams_img);
            self.axs[5].set_title("Pcams (LL)")
        else:
            self.axs[5].set_title("Pcams Missing")

        self.fig.suptitle(f"[{self.index + 1}/{len(self.images)}] {os.path.basename(self.images[self.index])}")
        self.fig.canvas.draw()

    def update_text(self, text):
        self.comment = text

    def move_files(self, dest_img_dir, dest_mask_dir, log_comment=False):
        img_path = self.images[self.index]
        pred_mask_path = self.new_pred_masks[self.index]

        dst_img = os.path.join(dest_img_dir, os.path.basename(img_path))
        dst_mask = os.path.join(dest_mask_dir, os.path.basename(pred_mask_path))

        shutil.move(img_path, dst_img)
        shutil.move(pred_mask_path, dst_mask)

        comment = self.comment + (" Rejected" if log_comment else " Accepted")
        with open(csv_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(img_path), os.path.basename(pred_mask_path), comment])

        del self.images[self.index]
        del self.orig_masks[self.index]
        del self.old_pred_masks[self.index]
        del self.new_pred_masks[self.index]
        # --- Clear cache and refresh ---
        self.cache.clear()
        # --- If images left, update display ---
        if self.images:
            self.update_display()
        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

        self.comment = ""
        self.textbox.set_val("")
        # self.update_display()

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


# --- Run Viewer ---
start_index = find_start_index(images, start_number)
viewer = ImageMaskViewerOptimized(images, orig_masks, new_pred_masks,old_pred_masks,
                                  pcams_dir,
                                  start_index=start_index, cache_size=15, prefetch_count=3)
