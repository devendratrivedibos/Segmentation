from tqdm import tqdm
import cv2
import numpy as np
import torch
from PIL import Image
import sys
import os
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import itertools

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))

from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large
from models.segformer.segformer import SegFormer
from models.unet.unet import UNet
from models.unet.mobilenet_unet import MobileV3Unet
from models.unet.vgg_unet import VGG16UNet
from models.fcn.fcn import fcn_resnet101
from models.unet.UnetPP import UNetPP
from random import shuffle

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
    (255, 100, 203): (9, "Sealed Joint - T"),
    (199, 21, 133): (10, "Sealed Joint - L"),
    (128, 0, 128): (11, "Punchout"),
    (112, 102, 255): (12, "Popout"),
    (255, 255, 255): (13, "Unclassified"),
    (255, 215, 0): (14, "Cracking"),
}


def main(imgs_root=None, prediction_save_path=None, weights_path=None, batch_size=2):
    num_classes = 14 + 1  #14 #5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = (0.478, 0.478, 0.478)  ##478 548
    std = (0.145, 0.145, 0.145)  ###145 146

    data_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # Collect images
    images_list = [img for img in os.listdir(imgs_root) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    shuffle(images_list)
    os.makedirs(prediction_save_path, exist_ok=True)

    # Model
    model = SegFormer(num_classes=15, phi="b0")
    model = UNetPP(in_channels=3, num_classes=num_classes)
    pretrain_weights = torch.load(weights_path, map_location=device)
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)
    model.eval()

    # Process in batches
    with torch.no_grad():
        try:
            for i in tqdm(range(0, len(images_list), batch_size), desc=f"Processing {os.path.basename(imgs_root)}"):
                batch_files = images_list[i:i + batch_size]
                batch_imgs, orig_names = [], []

                for image in batch_files:
                    original_img = cv2.imread(os.path.join(imgs_root, image))
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                    transformed = data_transform(image=original_img)
                    img = transformed["image"]
                    batch_imgs.append(img)
                    orig_names.append(image)

                # stack and forward pass
                batch_tensor = torch.stack(batch_imgs).to(device)
                outputs = model(batch_tensor)
                preds = outputs['out'].argmax(1).cpu().numpy().astype(np.uint8)

                # postprocess + save
                for pred, fname in zip(preds, orig_names):
                    # pred = cv2.resize(pred, (419, 1024), interpolation=cv2.INTER_NEAREST)
                    pred = join_directional_multiclass(pred, radius=25, line_width=2)  # ⬅️ Added here

                    pred = fix_fragmented_cracks_and_joints(pred)  # ⬅️ THIS SOLVES YOUR ISSUE
                    pred = remove_small_components_multiclass(pred)
                    # pred = extend_joint_seals_to_image_end(pred)  # ✅ FINAL STEP

                    pred_color = colorize_prediction(pred)
                    save_path = os.path.join(prediction_save_path, fname.split('.')[0] + '.png')
                    cv2.imwrite(save_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"❌ Error during processing batch starting at index {image}: {e}")
            raise e
    print("✅ Processing done for", imgs_root)


def colorize_prediction(prediction):
    """
    Convert class-wise prediction (H, W) to RGB image.
    """
    color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for rgb, (class_id, _) in COLOR_MAP.items():
        color_mask[prediction == class_id] = rgb
    return color_mask


# =====================================
# UTILITIES
# =====================================
def find_endpoints(contour):
    """Return two farthest points in contour (endpoints)."""
    pts = contour.reshape(-1, 2)
    max_dist = 0
    endpoints = (pts[0], pts[0])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_dist:
                max_dist = d
                endpoints = (pts[i], pts[j])
    return endpoints


def join_directional(mask, crack_type, radius=25, line_width=2):
    """
    Unified directional join function for cracks and sealed joints
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    joined = mask.copy()

    endpoints = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        p1, p2 = find_endpoints(cnt)
        endpoints.append((p1, p2))

    for (a1, a2), (b1, b2) in itertools.combinations(endpoints, 2):
        for p, q in [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]:

            dx = abs(int(p[0]) - int(q[0]))
            dy = abs(int(p[1]) - int(q[1]))

            # -------------------------
            # CRACKS
            # -------------------------
            if crack_type == "Longitudinal Crack":
                if dx <= radius and dy <= 100:
                    cv2.line(joined, tuple(p), tuple(q), 255, line_width)

            elif crack_type == "Transverse Crack":
                if dy <= radius and dx <= 50:
                    cv2.line(joined, tuple(p), tuple(q), 255, line_width)

            elif crack_type in ["Alligator", "Multiple Crack"]:
                if np.hypot(dx, dy) <= radius:
                    cv2.line(joined, tuple(p), tuple(q), 255, line_width)

            # -------------------------
            # SEALED JOINTS (5px WIDTH)
            # -------------------------
            elif crack_type == "Sealed Joint - L":
                # vertical preference, fixed 5px width
                if dx <= 10:
                    cv2.line(joined, tuple(p), tuple(q), 255, thickness=5)

            elif crack_type == "Sealed Joint - T":
                # horizontal preference, fixed 5px width
                if dy <= 10:
                    cv2.line(joined, tuple(p), tuple(q), 255, thickness=5)

    return joined


def join_directional_multiclass(pred_idx_map, radius=5, line_width=2):
    joined_idx = pred_idx_map.copy()
    inv_color_map = {v[0]: v[1] for v in COLOR_MAP.values()}

    for idx, name in inv_color_map.items():
        if name not in [
            "Longitudinal Crack",
            "Transverse Crack",
            "Alligator",
            "Multiple Crack",
            "Sealed Joint - L",
            "Sealed Joint - T"
        ]:
            continue

        mask = (pred_idx_map == idx).astype(np.uint8) * 255

        joined_mask = join_directional(
            mask,
            crack_type=name,
            radius=radius,
            line_width=line_width
        )

        joined_idx[joined_mask > 0] = idx

    return joined_idx


def overlay_mask_on_image(image, color_mask, alpha=0.5):
    """
    Overlay a multi-class RGB mask on a color image.
    """
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay


def remove_small_components_multiclass(mask):
    """
    Removes small connected components using class-specific area thresholds.
    """
    cleaned = np.zeros_like(mask, dtype=mask.dtype)

    # class-wise area thresholds (pixels)
    AREA_THRESHOLDS = {
        9: 100,   # Sealed Joint - T
        10: 100,  # Sealed Joint - L
        3: 100,    # Longitudinal Crack
        2: 100,    # Transverse Crack
        6: 100,    # Multiple Crack
        5: 500,   # Patches
        8: 100,   # Corner Break
    }
    # classes that should never be removed
    ALWAYS_KEEP = {0, 4, 7, 11, 12}  # background, pothole, spalling, corner, punchout, popout

    for cls in np.unique(mask):

        if cls in ALWAYS_KEEP:
            cleaned[mask == cls] = cls
            continue

        class_mask = (mask == cls).astype(np.uint8)
        if class_mask.sum() == 0:
            continue

        min_area = AREA_THRESHOLDS.get(cls, 100)  # default threshold

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            class_mask, connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = cls

    return cleaned


def get_orientation(coords):
    ys = coords[:, 0]
    xs = coords[:, 1]

    height = ys.max() - ys.min()
    width  = xs.max() - xs.min()

    if height > 3 * width:
        return "vertical"
    elif width > 3 * height:
        return "horizontal"
    else:
        return "other"


def fix_fragmented_cracks_and_joints(pred_idx_map, min_area=50):
    corrected = pred_idx_map.copy()

    # Class IDs
    LONG = 3
    TRANS = 2
    MULTI = 6
    JOINT_L = 10
    JOINT_T = 9

    # All relevant pixels
    valid_classes = [LONG, TRANS, MULTI, JOINT_L, JOINT_T]
    mask = np.isin(pred_idx_map, valid_classes).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    for lbl in range(1, num_labels):
        coords = np.column_stack(np.where(labels == lbl))
        if coords.shape[0] < min_area:
            continue

        component_classes = pred_idx_map[labels == lbl]
        orientation = get_orientation(coords)

        unique = set(component_classes.tolist())

        # --------------------
        # VERTICAL STRUCTURES
        # --------------------
        if orientation == "vertical":
            if JOINT_L in unique:
                corrected[labels == lbl] = JOINT_L
            else:
                corrected[labels == lbl] = LONG

        # --------------------
        # HORIZONTAL STRUCTURES
        # --------------------
        elif orientation == "horizontal":
            if JOINT_T in unique:
                corrected[labels == lbl] = JOINT_T
            else:
                corrected[labels == lbl] = TRANS

        # else: keep original labels (small / noisy blobs)

    return corrected


def extend_joint_seals_to_image_end(mask):
    """
    Extend joint seals to image boundaries using
    AVERAGE width of the existing joint seal component.

    - Sealed Joint - L (10): vertical extension
    - Sealed Joint - T (9): horizontal extension
    """
    h, w = mask.shape
    out = mask.copy()

    # ============================
    # Sealed Joint - L (Vertical)
    # ============================
    JOINT_L = 10
    joint_l = (mask == JOINT_L).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(joint_l, connectivity=8)

    for i in range(1, num_labels):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue

        y_min, y_max = ys.min(), ys.max()

        # ---- compute average width ----
        widths = []
        for y in np.unique(ys):
            row_xs = xs[ys == y]
            widths.append(row_xs.max() - row_xs.min() + 1)

        avg_width = int(round(np.mean(widths)))
        avg_width = max(avg_width, 1)

        # center x using median (stable)
        center_x = int(np.median(xs))
        x_min = max(0, center_x - avg_width // 2)
        x_max = min(w - 1, x_min + avg_width - 1)

        # extend UP
        if y_min > 0:
            out[0:y_min, x_min:x_max + 1] = JOINT_L

        # extend DOWN
        if y_max < h - 1:
            out[y_max + 1:h, x_min:x_max + 1] = JOINT_L

    # ============================
    # Sealed Joint - T (Horizontal)
    # ============================
    JOINT_T = 9
    joint_t = (mask == JOINT_T).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(joint_t, connectivity=8)

    for i in range(1, num_labels):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()

        # ---- compute average height ----
        heights = []
        for x in np.unique(xs):
            col_ys = ys[xs == x]
            heights.append(col_ys.max() - col_ys.min() + 1)

        avg_height = int(round(np.mean(heights)))
        avg_height = max(avg_height, 1)

        # center y using median
        center_y = int(np.median(ys))
        y_min = max(0, center_y - avg_height // 2)
        y_max = min(h - 1, y_min + avg_height - 1)

        # extend LEFT
        if x_min > 0:
            out[y_min:y_max + 1, 0:x_min] = JOINT_T

        # extend RIGHT
        if x_max < w - 1:
            out[y_min:y_max + 1, x_max + 1:w] = JOINT_T

    return out


if __name__ == "__main__":

    WEIGHTS_PATH = r"Y:\Devendra_Files\segmentation_training\weights\UNET_concrete_10may.pth"
    BATCH_SIZE = 4
    main(imgs_root=rf"C:\Sidhesh\Distress\Asphalt\Extra Training Data\NAGAUR-JODHPUR_2026-05-02_12-13-17\DataSet-2\Dataset_2.1\Data",
         prediction_save_path=rf"Z:\Devendra\CONCRETE\COMBINED_SPLITTED\TRAIN\SPLIT\VAL\PRED_10may",
         weights_path=WEIGHTS_PATH,
         batch_size=BATCH_SIZE)

    # WEIGHTS_PATH = r"Y:\Devendra_Files\segmentation_training\weights\concrete_best.pth"
    # BATCH_SIZE = 4
    # main(imgs_root=rf"Z:\Devendra\CONCRETE\COMBINED_SPLITTED\TRAIN\SPLIT\VAL\IMAGES",
    #      prediction_save_path=rf"Z:\Devendra\CONCRETE\COMBINED_SPLITTED\TRAIN\SPLIT\VAL\PRED_old",
    #      weights_path=WEIGHTS_PATH,
    #      batch_size=BATCH_SIZE)

