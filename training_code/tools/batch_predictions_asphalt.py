"""
This script processes a batch of images for semantic segmentation using a pre-trained UNet++ model.
It reads images from a specified directory, applies necessary transformations,
and generates segmentation masks. The predicted masks
"""
from tqdm import tqdm
from random import shuffle
import sys
import os
import itertools
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))

# from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
# from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large
# from models.segformer.segformer import SegFormer
# from models.unet.unet import UNet
# from models.unet.mobilenet_unet import MobileV3Unet
# from models.unet.vgg_unet import VGG16UNet
# from models.fcn.fcn import fcn_resnet101
from models.unet.UnetPP import UNetPP
from models.unet.UnetPP import UNetPP
from models.unet.UnetPP_backbone import build_unetpp_model

COLOR_MAP = {
    (0, 0, 0): (0, "Background"),
    (255, 0, 0): (1, "Alligator"),
    (0, 0, 255): (2, "Transverse Crack"),
    (0, 255, 0): (3, "Longitudinal Crack"),
    (139, 69, 19): (4, "Pothole"),
    (255, 165, 0): (5, "Patches"),
}


def main(imgs_root=None, prediction_save_path=None, weights_path=None, batch_size=4):
    num_classes = 5 + 1  #14 #5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = (0.4787, 0.4787, 0.4787)  # 478
    std = (0.1472, 0.1472, 0.1472)  # 145

    data_transform = A.Compose([
        # A.Resize(1024, 419),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(), ])

    # Collect images
    images_list = [img for img in os.listdir(imgs_root) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    shuffle(images_list)
    os.makedirs(prediction_save_path, exist_ok=True)

    model = UNetPP(in_channels=3, num_classes=num_classes, deep_supervision=True, base_channels=64)
    # model = VGG16UNet(num_classes=num_classes, pretrain_backbone=False)
    # model = build_unetpp_model(
    #                 encoder="resnet50",   # or efficientnet_b3
    #                 pretrained=False,
    #                 in_channels=3,
    #                 num_classes=num_classes,
    #                 dec_ch=320,
    #                 use_se=True,
    #                 use_attn_gates=True,
    #                 deep_supervision=True
    # )

    pretrain_weights = torch.load(weights_path, map_location=device)
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)
    model.eval()

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(images_list), batch_size), desc=f"Processing {os.path.basename(imgs_root)}"):
            batch_files = images_list[i:i + batch_size]
            batch_imgs, orig_names = [], []

            for image in batch_files:
                original_img = cv2.imread(os.path.join(imgs_root, image))
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                # original_img = cv2.resize(original_img, (1280, 3000), interpolation=cv2.INTER_NEAREST)
                # img = data_transform(original_img)
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
                # pred = join_directional_multiclass(pred, radius=25, line_width=2)  # ⬅️ Added here
                pred = remove_small_components_multiclass(pred, min_area=50)
                pred_color = colorize_prediction(pred)
                save_path = os.path.join(prediction_save_path, fname.split('.')[0] + '.png')
                cv2.imwrite(save_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

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
    pts = contour.reshape(-1, 2)
    if len(pts) < 2:
        return pts[0], pts[0]
    max_dist_sq = -1
    p1, p2 = pts[0], pts[0]

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = float(pts[i][0]) - float(pts[j][0])
            dy = float(pts[i][1]) - float(pts[j][1])

            dist_sq = dx * dx + dy * dy

            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                p1, p2 = pts[i], pts[j]

    return p1, p2


def join_directional(mask, crack_type, radius=5, line_width=2):
    """Join cracks geometrically in direction-aware fashion."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    endpoints = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        p1, p2 = find_endpoints(cnt)
        endpoints.append((p1, p2))

    joined = mask.copy()

    for (a1, a2), (b1, b2) in itertools.combinations(endpoints, 2):
        for p, q in [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]:
            dx, dy = abs(int(p[0]) - int(q[0])), abs(int(p[1]) - int(q[1]))

            if crack_type == "Longitudinal Crack":
                if dx <= radius and dy <= 25:  # vertical direction
                    cv2.line(joined, tuple(p), tuple(q), 255, line_width)

            elif crack_type == "Transverse Crack":
                if dy <= radius and dx <= 25:  # horizontal direction
                    cv2.line(joined, tuple(p), tuple(q), 255, line_width)

            elif crack_type == "Alligator":
                if np.hypot(dx, dy) <= radius:  # general small gaps
                    cv2.line(joined, tuple(p), tuple(q), 255, line_width)

    return joined


def join_directional_multiclass(pred_idx_map, radius=5, line_width=2):
    """
    Applies direction-aware joining to specific crack types in a class index map.
    """
    joined_idx = pred_idx_map.copy()
    inv_color_map = {v[0]: v[1] for v in COLOR_MAP.values()}  # idx → name

    for idx, name in inv_color_map.items():
        if name not in ["Longitudinal Crack", "Transverse Crack", "Alligator",
                        "Multiple Crack", "Sealed Joint - T", "Sealed Joint - L"]:
            continue

        mask = (pred_idx_map == idx).astype(np.uint8) * 255
        joined_mask = join_directional(mask, name, radius=radius, line_width=line_width)
        joined_idx[joined_mask > 0] = idx

    return joined_idx


def overlay_mask_on_image(image, color_mask, alpha=0.5):
    """
    Overlay a multi-class RGB mask on a color image.
    """
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay


def remove_small_components_multiclass(mask, min_area=200):
    """
    Removes small connected components per class in a multi-class segmentation mask.

    Args:
        mask (np.ndarray): Segmentation mask (H×W, dtype int).
                           0 = background, 1..N = classes.
        min_area (int): Minimum number of pixels to keep.
        min_width (int): Minimum bounding box width.
        min_height (int): Minimum bounding box height.

    Returns:
        np.ndarray: Cleaned segmentation mask.
    """
    cleaned = np.zeros_like(mask, dtype=mask.dtype)

    for cls in np.unique(mask):
        if cls in [0, 2, 4, 7, 8, 11, 12]:  # skip background
            continue

        class_mask = (mask == cls).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = cls
    cleaned[mask == 2] = 2
    cleaned[mask == 4] = 4
    cleaned[mask == 7] = 7
    cleaned[mask == 8] = 8
    cleaned[mask == 11] = 11
    cleaned[mask == 12] = 12
    return cleaned


if __name__ == "__main__":
    WEIGHTS_PATH = r"D:\Devendra_Files\segmentation_training\weights\4july\latest_checkpoint.pth"
    BATCH_SIZE = 4

    main(
        imgs_root=r"Z:\Devendra\ASPHALT\Asphalt_GoldenSet_Test\IMAGES",
        prediction_save_path=r"Z:\Devendra\ASPHALT\Asphalt_GoldenSet_Test\PRED_MASKS",
        weights_path=WEIGHTS_PATH,
        batch_size=BATCH_SIZE
    )