# train_utils/hybrid_loss.py
import math
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def one_hot_ignore_index(targets: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    """
    Convert [B,H,W] targets -> one-hot [B,C,H,W], zeroing out ignore_index locations.
    """
    # Clone to avoid modifying original
    t = targets.clone()
    if ignore_index >= 0:
        t[t == ignore_index] = 0  # temporary valid index
    oh = F.one_hot(t.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    if ignore_index >= 0:
        mask = (targets != ignore_index).unsqueeze(1).float()
        oh = oh * mask
    return oh


def normalize_weights(w: torch.Tensor,
                      mode: str = "mean1",
                      eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize only VALID (>0) weights.

    Missing classes (weight=0) remain zero and
    do not affect normalization.
    """

    w = w.clone()

    valid = w > 0

    if valid.sum() == 0:
        return w

    if mode == "mean1":
        m = w[valid].mean().clamp_min(eps)
        w[valid] = w[valid] / m

    elif mode == "max1":
        m = w[valid].max().clamp_min(eps)
        w[valid] = w[valid] / m

    return w


def effective_number_weights(class_counts: torch.Tensor, beta: float = 0.9999, eps: float = 1e-8) -> torch.Tensor:
    """
    Cui et al., "Class-Balanced Loss Based on Effective Number of Samples".
    Returns weights proportional to (1 - beta) / (1 - beta^n_c).
    """
    n = class_counts.float().clamp_min(1.0)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, device=n.device), n)
    w = (1.0 - beta) / effective_num.clamp_min(eps)
    return w


def detach_mean(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.detach().abs().mean().clamp_min(eps)


# -----------------------------
# Losses
# -----------------------------
class FocalCrossEntropy(nn.Module):
    """
    Multi-class focal loss on logits (CE + (1-pt)^gamma), with optional per-class alpha.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 1.5,
                 ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE per-pixel
        ce = F.cross_entropy(inputs, targets, weight=None, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce)  # prob of the true class
        loss = (1.0 - pt) ** self.gamma * ce

        if self.alpha is not None:
            with torch.no_grad():
                flat_t = targets.view(-1).clamp_min(0)  # [B*H*W]
                alpha_t = self.alpha.gather(0, flat_t)  # [B*H*W]
                alpha_t = alpha_t.view_as(targets)  # [B,H,W]
            loss = loss * alpha_t

        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float()
            loss = loss * mask

        if self.reduction == "mean":
            denom = (targets != self.ignore_index).float().sum().clamp_min(
                1.0) if self.ignore_index >= 0 else loss.numel()
            return loss.sum() / denom
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SoftDiceLoss(nn.Module):
    """
    Multi-class soft Dice loss with optional per-class weights.
    """

    def __init__(self, num_classes: int, class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = -100, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("cw", class_weights if class_weights is not None else None)
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        target_oh = one_hot_ignore_index(targets, self.num_classes, self.ignore_index)

        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask

        # Flatten
        p = probs.reshape(probs.size(0), probs.size(1), -1)
        t = target_oh.reshape(target_oh.size(0), target_oh.size(1), -1)

        intersection = (p * t).sum(-1)
        denom = p.sum(-1) + t.sum(-1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)  # [B,C]
        dice_loss_c = 1.0 - dice  # [B,C]
        # Ignore background class
        dice_loss_c = dice_loss_c[:, 1:]
        # Per-class weighting
        if self.cw is not None:
            w = self.cw[1:].view(1, -1).expand_as(dice_loss_c)
            dice_loss_c = dice_loss_c * w

        return dice_loss_c.mean()


class TverskyLoss(nn.Module):
    """
    Multi-class Tversky loss (Dice generalization) with per-class weights.
    alpha -> weight for FN (increase for recall)
    beta  -> weight for FP
    """

    def __init__(self, num_classes: int, alpha: float = 0.7, beta: float = 0.3,
                 class_weights: Optional[torch.Tensor] = None, ignore_index: int = -100, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("cw", class_weights if class_weights is not None else None)
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        target_oh = one_hot_ignore_index(targets, self.num_classes, self.ignore_index)

        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask

        p = probs.reshape(probs.size(0), probs.size(1), -1)
        t = target_oh.reshape(target_oh.size(0), target_oh.size(1), -1)

        tp = (p * t).sum(-1)
        fp = (p * (1 - t)).sum(-1)
        fn = ((1 - p) * t).sum(-1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss_c = 1.0 - tversky  # [B,C]
        # ignore background
        loss_c = loss_c[:, 1:]
        if self.cw is not None:
            w = self.cw[1:].view(1, -1).expand_as(loss_c)
            loss_c = loss_c * w

        return loss_c.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky: (1 - Tversky) ** gamma
    Good for very small/elongated structures (cracks).
    """

    def __init__(self, num_classes: int, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.33,
                 class_weights: Optional[torch.Tensor] = None, ignore_index: int = -100, smooth: float = 1e-6):
        super().__init__()
        self.tversky = TverskyLoss(num_classes, alpha, beta, class_weights, ignore_index, smooth)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.tversky(inputs, targets)  # already averaged scalar
        # We need per-class to raise to gamma; re-run internals to keep it simple & robust:
        # (To avoid double compute, you can refactor if needed.)
        return base ** self.gamma


# -----------------------------
# Combined Criterion (handles dict/Tensor, aux output, auto-balance)
# -----------------------------
class CombinedCriterion(nn.Module):
    """
    Flexible, stable criterion for imbalanced semantic segmentation.

    Choose any subset of:
      - CrossEntropy (weighted)
      - Focal CE
      - Soft Dice (per-class weighted)
      - Tversky (per-class weighted)
      - Focal Tversky

    Features:
      - ignore_index support
      - class weight normalization
      - auto-balance loss scales (optional)
      - supports model outputs as Tensor or dict with {'out', 'aux'}
    """

    def __init__(self,
                 num_classes: int,
                 ce_weight: float = 0.3,
                 focal_weight: float = 0.4,
                 dice_weight: float = 0.2,
                 tversky_weight: float = 0.0,
                 focal_tversky_weight: float = 0.3,
                 # loss options
                 ce_class_weights: Optional[torch.Tensor] = None,  # shape [C]
                 dice_class_weights: Optional[torch.Tensor] = None,  # shape [C]
                 tversky_class_weights: Optional[torch.Tensor] = None,  # shape [C]
                 alpha_tversky: float = 0.7,
                 beta_tversky: float = 0.3,
                 focal_gamma: float = 1.5,
                 focal_tversky_gamma: float = 1.33,
                 ignore_index: int = 255,
                 aux_weight: float = 0.5,
                 normalize_losses: bool = False):  #True
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.normalize_losses = normalize_losses

        # Normalize class weights for stability
        if ce_class_weights is not None:
            ce_class_weights = normalize_weights(ce_class_weights, "mean1")
        if dice_class_weights is not None:
            dice_class_weights = normalize_weights(dice_class_weights, "mean1")
        if tversky_class_weights is not None:
            tversky_class_weights = normalize_weights(tversky_class_weights, "mean1")

        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights, ignore_index=ignore_index)
        self.focal = FocalCrossEntropy(alpha=ce_class_weights, gamma=focal_gamma, ignore_index=ignore_index)
        self.dice = SoftDiceLoss(num_classes, class_weights=dice_class_weights, ignore_index=ignore_index)
        self.tversky = TverskyLoss(num_classes, alpha=alpha_tversky, beta=beta_tversky,
                                   class_weights=tversky_class_weights, ignore_index=ignore_index)
        self.ftversky = FocalTverskyLoss(num_classes, alpha=alpha_tversky, beta=beta_tversky,
                                         gamma=focal_tversky_gamma,
                                         class_weights=tversky_class_weights, ignore_index=ignore_index)

        # Weights to mix losses
        self.w_ce = ce_weight
        self.w_focal = focal_weight
        self.w_dice = dice_weight
        self.w_tversky = tversky_weight
        self.w_ftversky = focal_tversky_weight

    def _compute_losses(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        parts = {}
        if self.w_ce > 0:
            parts["ce"] = self.ce(logits, target)
        if self.w_focal > 0:
            parts["focal"] = self.focal(logits, target)
        if self.w_dice > 0:
            parts["dice"] = self.dice(logits, target)
        if self.w_tversky > 0:
            parts["tversky"] = self.tversky(logits, target)
        if self.w_ftversky > 0:
            parts["ftversky"] = self.ftversky(logits, target)
        return parts

    def _mix(self, parts):

        total = 0.0

        if "ce" in parts:
            total += self.w_ce * parts["ce"]

        if "focal" in parts:
            total += self.w_focal * parts["focal"]

        if "dice" in parts:
            total += self.w_dice * parts["dice"]

        if "tversky" in parts:
            total += self.w_tversky * parts["tversky"]

        if "ftversky" in parts:
            total += self.w_ftversky * parts["ftversky"]

        return total

    def forward(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        # Support dict (with aux) or tensor
        if isinstance(outputs, dict):
            assert "out" in outputs, "Model dict output must have key 'out'"
            main_parts = self._compute_losses(outputs["out"], target)
            loss = self._mix(main_parts)

            if "aux" in outputs and self.aux_weight > 0:
                aux_parts = self._compute_losses(outputs["aux"], target)
                loss = loss + self.aux_weight * self._mix(aux_parts)
            return loss

        elif torch.is_tensor(outputs):
            parts = self._compute_losses(outputs, target)
            return self._mix(parts)

        else:
            raise TypeError("outputs must be Tensor or dict with 'out' (and optional 'aux').")


def build_crack_criterion(
        num_classes: int,
        class_counts: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        ignore_index: int = 255
) -> CombinedCriterion:

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    ce_w = None
    dice_w = None
    tv_w = None

    if class_counts is not None:

        class_counts = class_counts.to(
            device=device,
            dtype=torch.float32
        )

        # --------------------------------------------------
        # VALID CLASSES (ignore classes having 0 pixels)
        # --------------------------------------------------
        valid = class_counts > 0

        total_pixels = class_counts[valid].sum()

        # ==================================================
        # Cross Entropy / Focal Weights
        # ==================================================
        ce_w = torch.zeros_like(class_counts)

        ce_w[valid] = total_pixels / class_counts[valid]

        # make rare classes stronger
        ce_w[valid] = torch.pow(ce_w[valid], 0.65)

        # normalize only valid classes
        ce_w = normalize_weights(
            ce_w,
            mode="mean1"
        )

        ce_w = torch.clamp(
            ce_w,
            min=0.5,
            max=12.0
        )

        # Missing classes receive zero weight
        ce_w[~valid] = 0.0

        # ==================================================
        # Dice / Tversky Weights
        # ==================================================
        dice_w = torch.zeros_like(class_counts)

        dice_w[valid] = total_pixels / class_counts[valid]

        dice_w = normalize_weights(
            dice_w,
            mode="mean1"
        )

        dice_w = torch.clamp(
            dice_w,
            min=0.5,
            max=10.0
        )

        dice_w[~valid] = 0.0

        # ==================================================
        # Manual weighting
        # ==================================================

        # Background
        ce_w[0] *= 0.10
        dice_w[0] *= 0.10

        # Class 1 = Alligator (missing)
        # Keep weight = 0

        # Class 2 = Transverse Crack
        ce_w[2] *= 1.5
        dice_w[2] *= 1.5

        # Class 3 = Longitudinal Crack
        ce_w[3] *= 3.0
        dice_w[3] *= 3.0

        # Class 4 = Pothole
        ce_w[4] *= 0.8
        dice_w[4] *= 0.8

        # Class 5 = Patch
        ce_w[5] *= 1.5
        dice_w[5] *= 1.5

        tv_w = dice_w.clone()

    print("\nLoaded Class Counts:")
    print(class_counts)

    print("\nFinal CE Weights")
    print(ce_w)

    print("\nFinal Dice Weights")
    print(dice_w)

    print("\nFinal Tversky Weights")
    print(tv_w)

    criterion = CombinedCriterion(
        num_classes=num_classes,

        ce_weight=0.20,
        focal_weight=0.20,
        dice_weight=0.20,
        tversky_weight=0.0,
        focal_tversky_weight=0.40,

        ce_class_weights=ce_w,
        dice_class_weights=dice_w,
        tversky_class_weights=tv_w,

        focal_gamma=2.0,

        alpha_tversky=0.6,
        beta_tversky=0.4,
        focal_tversky_gamma=1.5,

        ignore_index=ignore_index,
        aux_weight=0.5,
        normalize_losses=False
    ).to(device)

    return criterion
