# train_utils/hybrid_loss.py

from typing import Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def one_hot_ignore_index(
        targets: torch.Tensor,
        num_classes: int,
        ignore_index: int
) -> torch.Tensor:
    """
    Convert:
        targets [B, H, W]

    To:
        one-hot [B, C, H, W]

    Pixels equal to ignore_index are zeroed in all channels.
    """

    t = targets.clone()

    if ignore_index >= 0:
        ignore_mask = (t == ignore_index)
        t[ignore_mask] = 0

    oh = F.one_hot(
        t.long(),
        num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    if ignore_index >= 0:
        valid_mask = (
            targets != ignore_index
        ).unsqueeze(1).to(dtype=oh.dtype)

        oh = oh * valid_mask

    return oh


def normalize_weights(
        w: torch.Tensor,
        mode: str = "mean1",
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize only positive weights.

    Zero weights remain zero.
    """

    w = w.clone()

    valid = w > 0

    if not valid.any():
        return w

    if mode == "mean1":

        mean_value = w[valid].mean().clamp_min(eps)

        w[valid] = w[valid] / mean_value

    elif mode == "max1":

        max_value = w[valid].max().clamp_min(eps)

        w[valid] = w[valid] / max_value

    elif mode in ("none", None):

        pass

    else:

        raise ValueError(
            f"Unknown normalization mode: {mode}"
        )

    return w


def effective_number_weights(
        class_counts: torch.Tensor,
        beta: float = 0.9999,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Class-balanced weights from:

    Cui et al.
    "Class-Balanced Loss Based on Effective Number of Samples"
    """

    n = class_counts.float().clamp_min(1.0)

    beta_tensor = torch.tensor(
        beta,
        device=n.device,
        dtype=n.dtype
    )

    effective_num = 1.0 - torch.pow(
        beta_tensor,
        n
    )

    weights = (
        (1.0 - beta)
        / effective_num.clamp_min(eps)
    )

    return weights


def weighted_present_mean(
        loss_c: torch.Tensor,
        present: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute weighted mean over classes present in the GT.

    Parameters
    ----------
    loss_c:
        [B, C]

    present:
        [B, C] bool

    class_weights:
        [C] or None

    Important
    ---------
    Absent classes do not contribute to Dice/Tversky loss.

    This is especially important for extremely rare classes such as:
        - Transverse Crack
        - Longitudinal Crack
    """

    present_f = present.to(dtype=loss_c.dtype)

    if class_weights is None:

        effective_weights = present_f

    else:

        weights = class_weights.view(
            1, -1
        ).to(
            device=loss_c.device,
            dtype=loss_c.dtype
        )

        effective_weights = (
            weights * present_f
        )

    denominator = effective_weights.sum()

    if denominator.detach().item() <= 0:

        # Differentiable zero
        return loss_c.sum() * 0.0

    return (
        loss_c * effective_weights
    ).sum() / denominator.clamp_min(eps)


# ============================================================
# Focal Cross Entropy
# ============================================================

class FocalCrossEntropy(nn.Module):
    """
    Multi-class Focal Cross Entropy.

    Fixes:
    ------
    1. ignore_index=255 is handled safely before alpha lookup.
    2. Ignored pixels do not contribute to the denominator.
    3. Class alpha weights are applied only to valid pixels.
    """

    def __init__(
            self,
            alpha: Optional[torch.Tensor] = None,
            gamma: float = 2.0,
            ignore_index: int = 255,
            reduction: str = "mean"
    ):
        super().__init__()

        if alpha is not None:
            self.register_buffer(
                "alpha",
                alpha.clone().detach()
            )
        else:
            self.alpha = None

        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction


    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Valid pixel mask
        # ----------------------------------------------------

        if self.ignore_index >= 0:

            valid_mask = (
                targets != self.ignore_index
            )

        else:

            valid_mask = torch.ones_like(
                targets,
                dtype=torch.bool
            )

        # ----------------------------------------------------
        # Standard CE per pixel
        # ----------------------------------------------------

        ce = F.cross_entropy(
            inputs,
            targets,
            weight=None,
            ignore_index=self.ignore_index,
            reduction="none"
        )

        # Probability of true class
        pt = torch.exp(-ce)

        # Focal modulation
        loss = (
            (1.0 - pt).pow(self.gamma)
            * ce
        )

        # ----------------------------------------------------
        # Safe class alpha weighting
        # ----------------------------------------------------

        if self.alpha is not None:

            alpha_t = torch.zeros_like(
                loss
            )

            if valid_mask.any():

                valid_targets = targets[
                    valid_mask
                ].long()

                alpha_t[valid_mask] = self.alpha[
                    valid_targets
                ]

            loss = loss * alpha_t

        # Remove ignored pixels
        loss = loss * valid_mask.to(
            dtype=loss.dtype
        )

        # ----------------------------------------------------
        # Reduction
        # ----------------------------------------------------

        if self.reduction == "mean":

            denominator = valid_mask.sum().clamp_min(1)

            return (
                loss.sum()
                / denominator.to(loss.dtype)
            )

        elif self.reduction == "sum":

            return loss.sum()

        elif self.reduction == "none":

            return loss

        else:

            raise ValueError(
                f"Unknown reduction: {self.reduction}"
            )


# ============================================================
# Soft Dice Loss
# ============================================================

class SoftDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss.

    Important behavior:
    -------------------
    - Background is excluded.
    - Classes absent from a sample's GT are excluded.
    - Per-class weights use a proper weighted mean.

    This prevents rare crack classes from producing large overlap-loss
    penalties in crops where they do not exist.
    """

    def __init__(
            self,
            num_classes: int,
            class_weights: Optional[torch.Tensor] = None,
            ignore_index: int = 255,
            smooth: float = 1e-6
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

        if class_weights is not None:
            self.register_buffer(
                "cw",
                class_weights.clone().detach()
            )
        else:
            self.cw = None


    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:

        probs = F.softmax(
            inputs,
            dim=1
        )

        target_oh = one_hot_ignore_index(
            targets,
            self.num_classes,
            self.ignore_index
        )

        # ----------------------------------------------------
        # Ignore invalid pixels
        # ----------------------------------------------------

        if self.ignore_index >= 0:

            valid_mask = (
                targets != self.ignore_index
            ).unsqueeze(1).to(
                dtype=probs.dtype
            )

            probs = probs * valid_mask

        # ----------------------------------------------------
        # Flatten spatial dimensions
        # ----------------------------------------------------

        p = probs.reshape(
            probs.size(0),
            probs.size(1),
            -1
        )

        t = target_oh.reshape(
            target_oh.size(0),
            target_oh.size(1),
            -1
        )

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        intersection = (
            p * t
        ).sum(dim=-1)

        denominator = (
            p.sum(dim=-1)
            + t.sum(dim=-1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        loss_c = 1.0 - dice

        # ----------------------------------------------------
        # Remove background
        # ----------------------------------------------------

        loss_c = loss_c[:, 1:]
        t_fg = t[:, 1:]

        # ----------------------------------------------------
        # Only classes present in GT
        # ----------------------------------------------------

        present = (
            t_fg.sum(dim=-1) > 0
        )

        class_weights = None

        if self.cw is not None:
            class_weights = self.cw[1:]

        return weighted_present_mean(
            loss_c=loss_c,
            present=present,
            class_weights=class_weights
        )


# ============================================================
# Tversky Loss
# ============================================================

class TverskyLoss(nn.Module):
    """
    Multi-class Tversky Loss.

    Convention used here:

        denominator =
            TP
            + alpha * FN
            + beta  * FP

    Therefore:

        Higher alpha -> stronger FN penalty -> higher recall
        Higher beta  -> stronger FP penalty -> higher precision

    Important behavior:
    -------------------
    - Background excluded.
    - Absent GT classes excluded.
    - Proper weighted mean.
    """

    def __init__(
            self,
            num_classes: int,
            alpha: float = 0.6,
            beta: float = 0.4,
            class_weights: Optional[torch.Tensor] = None,
            ignore_index: int = 255,
            smooth: float = 1e-6
    ):
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = smooth

        if class_weights is not None:
            self.register_buffer(
                "cw",
                class_weights.clone().detach()
            )
        else:
            self.cw = None


    def per_class_loss(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ):
        """
        Return:
            loss_c   [B, C-1]
            present  [B, C-1]
        """

        probs = F.softmax(
            inputs,
            dim=1
        )

        target_oh = one_hot_ignore_index(
            targets,
            self.num_classes,
            self.ignore_index
        )

        # ----------------------------------------------------
        # Ignore invalid pixels
        # ----------------------------------------------------

        if self.ignore_index >= 0:

            valid_mask = (
                targets != self.ignore_index
            ).unsqueeze(1).to(
                dtype=probs.dtype
            )

            probs = probs * valid_mask

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        p = probs.reshape(
            probs.size(0),
            probs.size(1),
            -1
        )

        t = target_oh.reshape(
            target_oh.size(0),
            target_oh.size(1),
            -1
        )

        # ----------------------------------------------------
        # TP / FP / FN
        # ----------------------------------------------------

        tp = (
            p * t
        ).sum(dim=-1)

        fp = (
            p * (1.0 - t)
        ).sum(dim=-1)

        fn = (
            (1.0 - p) * t
        ).sum(dim=-1)

        # ----------------------------------------------------
        # Tversky score
        # ----------------------------------------------------

        score = (
            tp + self.smooth
        ) / (
            tp
            + self.alpha * fn
            + self.beta * fp
            + self.smooth
        )

        loss_c = 1.0 - score

        # ----------------------------------------------------
        # Remove background
        # ----------------------------------------------------

        loss_c = loss_c[:, 1:]
        t_fg = t[:, 1:]

        # Present classes only
        present = (
            t_fg.sum(dim=-1) > 0
        )

        return loss_c, present


    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:

        loss_c, present = self.per_class_loss(
            inputs,
            targets
        )

        class_weights = None

        if self.cw is not None:
            class_weights = self.cw[1:]

        return weighted_present_mean(
            loss_c=loss_c,
            present=present,
            class_weights=class_weights
        )


# ============================================================
# Focal Tversky Loss
# ============================================================

class FocalTverskyLoss(nn.Module):
    """
    True Focal Tversky Loss.

    Correct implementation:

        focal_loss_c = (1 - Tversky_c) ** gamma

    The focal exponent is applied PER CLASS before reduction.

    The old implementation incorrectly did:

        mean(TverskyLoss) ** gamma

    which is not true Focal Tversky.
    """

    def __init__(
            self,
            num_classes: int,
            alpha: float = 0.6,
            beta: float = 0.4,
            gamma: float = 1.5,
            class_weights: Optional[torch.Tensor] = None,
            ignore_index: int = 255,
            smooth: float = 1e-6
    ):
        super().__init__()

        self.gamma = gamma

        self.base_tversky = TverskyLoss(
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            class_weights=class_weights,
            ignore_index=ignore_index,
            smooth=smooth
        )


    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:

        loss_c, present = (
            self.base_tversky.per_class_loss(
                inputs,
                targets
            )
        )

        # Apply focal exponent PER CLASS
        focal_loss_c = loss_c.pow(
            self.gamma
        )

        class_weights = None

        if self.base_tversky.cw is not None:
            class_weights = (
                self.base_tversky.cw[1:]
            )

        return weighted_present_mean(
            loss_c=focal_loss_c,
            present=present,
            class_weights=class_weights
        )


# ============================================================
# Combined Criterion
# ============================================================

class CombinedCriterion(nn.Module):
    """
    Combined segmentation criterion.

    Supports:
        - Cross Entropy
        - Focal Cross Entropy
        - Soft Dice
        - Tversky
        - Focal Tversky

    Supports model outputs:
        Tensor

    Or:
        {
            "out": logits,
            "aux": aux_logits
        }
    """

    def __init__(
            self,
            num_classes: int,

            ce_weight: float = 0.20,
            focal_weight: float = 0.20,
            dice_weight: float = 0.20,
            tversky_weight: float = 0.0,
            focal_tversky_weight: float = 0.40,

            ce_class_weights: Optional[torch.Tensor] = None,
            dice_class_weights: Optional[torch.Tensor] = None,
            tversky_class_weights: Optional[torch.Tensor] = None,

            alpha_tversky: float = 0.6,
            beta_tversky: float = 0.4,

            focal_gamma: float = 2.0,
            focal_tversky_gamma: float = 1.5,

            ignore_index: int = 255,
            aux_weight: float = 0.5,

            normalize_class_weights: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

        # ----------------------------------------------------
        # Optional normalization
        #
        # Default FALSE because build_crack_criterion()
        # already generates final tuned weights.
        # ----------------------------------------------------

        if normalize_class_weights:

            if ce_class_weights is not None:
                ce_class_weights = normalize_weights(
                    ce_class_weights,
                    mode="mean1"
                )

            if dice_class_weights is not None:
                dice_class_weights = normalize_weights(
                    dice_class_weights,
                    mode="mean1"
                )

            if tversky_class_weights is not None:
                tversky_class_weights = normalize_weights(
                    tversky_class_weights,
                    mode="mean1"
                )

        # ----------------------------------------------------
        # Loss functions
        # ----------------------------------------------------

        self.ce = nn.CrossEntropyLoss(
            weight=ce_class_weights,
            ignore_index=ignore_index
        )

        self.focal = FocalCrossEntropy(
            alpha=ce_class_weights,
            gamma=focal_gamma,
            ignore_index=ignore_index
        )

        self.dice = SoftDiceLoss(
            num_classes=num_classes,
            class_weights=dice_class_weights,
            ignore_index=ignore_index
        )

        self.tversky = TverskyLoss(
            num_classes=num_classes,
            alpha=alpha_tversky,
            beta=beta_tversky,
            class_weights=tversky_class_weights,
            ignore_index=ignore_index
        )

        self.ftversky = FocalTverskyLoss(
            num_classes=num_classes,
            alpha=alpha_tversky,
            beta=beta_tversky,
            gamma=focal_tversky_gamma,
            class_weights=tversky_class_weights,
            ignore_index=ignore_index
        )

        # ----------------------------------------------------
        # Loss mixing weights
        # ----------------------------------------------------

        self.w_ce = ce_weight
        self.w_focal = focal_weight
        self.w_dice = dice_weight
        self.w_tversky = tversky_weight
        self.w_ftversky = focal_tversky_weight


    def _compute_losses(
            self,
            logits: torch.Tensor,
            target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        parts = {}

        if self.w_ce > 0:
            parts["ce"] = self.ce(
                logits,
                target
            )

        if self.w_focal > 0:
            parts["focal"] = self.focal(
                logits,
                target
            )

        if self.w_dice > 0:
            parts["dice"] = self.dice(
                logits,
                target
            )

        if self.w_tversky > 0:
            parts["tversky"] = self.tversky(
                logits,
                target
            )

        if self.w_ftversky > 0:
            parts["ftversky"] = self.ftversky(
                logits,
                target
            )

        return parts


    def _mix(
            self,
            parts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        total = None

        weighted_parts = []

        if "ce" in parts:
            weighted_parts.append(
                self.w_ce * parts["ce"]
            )

        if "focal" in parts:
            weighted_parts.append(
                self.w_focal * parts["focal"]
            )

        if "dice" in parts:
            weighted_parts.append(
                self.w_dice * parts["dice"]
            )

        if "tversky" in parts:
            weighted_parts.append(
                self.w_tversky * parts["tversky"]
            )

        if "ftversky" in parts:
            weighted_parts.append(
                self.w_ftversky * parts["ftversky"]
            )

        if not weighted_parts:
            raise RuntimeError(
                "At least one loss weight must be > 0."
            )

        total = weighted_parts[0]

        for value in weighted_parts[1:]:
            total = total + value

        return total


    def forward(
            self,
            outputs: Union[
                torch.Tensor,
                Dict[str, torch.Tensor]
            ],
            target: torch.Tensor
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Dict output with optional auxiliary output
        # ----------------------------------------------------

        if isinstance(outputs, dict):

            if "out" not in outputs:

                raise KeyError(
                    "Model dict output must contain key 'out'."
                )

            main_parts = self._compute_losses(
                outputs["out"],
                target
            )

            loss = self._mix(
                main_parts
            )

            if (
                "aux" in outputs
                and outputs["aux"] is not None
                and self.aux_weight > 0
            ):

                aux_parts = self._compute_losses(
                    outputs["aux"],
                    target
                )

                aux_loss = self._mix(
                    aux_parts
                )

                loss = (
                    loss
                    + self.aux_weight * aux_loss
                )

            return loss

        # ----------------------------------------------------
        # Tensor output
        # ----------------------------------------------------

        elif torch.is_tensor(outputs):

            parts = self._compute_losses(
                outputs,
                target
            )

            return self._mix(
                parts
            )

        else:

            raise TypeError(
                "outputs must be a Tensor or a dict "
                "containing 'out' and optional 'aux'."
            )


# ============================================================
# Criterion Builder
# ============================================================

def build_crack_criterion(
        num_classes: int,
        class_counts: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        ignore_index: int = 255
) -> CombinedCriterion:
    """
    Build criterion for:

        0 = Background
        1 = Alligator
        2 = Transverse Crack
        3 = Longitudinal Crack
        4 = Pothole
        5 = Patch

    Designed for severe imbalance where:
        - Transverse is extremely rare
        - Longitudinal is very rare
        - Both may be confused with Alligator
    """

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    ce_w = None
    dice_w = None
    tv_w = None

    # ========================================================
    # Generate class weights
    # ========================================================

    if class_counts is not None:

        class_counts = class_counts.to(
            device=device,
            dtype=torch.float32
        )

        if class_counts.numel() != num_classes:

            raise ValueError(
                f"class_counts contains "
                f"{class_counts.numel()} classes, "
                f"but num_classes={num_classes}"
            )

        # ----------------------------------------------------
        # Valid classes
        # ----------------------------------------------------

        valid = class_counts > 0

        if not valid.any():

            raise ValueError(
                "class_counts contains no valid classes."
            )

        total_pixels = class_counts[
            valid
        ].sum()

        # ====================================================
        # CE / Focal weights
        # ====================================================

        ce_w = torch.zeros_like(
            class_counts
        )

        # Inverse frequency
        ce_w[valid] = (
            total_pixels
            / class_counts[valid]
        )

        # Compress extreme imbalance
        ce_w[valid] = torch.pow(
            ce_w[valid],
            0.65
        )

        # Normalize before clamping/manual tuning
        ce_w = normalize_weights(
            ce_w,
            mode="mean1"
        )

        # Stable range
        ce_w[valid] = torch.clamp(
            ce_w[valid],
            min=0.5,
            max=12.0
        )

        # Missing classes remain zero
        ce_w[~valid] = 0.0

        # ====================================================
        # Dice / Tversky weights
        # ====================================================

        dice_w = torch.zeros_like(
            class_counts
        )

        dice_w[valid] = (
            total_pixels
            / class_counts[valid]
        )

        dice_w = normalize_weights(
            dice_w,
            mode="mean1"
        )

        dice_w[valid] = torch.clamp(
            dice_w[valid],
            min=0.5,
            max=10.0
        )

        dice_w[~valid] = 0.0

        # ====================================================
        # Manual tuning
        #
        # Current dataset problem:
        #
        # Transverse / Longitudinal
        #           ↓
        # predicted as Alligator
        #
        # Keep crack boost moderate because automatic weighting
        # already gives rare crack classes large weights.
        # ====================================================

        manual_ce = torch.tensor(
            [
                0.10,   # 0 Background
                0.50,   # 1 Alligator
                1.50,   # 2 Transverse
                1.50,   # 3 Longitudinal
                1.00,   # 4 Pothole
                1.00    # 5 Patch
            ],
            device=device,
            dtype=ce_w.dtype
        )

        manual_overlap = torch.tensor(
            [
                0.10,   # 0 Background
                0.75,   # 1 Alligator
                1.50,   # 2 Transverse
                1.50,   # 3 Longitudinal
                1.00,   # 4 Pothole
                1.00    # 5 Patch
            ],
            device=device,
            dtype=dice_w.dtype
        )

        # Safety for different class counts
        if num_classes != 6:

            raise ValueError(
                "The current manual crack weighting is configured "
                "for exactly 6 classes."
            )

        ce_w = (
            ce_w * manual_ce
        )

        dice_w = (
            dice_w * manual_overlap
        )

        # Ensure missing classes remain zero
        ce_w[~valid] = 0.0
        dice_w[~valid] = 0.0

        tv_w = dice_w.clone()

    # ========================================================
    # Print ACTUAL weights passed to losses
    # ========================================================

    print("\n" + "=" * 60)

    print("Loaded Class Counts:")
    print(class_counts)

    print("\nFinal CE / Focal Weights:")
    print(ce_w)

    print("\nFinal Dice Weights:")
    print(dice_w)

    print("\nFinal Tversky / Focal Tversky Weights:")
    print(tv_w)

    print("=" * 60 + "\n")

    # ========================================================
    # Build criterion
    # ========================================================

    criterion = CombinedCriterion(

        num_classes=num_classes,

        # ----------------------------------------------------
        # Loss mixture
        # ----------------------------------------------------

        ce_weight=0.20,

        focal_weight=0.20,

        dice_weight=0.20,

        tversky_weight=0.0,

        focal_tversky_weight=0.40,

        # ----------------------------------------------------
        # Class weights
        # ----------------------------------------------------

        ce_class_weights=ce_w,

        dice_class_weights=dice_w,

        tversky_class_weights=tv_w,

        # ----------------------------------------------------
        # Focal CE
        # ----------------------------------------------------

        focal_gamma=2.0,

        # ----------------------------------------------------
        # Tversky
        #
        # alpha = FN penalty
        # beta  = FP penalty
        #
        # 0.6 / 0.4 gives slightly more importance to recall.
        # ----------------------------------------------------

        alpha_tversky=0.60,

        beta_tversky=0.40,

        focal_tversky_gamma=1.50,

        # ----------------------------------------------------
        # General
        # ----------------------------------------------------

        ignore_index=ignore_index,

        aux_weight=0.5,

        # IMPORTANT:
        # Do not normalize again.
        # Printed weights = actual weights used.
        normalize_class_weights=False

    ).to(device)

    return criterion