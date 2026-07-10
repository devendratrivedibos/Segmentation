"""
TRAINING AND EVALUATION UTILITIES
"""

from pathlib import Path
from typing import Union, Dict, Optional

import torch
from torch import nn

from . import distributed_utils as utils
from .dice_coefficient_loss import (
    dice_loss,
    build_target,
)
from .hybrid_loss_asphalt import build_crack_criterion


# ============================================================
# Global configuration
# ============================================================

IGNORE_INDEX = 255

NUM_CLASSES = 6

# ------------------------------------------------------------
# Default device
#
# The training functions still use the device passed to them.
# This default device is only used to construct the global
# criterion.
# ------------------------------------------------------------

DEFAULT_DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load class counts
# ============================================================

project_root = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

counts_file = (
    project_root
    / "weights"
    / "class_counts_asphalt.pt"
)


def load_class_counts(
        file_path: Path,
        num_classes: int
) -> Optional[torch.Tensor]:
    """
    Load and validate class pixel counts.

    Expected shape:
        [num_classes]
    """

    if not file_path.exists():

        print(
            "\nWARNING:"
            f"\nClass-count file not found:"
            f"\n{file_path}"
        )

        return None

    class_counts = torch.load(
        file_path,
        map_location="cpu",
        weights_only=True
    )

    class_counts = torch.as_tensor(
        class_counts,
        dtype=torch.float32
    ).flatten()

    if class_counts.numel() != num_classes:

        raise ValueError(
            f"\nInvalid class-count file."
            f"\nExpected {num_classes} values."
            f"\nReceived {class_counts.numel()} values."
            f"\nFile: {file_path}"
        )

    if torch.any(class_counts < 0):

        raise ValueError(
            "Class counts cannot contain negative values."
        )

    print("\n" + "=" * 60)

    print("Loaded Class Counts:")

    for cls_idx, count in enumerate(class_counts):

        print(
            f"Class {cls_idx}: "
            f"{int(count.item()):,} pixels"
        )

    print("=" * 60)

    return class_counts


CLASS_COUNTS = load_class_counts(
    file_path=counts_file,
    num_classes=NUM_CLASSES
)


# ============================================================
# Build criterion
# ============================================================

criterion = build_crack_criterion(

    num_classes=NUM_CLASSES,

    class_counts=CLASS_COUNTS,

    device=DEFAULT_DEVICE,

    ignore_index=IGNORE_INDEX
)


# ============================================================
# Helpers
# ============================================================

def _main_output(
        outputs: Union[
            torch.Tensor,
            Dict[str, torch.Tensor]
        ]
) -> torch.Tensor:
    """
    Return primary logits:

        [B, C, H, W]

    Supports:

        Tensor

    or:

        {
            "out": Tensor,
            "aux": Tensor
        }
    """

    if isinstance(outputs, dict):

        if "out" not in outputs:

            raise KeyError(
                "Model dict output must contain key 'out'."
            )

        return outputs["out"]

    if torch.is_tensor(outputs):

        return outputs

    raise TypeError(
        "Model output must be a Tensor or dict."
    )


def _print_batch_debug(
        epoch: int,
        target: torch.Tensor,
        logits: torch.Tensor,
        loss: torch.Tensor,
        num_classes: int
):
    """
    Print class distribution for the first batch of each epoch.
    """

    with torch.no_grad():

        pred = logits.argmax(dim=1)
        print("\n" + "=" * 60)
        print(f"EPOCH {epoch} — FIRST BATCH DEBUG")
        print("=" * 60)
        print(f"Loss: {loss.detach().item():.6f}")
        print("\nGround Truth Pixel Counts:")

        for cls in range(num_classes):

            count = (target == cls).sum().item()

            percentage = (100.0 * count / max(target.numel(), 1))

            print(f"  Class {cls}: {count:,} ({percentage:.6f}%)")

        ignore_count = (
            target == IGNORE_INDEX
        ).sum().item()

        if ignore_count > 0:

            print(f"  Ignore ({IGNORE_INDEX}):{ignore_count:,}")

        print("\nPrediction Pixel Counts:")

        for cls in range(num_classes):

            count = (
                pred == cls
            ).sum().item()

            percentage = (
                100.0
                * count
                / max(pred.numel(), 1)
            )

            print(
                f"  Class {cls}: "
                f"{count:,} "
                f"({percentage:.6f}%)"
            )

        print("\nClasses Present:")

        print(
            "  GT:",
            torch.unique(target).detach().cpu().tolist()
        )

        print(
            "  Pred:",
            torch.unique(pred).detach().cpu().tolist()
        )

        print("=" * 60 + "\n")


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
        model,
        data_loader,
        device,
        num_classes
):
    """
    Evaluate model using:

        - Confusion matrix
        - Dice coefficient
    """

    model.eval()

    confmat = utils.ConfusionMatrix(
        num_classes
    )

    dice = utils.DiceCoefficient(
        num_classes=num_classes,
        ignore_index=IGNORE_INDEX
    )

    metric_logger = utils.MetricLogger(
        delimiter="  "
    )

    header = "Test:"

    for image, target in metric_logger.log_every(
            data_loader,
            50,
            header
    ):

        image = image.to(
            device,
            non_blocking=True
        )

        target = target.to(
            device,
            non_blocking=True
        )

        outputs = model(image)

        logits = _main_output(outputs)

        pred = logits.argmax(
            dim=1
        )

        # ----------------------------------------------------
        # Update metrics
        # ----------------------------------------------------

        confmat.update(
            target.flatten(),
            pred.flatten()
        )

        dice.update(
            logits,
            target
        )

    confmat.reduce_from_all_processes()

    dice.reduce_from_all_processes()

    return (
        confmat,
        dice.value.item()
    )


# ============================================================
# Training — one epoch
# ============================================================

def train_one_epoch(
        model,
        optimizer,
        data_loader,
        device,
        epoch,
        num_classes,
        lr_scheduler,
        print_freq=10,
        scaler=None,
        grad_clip_norm: float = 0.0
):
    """
    Train one epoch.

    Features:
    ---------
    - Hybrid crack loss
    - Tensor or dict model outputs
    - Optional AMP
    - Optional gradient clipping
    - Non-finite loss detection
    - First-batch class diagnostics
    """

    model.train()

    metric_logger = utils.MetricLogger(
        delimiter="  "
    )

    metric_logger.add_meter(
        "lr",
        utils.SmoothedValue(
            window_size=1,
            fmt="{value:.6f}"
        )
    )

    header = f"Epoch: [{epoch}]"

    # --------------------------------------------------------
    # AMP is enabled only when:
    #
    # 1. scaler exists
    # 2. training device is CUDA
    # --------------------------------------------------------

    amp_enabled = (
        scaler is not None
        and device.type == "cuda"
    )

    for batch_idx, (image, target) in enumerate(

            metric_logger.log_every(
                data_loader,
                print_freq,
                header
            )
    ):

        # ----------------------------------------------------
        # Move batch
        # ----------------------------------------------------

        image = image.to(
            device,
            non_blocking=True
        )

        target = target.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Clear gradients BEFORE forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.amp.autocast(
                device_type="cuda",
                enabled=amp_enabled
        ):

            outputs = model(image)

            loss = criterion(
                outputs,
                target
            )

        # ----------------------------------------------------
        # Check numerical stability
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            logits = _main_output(outputs)

            print("\n" + "!" * 60)

            print(
                "NON-FINITE LOSS DETECTED"
            )

            print(
                f"Epoch: {epoch}"
            )

            print(
                f"Batch: {batch_idx}"
            )

            print(
                f"Loss: {loss.detach().item()}"
            )

            print(
                "GT classes:",
                torch.unique(target)
                .detach()
                .cpu()
                .tolist()
            )

            print(
                "Logits finite:",
                torch.isfinite(logits)
                .all()
                .item()
            )

            print("!" * 60)

            raise FloatingPointError(
                "Training stopped because loss "
                "became NaN or Inf."
            )

        # ----------------------------------------------------
        # First-batch diagnostics
        # ----------------------------------------------------

        if batch_idx == 0:

            logits = _main_output(
                outputs
            )

            _print_batch_debug(

                epoch=epoch,

                target=target,

                logits=logits,

                loss=loss,

                num_classes=num_classes
            )

        # ----------------------------------------------------
        # Backward — AMP
        # ----------------------------------------------------

        if amp_enabled:

            scaler.scale(
                loss
            ).backward()

            if (
                grad_clip_norm is not None
                and grad_clip_norm > 0
            ):

                # Unscale before clipping
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_norm
                )

            scaler.step(
                optimizer
            )

            scaler.update()

        # ----------------------------------------------------
        # Backward — FP32
        # ----------------------------------------------------

        else:

            loss.backward()

            if (
                grad_clip_norm is not None
                and grad_clip_norm > 0
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_norm
                )

            optimizer.step()

        # ----------------------------------------------------
        # LR scheduler
        # ----------------------------------------------------

        if lr_scheduler is not None:

            lr_scheduler.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        lr = optimizer.param_groups[0][
            "lr"
        ]

        metric_logger.update(

            loss=loss.detach().item(),

            lr=lr
        )

    # --------------------------------------------------------
    # Epoch statistics
    # --------------------------------------------------------

    epoch_loss = (
        metric_logger
        .meters["loss"]
        .global_avg
    )

    return epoch_loss, lr


# ============================================================
# Backward-compatible alias
# ============================================================

def train_one_epoch_loss(
        model,
        optimizer,
        data_loader,
        device,
        epoch,
        num_classes,
        lr_scheduler,
        print_freq=10,
        scaler=None,
        grad_clip_norm: float = 0.0
):
    """
    Backward-compatible wrapper.

    Uses exactly the same implementation as train_one_epoch()
    to prevent the two functions from diverging.
    """

    return train_one_epoch(

        model=model,

        optimizer=optimizer,

        data_loader=data_loader,

        device=device,

        epoch=epoch,

        num_classes=num_classes,

        lr_scheduler=lr_scheduler,

        print_freq=print_freq,

        scaler=scaler,

        grad_clip_norm=grad_clip_norm
    )


# ============================================================
# Learning-rate scheduler
# ============================================================

def create_lr_scheduler(
        optimizer,
        num_step: int,
        epochs: int,
        warmup: bool = True,
        warmup_epochs: int = 1,
        warmup_factor: float = 1e-3
):
    """
    Polynomial LR decay with optional linear warmup.
    """

    if num_step <= 0:

        raise ValueError(
            "num_step must be greater than zero."
        )

    if epochs <= 0:

        raise ValueError(
            "epochs must be greater than zero."
        )

    if warmup:

        if warmup_epochs < 0:

            raise ValueError(
                "warmup_epochs cannot be negative."
            )

    else:

        warmup_epochs = 0

    total_steps = (
        epochs * num_step
    )

    warmup_steps = (
        warmup_epochs * num_step
    )

    decay_steps = max(
        total_steps - warmup_steps,
        1
    )


    def lr_lambda(step):

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        if (
            warmup
            and warmup_steps > 0
            and step < warmup_steps
        ):

            alpha = (
                float(step)
                / float(warmup_steps)
            )

            return (
                warmup_factor * (1.0 - alpha)
                + alpha
            )

        # ----------------------------------------------------
        # Polynomial decay
        # ----------------------------------------------------

        progress = (

            float(
                step - warmup_steps
            )

            / float(decay_steps)
        )

        # Prevent negative LR when scheduler receives
        # one or more extra steps.
        progress = min(
            max(progress, 0.0),
            1.0
        )

        return (
            1.0 - progress
        ) ** 0.9


    return torch.optim.lr_scheduler.LambdaLR(

        optimizer,

        lr_lambda=lr_lambda
    )