"""
Batch semantic segmentation inference using UNet++ ONNX Runtime.

Input:
    Images: 1024 x 419
    NCHW format

Output:
    Color segmentation masks

ONNX model:
    .onnx
    .onnx.data  <-- automatically loaded by ONNX Runtime
"""

import os
import cv2
import numpy as np
import onnxruntime as ort

from tqdm import tqdm


# =====================================================================
# CONFIG
# =====================================================================

ONNX_MODEL_PATH = (
    r"D:\Devendra_Files\segmentation_training\weights\25Aug\25Aug_best_epoch92_dice.onnx"
)

IMAGES_ROOT = (
    r"Z:\Devendra\ASPHALT\Asphalt_GoldenSet_Test"
    r"\IMAGES"
)

PREDICTION_SAVE_PATH = (
    r"Z:\Devendra\ASPHALT\Asphalt_GoldenSet_Test"
    r"\PRED_MASKS_ONNX"
)

BATCH_SIZE = 4

INPUT_HEIGHT = 1024
INPUT_WIDTH = 419


# =====================================================================
# CLASS / COLOR MAP
# =====================================================================

COLOR_MAP = {
    (0, 0, 0): (
        0,
        "Background"
    ),

    (255, 0, 0): (
        1,
        "Alligator"
    ),

    (0, 0, 255): (
        2,
        "Transverse Crack"
    ),

    (0, 255, 0): (
        3,
        "Longitudinal Crack"
    ),

    (139, 69, 19): (
        4,
        "Pothole"
    ),

    (255, 165, 0): (
        5,
        "Patches"
    ),
}


# =====================================================================
# NORMALIZATION
# =====================================================================

MEAN = np.array(
    [0.4787, 0.4787, 0.4787],
    dtype=np.float32
)

STD = np.array(
    [0.1472, 0.1472, 0.1472],
    dtype=np.float32
)


# =====================================================================
# CREATE ONNX SESSION
# =====================================================================

def create_onnx_session(model_path):

    print("=" * 70)
    print("LOADING ONNX MODEL")
    print("=" * 70)

    if not os.path.isfile(model_path):

        raise FileNotFoundError(
            f"ONNX model not found:\n{model_path}"
        )

    # ---------------------------------------------------------------
    # Check external data file
    # ---------------------------------------------------------------

    external_data_path = model_path + ".data"

    if os.path.isfile(external_data_path):

        print("\nONNX external weights found:")
        print(external_data_path)

        print(
            "\nONNX Runtime will automatically load "
            "the .data file."
        )

    else:

        print(
            "\nNo .onnx.data file found."
        )

        print(
            "If your ONNX model was exported using "
            "external data, make sure the .data file "
            "is beside the .onnx file."
        )

    # ---------------------------------------------------------------
    # Available providers
    # ---------------------------------------------------------------

    available_providers = ort.get_available_providers()

    print("\nAvailable ONNX Runtime providers:")

    for provider in available_providers:
        print("  ", provider)

    # ---------------------------------------------------------------
    # Select CUDA if available
    # ---------------------------------------------------------------

    if "CUDAExecutionProvider" in available_providers:

        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        print(
            "\nUsing CUDAExecutionProvider"
        )

    else:

        providers = [
            "CPUExecutionProvider"
        ]

        print(
            "\nCUDAExecutionProvider not available."
        )

        print(
            "Using CPUExecutionProvider"
        )

    # ---------------------------------------------------------------
    # Create session
    # ---------------------------------------------------------------

    session = ort.InferenceSession(
        model_path,
        providers=providers
    )

    print("\nONNX model loaded successfully.")

    return session


# =====================================================================
# PRINT MODEL INFORMATION
# =====================================================================

def print_model_info(session):

    print("\n" + "=" * 70)
    print("ONNX MODEL INFORMATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Input
    # ---------------------------------------------------------------

    inputs = session.get_inputs()

    for inp in inputs:

        print("\nInput:")
        print(
            f"  Name : {inp.name}"
        )

        print(
            f"  Shape: {inp.shape}"
        )

        print(
            f"  Type : {inp.type}"
        )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    outputs = session.get_outputs()

    for out in outputs:

        print("\nOutput:")
        print(
            f"  Name : {out.name}"
        )

        print(
            f"  Shape: {out.shape}"
        )

        print(
            f"  Type : {out.type}"
        )


# =====================================================================
# PREPROCESS IMAGE
# =====================================================================

def preprocess_image(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    # ---------------------------------------------------------------
    # BGR -> RGB
    # ---------------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ---------------------------------------------------------------
    # Resize
    #
    # Your PyTorch script assumes images are already 1024 x 419
    # because A.Resize() is commented out.
    #
    # ONNX model expects 1024 x 419.
    # ---------------------------------------------------------------

    if (
        image.shape[0] != INPUT_HEIGHT
        or
        image.shape[1] != INPUT_WIDTH
    ):

        image = cv2.resize(
            image,
            (
                INPUT_WIDTH,
                INPUT_HEIGHT
            ),
            interpolation=cv2.INTER_LINEAR
        )

    # ---------------------------------------------------------------
    # uint8 -> float32
    # ---------------------------------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0

    # ---------------------------------------------------------------
    # Normalize
    #
    # Same as:
    #
    # A.Normalize(
    #     mean=(0.4787,...),
    #     std=(0.1472,...)
    # )
    # ---------------------------------------------------------------

    image = (
        image - MEAN
    ) / STD

    # ---------------------------------------------------------------
    # HWC -> CHW
    # ---------------------------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # ---------------------------------------------------------------
    # Ensure contiguous memory
    # ---------------------------------------------------------------

    image = np.ascontiguousarray(
        image,
        dtype=np.float32
    )

    return image


# =====================================================================
# COLORIZE PREDICTION
# =====================================================================

def colorize_prediction(prediction):

    """
    Convert class index map:

        H x W

    into RGB image:

        H x W x 3
    """

    color_mask = np.zeros(
        (
            prediction.shape[0],
            prediction.shape[1],
            3
        ),
        dtype=np.uint8
    )

    for rgb, (
        class_id,
        class_name
    ) in COLOR_MAP.items():

        color_mask[
            prediction == class_id
        ] = rgb

    return color_mask


# =====================================================================
# REMOVE SMALL COMPONENTS
# =====================================================================

def remove_small_components_multiclass(
    mask,
    min_area=50
):

    """
    Remove small connected components for each
    foreground class.

    Class 0 = Background
    """

    cleaned = np.zeros_like(
        mask,
        dtype=mask.dtype
    )

    # ---------------------------------------------------------------
    # Process each class
    # ---------------------------------------------------------------

    for cls in np.unique(mask):

        # Background
        if cls == 0:
            continue

        class_mask = (
            mask == cls
        ).astype(np.uint8)

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                class_mask,
                connectivity=8
            )
        )

        # -----------------------------------------------------------
        # Keep components larger than min_area
        # -----------------------------------------------------------

        for i in range(
            1,
            num_labels
        ):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if area >= min_area:

                cleaned[
                    labels == i
                ] = cls

    return cleaned


# =====================================================================
# PROCESS BATCH
# =====================================================================

def process_batch(
    session,
    batch_images,
    batch_names,
    input_name,
    output_name,
    prediction_save_path
):

    # ---------------------------------------------------------------
    # Stack:
    #
    # [B, 3, 1024, 419]
    # ---------------------------------------------------------------

    batch_tensor = np.stack(
        batch_images,
        axis=0
    ).astype(np.float32)

    # ---------------------------------------------------------------
    # ONNX inference
    # ---------------------------------------------------------------

    outputs = session.run(
        [output_name],
        {
            input_name: batch_tensor
        }
    )

    output = outputs[0]

    # ---------------------------------------------------------------
    # Expected:
    #
    # [B, Classes, H, W]
    #
    # Example:
    #
    # [4, 6, 1024, 419]
    # ---------------------------------------------------------------

    if output.ndim != 4:

        raise RuntimeError(
            f"Unexpected ONNX output shape: "
            f"{output.shape}"
        )

    # ---------------------------------------------------------------
    # Argmax
    #
    # Same as:
    #
    # outputs['out'].argmax(1)
    #
    # ---------------------------------------------------------------

    predictions = np.argmax(
        output,
        axis=1
    ).astype(np.uint8)

    # ---------------------------------------------------------------
    # Process individual images
    # ---------------------------------------------------------------

    for pred, fname in zip(
        predictions,
        batch_names
    ):

        # -----------------------------------------------------------
        # Remove small components
        # -----------------------------------------------------------

        pred = remove_small_components_multiclass(
            pred,
            min_area=50
        )

        # -----------------------------------------------------------
        # Colorize
        # -----------------------------------------------------------

        pred_color = colorize_prediction(
            pred
        )

        # -----------------------------------------------------------
        # Output filename
        # -----------------------------------------------------------

        base_name = os.path.splitext(
            fname
        )[0]

        save_path = os.path.join(
            prediction_save_path,
            base_name + ".png"
        )

        # -----------------------------------------------------------
        # RGB -> BGR before OpenCV save
        # -----------------------------------------------------------

        cv2.imwrite(
            save_path,
            cv2.cvtColor(
                pred_color,
                cv2.COLOR_RGB2BGR
            )
        )


# =====================================================================
# MAIN
# =====================================================================

def main(
    imgs_root,
    prediction_save_path,
    onnx_model_path,
    batch_size=4
):

    print("\n" + "=" * 70)
    print("UNET++ ONNX RUNTIME SEGMENTATION")
    print("=" * 70)

    print("\nImages:")
    print(imgs_root)

    print("\nOutput:")
    print(prediction_save_path)

    print("\nONNX:")
    print(onnx_model_path)

    print("\nBatch size:")
    print(batch_size)

    # ---------------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------------

    os.makedirs(
        prediction_save_path,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Create ONNX session
    # ---------------------------------------------------------------

    session = create_onnx_session(
        onnx_model_path
    )

    # ---------------------------------------------------------------
    # Print model info
    # ---------------------------------------------------------------

    print_model_info(
        session
    )

    # ---------------------------------------------------------------
    # Input / Output names
    # ---------------------------------------------------------------

    input_name = session.get_inputs()[0].name

    output_name = session.get_outputs()[0].name

    print("\nUsing input name:")
    print(input_name)

    print("\nUsing output name:")
    print(output_name)

    # ---------------------------------------------------------------
    # Collect images
    # ---------------------------------------------------------------

    images_list = [
        img
        for img in os.listdir(imgs_root)
        if img.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg"
            )
        )
    ]

    # ---------------------------------------------------------------
    # Keep deterministic order
    #
    # Your PyTorch code used shuffle().
    # If you want EXACTLY the same random behavior,
    # use random.shuffle(images_list).
    # ---------------------------------------------------------------

    images_list.sort()

    print(
        f"\nTotal images: "
        f"{len(images_list)}"
    )

    if len(images_list) == 0:

        print(
            "\nNo images found."
        )

        return

    # ---------------------------------------------------------------
    # Batch processing
    # ---------------------------------------------------------------

    with tqdm(
        total=len(images_list),
        desc="Processing images"
    ) as progress:

        for i in range(
            0,
            len(images_list),
            batch_size
        ):

            batch_files = images_list[
                i:i + batch_size
            ]

            batch_imgs = []
            batch_names = []

            # -------------------------------------------------------
            # Preprocess batch
            # -------------------------------------------------------

            for fname in batch_files:

                image_path = os.path.join(
                    imgs_root,
                    fname
                )

                try:

                    image = preprocess_image(
                        image_path
                    )

                    batch_imgs.append(
                        image
                    )

                    batch_names.append(
                        fname
                    )

                except Exception as e:

                    print(
                        f"\nERROR processing "
                        f"{fname}: {e}"
                    )

            # -------------------------------------------------------
            # Skip empty batch
            # -------------------------------------------------------

            if len(batch_imgs) == 0:

                continue

            # -------------------------------------------------------
            # Inference
            # -------------------------------------------------------

            process_batch(
                session=session,
                batch_images=batch_imgs,
                batch_names=batch_names,
                input_name=input_name,
                output_name=output_name,
                prediction_save_path=prediction_save_path
            )

            # -------------------------------------------------------
            # Progress
            # -------------------------------------------------------

            progress.update(
                len(batch_imgs)
            )

    print(
        "\n" + "=" * 70
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOutput masks saved to:\n"
        f"{prediction_save_path}"
    )


# =====================================================================
# RUN
# =====================================================================

if __name__ == "__main__":

    main(

        imgs_root=IMAGES_ROOT,

        prediction_save_path=(
            PREDICTION_SAVE_PATH
        ),

        onnx_model_path=(
            ONNX_MODEL_PATH
        ),

        batch_size=BATCH_SIZE
    )