import os
import sys
import torch
import onnx

from ultralytics import YOLO

# ------------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, ".."))

from models.unet.UnetPP import UNetPP


# ==================================================================
# UNET++ ONNX WRAPPER
# ==================================================================

class UNetPPONNXWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        output = self.model(x)

        if isinstance(output, dict):
            return output["out"]

        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (list, tuple)):
            return output[0]

        raise RuntimeError(
            f"Unsupported UNet++ output type: {type(output)}"
        )


# ==================================================================
# COMMON ONNX EXPORTER
# ==================================================================

class ONNXExporter:

    def __init__(
            self,
            model_type,
            model_path,
            onnx_path=None,

            # Input size H, W
            input_size=(1024, 419),

            # YOLO settings
            yolo_opset=12,
            yolo_simplify=False,
            yolo_dynamic=False,

            # UNet++ settings
            num_classes=None,
            base_channels=64,
            deep_supervision=True,
            unet_opset=18,
            unet_dynamic=True,

            # Class names
            class_names=None,

            device=None
    ):

        self.model_type = model_type.lower()
        self.model_path = model_path
        self.onnx_path = onnx_path

        self.height, self.width = input_size

        self.yolo_opset = yolo_opset
        self.yolo_simplify = yolo_simplify
        self.yolo_dynamic = yolo_dynamic

        self.num_classes = num_classes
        self.base_channels = base_channels
        self.deep_supervision = deep_supervision
        self.unet_opset = unet_opset
        self.unet_dynamic = unet_dynamic

        self.class_names = class_names

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ==============================================================
    # MAIN EXPORT
    # ==============================================================

    def export(self):

        if self.model_type == "yolo":
            self._export_yolo()

        elif self.model_type in ["unet", "unetpp", "unet++"]:
            self._export_unet()

        else:
            raise ValueError(
                "Invalid model_type.\n"
                "Use 'yolo' or 'unetpp'."
            )

    # ==============================================================
    # YOLO EXPORT
    # ==============================================================

    def _export_yolo(self):

        print("\n" + "=" * 70)
        print("YOLO → ONNX EXPORT")
        print("=" * 70)

        # ----------------------------------------------------------
        # Check model
        # ----------------------------------------------------------

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"YOLO model not found:\n{self.model_path}"
            )

        print(f"\nYOLO model:")
        print(self.model_path)

        # ----------------------------------------------------------
        # Load YOLO
        # ----------------------------------------------------------

        print("\nLoading YOLO model...")

        model = YOLO(self.model_path)

        # ----------------------------------------------------------
        # Display model classes
        # ----------------------------------------------------------

        if hasattr(model, "names"):

            print("\nYOLO Classes:")

            for idx, name in model.names.items():
                print(f"  {idx}: {name}")

        # ----------------------------------------------------------
        # Export
        # ----------------------------------------------------------

        print("\nExporting YOLO to ONNX...")

        export_kwargs = {
            "format": "onnx",
            "imgsz": (self.height, self.width),
            "opset": self.yolo_opset,
            "simplify": self.yolo_simplify,
            "dynamic": self.yolo_dynamic,
        }

        # If user supplied an ONNX path, Ultralytics normally
        # derives the output path from the .pt location.
        #
        # Ultralytics export returns the actual exported path.
        exported_path = model.export(**export_kwargs)

        exported_path = str(exported_path)

        print("\nYOLO ONNX export successful.")

        print(f"\nONNX file:")
        print(exported_path)

        # ----------------------------------------------------------
        # Save classes.txt
        # ----------------------------------------------------------

        if hasattr(model, "names"):

            classes_path = os.path.join(
                os.path.dirname(exported_path),
                "classes.txt"
            )

            with open(
                    classes_path,
                    "w",
                    encoding="utf-8"
            ) as f:

                for idx in sorted(model.names.keys()):
                    f.write(
                        f"{model.names[idx]}\n"
                    )

            print("\nClasses saved:")
            print(classes_path)

        # ----------------------------------------------------------
        # Finish
        # ----------------------------------------------------------

        print("\n" + "=" * 70)
        print("YOLO EXPORT COMPLETE")
        print("=" * 70)

    # ==============================================================
    # UNET++ EXPORT
    # ==============================================================

    def _export_unet(self):

        print("\n" + "=" * 70)
        print("UNET++ → ONNX EXPORT")
        print("=" * 70)

        # ----------------------------------------------------------
        # Check checkpoint
        # ----------------------------------------------------------

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"UNet++ checkpoint not found:\n{self.model_path}"
            )

        print(f"\nCheckpoint:")
        print(self.model_path)

        print(f"\nInput size:")
        print(
            f"Height = {self.height}"
        )
        print(
            f"Width  = {self.width}"
        )

        print(f"\nDevice:")
        print(self.device)

        # ----------------------------------------------------------
        # Output path
        # ----------------------------------------------------------

        if self.onnx_path is None:
            self.onnx_path = os.path.splitext(
                self.model_path
            )[0] + ".onnx"

        output_dir = os.path.dirname(
            self.onnx_path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True
            )

        print("\nONNX output:")
        print(self.onnx_path)

        # ----------------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------------

        print("\nLoading checkpoint...")

        ckpt = torch.load(
            self.model_path,
            map_location="cpu"
        )

        # ----------------------------------------------------------
        # Extract state dict
        # ----------------------------------------------------------

        if isinstance(ckpt, dict):

            if "model" in ckpt:
                state_dict = ckpt["model"]

            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]

            else:
                state_dict = ckpt

        else:

            state_dict = ckpt

        # ----------------------------------------------------------
        # Remove module. prefix
        # ----------------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):
                key = key[len("module."):]

            cleaned_state_dict[key] = value

        state_dict = cleaned_state_dict

        # ----------------------------------------------------------
        # Number of classes
        # ----------------------------------------------------------

        if self.num_classes is None:

            num_classes = self._detect_unet_num_classes(
                state_dict
            )

        else:

            num_classes = self.num_classes

        print(
            f"\nUNet++ classes = {num_classes}"
        )

        print(
            f"Base channels = {self.base_channels}"
        )

        print(
            f"Deep supervision = {self.deep_supervision}"
        )

        # ----------------------------------------------------------
        # Create UNet++
        # ----------------------------------------------------------

        print("\nCreating UNet++ model...")

        model = UNetPP(
            in_channels=3,
            num_classes=num_classes,
            deep_supervision=self.deep_supervision,
            base_channels=self.base_channels
        )

        # ----------------------------------------------------------
        # Load weights
        # ----------------------------------------------------------

        print("\nLoading weights...")

        model.load_state_dict(
            state_dict,
            strict=True
        )

        print("Weights loaded successfully.")

        # ----------------------------------------------------------
        # Device
        # ----------------------------------------------------------

        model = model.to(self.device)
        model.eval()

        # ----------------------------------------------------------
        # Wrapper
        # ----------------------------------------------------------

        model_wrapper = UNetPPONNXWrapper(
            model
        )

        model_wrapper = model_wrapper.to(
            self.device
        )

        model_wrapper.eval()

        # ----------------------------------------------------------
        # Dummy input
        # ----------------------------------------------------------

        dummy_input = torch.randn(
            1,
            3,
            self.height,
            self.width,
            device=self.device
        )

        print(
            "\nInput shape:"
        )

        print(
            tuple(dummy_input.shape)
        )

        # ----------------------------------------------------------
        # Test PyTorch output
        # ----------------------------------------------------------

        print(
            "\nTesting PyTorch model..."
        )

        with torch.no_grad():

            pytorch_output = model_wrapper(
                dummy_input
            )

        print(
            "PyTorch output:"
        )

        print(
            tuple(pytorch_output.shape)
        )

        expected_shape = (
            1,
            num_classes,
            self.height,
            self.width
        )

        if tuple(pytorch_output.shape) != expected_shape:
            raise RuntimeError(
                "\nUnexpected PyTorch output shape.\n"
                f"Expected: {expected_shape}\n"
                f"Actual:   {tuple(pytorch_output.shape)}"
            )

        # ----------------------------------------------------------
        # Export ONNX
        # ----------------------------------------------------------

        print("\nExporting UNet++ to ONNX...")

        dynamic_axes = None

        if self.unet_dynamic:
            dynamic_axes = {
                "input": {
                    0: "batch_size"
                },
                "output": {
                    0: "batch_size"
                }
            }

        torch.onnx.export(

            model_wrapper,

            dummy_input,

            self.onnx_path,

            export_params=True,

            opset_version=self.unet_opset,

            do_constant_folding=True,

            input_names=[
                "input"
            ],

            output_names=[
                "output"
            ],

            dynamic_axes=dynamic_axes,

            external_data=True
        )

        print(
            "\nUNet++ ONNX export successful."
        )

        # ----------------------------------------------------------
        # External data
        # ----------------------------------------------------------

        external_data_path = (
                self.onnx_path + ".data"
        )

        if os.path.isfile(
                external_data_path
        ):
            print(
                "\nExternal weights:"
            )

            print(
                external_data_path
            )

            print(
                "\nKeep .onnx and .onnx.data "
                "in the same folder."
            )

        # ----------------------------------------------------------
        # Validate ONNX
        # ----------------------------------------------------------

        print(
            "\nValidating ONNX model..."
        )

        onnx_model = onnx.load(
            self.onnx_path,
            load_external_data=True
        )

        onnx.checker.check_model(
            onnx_model
        )

        print(
            "ONNX validation successful."
        )

        # ----------------------------------------------------------
        # Save classes
        # ----------------------------------------------------------

        if self.class_names:

            if len(self.class_names) != num_classes:
                print(
                    "\nWARNING:"
                )

                print(
                    f"class_names = "
                    f"{len(self.class_names)}"
                )

                print(
                    f"model classes = "
                    f"{num_classes}"
                )

            self._save_class_names(
                self.class_names
            )

        # ----------------------------------------------------------
        # Final information
        # ----------------------------------------------------------

        print("\n" + "=" * 70)
        print("UNET++ EXPORT COMPLETE")
        print("=" * 70)

        print(
            "\nInput:"
        )

        print(
            f"[batch_size, 3, "
            f"{self.height}, {self.width}]"
        )

        print(
            "\nOutput:"
        )

        print(
            f"[batch_size, {num_classes}, "
            f"{self.height}, {self.width}]"
        )

        print(
            f"\nONNX:"
        )

        print(
            self.onnx_path
        )

    # ==============================================================
    # DETECT UNET CLASSES
    # ==============================================================

    def _detect_unet_num_classes(
            self,
            state_dict
    ):

        for key, value in state_dict.items():

            key_lower = key.lower()

            if any(
                    name in key_lower
                    for name in [
                        "final",
                        "classifier",
                        "out_conv",
                        "final_conv"
                    ]
            ):

                if (
                        isinstance(value, torch.Tensor)
                        and len(value.shape) == 4
                ):
                    return value.shape[0]

        raise RuntimeError(
            "Could not detect UNet++ number of classes."
        )

    # ==============================================================
    # SAVE CLASSES
    # ==============================================================

    def _save_class_names(
            self,
            class_names
    ):

        classes_path = os.path.join(
            os.path.dirname(
                self.onnx_path
            ),
            "classes.txt"
        )

        with open(
                classes_path,
                "w",
                encoding="utf-8"
        ) as f:
            f.write(
                "\n".join(class_names)
            )

        print(
            f"\nclasses.txt saved:"
        )

        print(
            classes_path
        )


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":
    # ==============================================================
    # DEVICE
    # ==============================================================

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    # ==============================================================
    # --------------------------------------------------------------
    # OPTION 1: YOLO
    # --------------------------------------------------------------
    # ==============================================================

    yolo_exporter = ONNXExporter(

        model_type="yolo",

        model_path=(
            r"C:\Users\Admin\Code\survey-analytic"
            r"\ai_model\guide_rcc_metal_median.pt"
        ),

        input_size=(
            1024,
            419
        ),

        yolo_opset=12,

        yolo_simplify=False,

        # Fixed H/W is useful for your C3D pipeline
        yolo_dynamic=False,

        device=device
    )

    # Uncomment to export YOLO
    # yolo_exporter.export()

    # ==============================================================
    # --------------------------------------------------------------
    # OPTION 2: UNET++
    # --------------------------------------------------------------
    # ==============================================================

    unet_classes = [

        "Background",

        "Alligator",

        "Longitudinal Crack",

        "Transverse Crack",

        "Pothole",

        "Patches",

        "Multiple Crack",

        "Spalling",

        "Corner Break",

        "Sealed Joint - T",

        "Sealed Joint - L",

        "Punchout",

        "Popout",

        "White",

        "Unclassified"
    ]

    unet_exporter = ONNXExporter(

        model_type="unetpp",

        model_path=(
            r"D:\Devendra_Files"
            r"\segmentation_training"
            r"\weights\25Aug"
            r"\25Aug_best_epoch92_dice0.599.pth"
        ),

        onnx_path=(
            r"D:\Devendra_Files"
            r"\segmentation_training"
            r"\weights\25Aug"
            r"\25Aug_best_epoch92_dice.onnx"
        ),

        input_size=(
            1024,
            419
        ),

        num_classes=7,

        base_channels=64,

        deep_supervision=True,

        unet_opset=18,

        unet_dynamic=True,

        class_names=unet_classes,

        device=device
    )

    # Uncomment to export UNet++
    unet_exporter.export()