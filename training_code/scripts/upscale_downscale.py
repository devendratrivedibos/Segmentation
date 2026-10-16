import os
import cv2


def upscale_images(
    input_folder,
    output_folder=None,
    scale=2,
    interpolation=cv2.INTER_NEAREST,
    overwrite=True,
):
    """
    Upscale all images in a folder.

    Args:
        input_folder (str): Folder containing images.
        output_folder (str): Folder to save upscaled images.
                             Ignored if overwrite=True.
        scale (int): Upscaling factor.
        interpolation: OpenCV interpolation method.
        overwrite (bool): If True, replaces original images.
    """

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    if not overwrite:
        if output_folder is None:
            raise ValueError("Please provide an output folder when overwrite=False.")
        os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

    print(f"Found {len(files)} images.")

    for fname in files:
        in_path = os.path.join(input_folder, fname)

        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Skipping unreadable image: {fname}")
            continue

        h, w = img.shape[:2]

        # new_w = w * scale
        # new_h = h * scale

        new_w=w//scale
        new_h=h//scale

        upscaled = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=interpolation,
        )

        if overwrite:
            out_path = in_path
        else:
            out_path = os.path.join(output_folder, fname)

        cv2.imwrite(out_path, upscaled)

    print("Done!")


if __name__ == "__main__":
    input_folder = r"C:\Users\Admin\Downloads\Dataset\IMAGES"
    output_folder = r"R:\Sidhesh\Distress\Asphalt\L_AND_T_Upscaled\FOLDER 6\JPEGImages_DS"

    upscale_images(
        input_folder=input_folder,
        output_folder=output_folder,
        scale=2,                 
        interpolation=cv2.INTER_NEAREST,
        overwrite=True       
    )