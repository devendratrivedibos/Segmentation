r"""Upload a train/test dataset folder to ClearML.

Example:
    python training_code/tools/upload_dataset_clearml.py ^
        --data-path C:\path\to\dataset_root ^
        --clearml-project "Asphalt Segmentation" ^
        --dataset-name "asphalt-cracks-v1"
"""
import argparse
from pathlib import Path
from clearml import Dataset

REQUIRED_SPLITS = ("TRAIN", "VAL")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a dataset folder to ClearML Dataset.")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Root dataset folder. It must contain train and test subfolders.",
    )
    parser.add_argument(
        "--clearml-project",
        required=True,
        help="ClearML project name where the dataset will be created.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="ClearML dataset name.",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Optional dataset version, for example 1.0.0.",
    )
    parser.add_argument(
        "--output-uri",
        default=None,
        help="Optional upload destination. If omitted, ClearML uses the configured file server.",
    )
    parser.add_argument(
        "--description",
        default="Segmentation dataset with train and test splits.",
        help="Dataset description shown in ClearML.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional ClearML dataset tags, separated by spaces.",
    )
    return parser.parse_args()


def validate_dataset_root(data_path):
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist or is not a folder: {data_path}")

    missing_splits = [split for split in REQUIRED_SPLITS if not (data_path / split).is_dir()]
    if missing_splits:
        missing = ", ".join(missing_splits)
        raise FileNotFoundError(
            f"Dataset root must contain these missing split folders: {missing}"
        )

    return data_path


def main():
    args = parse_args()
    data_path = validate_dataset_root(args.data_path)

    dataset = Dataset.create(
        dataset_project=args.clearml_project,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        dataset_tags=args.tags,
        output_uri=args.output_uri,
        description=args.description,
    )

    print(f"Adding files from: {data_path}")
    dataset.add_files(path=str(data_path))

    print("Uploading dataset files to ClearML...")
    dataset.upload(output_url=args.output_uri)

    dataset.finalize()
    print("Dataset uploaded and finalized.")
    print(f"ClearML dataset id: {dataset.id}")


if __name__ == "__main__":
    main()
