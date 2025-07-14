"""Script to generate train/val/test splits for a 3D vision dataset.
Specifically adapted for a dataset with RGB images, point clouds, segmentation masks, and 3D bounding boxes.

Creates the following structure:
- dataset_root/
    - train/    
    - val/
    - test/
    - splits.json
    - train.txt
    - val.txt
    - test.txt

Each split folder contains subfolders for each sample, with the following files:
    - rgb.jpg: RGB image
    - pc.npy: Point cloud data
    - mask.npy: Segmentation mask
    - bbox3d.npy: 3D bounding boxes
Also generates metadata files for easy loading. Verifies dataset integrity before processing.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def discover_samples(dataset_root: str) -> List[str]:
    """
    Discover all sample folders in the dataset.

    Args:
        dataset_root: Path to the dataset root directory

    Returns:
        List of sample folder names
    """
    dataset_path = Path(dataset_root)
    samples = []

    for folder in dataset_path.iterdir():
        if folder.is_dir():
            # Check if folder contains all required files
            required_files = ["bbox3d.npy", "mask.npy", "pc.npy", "rgb.jpg"]
            if all((folder / file).exists() for file in required_files):
                samples.append(folder.name)

    print(f"Found {len(samples)} valid samples")
    return samples


def create_train_val_split(
    samples: List[str],
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits.

    Args:
        samples: List of sample IDs
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # First split: separate test set
    train_val_samples, test_samples = train_test_split(
        samples, test_size=test_ratio, random_state=random_seed
    )

    # Second split: separate train and val
    adjusted_val_ratio = val_ratio / (1 - test_ratio)
    train_samples, val_samples = train_test_split(
        train_val_samples, test_size=adjusted_val_ratio, random_state=random_seed
    )

    print(f"Train: {len(train_samples)} samples")
    print(f"Val: {len(val_samples)} samples")
    print(f"Test: {len(test_samples)} samples")

    return train_samples, val_samples, test_samples


def create_split_structure(
    dataset_root: str,
    train_samples: List[str],
    val_samples: List[str],
    test_samples: List[str],
    output_dir: str = "processed_dataset",
):
    """
    Create organized directory structure with train/val/test splits.

    Args:
        dataset_root: Original dataset path
        train_samples: List of training sample IDs
        val_samples: List of validation sample IDs
        test_samples: List of test sample IDs
        output_dir: Output directory name
    """
    dataset_path = Path(dataset_root)
    output_path = Path(output_dir)

    # Create output structure
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Copy files to appropriate splits
    splits = {"train": train_samples, "val": val_samples, "test": test_samples}

    for split_name, sample_list in splits.items():
        for sample_id in sample_list:
            src_dir = dataset_path / sample_id
            dst_dir = output_path / split_name / sample_id

            # Copy entire sample directory
            shutil.copytree(src_dir, dst_dir)

        print(f"Copied {len(sample_list)} samples to {split_name}/")


def create_metadata_files(
    train_samples: List[str],
    val_samples: List[str],
    test_samples: List[str],
    output_dir: str = "processed_dataset",
):
    """
    Create metadata files for easy loading.

    Args:
        train_samples: List of training sample IDs
        val_samples: List of validation sample IDs
        test_samples: List of test sample IDs
        output_dir: Output directory name
    """
    output_path = Path(output_dir)

    # Create split metadata
    metadata = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "total_samples": len(train_samples) + len(val_samples) + len(test_samples),
    }

    # Save metadata
    with open(output_path / "splits.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create individual split files
    for split_name, sample_list in metadata.items():
        if split_name != "total_samples":
            with open(output_path / f"{split_name}.txt", "w") as f:
                f.write("\n".join(sample_list))

    print(f"Created metadata files in {output_dir}/")


def verify_dataset_integrity(dataset_root: str) -> Dict[str, int]:
    """
    Verify dataset integrity and get statistics.

    Args:
        dataset_root: Path to dataset root

    Returns:
        Dictionary with dataset statistics
    """
    dataset_path = Path(dataset_root)
    stats = {"total_folders": 0, "valid_samples": 0, "missing_files": 0, "file_stats": {}}

    required_files = ["bbox3d.npy", "mask.npy", "pc.npy", "rgb.jpg"]

    for folder in dataset_path.iterdir():
        if folder.is_dir():
            stats["total_folders"] += 1

            missing = []
            for file in required_files:
                if not (folder / file).exists():
                    missing.append(file)

            if missing:
                stats["missing_files"] += 1
                print(f"Sample {folder.name} missing: {missing}")
            else:
                stats["valid_samples"] += 1

    return stats


def main():
    # Configuration
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits for a 3D vision dataset."
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="processed_dataset", help="Output directory name"
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument(
        "--random_seed", type=int, default=12345, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    DATASET_ROOT = args.dataset_root
    OUTPUT_DIR = args.output_dir
    VAL_RATIO = args.val_ratio
    TEST_RATIO = args.test_ratio
    RANDOM_SEED = args.random_seed

    print("Starting dataset preprocessing...")

    # Verify dataset integrity
    print("Verifying dataset integrity...")
    stats = verify_dataset_integrity(DATASET_ROOT)
    print(f"Total folders: {stats['total_folders']}")
    print(f"Valid samples: {stats['valid_samples']}")
    print(f"Invalid samples: {stats['missing_files']}")

    # Discover all samples
    print("Discovering samples...")
    samples = discover_samples(DATASET_ROOT)

    if not samples:
        print("No valid samples found!")
        return

    # Create train/val/test splits
    print("Creating train/val/test splits...")
    train_samples, val_samples, test_samples = create_train_val_split(
        samples, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, random_seed=RANDOM_SEED
    )

    # Create organized directory structure
    print("Creating organized directory structure...")
    create_split_structure(DATASET_ROOT, train_samples, val_samples, test_samples, OUTPUT_DIR)

    # Create metadata files
    print("Creating metadata files...")
    create_metadata_files(train_samples, val_samples, test_samples, OUTPUT_DIR)

    print(f"Dataset preprocessing complete!")
    print(f"Processed dataset saved to: {OUTPUT_DIR}/")
    print(f"- train/: {len(train_samples)} samples")
    print(f"- val/: {len(val_samples)} samples")
    print(f"- test/: {len(test_samples)} samples")


if __name__ == "__main__":
    main()
