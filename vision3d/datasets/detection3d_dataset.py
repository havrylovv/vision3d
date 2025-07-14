"""Core dataset for 3D detection tasks with RGB images, point clouds, masks, and 3D bounding boxes."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vision3d.utils.bbox_converter import corners_to_obb, reorder_corners_pca
from vision3d.utils.logging import configure_logger
from vision3d.utils.registry import DATASETS

logger = configure_logger(__name__.split(".")[-1], logging.INFO)


@DATASETS.register()
class Detection3DDataset(Dataset):
    """
    Dataset for 3D Detection with RGB images, point clouds, masks, and 3D bounding boxes.

    Each sample contains:
    - rgb.jpg: RGB image
    - pc.npy: Point cloud data
    - mask.npy: Segmentation mask
    - bbox3d.npy: 3D bounding boxes
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        transform: Optional[Dict[str, transforms.Compose]] = dict(),
        return_sample_id: bool = False,
        fix_bbox_corners_order: bool = True,
        bbox_corners_to_oob: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            dataset_root: Path to processed dataset root
            split: 'train', 'val', or 'test'
            transform: dictionary of transformations to apply to RGB images, point clouds, etc
                    Supported keys: 'rgb', 'pc', 'mask', 'bbox3d'
            return_sample_id: Whether to return sample IDs
            fix_bbox_corners_order: Whether to fix the order of bounding box corners to enforce a consistent orientation
            bbox_corners_to_oob: Whether to convert bounding box corners to oriented bounding boxes (OOB)
                                Map: [8,3] -> [10] of [size, center, quaternion]
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.return_sample_id = return_sample_id
        self.fix_bbox_corners_order = fix_bbox_corners_order
        self.bbox_corners_to_oob = bbox_corners_to_oob

        assert self.split in [
            "train",
            "val",
            "test",
        ], f"Split must be one of ['train', 'val', 'test'], but got '{self.split}'"
        assert self.transform.keys() <= {
            "rgb",
            "pc",
            "mask",
            "bbox3d",
        }, "Transform keys must be a subset of {'rgb', 'pc', 'mask', 'bbox3d'}"

        # Load sample list
        split_file = self.dataset_root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            self.sample_ids = [line.strip() for line in f.readlines()]

        logger.info(f"Loaded {len(self.sample_ids)} samples for {split} split")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Returns:
            inputs: Dictionary with 'rgb' (Tensor), 'pc' (Tensor), and optionally 'sample_id' (str)
            targets: Dictionary with 'mask' (Tensor), 'bbox3d' (Tensor), 'labels' (Tensor), and optionally 'sample_id' (str)
        Args:
            idx: Sample index
        """
        sample_id = self.sample_ids[idx]
        sample_path = self.dataset_root / self.split / sample_id

        # Load RGB image
        rgb_path = sample_path / "rgb.jpg"
        rgb = Image.open(rgb_path).convert("RGB")

        # Load point cloud
        pc_path = sample_path / "pc.npy"
        pc = np.load(pc_path).astype(np.float32)
        pc = torch.from_numpy(pc).float()

        # Load segmentation mask
        mask_path = sample_path / "mask.npy"
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask)

        # Load 3D bounding boxes
        bbox3d_path = sample_path / "bbox3d.npy"
        bbox3d = np.load(bbox3d_path).astype(np.float32)
        bbox3d = torch.from_numpy(bbox3d).float()

        # Apply transformations
        if "rgb" in self.transform:
            rgb = self.transform["rgb"](rgb)
        else:
            rgb = transforms.ToTensor()(rgb)
        if "pc" in self.transform:
            pc = self.transform["pc"](pc)
        if "mask" in self.transform:
            mask = self.transform["mask"](mask)
        if "bbox3d" in self.transform:
            bbox3d = self.transform["bbox3d"](bbox3d)

        if self.fix_bbox_corners_order:
            # Reorder bounding box corners using PCA to enforce a consistent orientation
            bbox3d = reorder_corners_pca(bbox3d)

        if self.bbox_corners_to_oob:
            # Convert bounding box corners to oriented bounding boxes (OOB)
            bbox3d = corners_to_obb(bbox3d)

        # Prepare output
        inputs = {
            "rgb": rgb.to(dtype=torch.float32),
            "pc": pc.to(dtype=torch.float32),
        }

        targets = {
            "mask": mask,
            "bbox3d": bbox3d.to(dtype=torch.float32),
            "labels": torch.ones(
                bbox3d.shape[0], dtype=torch.long
            ),  # class 0 - background, class 1 - object
        }

        if self.return_sample_id:
            inputs["sample_id"] = sample_id
            targets["sample_id"] = sample_id

        return inputs, targets

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a sample without loading the data.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample information
        """
        sample_id = self.sample_ids[idx]
        sample_path = self.dataset_root / self.split / sample_id

        # Get file sizes
        info = {
            "sample_id": sample_id,
            "rgb_path": str(sample_path / "rgb.jpg"),
            "pc_path": str(sample_path / "pc.npy"),
            "mask_path": str(sample_path / "mask.npy"),
            "bbox3d_path": str(sample_path / "bbox3d.npy"),
        }

        # Get point cloud shape
        pc = np.load(sample_path / "pc.npy")
        info["num_points"] = pc.shape[0]

        # Get bounding boxes count
        bbox3d = np.load(sample_path / "bbox3d.npy")
        info["num_boxes"] = bbox3d.shape[0] if bbox3d.ndim > 1 else 1

        return info


def collate_fn(
    batch: List[Tuple[Dict[str, Any], Dict[str, Any]]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Collate function for batching samples with possibly variable-sized tensors
    like masks or 3D bounding boxes.

    Args:
        batch: A list of (inputs, targets) tuples.

    Returns:
        A tuple (batched_inputs, batched_targets)
    """
    # Unzip list of tuples
    inputs_list, targets_list = zip(*batch)

    # Collate inputs
    batched_inputs = {
        "rgb": torch.stack([item["rgb"] for item in inputs_list], dim=0),
        "pc": torch.stack([item["pc"] for item in inputs_list], dim=0),
    }

    # Optional sample_id
    if "sample_id" in inputs_list[0]:
        batched_inputs["sample_id"] = [item["sample_id"] for item in inputs_list]

    # Collate targets
    batched_targets = {
        "mask": [target["mask"] for target in targets_list],  # list of [num_obj, H, W]
        "bbox3d": [
            target["bbox3d"] for target in targets_list
        ],  # list of [num_obj, 8, 3] or [num_obj, 10] (OOB)
        "labels": [target["labels"] for target in targets_list],  # list of [num_obj]
    }

    if "sample_id" in targets_list[0]:
        batched_targets["sample_id"] = [target["sample_id"] for target in targets_list]

    return batched_inputs, batched_targets
