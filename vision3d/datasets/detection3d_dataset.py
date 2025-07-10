import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import logging
from vision3d.utils.logger import configure_logger

logger = configure_logger(__name__, logging.INFO)

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
        split: str = 'train',
        transform: Optional[Dict[str, transforms.Compose]] = dict(),
        return_sample_id: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset_root: Path to processed dataset root
            split: 'train', 'val', or 'test'
            transform: dictionary of transformations to apply to RGB images, point clouds, etc
                    Supported keys: 'rgb', 'pc', 'mask', 'bbox3d'
            normalize_pc: Whether to normalize point cloud coordinates
            max_points: Maximum number of points to sample from point cloud
            return_sample_id: Whether to return sample IDs
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.return_sample_id = return_sample_id

        assert self.split in ['train', 'val', 'test'], f"Split must be one of ['train', 'val', 'test'], but got '{self.split}'"
        assert self.transform.keys() <= {'rgb', 'pc', 'mask', 'bbox3d'}, "Transform keys must be a subset of {'rgb', 'pc', 'mask', 'bbox3d'}"
        
        # Load sample list
        split_file = self.dataset_root / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.sample_ids = [line.strip() for line in f.readlines()]
        
        logger.info(f"Loaded {len(self.sample_ids)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
            - 'rgb': RGB image tensor [C, H, W]
            - 'pc': Point cloud tensor [3, H, W] (x, y, z)
            - 'mask': Segmentation mask tensor [num_objects, H, W]
            - 'bbox3d': 3D bounding boxes tensor [num_objects, 8, 3] (x, y, z)
            - 'sample_id': Sample ID (if return_sample_id=True)
        """
        sample_id = self.sample_ids[idx]
        sample_path = self.dataset_root / self.split / sample_id
        
        # Load RGB image
        rgb_path = sample_path / 'rgb.jpg'
        rgb = Image.open(rgb_path).convert('RGB')
        
        # Load point cloud
        pc_path = sample_path / 'pc.npy'
        pc = np.load(pc_path).astype(np.float32)
        pc = torch.from_numpy(pc).float()
                        
        # Load segmentation mask
        mask_path = sample_path / 'mask.npy'
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask)
        
        # Load 3D bounding boxes
        bbox3d_path = sample_path / 'bbox3d.npy'
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

        # Prepare output
        sample = {
            'rgb': rgb,
            'pc': pc,
            'mask': mask,
            'bbox3d': bbox3d
        }
        
        if self.return_sample_id:
            sample['sample_id'] = sample_id
        
        return sample
    
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
            'sample_id': sample_id,
            'rgb_path': str(sample_path / 'rgb.jpg'),
            'pc_path': str(sample_path / 'pc.npy'),
            'mask_path': str(sample_path / 'mask.npy'),
            'bbox3d_path': str(sample_path / 'bbox3d.npy')
        }
        
        # Get point cloud shape
        pc = np.load(sample_path / 'pc.npy')
        info['num_points'] = pc.shape[0]
        
        # Get bounding boxes count
        bbox3d = np.load(sample_path / 'bbox3d.npy')
        info['num_boxes'] = bbox3d.shape[0] if bbox3d.ndim > 1 else 1
        
        return info

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching samples with variable sizes.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    # Stack RGB images (assuming same size after transforms)
    #rgb_batch = torch.stack([sample['rgb'] for sample in batch])
    rgb_batch = [sample['rgb'] for sample in batch]  # List of tensors
    # Stack masks (assuming same size)
    #mask_batch = torch.stack([sample['mask'] for sample in batch])
    mask_batch = [sample['mask'] for sample in batch]
    # Handle variable-size point clouds
    pc_batch = [sample['pc'] for sample in batch]
    
    # Handle variable number of bounding boxes
    bbox3d_batch = [sample['bbox3d'] for sample in batch]
    
    result = {
        'rgb': rgb_batch,
        'pc': pc_batch, 
        'mask': mask_batch,
        'bbox3d': bbox3d_batch
    }
    
    # Include sample IDs if present
    if 'sample_id' in batch[0]:
        result['sample_id'] = [sample['sample_id'] for sample in batch]
    
    return result

def create_transforms_v1(split: str = 'train') -> transforms.Compose:
    """
    Create image transforms for training or validation.
    
    Args:
        split: 'train' or 'val'/'test'
        
    Returns:
        Composed transforms
    """
    if split == 'train':
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform


def create_transforms_modular_test():
    transform = dict()
    transform['rgb'] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    transform['pc'] = transforms.Compose([
        transforms.Lambda(lambda x: F.interpolate(x.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)),
    ])
    transform['mask'] = transforms.Compose([
        transforms.Resize((224, 224)),
    ])    
    return transform    

def test():
    dataset_root = "/home/hao1rng/sec_proj/processed_dataset"
    split = "train"
    transform = create_transforms_modular_test()
    
    dataset = Detection3DDataset(dataset_root=dataset_root, split=split, transform=transform, return_sample_id=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    for batch in dataloader:
        print("RGB batch shape:", batch['rgb'][0].shape, "Number of images:", len(batch['rgb']))
        print("Point cloud batch size:",  batch['pc'][0].shape, "Number of images:", len(batch['pc']))
        print("Mask batch shape:", batch['mask'][0].shape, "Number of masks:", len(batch['mask']))
        print("Bounding boxes batch size:",   batch['bbox3d'][0].shape, "Number of boxes:", len(batch['bbox3d']))
        if 'sample_id' in batch:
            print("Sample IDs:", batch['sample_id'])
        break

if __name__ == "__main__":
    test()