"""
Dataset loading utilities for remote sensing datasets.

Supports NWPU-RESISC45 and AID datasets.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from config import DatasetConfig, DATASET_CONFIGS


def get_transform(image_size: int = 256) -> transforms.Compose:
    """
    Get standard image transform for VAE evaluation.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_inverse_transform() -> transforms.Compose:
    """
    Get inverse transform to convert tensors back to images.
    
    Returns:
        Composed inverse transform
    """
    return transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
    ])


class RSDataset(Dataset):
    """
    Remote Sensing Dataset wrapper.
    
    Provides consistent interface for RESISC45 and AID datasets.
    """
    
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
    ):
        # Resolve path relative to project root if not absolute
        if not os.path.isabs(root):
            # Try to find project root (where run_experiments.py is)
            # Check multiple possible locations
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            
            # Try the resolved path first
            resolved_root = project_root / root
            if resolved_root.exists():
                root = str(resolved_root)
            else:
                # Fallback: try relative to current working directory
                if os.path.exists(root):
                    root = os.path.abspath(root)
                else:
                    # Last resort: try relative to project root as-is
                    root = str(project_root / root)
        
        self.root = os.path.abspath(root)
        self.image_size = image_size
        self.transform = transform or get_transform(image_size)
        
        # Find all images
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        if not os.path.exists(self.root):
            raise ValueError(
                f"Dataset root directory does not exist: {self.root}\n"
                f"Please ensure the dataset is available at this path."
            )
        
        for class_idx, class_name in enumerate(sorted(os.listdir(self.root))):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_names.append(class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in dataset root: {self.root}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def load_dataset(
    dataset_name: str,
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[RSDataset, DataLoader]:
    """
    Load a remote sensing dataset.
    
    Args:
        dataset_name: Name of the dataset ("RESISC45" or "AID")
        image_size: Target image size
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    dataset = RSDataset(
        root=config.root,
        image_size=image_size,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataset, dataloader


def load_all_datasets(
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict[str, Tuple[RSDataset, DataLoader]]:
    """
    Load all remote sensing datasets.
    
    Args:
        image_size: Target image size
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        
    Returns:
        Dictionary mapping dataset names to (dataset, dataloader) tuples
    """
    datasets = {}
    for name in DATASET_CONFIGS:
        print(f"Loading {name}...")
        datasets[name] = load_dataset(name, image_size, batch_size, num_workers)
    return datasets
