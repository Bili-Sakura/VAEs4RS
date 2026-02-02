"""
Dataset loading utilities for remote sensing datasets.

Supports NWPU-RESISC45 and AID datasets.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

try:
    from .config import DatasetConfig, DATASET_CONFIGS
except ImportError:
    from config import DatasetConfig, DATASET_CONFIGS


def get_transform(image_size: Optional[int] = None) -> transforms.Compose:
    """
    Get standard image transform for VAE evaluation.
    
    Args:
        image_size: Target image size. If None, images are loaded at their original size.
        
    Returns:
        Composed transform
    """
    transform_list = []
    
    # Only resize and crop if image_size is specified
    if image_size is not None:
        transform_list.extend([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ])
    
    # Always convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    return transforms.Compose(transform_list)


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


def collate_variable_size(batch: List[Tuple[torch.Tensor, int, str]]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Custom collate function for variable-sized images.
    
    Pads images to the maximum size in the batch.
    
    Args:
        batch: List of (image, label, path) tuples
        
    Returns:
        Tuple of (padded_images, labels, paths)
    """
    images, labels, paths = zip(*batch)
    
    # Find maximum height and width
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    channels = images[0].shape[0]
    
    # Pad all images to max size
    padded_images = []
    for img in images:
        h, w = img.shape[1], img.shape[2]
        pad_h = max_h - h
        pad_w = max_w - w
        
        # Pad with -1 (normalized value for black in [-1, 1] range)
        padded = torch.nn.functional.pad(
            img,
            (0, pad_w, 0, pad_h),
            mode='constant',
            value=-1.0
        )
        padded_images.append(padded)
    
    return torch.stack(padded_images), torch.tensor(labels), list(paths)


class RSDataset(Dataset):
    """
    Remote Sensing Dataset wrapper.
    
    Provides consistent interface for RESISC45 and AID datasets.
    """
    
    def __init__(
        self,
        root: str,
        image_size: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        classes: Optional[list[str]] = None,
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
        self.classes_filter = classes  # List of class names to include (None = all classes)
        
        # Find all images
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        if not os.path.exists(self.root):
            raise ValueError(
                f"Dataset root directory does not exist: {self.root}\n"
                f"Please ensure the dataset is available at this path."
            )
        
        # Get all available classes
        all_classes = sorted([d for d in os.listdir(self.root) 
                             if os.path.isdir(os.path.join(self.root, d))])
        
        # Filter classes if specified
        if self.classes_filter is not None:
            # Normalize class names (case-insensitive matching)
            classes_filter_lower = [c.lower() for c in self.classes_filter]
            selected_classes = [c for c in all_classes 
                              if c.lower() in classes_filter_lower]
            if not selected_classes:
                raise ValueError(
                    f"None of the specified classes {self.classes_filter} found in dataset. "
                    f"Available classes: {all_classes[:10]}..." if len(all_classes) > 10 else f"Available classes: {all_classes}"
                )
            all_classes = selected_classes
        
        # Build dataset
        for class_idx, class_name in enumerate(all_classes):
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
    image_size: Optional[int] = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    classes: Optional[list[str]] = None,
) -> Tuple[RSDataset, DataLoader]:
    """
    Load a remote sensing dataset.
    
    Args:
        dataset_name: Name of the dataset ("RESISC45" or "AID")
        image_size: Target image size. If None, images are loaded at their original size.
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        classes: Optional list of class names to filter (None = all classes)
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    dataset = RSDataset(
        root=config.root,
        image_size=image_size,
        classes=classes,
    )
    
    # Use custom collate function for variable-sized images
    # If image_size is None, images have variable sizes and need padding
    collate_fn = collate_variable_size if image_size is None else None
    
    # When using variable sizes, batch_size=1 is safer to avoid memory issues
    effective_batch_size = 1 if image_size is None else batch_size
    
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return dataset, dataloader


def load_all_datasets(
    image_size: Optional[int] = 256,
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict[str, Tuple[RSDataset, DataLoader]]:
    """
    Load all remote sensing datasets.
    
    Args:
        image_size: Target image size. If None, images are loaded at their original size.
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
