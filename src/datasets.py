"""
Dataset loading utilities for remote sensing datasets.

Supports NWPU-RESISC45 and AID datasets.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.errors import NotGeoreferencedWarning
    # Suppress NotGeoreferencedWarning - not needed for image processing
    # Set filter before any rasterio operations
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning, module='rasterio')
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

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
    
    Pads images to the maximum size in the batch, ensuring dimensions are divisible by 8
    (required by most VAE models for proper downsampling).
    
    Args:
        batch: List of (image, label, path) tuples
        
    Returns:
        Tuple of (padded_images, labels, paths)
    """
    images, labels, paths = zip(*batch)
    
    # Check if all images have the same size (common case, e.g., UCMerced)
    first_shape = images[0].shape
    all_same_size = all(img.shape == first_shape for img in images)
    
    if all_same_size:
        # Fast path: all images are the same size, just stack them
        return torch.stack(images), torch.tensor(labels), list(paths)
    
    # Variable-sized images: find maximum height and width
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    channels = images[0].shape[0]
    
    # Pad to next multiple of 8 to ensure VAE compatibility
    # Most VAEs require dimensions divisible by 8 (or at least 2) for proper downsampling
    # Using 8 ensures compatibility with all common VAE architectures
    def round_up_to_multiple(n: int, multiple: int = 8) -> int:
        """Round up to the next multiple."""
        return ((n + multiple - 1) // multiple) * multiple
    
    target_h = round_up_to_multiple(max_h, 8)
    target_w = round_up_to_multiple(max_w, 8)
    
    # Pad all images to target size
    padded_images = []
    for img in images:
        h, w = img.shape[1], img.shape[2]
        pad_h = target_h - h
        pad_w = target_w - w
        
        # Only pad if needed
        if pad_h > 0 or pad_w > 0:
            # Pad with -1 (normalized value for black in [-1, 1] range)
            padded = torch.nn.functional.pad(
                img,
                (0, pad_w, 0, pad_h),
                mode='constant',
                value=-1.0
            )
            padded_images.append(padded)
        else:
            padded_images.append(img)
    
    return torch.stack(padded_images), torch.tensor(labels), list(paths)


class RSDataset(Dataset):
    """
    Remote Sensing Dataset wrapper.
    
    Provides consistent interface for RESISC45, AID, and UCMerced datasets.
    Supports loading from split files (e.g., test set).
    """
    
    def __init__(
        self,
        root: str,
        image_size: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        classes: Optional[list[str]] = None,
        split_file: Optional[str] = None,
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
        self.split_file = split_file  # Path to split file (e.g., test set)
        
        # Load split file if provided
        split_filenames = None
        if split_file is not None:
            # Resolve split file path
            if not os.path.isabs(split_file):
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent
                resolved_split = project_root / split_file
                if resolved_split.exists():
                    split_file = str(resolved_split)
                elif os.path.exists(split_file):
                    split_file = os.path.abspath(split_file)
                else:
                    split_file = str(project_root / split_file)
            
            split_file = os.path.abspath(split_file)
            if not os.path.exists(split_file):
                raise ValueError(f"Split file does not exist: {split_file}")
            
            # Read split file (one filename per line)
            with open(split_file, 'r') as f:
                split_filenames = set(line.strip() for line in f if line.strip())
        
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
                    # If split file is provided, only include images in the split
                    if split_filenames is not None:
                        if img_name not in split_filenames:
                            continue
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
        
        if len(self.image_paths) == 0:
            split_info = f" (split file: {split_file})" if split_file else ""
            raise ValueError(f"No images found in dataset root: {self.root}{split_info}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Use rasterio for .tif/.tiff files, PIL for other formats
        if img_path.lower().endswith(('.tif', '.tiff')):
            if not HAS_RASTERIO:
                raise ImportError(
                    "rasterio is required to read .tif files. "
                    "Install it with: pip install rasterio"
                )
            
            # Read with rasterio (warnings suppressed for non-georeferenced images)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
                with rasterio.open(img_path) as src:
                    # Read all bands
                    data = src.read()  # Shape: (bands, height, width)
                    
                    # Handle different number of bands
                    if data.shape[0] == 1:
                        # Single band: convert to RGB by duplicating
                        data = np.repeat(data, 3, axis=0)
                    elif data.shape[0] == 2:
                        # Two bands: add a third band (duplicate second)
                        data = np.concatenate([data, data[-1:]], axis=0)
                    elif data.shape[0] > 3:
                        # More than 3 bands: take first 3
                        data = data[:3]
                    
                    # Normalize to 0-255 range if needed
                    # Handle different data types
                    if data.dtype == np.uint16:
                        # Scale 16-bit to 8-bit
                        data = (data / 65535.0 * 255.0).astype(np.uint8)
                    elif data.dtype != np.uint8:
                        # Normalize other types to 0-255
                        data_min = data.min()
                        data_max = data.max()
                        if data_max > data_min:
                            data = ((data - data_min) / (data_max - data_min) * 255.0).astype(np.uint8)
                        else:
                            data = np.zeros_like(data, dtype=np.uint8)
                    
                    # Transpose to (height, width, channels) for PIL
                    data = data.transpose(1, 2, 0)
                    
                    # Convert to PIL Image
                    image = Image.fromarray(data, mode='RGB')
        else:
            # Use PIL for other formats
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
    split_file: Optional[str] = None,
) -> Tuple[RSDataset, DataLoader]:
    """
    Load a remote sensing dataset.
    
    Args:
        dataset_name: Name of the dataset ("RESISC45", "AID", or "UCMerced")
        image_size: Target image size. If None, images are loaded at their original size.
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        classes: Optional list of class names to filter (None = all classes)
        split_file: Optional path to split file (e.g., test set). File should contain one filename per line.
        
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
        split_file=split_file,
    )
    
    # Use custom collate function for variable-sized images
    # If image_size is None, images have variable sizes and need padding
    collate_fn = collate_variable_size if image_size is None else None
    
    # When using variable sizes, use the requested batch size
    # The collate function will pad images to the maximum size in each batch
    # This can be memory-intensive with large batches and very different image sizes
    if image_size is None:
        effective_batch_size = batch_size
        if batch_size > 16:
            import warnings
            warnings.warn(
                f"Using batch_size={batch_size} with variable-sized images. "
                f"Images will be padded to the maximum size in each batch, which may use significant memory. "
                f"Consider using a smaller batch size or fixed image_size if you encounter memory issues.",
                UserWarning
            )
    else:
        effective_batch_size = batch_size
    
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
