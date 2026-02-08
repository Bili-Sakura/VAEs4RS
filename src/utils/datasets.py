"""
Dataset loading utilities for remote sensing datasets.
Supports RESISC45, AID, and UCMerced datasets.
"""

import os
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.utils.config import get_config, PROJECT_ROOT
from diffusers.image_processor import VaeImageProcessor

# Optional rasterio for TIFF support
try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def get_transform(image_size: Optional[int] = None) -> transforms.Compose:
    """Get standard image transform for VAE evaluation."""
    t = []
    if image_size is not None:
        t.extend([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ])
    t.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(t)


def get_inverse_transform() -> transforms.Compose:
    """Get inverse transform to convert tensors back to images."""
    return transforms.Compose([
        transforms.Lambda(lambda x: VaeImageProcessor.denormalize(x)),
    ])


class MultiScaleCrop:
    """Multi-Scale Crop augmentation for VAE training.

    For large images (e.g., 1024px), randomly crops to one of several target
    sizes.  Default: 256px with 80 % probability, 512px with 20 % probability.
    Images smaller than all crop sizes are returned unchanged.
    """

    def __init__(
        self,
        crop_sizes: Tuple[int, ...] = (256, 512),
        crop_probs: Tuple[float, ...] = (0.8, 0.2),
    ):
        if len(crop_sizes) != len(crop_probs):
            raise ValueError("crop_sizes and crop_probs must have the same length")
        self.crop_sizes = crop_sizes
        self.crop_probs = crop_probs

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size  # PIL: (width, height)
        min_dim = min(w, h)

        # Keep only crop sizes that fit within the image
        valid = [(s, p) for s, p in zip(self.crop_sizes, self.crop_probs) if s <= min_dim]
        if not valid:
            return img

        sizes, probs = zip(*valid)
        crop_size = random.choices(sizes, weights=probs, k=1)[0]
        return transforms.RandomCrop(crop_size)(img)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"crop_sizes={self.crop_sizes}, crop_probs={self.crop_probs})"
        )


def get_multiscale_train_transform(
    crop_sizes: Tuple[int, ...] = (256, 512),
    crop_probs: Tuple[float, ...] = (0.8, 0.2),
) -> transforms.Compose:
    """Get training transform with multi-scale random crop augmentation."""
    return transforms.Compose([
        MultiScaleCrop(crop_sizes=crop_sizes, crop_probs=crop_probs),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def collate_variable_size(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Custom collate for variable-sized images. Pads to max size in batch."""
    images, labels, paths = zip(*batch)
    
    # Fast path: all same size
    if all(img.shape == images[0].shape for img in images):
        return torch.stack(images), torch.tensor(labels), list(paths)
    
    # Pad to max dims (rounded to multiple of 8 for VAE compatibility)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    target_h = ((max_h + 7) // 8) * 8
    target_w = ((max_w + 7) // 8) * 8
    
    padded = []
    for img in images:
        pad_h, pad_w = target_h - img.shape[1], target_w - img.shape[2]
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=-1.0)
        padded.append(img)
    
    return torch.stack(padded), torch.tensor(labels), list(paths)


class RSDataset(Dataset):
    """Remote Sensing Dataset wrapper with support for split files and class filtering."""
    
    def __init__(
        self,
        root: str,
        image_size: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        classes: Optional[List[str]] = None,
        split_file: Optional[str] = None,
        multi_scale_crop: bool = False,
        crop_sizes: Tuple[int, ...] = (256, 512),
        crop_probs: Tuple[float, ...] = (0.8, 0.2),
    ):
        self.root = self._resolve_path(root)
        self.image_size = image_size
        if transform is not None:
            self.transform = transform
        elif multi_scale_crop:
            self.transform = get_multiscale_train_transform(crop_sizes, crop_probs)
        else:
            self.transform = get_transform(image_size)
        
        # Load split file if provided
        split_filenames = self._load_split_file(split_file) if split_file else None
        
        # Build dataset
        self.image_paths, self.labels, self.class_names = self._scan_directory(
            self.root, classes, split_filenames
        )
        
        if not self.image_paths:
            raise ValueError(f"No images found in {self.root}")
    
    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to project root."""
        if os.path.isabs(path) and os.path.exists(path):
            return path
        resolved = PROJECT_ROOT / path
        return str(resolved) if resolved.exists() else path
    
    def _load_split_file(self, split_file: str) -> set:
        """Load split file (one filename per line)."""
        path = self._resolve_path(split_file)
        if not os.path.exists(path):
            raise ValueError(f"Split file not found: {path}")
        with open(path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    
    def _scan_directory(
        self, root: str, classes: Optional[List[str]], split_filenames: Optional[set]
    ) -> Tuple[List[str], List[int], List[str]]:
        """Scan directory for images."""
        if not os.path.exists(root):
            raise ValueError(f"Dataset root not found: {root}")
        
        all_classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        
        # Filter classes if specified
        if classes:
            classes_lower = [c.lower() for c in classes]
            all_classes = [c for c in all_classes if c.lower() in classes_lower]
            if not all_classes:
                raise ValueError(f"No matching classes found for {classes}")
        
        image_paths, labels, class_names = [], [], []
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        
        for idx, class_name in enumerate(all_classes):
            class_dir = os.path.join(root, class_name)
            class_names.append(class_name)
            
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(valid_exts):
                    continue
                if split_filenames and img_name not in split_filenames:
                    continue
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(idx)
        
        return image_paths, labels, class_names
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = self._load_image(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path
    
    def _load_image(self, path: str) -> Image.Image:
        """Load image (supports TIFF with rasterio)."""
        if path.lower().endswith(('.tif', '.tiff')):
            if not HAS_RASTERIO:
                raise ImportError("rasterio required for TIFF files: pip install rasterio")
            return self._load_tiff(path)
        return Image.open(path).convert("RGB")
    
    def _load_tiff(self, path: str) -> Image.Image:
        """Load TIFF file with rasterio."""
        with rasterio.open(path) as src:
            data = src.read()
            
            # Ensure 3 channels
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] == 2:
                data = np.concatenate([data, data[-1:]], axis=0)
            elif data.shape[0] > 3:
                data = data[:3]
            
            # Normalize to 0-255
            if data.dtype == np.uint16:
                data = (data / 65535.0 * 255.0).astype(np.uint8)
            elif data.dtype != np.uint8:
                data = ((data - data.min()) / max(data.max() - data.min(), 1) * 255).astype(np.uint8)
            
            return Image.fromarray(data.transpose(1, 2, 0), mode='RGB')


def load_dataset(
    dataset_name: str,
    image_size: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    classes: Optional[List[str]] = None,
    split_file: Optional[str] = None,
) -> Tuple[RSDataset, DataLoader]:
    """Load a remote sensing dataset by name."""
    cfg = get_config()
    if dataset_name not in cfg.datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cfg.datasets.keys())}")
    
    ds_config = cfg.datasets[dataset_name]
    dataset = RSDataset(
        root=ds_config.root,
        image_size=image_size,
        classes=classes,
        split_file=split_file,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_size if image_size is None else None,
    )
    
    return dataset, dataloader


def load_all_datasets(
    image_size: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict:
    """Load all configured datasets."""
    cfg = get_config()
    return {name: load_dataset(name, image_size, batch_size, num_workers) for name in cfg.datasets}
