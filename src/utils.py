"""
Utility functions for VAE evaluation.
"""

import os
import random
from typing import Optional

import torch
import numpy as np


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> str:
    """
    Get the appropriate device.
    
    Args:
        device: Preferred device ("cuda", "cpu", or None for auto)
        
    Returns:
        Device string
    """
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_size(num_bytes: int) -> str:
    """
    Format byte size to human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def save_tensor_as_image(
    tensor: torch.Tensor,
    path: str,
    normalize: bool = True,
):
    """
    Save a tensor as an image file.
    
    Args:
        tensor: Image tensor, shape (3, H, W)
        path: Output path
        normalize: Whether to normalize from [-1, 1] to [0, 255]
    """
    from PIL import Image
    
    img = tensor.cpu().numpy()
    if normalize:
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
