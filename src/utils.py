"""
Utility functions for VAE evaluation.
"""

import os
import random
from typing import Optional

import torch
import numpy as np
from PIL import Image


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> str:
    """Get the appropriate device."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_tensor_as_image(tensor: torch.Tensor, path: str, normalize: bool = True):
    """Save a tensor as an image file. Tensor shape: (3, H, W), range [-1, 1]."""
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
