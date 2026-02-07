"""
Utility functions for VAE evaluation.
"""

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image

from diffusers.training_utils import set_seed  # noqa: F401 – re-exported
from diffusers.utils import pt_to_pil


def get_device(device: Optional[str] = None) -> str:
    """Get the appropriate device."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_tensor_as_image(tensor: torch.Tensor, path: str, normalize: bool = True):
    """Save a tensor as an image file. Tensor shape: (3, H, W), range [-1, 1]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if normalize:
        pil_img = pt_to_pil(tensor.unsqueeze(0))[0]
    else:
        # Already in [0, 1]: convert directly
        img = tensor.cpu().float().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        pil_img = Image.fromarray(img)
    pil_img.save(path)


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
