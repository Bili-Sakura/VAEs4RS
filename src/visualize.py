"""
Visualization utilities for VAE reconstruction results.
"""

import os
from typing import List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from diffusers.utils import pt_to_pil
from diffusers.training_utils import free_memory

from .models import load_vae
from .datasets import load_dataset


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor (3, H, W) in [-1, 1] to numpy image (H, W, 3) in [0, 255]."""
    pil_img = pt_to_pil(tensor.unsqueeze(0))[0]
    return np.array(pil_img)


def visualize_reconstructions(
    model_names: List[str],
    dataset_name: str,
    num_samples: int = 5,
    image_size: Optional[int] = 256,
    output_path: Optional[str] = None,
    device: str = "cuda",
):
    """Create grid visualization of original and reconstructed images."""
    # Load dataset
    dataset, dataloader = load_dataset(dataset_name, image_size, num_samples, num_workers=0)
    sample_batch = next(iter(dataloader))
    
    # Load models and reconstruct
    reconstructions = {}
    for model_name in model_names:
        print(f"Loading {model_name}...")
        try:
            vae = load_vae(model_name, device=device)
            model_dtype = next(vae.model.parameters()).dtype
            original = sample_batch[0].to(device, dtype=model_dtype)
            
            with torch.no_grad():
                recon = vae.reconstruct(original)
            reconstructions[model_name] = recon.float()
            
            del vae
            free_memory()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    # Create visualization
    num_rows = 1 + len(reconstructions)
    num_cols = num_samples
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows), squeeze=False)
    
    # Original images
    for col in range(num_cols):
        axes[0, col].imshow(tensor_to_image(original[col].float()))
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=12)
    
    # Reconstructions
    for row, (name, recon) in enumerate(reconstructions.items(), 1):
        for col in range(num_cols):
            axes[row, col].imshow(tensor_to_image(recon[col]))
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=12)
    
    plt.suptitle(f"VAE Reconstructions on {dataset_name}", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()
    return fig
