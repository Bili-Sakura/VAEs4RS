"""
Visualization utilities for VAE reconstruction results.

Generates qualitative comparisons between original and reconstructed images.
"""

import os
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Try relative imports first (when used as a package), fall back to absolute (when imported directly)
try:
    from .models import load_vae, VAEWrapper
    from .datasets import load_dataset, get_inverse_transform
except ImportError:
    # Fall back to absolute imports when src is in path
    from models import load_vae, VAEWrapper
    from datasets import load_dataset, get_inverse_transform


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to a numpy image.
    
    Args:
        tensor: Image tensor, shape (3, H, W), range [-1, 1]
        
    Returns:
        Numpy array, shape (H, W, 3), range [0, 255]
    """
    img = tensor.cpu().numpy()
    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


def visualize_reconstructions(
    model_names: List[str],
    dataset_name: str,
    num_samples: int = 5,
    image_size: int = 256,
    output_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Create a grid visualization of original and reconstructed images.
    
    Args:
        model_names: List of VAE model names to compare
        dataset_name: Name of the dataset
        num_samples: Number of sample images to show
        image_size: Target image size
        output_path: Path to save the figure
        device: Device to use
    """
    # Load dataset
    dataset, dataloader = load_dataset(
        dataset_name,
        image_size=image_size,
        batch_size=num_samples,
        num_workers=0,
    )
    
    # Get sample images
    sample_batch = next(iter(dataloader))
    
    # Load models and reconstruct
    reconstructions = {}
    for model_name in model_names:
        print(f"Loading {model_name}...")
        try:
            vae = load_vae(model_name, device=device)
            # Use model's dtype (models use float16/bfloat16)
            model_dtype = next(vae.model.parameters()).dtype
            original_images = sample_batch[0].to(device, dtype=model_dtype)
            with torch.no_grad():
                recon = vae.reconstruct(original_images)
            reconstructions[model_name] = recon.float()
            del vae
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    # Create visualization
    num_rows = 1 + len(reconstructions)  # Original + reconstructions
    num_cols = num_samples
    
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(3 * num_cols, 3 * num_rows),
        squeeze=False,
    )
    
    # Plot original images
    for col in range(num_cols):
        img = tensor_to_image(original_images[col].float())
        axes[0, col].imshow(img)
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=12)
    
    # Plot reconstructions
    for row, (model_name, recon) in enumerate(reconstructions.items(), 1):
        for col in range(num_cols):
            img = tensor_to_image(recon[col])
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(model_name, fontsize=12)
    
    plt.suptitle(f"VAE Reconstructions on {dataset_name}", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()
    return fig


def visualize_latent_space(
    model_name: str,
    dataset_name: str,
    num_samples: int = 1000,
    image_size: int = 256,
    output_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Visualize the latent space using t-SNE.
    
    Args:
        model_name: VAE model name
        dataset_name: Name of the dataset
        num_samples: Number of samples to visualize
        image_size: Target image size
        output_path: Path to save the figure
        device: Device to use
    """
    from sklearn.manifold import TSNE
    
    # Load model and dataset
    vae = load_vae(model_name, device=device)
    try:
        # Use model's dtype (models use float16/bfloat16)
        model_dtype = next(vae.model.parameters()).dtype
        dataset, dataloader = load_dataset(
            dataset_name,
            image_size=image_size,
            batch_size=32,
            num_workers=4,
        )
        
        # Extract latent representations
        latents = []
        labels = []
        count = 0
        
        for images, lbls, _ in dataloader:
            if count >= num_samples:
                break
            images = images.to(device, dtype=model_dtype)
            with torch.no_grad():
                z = vae.encode(images)
            latents.append(z.cpu().numpy().reshape(z.size(0), -1))
            labels.extend(lbls.numpy())
            count += z.size(0)
        
        latents = np.concatenate(latents, axis=0)[:num_samples]
        labels = np.array(labels)[:num_samples]
        
        # Apply t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        latents_2d = tsne.fit_transform(latents)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            latents_2d[:, 0],
            latents_2d[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.6,
            s=10,
        )
        ax.set_title(f"Latent Space ({model_name} on {dataset_name})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_path}")
        
        plt.show()
        return fig
    finally:
        # Clear GPU memory
        del vae
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example usage
    visualize_reconstructions(
        model_names=["SD21-VAE", "SDXL-VAE", "SD35-VAE"],
        dataset_name="RESISC45",
        num_samples=5,
        output_path="outputs/figures/reconstruction_comparison.png",
    )
