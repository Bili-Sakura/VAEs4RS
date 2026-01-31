"""
VAE model loading utilities.

Provides a unified interface for loading different VAE architectures.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from diffusers import AutoencoderKL

from config import VAEConfig, VAE_CONFIGS


class VAEWrapper(nn.Module):
    """
    Unified wrapper for different VAE architectures.
    
    Provides consistent encode/decode interface regardless of underlying model.
    """
    
    def __init__(self, model: AutoencoderKL, config: VAEConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.scaling_factor = config.scaling_factor
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.
        
        Args:
            x: Input images, shape (B, 3, H, W), range [-1, 1]
            
        Returns:
            Latent representations, shape (B, C, H//f, W//f)
        """
        posterior = self.model.encode(x).latent_dist
        z = posterior.sample()
        return z * self.scaling_factor
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.
        
        Args:
            z: Latent representations, shape (B, C, H//f, W//f)
            
        Returns:
            Reconstructed images, shape (B, 3, H, W), range [-1, 1]
        """
        z = z / self.scaling_factor
        return self.model.decode(z).sample
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and decode images (full reconstruction).
        
        Args:
            x: Input images, shape (B, 3, H, W), range [-1, 1]
            
        Returns:
            Reconstructed images, shape (B, 3, H, W), range [-1, 1]
        """
        z = self.encode(x)
        return self.decode(z)


def load_vae(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> VAEWrapper:
    """
    Load a VAE model by name.
    
    Args:
        model_name: Name of the VAE model (e.g., "SD21-VAE", "SDXL-VAE")
        device: Device to load the model on
        dtype: Data type for the model
        
    Returns:
        VAEWrapper instance
    """
    if model_name not in VAE_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VAE_CONFIGS.keys())}")
    
    config = VAE_CONFIGS[model_name]
    
    # Load the model
    if config.subfolder:
        model = AutoencoderKL.from_pretrained(
            config.pretrained_path,
            subfolder=config.subfolder,
            torch_dtype=dtype,
        )
    else:
        model = AutoencoderKL.from_pretrained(
            config.pretrained_path,
            torch_dtype=dtype,
        )
    
    model = model.to(device)
    model.eval()
    
    return VAEWrapper(model, config)


def load_all_vaes(
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> dict[str, VAEWrapper]:
    """
    Load all VAE models.
    
    Args:
        device: Device to load models on
        dtype: Data type for models
        
    Returns:
        Dictionary mapping model names to VAEWrapper instances
    """
    vaes = {}
    for name in VAE_CONFIGS:
        print(f"Loading {name}...")
        try:
            vaes[name] = load_vae(name, device, dtype)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    return vaes
