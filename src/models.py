"""
VAE model loading utilities.

Provides a unified interface for loading different VAE architectures.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
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
    dtype: Optional[torch.dtype] = None,
) -> VAEWrapper:
    """
    Load a VAE model by name.
    
    Args:
        model_name: Name of the VAE model (e.g., "SD21-VAE", "SDXL-VAE")
        device: Device to load the model on
        dtype: Data type for the model (default: bfloat16 for all models)
        
    Returns:
        VAEWrapper instance
        
    Note:
        All models default to bfloat16 for better numerical stability.
        Automatically falls back to float16 if the GPU doesn't support bfloat16
        (e.g., older GPUs like Turing/Pascal architectures).
        float16 can be explicitly specified if needed.
    """
    if model_name not in VAE_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VAE_CONFIGS.keys())}")
    
    # Get checkpoint path from config
    base_config = VAE_CONFIGS[model_name]
    ckpt_path = Path(base_config.pretrained_path)
    
    # Resolve relative paths (pretrained_path might be relative to project root)
    if not ckpt_path.is_absolute():
        # Try to resolve relative to current working directory first
        if not ckpt_path.exists():
            # Try relative to project root (where run_experiments.py is)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            ckpt_path = project_root / base_config.pretrained_path
    
    # Load config.json from checkpoint folder
    config_json_path = ckpt_path / "config.json"
    if config_json_path.exists():
        with open(config_json_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create VAEConfig from loaded config.json
        # Use values from config.json if available, otherwise fall back to base_config
        config = VAEConfig(
            name=model_name,
            pretrained_path=base_config.pretrained_path,
            subfolder=base_config.subfolder,
            scaling_factor=config_dict.get("scaling_factor", base_config.scaling_factor),
            latent_channels=config_dict.get("latent_channels", base_config.latent_channels),
            image_size=config_dict.get("image_size", base_config.image_size),
        )
    else:
        # Fall back to hardcoded config if config.json doesn't exist
        config = base_config
    
    # Default to bfloat16 for all models (better numerical stability)
    # Fall back to float16 if bfloat16 is not supported by the GPU
    if dtype is None:
        if device.startswith("cuda") and torch.cuda.is_available():
            # Check if GPU supports bfloat16 (Ampere+ architectures)
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            # For CPU or when CUDA is not available, use float16
            dtype = torch.float16
    
    # Ensure dtype is float16 or bfloat16
    # If an invalid dtype was provided, use bfloat16 if supported, otherwise float16
    if dtype not in (torch.float16, torch.bfloat16):
        if device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    
    # Final check: if bfloat16 was selected but GPU doesn't support it, fall back to float16
    if dtype == torch.bfloat16 and device.startswith("cuda") and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            dtype = torch.float16
    
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
    dtype: Optional[torch.dtype] = None,
) -> dict[str, VAEWrapper]:
    """
    Load all VAE models.
    
    Args:
        device: Device to load models on
        dtype: Data type for models (default: bfloat16 for all models)
        
    Returns:
        Dictionary mapping model names to VAEWrapper instances
        
    Note:
        All models default to bfloat16 for better numerical stability.
    """
    vaes = {}
    for name in VAE_CONFIGS:
        print(f"Loading {name}...")
        try:
            vaes[name] = load_vae(name, device, dtype)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    return vaes
