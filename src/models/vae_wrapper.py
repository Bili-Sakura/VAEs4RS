"""
VAE model loading utilities.
Unified interface for loading different VAE architectures.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from diffusers import AutoencoderKL, AutoencoderDC, AutoencoderKLQwenImage, AutoencoderKLFlux2

from ..utils.config import get_config, VAEConfig, PROJECT_ROOT

# VAE class mapping
VAE_CLASSES = {
    "AutoencoderKL": AutoencoderKL,
    "AutoencoderDC": AutoencoderDC,
    "AutoencoderKLQwenImage": AutoencoderKLQwenImage,
    "AutoencoderKLFlux2": AutoencoderKLFlux2,
}


class VAEWrapper(nn.Module):
    """Unified wrapper for different VAE architectures with consistent encode/decode interface."""
    
    def __init__(self, model: nn.Module, config: VAEConfig, vae_class_name: str = "AutoencoderKL"):
        super().__init__()
        self.model = model
        self.config = config
        self.scaling_factor = config.scaling_factor
        self.vae_class_name = vae_class_name
        self._original_shape = None
    
    def _get_pad_factor(self) -> int:
        """Get padding factor based on model type."""
        return 32 if self.config.name == "SANA-VAE" else 2
    
    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to be divisible by pad_factor."""
        h, w = x.shape[-2:]
        pad_factor = self._get_pad_factor()
        pad_h = (pad_factor - h % pad_factor) % pad_factor
        pad_w = (pad_factor - w % pad_factor) % pad_factor
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space. Input: (B, 3, H, W) in [-1, 1]."""
        self._original_shape = x.shape
        x = self._pad_input(x)
        
        # Qwen-VAE expects 5D input
        if self.vae_class_name == "AutoencoderKLQwenImage" and x.dim() == 4:
            x = x.unsqueeze(2)
        
        encoded = self.model.encode(x)
        z = encoded.latent_dist.sample() if hasattr(encoded, 'latent_dist') else \
            encoded.latent if hasattr(encoded, 'latent') else encoded
        
        # Remove frame dim for Qwen-VAE
        if self.vae_class_name == "AutoencoderKLQwenImage" and z.dim() == 5:
            z = z.squeeze(2)
        
        return z * self.scaling_factor
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor, original_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """Decode latents to images. Output: (B, 3, H, W) in [-1, 1]."""
        z = z / self.scaling_factor
        
        # Qwen-VAE expects 5D latents
        if self.vae_class_name == "AutoencoderKLQwenImage" and z.dim() == 4:
            z = z.unsqueeze(2)
        
        decoded = self.model.decode(z)
        result = decoded.sample if hasattr(decoded, 'sample') else decoded
        
        # Remove frame dim for Qwen-VAE
        if self.vae_class_name == "AutoencoderKLQwenImage" and result.dim() == 5:
            result = result.squeeze(2)
        
        # Crop to original size
        target_shape = original_shape or self._original_shape
        if target_shape is not None and result.shape[-2:] != target_shape[-2:]:
            result = result[..., :target_shape[-2], :target_shape[-1]]
        
        return result
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode (full reconstruction)."""
        original_shape = x.shape
        z = self.encode(x)
        return self.decode(z, original_shape=original_shape)


def _resolve_path(path: str) -> Path:
    """Resolve path relative to project root."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    resolved = PROJECT_ROOT / path
    return resolved if resolved.exists() else p


def _get_dtype(device: str, dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """Determine best dtype for device."""
    if dtype is not None and dtype in (torch.float16, torch.bfloat16):
        return dtype
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16


def load_vae(model_name: str, device: str = "cuda", dtype: Optional[torch.dtype] = None) -> VAEWrapper:
    """
    Load a VAE model by name.
    
    Args:
        model_name: Name of the VAE model (e.g., "SD21-VAE")
        device: Device to load the model on
        dtype: Data type (default: auto-detect best for device)
    
    Returns:
        VAEWrapper instance
    """
    cfg = get_config()
    if model_name not in cfg.vaes:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(cfg.vaes.keys())}")
    
    vae_config = cfg.vaes[model_name]
    ckpt_path = _resolve_path(vae_config.pretrained_path)
    
    # Detect VAE class from config.json
    vae_class = AutoencoderKL
    vae_class_name = "AutoencoderKL"
    config_json = ckpt_path / "config.json"
    
    if config_json.exists():
        with open(config_json, 'r') as f:
            model_cfg = json.load(f)
        class_name = model_cfg.get("_class_name", "AutoencoderKL")
        if class_name in VAE_CLASSES:
            vae_class = VAE_CLASSES[class_name]
            vae_class_name = class_name
        # Update config from model's config.json
        vae_config = VAEConfig(
            name=model_name,
            pretrained_path=vae_config.pretrained_path,
            scaling_factor=model_cfg.get("scaling_factor", vae_config.scaling_factor),
            latent_channels=model_cfg.get("latent_channels", vae_config.latent_channels),
            spatial_compression=vae_config.spatial_compression,
            subfolder=vae_config.subfolder,
        )
    
    # Load model
    dtype = _get_dtype(device, dtype)
    load_kwargs = dict(torch_dtype=dtype, ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
    
    if vae_config.subfolder:
        model = vae_class.from_pretrained(str(ckpt_path), subfolder=vae_config.subfolder, **load_kwargs)
    else:
        model = vae_class.from_pretrained(str(ckpt_path), **load_kwargs)
    
    model = model.to(device).eval()
    return VAEWrapper(model, vae_config, vae_class_name)


def load_all_vaes(device: str = "cuda", dtype: Optional[torch.dtype] = None) -> dict:
    """Load all configured VAE models."""
    cfg = get_config()
    vaes = {}
    for name in cfg.vaes:
        print(f"Loading {name}...")
        try:
            vaes[name] = load_vae(name, device, dtype)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    return vaes
