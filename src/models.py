"""
VAE model loading utilities.

Provides a unified interface for loading different VAE architectures.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from diffusers import AutoencoderKL, AutoencoderDC, AutoencoderKLQwenImage, AutoencoderKLFlux2

try:
    from .config import VAEConfig, VAE_CONFIGS
except ImportError:
    from config import VAEConfig, VAE_CONFIGS


# Mapping from class name strings to actual VAE classes
VAE_CLASS_MAP = {
    "AutoencoderKL": AutoencoderKL,
    "AutoencoderDC": AutoencoderDC,
    "AutoencoderKLQwenImage": AutoencoderKLQwenImage,
    "AutoencoderKLFlux2": AutoencoderKLFlux2,
}


class VAEWrapper(nn.Module):
    """
    Unified wrapper for different VAE architectures.
    
    Provides consistent encode/decode interface regardless of underlying model.
    """
    
    def __init__(self, model, config: VAEConfig, vae_class_name: str = "AutoencoderKL"):
        super().__init__()
        self.model = model
        self.config = config
        self.scaling_factor = config.scaling_factor
        self.vae_class_name = vae_class_name
        self._original_shape = None  # Store original shape for padding/cropping
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.
        
        Args:
            x: Input images, shape (B, 3, H, W), range [-1, 1]
            
        Returns:
            Latent representations, shape (B, C, H//f, W//f)
        """
        # Store original shape BEFORE any padding
        original_shape = x.shape
        h, w = x.shape[-2], x.shape[-1]
        
        # Determine padding factor based on model
        # SANA-VAE uses 32x spatial compression, so needs dimensions divisible by 32
        # Other models typically need dimensions divisible by 2
        if self.config.name == "SANA-VAE":
            pad_factor = 32
        else:
            pad_factor = 2
        
        # Pad to make dimensions divisible by pad_factor (required for pixel_unshuffle)
        pad_h = (pad_factor - h % pad_factor) % pad_factor
        pad_w = (pad_factor - w % pad_factor) % pad_factor
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Store the shape after padding for decode
        self._original_shape = x.shape if (pad_h > 0 or pad_w > 0) else original_shape
        
        # Qwen-VAE expects 5D input: (B, C, num_frame, H, W)
        # Add frame dimension if needed
        if self.vae_class_name == "AutoencoderKLQwenImage" and x.dim() == 4:
            x = x.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        
        encoded = self.model.encode(x)
        
        # AutoencoderKL returns EncoderOutput with latent_dist, AutoencoderDC returns with latent
        if hasattr(encoded, 'latent_dist'):
            z = encoded.latent_dist.sample()
        elif hasattr(encoded, 'latent'):
            z = encoded.latent
        else:
            z = encoded
        
        # Remove frame dimension if it was added (Qwen-VAE returns 5D latents)
        if self.vae_class_name == "AutoencoderKLQwenImage" and z.dim() == 5:
            z = z.squeeze(2)  # (B, C, 1, H//f, W//f) -> (B, C, H//f, W//f)
        
        return z * self.scaling_factor
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor, original_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """
        Decode latents to images.
        
        Args:
            z: Latent representations, shape (B, C, H//f, W//f)
            original_shape: Original input shape (B, C, H, W) for cropping. If None, uses stored shape.
            
        Returns:
            Reconstructed images, shape (B, 3, H, W), range [-1, 1]
        """
        z = z / self.scaling_factor
        
        # Qwen-VAE expects 5D latents: (B, C, num_frame, H//f, W//f)
        # Add frame dimension if needed
        if self.vae_class_name == "AutoencoderKLQwenImage" and z.dim() == 4:
            z = z.unsqueeze(2)  # (B, C, H//f, W//f) -> (B, C, 1, H//f, W//f)
        
        decoded = self.model.decode(z)
        
        # Handle different return types - simple check for sample attribute
        result = decoded.sample if hasattr(decoded, 'sample') else decoded
        
        # Remove frame dimension if it was added (Qwen-VAE returns 5D images)
        if self.vae_class_name == "AutoencoderKLQwenImage" and result.dim() == 5:
            result = result.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
        
        # Crop to original size if padding was applied
        target_shape = original_shape or self._original_shape
        if target_shape is not None and result.shape != target_shape:
            h, w = target_shape[-2], target_shape[-1]
            result = result[..., :h, :w]
        
        return result
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and decode images (full reconstruction).
        
        Args:
            x: Input images, shape (B, 3, H, W), range [-1, 1]
            
        Returns:
            Reconstructed images, shape (B, 3, H, W), range [-1, 1]
        """
        # Store original shape before encoding
        original_shape = x.shape
        z = self.encode(x)
        # Pass original shape explicitly to ensure correct cropping
        return self.decode(z, original_shape=original_shape)


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
    vae_class = AutoencoderKL  # Default fallback
    vae_class_name = "AutoencoderKL"  # Default class name
    if config_json_path.exists():
        with open(config_json_path, 'r') as f:
            config_dict = json.load(f)
        
        # Get the VAE class name from config.json
        class_name = config_dict.get("_class_name", "AutoencoderKL")
        vae_class_name = class_name
        if class_name in VAE_CLASS_MAP:
            vae_class = VAE_CLASS_MAP[class_name]
        else:
            print(f"Warning: Unknown VAE class '{class_name}' for {model_name}, falling back to AutoencoderKL")
        
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
    # Use ignore_mismatched_sizes=True to handle shape mismatches (e.g., SANA-VAE)
    # low_cpu_mem_usage=False is required when using ignore_mismatched_sizes=True
    if config.subfolder:
        model = vae_class.from_pretrained(
            config.pretrained_path,
            subfolder=config.subfolder,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )
    else:
        model = vae_class.from_pretrained(
            config.pretrained_path,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )
    
    model = model.to(device)
    model.eval()
    
    return VAEWrapper(model, config, vae_class_name)


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
