"""
VAE Model Statistics Calculator.
Calculates GFLOPs, parameters, compression ratios for VAE models.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from ..utils.config import get_config
from .vae_wrapper import load_vae


@dataclass
class VAEStatistics:
    """Statistics for a VAE model."""
    name: str
    total_params: int
    encoder_params: int
    decoder_params: int
    encode_gflops: float
    decode_gflops: float
    total_gflops: float
    spatial_compression: int
    latent_channels: int
    data_compression: float
    input_shape: Tuple[int, int, int]
    latent_shape: Tuple[int, int, int]
    dtype: torch.dtype
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params_millions": self.total_params / 1e6,
            "gflops": self.total_gflops,
            "spatial_compression": self.spatial_compression,
            "latent_channels": self.latent_channels,
            "data_compression": self.data_compression,
            "latent_shape": list(self.latent_shape),
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.name}: {self.total_params/1e6:.2f}M params, "
            f"{self.total_gflops:.2f} GFLOPs, {self.spatial_compression}x spatial, "
            f"{self.latent_channels} latent ch"
        )


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def calculate_flops(model: nn.Module, input_tensor: torch.Tensor, forward_fn=None) -> float:
    """Calculate FLOPs using PyTorch's FlopCounterMode."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as counter:
            if forward_fn:
                forward_fn(input_tensor)
            else:
                model(input_tensor)
        return float(counter.get_total_flops())
    except ImportError:
        return 0.0


def get_vae_statistics(
    model_name: str,
    input_size: Tuple[int, int, int] = (3, 256, 256),
    device: str = "cpu",
) -> VAEStatistics:
    """Calculate comprehensive statistics for a VAE model."""
    cfg = get_config()
    if model_name not in cfg.vaes:
        raise ValueError(f"Unknown model: {model_name}")
    
    vae = load_vae(model_name, device=device, dtype=torch.float32)
    model = vae.model
    model.eval()
    
    model_dtype = next(model.parameters()).dtype
    c, h, w = input_size
    dummy_input = torch.randn(1, c, h, w, device=device, dtype=model_dtype)
    
    # Get latent shape
    with torch.no_grad():
        latent = vae.encode(dummy_input)
    latent_c, latent_h, latent_w = latent.shape[1], latent.shape[2], latent.shape[3]
    
    # Count parameters
    total_params = count_params(model)
    encoder_params = count_params(model.encoder) if hasattr(model, 'encoder') else 0
    decoder_params = count_params(model.decoder) if hasattr(model, 'decoder') else 0
    
    # Calculate FLOPs
    encode_flops = calculate_flops(model, dummy_input, lambda x: vae.encode(x))
    decode_flops = calculate_flops(
        model,
        torch.randn(1, latent_c, latent_h, latent_w, device=device, dtype=model_dtype),
        lambda z: vae.decode(z)
    )
    
    # Compression ratios
    spatial_compression = h // latent_h
    data_compression = (c * h * w) / (latent_c * latent_h * latent_w)
    
    stats = VAEStatistics(
        name=model_name,
        total_params=total_params,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        encode_gflops=encode_flops / 1e9,
        decode_gflops=decode_flops / 1e9,
        total_gflops=(encode_flops + decode_flops) / 1e9,
        spatial_compression=spatial_compression,
        latent_channels=latent_c,
        data_compression=data_compression,
        input_shape=input_size,
        latent_shape=(latent_c, latent_h, latent_w),
        dtype=model_dtype,
    )
    
    del vae
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    return stats


def get_all_vae_statistics(
    input_size: Tuple[int, int, int] = (3, 256, 256),
    device: str = "cpu",
) -> Dict[str, VAEStatistics]:
    """Calculate statistics for all VAE models."""
    cfg = get_config()
    stats = {}
    for name in cfg.vaes:
        print(f"Calculating statistics for {name}...")
        try:
            stats[name] = get_vae_statistics(name, input_size, device)
            print(f"  {stats[name]}")
        except Exception as e:
            print(f"  Failed: {e}")
    return stats


def print_statistics_table(stats: Dict[str, VAEStatistics]):
    """Print VAE statistics as formatted table."""
    print("\n" + "="*100)
    print("VAE MODEL STATISTICS")
    print("="*100)
    print(f"{'Model':<12} {'Params (M)':>12} {'GFLOPs':>10} {'Spatial':>8} {'Latent Ch':>10} {'Compression':>12}")
    print("-"*100)
    
    for name, s in stats.items():
        print(f"{name:<12} {s.total_params/1e6:>12.2f} {s.total_gflops:>10.2f} "
              f"{s.spatial_compression:>7}x {s.latent_channels:>10} {s.data_compression:>11.2f}x")
    print("="*100)
