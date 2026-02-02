"""
VAE Model Statistics Calculator.

Provides utilities to calculate various VAE model statistics including:
- GFLOPs (Giga Floating Point Operations)
- Number of parameters
- Spatial compression ratio
- Latent channels
- Data compression ratio

Usage:
    from vae_statistics import get_vae_statistics, print_vae_statistics_table
    
    # Get statistics for a single VAE
    stats = get_vae_statistics("SD21-VAE", input_size=(3, 256, 256))
    
    # Get and print statistics for all VAEs
    all_stats = get_all_vae_statistics(input_size=(3, 256, 256))
    print_vae_statistics_table(all_stats)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from contextlib import nullcontext

# Try relative imports first (when used as a package), fall back to absolute (when imported directly)
try:
    from .config import VAE_CONFIGS, VAEConfig
except ImportError:
    # Fall back to absolute imports when src is in path
    from config import VAE_CONFIGS, VAEConfig


@dataclass
class VAEStatistics:
    """Statistics for a VAE model."""
    name: str
    # Parameter counts
    total_params: int
    encoder_params: int
    decoder_params: int
    trainable_params: int
    # FLOPs
    encode_gflops: float
    decode_gflops: float
    total_gflops: float  # encode + decode (full reconstruction)
    # Compression info
    spatial_compression_ratio: int  # e.g., 8 means 256x256 -> 32x32
    latent_channels: int
    data_compression_ratio: float  # input_elements / latent_elements
    # Additional info
    input_shape: Tuple[int, int, int]  # (C, H, W)
    latent_shape: Tuple[int, int, int]  # (C, H/f, W/f)
    scaling_factor: float
    dtype: torch.dtype  # Model data type (e.g., bfloat16, float16, float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "parameters": {
                "total": self.total_params,
                "encoder": self.encoder_params,
                "decoder": self.decoder_params,
                "trainable": self.trainable_params,
                "total_millions": self.total_params / 1e6,
            },
            "gflops": {
                "encode": self.encode_gflops,
                "decode": self.decode_gflops,
                "total": self.total_gflops,
            },
            "compression": {
                "spatial_compression_ratio": self.spatial_compression_ratio,
                "latent_channels": self.latent_channels,
                "data_compression_ratio": self.data_compression_ratio,
            },
            "shapes": {
                "input": list(self.input_shape),
                "latent": list(self.latent_shape),
            },
            "scaling_factor": self.scaling_factor,
            "dtype": str(self.dtype),
        }
    
    def __repr__(self) -> str:
        dtype_str = str(self.dtype).replace('torch.', '')
        return (
            f"VAEStatistics({self.name}: "
            f"{self.total_params/1e6:.2f}M params, "
            f"{self.total_gflops:.2f} GFLOPs, "
            f"spatial_compression={self.spatial_compression_ratio}x, "
            f"latent_channels={self.latent_channels}, "
            f"data_compression={self.data_compression_ratio:.2f}x, "
            f"dtype={dtype_str})"
        )


def count_module_parameters(module: nn.Module, only_trainable: bool = False) -> int:
    """
    Count parameters in a module.
    
    Args:
        module: PyTorch module
        only_trainable: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def calculate_flops_with_profiler(
    model: nn.Module,
    input_tensor: torch.Tensor,
    forward_fn: Optional[callable] = None,
) -> float:
    """
    Calculate FLOPs using PyTorch's FlopCounterMode.
    
    Args:
        model: PyTorch module
        input_tensor: Input tensor for the forward pass
        forward_fn: Optional custom forward function. If None, uses model(input_tensor).
        
    Returns:
        Total FLOPs as a float
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
        
        # FlopCounterMode no longer needs the model argument
        with FlopCounterMode(display=False) as flop_counter:
            if forward_fn is not None:
                forward_fn(input_tensor)
            else:
                model(input_tensor)
        
        total_flops = flop_counter.get_total_flops()
        return float(total_flops)
        
    except ImportError:
        # Fallback for older PyTorch versions
        return _estimate_flops_fallback(model, input_tensor, forward_fn)


def _estimate_flops_fallback(
    model: nn.Module,
    input_tensor: torch.Tensor,
    forward_fn: Optional[callable] = None,
) -> float:
    """
    Fallback FLOPs estimation using hooks.
    
    This is a simplified estimation that may not be as accurate as FlopCounterMode.
    """
    total_flops = 0
    hooks = []
    
    def count_conv_flops(module, input, output):
        nonlocal total_flops
        # FLOPs = 2 * K_h * K_w * C_in * C_out * H_out * W_out
        batch_size = input[0].shape[0]
        out_channels, in_channels = module.weight.shape[:2]
        kernel_h, kernel_w = module.kernel_size
        out_h, out_w = output.shape[2:]
        groups = module.groups
        
        flops = 2 * kernel_h * kernel_w * (in_channels // groups) * out_channels * out_h * out_w * batch_size
        total_flops += flops
    
    def count_linear_flops(module, input, output):
        nonlocal total_flops
        # FLOPs = 2 * in_features * out_features * batch_size
        batch_size = input[0].shape[0]
        flops = 2 * module.in_features * module.out_features * batch_size
        total_flops += flops
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(count_conv_flops))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(count_linear_flops))
    
    with torch.no_grad():
        if forward_fn is not None:
            forward_fn(input_tensor)
        else:
            model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    return float(total_flops)


def get_spatial_downsampling_factor(vae_wrapper, input_size: Tuple[int, int, int], device: str = "cpu") -> int:
    """
    Determine the spatial downsampling factor of a VAE encoder.
    
    Args:
        vae_wrapper: The VAE wrapper (VAEWrapper instance)
        input_size: Input size as (C, H, W)
        device: Device for computation
        
    Returns:
        Spatial downsampling factor (e.g., 8 for 256x256 -> 32x32)
    """
    # Get the actual dtype from the model
    model = vae_wrapper.model
    model_dtype = next(model.parameters()).dtype
    
    # Create dummy input with matching dtype
    c, h, w = input_size
    dummy_input = torch.randn(1, c, h, w, device=device, dtype=model_dtype)
    
    with torch.no_grad():
        # Use wrapper's encode method for unified interface
        latent = vae_wrapper.encode(dummy_input)
    
    latent_h, latent_w = latent.shape[2], latent.shape[3]
    
    # Calculate downsampling factor
    factor_h = h // latent_h
    factor_w = w // latent_w
    
    assert factor_h == factor_w, f"Non-square downsampling not supported: {factor_h}x vs {factor_w}x"
    
    return factor_h


def get_vae_statistics(
    model_name: str,
    input_size: Tuple[int, int, int] = (3, 256, 256),
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> VAEStatistics:
    """
    Calculate comprehensive statistics for a VAE model.
    
    Args:
        model_name: Name of the VAE model (from VAE_CONFIGS)
        input_size: Input image size as (C, H, W)
        device: Device for computation
        dtype: Data type for the model (default: float32 for accurate FLOPs)
        
    Returns:
        VAEStatistics object with all computed statistics
    """
    from models import load_vae
    
    if model_name not in VAE_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VAE_CONFIGS.keys())}")
    
    config = VAE_CONFIGS[model_name]
    
    # Load model - use float32 for accurate FLOPs calculation if explicitly requested
    # Otherwise let load_vae choose appropriate dtype (bfloat16/float16)
    if dtype is None:
        # Try float32 first, but load_vae will convert to bfloat16/float16
        # We'll use the actual model dtype for inputs
        vae = load_vae(model_name, device=device, dtype=torch.float32)
    else:
        vae = load_vae(model_name, device=device, dtype=dtype)
    
    model = vae.model
    if model is None:
        raise ValueError(f"Model failed to load for {model_name}")
    
    model.eval()
    
    # Get the actual dtype from the model (may differ from requested dtype)
    # Handle case where model might not have parameters (shouldn't happen, but be safe)
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        # Fallback to float32 if no parameters found (shouldn't happen for valid models)
        model_dtype = torch.float32
    
    # Create dummy input with matching dtype to avoid dtype mismatch errors
    c, h, w = input_size
    dummy_input = torch.randn(1, c, h, w, device=device, dtype=model_dtype)
    
    # Calculate spatial downsampling factor using wrapper
    spatial_factor = get_spatial_downsampling_factor(vae, input_size, device)
    
    # Get latent shape using wrapper's encode method
    with torch.no_grad():
        latent = vae.encode(dummy_input)
    latent_c, latent_h, latent_w = latent.shape[1], latent.shape[2], latent.shape[3]
    
    # Count parameters
    total_params = count_module_parameters(model)
    trainable_params = count_module_parameters(model, only_trainable=True)
    
    # Count encoder and decoder parameters separately
    encoder_params = 0
    decoder_params = 0
    
    if hasattr(model, 'encoder'):
        encoder_params = count_module_parameters(model.encoder)
    if hasattr(model, 'decoder'):
        decoder_params = count_module_parameters(model.decoder)
    
    # Also count quant_conv and post_quant_conv if present
    if hasattr(model, 'quant_conv') and model.quant_conv is not None:
        encoder_params += count_module_parameters(model.quant_conv)
    if hasattr(model, 'post_quant_conv') and model.post_quant_conv is not None:
        decoder_params += count_module_parameters(model.post_quant_conv)
    
    # Calculate encode FLOPs using wrapper's encode method
    def encode_forward(x):
        return vae.encode(x)
    
    encode_flops = calculate_flops_with_profiler(model, dummy_input, encode_forward)
    
    # Calculate decode FLOPs using wrapper's decode method
    # vae.decode() expects scaled latents (it divides by scaling_factor internally)
    # For FLOPs calculation, we use dummy latents with the same shape
    dummy_latent = torch.randn(1, latent_c, latent_h, latent_w, device=device, dtype=model_dtype)
    
    def decode_forward(z):
        return vae.decode(z)
    
    decode_flops = calculate_flops_with_profiler(model, dummy_latent, decode_forward)
    
    # Convert to GFLOPs
    encode_gflops = encode_flops / 1e9
    decode_gflops = decode_flops / 1e9
    total_gflops = encode_gflops + decode_gflops
    
    # Calculate data compression ratio
    input_elements = c * h * w
    latent_elements = latent_c * latent_h * latent_w
    data_compression_ratio = input_elements / latent_elements
    
    return VAEStatistics(
        name=model_name,
        total_params=total_params,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        trainable_params=trainable_params,
        encode_gflops=encode_gflops,
        decode_gflops=decode_gflops,
        total_gflops=total_gflops,
        spatial_compression_ratio=spatial_factor,
        latent_channels=latent_c,
        data_compression_ratio=data_compression_ratio,
        input_shape=input_size,
        latent_shape=(latent_c, latent_h, latent_w),
        scaling_factor=config.scaling_factor,
        dtype=model_dtype,
    )


def get_all_vae_statistics(
    input_size: Tuple[int, int, int] = (3, 256, 256),
    device: str = "cpu",
    model_names: Optional[list] = None,
) -> Dict[str, VAEStatistics]:
    """
    Calculate statistics for all VAE models.
    
    Args:
        input_size: Input image size as (C, H, W)
        device: Device for computation
        model_names: Optional list of model names. If None, uses all models.
        
    Returns:
        Dictionary mapping model names to VAEStatistics objects
    """
    if model_names is None:
        model_names = list(VAE_CONFIGS.keys())
    
    stats = {}
    for name in model_names:
        print(f"Calculating statistics for {name}...")
        try:
            stats[name] = get_vae_statistics(name, input_size, device)
            print(f"  {stats[name]}")
        except Exception as e:
            print(f"  Failed to calculate statistics for {name}: {e}")
    
    # Clear GPU cache after processing each model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    return stats


def print_vae_statistics_table(stats: Dict[str, VAEStatistics]):
    """
    Print VAE statistics as a formatted table.
    
    Args:
        stats: Dictionary mapping model names to VAEStatistics objects
    """
    print("\n" + "=" * 120)
    print("VAE MODEL STATISTICS")
    print("=" * 120)
    
    # Header
    print(
        f"{'Model':<12} "
        f"{'Params (M)':>12} "
        f"{'Enc (M)':>10} "
        f"{'Dec (M)':>10} "
        f"{'GFLOPs':>10} "
        f"{'Spatial':>8} "
        f"{'Latent Ch':>10} "
        f"{'Data Comp':>10} "
        f"{'Latent Shape':>16} "
        f"{'Dtype':>10}"
    )
    print("-" * 130)
    
    for name, s in stats.items():
        dtype_str = str(s.dtype).replace('torch.', '')
        print(
            f"{name:<12} "
            f"{s.total_params/1e6:>12.2f} "
            f"{s.encoder_params/1e6:>10.2f} "
            f"{s.decoder_params/1e6:>10.2f} "
            f"{s.total_gflops:>10.2f} "
            f"{s.spatial_compression_ratio:>7}x "
            f"{s.latent_channels:>10} "
            f"{s.data_compression_ratio:>9.2f}x "
            f"{str(s.latent_shape):>16} "
            f"{dtype_str:>10}"
        )
    
    print("=" * 130)
    print("\nLegend:")
    print("  - Params (M): Total parameters in millions")
    print("  - Enc (M): Encoder parameters in millions")
    print("  - Dec (M): Decoder parameters in millions")
    print("  - GFLOPs: Total GFLOPs for encode + decode (full reconstruction)")
    print("  - Spatial: Spatial compression ratio (e.g., 8x means 256x256 -> 32x32)")
    print("  - Latent Ch: Number of latent channels")
    print("  - Data Comp: Data compression ratio (input_elements / latent_elements)")
    print("  - Latent Shape: Shape of latent representation (C, H, W)")
    print("  - Dtype: Model data type (bfloat16, float16, or float32)")


def save_vae_statistics(
    stats: Dict[str, VAEStatistics],
    output_path: str,
):
    """
    Save VAE statistics to a JSON file.
    
    Args:
        stats: Dictionary mapping model names to VAEStatistics objects
        output_path: Path to the output JSON file
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    stats_dict = {name: s.to_dict() for name, s in stats.items()}
    
    with open(output_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"Statistics saved to: {output_path}")


def get_quick_statistics_from_config(
    model_name: str,
    input_size: Tuple[int, int, int] = (3, 256, 256),
) -> Dict[str, Any]:
    """
    Get quick statistics from config without loading the model.
    
    This is useful for getting basic info like latent channels and spatial
    compression ratio without the overhead of loading a model.
    
    Args:
        model_name: Name of the VAE model (from VAE_CONFIGS)
        input_size: Input image size as (C, H, W)
        
    Returns:
        Dictionary with quick statistics
    """
    if model_name not in VAE_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VAE_CONFIGS.keys())}")
    
    config = VAE_CONFIGS[model_name]
    
    c, h, w = input_size
    spatial_ratio = config.spatial_compression_ratio
    latent_channels = config.latent_channels
    
    # Calculate latent dimensions
    latent_h = h // spatial_ratio
    latent_w = w // spatial_ratio
    
    # Calculate data compression ratio
    input_elements = c * h * w
    latent_elements = latent_channels * latent_h * latent_w
    data_compression_ratio = input_elements / latent_elements
    
    return {
        "name": model_name,
        "scaling_factor": config.scaling_factor,
        "latent_channels": latent_channels,
        "spatial_compression_ratio": spatial_ratio,
        "data_compression_ratio": data_compression_ratio,
        "input_shape": list(input_size),
        "latent_shape": [latent_channels, latent_h, latent_w],
    }


def print_quick_statistics_table(
    input_size: Tuple[int, int, int] = (3, 256, 256),
    model_names: Optional[list] = None,
):
    """
    Print quick statistics table using config values only (no model loading).
    
    Args:
        input_size: Input image size as (C, H, W)
        model_names: Optional list of model names. If None, uses all models.
    """
    if model_names is None:
        model_names = list(VAE_CONFIGS.keys())
    
    print("\n" + "=" * 100)
    print("VAE QUICK STATISTICS (from config)")
    print("=" * 100)
    
    # Header
    print(
        f"{'Model':<12} "
        f"{'Scaling':>10} "
        f"{'Spatial':>8} "
        f"{'Latent Ch':>10} "
        f"{'Data Comp':>12} "
        f"{'Input Shape':>16} "
        f"{'Latent Shape':>16}"
    )
    print("-" * 100)
    
    for name in model_names:
        stats = get_quick_statistics_from_config(name, input_size)
        print(
            f"{name:<12} "
            f"{stats['scaling_factor']:>10.5f} "
            f"{stats['spatial_compression_ratio']:>7}x "
            f"{stats['latent_channels']:>10} "
            f"{stats['data_compression_ratio']:>11.2f}x "
            f"{str(tuple(stats['input_shape'])):>16} "
            f"{str(tuple(stats['latent_shape'])):>16}"
        )
    
    print("=" * 100)


def main():
    """Main function to calculate and display VAE statistics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate VAE Model Statistics")
    parser.add_argument("--model", type=str, default=None, help="Specific model name (default: all models)")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size (default: 256)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--models", type=str, nargs="+", help="List of models to analyze")
    parser.add_argument("--quick", action="store_true", help="Quick mode: show config-based stats only (no model loading)")
    args = parser.parse_args()
    
    input_size = (3, args.image_size, args.image_size)
    
    # Determine which models to process
    model_names = None
    if args.model:
        model_names = [args.model]
    elif args.models:
        model_names = args.models
    
    print(f"Calculating VAE statistics...")
    print(f"  Input size: {input_size}")
    print(f"  Device: {args.device}")
    print(f"  Models: {model_names if model_names else 'all'}")
    print(f"  Quick mode: {args.quick}")
    print()
    
    if args.quick:
        # Quick mode: use config-based stats only
        print_quick_statistics_table(input_size, model_names)
        
        # Save to file if requested
        if args.output:
            import json
            import os
            
            if model_names is None:
                model_names = list(VAE_CONFIGS.keys())
            
            stats_dict = {name: get_quick_statistics_from_config(name, input_size) for name in model_names}
            
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            print(f"\nStatistics saved to: {args.output}")
    else:
        # Full mode: load models and calculate comprehensive stats
        if model_names and len(model_names) == 1:
            stats = {model_names[0]: get_vae_statistics(model_names[0], input_size, args.device)}
        else:
            stats = get_all_vae_statistics(input_size, args.device, model_names)
        
        # Print table
        print_vae_statistics_table(stats)
        
        # Save to file if requested
        if args.output:
            save_vae_statistics(stats, args.output)


if __name__ == "__main__":
    main()
