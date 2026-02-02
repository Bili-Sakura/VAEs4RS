"""
Ablation study: VAEs as pre-processors for denoising and de-hazing.

Tests whether VAE reconstruction can clean noisy or hazy remote sensing images.
"""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Try relative imports first (when used as a package), fall back to absolute (when imported directly)
try:
    from .config import VAE_CONFIGS, DATASET_CONFIGS, EvalConfig
    from .models import load_vae, VAEWrapper
    from .datasets import load_dataset
    from .metrics import MetricCalculator, MetricResults
except ImportError:
    # Fall back to absolute imports when src is in path
    from config import VAE_CONFIGS, DATASET_CONFIGS, EvalConfig
    from models import load_vae, VAEWrapper
    from datasets import load_dataset
    from metrics import MetricCalculator, MetricResults


# =============================================================================
# Distortion Functions
# =============================================================================

def add_gaussian_noise(
    images: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    
    Args:
        images: Input images, shape (B, 3, H, W), range [-1, 1]
        sigma: Standard deviation of the noise
        
    Returns:
        Noisy images, same shape and range
    """
    noise = torch.randn_like(images) * sigma
    noisy = images + noise
    return noisy.clamp(-1, 1)


def add_haze(
    images: torch.Tensor,
    intensity: float = 0.5,
    atmospheric_light: float = 0.8,
) -> torch.Tensor:
    """
    Add synthetic haze to images using atmospheric scattering model.
    
    Hazy = I * t + A * (1 - t), where t is transmission
    
    Args:
        images: Input images, shape (B, 3, H, W), range [-1, 1]
        intensity: Haze intensity (0 = no haze, 1 = full haze)
        atmospheric_light: Atmospheric light value (normalized)
        
    Returns:
        Hazy images, same shape and range
    """
    # Convert to [0, 1] for haze model
    images_01 = (images + 1) / 2
    
    # Transmission map (uniform for simplicity)
    transmission = 1 - intensity
    
    # Atmospheric light
    A = atmospheric_light
    
    # Apply haze model
    hazy = images_01 * transmission + A * (1 - transmission)
    
    # Convert back to [-1, 1]
    hazy = hazy * 2 - 1
    return hazy.clamp(-1, 1)


def add_jpeg_compression(
    images: torch.Tensor,
    quality: int = 20,
) -> torch.Tensor:
    """
    Simulate JPEG compression artifacts.
    
    Args:
        images: Input images, shape (B, 3, H, W), range [-1, 1]
        quality: JPEG quality (1-100, lower = more compression)
        
    Returns:
        Compressed images, same shape and range
    """
    from PIL import Image
    from io import BytesIO
    
    batch_size = images.shape[0]
    compressed = torch.zeros_like(images)
    
    for i in range(batch_size):
        # Convert to PIL
        img = images[i].cpu().numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        pil_img = Image.fromarray(img)
        
        # Compress and decompress
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # Convert back to tensor
        img_arr = np.array(compressed_img).astype(np.float32)
        img_arr = img_arr / 255.0 * 2 - 1
        compressed[i] = torch.from_numpy(np.transpose(img_arr, (2, 0, 1)))
    
    return compressed.to(images.device)


# =============================================================================
# Ablation Study Functions
# =============================================================================

DISTORTION_FUNCTIONS = {
    "noise_0.1": lambda x: add_gaussian_noise(x, sigma=0.1),
    "noise_0.2": lambda x: add_gaussian_noise(x, sigma=0.2),
    "noise_0.3": lambda x: add_gaussian_noise(x, sigma=0.3),
    "haze_0.3": lambda x: add_haze(x, intensity=0.3),
    "haze_0.5": lambda x: add_haze(x, intensity=0.5),
    "haze_0.7": lambda x: add_haze(x, intensity=0.7),
    "jpeg_20": lambda x: add_jpeg_compression(x, quality=20),
    "jpeg_10": lambda x: add_jpeg_compression(x, quality=10),
}


def evaluate_denoising(
    vae: VAEWrapper,
    dataset_name: str,
    distortion_name: str,
    config: EvalConfig,
) -> Tuple[MetricResults, MetricResults]:
    """
    Evaluate VAE as a denoising/de-hazing pre-processor.
    
    Args:
        vae: VAE model wrapper
        dataset_name: Name of the dataset
        distortion_name: Name of the distortion to apply
        config: Evaluation configuration
        
    Returns:
        Tuple of (distorted_metrics, cleaned_metrics)
        - distorted_metrics: Metrics comparing distorted to original
        - cleaned_metrics: Metrics comparing VAE-cleaned to original
    """
    dataset, dataloader = load_dataset(
        dataset_name,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    
    distort_fn = DISTORTION_FUNCTIONS[distortion_name]
    
    distorted_calc = MetricCalculator(device=config.device, compute_fid=False)
    cleaned_calc = MetricCalculator(device=config.device, compute_fid=False)
    
    # Use model's dtype (models use float16/bfloat16)
    model_dtype = next(vae.model.parameters()).dtype
    
    for images, labels, paths in tqdm(dataloader, desc=f"Evaluating {distortion_name}"):
        images = images.to(config.device, dtype=model_dtype)
        
        # Apply distortion
        distorted = distort_fn(images)
        
        # Clean with VAE
        cleaned = vae.reconstruct(distorted)
        
        # Update metrics
        distorted_calc.update(images.float(), distorted.float())
        cleaned_calc.update(images.float(), cleaned.float())
    
    return distorted_calc.compute(), cleaned_calc.compute()


def run_ablation_study(
    model_names: List[str],
    dataset_name: str,
    distortion_names: List[str],
    config: EvalConfig,
) -> dict:
    """
    Run full ablation study across models and distortions.
    
    Args:
        model_names: List of VAE model names
        dataset_name: Name of the dataset
        distortion_names: List of distortion names
        config: Evaluation configuration
        
    Returns:
        Dictionary with all results
    """
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        print('='*60)
        
        try:
            vae = load_vae(model_name, device=config.device)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
        
        results[model_name] = {}
        
        for distortion_name in distortion_names:
            print(f"\n{model_name} / {distortion_name}:")
            
            distorted_metrics, cleaned_metrics = evaluate_denoising(
                vae, dataset_name, distortion_name, config
            )
            
            results[model_name][distortion_name] = {
                "distorted": distorted_metrics.to_dict(),
                "cleaned": cleaned_metrics.to_dict(),
                "improvement": {
                    "psnr": cleaned_metrics.psnr - distorted_metrics.psnr,
                    "ssim": cleaned_metrics.ssim - distorted_metrics.ssim,
                    "lpips": distorted_metrics.lpips - cleaned_metrics.lpips,  # Lower is better
                },
            }
            
            print(f"  Distorted: {distorted_metrics}")
            print(f"  Cleaned:   {cleaned_metrics}")
            print(f"  PSNR Improvement: {results[model_name][distortion_name]['improvement']['psnr']:.2f} dB")
        
        # Clear GPU memory
        del vae
        torch.cuda.empty_cache()
    
    return results


def print_ablation_table(results: dict):
    """Print ablation results as a formatted table."""
    print("\n" + "="*100)
    print("ABLATION STUDY: VAE as Pre-processor")
    print("="*100)
    
    print(f"{'Model':<12} {'Distortion':<12} {'Distorted PSNR':>14} {'Cleaned PSNR':>13} {'Improvement':>12}")
    print("-"*100)
    
    for model_name, distortions in results.items():
        for distortion_name, metrics in distortions.items():
            dist_psnr = metrics['distorted']['psnr']
            clean_psnr = metrics['cleaned']['psnr']
            improvement = metrics['improvement']['psnr']
            print(
                f"{model_name:<12} {distortion_name:<12} "
                f"{dist_psnr:>14.2f} {clean_psnr:>13.2f} {improvement:>+12.2f}"
            )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ablation Study: VAE as Pre-processor")
    parser.add_argument("--dataset", type=str, default="RESISC45", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    config = EvalConfig(
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Run ablation study
    results = run_ablation_study(
        model_names=["SD21-VAE", "SDXL-VAE", "SD35-VAE", "FLUX1-VAE"],
        dataset_name=args.dataset,
        distortion_names=["noise_0.1", "noise_0.2", "haze_0.3", "haze_0.5"],
        config=config,
    )
    
    print_ablation_table(results)
