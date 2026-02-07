"""
Ablation study: VAEs as pre-processors for denoising and de-hazing.
"""

from typing import List, Tuple, Dict, Callable

import torch
import numpy as np
from tqdm import tqdm

from diffusers.training_utils import free_memory

from src.utils.config import get_config, EvalConfig
from src.models.vae_wrapper import load_vae, VAEWrapper
from src.utils.datasets import load_dataset
from src.evaluation.metrics import MetricCalculator, MetricResults


# Distortion functions
def add_gaussian_noise(images: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to images in [-1, 1] range."""
    return (images + torch.randn_like(images) * sigma).clamp(-1, 1)


def add_haze(images: torch.Tensor, intensity: float = 0.5, atmospheric_light: float = 0.8) -> torch.Tensor:
    """Add synthetic haze using atmospheric scattering model."""
    images_01 = (images + 1) / 2
    transmission = 1 - intensity
    hazy = images_01 * transmission + atmospheric_light * (1 - transmission)
    return (hazy * 2 - 1).clamp(-1, 1)


DISTORTIONS: Dict[str, Callable] = {
    "noise_0.1": lambda x: add_gaussian_noise(x, 0.1),
    "noise_0.2": lambda x: add_gaussian_noise(x, 0.2),
    "noise_0.3": lambda x: add_gaussian_noise(x, 0.3),
    "haze_0.3": lambda x: add_haze(x, 0.3),
    "haze_0.5": lambda x: add_haze(x, 0.5),
    "haze_0.7": lambda x: add_haze(x, 0.7),
}


def evaluate_denoising(
    vae: VAEWrapper,
    dataset_name: str,
    distortion_name: str,
    config: EvalConfig,
) -> Tuple[MetricResults, MetricResults]:
    """Evaluate VAE as a denoising/de-hazing pre-processor."""
    dataset, dataloader = load_dataset(
        dataset_name, config.image_size, config.batch_size, config.num_workers
    )
    
    distort_fn = DISTORTIONS[distortion_name]
    distorted_calc = MetricCalculator(device=config.device, compute_fid=False)
    cleaned_calc = MetricCalculator(device=config.device, compute_fid=False)
    
    model_dtype = next(vae.model.parameters()).dtype
    
    for images, _, _ in tqdm(dataloader, desc=f"Evaluating {distortion_name}"):
        images = images.to(config.device, dtype=model_dtype)
        distorted = distort_fn(images)
        cleaned = vae.reconstruct(distorted)
        
        distorted_calc.update(images.float(), distorted.float())
        cleaned_calc.update(images.float(), cleaned.float())
    
    return distorted_calc.compute(), cleaned_calc.compute()


def run_ablation_study(
    model_names: List[str],
    dataset_name: str,
    distortion_names: List[str],
    config: EvalConfig,
) -> Dict:
    """Run ablation study across models and distortions."""
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}\nLoading {model_name}...\n{'='*60}")
        
        try:
            vae = load_vae(model_name, device=config.device)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
        
        results[model_name] = {}
        
        try:
            for distortion_name in distortion_names:
                print(f"\n{model_name} / {distortion_name}:")
                
                distorted, cleaned = evaluate_denoising(vae, dataset_name, distortion_name, config)
                
                results[model_name][distortion_name] = {
                    "distorted": distorted.to_dict(),
                    "cleaned": cleaned.to_dict(),
                    "improvement": {
                        "psnr": cleaned.psnr - distorted.psnr,
                        "ssim": cleaned.ssim - distorted.ssim,
                        "lpips": distorted.lpips - cleaned.lpips,
                    },
                }
                
                print(f"  Distorted: {distorted}")
                print(f"  Cleaned:   {cleaned}")
                print(f"  PSNR Improvement: {results[model_name][distortion_name]['improvement']['psnr']:.2f} dB")
        finally:
            del vae
            free_memory()
    
    return results


def print_ablation_table(results: Dict):
    """Print ablation results as formatted table."""
    print("\n" + "="*100 + "\nABLATION STUDY: VAE as Pre-processor\n" + "="*100)
    print(f"{'Model':<12} {'Distortion':<12} {'Distorted PSNR':>14} {'Cleaned PSNR':>13} {'Improvement':>12}")
    print("-"*100)
    
    for model_name, distortions in results.items():
        for distortion_name, m in distortions.items():
            print(f"{model_name:<12} {distortion_name:<12} "
                  f"{m['distorted']['psnr']:>14.2f} {m['cleaned']['psnr']:>13.2f} "
                  f"{m['improvement']['psnr']:>+12.2f}")
