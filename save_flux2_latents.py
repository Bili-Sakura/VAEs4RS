"""
Script to save FLUX2-VAE latents for both datasets.
Saves latents to /data/projects/VAEs4RS/datasets/BiliSakura/RS-Dataset-Latents/{dataset_name}/
using the same filename as the input (with .npz extension, compressed format with float16).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import EvalConfig
from evaluate import evaluate_all
from utils import set_seed

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    config = EvalConfig(
        batch_size=64,
        image_size=None,  # Use original image sizes
        output_dir="datasets/BiliSakura/VAEs4RS",
        device="cuda",
        seed=42,
    )
    
    print("="*80)
    print("Saving FLUX2-VAE Latents")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch size:  {config.batch_size}")
    print(f"  Image size:  Original sizes")
    print(f"  Device:      {config.device}")
    print(f"  Model:       FLUX2-VAE")
    print(f"  Latent output: /data/projects/VAEs4RS/datasets/BiliSakura/RS-Dataset-Latents/{{dataset_name}}/")
    print()
    
    # Create latent output directory
    latent_base_dir = Path("/data/projects/VAEs4RS/datasets/BiliSakura/RS-Dataset-Latents")
    latent_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created base directory: {latent_base_dir}")
    
    # Evaluate only FLUX2-VAE with latents saving enabled
    results = evaluate_all(
        config,
        dataset_classes=None,  # All classes
        results_dir=None,  # Don't save results/metrics, only latents
        skip_existing=False,
        save_images=False,  # Don't save images, only latents
        save_latents=True,  # Save latents
        model_names=["FLUX2-VAE"],  # Only FLUX2-VAE
        use_existing_images=False,
    )
    
    print("\n" + "="*80)
    print("LATENT SAVING COMPLETED")
    print("="*80)
    print(f"\nLatents saved to:")
    for dataset_name in ["RESISC45", "AID"]:
        dataset_latent_dir = latent_base_dir / dataset_name
        if dataset_latent_dir.exists():
            num_files = len(list(dataset_latent_dir.glob("*.npz")))
            total_size = sum(f.stat().st_size for f in dataset_latent_dir.glob("*.npz"))
            print(f"  {dataset_name}: {dataset_latent_dir} ({num_files} files, {total_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
