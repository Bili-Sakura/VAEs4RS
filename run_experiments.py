"""
Run all experiments for the VAEs4RS paper.

This script orchestrates the full evaluation pipeline:
1. Main experiment: Evaluate all VAEs on RESISC45 and AID datasets
2. Ablation study: Test VAEs as pre-processors for denoising and de-hazing
3. Generate visualizations for the paper

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --main-only        # Only main experiment
    python run_experiments.py --ablation-only    # Only ablation study
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import VAE_CONFIGS, DATASET_CONFIGS, EvalConfig
from evaluate import evaluate_all, save_results, print_results_table
from ablation import run_ablation_study, print_ablation_table
from visualize import visualize_reconstructions
from utils import set_seed


def run_main_experiment(config: EvalConfig) -> dict:
    """Run the main VAE reconstruction evaluation."""
    print("\n" + "="*80)
    print("MAIN EXPERIMENT: VAE Reconstruction Quality on Remote Sensing")
    print("="*80 + "\n")
    
    results = evaluate_all(config)
    save_results(results, config.output_dir)
    print_results_table(results)
    
    return results


def run_ablation_experiment(config: EvalConfig) -> dict:
    """Run the ablation study on denoising and de-hazing."""
    print("\n" + "="*80)
    print("ABLATION STUDY: VAE as Pre-processor for Denoising/De-hazing")
    print("="*80 + "\n")
    
    all_results = {}
    
    for dataset_name in DATASET_CONFIGS:
        print(f"\n--- Dataset: {dataset_name} ---\n")
        
        results = run_ablation_study(
            model_names=list(VAE_CONFIGS.keys()),
            dataset_name=dataset_name,
            distortion_names=["noise_0.1", "noise_0.2", "haze_0.3", "haze_0.5"],
            config=config,
        )
        
        all_results[dataset_name] = results
        print_ablation_table(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.output_dir, f"ablation_{timestamp}.json")
    os.makedirs(config.output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAblation results saved to: {output_path}")
    
    return all_results


def generate_visualizations(config: EvalConfig):
    """Generate qualitative visualizations for the paper."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    figures_dir = os.path.join(config.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Reconstruction comparison for each dataset
    for dataset_name in DATASET_CONFIGS:
        print(f"Generating reconstruction comparison for {dataset_name}...")
        try:
            visualize_reconstructions(
                model_names=list(VAE_CONFIGS.keys()),
                dataset_name=dataset_name,
                num_samples=5,
                image_size=config.image_size,
                output_path=os.path.join(figures_dir, f"reconstruction_{dataset_name}.png"),
                device=config.device,
            )
        except Exception as e:
            print(f"Failed to generate visualization for {dataset_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run VAEs4RS Experiments")
    parser.add_argument("--main-only", action="store_true", help="Only run main experiment")
    parser.add_argument("--ablation-only", action="store_true", help="Only run ablation study")
    parser.add_argument("--visualize-only", action="store_true", help="Only generate visualizations")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Configuration
    config = EvalConfig(
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    
    print("="*80)
    print("VAEs4RS: Zero-Shot VAE Study for Remote Sensing")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch size:  {config.batch_size}")
    print(f"  Image size:  {config.image_size}")
    print(f"  Device:      {config.device}")
    print(f"  Output dir:  {config.output_dir}")
    print(f"  Seed:        {config.seed}")
    print(f"\nModels: {', '.join(VAE_CONFIGS.keys())}")
    print(f"Datasets: {', '.join(DATASET_CONFIGS.keys())}")
    
    # Run experiments
    if args.main_only:
        run_main_experiment(config)
    elif args.ablation_only:
        run_ablation_experiment(config)
    elif args.visualize_only:
        generate_visualizations(config)
    else:
        # Run all
        run_main_experiment(config)
        run_ablation_experiment(config)
        generate_visualizations(config)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
