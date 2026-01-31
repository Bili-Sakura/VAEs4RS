"""
Main evaluation script for VAE reconstruction quality on remote sensing datasets.

Usage:
    python evaluate.py --model SD21-VAE --dataset RESISC45
    python evaluate.py --all  # Evaluate all models on all datasets
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from config import VAE_CONFIGS, DATASET_CONFIGS, EvalConfig
from models import load_vae, VAEWrapper
from datasets import load_dataset
from metrics import MetricCalculator, MetricResults


def evaluate_single(
    vae: VAEWrapper,
    dataset_name: str,
    config: EvalConfig,
) -> MetricResults:
    """
    Evaluate a single VAE on a single dataset.
    
    Args:
        vae: VAE model wrapper
        dataset_name: Name of the dataset
        config: Evaluation configuration
        
    Returns:
        MetricResults
    """
    dataset, dataloader = load_dataset(
        dataset_name,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    
    calculator = MetricCalculator(device=config.device, compute_fid=True)
    
    for images, labels, paths in tqdm(dataloader, desc=f"Evaluating"):
        images = images.to(config.device, dtype=torch.float16)
        reconstructed = vae.reconstruct(images)
        calculator.update(images.float(), reconstructed.float())
    
    return calculator.compute()


def evaluate_all(config: EvalConfig) -> dict:
    """
    Evaluate all VAE models on all datasets.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Dictionary with all results
    """
    results = {}
    
    for model_name in VAE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        print('='*60)
        
        try:
            vae = load_vae(model_name, device=config.device)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
        
        results[model_name] = {}
        
        for dataset_name in DATASET_CONFIGS:
            print(f"\nEvaluating {model_name} on {dataset_name}...")
            
            try:
                metrics = evaluate_single(vae, dataset_name, config)
                results[model_name][dataset_name] = metrics.to_dict()
                print(f"  {metrics}")
            except Exception as e:
                print(f"  Failed: {e}")
                results[model_name][dataset_name] = None
        
        # Clear GPU memory
        del vae
        torch.cuda.empty_cache()
    
    return results


def save_results(results: dict, output_dir: str):
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return output_path


def print_results_table(results: dict):
    """Print results as a formatted table."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Dataset':<12} {'Model':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FID':>8}")
    print("-"*80)
    
    for model_name, datasets in results.items():
        for dataset_name, metrics in datasets.items():
            if metrics is None:
                continue
            print(
                f"{dataset_name:<12} {model_name:<12} "
                f"{metrics['psnr']:>8.2f} {metrics['ssim']:>8.4f} "
                f"{metrics['lpips']:>8.4f} {metrics['fid']:>8.2f}"
            )
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAEs on Remote Sensing Datasets")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., SD21-VAE)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g., RESISC45)")
    parser.add_argument("--all", action="store_true", help="Evaluate all models on all datasets")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    config = EvalConfig(
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    if args.all:
        results = evaluate_all(config)
        save_results(results, config.output_dir)
        print_results_table(results)
    elif args.model and args.dataset:
        vae = load_vae(args.model, device=config.device)
        metrics = evaluate_single(vae, args.dataset, config)
        print(f"\n{args.model} on {args.dataset}: {metrics}")
        
        results = {args.model: {args.dataset: metrics.to_dict()}}
        save_results(results, config.output_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python evaluate.py --model SD21-VAE --dataset RESISC45")
        print("  python evaluate.py --all")


if __name__ == "__main__":
    main()
