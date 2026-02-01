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
from typing import Optional

import torch
from tqdm import tqdm

from config import VAE_CONFIGS, DATASET_CONFIGS, EvalConfig
from models import load_vae, VAEWrapper
from datasets import load_dataset
from metrics import MetricCalculator, MetricResults
from utils import save_tensor_as_image


def evaluate_single(
    vae: VAEWrapper,
    dataset_name: str,
    config: EvalConfig,
    classes: Optional[list[str]] = None,
    original_images_dir: Optional[Path] = None,
    reconstructed_images_dir: Optional[Path] = None,
) -> MetricResults:
    """
    Evaluate a single VAE on a single dataset.
    
    Args:
        vae: VAE model wrapper
        dataset_name: Name of the dataset
        config: Evaluation configuration
        classes: Optional list of class names to filter (None = all classes)
        original_images_dir: Optional directory to save original images (shared across models).
                             If None, original images are not saved.
        reconstructed_images_dir: Optional directory to save reconstructed images (per model).
                                  If None, reconstructed images are not saved.
        
    Returns:
        MetricResults
    """
    dataset, dataloader = load_dataset(
        dataset_name,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        classes=classes,
    )
    
    calculator = MetricCalculator(device=config.device, compute_fid=True)
    
    # Create image directories if saving images
    if original_images_dir is not None:
        original_images_dir.mkdir(parents=True, exist_ok=True)
    if reconstructed_images_dir is not None:
        reconstructed_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input dtype based on model dtype (models use float16/bfloat16)
    model_dtype = next(vae.model.parameters()).dtype
    
    for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader, desc=f"Evaluating")):
        images = images.to(config.device, dtype=model_dtype)
        reconstructed = vae.reconstruct(images)
        calculator.update(images.float(), reconstructed.float())
        
        # Save images if requested
        if original_images_dir is not None or reconstructed_images_dir is not None:
            images_cpu = images.float().cpu()
            reconstructed_cpu = reconstructed.float().cpu()
            
            for i in range(images_cpu.shape[0]):
                # Extract filename from path
                if paths and i < len(paths):
                    # Get the base filename from the original path
                    original_path = Path(paths[i])
                    filename = original_path.stem  # filename without extension
                    # Use batch index and sample index to ensure unique names
                    save_filename = f"{filename}_batch{batch_idx:04d}_idx{i:03d}.png"
                else:
                    # Fallback if paths not available
                    save_filename = f"batch{batch_idx:04d}_idx{i:03d}.png"
                
                # Save original image (only if directory provided and file doesn't exist)
                if original_images_dir is not None:
                    original_path = original_images_dir / save_filename
                    if not original_path.exists():
                        save_tensor_as_image(images_cpu[i], str(original_path), normalize=True)
                
                # Save reconstructed image
                if reconstructed_images_dir is not None:
                    reconstructed_path = reconstructed_images_dir / save_filename
                    save_tensor_as_image(reconstructed_cpu[i], str(reconstructed_path), normalize=True)
    
    return calculator.compute()


def load_results(results_path: Path) -> dict:
    """Load existing results from JSON file."""
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}


def save_results_incremental(results: dict, results_path: Path):
    """Save results incrementally, updating existing file."""
    # Load existing results if file exists
    existing_results = load_results(results_path)
    
    # Merge new results into existing results
    for model_name, datasets in results.items():
        if model_name not in existing_results:
            existing_results[model_name] = {}
        for dataset_name, metrics in datasets.items():
            existing_results[model_name][dataset_name] = metrics
    
    # Save updated results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    # Also save individual result file for structural organization
    for model_name, datasets in results.items():
        for dataset_name, metrics in datasets.items():
            if metrics is not None:
                # Create structured directory: model_name/dataset_name.json
                model_dir = results_path.parent / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                individual_path = model_dir / f"{dataset_name}.json"
                with open(individual_path, 'w') as f:
                    json.dump(metrics, f, indent=2)


def evaluate_all(
    config: EvalConfig, 
    dataset_classes: Optional[dict[str, list[str]]] = None,
    results_dir: Optional[Path] = None,
    skip_existing: bool = False,
    save_images: bool = True,
    model_names: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate VAE models on all datasets.
    
    Args:
        config: Evaluation configuration
        dataset_classes: Optional dict mapping dataset names to lists of class names to filter
        results_dir: Optional directory to save results incrementally. If None, results are not saved incrementally.
        skip_existing: If True, skip evaluations that already have results saved.
        save_images: If True, save generated/reconstructed images.
        model_names: Optional list of model names to evaluate. If None, evaluates all models.
        
    Returns:
        Dictionary with all results
    """
    results = {}
    results_path = None
    existing_results = {}
    
    # Determine which models to evaluate
    if model_names is None:
        model_names = list(VAE_CONFIGS.keys())
    else:
        # Validate model names
        invalid_models = [m for m in model_names if m not in VAE_CONFIGS]
        if invalid_models:
            raise ValueError(f"Unknown models: {invalid_models}. Available: {list(VAE_CONFIGS.keys())}")
    
    # Load existing results if incremental saving is enabled
    if results_dir is not None:
        results_path = results_dir / "results.json"
        if skip_existing and results_path.exists():
            existing_results = load_results(results_path)
            print(f"Loaded existing results from {results_path}")
    
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
        
        for dataset_name in DATASET_CONFIGS:
            classes = dataset_classes.get(dataset_name) if dataset_classes else None
            class_info = f" (classes: {', '.join(classes)})" if classes else ""
            
            # Check if we should skip this evaluation
            if skip_existing and results_path and results_path.exists():
                if (model_name in existing_results and 
                    dataset_name in existing_results[model_name] and
                    existing_results[model_name][dataset_name] is not None):
                    print(f"\nSkipping {model_name} on {dataset_name}{class_info} (already exists)...")
                    results[model_name][dataset_name] = existing_results[model_name][dataset_name]
                    continue
            
            print(f"\nEvaluating {model_name} on {dataset_name}{class_info}...")
            
            # Set up image saving directories
            # Original images are saved once per dataset (shared across models)
            # Reconstructed images are saved per model
            original_images_dir = None
            reconstructed_images_dir = None
            if save_images and results_dir is not None:
                original_images_dir = results_dir / dataset_name / "images" / "original"
                reconstructed_images_dir = results_dir / model_name / dataset_name / "images" / "reconstructed"
            
            try:
                metrics = evaluate_single(
                    vae, 
                    dataset_name, 
                    config, 
                    classes=classes,
                    original_images_dir=original_images_dir,
                    reconstructed_images_dir=reconstructed_images_dir,
                )
                results[model_name][dataset_name] = metrics.to_dict()
                print(f"  {metrics}")
                
                if reconstructed_images_dir is not None:
                    print(f"  Reconstructed images saved to {reconstructed_images_dir}")
                if original_images_dir is not None:
                    print(f"  Original images saved to {original_images_dir} (shared across models)")
                
                # Save incrementally after each evaluation
                if results_dir is not None:
                    save_results_incremental(
                        {model_name: {dataset_name: results[model_name][dataset_name]}},
                        results_path
                    )
                    print(f"  Results saved to {results_path}")
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
    parser.add_argument("--image-size", type=str, default="original", help="Image size (default: 'original' for original sizes). Specify an integer to resize (e.g., 256).")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    # Handle image_size argument (can be None for original sizes)
    if args.image_size.lower() in ['original', 'none', 'null']:
        image_size = None
    else:
        try:
            image_size = int(args.image_size)
        except ValueError:
            parser.error(f"Invalid image_size: {args.image_size}. Must be an integer or 'original'/'none'.")
    
    config = EvalConfig(
        batch_size=args.batch_size,
        image_size=image_size,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    if args.all:
        results = evaluate_all(config)
        save_results(results, config.output_dir)
        print_results_table(results)
    elif args.model and args.dataset:
        vae = load_vae(args.model, device=config.device)
        # For single evaluation, use model-specific directories
        output_path = Path(args.output_dir)
        original_images_dir = output_path / args.dataset / "images" / "original"
        reconstructed_images_dir = output_path / args.model / args.dataset / "images" / "reconstructed"
        metrics = evaluate_single(
            vae, 
            args.dataset, 
            config,
            original_images_dir=original_images_dir,
            reconstructed_images_dir=reconstructed_images_dir,
        )
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
