"""
Main evaluation script for VAE reconstruction quality on remote sensing datasets.

Usage:
    python evaluate.py --model SD21-VAE --dataset RESISC45
    python evaluate.py --all  # Evaluate all models on all datasets
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path to allow absolute imports
current_file = Path(__file__).resolve()
src_dir = current_file.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Try relative imports first (when used as a package), fall back to absolute (when imported directly)
try:
    from .config import VAE_CONFIGS, DATASET_CONFIGS, EvalConfig
    from .models import load_vae, VAEWrapper
    from .datasets import load_dataset
    from .metrics import MetricCalculator, MetricResults
    from .utils import save_tensor_as_image
except ImportError:
    # Fall back to absolute imports when src is in path
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
    latent_images_dir: Optional[Path] = None,
    skip_existing: bool = False,
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
        latent_images_dir: Optional directory to save latent representations as .npy files (per model).
                           If None, latent images are not saved.
        
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
    if latent_images_dir is not None:
        latent_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input dtype based on model dtype (models use float16/bfloat16)
    model_dtype = next(vae.model.parameters()).dtype
    
    skipped_count = 0
    saved_count = 0
    
    for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader, desc=f"Evaluating")):
        # Check which latents need to be saved (if skip_existing is enabled)
        # Do this BEFORE processing the batch to avoid unnecessary computation
        indices_to_save = None
        batch_skipped = 0
        
        if latent_images_dir is not None and skip_existing:
            indices_to_save = []
            batch_size = images.shape[0]
            for i in range(batch_size):
                if paths and i < len(paths):
                    original_path_obj = Path(paths[i])
                    latent_filename = original_path_obj.stem + ".npz"
                else:
                    latent_filename = f"batch{batch_idx:04d}_idx{i:03d}.npz"
                latent_path = latent_images_dir / latent_filename
                if not latent_path.exists():
                    indices_to_save.append(i)
                else:
                    skipped_count += 1
                    batch_skipped += 1
            
            # If all files exist, skip the entire batch (no encoding, no reconstruction)
            if len(indices_to_save) == 0:
                # Print progress for skipped batches
                if batch_skipped > 0 and (skipped_count % 1000 == 0 or batch_idx % 10 == 0):
                    tqdm.write(f"  Skipped batch {batch_idx}: {batch_skipped} files already exist (total skipped: {skipped_count})")
                continue  # Skip entire batch processing - no reconstruction, no encoding, no metrics
        
        # Process batch only if we need to (not all files exist)
        images = images.to(config.device, dtype=model_dtype)
        reconstructed = vae.reconstruct(images)
        calculator.update(images.float(), reconstructed.float())
        
        # Encode latents if needed
        if latent_images_dir is not None:
            if indices_to_save is not None:
                # Some files exist, encode all (we'll filter when saving)
                with torch.no_grad():
                    latents = vae.encode(images)
            else:
                # Not skipping existing, encode all latents
                with torch.no_grad():
                    latents = vae.encode(images)
        else:
            latents = None
        
        # Save images if requested
        if original_images_dir is not None or reconstructed_images_dir is not None or latent_images_dir is not None:
            images_cpu = images.float().cpu()
            reconstructed_cpu = reconstructed.float().cpu()
            
            for i in range(images_cpu.shape[0]):
                # Extract filename from path
                if paths and i < len(paths):
                    # Get the base filename from the original path
                    original_path = Path(paths[i])
                    filename = original_path.stem  # filename without extension
                    original_filename = original_path.name  # filename with extension
                    # Use batch index and sample index to ensure unique names
                    save_filename_base = f"{filename}_batch{batch_idx:04d}_idx{i:03d}"
                else:
                    # Fallback if paths not available
                    save_filename_base = f"batch{batch_idx:04d}_idx{i:03d}"
                    original_filename = f"batch{batch_idx:04d}_idx{i:03d}.png"
                
                # Save original image (only if directory provided and file doesn't exist)
                if original_images_dir is not None:
                    original_path = original_images_dir / f"{save_filename_base}.png"
                    if not original_path.exists():
                        save_tensor_as_image(images_cpu[i], str(original_path), normalize=True)
                
                # Save reconstructed image
                if reconstructed_images_dir is not None:
                    reconstructed_path = reconstructed_images_dir / f"{save_filename_base}.png"
                    save_tensor_as_image(reconstructed_cpu[i], str(reconstructed_path), normalize=True)
                
                # Save latent representation as compressed .npz file with float16
                if latent_images_dir is not None:
                    # Use original filename but replace extension with .npz
                    if paths and i < len(paths):
                        original_path_obj = Path(paths[i])
                        latent_filename = original_path_obj.stem + ".npz"
                    else:
                        latent_filename = f"batch{batch_idx:04d}_idx{i:03d}.npz"
                    latent_path = latent_images_dir / latent_filename
                    
                    # Save latent if we have it and it needs to be saved
                    # If indices_to_save is None (skip_existing=False), save all
                    # If indices_to_save is set (skip_existing=True), only save indices in the list
                    if latents is not None and (indices_to_save is None or i in indices_to_save):
                        # Save latent as compressed npz with float16 for maximum compression
                        # Convert to float16 for smaller file size (sufficient precision for latents)
                        latent_np = latents[i].cpu().float().numpy().astype(np.float16)
                        np.savez_compressed(str(latent_path), latent=latent_np)
                        saved_count += 1
                        # Print progress every 100 files
                        total_processed = saved_count + skipped_count
                        if total_processed % 500 == 0:
                            tqdm.write(f"  Progress: Saved {saved_count} latents, Skipped {skipped_count} existing files")
    
    # Print final summary
    if skip_existing and latent_images_dir is not None:
        print(f"\n  Final summary:")
        if skipped_count > 0:
            print(f"    Skipped {skipped_count} existing latent files")
        if saved_count > 0:
            print(f"    Saved {saved_count} new latent files")
        if skipped_count == 0 and saved_count == 0:
            print(f"    No latents processed")
    
    return calculator.compute()


def load_image_as_tensor(image_path: Path, device: str = "cuda") -> torch.Tensor:
    """
    Load an image from disk and convert to tensor in [-1, 1] range.
    
    Args:
        image_path: Path to image file
        device: Device to load tensor on
        
    Returns:
        Image tensor in shape (C, H, W) with values in [-1, 1]
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0  # [0, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img_tensor = (img_tensor - 0.5) / 0.5  # [0, 1] -> [-1, 1]
    return img_tensor.to(device)


def evaluate_from_existing_images(
    dataset_name: str,
    config: EvalConfig,
    original_images_dir: Path,
    reconstructed_images_dir: Path,
    classes: Optional[list[str]] = None,
) -> MetricResults:
    """
    Evaluate metrics using existing original and reconstructed images.
    
    Args:
        dataset_name: Name of the dataset
        config: Evaluation configuration
        original_images_dir: Directory containing original images
        reconstructed_images_dir: Directory containing reconstructed images
        classes: Optional list of class names to filter (None = all classes)
        
    Returns:
        MetricResults
    """
    calculator = MetricCalculator(device=config.device, compute_fid=True)
    
    # Get all reconstructed image files
    reconstructed_files = list(reconstructed_images_dir.glob("*.png"))
    
    if len(reconstructed_files) == 0:
        raise ValueError(
            f"No reconstructed images found in {reconstructed_images_dir}. "
            f"Make sure the images exist."
        )
    
    # Filter to only files that have matching original images
    valid_pairs = []
    for reconstructed_image_path in reconstructed_files:
        save_filename = reconstructed_image_path.name
        original_image_path = original_images_dir / save_filename
        if original_image_path.exists():
            valid_pairs.append((original_image_path, reconstructed_image_path))
    
    if len(valid_pairs) == 0:
        raise ValueError(
            f"No matching image pairs found between {original_images_dir} and {reconstructed_images_dir}. "
            f"Make sure the images exist and follow the naming convention."
        )
    
    # Process images in batches
    batch_size = config.batch_size
    matched_count = 0
    
    def pad_to_size(img, target_h, target_w):
        """Pad image to target size."""
        h, w = img.shape[1], img.shape[2]
        pad_h = target_h - h
        pad_w = target_w - w
        return torch.nn.functional.pad(
            img,
            (0, pad_w, 0, pad_h),
            mode='constant',
            value=-1.0
        )
    
    for batch_start in tqdm(range(0, len(valid_pairs), batch_size), desc="Evaluating from existing images"):
        batch_end = min(batch_start + batch_size, len(valid_pairs))
        batch_pairs = valid_pairs[batch_start:batch_end]
        
        # Load batch of images
        original_batch = []
        reconstructed_batch = []
        
        for orig_path, recon_path in batch_pairs:
            try:
                original_img = load_image_as_tensor(orig_path, config.device)
                reconstructed_img = load_image_as_tensor(recon_path, config.device)
                
                # Ensure same size (pad if necessary)
                if original_img.shape != reconstructed_img.shape:
                    max_h = max(original_img.shape[1], reconstructed_img.shape[1])
                    max_w = max(original_img.shape[2], reconstructed_img.shape[2])
                    original_img = pad_to_size(original_img, max_h, max_w)
                    reconstructed_img = pad_to_size(reconstructed_img, max_h, max_w)
                
                original_batch.append(original_img)
                reconstructed_batch.append(reconstructed_img)
            except Exception as e:
                print(f"Warning: Failed to process {orig_path.name}: {e}")
                continue
        
        if len(original_batch) == 0:
            continue
        
        # Find max dimensions in batch for padding to same size
        max_h = max(img.shape[1] for img in original_batch + reconstructed_batch)
        max_w = max(img.shape[2] for img in original_batch + reconstructed_batch)
        
        # Pad all images to same size and stack into batch
        original_batch_padded = []
        reconstructed_batch_padded = []
        for orig_img, recon_img in zip(original_batch, reconstructed_batch):
            if orig_img.shape[1] != max_h or orig_img.shape[2] != max_w:
                orig_img = pad_to_size(orig_img, max_h, max_w)
            if recon_img.shape[1] != max_h or recon_img.shape[2] != max_w:
                recon_img = pad_to_size(recon_img, max_h, max_w)
            original_batch_padded.append(orig_img)
            reconstructed_batch_padded.append(recon_img)
        
        # Stack into batch tensors
        original_batch_tensor = torch.stack(original_batch_padded)
        reconstructed_batch_tensor = torch.stack(reconstructed_batch_padded)
        
        # Update metrics with batch
        calculator.update(original_batch_tensor, reconstructed_batch_tensor)
        matched_count += len(original_batch_padded)
    
    if matched_count == 0:
        raise ValueError(
            f"No matching image pairs found between {original_images_dir} and {reconstructed_images_dir}. "
            f"Make sure the images exist and follow the naming convention."
        )
    
    print(f"Evaluated {matched_count} image pairs")
    return calculator.compute()


def load_results(results_path: Path) -> dict:
    """Load existing results from JSON file."""
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}


def sync_individual_results_to_main(results_path: Path):
    """
    Sync individual result JSON files (model_name/dataset_name.json) back into main results.json.
    This ensures that if individual files exist but main results.json is missing them, they get synced.
    """
    if not results_path.exists():
        return
    
    results_dir = results_path.parent
    existing_results = load_results(results_path)
    updated = False
    
    # Scan for individual result files: model_name/dataset_name.json
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        if model_name not in existing_results:
            existing_results[model_name] = {}
        
        # Look for dataset JSON files in this model directory
        for dataset_file in model_dir.glob("*.json"):
            dataset_name = dataset_file.stem  # filename without extension
            
            # Skip if it's not a dataset result file (e.g., metadata.json)
            if dataset_name in ["metadata", "results"]:
                continue
            
            # Load individual result file
            try:
                with open(dataset_file, 'r') as f:
                    metrics = json.load(f)
                
                # Only update if current value is None or missing
                if (dataset_name not in existing_results[model_name] or 
                    existing_results[model_name][dataset_name] is None):
                    existing_results[model_name][dataset_name] = metrics
                    updated = True
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {dataset_file}: {e}")
    
    # Save updated results if anything changed
    if updated:
        with open(results_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print(f"Synced individual result files into {results_path}")


def save_results_incremental(results: dict, results_path: Path):
    """Save results incrementally, updating existing file."""
    # Load existing results if file exists
    existing_results = load_results(results_path)
    
    # Merge new results into existing results
    for model_name, datasets in results.items():
        if model_name not in existing_results:
            existing_results[model_name] = {}
        for dataset_name, metrics in datasets.items():
            # Only update if metrics is not None (preserve existing non-null results)
            if metrics is not None:
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
    save_latents: bool = False,
    model_names: Optional[list[str]] = None,
    use_existing_images: bool = False,
) -> dict:
    """
    Evaluate VAE models on all datasets.
    
    Args:
        config: Evaluation configuration
        dataset_classes: Optional dict mapping dataset names to lists of class names to filter
        results_dir: Optional directory to save results incrementally. If None, results are not saved incrementally.
        skip_existing: If True, skip evaluations that already have results saved.
        save_images: If True, save generated/reconstructed images.
        save_latents: If True, save latent representations as .npy files.
        model_names: Optional list of model names to evaluate. If None, evaluates all models.
        use_existing_images: If True, evaluate metrics from existing reconstructed images instead of regenerating.
        
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
        # Sync individual result files into main results.json
        if results_path.exists():
            sync_individual_results_to_main(results_path)
        if skip_existing and results_path.exists():
            existing_results = load_results(results_path)
            print(f"Loaded existing results from {results_path}")
    
    for model_name in model_names:
        results[model_name] = {}
        
        # Only load VAE if not using existing images
        vae = None
        if not use_existing_images:
            print(f"\n{'='*60}")
            print(f"Loading {model_name}...")
            print('='*60)
            
            try:
                vae = load_vae(model_name, device=config.device)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        
        try:
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
                
                # Set up image directories
                original_images_dir = None
                reconstructed_images_dir = None
                latent_images_dir = None
                if results_dir is not None:
                    original_images_dir = results_dir / dataset_name / "images" / "original"
                    reconstructed_images_dir = results_dir / model_name / dataset_name / "images" / "reconstructed"
                
                # Set up latent directory (independent of results_dir, always use custom path when save_latents is True)
                if save_latents:
                    # Use custom directory for latents: /data/projects/VAEs4RS/datasets/BiliSakura/RS-Dataset-Latents/{dataset_name}/
                    latent_images_dir = Path("/data/projects/VAEs4RS/datasets/BiliSakura/RS-Dataset-Latents") / dataset_name
                
                # Check if we should use existing images
                if use_existing_images:
                    if reconstructed_images_dir is None or not reconstructed_images_dir.exists():
                        print(f"\nSkipping {model_name} on {dataset_name}{class_info} (no reconstructed images found at {reconstructed_images_dir})...")
                        results[model_name][dataset_name] = None
                        continue
                    
                    if original_images_dir is None or not original_images_dir.exists():
                        print(f"\nSkipping {model_name} on {dataset_name}{class_info} (no original images found at {original_images_dir})...")
                        results[model_name][dataset_name] = None
                        continue
                    
                    print(f"\nEvaluating {model_name} on {dataset_name}{class_info} from existing images...")
                    
                    try:
                        metrics = evaluate_from_existing_images(
                            dataset_name,
                            config,
                            original_images_dir=original_images_dir,
                            reconstructed_images_dir=reconstructed_images_dir,
                            classes=classes,
                        )
                        results[model_name][dataset_name] = metrics.to_dict()
                        print(f"  {metrics}")
                        
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
                else:
                    # Normal evaluation with VAE model
                    print(f"\nEvaluating {model_name} on {dataset_name}{class_info}...")
                    
                    # Set up image saving directories
                    original_images_dir_save = original_images_dir if save_images else None
                    reconstructed_images_dir_save = reconstructed_images_dir if save_images else None
                    latent_images_dir_save = latent_images_dir if save_latents else None
                    
                    try:
                        metrics = evaluate_single(
                            vae, 
                            dataset_name, 
                            config, 
                            classes=classes,
                            original_images_dir=original_images_dir_save,
                            reconstructed_images_dir=reconstructed_images_dir_save,
                            latent_images_dir=latent_images_dir_save,
                            skip_existing=skip_existing,
                        )
                        results[model_name][dataset_name] = metrics.to_dict()
                        print(f"  {metrics}")
                        
                        if reconstructed_images_dir_save is not None:
                            print(f"  Reconstructed images saved to {reconstructed_images_dir_save}")
                        if original_images_dir_save is not None:
                            print(f"  Original images saved to {original_images_dir_save} (shared across models)")
                        if latent_images_dir_save is not None:
                            # Count saved files (check for .npz files, not .npy)
                            num_saved = len(list(latent_images_dir_save.glob("*.npz"))) if latent_images_dir_save.exists() else 0
                            print(f"  Latent representations saved to {latent_images_dir_save} ({num_saved} files)")
                        
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
        finally:
            # Clear GPU memory after processing all datasets for this model
            if vae is not None:
                del vae
                if config.device.startswith("cuda"):
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
        try:
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
        finally:
            # Clear GPU memory
            del vae
            torch.cuda.empty_cache()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python evaluate.py --model SD21-VAE --dataset RESISC45")
        print("  python evaluate.py --all")


if __name__ == "__main__":
    main()
