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
from typing import Optional, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import VAE_CONFIGS, DATASET_CONFIGS, EvalConfig
from evaluate import evaluate_all, print_results_table
from ablation import run_ablation_study, print_ablation_table
from visualize import visualize_reconstructions
from utils import set_seed


def get_results_dir(output_dir: str) -> Path:
    """
    Get the results directory (single folder, no timestamps).
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Path to the results directory
    """
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_results_with_metadata(
    results: dict,
    output_dir: str,
    config: EvalConfig,
    dataset_classes: Optional[Dict[str, List[str]]] = None,
    results_dir: Optional[Path] = None,
    model_names: Optional[List[str]] = None,
) -> str:
    """
    Save evaluation results with structured layout and metadata.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Base output directory
        config: Evaluation configuration
        dataset_classes: Optional dict mapping dataset names to filtered classes
        results_dir: Optional existing results directory. If None, uses output_dir directly.
        model_names: Optional list of model names evaluated. If None, uses all models from results.
        
    Returns:
        Path to the saved results directory
    """
    # Use provided results_dir or get/create one
    if results_dir is None:
        results_dir = get_results_dir(output_dir)
    else:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results JSON (merge with existing results to preserve other results)
    results_path = results_dir / "results.json"
    # Load existing results if file exists
    existing_results = {}
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing results from {results_path}: {e}")
            existing_results = {}
    
    # Merge new results into existing results
    for model_name, datasets in results.items():
        if model_name not in existing_results:
            existing_results[model_name] = {}
        for dataset_name, metrics in datasets.items():
            # Only update if metrics is not None (preserve existing non-null results)
            if metrics is not None:
                existing_results[model_name][dataset_name] = metrics
    
    # Save merged results
    with open(results_path, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    # Create metadata
    metadata = {
        "experiment_info": {
            "experiment_type": "VAE Reconstruction Quality Evaluation",
            "description": "Zero-shot evaluation of various VAE models on remote sensing datasets",
        },
        "configuration": {
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "device": config.device,
            "seed": config.seed,
            "num_workers": config.num_workers,
        },
        "models_evaluated": model_names if model_names else list(results.keys()),
        "datasets_evaluated": {},
        "class_filtering": dataset_classes if dataset_classes else None,
    }
    
    # Add dataset information
    for dataset_name in DATASET_CONFIGS:
        dataset_config = DATASET_CONFIGS[dataset_name]
        filtered_classes = dataset_classes.get(dataset_name) if dataset_classes else None
        metadata["datasets_evaluated"][dataset_name] = {
            "root_path": dataset_config.root,
            "num_classes": dataset_config.num_classes,
            "image_size": dataset_config.image_size,
            "filtered_classes": filtered_classes,
            "note": f"Evaluated {'only ' + ', '.join(filtered_classes) if filtered_classes else 'all classes'}"
        }
    
    # Add model configurations
    metadata["model_configurations"] = {}
    models_to_include = model_names if model_names else list(results.keys())
    for model_name in models_to_include:
        if model_name not in VAE_CONFIGS:
            continue
        model_config = VAE_CONFIGS[model_name]
        metadata["model_configurations"][model_name] = {
            "pretrained_path": model_config.pretrained_path,
            "subfolder": model_config.subfolder,
            "scaling_factor": model_config.scaling_factor,
            "latent_channels": model_config.latent_channels,
        }
    
    # Save metadata (merge with existing instead of overwrite)
    metadata_path = results_dir / "metadata.json"
    existing_metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing metadata from {metadata_path}: {e}")
            existing_metadata = {}

    merged_metadata = existing_metadata if isinstance(existing_metadata, dict) else {}
    merged_metadata["experiment_info"] = metadata["experiment_info"]
    merged_metadata["configuration"] = metadata["configuration"]
    merged_metadata["class_filtering"] = metadata["class_filtering"]

    # Merge models evaluated (preserve existing ordering)
    existing_models = merged_metadata.get("models_evaluated")
    if not isinstance(existing_models, list):
        existing_models = []
    for model_name in metadata["models_evaluated"]:
        if model_name not in existing_models:
            existing_models.append(model_name)
    merged_metadata["models_evaluated"] = existing_models

    # Merge datasets and model configurations
    merged_metadata.setdefault("datasets_evaluated", {})
    if not isinstance(merged_metadata["datasets_evaluated"], dict):
        merged_metadata["datasets_evaluated"] = {}
    merged_metadata["datasets_evaluated"].update(metadata["datasets_evaluated"])

    merged_metadata.setdefault("model_configurations", {})
    if not isinstance(merged_metadata["model_configurations"], dict):
        merged_metadata["model_configurations"] = {}
    merged_metadata["model_configurations"].update(metadata["model_configurations"])

    with open(metadata_path, 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    
    # Create README with instructions
    readme_content = f"""# VAE Evaluation Results

## Experiment Information

- **Experiment Type**: VAE Reconstruction Quality Evaluation
- **Description**: Zero-shot evaluation of various VAE models on remote sensing datasets

## How These Results Were Generated

### Configuration
- **Batch Size**: {config.batch_size}
- **Image Size**: {config.image_size}
- **Device**: {config.device}
- **Random Seed**: {config.seed}
- **Number of Workers**: {config.num_workers}

### Models Evaluated
{chr(10).join(f"- **{name}**: {VAE_CONFIGS[name].pretrained_path}" + (f" (subfolder: {VAE_CONFIGS[name].subfolder})" if VAE_CONFIGS[name].subfolder else "") for name in (model_names if model_names else list(results.keys())) if name in VAE_CONFIGS)}

### Datasets Evaluated
{chr(10).join(f"- **{name}**: {cfg.root}{chr(10)}  - Classes: {'Only ' + ', '.join((dataset_classes or {}).get(name, [])) if dataset_classes and dataset_classes.get(name) else 'All classes'}{chr(10)}  - Total classes in dataset: {cfg.num_classes}" for name, cfg in DATASET_CONFIGS.items())}

### Metrics Computed
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better, measured in dB
- **SSIM** (Structural Similarity Index): Higher is better, range [0, 1]
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better, range [0, 1]
- **FID** (Fréchet Inception Distance): Lower is better

### File Structure
- `results.json`: Complete evaluation results with all metrics
- `metadata.json`: Detailed metadata about the experiment configuration
- `README.md`: This file
- `{model_name}/{dataset_name}.json`: Individual result file for each model-dataset combination
- `{dataset_name}/images/original/`: Directory containing original input images (shared across all models)
- `{model_name}/{dataset_name}/images/reconstructed/`: Directory containing VAE reconstructed images (per model)
- `{model_name}/{dataset_name}/images/latent/`: Directory containing latent representations as .npy files (per model, if --save-latents flag is used)

### Usage
To load and analyze these results:

```python
import json
from pathlib import Path

# Load results
results_path = Path("{results_dir}") / "results.json"
with open(results_path, 'r') as f:
    results = json.load(f)

# Access results for a specific model and dataset
model_name = "SD21-VAE"
dataset_name = "RESISC45"
metrics = results[model_name][dataset_name]
print(f"PSNR: {{metrics['psnr']:.2f}} dB")
print(f"SSIM: {{metrics['ssim']:.4f}}")
print(f"LPIPS: {{metrics['lpips']:.4f}}")
print(f"FID: {{metrics['fid']:.2f}}")
```

### Notes
- Image size: {'Original sizes (no resizing)' if config.image_size is None else f'Resized to {config.image_size}x{config.image_size}'}
- Results are averaged across all samples in the filtered dataset
- Class filtering was applied: {('Yes' if dataset_classes else 'No')}
"""
    
    readme_path = results_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"  - Results: {results_path}")
    print(f"  - Metadata: {metadata_path}")
    print(f"  - README: {readme_path}")
    
    return str(results_dir)


def run_main_experiment(
    config: EvalConfig, 
    dataset_classes: Optional[Dict[str, List[str]]] = None,
    skip_existing: bool = False,
    save_images: bool = True,
    save_latents: bool = False,
    model_names: Optional[List[str]] = None,
    use_existing_images: bool = False,
    dataset_split_files: Optional[Dict[str, str]] = None,
    dataset_names: Optional[List[str]] = None,
) -> dict:
    """Run the main VAE reconstruction evaluation."""
    print("\n" + "="*80)
    print("MAIN EXPERIMENT: VAE Reconstruction Quality on Remote Sensing")
    print("="*80 + "\n")
    
    # Get results directory (single folder, no timestamps)
    results_dir = get_results_dir(config.output_dir)
    
    # Evaluate with incremental saving
    results = evaluate_all(
        config, 
        dataset_classes=dataset_classes,
        results_dir=results_dir,
        skip_existing=skip_existing,
        save_images=save_images,
        save_latents=save_latents,
        model_names=model_names,
        use_existing_images=use_existing_images,
        dataset_split_files=dataset_split_files,
        dataset_names=dataset_names,
    )
    
    # Save final metadata (results.json already saved incrementally)
    save_results_with_metadata(results, config.output_dir, config, dataset_classes, results_dir=results_dir, model_names=model_names)
    print_results_table(results)
    
    return results


def run_ablation_experiment(config: EvalConfig, model_names: Optional[List[str]] = None) -> dict:
    """Run the ablation study on denoising and de-hazing."""
    print("\n" + "="*80)
    print("ABLATION STUDY: VAE as Pre-processor for Denoising/De-hazing")
    print("="*80 + "\n")
    
    all_results = {}
    
    for dataset_name in DATASET_CONFIGS:
        print(f"\n--- Dataset: {dataset_name} ---\n")
        
        results = run_ablation_study(
            model_names=model_names if model_names else list(VAE_CONFIGS.keys()),
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


def generate_visualizations(config: EvalConfig, model_names: Optional[List[str]] = None):
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
                model_names=model_names if model_names else list(VAE_CONFIGS.keys()),
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
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--image-size", type=str, default="original", help="Image size (default: 'original' for original sizes). Specify an integer to resize (e.g., 256).")
    parser.add_argument("--output-dir", type=str, default="datasets/BiliSakura/VAEs4RS", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--classes", type=str, nargs="+", help="Filter classes for datasets. Formats: DATASET (all classes), DATASET: (all classes), DATASET:* (all classes), or DATASET:CLASS1,CLASS2 (specific classes). Examples: AID (all), RESISC45:airport (one class), AID:Airport,Beach RESISC45:* (mixed)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip evaluations that already have results saved")
    parser.add_argument("--no-save-images", action="store_true", help="Don't save generated/reconstructed images")
    parser.add_argument("--save-latents", action="store_true", help="Save latent representations as .npy files")
    parser.add_argument("--use-existing-images", action="store_true", help="Evaluate metrics from existing reconstructed images instead of regenerating them")
    parser.add_argument("--models", type=str, nargs="+", help="Specify VAE models to evaluate (e.g., --models SD21-VAE SDXL-VAE). If not specified, all models are evaluated.")
    parser.add_argument("--datasets", type=str, nargs="+", help="Specify datasets to evaluate (e.g., --datasets UCMerced RESISC45). If not specified, all datasets are evaluated.")
    parser.add_argument("--split-file", type=str, help="Path to split file for a dataset. Format: DATASET:PATH (e.g., UCMerced:datasets/torchgeo/ucmerced/uc_merced-test.txt)")
    args = parser.parse_args()
    
    # Parse class filtering arguments
    dataset_classes = None
    if args.classes:
        dataset_classes = {}
        for class_arg in args.classes:
            if ':' not in class_arg:
                # No colon means full dataset (all classes)
                dataset_name = class_arg
                if dataset_name not in DATASET_CONFIGS:
                    parser.error(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
                dataset_classes[dataset_name] = None  # None means all classes
            else:
                dataset_name, classes_str = class_arg.split(':', 1)
                if dataset_name not in DATASET_CONFIGS:
                    parser.error(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
                # Empty string or '*' means full dataset (all classes)
                if not classes_str.strip() or classes_str.strip() == '*':
                    dataset_classes[dataset_name] = None  # None means all classes
                else:
                    classes_list = [c.strip() for c in classes_str.split(',')]
                    dataset_classes[dataset_name] = classes_list
    
    # Parse model filtering arguments
    model_names = None
    if args.models:
        model_names = args.models
        invalid_models = [m for m in model_names if m not in VAE_CONFIGS]
        if invalid_models:
            parser.error(f"Unknown models: {invalid_models}. Available: {list(VAE_CONFIGS.keys())}")
    
    # Parse dataset filtering arguments
    dataset_names = None
    if args.datasets:
        dataset_names = args.datasets
        invalid_datasets = [d for d in dataset_names if d not in DATASET_CONFIGS]
        if invalid_datasets:
            parser.error(f"Unknown datasets: {invalid_datasets}. Available: {list(DATASET_CONFIGS.keys())}")
    
    # Parse split file arguments
    dataset_split_files = None
    if args.split_file:
        dataset_split_files = {}
        if ':' not in args.split_file:
            parser.error(f"Invalid split-file format: {args.split_file}. Expected format: DATASET:PATH")
        dataset_name, split_path = args.split_file.split(':', 1)
        if dataset_name not in DATASET_CONFIGS:
            parser.error(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
        dataset_split_files[dataset_name] = split_path.strip()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Handle image_size argument (can be None for original sizes)
    if args.image_size.lower() in ['original', 'none', 'null']:
        image_size = None
    else:
        try:
            image_size = int(args.image_size)
        except ValueError:
            parser.error(f"Invalid image_size: {args.image_size}. Must be an integer or 'original'/'none'.")
    
    # Configuration
    config = EvalConfig(
        batch_size=args.batch_size,
        image_size=image_size,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    
    print("="*80)
    print("VAEs4RS: Zero-Shot VAE Study for Remote Sensing")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch size:  {config.batch_size}")
    print(f"  Image size:  {config.image_size if config.image_size else 'Original sizes'}")
    print(f"  Device:      {config.device}")
    print(f"  Output dir:  {config.output_dir}")
    print(f"  Seed:        {config.seed}")
    print(f"\nModels: {', '.join(model_names if model_names else VAE_CONFIGS.keys())}")
    print(f"Datasets: {', '.join(dataset_names if dataset_names else DATASET_CONFIGS.keys())}")
    if dataset_classes:
        print(f"\nClass Filtering:")
        for dataset_name, classes in dataset_classes.items():
            if classes is None:
                print(f"  {dataset_name}: all classes")
            else:
                print(f"  {dataset_name}: {', '.join(classes)}")
    if dataset_split_files:
        print(f"\nSplit Files:")
        for dataset_name, split_file in dataset_split_files.items():
            print(f"  {dataset_name}: {split_file}")
    
    # Run experiments
    save_images = not args.no_save_images
    save_latents = args.save_latents
    if args.main_only:
        run_main_experiment(config, dataset_classes=dataset_classes, skip_existing=args.skip_existing, save_images=save_images, save_latents=save_latents, model_names=model_names, use_existing_images=args.use_existing_images, dataset_split_files=dataset_split_files, dataset_names=dataset_names)
    elif args.ablation_only:
        run_ablation_experiment(config, model_names=model_names)
    elif args.visualize_only:
        generate_visualizations(config, model_names=model_names)
    else:
        # Run all
        run_main_experiment(config, dataset_classes=dataset_classes, skip_existing=args.skip_existing, save_images=save_images, save_latents=save_latents, model_names=model_names, use_existing_images=args.use_existing_images, dataset_split_files=dataset_split_files, dataset_names=dataset_names)
        run_ablation_experiment(config, model_names=model_names)
        generate_visualizations(config, model_names=model_names)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)


# CLI demo examples:
# 
# Quick test with specific classes (uses original image sizes by default):
# HF_HUB_CACHE=/data/projects/VAEs4RS/models/BiliSakura/VAEs python run_experiments.py --main-only --classes AID:Airport RESISC45:airport
#
# Full evaluation on all classes with original sizes:
# HF_HUB_CACHE=/data/projects/VAEs4RS/models/BiliSakura/VAEs python run_experiments.py --main-only
#
# To resize images to a fixed size (e.g., 256x256):
# HF_HUB_CACHE=/data/projects/VAEs4RS/models/BiliSakura/VAEs python run_experiments.py --main-only --image-size 256
#
# Results are saved to: datasets/BiliSakura/VAEs4RS/<timestamp>/
if __name__ == "__main__":
    main()
