#!/usr/bin/env python3
"""
VAEs4RS: Zero-Shot VAE Study for Remote Sensing

Main entry point. All configuration is in config.yaml.

Usage:
    python run_experiments.py              # Run main evaluation
    python run_experiments.py --ablation   # Run ablation study
    python run_experiments.py --visualize  # Generate visualizations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_config
from src.evaluate import evaluate_all, print_results_table
from src.ablation import run_ablation_study, print_ablation_table
from src.visualize import visualize_reconstructions
from src.utils import set_seed


def run_main(cfg):
    """Run main VAE reconstruction evaluation."""
    print("\n" + "="*80)
    print("MAIN EXPERIMENT: VAE Reconstruction Quality on Remote Sensing")
    print("="*80 + "\n")
    
    results = evaluate_all(
        config=cfg.eval,
        models=cfg.experiment.models,
        datasets=cfg.experiment.datasets,
        class_filter=cfg.experiment.class_filter,
        split_files=cfg.experiment.split_files,
        skip_existing=cfg.eval.skip_existing,
        save_images=cfg.eval.save_images,
        save_latents=cfg.eval.save_latents,
    )
    
    print_results_table(results)
    return results


def run_ablation(cfg):
    """Run ablation study on denoising/de-hazing."""
    print("\n" + "="*80)
    print("ABLATION STUDY: VAE as Pre-processor for Denoising/De-hazing")
    print("="*80 + "\n")
    
    models = cfg.experiment.models or list(cfg.vaes.keys())
    datasets = cfg.experiment.datasets or list(cfg.datasets.keys())
    
    all_results = {}
    for dataset_name in datasets:
        print(f"\n--- Dataset: {dataset_name} ---\n")
        results = run_ablation_study(
            model_names=models,
            dataset_name=dataset_name,
            distortion_names=cfg.ablation_distortions,
            config=cfg.eval,
        )
        all_results[dataset_name] = results
        print_ablation_table(results)
    
    return all_results


def run_visualize(cfg):
    """Generate visualizations."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    models = cfg.experiment.models or list(cfg.vaes.keys())
    datasets = cfg.experiment.datasets or list(cfg.datasets.keys())
    
    for dataset_name in datasets:
        print(f"Generating reconstruction comparison for {dataset_name}...")
        try:
            visualize_reconstructions(
                model_names=models,
                dataset_name=dataset_name,
                num_samples=cfg.visualization_num_samples,
                image_size=cfg.eval.image_size,
                output_path=f"{cfg.eval.output_dir}/figures/reconstruction_{dataset_name}.png",
                device=cfg.eval.device,
            )
        except Exception as e:
            print(f"Failed: {e}")


def main():
    cfg = get_config()
    set_seed(cfg.eval.seed)
    
    # Print configuration
    print("="*80)
    print("VAEs4RS: Zero-Shot VAE Study for Remote Sensing")
    print("="*80)
    print(f"\nConfiguration (from config.yaml):")
    print(f"  Batch size:  {cfg.eval.batch_size}")
    print(f"  Image size:  {cfg.eval.image_size or 'Original sizes'}")
    print(f"  Device:      {cfg.eval.device}")
    print(f"  Output dir:  {cfg.eval.output_dir}")
    print(f"  Seed:        {cfg.eval.seed}")
    print(f"\nModels: {', '.join(cfg.experiment.models or cfg.vaes.keys())}")
    print(f"Datasets: {', '.join(cfg.experiment.datasets or cfg.datasets.keys())}")
    
    # Check command line args for mode
    run_abl = "--ablation" in sys.argv
    run_vis = "--visualize" in sys.argv
    run_main_exp = not run_abl and not run_vis
    
    if run_main_exp:
        run_main(cfg)
    
    if run_abl or cfg.ablation_enabled:
        run_ablation(cfg)
    
    if run_vis or cfg.visualization_enabled:
        run_visualize(cfg)
    
    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
