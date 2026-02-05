"""
Main evaluation script for VAE reconstruction quality on remote sensing datasets.
"""

import gc
import json
from pathlib import Path
from typing import Optional, List, Dict

import torch
import numpy as np
from tqdm import tqdm

from .config import get_config, EvalConfig, PROJECT_ROOT
from .models import load_vae, VAEWrapper
from .datasets import load_dataset
from .metrics import MetricCalculator, MetricResults
from .utils import save_tensor_as_image


def evaluate_single(
    vae: VAEWrapper,
    dataset_name: str,
    config: EvalConfig,
    classes: Optional[List[str]] = None,
    original_dir: Optional[Path] = None,
    reconstructed_dir: Optional[Path] = None,
    latent_dir: Optional[Path] = None,
    skip_existing: bool = False,
    split_file: Optional[str] = None,
) -> MetricResults:
    """Evaluate a single VAE on a single dataset."""
    dataset, dataloader = load_dataset(
        dataset_name,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        classes=classes,
        split_file=split_file,
    )
    
    calculator = MetricCalculator(
        device=config.device,
        compute_fid=True,
        compute_cmmd=config.compute_cmmd,
        fid_feature_extractor=config.fid_feature_extractor,
        cmmd_clip_model=config.cmmd_clip_model,
        cmmd_batch_size=min(config.batch_size, config.cmmd_batch_size),
        mmd_chunk_size=config.mmd_chunk_size,
    )
    
    # Create directories
    for d in [original_dir, reconstructed_dir, latent_dir]:
        if d:
            d.mkdir(parents=True, exist_ok=True)
    
    model_dtype = next(vae.model.parameters()).dtype
    
    for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader, desc="Evaluating")):
        # Skip if all latents exist
        if latent_dir and skip_existing:
            all_exist = all(
                (latent_dir / (Path(p).stem + ".npz")).exists() 
                for p in paths if p
            )
            if all_exist:
                continue
        
        images = images.to(config.device, dtype=model_dtype)
        
        # Encode/decode
        if latent_dir:
            latents = vae.encode(images)
            reconstructed = vae.decode(latents, original_shape=images.shape)
        else:
            reconstructed = vae.reconstruct(images)
            latents = None
        
        # Update metrics
        calculator.update(images.float(), reconstructed.float())
        
        # Save outputs
        if original_dir or reconstructed_dir or latent_dir:
            images_cpu = images.float().cpu()
            recon_cpu = reconstructed.float().cpu()
            latents_cpu = latents.cpu() if latents is not None else None
            
            for i, path in enumerate(paths):
                stem = Path(path).stem if path else f"batch{batch_idx:04d}_idx{i:03d}"
                
                if original_dir:
                    out_path = original_dir / f"{stem}.png"
                    if not out_path.exists():
                        save_tensor_as_image(images_cpu[i], str(out_path))
                
                if reconstructed_dir:
                    save_tensor_as_image(recon_cpu[i], str(reconstructed_dir / f"{stem}.png"))
                
                if latent_dir and latents_cpu is not None:
                    np.savez_compressed(
                        str(latent_dir / f"{stem}.npz"),
                        latent=latents_cpu[i].float().numpy().astype(np.float16)
                    )
        
        # Memory management
        del images, reconstructed
        if latents is not None:
            del latents
        if config.device.startswith("cuda") and batch_idx % 3 == 0:
            torch.cuda.empty_cache()
    
    return calculator.compute()


def evaluate_all(
    config: Optional[EvalConfig] = None,
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    class_filter: Optional[Dict[str, List[str]]] = None,
    split_files: Optional[Dict[str, str]] = None,
    skip_existing: bool = False,
    save_images: bool = False,
    save_latents: bool = False,
) -> Dict:
    """
    Evaluate VAE models on datasets.
    Uses config.yaml if config not provided.
    """
    cfg = get_config()
    if config is None:
        config = cfg.eval
    
    # Use experiment settings from config if not overridden
    models = models or cfg.experiment.models or list(cfg.vaes.keys())
    datasets = datasets or cfg.experiment.datasets or list(cfg.datasets.keys())
    class_filter = class_filter or cfg.experiment.class_filter
    split_files = split_files or cfg.experiment.split_files
    
    results_dir = Path(config.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"
    
    # Load existing results
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
    
    results = {}
    
    for model_name in models:
        results[model_name] = {}
        
        # Clear memory
        if config.device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n{'='*60}\nLoading {model_name}...\n{'='*60}")
        
        try:
            vae = load_vae(model_name, device=config.device)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
        
        try:
            for dataset_name in datasets:
                classes = class_filter.get(dataset_name) if class_filter else None
                split_file = split_files.get(dataset_name) if split_files else None
                
                # Skip if exists
                if skip_existing and model_name in existing and dataset_name in existing[model_name]:
                    if existing[model_name][dataset_name] is not None:
                        print(f"Skipping {model_name} on {dataset_name} (exists)")
                        results[model_name][dataset_name] = existing[model_name][dataset_name]
                        continue
                
                print(f"\nEvaluating {model_name} on {dataset_name}...")
                
                # Setup directories
                orig_dir = results_dir / dataset_name / "images" / "original" if save_images else None
                recon_dir = results_dir / model_name / dataset_name / "images" / "reconstructed" if save_images else None
                latent_dir = results_dir / model_name / dataset_name / "latents" if save_latents else None
                
                try:
                    metrics = evaluate_single(
                        vae, dataset_name, config,
                        classes=classes,
                        original_dir=orig_dir,
                        reconstructed_dir=recon_dir,
                        latent_dir=latent_dir,
                        skip_existing=skip_existing,
                        split_file=split_file,
                    )
                    results[model_name][dataset_name] = metrics.to_dict()
                    print(f"  {metrics}")
                    
                    # Save incrementally
                    _save_results(results, results_path, existing)
                except Exception as e:
                    print(f"  Failed: {e}")
                    results[model_name][dataset_name] = None
        finally:
            del vae
            if config.device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()
    
    return results


def _save_results(results: Dict, path: Path, existing: Dict):
    """Save results, merging with existing."""
    for model, datasets in results.items():
        if model not in existing:
            existing[model] = {}
        for dataset, metrics in datasets.items():
            if metrics is not None:
                existing[model][dataset] = metrics
    
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2)


def print_results_table(results: Dict):
    """Print results as formatted table."""
    print("\n" + "="*80 + "\nRESULTS SUMMARY\n" + "="*80)
    
    has_cmmd = any(
        m and m.get('cmmd') is not None
        for d in results.values() for m in d.values()
    )
    
    header = f"{'Model':<12} {'Dataset':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FID':>8}"
    if has_cmmd:
        header += f" {'CMMD':>8}"
    print(header + "\n" + "-"*80)
    
    for model, datasets in results.items():
        for dataset, m in datasets.items():
            if m is None:
                continue
            fid = f"{m['fid']:>8.2f}" if m.get('fid') else "     N/A"
            line = f"{model:<12} {dataset:<12} {m['psnr']:>8.2f} {m['ssim']:>8.4f} {m['lpips']:>8.4f} {fid}"
            if has_cmmd:
                cmmd = f"{m['cmmd']:>8.2f}" if m.get('cmmd') else "     N/A"
                line += f" {cmmd}"
            print(line)
    
    print("="*80)
