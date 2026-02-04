"""
Run CMMD evaluation for RESISC45 and AID datasets on all models.

Usage:
    python run_cmmd_resisc45_aid.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import EvalConfig
from evaluate import evaluate_all, print_results_table, save_results_incremental
from pathlib import Path

def main():
    """Run CMMD evaluation for RESISC45 and AID on all models."""
    
    # Create config with CMMD enabled
    config = EvalConfig(
        batch_size=64,
        image_size=None,  # Use original image sizes
        output_dir="outputs",
        device="cuda",
        compute_cmmd=True,  # Enable CMMD
        cmmd_clip_model="/data/projects/VAEs4RS/models/BiliSakura/Remote-CLIP-ViT-L-14/transformers"  # Use local CLIP model (transformers subdirectory)
    )
    
    # Results directory
    results_dir = Path(config.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Running CMMD evaluation for RESISC45 and AID datasets on all models")
    print("="*80)
    print(f"Batch size: {config.batch_size}")
    print(f"Output directory: {results_dir}")
    print(f"CMMD CLIP model: {config.cmmd_clip_model}")
    print("="*80)
    
    # Run evaluation for RESISC45 and AID, all models
    results = evaluate_all(
        config,
        dataset_names=["RESISC45", "AID"],  # RESISC45 and AID datasets
        model_names=None,  # All models
        results_dir=results_dir,
        skip_existing=False,  # Don't skip existing (recompute CMMD)
        save_images=False,  # Don't save images (faster, just compute metrics)
        update_cmmd_only=True,  # Only update CMMD field, preserve other metrics
    )
    
    # Print results table
    print_results_table(results)
    
    # Save results (already saved incrementally during evaluation, but save final merged results)
    results_path = results_dir / "results.json"
    save_results_incremental(results, results_path, update_cmmd_only=True)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
