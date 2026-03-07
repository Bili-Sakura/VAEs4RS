#!/usr/bin/env python3
"""
Visualize VAE latent features using t-SNE.

Encodes images from a remote sensing dataset through one or more VAE
models, randomly samples per-pixel latent vectors, and produces
density-coloured t-SNE scatter plots.

Adapted from LightningDiT (https://github.com/hustvl/LightningDiT).

Usage:
    # Single model
    python scripts/visualize_latents.py --model SD21-VAE --dataset RESISC45

    # Multiple models (comparison grid)
    python scripts/visualize_latents.py \
        --model SD21-VAE SDXL-VAE FLUX1-VAE \
        --dataset RESISC45 --sample_num 5000

    # Custom output directory
    python scripts/visualize_latents.py \
        --model SD21-VAE --dataset UCMerced \
        --output_dir outputs/latent_tsne
"""

import argparse
import json
import os
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.config import get_config
from src.utils.helpers import set_seed
from src.models.vae_wrapper import load_vae
from src.utils.datasets import load_dataset
from src.evaluation.latent_viz import (
    extract_latent_pixels,
    compute_tsne,
    calculate_uniformity_metrics,
    plot_tsne,
    plot_tsne_comparison,
)

from diffusers.training_utils import free_memory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize VAE latent space via t-SNE."
    )
    p.add_argument(
        "--model", nargs="+", default=None,
        help="VAE model name(s). Default: all configured models.",
    )
    p.add_argument(
        "--dataset", type=str, default="RESISC45",
        help="Dataset name (must match config.yaml).",
    )
    p.add_argument("--sample_num", type=int, default=10000,
                   help="Number of per-pixel latent vectors to sample.")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=None,
                   help="Resize images (default: dataset native size).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs/latent_tsne")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = get_config()
    model_names = args.model or list(cfg.vaes.keys())

    # Load dataset once
    _, dataloader = load_dataset(
        args.dataset,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    all_tsne = {}
    all_metrics = {}

    for name in model_names:
        print(f"\n{'='*60}\n{name} on {args.dataset}\n{'='*60}")
        try:
            vae = load_vae(name, device=args.device)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

        try:
            features = extract_latent_pixels(
                vae, dataloader,
                sample_num=args.sample_num,
                device=args.device,
                seed=args.seed,
            )
            print(f"  Extracted {features.shape[0]} latent vectors "
                  f"(dim={features.shape[1]})")

            tsne_results = compute_tsne(
                features,
                perplexity=args.perplexity,
                max_iter=args.max_iter,
                seed=args.seed,
            )

            metrics = calculate_uniformity_metrics(tsne_results)
            print(f"  Uniformity: entropy={metrics['normalized_entropy']:.4f}  "
                  f"gini={metrics['gini_coefficient']:.4f}")

            # Save individual plot
            fig_path = os.path.join(out_dir, f"{name}_tsne.png")
            plot_tsne(
                tsne_results,
                output_path=fig_path,
                title=f"{name} – {args.dataset}",
            )
            print(f"  Saved {fig_path}")

            all_tsne[name] = tsne_results
            all_metrics[name] = metrics
        finally:
            del vae
            free_memory()

    # Comparison figure
    if len(all_tsne) > 1:
        cmp_path = os.path.join(out_dir, "comparison_tsne.png")
        plot_tsne_comparison(
            all_tsne,
            output_path=cmp_path,
            suptitle=f"Latent t-SNE – {args.dataset}",
        )
        print(f"\nSaved comparison → {cmp_path}")

    # Save metrics JSON
    metrics_path = os.path.join(out_dir, "tsne_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    # Summary table
    print(f"\n{'Model':<14} {'Entropy':>10} {'Gini':>10} {'CV':>10}")
    print("-" * 48)
    for name, m in all_metrics.items():
        print(f"{name:<14} {m['normalized_entropy']:>10.4f} "
              f"{m['gini_coefficient']:>10.4f} {m['density_cv']:>10.4f}")


if __name__ == "__main__":
    main()
