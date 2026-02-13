#!/usr/bin/env python3
"""
Plot bar charts for quick VAE reconstruction results (MAE, PSNR, SSIM).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

# Data: (modality, vae, mae, psnr, ssim)
DATA = [
    ("IR", "SD21-VAE", 0.0213, 28.49, 0.7594),
    ("IR", "SDXL-VAE", 0.0208, 28.68, 0.7722),
    ("IR", "SD35-VAE", 0.0164, 31.83, 0.8858),
    ("IR", "FLUX1-VAE", 0.0148, 33.53, 0.9266),
    ("IR", "FLUX2-VAE", 0.0091, 36.80, 0.9548),
    ("IR", "SANA-VAE", 0.0256, 26.73, 0.7027),
    ("IR", "Qwen-VAE", 0.0130, 32.67, 0.8938),
    ("IR", "MOVQGAN-67M", 0.0218, 28.50, 0.7720),
    ("IR", "MOVQGAN-102M", 0.0213, 28.82, 0.7825),
    ("IR", "MOVQGAN-270M", 0.0222, 28.81, 0.7819),
    ("IR", "VQDIFFUSION-VQVAE", 0.0256, 27.40, 0.7754),
    ("IR", "IBQ-VQVAE-1024", 0.0348, 25.25, 0.6218),
    ("IR", "IBQ-VQVAE-8192", 0.0295, 26.39, 0.6836),
    ("IR", "IBQ-VQVAE-16384", 0.0269, 26.75, 0.6927),
    ("IR", "IBQ-VQVAE-262144", 0.0299, 26.96, 0.7089),
    ("EO", "SD21-VAE", 0.0102, 34.69, 0.9331),
    ("EO", "SDXL-VAE", 0.0093, 35.30, 0.9428),
    ("EO", "SD35-VAE", 0.0051, 41.45, 0.9810),
    ("EO", "FLUX1-VAE", 0.0032, 44.64, 0.9930),
    ("EO", "FLUX2-VAE", 0.0042, 43.83, 0.9903),
    ("EO", "SANA-VAE", 0.0146, 31.43, 0.8885),
    ("EO", "Qwen-VAE", 0.0049, 42.17, 0.9833),
    ("EO", "MOVQGAN-67M", 0.0114, 34.64, 0.9333),
    ("EO", "MOVQGAN-102M", 0.0106, 35.26, 0.9404),
    ("EO", "MOVQGAN-270M", 0.0107, 35.31, 0.9413),
    ("EO", "VQDIFFUSION-VQVAE", 0.0147, 31.56, 0.9161),
    ("EO", "IBQ-VQVAE-1024", 0.0279, 26.09, 0.7905),
    ("EO", "IBQ-VQVAE-8192", 0.0239, 27.36, 0.8217),
    ("EO", "IBQ-VQVAE-16384", 0.0237, 27.56, 0.8291),
    ("EO", "IBQ-VQVAE-262144", 0.0209, 28.33, 0.8559),
    ("RGB", "SD21-VAE", 0.0299, 25.64, 0.6732),
    ("RGB", "SDXL-VAE", 0.0291, 25.73, 0.6880),
    ("RGB", "SD35-VAE", 0.0228, 28.50, 0.8221),
    ("RGB", "FLUX1-VAE", 0.0196, 30.21, 0.8753),
    ("RGB", "FLUX2-VAE", 0.0145, 32.76, 0.9173),
    ("RGB", "SANA-VAE", 0.0360, 23.94, 0.6047),
    ("RGB", "Qwen-VAE", 0.0209, 28.45, 0.8231),
    ("RGB", "MOVQGAN-67M", 0.0308, 25.62, 0.6863),
    ("RGB", "MOVQGAN-102M", 0.0299, 25.89, 0.6980),
    ("RGB", "MOVQGAN-270M", 0.0303, 25.96, 0.6975),
    ("RGB", "VQDIFFUSION-VQVAE", 0.0350, 24.38, 0.6949),
    ("RGB", "IBQ-VQVAE-1024", 0.0442, 22.93, 0.5285),
    ("RGB", "IBQ-VQVAE-8192", 0.0386, 23.80, 0.5828),
    ("RGB", "IBQ-VQVAE-16384", 0.0376, 24.02, 0.5920),
    ("RGB", "IBQ-VQVAE-262144", 0.0385, 24.35, 0.6126),
    ("SAR", "SD21-VAE", 0.0057, 42.68, 0.9789),
    ("SAR", "SDXL-VAE", 0.0063, 42.10, 0.9778),
    ("SAR", "SD35-VAE", 0.0095, 39.95, 0.9823),
    ("SAR", "FLUX1-VAE", 0.0087, 40.83, 0.9904),
    ("SAR", "FLUX2-VAE", 0.0057, 43.84, 0.9938),
    ("SAR", "SANA-VAE", 0.0094, 38.27, 0.9419),
    ("SAR", "Qwen-VAE", 0.0022, 51.04, 0.9967),
    ("SAR", "MOVQGAN-67M", 0.0084, 39.56, 0.9709),
    ("SAR", "MOVQGAN-102M", 0.0077, 40.17, 0.9758),
    ("SAR", "MOVQGAN-270M", 0.0074, 40.57, 0.9748),
    ("SAR", "VQDIFFUSION-VQVAE", 0.0175, 33.28, 0.9346),
    ("SAR", "IBQ-VQVAE-1024", 0.0289, 28.38, 0.7957),
    ("SAR", "IBQ-VQVAE-8192", 0.0196, 31.97, 0.8851),
    ("SAR", "IBQ-VQVAE-16384", 0.0179, 32.69, 0.8915),
    ("SAR", "IBQ-VQVAE-262144", 0.0245, 30.47, 0.8855),
]

MODALITIES = ["IR", "EO", "RGB", "SAR"]
MODALITY_COLORS = {"IR": "#E63946", "EO": "#457B9D", "RGB": "#2A9D8F", "SAR": "#E9C46A"}
VAE_ORDER = [
    "SD21-VAE", "SDXL-VAE", "SD35-VAE", "FLUX1-VAE", "FLUX2-VAE",
    "SANA-VAE", "Qwen-VAE", "MOVQGAN-67M", "MOVQGAN-102M", "MOVQGAN-270M",
    "VQDIFFUSION-VQVAE", "IBQ-VQVAE-1024", "IBQ-VQVAE-8192",
    "IBQ-VQVAE-16384", "IBQ-VQVAE-262144",
]


def build_data_dict():
    """Build nested dict: data[modality][vae] = (mae, psnr, ssim)."""
    data = {m: {} for m in MODALITIES}
    for mod, vae, mae, psnr, ssim in DATA:
        data[mod][vae] = (mae, psnr, ssim)
    return data


def sorted_vaes_by_avg_psnr(data):
    """VAEs sorted best (highest avg PSNR) to worst."""
    avg_psnr = {
        v: np.mean([data[m].get(v, (0, 0, 0))[1] for m in MODALITIES])
        for v in VAE_ORDER
    }
    return sorted(VAE_ORDER, key=lambda v: avg_psnr[v], reverse=True)


def sorted_vaes_for_modality(data, modality, metric_idx, higher_is_better):
    """VAEs sorted best to worst for a given modality and metric."""
    vals = [(v, data[modality].get(v, (0, 0, 0))[metric_idx]) for v in VAE_ORDER]
    return [v for v, _ in sorted(vals, key=lambda x: x[1], reverse=higher_is_better)]


# Colormap: green (best) → yellow → red (worst), perceptually uniform for many bars
CMAP_RANK = plt.cm.RdYlGn_r


def vae_colors_and_order(data, metric_idx: int, higher_is_better: bool):
    """Return (colors dict, sorted_vaes best→worst) based on average rank across modalities."""
    ranks = {v: [] for v in VAE_ORDER}
    for mod in MODALITIES:
        order = sorted_vaes_for_modality(data, mod, metric_idx, higher_is_better)
        for r, v in enumerate(order):
            ranks[v].append(r)
    avg_rank = {v: np.mean(ranks[v]) for v in VAE_ORDER}
    sorted_vaes = sorted(VAE_ORDER, key=lambda v: avg_rank[v])
    n = len(VAE_ORDER)
    colors = {v: CMAP_RANK(sorted_vaes.index(v) / max(n - 1, 1)) for v in VAE_ORDER}
    return colors, sorted_vaes


def plot_metric_by_modality(pdf_dir: Path, png_dir: Path, metric_name: str, metric_idx: int,
                            ylabel: str, ylim: tuple, higher_is_better: bool):
    """Bar chart: x-axis = modality, bars = one per VAE, color = rank (green=best→red=worst)."""
    data = build_data_dict()
    x = np.arange(len(MODALITIES))
    n_vaes = len(VAE_ORDER)
    width = 0.9 / n_vaes
    vae_colors, legend_order = vae_colors_and_order(data, metric_idx, higher_is_better)

    fig, ax = plt.subplots(figsize=(12, 6))
    for mod_idx, mod in enumerate(MODALITIES):
        order = sorted_vaes_for_modality(data, mod, metric_idx, higher_is_better)
        vals = [data[mod].get(v, (0, 0, 0))[metric_idx] for v in order]
        for vae_idx, (vae, val) in enumerate(zip(order, vals)):
            offset = (vae_idx - (n_vaes - 1) / 2) * width
            ax.bar(mod_idx + offset, val, width * 0.95, color=vae_colors[vae],
                   edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Modality")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Quick Reconstruction: {metric_name} by Modality (10 images each, best→worst)")
    ax.set_xticks(x, MODALITIES)
    ax.set_ylim(ylim)
    # Legend: VAE names in 2 rows, ordered best→worst
    legend_elements = [Patch(facecolor=vae_colors[v], edgecolor="white", label=v) for v in legend_order]
    ax.legend(handles=legend_elements, loc="upper right", ncol=8, fontsize=7)
    fig.tight_layout()
    base = f"quick_recon_{metric_name.lower().replace(' ', '_')}"
    fig.savefig(png_dir / f"{base}.png", dpi=150, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{base}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_dir / (base + '.png')}, {pdf_dir / (base + '.pdf')}")


def main():
    root = Path(__file__).resolve().parent.parent
    pdf_dir = root / "manuscript" / "ICLR26_ML4RS_Workshop_Template" / "figures"
    png_dir = root / "assets"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    # Single style: modality on x-axis, bars = models, color = best (green) → worst (red)
    plot_metric_by_modality(pdf_dir, png_dir, "PSNR", 1, "PSNR ↑", (0, 55), higher_is_better=True)
    plot_metric_by_modality(pdf_dir, png_dir, "MAE", 0, "MAE ↓", (0, 0.06), higher_is_better=False)
    plot_metric_by_modality(pdf_dir, png_dir, "SSIM", 2, "SSIM ↑", (0.5, 1.02), higher_is_better=True)


if __name__ == "__main__":
    main()
