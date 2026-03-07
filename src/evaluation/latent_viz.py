"""
Latent feature visualization using t-SNE.

Adapted from LightningDiT (https://github.com/hustvl/LightningDiT)
for remote sensing VAE latent space analysis.

Provides:
- Per-pixel latent sampling from VAE-encoded images
- t-SNE dimensionality reduction and density-coloured scatter plots
- Uniformity metrics (entropy, Gini, coefficient of variation)
- Support for comparing multiple VAE models side-by-side
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_latent_pixels(
    vae,
    dataloader,
    sample_num: int = 10000,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """Extract randomly-sampled per-pixel latent features from a VAE.

    For each image in the dataloader the VAE encoder produces a latent map
    of shape ``(C, H', W')``.  A single spatial position ``(h, w)`` is
    sampled uniformly at random and the resulting ``C``-dimensional vector
    is collected.  The process repeats until *sample_num* vectors have been
    gathered.

    Args:
        vae: A ``VAEWrapper`` (or any object whose ``.encode()`` method
            returns scaled latent tensors of shape ``(B, C, H', W')``).
        dataloader: Yields ``(images, labels, paths)`` batches.
        sample_num: Number of per-pixel feature vectors to collect.
        device: Torch device for encoding.
        seed: Random seed for reproducible spatial sampling.

    Returns:
        ``torch.Tensor`` of shape ``(sample_num, C)`` on CPU.
    """
    rng = np.random.RandomState(seed)
    model_dtype = next(vae.model.parameters()).dtype

    collected: List[torch.Tensor] = []
    remaining = sample_num

    for images, _labels, _paths in dataloader:
        if remaining <= 0:
            break

        images = images.to(device, dtype=model_dtype)
        latents = vae.encode(images)  # (B, C, H', W')

        b, _c, h, w = latents.shape
        take = min(b, remaining)

        for i in range(take):
            hi = rng.randint(0, h)
            wi = rng.randint(0, w)
            collected.append(latents[i, :, hi, wi].float().cpu())

        remaining -= take

    if not collected:
        raise RuntimeError("No latent features collected – is the dataloader empty?")

    return torch.stack(collected[:sample_num])


# ---------------------------------------------------------------------------
# Uniformity metrics  (adapted from LightningDiT)
# ---------------------------------------------------------------------------

def calculate_uniformity_metrics(tsne_results: np.ndarray) -> Dict[str, float]:
    """Calculate uniformity metrics for a 2-D t-SNE embedding.

    Args:
        tsne_results: ``(N, 2)`` array of t-SNE coordinates.

    Returns:
        Dict with keys ``density_std``, ``density_cv``,
        ``normalized_entropy``, and ``gini_coefficient``.
    """
    kde = gaussian_kde(tsne_results.T)
    density = kde(tsne_results.T)

    density_mean = np.mean(density)
    density_std = np.std(density)
    density_cv = density_std / density_mean

    # Entropy
    density_norm = density / np.sum(density)
    entropy = -np.sum(density_norm * np.log2(density_norm + 1e-10))
    max_entropy = np.log2(len(density))
    normalized_entropy = entropy / max_entropy

    # Gini coefficient
    sorted_density = np.sort(density)
    n = len(sorted_density)
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * sorted_density)) / (n * np.sum(sorted_density))

    return {
        "density_std": float(density_std),
        "density_cv": float(density_cv),
        "normalized_entropy": float(normalized_entropy),
        "gini_coefficient": float(gini),
    }


# ---------------------------------------------------------------------------
# t-SNE computation
# ---------------------------------------------------------------------------

def compute_tsne(
    features: torch.Tensor,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Run t-SNE on a feature matrix.

    Args:
        features: ``(N, C)`` tensor of feature vectors.
        perplexity: t-SNE perplexity.
        max_iter: Optimisation iterations.
        seed: Random seed.

    Returns:
        ``(N, 2)`` numpy array of 2-D coordinates.
    """
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        max_iter=max_iter,
    )
    return tsne.fit_transform(features.numpy())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tsne(
    tsne_results: np.ndarray,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    alpha: float = 0.6,
    dpi: int = 150,
) -> plt.Figure:
    """Create a density-coloured t-SNE scatter plot.

    Args:
        tsne_results: ``(N, 2)`` array.
        output_path: If given, save the figure to this path.
        title: Optional plot title.
        figsize: Figure size in inches.
        cmap: Matplotlib colour-map name.
        alpha: Point transparency.
        dpi: Output resolution.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    kde = gaussian_kde(tsne_results.T)
    density = kde(tsne_results.T)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=density,
        cmap=cmap,
        alpha=alpha,
        s=8,
        edgecolors="none",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=14)

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_tsne_comparison(
    model_results: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    figsize_per_plot: Tuple[int, int] = (6, 5),
    cmap: str = "viridis",
    alpha: float = 0.6,
    dpi: int = 150,
) -> plt.Figure:
    """Plot t-SNE embeddings for multiple VAE models in a single row.

    Args:
        model_results: ``{model_name: tsne_array}`` mapping.
        output_path: Save path.
        suptitle: Super-title for the figure.
        figsize_per_plot: Size of each subplot.
        cmap: Colour-map.
        alpha: Point transparency.
        dpi: Output resolution.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    n = len(model_results)
    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_plot[0] * n, figsize_per_plot[1]),
        squeeze=False,
    )

    for idx, (name, tsne_arr) in enumerate(model_results.items()):
        ax = axes[0, idx]
        kde = gaussian_kde(tsne_arr.T)
        density = kde(tsne_arr.T)
        ax.scatter(
            tsne_arr[:, 0],
            tsne_arr[:, 1],
            c=density,
            cmap=cmap,
            alpha=alpha,
            s=8,
            edgecolors="none",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=12)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def visualize_latent_tsne(
    vae,
    dataloader,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    sample_num: int = 10000,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """End-to-end: extract latents → t-SNE → plot → return results.

    Args:
        vae: ``VAEWrapper`` instance.
        dataloader: Dataset dataloader.
        output_path: Optional path to save the figure.
        title: Optional plot title.
        sample_num: Number of per-pixel vectors to sample.
        perplexity: t-SNE perplexity.
        max_iter: t-SNE iterations.
        device: Torch device.
        seed: Random seed.

    Returns:
        ``(tsne_results, uniformity_metrics)`` tuple.
    """
    features = extract_latent_pixels(vae, dataloader, sample_num, device, seed)
    tsne_results = compute_tsne(features, perplexity, max_iter, seed)
    metrics = calculate_uniformity_metrics(tsne_results)
    plot_tsne(tsne_results, output_path=output_path, title=title)
    plt.close("all")
    return tsne_results, metrics
