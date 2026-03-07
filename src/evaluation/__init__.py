"""
Evaluation modules: metrics, evaluation loop, ablation studies, and visualization.
"""

from .evaluate import evaluate_single, evaluate_all, print_results_table
from .metrics import MetricCalculator, MetricResults
from .ablation import run_ablation_study, print_ablation_table
from .visualize import visualize_reconstructions
from .latent_viz import (
    extract_latent_pixels,
    compute_tsne,
    calculate_uniformity_metrics,
    plot_tsne,
    plot_tsne_comparison,
    visualize_latent_tsne,
)

__all__ = [
    "evaluate_single",
    "evaluate_all",
    "print_results_table",
    "MetricCalculator",
    "MetricResults",
    "run_ablation_study",
    "print_ablation_table",
    "visualize_reconstructions",
    "extract_latent_pixels",
    "compute_tsne",
    "calculate_uniformity_metrics",
    "plot_tsne",
    "plot_tsne_comparison",
    "visualize_latent_tsne",
]
