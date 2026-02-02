"""
VAEs4RS: Evaluating VAE Robustness on Remote Sensing Data.

This package provides tools for evaluating variational autoencoders
pre-trained on natural images when applied to remote sensing datasets.
"""

from .config import VAE_CONFIGS, DATASET_CONFIGS, EvalConfig, VAEConfig
from .models import load_vae, VAEWrapper
from .datasets import load_dataset, RSDataset
from .metrics import MetricCalculator, MetricResults
from .vae_statistics import (
    VAEStatistics,
    get_vae_statistics,
    get_all_vae_statistics,
    print_vae_statistics_table,
    save_vae_statistics,
    get_quick_statistics_from_config,
    print_quick_statistics_table,
)

__version__ = "0.1.0"
__all__ = [
    "VAE_CONFIGS",
    "DATASET_CONFIGS", 
    "EvalConfig",
    "VAEConfig",
    "load_vae",
    "VAEWrapper",
    "load_dataset",
    "RSDataset",
    "MetricCalculator",
    "MetricResults",
    "VAEStatistics",
    "get_vae_statistics",
    "get_all_vae_statistics",
    "print_vae_statistics_table",
    "save_vae_statistics",
    "get_quick_statistics_from_config",
    "print_quick_statistics_table",
]
