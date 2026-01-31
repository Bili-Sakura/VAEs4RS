"""
VAEs4RS: Evaluating VAE Robustness on Remote Sensing Data.

This package provides tools for evaluating variational autoencoders
pre-trained on natural images when applied to remote sensing datasets.
"""

from .config import VAE_CONFIGS, DATASET_CONFIGS, EvalConfig
from .models import load_vae, VAEWrapper
from .datasets import load_dataset, RSDataset
from .metrics import MetricCalculator, MetricResults

__version__ = "0.1.0"
__all__ = [
    "VAE_CONFIGS",
    "DATASET_CONFIGS", 
    "EvalConfig",
    "load_vae",
    "VAEWrapper",
    "load_dataset",
    "RSDataset",
    "MetricCalculator",
    "MetricResults",
]
