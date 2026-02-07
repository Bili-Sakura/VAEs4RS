"""
Model modules: VAE loading, wrapping, and statistics.
"""

from .vae_wrapper import (
    load_vae,
    load_all_vaes,
    VAEWrapper,
    VAE_CLASSES,
)
from .vae_statistics import (
    VAEStatistics,
    get_vae_statistics,
    get_all_vae_statistics,
    count_params,
    calculate_flops,
    print_statistics_table,
)

__all__ = [
    "load_vae",
    "load_all_vaes",
    "VAEWrapper",
    "VAE_CLASSES",
    "VAEStatistics",
    "get_vae_statistics",
    "get_all_vae_statistics",
    "count_params",
    "calculate_flops",
    "print_statistics_table",
]
