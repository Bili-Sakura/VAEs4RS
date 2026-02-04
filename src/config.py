"""
Configuration for VAE models and datasets.

Defines HuggingFace model paths and dataset configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, Any

# Use Any for the feature extractor type to avoid requiring torch import at module level
# Users can pass any torch.nn.Module instance


# =============================================================================
# VAE Model Configurations
# =============================================================================

@dataclass
class VAEConfig:
    """Configuration for a VAE model."""
    name: str
    pretrained_path: str
    subfolder: Optional[str] = None
    scaling_factor: float = 0.18215
    latent_channels: int = 4
    image_size: int = 256
    spatial_compression_ratio: int = 8  # Spatial downsampling factor (e.g., 8 means 256x256 -> 32x32)


VAE_CONFIGS = {
    "SD21-VAE": VAEConfig(
        name="SD21-VAE",
        pretrained_path="models/BiliSakura/VAEs/SD21-VAE",
        scaling_factor=0.18215,
        latent_channels=4,
        spatial_compression_ratio=8,
    ),
    "SDXL-VAE": VAEConfig(
        name="SDXL-VAE",
        pretrained_path="models/BiliSakura/VAEs/SDXL-VAE",
        scaling_factor=0.13025,
        latent_channels=4,
        spatial_compression_ratio=8,
    ),
    "SD35-VAE": VAEConfig(
        name="SD35-VAE",
        pretrained_path="models/BiliSakura/VAEs/SD35-VAE",
        scaling_factor=1.5305,
        latent_channels=16,
        spatial_compression_ratio=8,
    ),
    "FLUX1-VAE": VAEConfig(
        name="FLUX1-VAE",
        pretrained_path="models/BiliSakura/VAEs/FLUX1-VAE",
        scaling_factor=0.3611,
        latent_channels=16,
        spatial_compression_ratio=8,
    ),
    "FLUX2-VAE": VAEConfig(
        name="FLUX2-VAE",
        pretrained_path="models/BiliSakura/VAEs/FLUX2-VAE",
        scaling_factor=0.3611,
        latent_channels=32,
        spatial_compression_ratio=8,
    ),
    "SANA-VAE": VAEConfig(
        name="SANA-VAE",
        pretrained_path="models/BiliSakura/VAEs/SANA-VAE",
        scaling_factor=0.41407,
        latent_channels=32,
        spatial_compression_ratio=32,  # SANA uses 32x spatial compression
    ),
    "Qwen-VAE": VAEConfig(
        name="Qwen-VAE",
        pretrained_path="models/BiliSakura/VAEs/Qwen-VAE",
        scaling_factor=0.41407,
        latent_channels=16,
        spatial_compression_ratio=8,
    ),
}


# =============================================================================
# Dataset Configurations
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    root: str
    image_size: int
    num_classes: int


DATASET_CONFIGS = {
    "RESISC45": DatasetConfig(
        name="RESISC45",
        root="datasets/blanchon/RESISC45/data",
        image_size=256,
        num_classes=45,
    ),
    "AID": DatasetConfig(
        name="AID",
        root="datasets/blanchon/AID/data",
        image_size=600,
        num_classes=30,
    ),
    "UCMerced": DatasetConfig(
        name="UCMerced",
        root="datasets/torchgeo/ucmerced/UCMerced_LandUse/Images",
        image_size=256,
        num_classes=21,
    ),
}


# =============================================================================
# Evaluation Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"
    output_dir: str = "outputs"
    seed: int = 42
    image_size: Optional[int] = None  # Resize all images to this size. If None, use original sizes.
    fid_feature_extractor: Optional[Any] = None  # Optional custom feature extractor (torch.nn.Module) for FID metric
    compute_cmmd: bool = False  # Whether to compute CMMD metric (requires transformers)
    cmmd_clip_model: str = "/data/projects/VAEs4RS/models/BiliSakura/Remote-CLIP-ViT-L-14/transformers"  # CLIP/SigLIP model path/name for CMMD (local path or HuggingFace model name, auto-detects CLIP vs SigLIP)
