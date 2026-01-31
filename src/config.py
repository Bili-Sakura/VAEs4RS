"""
Configuration for VAE models and datasets.

Defines HuggingFace model paths and dataset configurations.
"""

from dataclasses import dataclass, field
from typing import Optional


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


VAE_CONFIGS = {
    "SD21-VAE": VAEConfig(
        name="SD21-VAE",
        pretrained_path="stabilityai/sd-vae-ft-ema",
        scaling_factor=0.18215,
        latent_channels=4,
    ),
    "SDXL-VAE": VAEConfig(
        name="SDXL-VAE",
        pretrained_path="madebyollin/sdxl-vae-fp16-fix",
        scaling_factor=0.13025,
        latent_channels=4,
    ),
    "SD35-VAE": VAEConfig(
        name="SD35-VAE",
        pretrained_path="stabilityai/stable-diffusion-3.5-large",
        subfolder="vae",
        scaling_factor=1.5305,
        latent_channels=16,
    ),
    "FLUX1-VAE": VAEConfig(
        name="FLUX1-VAE",
        pretrained_path="black-forest-labs/FLUX.1-dev",
        subfolder="vae",
        scaling_factor=0.3611,
        latent_channels=16,
    ),
    "FLUX2-VAE": VAEConfig(
        name="FLUX2-VAE",
        pretrained_path="black-forest-labs/FLUX.1-schnell",  # Placeholder for FLUX.2
        subfolder="vae",
        scaling_factor=0.3611,
        latent_channels=16,
    ),
    "SANA-VAE": VAEConfig(
        name="SANA-VAE",
        pretrained_path="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        subfolder="vae",
        scaling_factor=0.41407,
        latent_channels=32,
    ),
    "Qwen-VAE": VAEConfig(
        name="Qwen-VAE",
        pretrained_path="mit-han-lab/dc-ae-f32c32-sana-1.0",
        scaling_factor=0.41407,
        latent_channels=32,
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
        root="datasets/blanchon/RESISC45",
        image_size=256,
        num_classes=45,
    ),
    "AID": DatasetConfig(
        name="AID",
        root="datasets/blanchon/AID",
        image_size=600,
        num_classes=30,
    ),
}


# =============================================================================
# Evaluation Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"
    output_dir: str = "outputs"
    seed: int = 42
    image_size: int = 256  # Resize all images to this size for fair comparison
