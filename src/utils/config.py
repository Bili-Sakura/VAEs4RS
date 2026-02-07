"""
Configuration loader for VAEs4RS.
Loads all settings from config.yaml in project root.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

# Project root (parent of src/utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_yaml_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class VAEConfig:
    """VAE model configuration."""
    name: str
    pretrained_path: str
    scaling_factor: float = 0.18215
    latent_channels: int = 4
    spatial_compression: int = 8
    subfolder: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    root: str
    image_size: int
    num_classes: int


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    batch_size: int = 64
    num_workers: int = 8
    device: str = "cuda"
    seed: int = 42
    image_size: Optional[int] = None
    output_dir: str = "outputs"
    skip_existing: bool = False
    save_images: bool = False
    save_latents: bool = False
    compute_cmmd: bool = True
    cmmd_clip_model: str = "models/BiliSakura/Remote-CLIP-ViT-L-14/transformers"
    cmmd_batch_size: int = 32
    mmd_chunk_size: int = 1000
    fid_feature_extractor: Optional[Any] = None


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    models: Optional[List[str]] = None
    datasets: Optional[List[str]] = None
    class_filter: Optional[Dict[str, List[str]]] = None
    split_files: Optional[Dict[str, str]] = None


@dataclass
class Config:
    """Main configuration container."""
    eval: EvalConfig
    experiment: ExperimentConfig
    vaes: Dict[str, VAEConfig]
    datasets: Dict[str, DatasetConfig]
    ablation_enabled: bool = False
    ablation_distortions: List[str] = field(default_factory=lambda: ["noise_0.1", "noise_0.2"])
    visualization_enabled: bool = False
    visualization_num_samples: int = 5


def load_config(config_path: Path = CONFIG_PATH) -> Config:
    """Load and parse configuration from YAML file."""
    raw = load_yaml_config(config_path)
    
    # Parse eval config
    eval_raw = raw.get("eval", {})
    eval_config = EvalConfig(
        batch_size=eval_raw.get("batch_size", 64),
        num_workers=eval_raw.get("num_workers", 8),
        device=eval_raw.get("device", "cuda"),
        seed=eval_raw.get("seed", 42),
        image_size=eval_raw.get("image_size"),
        output_dir=eval_raw.get("output_dir", "outputs"),
        skip_existing=eval_raw.get("skip_existing", False),
        save_images=eval_raw.get("save_images", False),
        save_latents=eval_raw.get("save_latents", False),
        compute_cmmd=eval_raw.get("compute_cmmd", True),
        cmmd_clip_model=eval_raw.get("cmmd_clip_model", "models/BiliSakura/Remote-CLIP-ViT-L-14/transformers"),
        cmmd_batch_size=eval_raw.get("cmmd_batch_size", 32),
        mmd_chunk_size=eval_raw.get("mmd_chunk_size", 1000),
    )
    
    # Parse experiment config
    exp_raw = raw.get("experiment", {})
    experiment_config = ExperimentConfig(
        models=exp_raw.get("models"),
        datasets=exp_raw.get("datasets"),
        class_filter=exp_raw.get("class_filter"),
        split_files=exp_raw.get("split_files"),
    )
    
    # Parse VAE configs
    vaes = {}
    for name, cfg in raw.get("vaes", {}).items():
        vaes[name] = VAEConfig(
            name=name,
            pretrained_path=cfg.get("pretrained_path", ""),
            scaling_factor=cfg.get("scaling_factor", 0.18215),
            latent_channels=cfg.get("latent_channels", 4),
            spatial_compression=cfg.get("spatial_compression", 8),
            subfolder=cfg.get("subfolder"),
        )
    
    # Parse dataset configs
    datasets = {}
    for name, cfg in raw.get("datasets", {}).items():
        datasets[name] = DatasetConfig(
            name=name,
            root=cfg.get("root", ""),
            image_size=cfg.get("image_size", 256),
            num_classes=cfg.get("num_classes", 10),
        )
    
    # Ablation settings
    ablation_raw = raw.get("ablation", {})
    ablation_enabled = ablation_raw.get("enabled", False)
    ablation_distortions = ablation_raw.get("distortions", ["noise_0.1", "noise_0.2"])
    
    # Visualization settings
    vis_raw = raw.get("visualization", {})
    visualization_enabled = vis_raw.get("enabled", False)
    visualization_num_samples = vis_raw.get("num_samples", 5)
    
    return Config(
        eval=eval_config,
        experiment=experiment_config,
        vaes=vaes,
        datasets=datasets,
        ablation_enabled=ablation_enabled,
        ablation_distortions=ablation_distortions,
        visualization_enabled=visualization_enabled,
        visualization_num_samples=visualization_num_samples,
    )


# Lazy-loaded global config
_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Get the global configuration (lazy-loaded)."""
    global _config
    if _config is None or reload:
        _config = load_config()
    return _config


# Backward compatibility: expose dicts for old code
def get_vae_configs() -> Dict[str, VAEConfig]:
    """Get VAE configurations dict."""
    return get_config().vaes


def get_dataset_configs() -> Dict[str, DatasetConfig]:
    """Get dataset configurations dict."""
    return get_config().datasets


# For backward compatibility
VAE_CONFIGS = property(lambda self: get_vae_configs())
DATASET_CONFIGS = property(lambda self: get_dataset_configs())
