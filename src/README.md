# VAEs4RS Source Code

Modular implementation for VAE evaluation and training on remote sensing datasets.

## Structure

```
src/
├── __init__.py               # Top-level package (backward-compatible exports)
├── models/                   # VAE model loading and analysis
│   ├── vae_wrapper.py        # VAEWrapper, load_vae, load_all_vaes
│   └── vae_statistics.py     # Model statistics (params, FLOPs)
├── utils/                    # Configuration, datasets, helpers
│   ├── config.py             # Loads settings from config.yaml
│   ├── datasets.py           # Remote sensing dataset loaders
│   └── helpers.py            # General utilities (set_seed, save_tensor_as_image)
├── training/                 # Training and fine-tuning utilities
│   └── train_utils.py        # VAE preparation, loss, optimizer, dataset
├── evaluation/               # Evaluation and visualization
│   ├── evaluate.py           # Main evaluation loop
│   ├── metrics.py            # PSNR, SSIM, LPIPS, FID, CMMD
│   ├── ablation.py           # Ablation study (noise/haze)
│   └── visualize.py          # Visualization utilities
scripts/
├── train_vae.py              # Train any VAE (single/multi-GPU)
├── run_experiments.py        # Run evaluation experiments
└── streamlit_app.py          # Interactive reconstruction viewer
configs/
├── train_vae.yaml            # Generic training config (any VAE)
└── train_rs_vae.yaml         # Single-channel RS training config
```

## Usage

All modules are importable from the top-level package:

```python
from src import load_vae, load_dataset, evaluate_all, get_config

cfg = get_config()
results = evaluate_all(cfg.eval)
```

Or import from subpackages directly:

```python
from src.models import load_vae, VAEWrapper
from src.training import prepare_vae_for_training, vae_loss
from src.evaluation import MetricCalculator
from src.utils import get_config
```

## Supported VAE Architectures for Training

| Architecture | Models | Partial Freeze | Conv Replacement |
|-------------|--------|:--------------:|:----------------:|
| AutoencoderKL | SD21-VAE, SDXL-VAE, SD35-VAE, FLUX1-VAE | ✅ | ✅ |
| AutoencoderKLFlux2 | FLUX2-VAE | ✅ | ✅ |
| AutoencoderKLQwenImage | Qwen-VAE | ✅ | ✅ |
| AutoencoderDC | SANA-VAE | Full only | ❌ |
