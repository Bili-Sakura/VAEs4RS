# VAEs4RS Source Code

Minimal, modular implementation for VAE evaluation on remote sensing datasets.

## Modules

| Module | Description |
|--------|-------------|
| `config.py` | Loads settings from `config.yaml` |
| `models.py` | VAE loading with unified wrapper |
| `datasets.py` | Remote sensing dataset loaders |
| `metrics.py` | PSNR, SSIM, LPIPS, FID, CMMD |
| `evaluate.py` | Evaluation logic |
| `ablation.py` | Ablation study (noise/haze) |
| `visualize.py` | Visualization utilities |
| `utils.py` | General utilities |
| `vae_statistics.py` | Model statistics (params, FLOPs) |

## Usage

All modules are importable:

```python
from src import load_vae, load_dataset, evaluate_all, get_config

cfg = get_config()
results = evaluate_all(cfg.eval)
```
