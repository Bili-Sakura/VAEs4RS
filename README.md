# VAEs4RS: Zero-Shot VAE Study for Remote Sensing

Evaluation framework for comparing VAE architectures on remote sensing datasets.

## Quick Start

1. **Configure** - Edit `config.yaml` with your settings:
   ```yaml
   eval:
     batch_size: 64
     device: cuda
     compute_cmmd: true

   experiment:
     models:
       - SD21-VAE
       - SDXL-VAE
     datasets:
       - RESISC45
       - UCMerced
   ```

2. **Run** - Execute the evaluation:
   ```bash
   python run_experiments.py
   ```

## Configuration

All settings are in `config.yaml`. No command-line arguments needed.

### Key Sections

- **eval**: Evaluation settings (batch_size, device, image_size, etc.)
- **vaes**: VAE model configurations
- **datasets**: Dataset paths and settings
- **experiment**: What to run (models, datasets, class filters)
- **ablation**: Ablation study settings
- **visualization**: Visualization settings

### Examples

Run with all defaults from config.yaml:
```bash
python run_experiments.py
```

Run ablation study:
```bash
python run_experiments.py --ablation
```

Generate visualizations:
```bash
python run_experiments.py --visualize
```

## Project Structure

```
.
├── config.yaml          # All configuration in one place
├── run_experiments.py   # Main entry point
├── streamlit_app.py     # Interactive viewer
└── src/
    ├── config.py        # Config loader
    ├── models.py        # VAE loading utilities
    ├── datasets.py      # Dataset loading
    ├── metrics.py       # PSNR, SSIM, LPIPS, FID, CMMD
    ├── evaluate.py      # Evaluation logic
    ├── ablation.py      # Ablation studies
    ├── visualize.py     # Visualization utilities
    ├── utils.py         # General utilities
    └── vae_statistics.py # Model statistics
```

## Supported VAEs

- SD21-VAE
- SDXL-VAE
- SD35-VAE
- FLUX1-VAE
- FLUX2-VAE
- SANA-VAE
- Qwen-VAE

## Supported Datasets

- RESISC45
- AID
- UCMerced

## Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (dB, higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)
- **FID**: Fréchet Inception Distance (lower is better)
- **CMMD**: CLIP Maximum Mean Discrepancy (lower is better)

## API Usage

```python
from src import load_vae, load_dataset, evaluate_single, get_config

# Load config
cfg = get_config()

# Load a VAE
vae = load_vae("SD21-VAE", device="cuda")

# Load a dataset
dataset, dataloader = load_dataset("RESISC45", image_size=256)

# Evaluate
metrics = evaluate_single(vae, "RESISC45", cfg.eval)
print(metrics)  # PSNR: 28.45 dB | SSIM: 0.8234 | LPIPS: 0.1234
```

## Interactive Viewer

Launch the Streamlit app to browse reconstruction results:

```bash
streamlit run streamlit_app.py
```

## Installation

```bash
pip install torch torchvision diffusers torchmetrics transformers pyyaml pillow tqdm matplotlib
```

## License

See [LICENSE](LICENSE).
