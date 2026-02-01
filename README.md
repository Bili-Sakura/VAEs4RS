# VAEs4RS

**The Robustness of Natural English Priors in Remote Sensing: A Zero-Shot VAE Study**

Are pre-trained VAEs good zero-shot remote sensing image reconstructors?

This repository evaluates variational autoencoders (VAEs) pre-trained on natural image datasets when applied to remote sensing data in a zero-shot manner.

## VAE Models Evaluated

| Model | Source | Latent Channels |
|-------|--------|-----------------|
| SD21-VAE | Stable Diffusion 2.1 | 4 |
| SDXL-VAE | Stable Diffusion XL | 4 |
| SD35-VAE | Stable Diffusion 3.5 | 16 |
| FLUX1-VAE | FLUX.1 | 16 |
| FLUX2-VAE | FLUX.2 | 32 |
| SANA-VAE | SANA (DC-AE) | 32 |
| Qwen-VAE | Qwen-Image | 16 |

## Datasets

- **NWPU-RESISC45**: 31,500 images, 45 classes, 256×256 pixels
- **AID**: 10,000 images, 30 classes, 600×600 pixels

## Project Structure

```
VAEs4RS/
├── src/
│   ├── config.py       # VAE and dataset configurations
│   ├── models.py       # VAE model loading utilities
│   ├── datasets.py     # Dataset loading for RS datasets
│   ├── metrics.py      # PSNR, SSIM, LPIPS, FID metrics
│   ├── evaluate.py     # Main evaluation script
│   ├── visualize.py    # Visualization utilities
│   ├── ablation.py     # Denoising/de-hazing ablation study
│   └── utils.py        # Helper utilities
├── scripts/
│   ├── download_datasets.py    # Dataset download instructions
│   └── generate_latex_table.py # Generate LaTeX tables
├── run_experiments.py  # Run all experiments
└── datasets/
    └── blanchon/
        ├── RESISC45/
        └── AID/
```

## Quick Start

### 1. Evaluate a single model on a single dataset

```bash
cd src
python evaluate.py --model SD21-VAE --dataset RESISC45
```

### 2. Evaluate all models on all datasets

```bash
cd src
python evaluate.py --all
```

### 3. Run the complete experiment pipeline

```bash
python run_experiments.py
```

### 4. Run only ablation study (denoising/de-hazing)

```bash
python run_experiments.py --ablation-only
```

## Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity Index): Higher is better
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better
- **rFID** (Reconstruction FID): Lower is better

## Usage Examples

### Load a VAE model

```python
from src.models import load_vae

vae = load_vae("SD21-VAE", device="cuda")

# Reconstruct images
reconstructed = vae.reconstruct(images)  # images: (B, 3, H, W), range [-1, 1]
```

### Compute metrics

```python
from src.metrics import MetricCalculator

calculator = MetricCalculator(device="cuda")
calculator.update(original_images, reconstructed_images)
results = calculator.compute()
print(results)  # PSNR: 28.5 dB | SSIM: 0.92 | LPIPS: 0.05 | FID: 12.3
```

### Load dataset

```python
from src.datasets import load_dataset

dataset, dataloader = load_dataset("RESISC45", image_size=256, batch_size=16)

for images, labels, paths in dataloader:
    # images: (B, 3, 256, 256), range [-1, 1]
    pass
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{vaes4rs2026,
  title={The Robustness of Natural English Priors in Remote Sensing: A Zero-Shot VAE Study},
  author={Anonymous},
  booktitle={ICLR 2026 Workshop on Machine Learning for Remote Sensing},
  year={2026}
}
```

## License

MIT License
