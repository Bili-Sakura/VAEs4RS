# Source Code Documentation

This directory contains the core implementation for evaluating VAEs on remote sensing datasets.

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

## Code Usage Examples

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

#### Using custom feature extractor for FID

You can use a custom pre-trained model as the feature extractor for FID instead of the default Inception v3:

```python
import torch
import torch.nn as nn
import torchvision.models as models
from src.metrics import MetricCalculator

# Example: Use ResNet50 as feature extractor
resnet50 = models.resnet50(pretrained=True)
# Remove the final classification layer to get features
feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
# Flatten the output
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
feature_extractor = nn.Sequential(feature_extractor, Flatten())

# Use custom feature extractor
calculator = MetricCalculator(
    device="cuda",
    compute_fid=True,
    fid_feature_extractor=feature_extractor
)

# Or pass it via EvalConfig
from src.config import EvalConfig
config = EvalConfig(
    batch_size=16,
    device="cuda",
    fid_feature_extractor=feature_extractor  # Your custom model
)
```

**Note:** The feature extractor should be a `torch.nn.Module` that:
- Takes images as input (shape: `(B, C, H, W)`)
- Returns features with shape `(B, num_features)` where `num_features` is the feature dimension
- The model will be automatically moved to the specified device and set to eval mode

### Load dataset

```python
from src.datasets import load_dataset

dataset, dataloader = load_dataset("RESISC45", image_size=256, batch_size=16)

for images, labels, paths in dataloader:
    # images: (B, 3, 256, 256), range [-1, 1]
    pass
```

### Evaluate a single model-dataset combination

```python
from src.evaluate import evaluate_single
from src.models import load_vae
from src.config import EvalConfig

# Create evaluation configuration
config = EvalConfig(
    batch_size=16,
    image_size=256,
    device="cuda",
    output_dir="outputs"
)

# Load VAE model
vae = load_vae("FLUX2-VAE", device="cuda")

# Evaluate on dataset
metrics = evaluate_single(vae, "RESISC45", config)
print(metrics)  # MetricResults with psnr, ssim, lpips, fid
```

### Run ablation study (denoising/de-hazing)

```python
from src.ablation import run_ablation_study
from src.config import EvalConfig

config = EvalConfig(
    batch_size=16,
    image_size=256,
    device="cuda"
)

results = run_ablation_study(
    model_names=["SD21-VAE", "FLUX2-VAE"],
    dataset_name="RESISC45",
    distortion_names=["noise_0.1", "noise_0.2", "haze_0.3", "haze_0.5"],
    config=config
)
```

### Visualize reconstructions

```python
from src.visualize import visualize_reconstructions

visualize_reconstructions(
    model_names=["SD21-VAE", "SDXL-VAE", "FLUX2-VAE"],
    dataset_name="RESISC45",
    num_samples=5,
    image_size=256,
    output_path="reconstructions.png",
    device="cuda"
)
```

## Module Documentation

### `config.py`
Configuration management for VAE models and datasets.

- `VAE_CONFIGS`: Dictionary mapping model names to their configurations
- `DATASET_CONFIGS`: Dictionary mapping dataset names to their configurations
- `EvalConfig`: Dataclass for evaluation configuration (batch size, image size, device, etc.)

### `models.py`
VAE model loading and wrapper utilities.

- `load_vae(model_name, device)`: Load a VAE model by name
- `VAEWrapper`: Wrapper class for VAE models with `reconstruct()` method

### `datasets.py`
Dataset loading for remote sensing datasets.

- `load_dataset(dataset_name, image_size, batch_size, num_workers, classes)`: Load a dataset and create a DataLoader
- Supports RESISC45 and AID datasets
- Optional class filtering for subset evaluation

### `metrics.py`
Evaluation metrics for reconstruction quality.

- `MetricCalculator`: Computes PSNR, SSIM, LPIPS, and FID metrics
- `MetricResults`: Dataclass containing metric values
- Supports incremental updates for large datasets

### `evaluate.py`
Main evaluation script and utilities.

- `evaluate_single()`: Evaluate a single VAE on a single dataset
- `evaluate_all()`: Evaluate all models on all datasets
- `evaluate_from_existing_images()`: Compute metrics from saved images
- Supports incremental result saving and skipping existing evaluations

### `ablation.py`
Ablation study for denoising and de-hazing applications.

- `run_ablation_study()`: Run ablation experiments with noise and haze distortions
- `evaluate_denoising()`: Evaluate VAE as a denoising/de-hazing pre-processor

### `visualize.py`
Visualization utilities for qualitative analysis.

- `visualize_reconstructions()`: Generate side-by-side comparison images
- Creates visualizations showing original vs. reconstructed images for multiple models

### `utils.py`
Helper utilities.

- `set_seed()`: Set random seeds for reproducibility
- `save_tensor_as_image()`: Save tensor images to disk

### `vae_statistics.py`
VAE model statistics and analysis.

- `get_vae_statistics()`: Compute model statistics (GFLOPs, spatial compression ratio, latent channels, latent shape, etc.)
- Useful for understanding model complexity and capabilities

## Advanced Usage

### Custom class filtering

```python
from src.evaluate import evaluate_all
from src.config import EvalConfig

config = EvalConfig(batch_size=16, image_size=256, device="cuda")

# Evaluate only specific classes
dataset_classes = {
    "RESISC45": ["airport", "beach", "bridge"],
    "AID": ["Airport", "Beach"]
}

results = evaluate_all(config, dataset_classes=dataset_classes)
```

### Incremental evaluation with result saving

```python
from src.evaluate import evaluate_all
from src.config import EvalConfig
from pathlib import Path

config = EvalConfig(
    batch_size=16,
    image_size=256,
    device="cuda",
    output_dir="results"
)

results_dir = Path("results")
results = evaluate_all(
    config,
    results_dir=results_dir,
    skip_existing=True,  # Skip already evaluated model-dataset pairs
    save_images=True    # Save reconstructed images
)
```

### Evaluate from existing images

```python
from src.evaluate import evaluate_from_existing_images
from src.config import EvalConfig
from pathlib import Path

config = EvalConfig(device="cuda")
metrics = evaluate_from_existing_images(
    dataset_name="RESISC45",
    config=config,
    original_images_dir=Path("results/RESISC45/images/original"),
    reconstructed_images_dir=Path("results/FLUX2-VAE/RESISC45/images/reconstructed")
)
```

## Command-Line Interface

### `evaluate.py`

```bash
# Single model-dataset evaluation
python evaluate.py --model SD21-VAE --dataset RESISC45 --batch-size 16

# Evaluate all models on all datasets
python evaluate.py --all --batch-size 64 --device cuda

# Custom image size
python evaluate.py --all --image-size 256
```

### `run_experiments.py`

```bash
# Run all experiments
python run_experiments.py

# Main experiment only
python run_experiments.py --main-only

# Ablation study only
python run_experiments.py --ablation-only

# Custom configuration
python run_experiments.py --batch-size 32 --image-size 256 --device cuda

# Filter specific classes
python run_experiments.py --classes RESISC45:airport,beach AID:Airport

# Evaluate specific models only
python run_experiments.py --models SD21-VAE FLUX2-VAE

# Skip existing results
python run_experiments.py --skip-existing

# Don't save images (faster, less storage)
python run_experiments.py --no-save-images

# Evaluate metrics from existing images
python run_experiments.py --use-existing-images
```

## Notes

- All models use bfloat16 precision for efficiency
- Images are normalized to [-1, 1] range for VAE input/output
- Results are saved incrementally to avoid data loss
- Individual result files are saved per model-dataset combination for better organization
- Original images are saved once and shared across all models to save storage
