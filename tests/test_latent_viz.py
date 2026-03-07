"""Tests for latent feature visualization using t-SNE."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np
import torch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.evaluation.latent_viz import (
    extract_latent_pixels,
    compute_tsne,
    calculate_uniformity_metrics,
    plot_tsne,
    plot_tsne_comparison,
    visualize_latent_tsne,
)


# ---- Fixtures ------------------------------------------------------------

class _FakeVAE:
    """Minimal VAE mock that returns fixed-shape latent tensors."""

    def __init__(self, latent_channels: int = 4, spatial: int = 8):
        self.latent_channels = latent_channels
        self.spatial = spatial
        self.model = MagicMock()
        # Create a real parameter so next(vae.model.parameters()) works
        param = torch.nn.Parameter(torch.zeros(1))
        self.model.parameters = MagicMock(return_value=iter([param]))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.randn(b, self.latent_channels, self.spatial, self.spatial)


def _fake_dataloader(n_batches: int = 5, batch_size: int = 8, img_size: int = 32):
    """Yield (images, labels, paths) tuples."""
    for _ in range(n_batches):
        images = torch.randn(batch_size, 3, img_size, img_size)
        labels = torch.zeros(batch_size, dtype=torch.long)
        paths = ["fake.png"] * batch_size
        yield images, labels, paths


# ---- Tests: extract_latent_pixels ----------------------------------------

class TestExtractLatentPixels:
    def test_output_shape(self):
        vae = _FakeVAE(latent_channels=4)
        dl = list(_fake_dataloader(n_batches=3, batch_size=8))
        feats = extract_latent_pixels(vae, dl, sample_num=10, device="cpu")
        assert feats.shape == (10, 4)

    def test_respects_sample_num(self):
        vae = _FakeVAE(latent_channels=16)
        dl = list(_fake_dataloader(n_batches=5, batch_size=8))
        feats = extract_latent_pixels(vae, dl, sample_num=7, device="cpu")
        assert feats.shape[0] == 7

    def test_empty_dataloader_raises(self):
        vae = _FakeVAE()
        with pytest.raises(RuntimeError, match="No latent features"):
            extract_latent_pixels(vae, iter([]), sample_num=10, device="cpu")

    def test_cpu_tensor(self):
        vae = _FakeVAE()
        dl = list(_fake_dataloader(n_batches=2, batch_size=4))
        feats = extract_latent_pixels(vae, dl, sample_num=5, device="cpu")
        assert feats.device == torch.device("cpu")


# ---- Tests: compute_tsne ------------------------------------------------

class TestComputeTSNE:
    def test_output_shape(self):
        features = torch.randn(50, 8)
        result = compute_tsne(features, perplexity=5, max_iter=250)
        assert result.shape == (50, 2)

    def test_numpy_output(self):
        features = torch.randn(50, 8)
        result = compute_tsne(features, perplexity=5, max_iter=250)
        assert isinstance(result, np.ndarray)


# ---- Tests: uniformity metrics -------------------------------------------

class TestUniformityMetrics:
    def test_keys_present(self):
        data = np.random.randn(100, 2)
        metrics = calculate_uniformity_metrics(data)
        assert "density_std" in metrics
        assert "density_cv" in metrics
        assert "normalized_entropy" in metrics
        assert "gini_coefficient" in metrics

    def test_entropy_range(self):
        data = np.random.randn(200, 2)
        metrics = calculate_uniformity_metrics(data)
        assert 0 < metrics["normalized_entropy"] <= 1.0

    def test_gini_range(self):
        data = np.random.randn(200, 2)
        metrics = calculate_uniformity_metrics(data)
        assert 0 <= metrics["gini_coefficient"] <= 1.0


# ---- Tests: plotting -----------------------------------------------------

class TestPlotTSNE:
    def test_returns_figure(self):
        data = np.random.randn(50, 2)
        import matplotlib.pyplot as plt
        fig = plot_tsne(data)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_to_file(self, tmp_path):
        data = np.random.randn(50, 2)
        out = str(tmp_path / "test_tsne.png")
        plot_tsne(data, output_path=out)
        assert Path(out).exists()
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_with_title(self):
        data = np.random.randn(50, 2)
        import matplotlib.pyplot as plt
        fig = plot_tsne(data, title="Test Title")
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestPlotTSNEComparison:
    def test_returns_figure(self):
        results = {
            "ModelA": np.random.randn(50, 2),
            "ModelB": np.random.randn(50, 2),
        }
        import matplotlib.pyplot as plt
        fig = plot_tsne_comparison(results)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_to_file(self, tmp_path):
        results = {
            "ModelA": np.random.randn(50, 2),
            "ModelB": np.random.randn(50, 2),
        }
        out = str(tmp_path / "cmp.png")
        plot_tsne_comparison(results, output_path=out)
        assert Path(out).exists()
        import matplotlib.pyplot as plt
        plt.close("all")


# ---- Tests: end-to-end helper -------------------------------------------

class TestVisualizeLatentTSNE:
    def test_returns_tuple(self, tmp_path):
        vae = _FakeVAE(latent_channels=4)
        dl = list(_fake_dataloader(n_batches=3, batch_size=8))
        out = str(tmp_path / "e2e.png")
        tsne_res, metrics = visualize_latent_tsne(
            vae, dl, output_path=out, sample_num=20,
            perplexity=5, max_iter=250, device="cpu",
        )
        assert tsne_res.shape == (20, 2)
        assert "normalized_entropy" in metrics
        assert Path(out).exists()
