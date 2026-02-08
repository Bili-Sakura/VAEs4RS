"""Tests for quick VAE reconstruction helper utilities."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.quick_vae_reconstruction import (  # noqa: E402
    _collect_image_paths,
    _normalize_image_array,
    compute_reconstruction_metrics,
    reduce_reconstruction_channels,
)


def test_normalize_image_array_scales_to_unit_range():
    arr = np.array([[0, 128, 255]], dtype=np.uint8)
    norm = _normalize_image_array(arr)
    assert norm.dtype == np.float32
    assert norm.min() == 0.0
    assert norm.max() == pytest.approx(1.0)
    assert norm[0, 1] == pytest.approx(128 / 255.0)


def test_reduce_reconstruction_channels_single_channel_means():
    recon = torch.tensor(
        [[[[0.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    )
    reduced = reduce_reconstruction_channels(recon, orig_channels=1)
    expected = recon.mean(dim=1, keepdim=True)
    assert torch.allclose(reduced, expected)


def test_collect_image_paths_filters_extensions(tmp_path: Path):
    keep = tmp_path / "keep.png"
    drop = tmp_path / "drop.txt"
    keep.touch()
    drop.touch()

    found = _collect_image_paths(tmp_path, exts=[".png"])
    assert found == [keep]


def test_compute_reconstruction_metrics_identical_images():
    target = torch.ones(1, 1, 4, 4)
    metrics = compute_reconstruction_metrics(target.clone(), target)
    assert metrics["mae"] == 0.0
    assert math.isinf(metrics["psnr"])
    assert metrics["ssim"] == pytest.approx(1.0)
