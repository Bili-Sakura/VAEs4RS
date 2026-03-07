"""Tests for KID metric and RS-domain metric integration."""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.evaluation.metrics import MetricCalculator, MetricResults


# ---- Tests: MetricResults dataclass --------------------------------------

class TestMetricResults:
    def test_basic_fields(self):
        r = MetricResults(psnr=30.0, ssim=0.95, lpips=0.05)
        assert r.psnr == 30.0
        assert r.ssim == 0.95
        assert r.lpips == 0.05
        assert r.fid is None
        assert r.kid is None
        assert r.cmmd is None
        assert r.fid_rs is None
        assert r.kid_rs is None
        assert r.lpips_rs is None

    def test_all_fields(self):
        r = MetricResults(
            psnr=30.0, ssim=0.95, lpips=0.05,
            fid=10.0, kid=0.01, cmmd=5.0,
            fid_rs=8.0, kid_rs=0.008, lpips_rs=0.04,
        )
        assert r.kid == 0.01
        assert r.fid_rs == 8.0
        assert r.kid_rs == 0.008
        assert r.lpips_rs == 0.04

    def test_to_dict_contains_new_keys(self):
        r = MetricResults(psnr=30.0, ssim=0.95, lpips=0.05, kid=0.01)
        d = r.to_dict()
        assert "kid" in d
        assert "fid_rs" in d
        assert "kid_rs" in d
        assert "lpips_rs" in d

    def test_repr_includes_kid(self):
        r = MetricResults(psnr=30.0, ssim=0.95, lpips=0.05, kid=0.01)
        s = repr(r)
        assert "KID" in s

    def test_repr_includes_rs_metrics(self):
        r = MetricResults(
            psnr=30.0, ssim=0.95, lpips=0.05,
            fid_rs=8.0, kid_rs=0.008, lpips_rs=0.04,
        )
        s = repr(r)
        assert "FID(rs)" in s
        assert "KID(rs)" in s
        assert "LPIPS(rs)" in s


# ---- Tests: MetricCalculator with KID -----------------------------------

class TestMetricCalculatorKID:
    def test_kid_disabled_by_default(self):
        calc = MetricCalculator(device="cpu", compute_fid=False)
        assert calc.kid is None

    def test_kid_enabled(self):
        calc = MetricCalculator(device="cpu", compute_fid=False, compute_kid=True)
        # KID should be initialised (if torchmetrics supports it)
        if calc.kid is not None:
            # Feed enough data for the default subset_size (50)
            for _ in range(2):
                orig = torch.randn(32, 3, 32, 32).clamp(-1, 1)
                recon = (orig + 0.1 * torch.randn_like(orig)).clamp(-1, 1)
                calc.update(orig, recon)
            results = calc.compute()
            assert results.kid is not None


# ---- Tests: MetricCalculator basic (no FID/KID to avoid torch-fidelity) --

class TestMetricCalculatorBasic:
    def test_psnr_ssim_lpips(self):
        calc = MetricCalculator(device="cpu", compute_fid=False, compute_kid=False)
        orig = torch.randn(4, 3, 32, 32).clamp(-1, 1)
        recon = (orig + 0.05 * torch.randn_like(orig)).clamp(-1, 1)
        calc.update(orig, recon)
        results = calc.compute()
        assert results.psnr > 0
        assert 0 < results.ssim <= 1
        assert results.lpips >= 0
        assert results.fid is None
        assert results.kid is None

    def test_reset(self):
        calc = MetricCalculator(device="cpu", compute_fid=False, compute_kid=False)
        orig = torch.randn(4, 3, 32, 32).clamp(-1, 1)
        calc.update(orig, orig)
        calc.reset()
        assert len(calc.psnr_values) == 0
        assert len(calc.ssim_values) == 0
        assert len(calc.lpips_values) == 0
        assert len(calc.lpips_rs_values) == 0

    def test_rs_metrics_disabled_without_extractors(self):
        calc = MetricCalculator(device="cpu", compute_fid=False)
        assert calc.fid_rs is None
        assert calc.kid_rs is None
        assert calc.lpips_rs_metric is None
