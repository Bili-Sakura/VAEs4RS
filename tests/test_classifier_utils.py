"""Tests for classifier creation and feature extractor utilities."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.training.classifier_utils import (
    create_vgg16,
    create_inception_v3,
    RSVGGFeatures,
    RSInceptionFeatures,
    RSLPIPSMetric,
)


# ---- Tests: model creation -----------------------------------------------

class TestCreateVGG16:
    def test_output_classes(self):
        model = create_vgg16(num_classes=45)
        assert model.classifier[-1].out_features == 45

    def test_forward_pass(self):
        model = create_vgg16(num_classes=10)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_different_num_classes(self):
        for nc in [21, 30, 45]:
            model = create_vgg16(num_classes=nc)
            assert model.classifier[-1].out_features == nc


class TestCreateInceptionV3:
    def test_output_classes(self):
        model = create_inception_v3(num_classes=45)
        assert model.fc.out_features == 45
        assert model.AuxLogits.fc.out_features == 45

    def test_forward_pass_eval(self):
        model = create_inception_v3(num_classes=10)
        model.eval()
        x = torch.randn(2, 3, 299, 299)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_forward_pass_train(self):
        model = create_inception_v3(num_classes=10)
        model.train()
        x = torch.randn(2, 3, 299, 299)
        out = model(x)
        # In train mode, InceptionV3 returns InceptionOutputs
        if isinstance(out, tuple):
            logits, aux = out
        else:
            logits = out.logits
            aux = out.aux_logits
        assert logits.shape == (2, 10)
        assert aux.shape == (2, 10)


# ---- Tests: feature extractors (with saved checkpoint) -------------------

@pytest.fixture(scope="module")
def vgg_checkpoint(tmp_path_factory):
    """Save a fresh VGG16 checkpoint for testing."""
    model = create_vgg16(num_classes=21)
    path = tmp_path_factory.mktemp("ckpt") / "vgg16.pth"
    torch.save(model.state_dict(), path)
    return str(path)


@pytest.fixture(scope="module")
def inception_checkpoint(tmp_path_factory):
    """Save a fresh InceptionV3 checkpoint for testing."""
    model = create_inception_v3(num_classes=21)
    path = tmp_path_factory.mktemp("ckpt") / "inception.pth"
    torch.save(model.state_dict(), path)
    return str(path)


class TestRSVGGFeatures:
    def test_output_stages(self, vgg_checkpoint):
        extractor = RSVGGFeatures(vgg_checkpoint, num_classes=21, device="cpu")
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            feats = extractor(x)
        assert len(feats) == 5
        for f in feats:
            assert f.shape[0] == 2

    def test_frozen_parameters(self, vgg_checkpoint):
        extractor = RSVGGFeatures(vgg_checkpoint, num_classes=21, device="cpu")
        extractor.eval()
        for p in extractor.parameters():
            assert not p.requires_grad


class TestRSInceptionFeatures:
    def test_output_shape(self, inception_checkpoint):
        extractor = RSInceptionFeatures(inception_checkpoint, num_classes=21, device="cpu")
        x = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
        with torch.no_grad():
            out = extractor(x)
        assert out.shape == (2, 2048)

    def test_frozen_parameters(self, inception_checkpoint):
        extractor = RSInceptionFeatures(inception_checkpoint, num_classes=21, device="cpu")
        for p in extractor.parameters():
            assert not p.requires_grad


class TestRSLPIPSMetric:
    def test_distance_zero_for_identical(self, vgg_checkpoint):
        metric = RSLPIPSMetric(vgg_checkpoint, num_classes=21, device="cpu")
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            dist = metric(x, x)
        assert dist.item() == pytest.approx(0.0, abs=1e-4)

    def test_distance_nonzero_for_different(self, vgg_checkpoint):
        metric = RSLPIPSMetric(vgg_checkpoint, num_classes=21, device="cpu")
        x = torch.randn(2, 3, 64, 64)
        y = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            dist = metric(x, y)
        assert dist.item() > 0
