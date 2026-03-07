"""
Utilities for training VGG16 and InceptionV3 classifiers from scratch
on remote sensing datasets (RESISC45, AID, UCMerced).

Trained models serve as domain-specific feature extractors for
FID(rs), KID(rs), and LPIPS(rs) metrics.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier creation
# ---------------------------------------------------------------------------

def create_vgg16(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Create a VGG16 classifier (with batch normalization).

    Args:
        num_classes: Number of output classes.
        pretrained: If True, use ImageNet-pretrained weights; otherwise
            train from scratch.

    Returns:
        ``torchvision.models.VGG`` model.
    """
    weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
    model = models.vgg16_bn(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def create_inception_v3(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Create an InceptionV3 classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: If True, use ImageNet-pretrained weights; otherwise
            train from scratch.

    Returns:
        ``torchvision.models.Inception3`` model.
    """
    weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
    model = models.inception_v3(weights=weights, init_weights=(weights is None))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Feature extractors for RS metrics
# ---------------------------------------------------------------------------

class RSInceptionFeatures(nn.Module):
    """Feature extractor wrapping an RS-trained InceptionV3.

    Extracts the 2048-dimensional feature vector from the average-pooling
    layer, suitable for use with ``FrechetInceptionDistance`` and
    ``KernelInceptionDistance`` from ``torchmetrics``.

    Input images should be ``uint8`` tensors in ``[0, 255]`` with shape
    ``(B, 3, H, W)`` (they are normalised internally).
    """

    def __init__(self, checkpoint_path: str, num_classes: int, device: str = "cpu"):
        super().__init__()
        model = create_inception_v3(num_classes=num_classes, pretrained=False)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Keep everything up to the average-pool layer
        self.conv = nn.Sequential(
            model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
            model.Mixed_6a, model.Mixed_6b, model.Mixed_6c,
            model.Mixed_6d, model.Mixed_6e,
            model.Mixed_7a, model.Mixed_7b, model.Mixed_7c,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.requires_grad_(False)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise uint8 [0,255] → float [0,1] → ImageNet-style
        x = x.float() / 255.0
        x = nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        x = self.conv(x)
        x = self.avgpool(x)
        return x.flatten(1)  # (B, 2048)


class RSVGGFeatures(nn.Module):
    """Multi-scale feature extractor wrapping an RS-trained VGG16-BN.

    Extracts features from five relu stages (after max-pool), producing a
    list of tensors suitable for computing perceptual distances in
    ``RSLPIPSMetric``.
    """

    # VGG16-BN feature-layer indices (right after each max-pool)
    _STAGE_INDICES = (6, 13, 23, 33, 43)

    def __init__(self, checkpoint_path: str, num_classes: int, device: str = "cpu"):
        super().__init__()
        model = create_vgg16(num_classes=num_classes, pretrained=False)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self._STAGE_INDICES:
            self.slices.append(nn.Sequential(*list(model.features.children())[prev:idx + 1]))
            prev = idx + 1
        self.requires_grad_(False)
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return feature maps from five stages.

        Args:
            x: Images in ``[-1, 1]`` range, shape ``(B, 3, H, W)``.
        """
        feats = []
        h = x
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return tuple(feats)


class RSLPIPSMetric(nn.Module):
    """LPIPS-style perceptual metric using RS-trained VGG16-BN features.

    Computes the mean L2 distance (with channel normalisation) across
    five VGG feature stages between two images.
    """

    def __init__(self, checkpoint_path: str, num_classes: int, device: str = "cpu"):
        super().__init__()
        self.features = RSVGGFeatures(checkpoint_path, num_classes, device)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute perceptual distance.

        Args:
            x, y: Images in ``[-1, 1]`` range, shape ``(B, 3, H, W)``.

        Returns:
            Scalar mean perceptual distance.
        """
        feats_x = self.features(x)
        feats_y = self.features(y)
        dist = torch.tensor(0.0, device=x.device)
        for fx, fy in zip(feats_x, feats_y):
            # Unit-normalise along channel dimension
            fx = fx / (fx.norm(dim=1, keepdim=True) + 1e-10)
            fy = fy / (fy.norm(dim=1, keepdim=True) + 1e-10)
            dist = dist + (fx - fy).pow(2).mean()
        return dist / len(feats_x)
