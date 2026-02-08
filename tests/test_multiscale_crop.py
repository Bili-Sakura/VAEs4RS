"""Tests for MultiScaleCrop augmentation and RSDataset integration."""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.datasets import MultiScaleCrop, get_multiscale_train_transform, RSDataset


# ---- Tests: MultiScaleCrop transform ------------------------------------

class TestMultiScaleCrop:
    def test_crop_large_image_produces_valid_size(self):
        """A 1024px image should be cropped to 256 or 512."""
        crop = MultiScaleCrop(crop_sizes=(256, 512), crop_probs=(0.8, 0.2))
        img = Image.new("RGB", (1024, 1024))
        result = crop(img)
        assert result.size in [(256, 256), (512, 512)]

    def test_small_image_unchanged(self):
        """Image smaller than all crop sizes is returned unchanged."""
        crop = MultiScaleCrop(crop_sizes=(256, 512), crop_probs=(0.8, 0.2))
        img = Image.new("RGB", (128, 128))
        result = crop(img)
        assert result.size == (128, 128)

    def test_256px_image_returns_256(self):
        """A 256px image can only be cropped to 256 (passes through)."""
        crop = MultiScaleCrop(crop_sizes=(256, 512), crop_probs=(0.8, 0.2))
        img = Image.new("RGB", (256, 256))
        result = crop(img)
        assert result.size == (256, 256)

    def test_512px_image_produces_valid_size(self):
        """A 512px image can be cropped to 256 or 512."""
        crop = MultiScaleCrop(crop_sizes=(256, 512), crop_probs=(0.8, 0.2))
        img = Image.new("RGB", (512, 512))
        result = crop(img)
        assert result.size in [(256, 256), (512, 512)]

    def test_probability_distribution(self):
        """Over many samples, 256 crops should be much more frequent than 512."""
        crop = MultiScaleCrop(crop_sizes=(256, 512), crop_probs=(0.8, 0.2))
        img = Image.new("RGB", (1024, 1024))
        counts = {256: 0, 512: 0}
        for _ in range(500):
            result = crop(img)
            counts[result.size[0]] += 1
        # 80% of 500 = 400; allow generous margin
        assert counts[256] > counts[512]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            MultiScaleCrop(crop_sizes=(256,), crop_probs=(0.8, 0.2))

    def test_non_square_image(self):
        """Non-square images: crop size limited by the smaller dimension."""
        crop = MultiScaleCrop(crop_sizes=(256, 512), crop_probs=(0.8, 0.2))
        img = Image.new("RGB", (1024, 300))  # min dim = 300, so only 256 fits
        result = crop(img)
        assert result.size == (256, 256)

    def test_repr(self):
        crop = MultiScaleCrop(crop_sizes=(256, 512), crop_probs=(0.8, 0.2))
        assert "MultiScaleCrop" in repr(crop)
        assert "256" in repr(crop)


# ---- Tests: get_multiscale_train_transform -------------------------------

class TestGetMultiscaleTrainTransform:
    def test_returns_compose(self):
        t = get_multiscale_train_transform()
        from torchvision.transforms import Compose
        assert isinstance(t, Compose)

    def test_output_is_normalized_tensor(self):
        t = get_multiscale_train_transform(crop_sizes=(64,), crop_probs=(1.0,))
        img = Image.new("RGB", (128, 128), color=(128, 128, 128))
        tensor = t(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)
        # Should be in [-1, 1] range after normalization
        assert tensor.min() >= -1.0 - 1e-5
        assert tensor.max() <= 1.0 + 1e-5


# ---- Tests: RSDataset with multi_scale_crop ------------------------------

class TestRSDatasetMultiScaleCrop:
    def _create_dataset_dir(self, tmp_path, image_size=1024, num_images=3):
        """Helper to create a class-based dataset directory."""
        class_dir = tmp_path / "classA"
        class_dir.mkdir()
        for i in range(num_images):
            Image.new("RGB", (image_size, image_size)).save(class_dir / f"img{i}.png")
        return tmp_path

    def test_multi_scale_crop_enabled(self, tmp_path):
        root = self._create_dataset_dir(tmp_path, image_size=1024)
        ds = RSDataset(str(root), multi_scale_crop=True)
        img_tensor, label, path = ds[0]
        assert img_tensor.shape[0] == 3  # RGB channels
        assert img_tensor.shape[1] in (256, 512)
        assert img_tensor.shape[2] in (256, 512)

    def test_multi_scale_crop_disabled_uses_default(self, tmp_path):
        root = self._create_dataset_dir(tmp_path, image_size=1024)
        ds = RSDataset(str(root), image_size=256, multi_scale_crop=False)
        img_tensor, label, path = ds[0]
        assert img_tensor.shape == (3, 256, 256)

    def test_custom_transform_takes_precedence(self, tmp_path):
        """Explicit transform overrides multi_scale_crop."""
        from torchvision import transforms
        custom = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ])
        root = self._create_dataset_dir(tmp_path, image_size=1024)
        ds = RSDataset(str(root), transform=custom, multi_scale_crop=True)
        img_tensor, _, _ = ds[0]
        assert img_tensor.shape == (3, 64, 64)

    def test_custom_crop_sizes(self, tmp_path):
        root = self._create_dataset_dir(tmp_path, image_size=1024)
        ds = RSDataset(
            str(root),
            multi_scale_crop=True,
            crop_sizes=(128, 256),
            crop_probs=(0.5, 0.5),
        )
        img_tensor, _, _ = ds[0]
        assert img_tensor.shape[1] in (128, 256)

    def test_256px_source_kept_with_multi_scale(self, tmp_path):
        """256px source images are kept at 256 when multi_scale_crop is on."""
        root = self._create_dataset_dir(tmp_path, image_size=256)
        ds = RSDataset(str(root), multi_scale_crop=True)
        img_tensor, _, _ = ds[0]
        assert img_tensor.shape == (3, 256, 256)
