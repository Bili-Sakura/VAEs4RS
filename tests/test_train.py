"""Tests for single-channel VAE modification and training utilities."""

import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from diffusers import AutoencoderKL

# Import the train module directly (avoid triggering heavy src/__init__.py)
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import train as _train_mod  # noqa: E402

replace_encoder_conv_in = _train_mod.replace_encoder_conv_in
replace_decoder_conv_out = _train_mod.replace_decoder_conv_out
prepare_vae_for_training = _train_mod.prepare_vae_for_training
get_trainable_parameters = _train_mod.get_trainable_parameters
log_trainable_summary = _train_mod.log_trainable_summary
SingleChannelRSDataset = _train_mod.SingleChannelRSDataset
vae_loss = _train_mod.vae_loss


# ---- Fixtures ------------------------------------------------------------

@pytest.fixture(scope="module")
def vae_config():
    """Minimal AutoencoderKL config for fast testing."""
    return dict(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 64),
        latent_channels=4,
        layers_per_block=1,
    )


@pytest.fixture()
def vae(vae_config):
    """Fresh small VAE for each test."""
    return AutoencoderKL(**vae_config)


# ---- Tests: conv replacement ---------------------------------------------

class TestReplaceEncoderConvIn:
    def test_changes_in_channels(self, vae):
        assert vae.encoder.conv_in.in_channels == 3
        replace_encoder_conv_in(vae.encoder, in_channels=1)
        assert vae.encoder.conv_in.in_channels == 1

    def test_preserves_out_channels(self, vae):
        out_ch = vae.encoder.conv_in.out_channels
        replace_encoder_conv_in(vae.encoder, in_channels=1)
        assert vae.encoder.conv_in.out_channels == out_ch

    def test_weight_shape(self, vae):
        replace_encoder_conv_in(vae.encoder, in_channels=1)
        w = vae.encoder.conv_in.weight
        assert w.shape[1] == 1  # in_channels

    def test_bias_copied(self, vae):
        old_bias = vae.encoder.conv_in.bias.clone()
        replace_encoder_conv_in(vae.encoder, in_channels=1)
        assert torch.allclose(vae.encoder.conv_in.bias, old_bias)


class TestReplaceDecoderConvOut:
    def test_changes_out_channels(self, vae):
        assert vae.decoder.conv_out.out_channels == 3
        replace_decoder_conv_out(vae.decoder, out_channels=1)
        assert vae.decoder.conv_out.out_channels == 1

    def test_preserves_in_channels(self, vae):
        in_ch = vae.decoder.conv_out.in_channels
        replace_decoder_conv_out(vae.decoder, out_channels=1)
        assert vae.decoder.conv_out.in_channels == in_ch

    def test_weight_shape(self, vae):
        replace_decoder_conv_out(vae.decoder, out_channels=1)
        w = vae.decoder.conv_out.weight
        assert w.shape[0] == 1  # out_channels


# ---- Tests: prepare_vae_for_training ------------------------------------

class TestPrepareVaeForTraining:
    def test_single_channel_io(self, vae):
        prepare_vae_for_training(vae, in_channels=1, out_channels=1)
        assert vae.encoder.conv_in.in_channels == 1
        assert vae.decoder.conv_out.out_channels == 1

    def test_most_params_frozen(self, vae):
        prepare_vae_for_training(
            vae, in_channels=1, out_channels=1,
            trainable_encoder_blocks=0, trainable_decoder_blocks=0,
        )
        total = sum(p.numel() for p in vae.parameters())
        trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        # With 0 extra blocks, only conv_in, conv_norm_out, conv_out are trainable
        assert trainable < total
        assert trainable > 0

    def test_trainable_params_exist(self, vae):
        prepare_vae_for_training(vae, in_channels=1, out_channels=1)
        params = get_trainable_parameters(vae)
        assert len(params) > 0

    def test_encoder_conv_in_trainable(self, vae):
        prepare_vae_for_training(vae, in_channels=1, out_channels=1)
        for p in vae.encoder.conv_in.parameters():
            assert p.requires_grad

    def test_decoder_conv_out_trainable(self, vae):
        prepare_vae_for_training(vae, in_channels=1, out_channels=1)
        for p in vae.decoder.conv_out.parameters():
            assert p.requires_grad

    def test_decoder_conv_norm_out_trainable(self, vae):
        prepare_vae_for_training(vae, in_channels=1, out_channels=1)
        for p in vae.decoder.conv_norm_out.parameters():
            assert p.requires_grad

    def test_mid_block_frozen(self, vae):
        prepare_vae_for_training(
            vae, in_channels=1, out_channels=1,
            trainable_encoder_blocks=0, trainable_decoder_blocks=0,
        )
        for p in vae.encoder.mid_block.parameters():
            assert not p.requires_grad
        for p in vae.decoder.mid_block.parameters():
            assert not p.requires_grad

    def test_quant_conv_frozen(self, vae):
        prepare_vae_for_training(vae, in_channels=1, out_channels=1)
        for p in vae.quant_conv.parameters():
            assert not p.requires_grad
        for p in vae.post_quant_conv.parameters():
            assert not p.requires_grad


# ---- Tests: log_trainable_summary ----------------------------------------

def test_log_trainable_summary(vae):
    prepare_vae_for_training(vae, in_channels=1, out_channels=1)
    trainable, total = log_trainable_summary(vae)
    assert 0 < trainable < total


# ---- Tests: forward pass -------------------------------------------------

class TestForwardPass:
    def test_encode_decode_single_channel(self, vae):
        prepare_vae_for_training(vae, in_channels=1, out_channels=1)
        vae.eval()
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            posterior = vae.encode(x).latent_dist
            z = posterior.mode()
            out = vae.decode(z).sample
        assert out.shape[0] == 2
        assert out.shape[1] == 1  # single-channel output

    def test_vae_loss_runs(self, vae):
        prepare_vae_for_training(vae, in_channels=1, out_channels=1)
        vae.train()
        x = torch.randn(2, 1, 64, 64)
        loss, metrics = vae_loss(vae, x, reconstruction_weight=1.0, kl_weight=1e-6)
        assert loss.shape == ()
        assert loss.requires_grad
        assert "recon_loss" in metrics
        assert "kl_loss" in metrics
        assert "total_loss" in metrics


# ---- Tests: dataset (basic) ----------------------------------------------

class TestSingleChannelRSDataset:
    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SingleChannelRSDataset(str(tmp_path / "nonexistent"))

    def test_empty_root_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError):
            SingleChannelRSDataset(str(empty_dir))

    def test_loads_grayscale_images(self, tmp_path):
        from PIL import Image
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        for i in range(3):
            Image.new("L", (64, 64)).save(img_dir / f"img{i}.png")

        ds = SingleChannelRSDataset(str(img_dir), image_size=32)
        assert len(ds) == 3
        sample = ds[0]
        assert sample.shape == (1, 32, 32)

    def test_rgb_converted_to_grayscale(self, tmp_path):
        from PIL import Image
        img_dir = tmp_path / "rgb"
        img_dir.mkdir()
        Image.new("RGB", (64, 64)).save(img_dir / "color.png")

        ds = SingleChannelRSDataset(str(img_dir), image_size=32)
        assert ds[0].shape == (1, 32, 32)
