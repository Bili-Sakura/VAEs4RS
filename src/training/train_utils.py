"""
Training utilities for fine-tuning VAE models on remote sensing images.

Supports all VAE architectures in the repository (AutoencoderKL, AutoencoderDC,
AutoencoderKLQwenImage, AutoencoderKLFlux2). For AutoencoderKL-family models,
supports modifying input/output conv layers for single-channel I/O and selective
layer freezing. All architectures support full fine-tuning.
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL, AutoencoderDC, AutoencoderKLQwenImage, AutoencoderKLFlux2

logger = logging.getLogger(__name__)

# VAE class mapping (same as in models/vae_wrapper.py)
VAE_CLASSES = {
    "AutoencoderKL": AutoencoderKL,
    "AutoencoderDC": AutoencoderDC,
    "AutoencoderKLQwenImage": AutoencoderKLQwenImage,
    "AutoencoderKLFlux2": AutoencoderKLFlux2,
}

# Architectures that support conv_in/conv_out replacement and block-level freezing
_BLOCK_FREEZE_ARCHITECTURES = (AutoencoderKL, AutoencoderKLFlux2, AutoencoderKLQwenImage)


def load_vae_for_training(
    pretrained_path: str,
    subfolder: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    """Load a VAE model for training, auto-detecting the architecture.

    Args:
        pretrained_path: HuggingFace model ID or local path.
        subfolder: Optional subfolder within the model directory.

    Returns:
        (model, vae_class_name) tuple.
    """
    ckpt_path = Path(pretrained_path)
    config_json = ckpt_path / "config.json"
    if subfolder:
        config_json = ckpt_path / subfolder / "config.json"

    vae_class = AutoencoderKL
    vae_class_name = "AutoencoderKL"
    if config_json.exists():
        with open(config_json, "r") as f:
            model_cfg = json.load(f)
        class_name = model_cfg.get("_class_name", "AutoencoderKL")
        if class_name in VAE_CLASSES:
            vae_class = VAE_CLASSES[class_name]
            vae_class_name = class_name

    load_kwargs = dict(ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
    if subfolder:
        model = vae_class.from_pretrained(str(ckpt_path), subfolder=subfolder, **load_kwargs)
    else:
        model = vae_class.from_pretrained(str(ckpt_path), **load_kwargs)

    return model, vae_class_name


# ---------------------------------------------------------------------------
# Model modification utilities
# ---------------------------------------------------------------------------

def replace_encoder_conv_in(encoder: nn.Module, in_channels: int = 1) -> None:
    """Replace the encoder's first conv layer to accept `in_channels` input.

    Initializes the new weights by averaging the pre-trained 3-channel
    weights along the input-channel dimension.
    """
    old = encoder.conv_in
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        # Average across old input channels → (out_c, 1, kH, kW)
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    encoder.conv_in = new


def replace_decoder_conv_out(decoder: nn.Module, out_channels: int = 1) -> None:
    """Replace the decoder's last conv layer to produce `out_channels` output.

    Initializes the new weights by averaging the pre-trained 3-channel
    weights along the output-channel dimension.
    """
    old = decoder.conv_out
    new = nn.Conv2d(
        old.in_channels,
        out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        # Average across old output channels → (1, in_c, kH, kW)
        new.weight.copy_(old.weight.mean(dim=0, keepdim=True).repeat(out_channels, 1, 1, 1))
        if old.bias is not None:
            new.bias.copy_(old.bias.mean(dim=0, keepdim=True).repeat(out_channels))
    decoder.conv_out = new


def prepare_vae_for_training(
    vae: nn.Module,
    in_channels: int = 3,
    out_channels: int = 3,
    trainable_encoder_blocks: int = 1,
    trainable_decoder_blocks: int = 1,
    train_all_params: bool = False,
) -> nn.Module:
    """Prepare a pre-trained VAE for fine-tuning.

    Supports all VAE architectures in the repository. For AutoencoderKL-family
    models, optionally replaces encoder/decoder conv layers for different
    channel counts and applies selective layer freezing. For other architectures
    (e.g. AutoencoderDC), only full fine-tuning is supported.

    Args:
        vae: Pre-trained VAE model (any supported architecture).
        in_channels: Number of input channels. Conv replacement is applied
            only when this differs from the model's default *and* the
            architecture supports it.
        out_channels: Number of output channels (same conditions as above).
        trainable_encoder_blocks: Number of encoder down-blocks to unfreeze
            (counting from the input side). Ignored when *train_all_params* is
            True or the architecture does not support block-level freezing.
        trainable_decoder_blocks: Number of decoder up-blocks to unfreeze
            (counting from the output side). Same conditions as above.
        train_all_params: If True, all parameters are trainable (full fine-tuning).

    Steps for AutoencoderKL-family models:
        1. Replace encoder.conv_in for *in_channels* input (if needed).
        2. Replace decoder.conv_out for *out_channels* output (if needed).
        3. If *train_all_params*: unfreeze everything.
           Otherwise: freeze all, then selectively unfreeze:
           - encoder.conv_in  (newly created)
           - first *trainable_encoder_blocks* encoder down-blocks
           - last *trainable_decoder_blocks* decoder up-blocks
           - decoder.conv_norm_out
           - decoder.conv_out  (newly created)

    For other architectures (e.g. AutoencoderDC):
        All parameters are set to trainable (full fine-tuning).
    """
    supports_block_freeze = isinstance(vae, _BLOCK_FREEZE_ARCHITECTURES)

    # --- Conv replacement (only for supported architectures) ---
    if supports_block_freeze and hasattr(vae, "encoder") and hasattr(vae.encoder, "conv_in"):
        current_in = vae.encoder.conv_in.in_channels
        if in_channels != current_in:
            replace_encoder_conv_in(vae.encoder, in_channels)

    if supports_block_freeze and hasattr(vae, "decoder") and hasattr(vae.decoder, "conv_out"):
        current_out = vae.decoder.conv_out.out_channels
        if out_channels != current_out:
            replace_decoder_conv_out(vae.decoder, out_channels)

    if train_all_params or not supports_block_freeze:
        # --- Full fine-tuning: all parameters trainable ---
        vae.requires_grad_(True)
    else:
        # --- Partial fine-tuning (AutoencoderKL-family only) ---
        # Freeze everything first
        vae.requires_grad_(False)

        # Selectively unfreeze encoder input layers
        if hasattr(vae, "encoder") and hasattr(vae.encoder, "conv_in"):
            for p in vae.encoder.conv_in.parameters():
                p.requires_grad = True

        if hasattr(vae, "encoder") and hasattr(vae.encoder, "down_blocks"):
            n_down = len(vae.encoder.down_blocks)
            for i in range(min(trainable_encoder_blocks, n_down)):
                for p in vae.encoder.down_blocks[i].parameters():
                    p.requires_grad = True

        # Selectively unfreeze decoder output layers
        if hasattr(vae, "decoder") and hasattr(vae.decoder, "conv_norm_out"):
            for p in vae.decoder.conv_norm_out.parameters():
                p.requires_grad = True
        if hasattr(vae, "decoder") and hasattr(vae.decoder, "conv_out"):
            for p in vae.decoder.conv_out.parameters():
                p.requires_grad = True

        if hasattr(vae, "decoder") and hasattr(vae.decoder, "up_blocks"):
            n_up = len(vae.decoder.up_blocks)
            for i in range(max(0, n_up - trainable_decoder_blocks), n_up):
                for p in vae.decoder.up_blocks[i].parameters():
                    p.requires_grad = True

    return vae


def get_trainable_parameters(vae: nn.Module) -> List[nn.Parameter]:
    """Return only the trainable parameters of the VAE."""
    return [p for p in vae.parameters() if p.requires_grad]


def log_trainable_summary(vae: nn.Module) -> Tuple[int, int]:
    """Log and return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in vae.parameters())
    trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", pct,
    )
    return trainable, total


def create_optimizer(
    params: Iterable[nn.Parameter],
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> torch.optim.Optimizer:
    """Create an optimizer from a name string and common hyperparameters."""
    params_list = list(params)
    if any(isinstance(p, dict) for p in params_list):
        raise TypeError(
            "create_optimizer expects raw parameters; pass an iterable of parameters and let "
            "the helper build any required parameter groups internally."
        )
    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params_list,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
    elif name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError as exc:
            raise ImportError(
                "Prodigy optimizer requested but prodigyopt is not installed. "
                "Install with `pip install prodigyopt`."
            ) from exc
        return Prodigy(
            params_list,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
    elif name == "muon":
        try:
            from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
        except ImportError as exc:
            raise ImportError(
                "Muon optimizer requested but muon is not installed. "
                "Install with `pip install git+https://github.com/KellerJordan/Muon`."
            ) from exc
        # Muon is intended for matrix-like weights; biases/norms use AdamW.
        muon_params = [p for p in params_list if p.ndim >= 2]
        aux_params = [p for p in params_list if p.ndim < 2]
        if not muon_params:
            raise ValueError(
                "Muon optimizer requires parameters with ndim >= 2; "
                "none were found in the provided parameter list."
            )
        # Muon param groups use standard optimizer keys plus the Muon-specific "use_muon" flag.
        muon_group = dict(
            params=muon_params,
            use_muon=True,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        param_groups = [muon_group]
        if aux_params:
            param_groups.append(
                dict(
                    params=aux_params,
                    use_muon=False,
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay,
                )
            )
        if torch.distributed.is_initialized():
            optimizer_cls = MuonWithAuxAdam
        else:
            optimizer_cls = SingleDeviceMuonWithAuxAdam
        return optimizer_cls(param_groups)
    raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. "
        "Choose from: adamw, muon, prodigy."
    )


# ---------------------------------------------------------------------------
# Single-channel remote sensing dataset
# ---------------------------------------------------------------------------

class SingleChannelRSDataset(Dataset):
    """Dataset for single-channel remote sensing images (IR / SAR / EO).

    Loads images as grayscale from a directory tree::

        root/
            classA/
                img001.png
                img002.tif
            classB/
                ...

    or a flat directory of images (no class sub-folders).
    """

    EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.image_paths = self._scan(self.root)
        if not self.image_paths:
            raise ValueError(f"No images found in {self.root}")

        self.transform = transform or self._default_transform(image_size)

    # -- scanning ----------------------------------------------------------

    def _scan(self, root: Path) -> List[str]:
        paths: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in sorted(filenames):
                if fn.lower().endswith(self.EXTENSIONS):
                    paths.append(os.path.join(dirpath, fn))
        return paths

    # -- transforms --------------------------------------------------------

    @staticmethod
    def _default_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),                    # [0, 1]
            transforms.Normalize([0.5], [0.5]),       # [-1, 1]
        ])

    # -- loading -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        image = Image.open(path).convert("L")         # force grayscale
        return self.transform(image)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def vae_loss(
    vae: nn.Module,
    images: torch.Tensor,
    reconstruction_weight: float = 1.0,
    kl_weight: float = 1e-6,
) -> Tuple[torch.Tensor, dict]:
    """Compute VAE reconstruction + KL loss for any supported architecture.

    Handles different encode APIs:
    - AutoencoderKL / AutoencoderKLFlux2 / AutoencoderKLQwenImage: latent_dist
    - AutoencoderDC: latent (no KL divergence)

    Args:
        vae: VAE model (must be in train mode for unfrozen layers).
        images: (B, C, H, W) tensor in [-1, 1].
        reconstruction_weight: weight for MSE reconstruction loss.
        kl_weight: weight for KL divergence loss.

    Returns:
        (total_loss, {recon_loss, kl_loss, total_loss})
    """
    # Encode
    encoded = vae.encode(images)
    has_latent_dist = hasattr(encoded, "latent_dist") and encoded.latent_dist is not None

    if has_latent_dist:
        posterior = encoded.latent_dist
        z = posterior.sample()
    elif hasattr(encoded, "latent"):
        z = encoded.latent
        posterior = None
    else:
        z = encoded
        posterior = None

    # Decode
    decoded = vae.decode(z)
    reconstructed = decoded.sample if hasattr(decoded, "sample") else decoded

    # Crop to match input (in case of padding artifacts)
    if reconstructed.shape != images.shape:
        reconstructed = reconstructed[..., : images.shape[-2], : images.shape[-1]]

    # Losses
    recon_loss = F.mse_loss(reconstructed, images)

    if posterior is not None:
        kl_loss = posterior.kl().mean()
    else:
        kl_loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)

    total = reconstruction_weight * recon_loss + kl_weight * kl_loss
    return total, {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total.item(),
    }
