#!/usr/bin/env python3
"""Quick VAE reconstruction sanity check for SAR/IR/EO imagery.

The script loads a diffusers ``AutoencoderKL`` from ``./models`` (defaults to
``./models/BiliSakura/VAEs``), expands single-channel inputs to three channels
as expected by the VAE, reconstructs them, averages the three output channels
back to one channel for SAR/IR inputs, and reports MAE / PSNR / SSIM.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL, AutoencoderDC, AutoencoderKLFlux2, AutoencoderKLQwenImage
from PIL import Image

try:
    _RESAMPLING = Image.Resampling  # Pillow >= 9
except AttributeError:  # pragma: no cover - fallback for older Pillow
    _RESAMPLING = Image
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

DEFAULT_VAE_PATH = "./models/BiliSakura/VAEs"  # Repository default; override for other checkpoints.
DEFAULT_EXTS = (".png", ".tif", ".tiff")
_VAE_CLASSES = {
    "AutoencoderKL": AutoencoderKL,
    "AutoencoderDC": AutoencoderDC,
    "AutoencoderKLFlux2": AutoencoderKLFlux2,
    "AutoencoderKLQwenImage": AutoencoderKLQwenImage,
}


def _normalize_image_array(arr: np.ndarray) -> np.ndarray:
    """Normalize a H×W×C image array to [0, 1] float32."""
    arr = arr.astype(np.float32)
    arr_max = arr.max()
    if arr_max > 1.0:
        arr = arr / (65535.0 if arr_max > 255.0 else 255.0)
    return arr


def load_image_to_three_channels(
    path: Path, resolution: int, device: torch.device
) -> Tuple[torch.Tensor, int]:
    """Load an image, resize, and expand to three channels for the VAE.

    Returns
    -------
    tensor : torch.Tensor
        Tensor of shape (1, 3, H, W) in [0, 1] on the requested device.
    orig_channels : int
        Number of channels in the source image before expansion.
    """
    img = Image.open(path)
    if resolution <= 0:
        raise ValueError("resolution must be positive when provided")
    img = img.resize((resolution, resolution), _RESAMPLING.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = _normalize_image_array(arr)

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    orig_channels = arr.shape[2]

    if orig_channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif orig_channels == 2:
        # Keep both channels and duplicate the first to reach three channels: [c0, c1, c0].
        arr = np.concatenate([arr, arr[:, :, :1]], axis=2)
    elif orig_channels > 3:
        arr = arr[:, :, :3]

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, orig_channels


def reduce_reconstruction_channels(recon: torch.Tensor, orig_channels: int) -> torch.Tensor:
    """Convert a 3-channel reconstruction back to the original channel layout."""
    if orig_channels == 1:
        return recon.mean(dim=1, keepdim=True)
    desired_channels = min(orig_channels, recon.shape[1])
    return recon[:, :desired_channels]


def compute_reconstruction_metrics(
    recon: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    """Compute MAE, PSNR, and SSIM for a single reconstructed sample."""
    recon = recon.clamp(0, 1)
    target = target.clamp(0, 1)
    _, _, h, w = recon.shape
    min_side = min(h, w)
    # SSIM expects an odd kernel; cap at 11 to match the torchmetrics default while
    # staying robust for small inputs.
    if min_side < 3:
        kernel = 1
    else:
        kernel = min(min_side, 11)
        if kernel % 2 == 0:
            kernel = kernel - 1
    use_gaussian = min_side >= 11

    mae = torch.mean(torch.abs(recon - target)).item()
    psnr = peak_signal_noise_ratio(recon, target, data_range=1.0).item()
    ssim = structural_similarity_index_measure(
        recon, target, data_range=1.0, kernel_size=kernel, gaussian_kernel=use_gaussian
    ).item()
    return {"mae": float(mae), "psnr": float(psnr), "ssim": float(ssim)}


def _load_vae_from_path(vae_path: Path, device: torch.device) -> Tuple[torch.nn.Module, float, str]:
    """Load a VAE checkpoint, detecting the correct class from ``config.json``."""
    config_json = vae_path / "config.json"
    vae_class = AutoencoderKL
    scaling_factor = 1.0
    class_name = "AutoencoderKL"

    if config_json.exists():
        with config_json.open() as fh:
            cfg = json.load(fh)
        class_name = cfg.get("_class_name", class_name)
        scaling_factor = float(cfg.get("scaling_factor", scaling_factor))
        vae_class = _VAE_CLASSES.get(class_name, AutoencoderKL)

    model = vae_class.from_pretrained(str(vae_path)).to(device)
    model.eval()
    return model, scaling_factor, class_name


def _collect_image_paths(root: Path, exts: Sequence[str]) -> List[Path]:
    """Return a sorted list of image paths under ``root``."""
    wanted = {ext.lower() for ext in exts}
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted
    )


def _save_reconstruction(
    path: Path, recon: torch.Tensor, orig_channels: int, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    recon = recon.squeeze(0).detach().cpu()
    if orig_channels == 1:
        recon_np = recon[0].clamp(0, 1).numpy()
        img = Image.fromarray((recon_np * 255).astype(np.uint8), mode="L")
    else:
        recon_np = recon.permute(1, 2, 0).clamp(0, 1).numpy()
        img = Image.fromarray((recon_np * 255).astype(np.uint8))
    img.save(out_dir / path.name)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae_path = Path(args.vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(
            f"VAE path not found: {vae_path}. Pass --vae-path to point to your checkpoint if different."
        )

    vae, scaling_factor, vae_class_name = _load_vae_from_path(vae_path, device)

    image_paths = _collect_image_paths(Path(args.input_dir), args.extensions)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(
            f"No images with extensions {args.extensions} found under {args.input_dir}"
        )

    results: List[Dict[str, float]] = []
    for path in image_paths:
        input_tensor, orig_channels = load_image_to_three_channels(
            path, args.resolution, device
        )
        vae_input = input_tensor * 2 - 1

        with torch.no_grad():
            enc_in = vae_input
            if vae_class_name == "AutoencoderKLQwenImage" and enc_in.dim() == 4:
                enc_in = enc_in.unsqueeze(2)
            encoded = vae.encode(enc_in)
            if hasattr(encoded, "latent_dist"):
                latents = encoded.latent_dist.mean
            elif hasattr(encoded, "latent"):
                latents = encoded.latent
            else:
                latents = encoded
            latents = latents * scaling_factor

            if vae_class_name == "AutoencoderKLQwenImage" and latents.dim() == 4:
                latents = latents.unsqueeze(2)
            decoded = vae.decode(latents / scaling_factor)
            if hasattr(decoded, "sample"):
                decoded = decoded.sample
            if vae_class_name == "AutoencoderKLQwenImage" and decoded.dim() == 5:
                decoded = decoded.squeeze(2)
            decoded = decoded[..., : input_tensor.shape[-2], : input_tensor.shape[-1]]

        recon = (decoded.clamp(-1, 1) + 1) / 2
        recon_for_metrics = reduce_reconstruction_channels(recon, orig_channels)
        target_for_metrics = input_tensor[:, : recon_for_metrics.shape[1]]

        metrics = compute_reconstruction_metrics(recon_for_metrics, target_for_metrics)
        results.append(metrics)

        print(
            f"[{path.name}] MAE: {metrics['mae']:.4f} | "
            f"PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}"
        )

        if args.output_dir:
            _save_reconstruction(path, recon_for_metrics, orig_channels, Path(args.output_dir))

    # Summary
    mae_values = [m["mae"] for m in results]
    ssim_values = [m["ssim"] for m in results]
    mse_values = []
    for m in results:
        psnr_val = m["psnr"]
        if math.isfinite(psnr_val):
            mse_values.append(10 ** (-psnr_val / 10.0))
        else:
            mse_values.append(0.0)

    mean_mae = np.mean(mae_values)
    mean_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)
    # PSNR = 10 * log10(MAX^2 / MSE) with MAX=1.0 for normalized images.
    eps = 1e-12
    mean_psnr = float("inf") if avg_mse <= eps else 10 * math.log10(1.0 / max(avg_mse, eps))
    print(
        f"\nAveraged over {len(results)} image(s): "
        f"MAE={mean_mae:.4f}, PSNR={mean_psnr:.2f}, SSIM={mean_ssim:.4f}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing SAR/IR/EO images to reconstruct.",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default=DEFAULT_VAE_PATH,
        help="Path to the diffusers AutoencoderKL (defaults to ./models/BiliSakura/VAEs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device string (default: "cuda" if available else "cpu").',
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resize resolution fed to the VAE (square).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTS),
        help="Image extensions to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help("Optional directory to write reconstructed outputs."),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help("Optionally limit the number of images processed."),
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(args=argv)
    run(args)


if __name__ == "__main__":
    main()
