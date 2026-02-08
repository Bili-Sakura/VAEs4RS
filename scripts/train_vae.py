#!/usr/bin/env python3
"""
Train (fine-tune) any VAE model on remote sensing images.

Supports all VAE architectures in the repository:
- AutoencoderKL (SD21-VAE, SDXL-VAE, SD35-VAE)
- AutoencoderKLFlux2 (FLUX2-VAE)
- AutoencoderKLQwenImage (Qwen-VAE)
- AutoencoderDC (SANA-VAE)
- AutoencoderKL (FLUX1-VAE)

For single-channel remote sensing (IR, SAR, EO), set model.in_channels and
model.out_channels to 1 in the config. Conv layer replacement is automatically
applied for AutoencoderKL-family models.

Usage:
    # Single GPU
    python scripts/train_vae.py --config configs/train_rs_vae.yaml

    # Multi-GPU with accelerate
    accelerate launch scripts/train_vae.py --config configs/train_rs_vae.yaml

    # Override specific settings
    python scripts/train_vae.py --config configs/train_vae.yaml \
        --pretrained_path stabilityai/sd-vae-ft-mse \
        --train_dir datasets/sar/train \
        --output_dir outputs/sar_vae \
        --num_epochs 50
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.training.train_utils import (
    load_vae_for_training,
    prepare_vae_for_training,
    get_trainable_parameters,
    log_trainable_summary,
    vae_loss,
    create_optimizer,
    SingleChannelRSDataset,
)

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_train_config(path: str) -> dict:
    """Load YAML training config and return a flat namespace-like dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune any VAE for remote sensing images."
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_rs_vae.yaml",
        help="Path to training config YAML.",
    )
    # Allow CLI overrides for the most common settings
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--train_all_params", action="store_true", default=None,
        help="Train all parameters (full fine-tuning) instead of only selected layers.",
    )
    return parser.parse_args()


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with non-None CLI arguments."""
    overrides = {
        "model.pretrained_path": args.pretrained_path,
        "data.train_dir": args.train_dir,
        "data.val_dir": args.val_dir,
        "training.output_dir": args.output_dir,
        "training.num_epochs": args.num_epochs,
        "training.batch_size": args.batch_size,
        "training.learning_rate": args.learning_rate,
        "training.seed": args.seed,
        "training.resume_from_checkpoint": args.resume_from_checkpoint,
        "freeze.train_all_params": args.train_all_params,
    }
    for dotkey, value in overrides.items():
        if value is None:
            continue
        keys = dotkey.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(cfg: dict, *keys, default=None):
    """Nested dict get: _get(cfg, 'training', 'seed', default=42)."""
    d = cfg
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_train_config(args.config)
    cfg = merge_args(cfg, args)

    # Shorthand accessors
    seed = _get(cfg, "training", "seed", default=42)
    output_dir = _get(cfg, "training", "output_dir", default="outputs/train_vae")
    num_epochs = _get(cfg, "training", "num_epochs", default=100)
    batch_size = _get(cfg, "training", "batch_size", default=8)
    grad_accum = _get(cfg, "training", "gradient_accumulation_steps", default=1)
    mixed_precision = _get(cfg, "training", "mixed_precision", default="bf16")
    lr = _get(cfg, "training", "learning_rate", default=1e-4)
    optimizer_name = _get(cfg, "training", "optimizer", default="adamw")
    weight_decay = _get(cfg, "training", "weight_decay", default=0.01)
    adam_beta1 = _get(cfg, "training", "adam_beta1", default=0.9)
    adam_beta2 = _get(cfg, "training", "adam_beta2", default=0.999)
    max_grad_norm = _get(cfg, "training", "max_grad_norm", default=1.0)
    lr_scheduler_type = _get(cfg, "training", "lr_scheduler", default="cosine")
    warmup_steps = _get(cfg, "training", "warmup_steps", default=500)
    recon_w = _get(cfg, "training", "reconstruction_weight", default=1.0)
    kl_w = _get(cfg, "training", "kl_weight", default=1e-6)
    log_every = _get(cfg, "training", "log_every_n_steps", default=100)
    save_every = _get(cfg, "training", "save_every_n_epochs", default=10)
    resume_ckpt = _get(cfg, "training", "resume_from_checkpoint")

    pretrained_path = _get(cfg, "model", "pretrained_path", default="stabilityai/sd-vae-ft-mse")
    subfolder = _get(cfg, "model", "subfolder")
    in_channels = _get(cfg, "model", "in_channels", default=3)
    out_channels = _get(cfg, "model", "out_channels", default=3)

    trainable_enc = _get(cfg, "freeze", "trainable_encoder_blocks", default=1)
    trainable_dec = _get(cfg, "freeze", "trainable_decoder_blocks", default=1)
    train_all_params = _get(cfg, "freeze", "train_all_params", default=False)
    asymmetric = _get(cfg, "freeze", "asymmetric", default=False)

    train_dir = _get(cfg, "data", "train_dir")
    val_dir = _get(cfg, "data", "val_dir")
    image_size = _get(cfg, "data", "image_size", default=256)
    num_workers = _get(cfg, "data", "num_workers", default=4)

    # ---- Accelerator -----------------------------------------------------
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=output_dir,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if seed is not None:
        set_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # ---- Model -----------------------------------------------------------
    logger.info("Loading pre-trained VAE from %s", pretrained_path)
    vae, vae_class_name = load_vae_for_training(pretrained_path, subfolder=subfolder)
    logger.info("Detected VAE architecture: %s", vae_class_name)

    logger.info(
        "Preparing VAE for training (in=%d, out=%d, train_all_params=%s, asymmetric=%s)",
        in_channels, out_channels, train_all_params, asymmetric,
    )
    vae = prepare_vae_for_training(
        vae,
        in_channels=in_channels,
        out_channels=out_channels,
        trainable_encoder_blocks=trainable_enc,
        trainable_decoder_blocks=trainable_dec,
        train_all_params=train_all_params,
        asymmetric=asymmetric,
    )
    trainable_count, total_count = log_trainable_summary(vae)

    # ---- Dataset ---------------------------------------------------------
    if train_dir is None:
        raise ValueError("data.train_dir must be specified in the config or via --train_dir")

    train_dataset = SingleChannelRSDataset(train_dir, image_size=image_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info("Train dataset: %d images from %s", len(train_dataset), train_dir)

    val_dataloader = None
    if val_dir:
        val_dataset = SingleChannelRSDataset(val_dir, image_size=image_size)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info("Val dataset: %d images from %s", len(val_dataset), val_dir)

    # ---- Optimizer & Scheduler -------------------------------------------
    trainable_params = get_trainable_parameters(vae)
    optimizer_obj = create_optimizer(
        trainable_params,
        optimizer_name=optimizer_name,
        learning_rate=lr,
        weight_decay=weight_decay,
        beta1=adam_beta1,
        beta2=adam_beta2,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / grad_accum
    )
    max_train_steps = num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer_obj,
        num_warmup_steps=warmup_steps * grad_accum,
        num_training_steps=max_train_steps * grad_accum,
    )

    # ---- Prepare with Accelerator ----------------------------------------
    vae, optimizer_obj, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer_obj, train_dataloader, lr_scheduler,
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # ---- Resume -----------------------------------------------------------
    global_step = 0
    start_epoch = 0
    if resume_ckpt and os.path.isdir(resume_ckpt):
        logger.info("Resuming from checkpoint: %s", resume_ckpt)
        accelerator.load_state(resume_ckpt)
        step_str = os.path.basename(resume_ckpt).replace("checkpoint-", "")
        global_step = int(step_str) if step_str.isdigit() else 0
        start_epoch = global_step // num_update_steps_per_epoch

    # ---- Logging ----------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers("vae_training", config=cfg)

    # ---- Training Loop ----------------------------------------------------
    logger.info("***** Starting Training *****")
    logger.info("  VAE architecture = %s", vae_class_name)
    logger.info("  Trainable params = %s / %s (%.2f%%)",
                f"{trainable_count:,}", f"{total_count:,}",
                100.0 * trainable_count / total_count if total_count > 0 else 0.0)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num epochs = %d", num_epochs)
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Gradient accumulation steps = %d", grad_accum)
    logger.info("  Total optimization steps = %d", max_train_steps)

    for epoch in range(start_epoch, num_epochs):
        vae.train()
        epoch_loss = 0.0

        progress = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for step, images in enumerate(progress):
            with accelerator.accumulate(vae):
                loss, metrics = vae_loss(
                    vae, images,
                    reconstruction_weight=recon_w,
                    kl_weight=kl_w,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, max_grad_norm)

                optimizer_obj.step()
                lr_scheduler.step()
                optimizer_obj.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += metrics["total_loss"]

                if global_step % log_every == 0:
                    accelerator.log(
                        {
                            "train/recon_loss": metrics["recon_loss"],
                            "train/kl_loss": metrics["kl_loss"],
                            "train/total_loss": metrics["total_loss"],
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            progress.set_postfix(loss=f"{metrics['total_loss']:.4f}")

        # -- Epoch-level logging -------------------------------------------
        # epoch_loss accumulates once per optimizer step (sync_gradients)
        avg_loss = epoch_loss / max(num_update_steps_per_epoch, 1)
        logger.info("Epoch %d – avg loss: %.5f", epoch + 1, avg_loss)

        # -- Validation ----------------------------------------------------
        if val_dataloader is not None:
            vae.eval()
            val_loss_accum = 0.0
            val_steps = 0
            with torch.no_grad():
                for images in val_dataloader:
                    _, val_metrics = vae_loss(vae, images, recon_w, kl_w)
                    val_loss_accum += val_metrics["total_loss"]
                    val_steps += 1
            avg_val = val_loss_accum / max(val_steps, 1)
            logger.info("Epoch %d – val loss: %.5f", epoch + 1, avg_val)
            if accelerator.is_main_process:
                accelerator.log({"val/total_loss": avg_val}, step=global_step)

        # -- Checkpoint ----------------------------------------------------
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(ckpt_dir)
                logger.info("Saved checkpoint to %s", ckpt_dir)

                # Also save the unwrapped VAE for easy loading later
                unwrapped = accelerator.unwrap_model(vae)
                unwrapped.save_pretrained(
                    os.path.join(output_dir, f"vae-epoch-{epoch + 1}")
                )

    # ---- End training -----------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(vae)
        unwrapped.save_pretrained(os.path.join(output_dir, "vae-final"))
        logger.info("Training complete. Final model saved to %s/vae-final", output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
