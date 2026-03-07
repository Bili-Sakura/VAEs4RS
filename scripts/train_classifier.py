#!/usr/bin/env python3
"""
Train a VGG16-BN or InceptionV3 classifier from scratch on a remote
sensing dataset (RESISC45, AID, UCMerced).

The trained model is saved as a plain ``state_dict`` and can later be
loaded by ``RSInceptionFeatures`` or ``RSVGGFeatures`` to serve as a
domain-specific feature extractor for FID(rs), KID(rs), and LPIPS(rs).

Usage:
    # Single GPU
    python scripts/train_classifier.py --config configs/train_classifier.yaml

    # Multi-GPU with Accelerate
    accelerate launch scripts/train_classifier.py \
        --config configs/train_classifier.yaml

    # Override per run
    python scripts/train_classifier.py \
        --config configs/train_classifier.yaml \
        --arch inception_v3 --dataset AID --num_epochs 100
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

# Ensure project root on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.training.classifier_utils import create_vgg16, create_inception_v3
from src.utils.datasets import RSDataset

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _get(cfg: dict, *keys, default=None):
    d = cfg
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d


def load_train_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VGG/InceptionV3 on RS data.")
    p.add_argument("--config", type=str, default="configs/train_classifier.yaml")
    p.add_argument("--arch", type=str, default=None, choices=["vgg16", "inception_v3"])
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    return p.parse_args()


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    mapping = {
        "model.arch": args.arch,
        "data.dataset": args.dataset,
        "training.output_dir": args.output_dir,
        "training.num_epochs": args.num_epochs,
        "training.batch_size": args.batch_size,
        "training.learning_rate": args.learning_rate,
        "training.seed": args.seed,
        "data.image_size": args.image_size,
    }
    for dotkey, value in mapping.items():
        if value is None:
            continue
        keys = dotkey.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return cfg


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _classification_transforms(image_size: int, is_train: bool):
    """Return augmentation / evaluation transforms for classification."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _train_val_split(dataset: RSDataset, val_ratio: float = 0.2, seed: int = 42):
    """Stratified train/val split by class label."""
    from collections import defaultdict
    import random

    rng = random.Random(seed)
    class_indices: dict = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)

    train_indices, val_indices = [], []
    for indices in class_indices.values():
        indices = indices.copy()
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _collate_fn(batch):
    """Collate that returns (images, labels) only (drops paths)."""
    images, labels, _ = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_train_config(args.config)
    cfg = merge_args(cfg, args)

    # Read config values
    arch = _get(cfg, "model", "arch", default="vgg16")
    pretrained = _get(cfg, "model", "pretrained", default=False)
    num_classes_cfg = _get(cfg, "model", "num_classes")

    seed = _get(cfg, "training", "seed", default=42)
    output_dir = _get(cfg, "training", "output_dir", default="outputs/classifiers")
    num_epochs = _get(cfg, "training", "num_epochs", default=100)
    batch_size = _get(cfg, "training", "batch_size", default=32)
    grad_accum = _get(cfg, "training", "gradient_accumulation_steps", default=1)
    mixed_precision = _get(cfg, "training", "mixed_precision", default="bf16")
    lr = _get(cfg, "training", "learning_rate", default=1e-3)
    optimizer_name = _get(cfg, "training", "optimizer", default="adamw")
    weight_decay = _get(cfg, "training", "weight_decay", default=1e-4)
    adam_beta1 = _get(cfg, "training", "adam_beta1", default=0.9)
    adam_beta2 = _get(cfg, "training", "adam_beta2", default=0.999)
    max_grad_norm = _get(cfg, "training", "max_grad_norm", default=1.0)
    lr_scheduler_type = _get(cfg, "training", "lr_scheduler", default="cosine")
    warmup_ratio = _get(cfg, "training", "warmup_ratio", default=0.05)
    log_every = _get(cfg, "training", "log_every_n_steps", default=50)
    save_every = _get(cfg, "training", "save_every_n_epochs", default=10)
    eval_every = _get(cfg, "training", "eval_every_n_epochs", default=1)

    dataset_name = _get(cfg, "data", "dataset", default="RESISC45")
    image_size = _get(cfg, "data", "image_size", default=256)
    num_workers = _get(cfg, "data", "num_workers", default=4)
    val_ratio = _get(cfg, "data", "val_ratio", default=0.2)

    # Adjust image size for InceptionV3 (requires 299)
    if arch == "inception_v3" and image_size < 299:
        image_size = 299
        logger.info("InceptionV3 requires image_size >= 299; set to 299.")

    # Per-dataset output directory
    output_dir = os.path.join(output_dir, f"{arch}_{dataset_name}")

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
    if seed is not None:
        set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Dataset ---------------------------------------------------------
    train_transform = _classification_transforms(image_size, is_train=True)
    val_transform = _classification_transforms(image_size, is_train=False)

    full_dataset = RSDataset(
        root=_resolve_dataset_root(dataset_name),
        image_size=None,
        transform=train_transform,
    )
    num_classes = num_classes_cfg or len(full_dataset.class_names)
    logger.info("Dataset: %s  classes=%d  total=%d", dataset_name, num_classes, len(full_dataset))

    train_subset, val_subset = _train_val_split(full_dataset, val_ratio, seed)

    # Replace transform for validation subset.
    # We create a second dataset with val_transform sharing the same file list.
    val_dataset = RSDataset(
        root=_resolve_dataset_root(dataset_name),
        image_size=None,
        transform=val_transform,
    )

    class _SubsetByIndices:
        """Thin wrapper that indexes into a full dataset via a list of indices."""
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    val_wrapped = _SubsetByIndices(val_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_wrapped, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_fn,
    )

    logger.info("Train: %d  Val: %d", len(train_subset), len(val_wrapped))

    # ---- Model -----------------------------------------------------------
    if arch == "vgg16":
        model = create_vgg16(num_classes, pretrained=pretrained)
    elif arch == "inception_v3":
        model = create_inception_v3(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s  params=%s", arch, f"{total_params:,}")

    # ---- Optimizer / Scheduler -------------------------------------------
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2),
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9,
            weight_decay=weight_decay,
        )

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    max_steps = num_epochs * steps_per_epoch
    warmup_steps = int(max_steps * warmup_ratio)

    scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * grad_accum,
        num_training_steps=max_steps * grad_accum,
    )

    criterion = nn.CrossEntropyLoss()

    # ---- Accelerator prepare ---------------------------------------------
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler,
    )
    val_loader = accelerator.prepare(val_loader)

    if accelerator.is_main_process:
        accelerator.init_trackers("classifier_training", config=cfg)

    # ---- Training loop ---------------------------------------------------
    logger.info("***** Starting Training *****")
    logger.info("  Architecture = %s", arch)
    logger.info("  Dataset = %s (%d classes)", dataset_name, num_classes)
    logger.info("  Epochs = %d, Batch = %d, Steps = %d", num_epochs, batch_size, max_steps)

    def _extract_logits_and_loss(outputs, labels):
        """Return (logits, loss) handling InceptionV3's auxiliary outputs."""
        if isinstance(outputs, tuple):
            logits, aux_logits = outputs
            loss = criterion(logits, labels) + 0.4 * criterion(aux_logits, labels)
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
            loss = criterion(logits, labels)
            if hasattr(outputs, "aux_logits") and outputs.aux_logits is not None:
                loss = loss + 0.4 * criterion(outputs.aux_logits, labels)
        else:
            logits = outputs
            loss = criterion(logits, labels)
        return logits, loss

    global_step = 0
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",
                     disable=not accelerator.is_local_main_process)

        for images, labels in pbar:
            with accelerator.accumulate(model):
                outputs = model(images)
                logits, loss = _extract_logits_and_loss(outputs, labels)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if global_step % log_every == 0:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                    }, step=global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = correct / max(total, 1)
        avg_loss = running_loss / max(steps_per_epoch, 1)
        logger.info("Epoch %d – loss: %.4f  train_acc: %.4f", epoch + 1, avg_loss, train_acc)

        # ---- Validation --------------------------------------------------
        if (epoch + 1) % eval_every == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    logits, batch_loss = _extract_logits_and_loss(outputs, labels)
                    val_loss += batch_loss.item()
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)
                    val_steps += 1

            val_acc = val_correct / max(val_total, 1)
            avg_val_loss = val_loss / max(val_steps, 1)
            logger.info("Epoch %d – val_loss: %.4f  val_acc: %.4f",
                        epoch + 1, avg_val_loss, val_acc)

            if accelerator.is_main_process:
                accelerator.log({
                    "val/loss": avg_val_loss,
                    "val/accuracy": val_acc,
                }, step=global_step)

                if val_acc > best_acc:
                    best_acc = val_acc
                    _save_model(accelerator.unwrap_model(model),
                                os.path.join(output_dir, "best_model.pth"),
                                arch, dataset_name, num_classes, epoch + 1, val_acc)
                    logger.info("New best val_acc: %.4f – saved best_model.pth", val_acc)

        # ---- Checkpoint --------------------------------------------------
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            if accelerator.is_main_process:
                _save_model(accelerator.unwrap_model(model),
                            os.path.join(output_dir, f"checkpoint-epoch{epoch+1}.pth"),
                            arch, dataset_name, num_classes, epoch + 1)

    # ---- End training ----------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_model(accelerator.unwrap_model(model),
                    os.path.join(output_dir, "final_model.pth"),
                    arch, dataset_name, num_classes, num_epochs)
        logger.info("Training complete.  best_val_acc=%.4f  saved to %s", best_acc, output_dir)

    accelerator.end_training()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dataset_root(dataset_name: str) -> str:
    """Look up the dataset root from the project config.yaml."""
    from src.utils.config import get_config
    cfg = get_config()
    if dataset_name not in cfg.datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(cfg.datasets.keys())}")
    return cfg.datasets[dataset_name].root


def _save_model(model: nn.Module, path: str, arch: str, dataset: str,
                num_classes: int, epoch: int, val_acc: float = 0.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    # Save companion metadata
    import json
    meta = {"arch": arch, "dataset": dataset, "num_classes": num_classes,
            "epoch": epoch, "val_acc": val_acc}
    meta_path = path.replace(".pth", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
