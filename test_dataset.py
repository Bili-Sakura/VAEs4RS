#!/usr/bin/env python
"""Quick test script to verify dataset loading works."""

import sys
sys.path.insert(0, 'src')

from config import DATASET_CONFIGS
from datasets import load_dataset
import os

print("="*60)
print("Testing RESISC45 Dataset Configuration")
print("="*60)

# Check config
config = DATASET_CONFIGS["RESISC45"]
print(f"\nConfig root path: {config.root}")

# Test dataset loading
print("\nLoading dataset...")
try:
    dataset, dataloader = load_dataset('RESISC45', image_size=256, batch_size=1, num_workers=0)
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Found {len(dataset)} images")
    print(f"  - Found {len(dataset.class_names)} classes")
    print(f"  - Dataset root: {dataset.root}")
    print(f"  - First 5 classes: {dataset.class_names[:5]}")
    print(f"  - Sample image: {dataset.image_paths[0] if dataset.image_paths else 'None'}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Dataset loading test completed successfully!")
print("="*60)
