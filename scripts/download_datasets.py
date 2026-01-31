"""
Download and prepare remote sensing datasets.

Datasets:
- NWPU-RESISC45: https://www.tensorflow.org/datasets/catalog/resisc45
- AID: https://captain-whu.github.io/AID/

Note: These datasets may require manual download due to licensing.
This script provides instructions and optional auto-download if available.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DATASET_CONFIGS


def check_dataset_exists(dataset_name: str) -> bool:
    """Check if a dataset already exists."""
    if dataset_name not in DATASET_CONFIGS:
        return False
    
    config = DATASET_CONFIGS[dataset_name]
    root = Path(config.root)
    
    if not root.exists():
        return False
    
    # Check if there are any subdirectories (class folders)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    return len(subdirs) > 0


def print_download_instructions():
    """Print instructions for downloading datasets."""
    print("="*80)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*80)
    
    print("""
NWPU-RESISC45:
--------------
1. Download from: https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68
   Or: https://gcheng-nwpu.github.io/#Datasets
2. Extract to: datasets/blanchon/RESISC45/
3. Expected structure:
   datasets/blanchon/RESISC45/
   ├── airplane/
   │   ├── airplane_001.jpg
   │   └── ...
   ├── airport/
   └── ...

AID (Aerial Image Dataset):
---------------------------
1. Download from: https://captain-whu.github.io/AID/
2. Extract to: datasets/blanchon/AID/
3. Expected structure:
   datasets/blanchon/AID/
   ├── Airport/
   │   ├── airport_1.jpg
   │   └── ...
   ├── BareLand/
   └── ...

Note: Both datasets require accepting academic usage terms.
""")


def main():
    print("Checking dataset availability...\n")
    
    for name in DATASET_CONFIGS:
        exists = check_dataset_exists(name)
        status = "✓ Found" if exists else "✗ Not found"
        print(f"  {name}: {status}")
    
    all_exist = all(check_dataset_exists(name) for name in DATASET_CONFIGS)
    
    if not all_exist:
        print_download_instructions()
    else:
        print("\n✓ All datasets are available!")


if __name__ == "__main__":
    main()
