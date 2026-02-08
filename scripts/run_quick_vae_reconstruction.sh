#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

VAE_ROOT="${REPO_ROOT}/models/BiliSakura/VAEs"
VAES=(
  "SD21-VAE"
  "SD35-VAE"
  "FLUX2-VAE"
)

DATASETS=(
  "IR|datasets/BiliSakura/MACIV-T-2025-Structure-Refined/rgb2ir_crop_aug/train/target|1024"
  "EO|datasets/BiliSakura/MACIV-T-2025-Structure-Refined/sar2eo/train/target|256"
  "RGB|datasets/BiliSakura/MACIV-T-2025-Structure-Refined/sar2rgb_crop_aug/train/target|1024"
  "SAR|datasets/BiliSakura/MACIV-T-2025-Structure-Refined/sar2rgb_crop_aug/train/input|1024"
)

MAX_IMAGES=10

for vae in "${VAES[@]}"; do
  vae_path="${VAE_ROOT}/${vae}"
  for entry in "${DATASETS[@]}"; do
    IFS="|" read -r label rel_path resolution <<< "${entry}"
    input_dir="${REPO_ROOT}/${rel_path}"
    echo "==> VAE=${vae} | DATA=${label} | RES=${resolution} | MAX=${MAX_IMAGES}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/quick_vae_reconstruction.py" \
      --input-dir "${input_dir}" \
      --vae-path "${vae_path}" \
      --max-images "${MAX_IMAGES}" \
      --resolution "${resolution}"
  done
done
