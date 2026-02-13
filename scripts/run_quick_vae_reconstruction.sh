#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

VAE_ROOT="${REPO_ROOT}/models/BiliSakura/VAEs"
# All VAEs under models/BiliSakura/VAEs/ (base KL, DC, Flux2, Qwen, SANA, VQModel, VQDIFFUSION-VQVAE, IBQ-VQVAE-*)
VAES=(
  # "SD21-VAE"
  # "SDXL-VAE"
  # "SD35-VAE"
  # "FLUX1-VAE"
  # "FLUX2-VAE"
  # "SANA-VAE"
  # "Qwen-VAE"
  # "MOVQGAN-67M"
  # "MOVQGAN-102M"
  # "MOVQGAN-270M"
  "VQDIFFUSION-VQVAE"
  "IBQ-VQVAE-1024"
  "IBQ-VQVAE-8192"
  "IBQ-VQVAE-16384"
  "IBQ-VQVAE-262144"
)

# Per-VAE resolution override: 0 = use original image resolution (no resize)
declare -A VAE_RES_OVERRIDE=(
  ["MOVQGAN-67M"]=0
  ["MOVQGAN-102M"]=0
  ["MOVQGAN-270M"]=0
  ["VQDIFFUSION-VQVAE"]=0
  ["IBQ-VQVAE-1024"]=0
  ["IBQ-VQVAE-8192"]=0
  ["IBQ-VQVAE-16384"]=0
  ["IBQ-VQVAE-262144"]=0
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
  if [[ ! -d "${vae_path}" ]]; then
    echo "==> Skipping ${vae} (not found at ${vae_path})"
    continue
  fi
  for entry in "${DATASETS[@]}"; do
    IFS="|" read -r label rel_path resolution <<< "${entry}"
    input_dir="${REPO_ROOT}/${rel_path}"
    # Use per-VAE resolution override if set (e.g. MOVQGAN expects 256)
    res="${VAE_RES_OVERRIDE[$vae]:-$resolution}"
    echo "==> VAE=${vae} | DATA=${label} | RES=${res} | MAX=${MAX_IMAGES}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/quick_vae_reconstruction.py" \
      --input-dir "${input_dir}" \
      --vae-path "${vae_path}" \
      --max-images "${MAX_IMAGES}" \
      --resolution "${res}"
  done
done
