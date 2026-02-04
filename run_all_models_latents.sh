#!/bin/bash
# Run all VAE models with latents saving, skip existing, no image saving
# Run in background with nohup

# Set HuggingFace cache directory
export HF_HUB_CACHE=/data/projects/VAEs4RS/models/BiliSakura/VAEs

# Create log directory if it doesn't exist
mkdir -p logs

# Run with nohup
nohup python run_experiments.py \
    --main-only \
    --no-save-images \
    --save-latents \
    --batch-size 32 \
    --image-size original \
    --datasets UCMerced \
    --output-dir datasets/BiliSakura/VAEs4RS \
    --device cuda \
    --seed 42 \
    > logs/run_all_models_latents_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Print the PID
echo "Started process with PID: $!"
echo "Log file: logs/run_all_models_latents_$(date +%Y%m%d_%H%M%S).log"
echo "To check progress: tail -f logs/run_all_models_latents_*.log"
echo "To check if still running: ps aux | grep run_experiments.py"
