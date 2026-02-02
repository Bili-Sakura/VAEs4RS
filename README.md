# VAEs4RS

**The Robustness of Natural English Priors in Remote Sensing: A Zero-Shot VAE Study**

Are pre-trained VAEs good zero-shot remote sensing image reconstructors?

This repository evaluates variational autoencoders (VAEs) pre-trained on natural image datasets when applied to remote sensing data in a zero-shot manner.

## Results and Findings

![Quantitative and qualitative results](./assets/results.png)

*Columns: Ground Truth \| SD21-VAE \| SDXL-VAE \| SD35-VAE \| FLUX1-VAE \| FLUX2-VAE \| SANA-VAE \| Qwen-VAE*  
*Rows: 8 samples (RESISC45 \| All)*

### Quantitative Results

Our evaluation demonstrates that pre-trained VAEs achieve strong zero-shot reconstruction performance on remote sensing data. Key quantitative results:

| Model | GFLOPs | Latent Shape | Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|-------|--------|--------------|---------|--------|--------|---------|-------|
| **FLUX2-VAE** | 895.71 | (32,32,32) | RESISC45 | **33.42** | **0.925** | **0.021** | **0.46** |
| **FLUX2-VAE** | 895.71 | (32,32,32) | AID | **34.46** | **0.926** | **0.022** | **0.37** |
| Qwen-VAE | 1143.88 | (16,32,32) | RESISC45 | 31.62 | 0.875 | 0.080 | 9.51 |
| Qwen-VAE | 1143.88 | (16,32,32) | AID | 33.17 | 0.890 | 0.077 | 0.42 |
| SD35-VAE | 895.25 | (16,32,32) | RESISC45 | 30.88 | 0.862 | 0.035 | 1.11 |
| SD35-VAE | 895.25 | (16,32,32) | AID | 31.97 | 0.877 | 0.037 | 0.69 |
| SDXL-VAE | 894.91 | (4,32,32) | RESISC45 | 26.81 | 0.685 | 0.100 | 32.36 |
| SDXL-VAE | 894.91 | (4,32,32) | AID | 28.16 | 0.755 | 0.086 | 24.81 |
| SD21-VAE | 894.91 | (4,32,32) | RESISC45 | 25.84 | 0.640 | 0.091 | 29.03 |
| SD21-VAE | 894.91 | (4,32,32) | AID | 27.19 | 0.715 | 0.082 | 22.26 |
| SANA-VAE | 846.76 | (32,8,8) | RESISC45 | 24.25 | 0.558 | 0.124 | 8.73 |
| SANA-VAE | 846.76 | (32,8,8) | AID | N/A* | N/A* | N/A* | N/A* |

*SANA-VAE cannot process AID images (600×600px) because 600 is not divisible by 32 (SANA-VAE's spatial compression factor).

## Insights and Conclusion

**Insights 1:** We find that VAEs reconstruct remote sensing images remarkably well, with reconstructions appearing visually nearly identical to the input. We argue that VAEs may have the potential to implicitly deblur and denoise input images, where the reconstructed image serves as a better data source for model training (e.g., representation learning) with possibly improved statistics.

**Insights 2:** As the compression appears effectively lossless, we argue for directly storing latent representations instead of original images as datasets to reduce storage requirements.

In this work, we explored the robustness of natural image priors in VAEs for remote sensing. Our findings indicate that these models, when used zero-shot, can provide significant utility in data compression across various categories. We will release the reconstructed images along with their corresponding latents for community exploration and further research.

## Quick Start

For code usage, installation, and detailed documentation, see [src/README.md](src/README.md).

**Resources:**
- VAE Models: [https://huggingface.co/BiliSakura/VAEs](https://huggingface.co/BiliSakura/VAEs)
- Datasets: [https://huggingface.co/blanchon/AID](https://huggingface.co/blanchon/AID) and [https://huggingface.co/blanchon/RESISC45](https://huggingface.co/blanchon/RESISC45)
- Latents Dataset (FLUX2-VAE): [https://huggingface.co/datasets/BiliSakura/RS-Dataset-Latents](https://huggingface.co/datasets/BiliSakura/RS-Dataset-Latents) - Latents version of AID and RESISC45 using FLUX2-VAE

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026robustness,
  author = {Zhenyuan Chen and Feng Zhang},
  title = {THE ROBUSTNESS OF NATURAL IMAGE PRIORS IN REMOTE SENSING: A ZERO-SHOT VAE STUDY},
  year = {2026}
}
```
