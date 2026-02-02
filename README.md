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

| Model | GFLOPs | Latent Shape | PSNR ↑ (RESISC45) | PSNR ↑ (AID) | SSIM ↑ (RESISC45) | SSIM ↑ (AID) | LPIPS ↓ (RESISC45) | LPIPS ↓ (AID) | FID ↓ (RESISC45) | FID ↓ (AID) |
|-------|--------|--------------|------------------:|-------------:|------------------:|-------------:|-------------------:|--------------:|-----------------:|------------:|
| SD21-VAE | 894.91 | (4,32,32) | 25.71 | 26.66 | 0.672 | 0.709 | 0.095 | 0.094 | 4.13 | 3.08 |
| SDXL-VAE | 894.91 | (4,32,32) | 25.83 | 26.80 | 0.692 | 0.726 | 0.098 | 0.098 | 4.98 | 3.11 |
| SD35-VAE | 895.25 | (16,32,32) | 29.71 | 30.72 | 0.862 | 0.876 | 0.035 | 0.037 | 1.11 | 0.69 |
| FLUX1-VAE | 895.25 | (16,32,32) | 33.30 | 33.63 | 0.923 | 0.918 | 0.022 | 0.025 | 0.38 | 0.26 |
| **FLUX2-VAE** | 895.71 | (32,32,32) | **33.42** | **34.46** | **0.925** | **0.926** | **0.021** | **0.022** | **0.46** | **0.37** |
| SANA-VAE | 846.76 | (32,8,8) | 23.36 | N/A* | 0.558 | N/A* | 0.124 | N/A* | 8.69 | N/A* |
| Qwen-VAE | 1143.88 | (16,32,32) | 30.38 | 31.46 | 0.874 | 0.889 | 0.080 | 0.077 | 9.51 | 0.42 |

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
