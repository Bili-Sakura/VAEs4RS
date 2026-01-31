"""
Reconstruction quality metrics for VAE evaluation.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- rFID (Reconstruction FID)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)


@dataclass
class MetricResults:
    """Container for metric results."""
    psnr: float
    ssim: float
    lpips: float
    fid: Optional[float] = None
    
    def __repr__(self) -> str:
        fid_str = f"{self.fid:.2f}" if self.fid is not None else "N/A"
        return (
            f"PSNR: {self.psnr:.2f} dB | "
            f"SSIM: {self.ssim:.4f} | "
            f"LPIPS: {self.lpips:.4f} | "
            f"FID: {fid_str}"
        )
    
    def to_dict(self) -> dict:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "fid": self.fid,
        }


class MetricCalculator:
    """
    Calculator for image reconstruction metrics.
    
    Computes PSNR, SSIM, LPIPS, and optionally FID between original
    and reconstructed images.
    """
    
    def __init__(self, device: str = "cuda", compute_fid: bool = True):
        self.device = device
        self.compute_fid = compute_fid
        
        # Initialize metrics
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)  # [-1, 1] range
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        
        if compute_fid:
            self.fid = FrechetInceptionDistance(normalize=True).to(device)
        
        # Accumulate values
        self.psnr_values: List[float] = []
        self.ssim_values: List[float] = []
        self.lpips_values: List[float] = []
    
    def reset(self):
        """Reset accumulated metrics."""
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []
        if self.compute_fid:
            self.fid.reset()
    
    @torch.no_grad()
    def update(self, original: torch.Tensor, reconstructed: torch.Tensor):
        """
        Update metrics with a batch of images.
        
        Args:
            original: Original images, shape (B, 3, H, W), range [-1, 1]
            reconstructed: Reconstructed images, shape (B, 3, H, W), range [-1, 1]
        """
        # Clamp reconstructed images to valid range
        reconstructed = reconstructed.clamp(-1, 1)
        
        # Compute per-batch metrics
        self.psnr_values.append(self.psnr(reconstructed, original).item())
        self.ssim_values.append(self.ssim(reconstructed, original).item())
        self.lpips_values.append(self.lpips(reconstructed, original).item())
        
        # Update FID
        if self.compute_fid:
            # Convert to [0, 1] range for FID
            orig_01 = (original + 1) / 2
            recon_01 = (reconstructed + 1) / 2
            
            # Convert to uint8 for FID
            orig_uint8 = (orig_01 * 255).to(torch.uint8)
            recon_uint8 = (recon_01 * 255).to(torch.uint8)
            
            self.fid.update(orig_uint8, real=True)
            self.fid.update(recon_uint8, real=False)
    
    def compute(self) -> MetricResults:
        """
        Compute final metrics.
        
        Returns:
            MetricResults with averaged metrics
        """
        psnr_avg = np.mean(self.psnr_values)
        ssim_avg = np.mean(self.ssim_values)
        lpips_avg = np.mean(self.lpips_values)
        
        fid_value = None
        if self.compute_fid:
            fid_value = self.fid.compute().item()
        
        return MetricResults(
            psnr=psnr_avg,
            ssim=ssim_avg,
            lpips=lpips_avg,
            fid=fid_value,
        )


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute PSNR between two images.
    
    Args:
        original: Original image tensor
        reconstructed: Reconstructed image tensor
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    # For [-1, 1] range, max value is 2
    return (20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse)).item()
