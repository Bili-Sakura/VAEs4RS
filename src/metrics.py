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
)

# Try to import FID, but handle missing torch-fidelity gracefully
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    FID_AVAILABLE = False
    FrechetInceptionDistance = None
    import warnings
    warnings.warn(
        f"FrechetInceptionDistance is not available: {e}. "
        "FID will be skipped. Install with: pip install torchmetrics[image] or pip install torch-fidelity"
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
        # Only compute FID if requested and available
        self.compute_fid = compute_fid and FID_AVAILABLE
        
        # Initialize metrics
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)  # [-1, 1] range
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        
        if self.compute_fid:
            try:
                # Use normalize=False since we'll provide uint8 images in [0, 255] range
                self.fid = FrechetInceptionDistance(normalize=False).to(device)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to initialize FrechetInceptionDistance: {e}. "
                    "FID will be skipped. Install with: pip install torchmetrics[image] or pip install torch-fidelity"
                )
                self.compute_fid = False
                self.fid = None
        else:
            self.fid = None
        
        # Accumulate values
        self.psnr_values: List[float] = []
        self.ssim_values: List[float] = []
        self.lpips_values: List[float] = []
    
    def reset(self):
        """Reset accumulated metrics."""
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []
        # Reset metric objects to clear their internal state
        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()
        if self.compute_fid and self.fid is not None:
            self.fid.reset()
    
    @torch.no_grad()
    def update(self, original: torch.Tensor, reconstructed: torch.Tensor):
        """
        Update metrics with a batch of images.
        
        Args:
            original: Original images, shape (B, 3, H, W), range [-1, 1]
            reconstructed: Reconstructed images, shape (B, 3, H, W), range [-1, 1]
        """
        # Clamp images to valid range [-1, 1] for safety
        original = original.clamp(-1, 1)
        reconstructed = reconstructed.clamp(-1, 1)
        
        # Compute per-batch metrics and reset immediately to prevent state accumulation
        # This prevents slowdown as more batches are processed
        self.psnr_values.append(self.psnr(reconstructed, original).item())
        self.psnr.reset()  # Reset to prevent state accumulation
        
        self.ssim_values.append(self.ssim(reconstructed, original).item())
        self.ssim.reset()  # Reset to prevent state accumulation
        
        self.lpips_values.append(self.lpips(reconstructed, original).item())
        self.lpips.reset()  # Reset to prevent state accumulation
        
        # Update FID (FID intentionally accumulates state for final computation)
        if self.compute_fid and self.fid is not None:
            # Convert from [-1, 1] to [0, 1] range
            orig_01 = (original + 1) / 2
            recon_01 = (reconstructed + 1) / 2
            
            # Clamp to valid [0, 1] range before conversion
            orig_01 = orig_01.clamp(0, 1)
            recon_01 = recon_01.clamp(0, 1)
            
            # Convert to uint8 [0, 255] for FID (normalize=False expects uint8)
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
        if self.compute_fid and self.fid is not None:
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
