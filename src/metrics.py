"""
Reconstruction quality metrics for VAE evaluation.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- rFID (Reconstruction FID)
- CMMD (CLIP Maximum Mean Discrepancy)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union
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

# Try to import transformers for CMMD
try:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    CLIPImageProcessor = None
    CLIPVisionModelWithProjection = None
    import warnings
    warnings.warn(
        "transformers is not available. CMMD will be skipped. "
        "Install with: pip install transformers"
    )


# =============================================================================
# CMMD Implementation (based on sayakpaul/cmmd-pytorch)
# =============================================================================

# MMD parameters from the CMMD paper
_CMMD_SIGMA = 10  # Bandwidth parameter for Gaussian RBF kernel
_CMMD_SCALE = 1000  # Scale factor for human readability
_CLIP_MODEL_NAME = "/data/projects/VAEs4RS/models/BiliSakura/Git-RSCLIP-ViT-L-16"  # Local CLIP model path


def _compute_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Memory-efficient MMD implementation.
    
    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    
    Args:
        x: First set of embeddings, shape (n, embedding_dim)
        y: Second set of embeddings, shape (m, embedding_dim)
        
    Returns:
        MMD distance between x and y embedding sets
    """
    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))
    
    gamma = 1 / (2 * _CMMD_SIGMA**2)
    
    # Compute kernel matrices
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + 
                           torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + 
                           torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + 
                           torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    
    return _CMMD_SCALE * (k_xx + k_yy - 2 * k_xy)


class CMMDCalculator:
    """
    CMMD (CLIP Maximum Mean Discrepancy) calculator.
    
    Uses CLIP embeddings and MMD distance to compute CMMD between original
    and reconstructed images.
    """
    
    def __init__(self, device: str = "cuda", clip_model_name: str = _CLIP_MODEL_NAME):
        """
        Initialize CMMD calculator.
        
        Args:
            device: Device to run computations on
            clip_model_name: HuggingFace model name or local path for CLIP 
                           (default: /data/projects/VAEs4RS/models/BiliSakura/Git-RSCLIP-ViT-L-16)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for CMMD. Install with: pip install transformers"
            )
        
        self.device = device
        self.clip_model_name = clip_model_name
        
        # Initialize CLIP model and processor (works with both HuggingFace model names and local paths)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).eval()
        self.model = self.model.to(device)
        
        # Set model to eval mode and disable gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.input_image_size = self.image_processor.crop_size["height"]
        
        # Accumulate embeddings
        self.original_embeddings: List[torch.Tensor] = []
        self.reconstructed_embeddings: List[torch.Tensor] = []
    
    def _resize_bicubic(self, images: torch.Tensor) -> torch.Tensor:
        """Resize images to CLIP input size using bicubic interpolation."""
        # images: (B, 3, H, W)
        images = torch.nn.functional.interpolate(
            images, 
            size=(self.input_image_size, self.input_image_size), 
            mode="bicubic",
            align_corners=False
        )
        return images
    
    @torch.no_grad()
    def _compute_clip_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute CLIP embeddings for images.
        
        Args:
            images: Image tensor, shape (B, 3, H, W), range [-1, 1]
            
        Returns:
            Embeddings, shape (B, embedding_dim)
        """
        # Convert from [-1, 1] to [0, 1] range
        images_01 = (images + 1) / 2
        images_01 = images_01.clamp(0, 1)
        
        # Resize to CLIP input size
        images_resized = self._resize_bicubic(images_01)
        
        # Convert to numpy for processor (processor expects numpy arrays)
        # Shape: (B, H, W, 3) for processor
        images_np = images_resized.permute(0, 2, 3, 1).cpu().numpy()
        
        # Process images
        inputs = self.image_processor(
            images=images_np,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        image_embs = self.model(**inputs).image_embeds
        
        # Normalize embeddings
        image_embs = image_embs / torch.linalg.norm(image_embs, dim=-1, keepdims=True)
        
        return image_embs
    
    def reset(self):
        """Reset accumulated embeddings."""
        self.original_embeddings = []
        self.reconstructed_embeddings = []
    
    def update(self, original: torch.Tensor, reconstructed: torch.Tensor):
        """
        Update CMMD with a batch of images.
        
        Args:
            original: Original images, shape (B, 3, H, W), range [-1, 1]
            reconstructed: Reconstructed images, shape (B, 3, H, W), range [-1, 1]
        """
        # Clamp images to valid range
        original = original.clamp(-1, 1)
        reconstructed = reconstructed.clamp(-1, 1)
        
        # Compute CLIP embeddings
        orig_embs = self._compute_clip_embeddings(original)
        recon_embs = self._compute_clip_embeddings(reconstructed)
        
        # Accumulate embeddings (store on CPU to save GPU memory)
        self.original_embeddings.append(orig_embs.cpu())
        self.reconstructed_embeddings.append(recon_embs.cpu())
    
    def compute(self) -> float:
        """
        Compute CMMD distance.
        
        Returns:
            CMMD value (lower is better)
        """
        if len(self.original_embeddings) == 0:
            raise ValueError("No embeddings accumulated. Call update() first.")
        
        # Concatenate all embeddings
        orig_all = torch.cat(self.original_embeddings, dim=0)
        recon_all = torch.cat(self.reconstructed_embeddings, dim=0)
        
        # Move to device for computation
        orig_all = orig_all.to(self.device)
        recon_all = recon_all.to(self.device)
        
        # Compute MMD
        cmmd_value = _compute_mmd(orig_all, recon_all)
        
        return cmmd_value.item()


@dataclass
class MetricResults:
    """Container for metric results."""
    psnr: float
    ssim: float
    lpips: float
    fid: Optional[float] = None
    cmmd: Optional[float] = None
    
    def __repr__(self) -> str:
        fid_str = f"{self.fid:.2f}" if self.fid is not None else "N/A"
        cmmd_str = f"{self.cmmd:.2f}" if self.cmmd is not None else "N/A"
        return (
            f"PSNR: {self.psnr:.2f} dB | "
            f"SSIM: {self.ssim:.4f} | "
            f"LPIPS: {self.lpips:.4f} | "
            f"FID: {fid_str} | "
            f"CMMD: {cmmd_str}"
        )
    
    def to_dict(self) -> dict:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "fid": self.fid,
            "cmmd": self.cmmd,
        }


class MetricCalculator:
    """
    Calculator for image reconstruction metrics.
    
    Computes PSNR, SSIM, LPIPS, and optionally FID and CMMD between original
    and reconstructed images.
    
    Args:
        device: Device to run computations on (default: "cuda")
        compute_fid: Whether to compute FID metric (default: True)
        compute_cmmd: Whether to compute CMMD metric (default: False)
        fid_feature_extractor: Optional custom feature extractor model for FID.
                              Should be a torch.nn.Module that takes images and returns
                              features with shape (N, num_features). If None, uses default
                              Inception v3 model. The model will be moved to the specified device.
        cmmd_clip_model: CLIP model path or HuggingFace model name for CMMD 
                        (default: /data/projects/VAEs4RS/models/BiliSakura/Git-RSCLIP-ViT-L-16)
    """
    
    def __init__(
        self, 
        device: str = "cuda", 
        compute_fid: bool = True,
        compute_cmmd: bool = False,
        fid_feature_extractor: Optional[nn.Module] = None,
        cmmd_clip_model: str = _CLIP_MODEL_NAME
    ):
        self.device = device
        # Only compute FID if requested and available
        self.compute_fid = compute_fid and FID_AVAILABLE
        # Only compute CMMD if requested and available
        self.compute_cmmd = compute_cmmd and TRANSFORMERS_AVAILABLE
        
        # Initialize metrics
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)  # [-1, 1] range
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        
        if self.compute_fid:
            try:
                # Prepare feature extractor if provided
                feature_extractor = None
                if fid_feature_extractor is not None:
                    # Move feature extractor to device and set to eval mode
                    feature_extractor = fid_feature_extractor.to(device).eval()
                    # Ensure it's wrapped in no_grad context for efficiency
                    for param in feature_extractor.parameters():
                        param.requires_grad = False
                
                # Use normalize=False since we'll provide uint8 images in [0, 255] range
                # Pass custom feature extractor if provided
                self.fid = FrechetInceptionDistance(
                    normalize=False,
                    feature=feature_extractor
                ).to(device)
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
        
        if self.compute_cmmd:
            try:
                self.cmmd = CMMDCalculator(device=device, clip_model_name=cmmd_clip_model)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to initialize CMMDCalculator: {e}. "
                    "CMMD will be skipped. Install with: pip install transformers"
                )
                self.compute_cmmd = False
                self.cmmd = None
        else:
            self.cmmd = None
        
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
        if self.compute_cmmd and self.cmmd is not None:
            self.cmmd.reset()
    
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
        
        # Update CMMD (CMMD accumulates embeddings for final computation)
        if self.compute_cmmd and self.cmmd is not None:
            self.cmmd.update(original, reconstructed)
    
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
        
        cmmd_value = None
        if self.compute_cmmd and self.cmmd is not None:
            cmmd_value = self.cmmd.compute()
        
        return MetricResults(
            psnr=psnr_avg,
            ssim=ssim_avg,
            lpips=lpips_avg,
            fid=fid_value,
            cmmd=cmmd_value,
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
