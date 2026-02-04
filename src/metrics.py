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
    from transformers import (
        CLIPImageProcessor, 
        CLIPVisionModelWithProjection,
        SiglipImageProcessor, 
        SiglipModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    CLIPImageProcessor = None
    CLIPVisionModelWithProjection = None
    SiglipImageProcessor = None
    SiglipModel = None
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
_CLIP_MODEL_NAME = "/data/projects/VAEs4RS/models/BiliSakura/Remote-CLIP-ViT-L-14/transformers"  # Local CLIP model path (transformers subdirectory)


def _compute_mmd(x: torch.Tensor, y: torch.Tensor, chunk_size: int = 1000) -> float:
    """
    Memory-efficient batched MMD implementation.
    
    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    
    Uses batched computation to avoid O(n²) memory usage for large datasets.
    Clears GPU cache periodically to prevent memory fragmentation.
    
    Args:
        x: First set of embeddings, shape (n, embedding_dim)
        y: Second set of embeddings, shape (m, embedding_dim)
        chunk_size: Size of chunks for batched computation (default: 1000)
        
    Returns:
        MMD distance between x and y embedding sets
    """
    n = x.shape[0]
    m = y.shape[0]
    gamma = 1 / (2 * _CMMD_SIGMA**2)
    device = x.device
    
    # Compute squared norms for both sets
    x_sqnorms = torch.sum(x ** 2, dim=1)  # (n,)
    y_sqnorms = torch.sum(y ** 2, dim=1)  # (m,)
    
    # Compute k_xx in chunks to avoid O(n²) memory
    # Process all pairs in chunks (full n×n matrix)
    k_xx_sum = 0.0
    k_xx_count = 0
    num_chunks_xx = (n + chunk_size - 1) // chunk_size
    chunk_idx = 0
    
    for i in range(0, n, chunk_size):
        x_chunk_i = x[i:i + chunk_size]  # (chunk_i, embedding_dim)
        x_sqnorms_chunk_i = x_sqnorms[i:i + chunk_size]  # (chunk_i,)
        
        for j in range(0, n, chunk_size):
            x_chunk_j = x[j:min(j + chunk_size, n)]  # (chunk_j, embedding_dim)
            x_sqnorms_chunk_j = x_sqnorms[j:min(j + chunk_size, n)]  # (chunk_j,)
            
            # Compute kernel matrix chunk: (chunk_i, chunk_j)
            dists_sq = (
                x_sqnorms_chunk_i.unsqueeze(1) + 
                x_sqnorms_chunk_j.unsqueeze(0) - 
                2 * torch.matmul(x_chunk_i, x_chunk_j.T)
            )
            kernel_vals = torch.exp(-gamma * dists_sq)
            k_xx_sum += torch.sum(kernel_vals).item()  # Move to CPU immediately
            k_xx_count += kernel_vals.numel()
            
            # Clear intermediate tensors
            del dists_sq, kernel_vals
            
            # Clear GPU cache periodically (every 10 chunks)
            chunk_idx += 1
            if chunk_idx % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
    
    k_xx = k_xx_sum / k_xx_count if k_xx_count > 0 else 0.0
    
    # Clear GPU cache after k_xx computation
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Compute k_yy in chunks (same as k_xx)
    k_yy_sum = 0.0
    k_yy_count = 0
    chunk_idx = 0
    
    for i in range(0, m, chunk_size):
        y_chunk_i = y[i:i + chunk_size]  # (chunk_i, embedding_dim)
        y_sqnorms_chunk_i = y_sqnorms[i:i + chunk_size]  # (chunk_i,)
        
        for j in range(0, m, chunk_size):
            y_chunk_j = y[j:min(j + chunk_size, m)]  # (chunk_j, embedding_dim)
            y_sqnorms_chunk_j = y_sqnorms[j:min(j + chunk_size, m)]  # (chunk_j,)
            
            dists_sq = (
                y_sqnorms_chunk_i.unsqueeze(1) + 
                y_sqnorms_chunk_j.unsqueeze(0) - 
                2 * torch.matmul(y_chunk_i, y_chunk_j.T)
            )
            kernel_vals = torch.exp(-gamma * dists_sq)
            k_yy_sum += torch.sum(kernel_vals).item()  # Move to CPU immediately
            k_yy_count += kernel_vals.numel()
            
            # Clear intermediate tensors
            del dists_sq, kernel_vals
            
            # Clear GPU cache periodically
            chunk_idx += 1
            if chunk_idx % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
    
    k_yy = k_yy_sum / k_yy_count if k_yy_count > 0 else 0.0
    
    # Clear GPU cache after k_yy computation
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Compute k_xy in chunks (cross terms - full matrix)
    k_xy_sum = 0.0
    k_xy_count = 0
    chunk_idx = 0
    
    for i in range(0, n, chunk_size):
        x_chunk = x[i:i + chunk_size]  # (chunk_size, embedding_dim)
        x_sqnorms_chunk = x_sqnorms[i:i + chunk_size]  # (chunk_size,)
        
        for j in range(0, m, chunk_size):
            y_chunk = y[j:j + chunk_size]  # (chunk_size, embedding_dim)
            y_sqnorms_chunk = y_sqnorms[j:j + chunk_size]  # (chunk_size,)
            
            # Compute all pairwise distances between x_chunk and y_chunk
            # Shape: (chunk_size_x, chunk_size_y)
            dists_sq = (
                x_sqnorms_chunk.unsqueeze(1) + 
                y_sqnorms_chunk.unsqueeze(0) - 
                2 * torch.matmul(x_chunk, y_chunk.T)
            )
            kernel_vals = torch.exp(-gamma * dists_sq)
            k_xy_sum += torch.sum(kernel_vals).item()  # Move to CPU immediately
            k_xy_count += kernel_vals.numel()
            
            # Clear intermediate tensors
            del dists_sq, kernel_vals
            
            # Clear GPU cache periodically
            chunk_idx += 1
            if chunk_idx % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
    
    k_xy = k_xy_sum / k_xy_count if k_xy_count > 0 else 0.0
    
    # Final cache clear
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return _CMMD_SCALE * (k_xx + k_yy - 2 * k_xy)


def _detect_model_type(model_path: str) -> str:
    """
    Detect whether the model is CLIP or SigLIP based on config.json.
    
    Args:
        model_path: Path to the model directory or HuggingFace model name
        
    Returns:
        "clip" or "siglip"
    """
    import json
    from pathlib import Path
    
    # Check if it's a local path
    model_path_obj = Path(model_path)
    
    # Try to find config.json
    config_path = model_path_obj / "config.json"
    if not config_path.exists():
        # Try transformers subdirectory (common for SigLIP models like Git-RSCLIP)
        config_path = model_path_obj / "transformers" / "config.json"
        if config_path.exists():
            return "siglip"  # Found in transformers subdirectory, likely SigLIP
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_type = config.get("model_type", "").lower()
            if "siglip" in model_type:
                return "siglip"
            elif "clip" in model_type:
                return "clip"
        except Exception:
            pass
    
    # If it's a HuggingFace model name (contains "/"), try to infer from name
    if "/" in model_path and not model_path_obj.exists():
        # HuggingFace model name - check if it contains "siglip" or "clip"
        model_path_lower = model_path.lower()
        if "siglip" in model_path_lower:
            return "siglip"
        elif "clip" in model_path_lower:
            return "clip"
        # Default for HuggingFace: try CLIP first (more common)
        return "clip"
    
    # Default: try SigLIP first (for Git-RSCLIP), fall back to CLIP
    # Check if transformers subdirectory exists (SigLIP models often have this structure)
    if (model_path_obj / "transformers" / "config.json").exists():
        return "siglip"
    
    return "clip"  # Default to CLIP


class CMMDCalculator:
    """
    CMMD (CLIP Maximum Mean Discrepancy) calculator.
    
    Uses CLIP or SigLIP embeddings and MMD distance to compute CMMD between original
    and reconstructed images. Automatically detects model type (CLIP or SigLIP) and uses
    the appropriate classes.
    """
    
    def __init__(self, device: str = "cuda", clip_model_name: str = _CLIP_MODEL_NAME, cmmd_batch_size: int = 32, mmd_chunk_size: int = 1000):
        """
        Initialize CMMD calculator.
        
        Args:
            device: Device to run computations on
            clip_model_name: HuggingFace model name or local path for CLIP/SigLIP model 
                           (default: /data/projects/VAEs4RS/models/BiliSakura/Remote-CLIP-ViT-L-14/transformers)
            cmmd_batch_size: Batch size for CLIP embedding computation (default: 32).
                            Large batches will be automatically chunked to this size.
            mmd_chunk_size: Chunk size for MMD computation to avoid O(n²) memory (default: 1000).
                           Smaller values use less memory but may be slower.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for CMMD. Install with: pip install transformers"
            )
        
        self.device = device
        self.clip_model_name = clip_model_name
        self.cmmd_batch_size = cmmd_batch_size
        self.mmd_chunk_size = mmd_chunk_size
        
        # Detect model type (CLIP or SigLIP) and initialize accordingly
        self.model_type = _detect_model_type(clip_model_name)
        
        if self.model_type == "siglip":
            # Handle SigLIP models (e.g., Git-RSCLIP)
            # Check if transformers subdirectory exists (common structure)
            from pathlib import Path
            model_path = Path(clip_model_name)
            if model_path.exists() and (model_path / "transformers" / "config.json").exists():
                siglip_path = str(model_path / "transformers")
            else:
                siglip_path = clip_model_name
            
            try:
                self.image_processor = SiglipImageProcessor.from_pretrained(siglip_path)
                self.model = SiglipModel.from_pretrained(siglip_path).eval()
                self.use_siglip = True
            except Exception as e:
                # Fall back to CLIP if SigLIP loading fails
                import warnings
                warnings.warn(
                    f"Failed to load SigLIP model from {siglip_path}: {e}. "
                    "Falling back to CLIP model."
                )
                self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
                self.model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).eval()
                self.use_siglip = False
        else:
            # Use standard CLIP
            try:
                self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
                self.model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).eval()
                self.use_siglip = False
            except Exception as e:
                # Fall back to SigLIP if CLIP loading fails
                import warnings
                warnings.warn(
                    f"Failed to load CLIP model from {clip_model_name}: {e}. "
                    "Trying SigLIP model instead."
                )
                from pathlib import Path
                model_path = Path(clip_model_name)
                if model_path.exists() and (model_path / "transformers" / "config.json").exists():
                    siglip_path = str(model_path / "transformers")
                else:
                    siglip_path = clip_model_name
                self.image_processor = SiglipImageProcessor.from_pretrained(siglip_path)
                self.model = SiglipModel.from_pretrained(siglip_path).eval()
                self.use_siglip = True
        
        self.model = self.model.to(device)
        
        # Set model to eval mode and disable gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get input image size - handle both CLIP and SigLIP processors
        if hasattr(self.image_processor, 'crop_size') and isinstance(self.image_processor.crop_size, dict):
            # CLIPImageProcessor has crop_size as dict
            self.input_image_size = self.image_processor.crop_size["height"]
        elif hasattr(self.image_processor, 'size') and isinstance(self.image_processor.size, dict):
            # SiglipImageProcessor might have size as dict
            self.input_image_size = self.image_processor.size.get("height") or self.image_processor.size.get("shortest_edge", 224)
        elif hasattr(self.image_processor, 'size') and isinstance(self.image_processor.size, int):
            # SiglipImageProcessor might have size as int
            self.input_image_size = self.image_processor.size
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'image_size'):
            # Try to get from model config
            self.input_image_size = self.model.config.image_size
        else:
            # Default fallback
            self.input_image_size = 224
        
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
        Compute CLIP or SigLIP embeddings for images.
        
        Processes batches in chunks to avoid memory issues with large batches.
        
        Args:
            images: Image tensor, shape (B, 3, H, W), range [-1, 1]
            
        Returns:
            Embeddings, shape (B, embedding_dim)
        """
        batch_size = images.shape[0]
        
        # If batch is small enough, process directly
        if batch_size <= self.cmmd_batch_size:
            return self._compute_clip_embeddings_chunk(images)
        
        # Process in chunks
        all_embeddings = []
        for i in range(0, batch_size, self.cmmd_batch_size):
            chunk = images[i:i + self.cmmd_batch_size]
            chunk_embs = self._compute_clip_embeddings_chunk(chunk)
            all_embeddings.append(chunk_embs)
        
        return torch.cat(all_embeddings, dim=0)
    
    @torch.no_grad()
    def _compute_clip_embeddings_chunk(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute CLIP or SigLIP embeddings for a chunk of images.
        
        Args:
            images: Image tensor, shape (B, 3, H, W), range [-1, 1]
            
        Returns:
            Embeddings, shape (B, embedding_dim)
        """
        # Convert from [-1, 1] to [0, 1] range
        images_01 = (images + 1) / 2
        images_01 = images_01.clamp(0, 1)
        
        # Resize to model input size
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
        
        # Get embeddings based on model type
        if self.use_siglip:
            # SigLIP models use get_image_features()
            image_embs = self.model.get_image_features(**inputs)
        else:
            # CLIP models use .image_embeds attribute
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
        Compute CMMD distance using batched MMD computation to avoid O(n²) memory usage.
        
        The MMD computation itself is optimized to process embeddings in chunks,
        avoiding the need to create full n×n kernel matrices in memory.
        GPU cache is cleared periodically to prevent memory fragmentation.
        
        Returns:
            CMMD value (lower is better)
        """
        if len(self.original_embeddings) == 0:
            raise ValueError("No embeddings accumulated. Call update() first.")
        
        # Clear GPU cache before starting computation
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        # Concatenate all embeddings
        orig_all = torch.cat(self.original_embeddings, dim=0)
        recon_all = torch.cat(self.reconstructed_embeddings, dim=0)
        
        # Move to device for computation
        orig_all = orig_all.to(self.device)
        recon_all = recon_all.to(self.device)
        
        # Compute MMD with batched computation (avoids O(n²) memory)
        cmmd_value = _compute_mmd(orig_all, recon_all, chunk_size=self.mmd_chunk_size)
        
        # Clear GPU cache after computation
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        # Clear embeddings from memory
        del orig_all, recon_all
        
        return cmmd_value


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
        cmmd_clip_model: CLIP/SigLIP model path or HuggingFace model name for CMMD 
                        (default: /data/projects/VAEs4RS/models/BiliSakura/Remote-CLIP-ViT-L-14/transformers)
                        Supports both CLIP and SigLIP models (auto-detected)
        cmmd_batch_size: Batch size for CLIP embedding computation in CMMD (default: 32).
                        Large batches will be automatically chunked to this size.
        mmd_chunk_size: Chunk size for MMD computation to avoid O(n²) memory (default: 1000).
                       Smaller values use less memory but may be slower.
    """
    
    def __init__(
        self, 
        device: str = "cuda", 
        compute_fid: bool = True,
        compute_cmmd: bool = False,
        fid_feature_extractor: Optional[nn.Module] = None,
        cmmd_clip_model: str = _CLIP_MODEL_NAME,
        cmmd_batch_size: int = 32,
        mmd_chunk_size: int = 1000
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
                if fid_feature_extractor is not None:
                    # Move feature extractor to device and set to eval mode
                    feature_extractor = fid_feature_extractor.to(device).eval()
                    # Ensure it's wrapped in no_grad context for efficiency
                    for param in feature_extractor.parameters():
                        param.requires_grad = False
                    
                    # Use normalize=False since we'll provide uint8 images in [0, 255] range
                    # Pass custom feature extractor
                    self.fid = FrechetInceptionDistance(
                        normalize=False,
                        feature=feature_extractor
                    ).to(device)
                else:
                    # Use normalize=False since we'll provide uint8 images in [0, 255] range
                    # Don't pass feature parameter when None (some versions don't support it)
                    self.fid = FrechetInceptionDistance(
                        normalize=False
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
                self.cmmd = CMMDCalculator(
                    device=device, 
                    clip_model_name=cmmd_clip_model,
                    cmmd_batch_size=cmmd_batch_size,
                    mmd_chunk_size=mmd_chunk_size
                )
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
