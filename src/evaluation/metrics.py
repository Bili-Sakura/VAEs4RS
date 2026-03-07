"""
Reconstruction quality metrics for VAE evaluation.
PSNR, SSIM, LPIPS, FID, KID, CMMD, and RS-domain variants.
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

# FID (optional, requires torch-fidelity)
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except (ImportError, RuntimeError):
    FID_AVAILABLE = False
    FrechetInceptionDistance = None

# KID (optional, requires torch-fidelity)
try:
    from torchmetrics.image.kid import KernelInceptionDistance
    KID_AVAILABLE = True
except (ImportError, RuntimeError):
    KID_AVAILABLE = False
    KernelInceptionDistance = None

# CMMD (optional, requires transformers)
try:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor, SiglipModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    CLIPImageProcessor = CLIPVisionModelWithProjection = SiglipImageProcessor = SiglipModel = None


# MMD parameters
_CMMD_SIGMA = 10
_CMMD_SCALE = 1000


def _compute_mmd(x: torch.Tensor, y: torch.Tensor, chunk_size: int = 1000) -> float:
    """Memory-efficient batched MMD with Gaussian RBF kernel."""
    gamma = 1 / (2 * _CMMD_SIGMA**2)
    device = x.device
    
    x_sqnorms = (x ** 2).sum(dim=1)
    y_sqnorms = (y ** 2).sum(dim=1)
    
    def kernel_sum(a, b, a_sq, b_sq, chunk_size):
        total = 0.0
        count = 0
        for i in range(0, len(a), chunk_size):
            for j in range(0, len(b), chunk_size):
                ai, bj = a[i:i+chunk_size], b[j:j+chunk_size]
                dists = a_sq[i:i+chunk_size].unsqueeze(1) + b_sq[j:j+chunk_size].unsqueeze(0) - 2 * ai @ bj.T
                total += torch.exp(-gamma * dists).sum().item()
                count += ai.shape[0] * bj.shape[0]
            if device.type == 'cuda' and i % (chunk_size * 10) == 0:
                torch.cuda.empty_cache()
        return total / count if count > 0 else 0.0
    
    k_xx = kernel_sum(x, x, x_sqnorms, x_sqnorms, chunk_size)
    k_yy = kernel_sum(y, y, y_sqnorms, y_sqnorms, chunk_size)
    k_xy = kernel_sum(x, y, x_sqnorms, y_sqnorms, chunk_size)
    
    return _CMMD_SCALE * (k_xx + k_yy - 2 * k_xy)


class CMMDCalculator:
    """CMMD calculator using CLIP/SigLIP embeddings."""
    
    def __init__(self, device: str = "cuda", clip_model_name: str = "", batch_size: int = 32, chunk_size: int = 1000):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for CMMD: pip install transformers")
        
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.original_embeddings: List[torch.Tensor] = []
        self.reconstructed_embeddings: List[torch.Tensor] = []
        
        # Load model (auto-detect CLIP vs SigLIP)
        self._load_model(clip_model_name)
    
    def _load_model(self, model_path: str):
        """Load CLIP or SigLIP model."""
        from pathlib import Path
        import json
        
        path = Path(model_path)
        config_file = path / "config.json"
        
        # Detect model type
        use_siglip = False
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            use_siglip = "siglip" in cfg.get("model_type", "").lower()
        elif (path / "transformers" / "config.json").exists():
            use_siglip = True
            model_path = str(path / "transformers")
        
        try:
            if use_siglip:
                self.processor = SiglipImageProcessor.from_pretrained(model_path)
                self.model = SiglipModel.from_pretrained(model_path).eval().to(self.device)
            else:
                self.processor = CLIPImageProcessor.from_pretrained(model_path)
                self.model = CLIPVisionModelWithProjection.from_pretrained(model_path).eval().to(self.device)
            self.use_siglip = use_siglip
        except Exception as e:
            # Fallback to other type
            try:
                self.processor = CLIPImageProcessor.from_pretrained(model_path)
                self.model = CLIPVisionModelWithProjection.from_pretrained(model_path).eval().to(self.device)
                self.use_siglip = False
            except:
                raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Get input size
        if hasattr(self.processor, 'crop_size') and isinstance(self.processor.crop_size, dict):
            self.input_size = self.processor.crop_size["height"]
        elif hasattr(self.processor, 'size'):
            s = self.processor.size
            self.input_size = s.get("height", s.get("shortest_edge", 224)) if isinstance(s, dict) else s
        else:
            self.input_size = 224
    
    @torch.no_grad()
    def _get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP/SigLIP embeddings for images in [-1, 1] range."""
        # Convert to [0, 1] and resize
        images = ((images + 1) / 2).clamp(0, 1)
        images = nn.functional.interpolate(images, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        
        # Process in batches
        all_embs = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i+self.batch_size].permute(0, 2, 3, 1).cpu().numpy()
            inputs = self.processor(images=batch, do_normalize=True, do_center_crop=False, do_resize=False, do_rescale=False, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            embs = self.model.get_image_features(**inputs) if self.use_siglip else self.model(**inputs).image_embeds
            all_embs.append(embs / embs.norm(dim=-1, keepdim=True))
        
        return torch.cat(all_embs, dim=0)
    
    def reset(self):
        self.original_embeddings = []
        self.reconstructed_embeddings = []
    
    def update(self, original: torch.Tensor, reconstructed: torch.Tensor):
        self.original_embeddings.append(self._get_embeddings(original.clamp(-1, 1)).cpu())
        self.reconstructed_embeddings.append(self._get_embeddings(reconstructed.clamp(-1, 1)).cpu())
    
    def compute(self) -> float:
        if not self.original_embeddings:
            raise ValueError("No embeddings accumulated")
        
        orig = torch.cat(self.original_embeddings).to(self.device)
        recon = torch.cat(self.reconstructed_embeddings).to(self.device)
        result = _compute_mmd(orig, recon, self.chunk_size)
        
        del orig, recon
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        return result


@dataclass
class MetricResults:
    """Container for metric results."""
    psnr: float
    ssim: float
    lpips: float
    fid: Optional[float] = None
    kid: Optional[float] = None
    cmmd: Optional[float] = None
    fid_rs: Optional[float] = None
    kid_rs: Optional[float] = None
    lpips_rs: Optional[float] = None
    
    def __repr__(self):
        parts = [f"PSNR: {self.psnr:.2f} dB", f"SSIM: {self.ssim:.4f}", f"LPIPS: {self.lpips:.4f}"]
        if self.fid is not None:
            parts.append(f"FID: {self.fid:.2f}")
        if self.kid is not None:
            parts.append(f"KID: {self.kid:.4f}")
        if self.cmmd is not None:
            parts.append(f"CMMD: {self.cmmd:.2f}")
        if self.fid_rs is not None:
            parts.append(f"FID(rs): {self.fid_rs:.2f}")
        if self.kid_rs is not None:
            parts.append(f"KID(rs): {self.kid_rs:.4f}")
        if self.lpips_rs is not None:
            parts.append(f"LPIPS(rs): {self.lpips_rs:.4f}")
        return " | ".join(parts)
    
    def to_dict(self) -> dict:
        return {
            "psnr": self.psnr, "ssim": self.ssim, "lpips": self.lpips,
            "fid": self.fid, "kid": self.kid, "cmmd": self.cmmd,
            "fid_rs": self.fid_rs, "kid_rs": self.kid_rs, "lpips_rs": self.lpips_rs,
        }


class MetricCalculator:
    """Calculator for image reconstruction metrics.

    Supports standard metrics (PSNR, SSIM, LPIPS, FID, KID, CMMD) and
    remote-sensing domain variants (FID(rs), KID(rs), LPIPS(rs)) that
    use RS-trained InceptionV3 / VGG feature extractors.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        compute_fid: bool = True,
        compute_kid: bool = False,
        compute_cmmd: bool = False,
        fid_feature_extractor: Optional[nn.Module] = None,
        cmmd_clip_model: str = "",
        cmmd_batch_size: int = 32,
        mmd_chunk_size: int = 1000,
        rs_inception_extractor: Optional[nn.Module] = None,
        rs_vgg_metric: Optional[nn.Module] = None,
    ):
        self.device = device
        self.compute_fid = compute_fid and FID_AVAILABLE
        self.compute_kid = compute_kid and KID_AVAILABLE
        self.compute_cmmd = compute_cmmd and TRANSFORMERS_AVAILABLE
        
        # Core metrics
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        
        # FID
        self.fid = None
        if self.compute_fid:
            try:
                kwargs = {"normalize": False}
                if fid_feature_extractor:
                    fid_feature_extractor = fid_feature_extractor.to(device).eval()
                    kwargs["feature"] = fid_feature_extractor
                self.fid = FrechetInceptionDistance(**kwargs).to(device)
            except Exception:
                self.compute_fid = False
        
        # KID
        self.kid = None
        if self.compute_kid:
            try:
                kwargs = {"normalize": False, "subset_size": 50}
                self.kid = KernelInceptionDistance(**kwargs).to(device)
            except Exception:
                self.compute_kid = False
        
        # CMMD
        self.cmmd = None
        if self.compute_cmmd and cmmd_clip_model:
            try:
                self.cmmd = CMMDCalculator(device, cmmd_clip_model, cmmd_batch_size, mmd_chunk_size)
            except Exception:
                self.compute_cmmd = False
        
        # ---- RS-domain metrics -------------------------------------------
        # FID(rs) / KID(rs) using RS-trained InceptionV3 features
        self.fid_rs = None
        self.kid_rs = None
        self.compute_fid_rs = rs_inception_extractor is not None and FID_AVAILABLE
        self.compute_kid_rs = rs_inception_extractor is not None and KID_AVAILABLE
        if rs_inception_extractor is not None:
            rs_inception_extractor = rs_inception_extractor.to(device).eval()
            if FID_AVAILABLE:
                try:
                    self.fid_rs = FrechetInceptionDistance(
                        feature=rs_inception_extractor, normalize=False,
                    ).to(device)
                except Exception:
                    self.compute_fid_rs = False
            if KID_AVAILABLE:
                try:
                    self.kid_rs = KernelInceptionDistance(
                        feature=rs_inception_extractor, normalize=False,
                        subset_size=50,
                    ).to(device)
                except Exception:
                    self.compute_kid_rs = False
        
        # LPIPS(rs) using RS-trained VGG features
        self.lpips_rs_metric = None
        self.compute_lpips_rs = rs_vgg_metric is not None
        if rs_vgg_metric is not None:
            self.lpips_rs_metric = rs_vgg_metric.to(device).eval()
        
        self.psnr_values: List[float] = []
        self.ssim_values: List[float] = []
        self.lpips_values: List[float] = []
        self.lpips_rs_values: List[float] = []
    
    def reset(self):
        self.psnr_values, self.ssim_values, self.lpips_values = [], [], []
        self.lpips_rs_values = []
        for m in [self.psnr, self.ssim, self.lpips]:
            m.reset()
        if self.fid:
            self.fid.reset()
        if self.kid:
            self.kid.reset()
        if self.cmmd:
            self.cmmd.reset()
        if self.fid_rs:
            self.fid_rs.reset()
        if self.kid_rs:
            self.kid_rs.reset()
    
    @torch.no_grad()
    def update(self, original: torch.Tensor, reconstructed: torch.Tensor):
        """Update metrics with a batch of images in [-1, 1] range."""
        original = original.clamp(-1, 1)
        reconstructed = reconstructed.clamp(-1, 1)
        
        # Per-batch metrics
        self.psnr_values.append(self.psnr(reconstructed, original).item())
        self.psnr.reset()
        self.ssim_values.append(self.ssim(reconstructed, original).item())
        self.ssim.reset()
        self.lpips_values.append(self.lpips(reconstructed, original).item())
        self.lpips.reset()
        
        # Shared uint8 conversion for FID / KID / FID(rs) / KID(rs)
        orig_uint8 = (((original + 1) / 2).clamp(0, 1) * 255).to(torch.uint8)
        recon_uint8 = (((reconstructed + 1) / 2).clamp(0, 1) * 255).to(torch.uint8)
        
        # FID (accumulates)
        if self.fid:
            self.fid.update(orig_uint8, real=True)
            self.fid.update(recon_uint8, real=False)
        
        # KID (accumulates)
        if self.kid:
            self.kid.update(orig_uint8, real=True)
            self.kid.update(recon_uint8, real=False)
        
        # CMMD (accumulates)
        if self.cmmd:
            self.cmmd.update(original, reconstructed)
        
        # ---- RS-domain metrics -------------------------------------------
        if self.fid_rs:
            self.fid_rs.update(orig_uint8, real=True)
            self.fid_rs.update(recon_uint8, real=False)
        
        if self.kid_rs:
            self.kid_rs.update(orig_uint8, real=True)
            self.kid_rs.update(recon_uint8, real=False)
        
        if self.lpips_rs_metric is not None:
            self.lpips_rs_values.append(
                self.lpips_rs_metric(original, reconstructed).item()
            )
    
    def compute(self) -> MetricResults:
        kid_val = None
        if self.kid:
            kid_mean, _ = self.kid.compute()
            kid_val = kid_mean.item()
        
        kid_rs_val = None
        if self.kid_rs:
            kid_rs_mean, _ = self.kid_rs.compute()
            kid_rs_val = kid_rs_mean.item()
        
        return MetricResults(
            psnr=np.mean(self.psnr_values),
            ssim=np.mean(self.ssim_values),
            lpips=np.mean(self.lpips_values),
            fid=self.fid.compute().item() if self.fid else None,
            kid=kid_val,
            cmmd=self.cmmd.compute() if self.cmmd else None,
            fid_rs=self.fid_rs.compute().item() if self.fid_rs else None,
            kid_rs=kid_rs_val,
            lpips_rs=np.mean(self.lpips_rs_values) if self.lpips_rs_values else None,
        )
