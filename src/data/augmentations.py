from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class SynchronizedPhotometricAugmentation:
    """Photometric-only augmentation synchronized across all views in one sample."""

    enabled: bool = False
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    gamma: float = 0.08
    noise_std: float = 0.01
    blur_prob: float = 0.2
    jpeg_prob: float = 0.2
    jpeg_quality_min: int = 75
    jpeg_quality_max: int = 95

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SynchronizedPhotometricAugmentation":
        aug_cfg = cfg.get("data", {}).get("augmentation", {})
        return cls(
            enabled=bool(aug_cfg.get("enabled", False)),
            brightness=float(aug_cfg.get("brightness", 0.1)),
            contrast=float(aug_cfg.get("contrast", 0.1)),
            saturation=float(aug_cfg.get("saturation", 0.1)),
            gamma=float(aug_cfg.get("gamma", 0.08)),
            noise_std=float(aug_cfg.get("noise_std", 0.01)),
            blur_prob=float(aug_cfg.get("blur_prob", 0.2)),
            jpeg_prob=float(aug_cfg.get("jpeg_prob", 0.2)),
            jpeg_quality_min=int(aug_cfg.get("jpeg_quality_min", 75)),
            jpeg_quality_max=int(aug_cfg.get("jpeg_quality_max", 95)),
        )

    def __call__(self, images: torch.Tensor, seed: int = None) -> torch.Tensor:
        """Apply augmentation to [V,C,H,W] tensor in [0,1]."""
        out, _ = self.apply_with_info(images, seed=seed)
        return out

    def apply_with_info(self, images: torch.Tensor, seed: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply augmentation and return both image tensor and sampled parameters.
        
        Args:
            images: Input images [V,C,H,W] in range [0,1]
            seed: Optional seed for reproducible augmentation per subject
        """
        info: Dict[str, Any] = {"enabled": bool(self.enabled)}
        if (not self.enabled) or images.ndim != 4:
            return images, info
        
        # Set random seed for this subject if provided
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        out = images.clone()

        # Shared parameters across all views.
        b = self._sample_factor(self.brightness, generator)
        c = self._sample_factor(self.contrast, generator)
        sat = self._sample_factor(self.saturation, generator)
        g = self._sample_gamma(self.gamma, generator)
        info.update({"brightness_factor": b, "contrast_factor": c, "saturation_factor": sat, "gamma_factor": g})

        out = out * b
        out = (out - 0.5) * c + 0.5

        gray = out.mean(dim=1, keepdim=True)
        out = gray + sat * (out - gray)

        out = out.clamp(0.0, 1.0)
        out = out.pow(g)

        sigma = 0.0
        if self.noise_std > 0:
            sigma = float(torch.rand(1, generator=generator).item()) * self.noise_std
            if sigma > 0:
                # Generate noise with the same generator for reproducibility
                noise_shape = out[:1].shape
                if generator is not None:
                    noise = torch.randn(noise_shape, generator=generator, device=out.device, dtype=out.dtype) * sigma
                else:
                    noise = torch.randn_like(out[:1]) * sigma
                out = out + noise
        info["noise_sigma"] = sigma

        blur_applied = bool(self.blur_prob > 0 and torch.rand(1, generator=generator).item() < self.blur_prob)
        if blur_applied:
            out = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
        info["blur_applied"] = blur_applied

        out = out.clamp(0.0, 1.0)

        jpeg_applied = False
        jpeg_quality = -1
        if self.jpeg_prob > 0 and torch.rand(1, generator=generator).item() < self.jpeg_prob:
            jpeg_applied = True
            jpeg_quality = int(
                torch.randint(self.jpeg_quality_min, self.jpeg_quality_max + 1, (1,), generator=generator).item()
            )
            out = self._apply_jpeg(out, jpeg_quality)
        info["jpeg_applied"] = jpeg_applied
        info["jpeg_quality"] = jpeg_quality

        return out.clamp(0.0, 1.0), info

    @staticmethod
    def _sample_factor(amount: float, generator=None) -> float:
        if amount <= 0:
            return 1.0
        lo = 1.0 - amount
        hi = 1.0 + amount
        return float((lo + (hi - lo) * torch.rand(1, generator=generator)).item())

    @staticmethod
    def _sample_gamma(amount: float, generator=None) -> float:
        if amount <= 0:
            return 1.0
        lo = max(0.1, 1.0 - amount)
        hi = 1.0 + amount
        return float((lo + (hi - lo) * torch.rand(1, generator=generator)).item())

    @staticmethod
    def _apply_jpeg(images: torch.Tensor, quality: int) -> torch.Tensor:
        v, _, _, _ = images.shape
        out = []
        for i in range(v):
            arr = (images[i].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")
            pil_img = Image.fromarray(arr, mode="RGB")
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            decoded = Image.open(buf).convert("RGB")
            t = torch.from_numpy(np.array(decoded, dtype=np.uint8, copy=True))
            t = t.permute(2, 0, 1).float() / 255.0
            out.append(t)
        return torch.stack(out, dim=0).to(images.device)
