"""Shared NLF-GS device setup and Lightning module construction (train + inference)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from backbone.resnet_plain import PlainResNet50FeatureExtractor
from decoder.gaussian_decoder import GaussianDecoder
from encoder.feature_extractor import FeatureExtractor
from render.gaussian_renderer import GsplatRenderer
from training.nlfgs import NlfGaussianModel


def device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    device_str = None
    if isinstance(cfg, dict):
        sys_cfg = cfg.get("sys")
        if isinstance(sys_cfg, dict):
            device_str = sys_cfg.get("device")
    if not device_str:
        device_str = "cpu"
    return torch.device(device_str)


def gsplat_renderer_if_cuda(device: torch.device) -> Optional[GsplatRenderer]:
    """gsplat rasterization is CUDA-only in this project."""
    return GsplatRenderer() if device.type == "cuda" else None


def apply_matmul_precision_for_device(cfg: Dict[str, Any], device: torch.device) -> None:
    try:
        matmul_prec = None
        if isinstance(cfg, dict):
            sys_cfg = cfg.get("sys")
            if isinstance(sys_cfg, dict):
                matmul_prec = sys_cfg.get("matmul_precision")
        if device.type == "cuda":
            torch.set_float32_matmul_precision(matmul_prec or "high")
    except Exception:
        pass


def build_nlf_gaussian_model(cfg: Dict[str, Any], device: torch.device) -> NlfGaussianModel:
    backbone_cfg = cfg.get("backbone") or {}
    if not isinstance(backbone_cfg, dict):
        backbone_cfg = {}
    train_cfg = cfg.get("train", {})
    train_decoder_only = bool(train_cfg.get("train_decoder_only", True))
    encoder = str(backbone_cfg.get("encoder", "fpn")).strip().lower()

    if encoder in ("plain", "resnet_plain", "resnet50_plain"):
        plain = backbone_cfg.get("plain") if isinstance(backbone_cfg.get("plain"), dict) else {}
        proj = plain.get("proj_channels", backbone_cfg.get("plain_proj_channels"))
        if proj is None:
            proj = 256
        proj_int: Optional[int]
        if proj is False or str(proj).lower() in ("none", "null", ""):
            proj_int = None
        else:
            pi = int(proj)
            proj_int = None if pi <= 0 else pi
        pretrained_fb = bool(plain.get("pretrained_fallback", True))
        backbone = PlainResNet50FeatureExtractor(
            weights_path=backbone_cfg.get("resnet50_weights_path"),
            frozen=train_decoder_only,
            proj_channels=proj_int,
            pretrained_fallback=pretrained_fb,
        )
    else:
        fpn_levels = tuple(backbone_cfg.get("fpn_levels", ["p2", "p3", "p4"]))
        backbone = FeatureExtractor(
            fpn_levels=fpn_levels,
            resnet_weights_path=backbone_cfg.get("resnet50_weights_path"),
            freeze_resnet_fpn=train_decoder_only,
        )
    decoder = GaussianDecoder()
    renderer = gsplat_renderer_if_cuda(device)
    return NlfGaussianModel(
        backbone=backbone,
        decoder=decoder,
        renderer=renderer,
        train_decoder_only=train_decoder_only,
    )
