"""Shared NLF-GS device setup and Lightning module construction (train + inference)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

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
    backbone_cfg = cfg.get("backbone", {})
    train_cfg = cfg.get("train", {})
    train_decoder_only = bool(train_cfg.get("train_decoder_only", True))
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
