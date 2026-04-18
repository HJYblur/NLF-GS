"""Training helpers: batch unpacking, optimizer/scheduler, optional GPU VRAM profiling."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info


# ---------------------------------------------------------------------------
# Batch unpacking (AvatarDataset / ViewsChunkedDataset)
# ---------------------------------------------------------------------------


def unpack_training_batch(
    batch: Dict[str, Any],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Tuple[int, int, int],
    Any,
    Optional[List[str]],
    torch.Tensor,
    torch.Tensor,
]:
    """Extract tensors from the dataset batch and normalize leading batch dimensions."""
    assert "images_float" in batch and "images_uint8" in batch, (
        "Batch missing 'images_float' or 'images_uint8' key"
    )

    img_float = batch["images_float"]
    if img_float.ndim == 5 and img_float.shape[0] == 1:
        img_float = img_float[0]

    img_uint8 = batch["images_uint8"]
    if img_uint8.ndim == 5 and img_uint8.shape[0] == 1:
        img_uint8 = img_uint8[0]

    masks_float = batch.get("masks_float", None)
    if masks_float is not None:
        if masks_float.ndim == 5 and masks_float.shape[0] == 1:
            masks_float = masks_float[0]

    B, _, H, W = img_float.shape

    subject = batch.get("subject", None)
    if isinstance(subject, (list, tuple)):
        subject = subject[0]

    view_names = batch.get("view_names", None)
    if isinstance(view_names, (list, tuple)):
        if len(view_names) == 1 and isinstance(view_names[0], (list, tuple)):
            view_names = list(view_names[0])
        else:
            view_names = [
                vn[0] if isinstance(vn, (list, tuple)) else vn for vn in view_names
            ]

    vertices3d = batch.get("vertices3d", None)
    if vertices3d is not None:
        if vertices3d.ndim == 3 and vertices3d.shape[0] == 1:
            vertices3d = vertices3d[0]
    else:
        vertices3d = torch.empty(0, 3)

    vertices2d = batch.get("vertices2d", None)
    if vertices2d is not None:
        if vertices2d.ndim == 4 and vertices2d.shape[0] == 1:
            vertices2d = vertices2d[0]
    else:
        vertices2d = torch.empty(B, 0, 2)

    return img_float, img_uint8, masks_float, (B, H, W), subject, view_names, vertices3d, vertices2d


# ---------------------------------------------------------------------------
# AdamW param groups + cosine LR (enable via configure_optimizers in NLF-GS)
# ---------------------------------------------------------------------------


def _is_no_decay(name: str, param: torch.Tensor) -> bool:
    if param.ndim == 1:
        return True
    if name.endswith(".bias"):
        return True
    n = name.lower()
    if "bn" in n or "norm" in n:
        return True
    return False


def configure_nlf_gaussian_optimizers(pl_module: "L.LightningModule") -> Dict[str, Any]:
    """Build AdamW (decoder + optional fusion + backbone) and step-wise cosine LR."""
    base_lr = float(pl_module.hparams.lr)
    bb_mult = float(getattr(pl_module.hparams, "bb_lr_mult", 0.1))
    wd = float(pl_module.hparams.wd)

    dec_decay, dec_no_decay = [], []
    for mod in (pl_module.decoder, pl_module.view_fusion):
        if mod is None:
            continue
        for n, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            (dec_no_decay if _is_no_decay(n, p) else dec_decay).append(p)

    bb_decay, bb_no_decay = [], []
    for n, p in pl_module.backbone.named_parameters():
        if not p.requires_grad:
            continue
        (bb_no_decay if _is_no_decay(n, p) else bb_decay).append(p)

    optimizer = torch.optim.AdamW(
        [
            {
                "name": "decoder_decay",
                "params": dec_decay,
                "lr": base_lr,
                "weight_decay": wd,
            },
            {
                "name": "decoder_no_decay",
                "params": dec_no_decay,
                "lr": base_lr,
                "weight_decay": 0.0,
            },
            {
                "name": "backbone_decay",
                "params": bb_decay,
                "lr": base_lr * bb_mult,
                "weight_decay": wd,
            },
            {
                "name": "backbone_no_decay",
                "params": bb_no_decay,
                "lr": base_lr * bb_mult,
                "weight_decay": 0.0,
            },
        ],
        betas=tuple(pl_module.hparams.betas)
        if isinstance(pl_module.hparams.betas, (list, tuple))
        else (0.9, 0.99),
        eps=float(pl_module.hparams.eps),
    )

    total_steps = int(pl_module.trainer.estimated_stepping_batches)
    warmup_ratio = float(pl_module.hparams.warmup_ratio)
    warmup_ratio = min(1.0, max(0.0, warmup_ratio))
    warmup_steps = int(warmup_ratio * max(1, total_steps))
    min_lr_ratio = float(getattr(pl_module.hparams, "min_lr_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        denom = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / denom
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ---------------------------------------------------------------------------
# GPU memory profiling callback (enable with train.profile_gpu: true)
# ---------------------------------------------------------------------------


def _gpu_mem_mb() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_MB": torch.cuda.memory_allocated() / 1024 ** 2,
        "reserved_MB": torch.cuda.memory_reserved() / 1024 ** 2,
        "peak_allocated_MB": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_reserved_MB": torch.cuda.max_memory_reserved() / 1024 ** 2,
    }


def _log_gpu_mem(tag: str) -> None:
    stats = _gpu_mem_mb()
    if not stats:
        return
    parts = " ".join(f"{k}={v:.2f}" for k, v in stats.items())
    rank_zero_info(f"[GpuMemoryProfiler] {tag}: {parts}")


class GpuMemoryProfilerCallback(L.Callback):
    """Logs CUDA VRAM at train boundaries when ``train.profile_gpu`` is enabled."""

    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _log_gpu_mem("epoch_start")

    def on_train_epoch_end(self, trainer, pl_module):
        _log_gpu_mem("epoch_end")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _log_gpu_mem(f"batch_{batch_idx}_start (after data load)")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _log_gpu_mem(f"batch_{batch_idx}_end   (after optim step)")

    def on_before_backward(self, trainer, pl_module, loss):
        _log_gpu_mem("before_backward")

    def on_after_backward(self, trainer, pl_module):
        _log_gpu_mem("after_backward")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        _log_gpu_mem("before_optimizer_step")
