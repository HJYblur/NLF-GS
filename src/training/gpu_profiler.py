"""Lightweight GPU-memory profiling callback for Lightning.

Logs allocated / reserved / peak VRAM at key points during training so you
can identify which section (data-loading, forward, backward, optimizer step)
consumes the most GPU memory.

Enable by setting ``train.profile_gpu: True`` in your config YAML.
"""

import torch
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info


def _gpu_mem_mb() -> dict:
    """Return a snapshot of CUDA memory stats (in MB) for the current device."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_MB": torch.cuda.memory_allocated() / 1024 ** 2,
        "reserved_MB": torch.cuda.memory_reserved() / 1024 ** 2,
        "peak_allocated_MB": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_reserved_MB": torch.cuda.max_memory_reserved() / 1024 ** 2,
    }


def _log_mem(tag: str) -> None:
    stats = _gpu_mem_mb()
    if not stats:
        return
    parts = " ".join(f"{k}={v:.2f}" for k, v in stats.items())
    rank_zero_info(f"[GpuMemoryProfiler] {tag}: {parts}")


class GpuMemoryProfilerCallback(L.Callback):
    """Logs GPU memory at every training-step boundary.

    The callback resets peak-memory counters at the start of each training
    step so you get *per-step* peaks, making it easy to compare across steps
    and find the memory high-water mark.
    """

    # ------------------------------------------------------------------
    # Per-epoch
    # ------------------------------------------------------------------
    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _log_mem("epoch_start")

    def on_train_epoch_end(self, trainer, pl_module):
        _log_mem("epoch_end")

    # ------------------------------------------------------------------
    # Per-step (batch)
    # ------------------------------------------------------------------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _log_mem(f"batch_{batch_idx}_start (after data load)")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _log_mem(f"batch_{batch_idx}_end   (after optim step)")

    def on_before_backward(self, trainer, pl_module, loss):
        _log_mem("before_backward")

    def on_after_backward(self, trainer, pl_module):
        _log_mem("after_backward")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        _log_mem("before_optimizer_step")
