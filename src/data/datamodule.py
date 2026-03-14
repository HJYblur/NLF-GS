from typing import Optional, Dict, Any, Callable

import math
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import lightning as L

from src.data.datasets import AvatarDataset, ViewsChunkedDataset


def worker_init_fn(worker_id):
    """Initialize each DataLoader worker with a unique random seed.
    
    This prevents reproducibility issues in multi-worker scenarios.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class AvatarDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping AvatarDataset with simple train/val split."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.train_ds: Optional[torch.utils.data.Dataset] = None
        self.val_ds: Optional[torch.utils.data.Dataset] = None

    def setup(self, stage: Optional[str] = None):
        data_cfg = self.cfg.get("data", {})
        train_cfg = self.cfg.get("train", {})
        train_base_ds = AvatarDataset(
            root=data_cfg.get("root", "processed"),
            transform=None,
        )
        val_base_ds = AvatarDataset(
            root=data_cfg.get("root", "processed"),
            transform=None,
        )
        # Chunk views sequentially based on desired views-per-batch (use train batch_size)
        chunk_size = int(train_cfg.get("batch_size", 4))

        n = len(train_base_ds)
        val_ratio = float(train_cfg.get("val_ratio", 0.0))
        if val_ratio > 0.0 and n > 1:
            n_val = max(1, int(math.floor(n * val_ratio)))
            idx = torch.randperm(n).tolist()
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
            if len(train_idx) == 0:  # fallback to at least one train sample
                train_idx, val_idx = idx[:-1], idx[-1:]
            train_base = Subset(train_base_ds, train_idx)
            val_base = Subset(val_base_ds, val_idx)
            self.train_ds = ViewsChunkedDataset(train_base, chunk_size)
            self.val_ds = ViewsChunkedDataset(val_base, chunk_size)
        else:
            self.train_ds = ViewsChunkedDataset(train_base_ds, chunk_size)
            self.val_ds = None

    def train_dataloader(self) -> DataLoader:
        train_cfg = self.cfg.get("train", {})
        return DataLoader(
            self.train_ds,
            # Each batch is one sequential chunk of views
            batch_size=1,
            num_workers=int(train_cfg.get("num_workers", 2)),
            shuffle=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_ds is None:
            return None
        train_cfg = self.cfg.get("train", {})
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=int(train_cfg.get("num_workers", 2)),
            shuffle=False,
            worker_init_fn=worker_init_fn,
        )
