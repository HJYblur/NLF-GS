from typing import Optional, Dict, Any, Callable
from pathlib import Path
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
        processed_root = data_cfg.get("processed_root", "processed")

        # Build a single base dataset and create deterministic random split by subject.
        processed_root = Path(processed_root)
        base_ds = AvatarDataset(root=processed_root, transform=None)

        # Chunk views sequentially based on desired views-per-batch (use train batch_size)
        chunk_size = int(train_cfg.get("batch_size", 4))

        n = len(base_ds)

        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        val_file = out_dir / "split_val.txt"
        train_file = out_dir / "split_train.txt"

        subjects = [rec["subject"] for rec in base_ds._records]
        subject_to_idx = {s: i for i, s in enumerate(subjects)}

        if val_file.exists() and train_file.exists():
            # Use fixed split from files
            val_subjects = [l.strip() for l in val_file.read_text().splitlines() if l.strip()]
            train_subjects = [l.strip() for l in train_file.read_text().splitlines() if l.strip()]

            missing_val = [s for s in val_subjects if s not in subject_to_idx]
            missing_train = [s for s in train_subjects if s not in subject_to_idx]
            if missing_val:
                print(f"Warning: {len(missing_val)} subjects in data/split_val.txt not found in dataset: {missing_val}")
            if missing_train:
                print(f"Warning: {len(missing_train)} subjects in data/split_train.txt not found in dataset: {missing_train}")

            val_idx = [subject_to_idx[s] for s in val_subjects if s in subject_to_idx]
            train_idx = [subject_to_idx[s] for s in train_subjects if s in subject_to_idx]

        else:
            # Number of validation subjects to reserve (default 100)
            n_val_requested = int(train_cfg.get("val_count", 100))
            n_val = min(n_val_requested, max(0, n))

            # Deterministic RNG seed for reproducibility
            seed = int(train_cfg.get("seed", 42))
            rng = np.random.RandomState(seed)

            indices = np.arange(len(subjects))
            rng.shuffle(indices)

            # Assign val/train indices
            val_idx = indices[:n_val].tolist()
            train_idx = indices[n_val:].tolist()
            if len(train_idx) == 0 and len(val_idx) > 0:
                # keep at least one training subject
                train_idx = val_idx[:-1]
                val_idx = val_idx[-1:]

            # Persist generated splits for reproducibility
            val_subjects = [subjects[i] for i in val_idx]
            train_subjects = [subjects[i] for i in train_idx]
            with open(val_file, "w") as f:
                f.write("\n".join(val_subjects) + ("\n" if val_subjects else ""))
            with open(train_file, "w") as f:
                f.write("\n".join(train_subjects) + ("\n" if train_subjects else ""))

        # Create Subset datasets for train/val
        # THIS TAKES FREAKING AGES !!! (I think the Subset)
        if len(train_idx) > 0:
            train_base = Subset(base_ds, train_idx)
            self.train_ds = ViewsChunkedDataset(train_base, chunk_size)
        else:
            self.train_ds = None

        if len(val_idx) > 0:
            val_base = Subset(base_ds, val_idx)
            self.val_ds = ViewsChunkedDataset(val_base, chunk_size)
        else:
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
