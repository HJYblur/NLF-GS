import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from avatar_utils.config import get_config


VIEW_ORDER = ["front", "back", "left", "right"]
IMG_EXTS = (".png", ".jpg", ".jpeg")


class AvatarDataset(Dataset):
    """
    Minimal dataset for processed inputs.

        Layout per subject (new naming):
            processed/<subject>/
                <subject>_front.(png|jpg|jpeg)
                <subject>_back.(png|jpg|jpeg)
                <subject>_left.(png|jpg|jpeg)
                <subject>_right.(png|jpg|jpeg)

    Behavior is controlled by config value `data.num_views`:
      - If num_views == 1: only the 'front' view is loaded.
      - If num_views == 4: load the four views in VIEW_ORDER.

    Outputs per sample:
      - images_float: torch.FloatTensor [V, C, H, W], normalized to [0,1]
      - images_uint8: torch.Uint8Tensor [V, C, H, W]
      - subject: str
      - view_names: List[str]
      - vertices3d: Optional[torch.FloatTensor] [N, 3]
      - vertices2d: Optional[torch.FloatTensor] [N, 2]
    """

    def __init__(self, root: str, transform: Optional[Any] = None):
        # Config
        cfg = get_config()
        self.debug: bool = bool(cfg.get("sys", {}).get("debug", False))
        self.num_views: int = int(cfg.get("data", {}).get("num_views", 1))
        self.root = Path(root)
        self.smplx_root = Path(cfg.get("data", {}).get("smplx_root", "data/THuman_2.0_smplx_params"))

        # Index subjects and required views
        self._records: List[Dict[str, Any]] = []
        self._index_subjects()
        if len(self._records) == 0:
            raise FileNotFoundError(
                f"No valid subjects found under {self.root}. Expected files like front.jpg/png, etc."
            )

    def __len__(self) -> int:
        if self.debug:
            return min(4, len(self._records))
        else:
            return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._records[idx]
        view_paths: List[Path] = rec["view_paths"]
        view_names: List[str] = rec["view_names"]

        imgs_f: List[torch.Tensor] = []
        imgs_u8: List[torch.Tensor] = []
        for p in view_paths:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img)
            f = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
            u8 = torch.from_numpy(arr.astype(np.uint8)).permute(2, 0, 1)
            imgs_f.append(f)
            imgs_u8.append(u8)

        images_float = torch.stack(imgs_f, dim=0)  # [V,C,H,W]
        images_uint8 = torch.stack(imgs_u8, dim=0)  # [V,C,H,W]

        # Placeholder augmentations hook (no-op for now)
        images_float, images_uint8 = apply_augmentations(
            images_float, images_uint8, rec["subject"]
        )

        return {
            "images_float": images_float,
            "images_uint8": images_uint8,
            "subject": rec["subject"],
            "view_names": view_names,
        }

    def _index_subjects(self) -> None:
        """Collect subjects with the required views present."""

        def find_view_file(
            subj_dir: Path, basename: str, subject_name: str
        ) -> Optional[Path]:
            """Find view file using new '<subject>_<view>' naming."""
            # Prefer new naming scheme
            for ext in IMG_EXTS:
                p_new = subj_dir / f"{subject_name}_{basename}{ext}"
                if p_new.exists():
                    return p_new
            return None

        # Find subject directories
        assert self.root.is_dir(), "Root directory is not valid."
        candidates: List[Path] = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir():
                candidates.append(child)

        needed = ["front"] if self.num_views == 1 else VIEW_ORDER

        for subj_dir in candidates:
            paths: List[Path] = []
            ok = True
            for v in needed:
                vp = find_view_file(subj_dir, v, subj_dir.name)
                if vp is None:
                    ok = False
                    break
                paths.append(vp)
            if ok:
                self._records.append(
                    {
                        "subject": subj_dir.name,
                        "view_paths": paths,
                        "view_names": needed,
                    }
                )


class ViewsChunkedDataset(Dataset):
    """Yield sequential view chunks from a subject-level dataset.

    For each subject sample with images_float [V,C,H,W], this wrapper produces
    items with images_float [K,C,H,W], where K=chunk_size (last chunk may be smaller).
    """

    def __init__(self, base_ds: Dataset, chunk_size: int):
        self.base_ds = base_ds
        self.chunk_size = max(1, int(chunk_size))
        self._chunks: List[Tuple[int, int, int]] = []
        for i in range(len(base_ds)):
            sample = base_ds[i]
            V = int(sample["images_float"].shape[0])
            for start in range(0, V, self.chunk_size):
                end = min(start + self.chunk_size, V)
                self._chunks.append((i, start, end))

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx, start, end = self._chunks[idx]
        sample = self.base_ds[base_idx]
        out: Dict[str, Any] = {
            "images_float": sample["images_float"][start:end],
            "images_uint8": sample["images_uint8"][start:end],
            "subject": sample.get("subject", ""),
        }
        if "view_names" in sample:
            out["view_names"] = sample["view_names"][start:end]
        return out
