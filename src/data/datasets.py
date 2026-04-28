from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from avatar_utils.config import get_config
from avatar_utils.smplx_loader import load_smplx_coord3d
from avatar_utils.smplx_loader import vertices_3d_to_2d
from avatar_utils.camera import load_camera_mapping


VIEW_ORDER = ["front", "back", "left", "right"]
IMG_EXTS = (".png", ".jpg", ".jpeg")


class AvatarDataset(Dataset):
    """
    Minimal dataset for processed inputs.

        Layout per subject (new naming):
            processed/<subject>/
                <subject>_front.(png|jpg|jpeg)
                <subject>_front_mask.(png|jpg|jpeg)
                <subject>_back.(png|jpg|jpeg)
                <subject>_back_mask.(png|jpg|jpeg)
                <subject>_left.(png|jpg|jpeg)
                <subject>_left_mask.(png|jpg|jpeg)
                <subject>_right.(png|jpg|jpeg)
                <subject>_right_mask.(png|jpg|jpeg)
                
        Layout per smplx_param (preferred):
            processed/<subject>/<subject>_smplx.pkl

    The dataset always loads all canonical views in VIEW_ORDER.
    Training-time fusion behavior is controlled by `data.num_views` in the trainer
    (1 = no fusion, 4 = fuse multi-view features).

    Outputs per sample:
      - images_float: torch.FloatTensor [V, C, H, W], normalized to [0,1]
      - images_uint8: torch.Uint8Tensor [V, C, H, W]
      - masks_float: torch.FloatTensor [V, 1, H, W], normalized to [0,1]
      - subject: str
      - view_names: List[str]
      - vertices3d: Optional[torch.FloatTensor] [N, 3]
    """

    def __init__(self, root: str, transform: Optional[Any] = None):
        # Config
        cfg = get_config()
        self.debug: bool = bool(cfg.get("sys", {}).get("debug", False))
        image_size = cfg.get("data", {}).get("image_size", (1024, 1024))
        self.target_w: int = int(image_size[0])
        self.target_h: int = int(image_size[1])
        self.root = Path(root)
        data_cfg = cfg.get("data", {})
        self.smplx_root = Path(
            data_cfg.get(
                "smplx_root",
                data_cfg.get("processed_root", "processed"),
            )
        )

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
        masks_f: List[torch.Tensor] = []
        for p in view_paths:
            img = Image.open(p).convert("RGB")
            if img.size != (self.target_w, self.target_h):
                img = img.resize((self.target_w, self.target_h), Image.BILINEAR)
            arr = np.asarray(img)
            f = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
            imgs_f.append(f)
            
            # Load corresponding mask image
            mask_path = p.parent / f"{p.stem}_mask{p.suffix}"
            if mask_path.exists():
                mask_img = Image.open(mask_path).convert("L")  # Load as grayscale
                if mask_img.size != (self.target_w, self.target_h):
                    mask_img = mask_img.resize((self.target_w, self.target_h), Image.BILINEAR)
                mask_arr = np.asarray(mask_img)
                mask_f = torch.from_numpy(mask_arr.astype(np.float32) / 255.0).unsqueeze(0)  # [1,H,W]
            else:
                # If mask doesn't exist, create an all-ones mask
                mask_f = torch.ones(1, self.target_h, self.target_w, dtype=torch.float32)
            masks_f.append(mask_f)

        images_float = torch.stack(imgs_f, dim=0)  # [V,C,H,W]
        masks_float = torch.stack(masks_f, dim=0)  # [V,1,H,W]

        images_uint8 = (images_float.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)  # [V,C,H,W]

        # Load SMPL-X 3D vertices (prefer canonical smplx_param.pkl).
        subject = rec["subject"]
        candidates = [
            self.smplx_root / subject / "smplx_param.pkl",
            self.smplx_root / subject / f"{subject}_smplx.pkl",
            rec["subject_dir"] / "smplx_param.pkl",
            rec["subject_dir"] / f"{subject}_smplx.pkl",
            self.smplx_root / subject / "mesh_smplx.obj",
        ]
        smplx_path = next((p for p in candidates if p.exists()), None)
        if smplx_path is not None:
            vertices3d = load_smplx_coord3d(str(smplx_path))
        else:
            vertices3d = torch.empty(0, 3, dtype=torch.float32)

        # Project 3D vertices to 2D for each view using precomputed camera params
        if vertices3d.shape[0] > 0:
            viewmats, Ks = load_camera_mapping(view_names)  # (V,4,4), (V,3,3)
            verts2d_list = []
            for v_idx in range(viewmats.shape[0]):
                v2d = vertices_3d_to_2d(vertices3d, Ks[v_idx], viewmats[v_idx])
                verts2d_list.append(v2d)
            vertices2d = torch.stack(verts2d_list, dim=0)  # (V, Nv, 2)
        else:
            V = len(view_names)
            vertices2d = torch.empty(V, 0, 2, dtype=torch.float32)

        return {
            "images_float": images_float,
            "images_uint8": images_uint8,
            "masks_float": masks_float,
            "subject": rec["subject"],
            "view_names": view_names,
            "vertices3d": vertices3d,
            "vertices2d": vertices2d,
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

        needed = VIEW_ORDER

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
                        "subject_dir": subj_dir,
                        "view_paths": paths,
                        "view_names": list(needed),
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
        if "masks_float" in sample:
            out["masks_float"] = sample["masks_float"][start:end]
        if "view_names" in sample:
            out["view_names"] = sample["view_names"][start:end]
        if "vertices3d" in sample:
            out["vertices3d"] = sample["vertices3d"]
        if "vertices2d" in sample:
            out["vertices2d"] = sample["vertices2d"][start:end]
        return out
