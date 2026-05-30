#!/usr/bin/env python3
"""
Convert Lemon's processed THuman-style data to the directory layout expected by GHG.

Expected source layout:
    processed/
      0001/
        0001_0.png
        0001_0_mask.png
        0001_15.png
        0001_15_mask.png
        ...
        smpl_param.pkl
        smplx_param.pkl

Expected camera layout:
    THuman_cameras/
      thuman_0.json
      thuman_15.json
      ...
      thuman_345.json

Default GHG input mapping:
    GHG view 0  <- source angle 0
    GHG view 6  <- source angle 135
    GHG view 11 <- source angle 255

Default output:
    Generalizable-Human-Gaussians/datasets/THuman/
    val/img/
    ├── 0004_000/
    │   └── 0.jpg
    ├── 0004_006/
    │   └── 0.jpg
    └── 0004_011/
        └── 0.jpg

    val/mask/
    ├── 0004_000/
    │   └── 0.png
    ├── 0004_006/
    │   └── 0.png
    └── 0004_011/
        └── 0.png

    val/parm/
    ├── 0004_000/
    │   ├── 0_intrinsic.npy
    │   └── 0_extrinsic.npy
    ├── 0004_006/
    │   ├── 0_intrinsic.npy
    │   └── 0_extrinsic.npy
    └── 0004_011/
        ├── 0_intrinsic.npy
        └── 0_extrinsic.npy

Notes:
- This script only creates the data layout and camera .npy files.
- It can create the required map folders, but the actual position/visibility maps still need
  GHG's preprocessing scripts and valid SMPL-X OBJ meshes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


MAP_DIRS = [
    "position_map_uv_space",
    "position_map_uv_space_outer_shell_1",
    "position_map_uv_space_outer_shell_2",
    "position_map_uv_space_outer_shell_3",
    "position_map_uv_space_outer_shell_4",
    "visibility_map_uv_space",
    "visibility_map_uv_space_outer_shell_1",
    "visibility_map_uv_space_outer_shell_2",
    "visibility_map_uv_space_outer_shell_3",
    "visibility_map_uv_space_outer_shell_4",
]


def parse_map(spec: str) -> Dict[int, int]:
    """
    Parse '0:0,6:135,11:255' into {0: 0, 6: 135, 11: 255}.
    Left side: GHG view/subview id.
    Right side: source camera angle in your processed files.
    """
    out: Dict[int, int] = {}
    spec = (spec or "").strip()
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid mapping item {item!r}; expected '<id>:<angle>'")
        left, right = item.split(":", 1)
        out[int(left)] = int(right)
    return out


def list_subjects(processed_root: Path, subjects_arg: str | None) -> List[str]:
    if subjects_arg:
        return [x.strip() for x in subjects_arg.split(",") if x.strip()]

    subjects = []
    for p in sorted(processed_root.iterdir()):
        if p.is_dir():
            subjects.append(p.name)
    if not subjects:
        raise FileNotFoundError(f"No subject folders found under {processed_root}")
    return subjects


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_unlink(path: Path, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def save_image_as_jpg(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    ensure_parent(dst)
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow is required: pip install pillow") from e

    if not src.exists():
        raise FileNotFoundError(src)
    img = Image.open(src).convert("RGB")
    img.save(dst, quality=95)


def save_mask_as_png(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    ensure_parent(dst)
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow is required: pip install pillow") from e

    if not src.exists():
        raise FileNotFoundError(src)
    mask = Image.open(src)
    # Keep masks simple for GHG. If the source is RGB/RGBA, convert to single-channel.
    if mask.mode not in ("1", "L"):
        mask = mask.convert("L")
    mask.save(dst)


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def load_camera(camera_root: Path, camera_prefix: str, angle: int) -> Tuple[np.ndarray, np.ndarray]:
    cam_path = camera_root / f"{camera_prefix}_{angle}.json"
    if not cam_path.exists():
        raise FileNotFoundError(f"Camera json not found: {cam_path}")

    with open(cam_path, "r", encoding="utf-8") as f:
        cam = json.load(f)

    if "K" not in cam:
        raise KeyError(f"{cam_path} has no 'K' field")
    K = np.asarray(cam["K"], dtype=np.float32)

    # Your JSON says type='w2c'. GHG wants world-to-camera [R|T].
    # So we use viewmat[:3, :4] directly when available.
    if "viewmat" in cam:
        viewmat = np.asarray(cam["viewmat"], dtype=np.float32)
        if viewmat.shape != (4, 4):
            raise ValueError(f"{cam_path}: expected viewmat shape (4, 4), got {viewmat.shape}")
        extr = viewmat[:3, :4].astype(np.float32)
    elif "R" in cam and "T" in cam:
        R = np.asarray(cam["R"], dtype=np.float32)
        T = np.asarray(cam["T"], dtype=np.float32).reshape(3, 1)
        extr = np.concatenate([R, T], axis=1).astype(np.float32)
    else:
        raise KeyError(f"{cam_path} has neither 'viewmat' nor both 'R' and 'T'")

    return K, extr


def write_camera_files(
    out_root: Path,
    sample_name: str,
    frame_id: int,
    K: np.ndarray,
    extr: np.ndarray,
    overwrite: bool,
) -> None:
    parm_dir = out_root / "parm" / sample_name
    parm_dir.mkdir(parents=True, exist_ok=True)

    base = f"{frame_id}"
    k_path = parm_dir / f"{base}_intrinsic.npy"
    e_path = parm_dir / f"{base}_extrinsic.npy"

    if overwrite or not k_path.exists():
        np.save(k_path, K)
    if overwrite or not e_path.exists():
        np.save(e_path, extr)


def copy_view(
    processed_root: Path,
    camera_root: Path,
    out_root: Path,
    subject: str,
    sample_name: str,
    frame_id: int,
    source_angle: int,
    camera_prefix: str,
    image_ext: str,
    mask_ext: str,
    mask_suffix: str,
    overwrite: bool,
) -> None:
    src_dir = processed_root / subject
    img_src = src_dir / f"{subject}_{source_angle}.{image_ext.lstrip('.')}"
    mask_src = src_dir / f"{subject}_{source_angle}{mask_suffix}.{mask_ext.lstrip('.')}"

    file_base = f"{frame_id}"
    img_dst = out_root / "img" / sample_name / f"{file_base}.jpg"
    mask_dst = out_root / "mask" / sample_name / f"{file_base}.png"

    save_image_as_jpg(img_src, img_dst, overwrite=overwrite)
    save_mask_as_png(mask_src, mask_dst, overwrite=overwrite)

    K, extr = load_camera(camera_root, camera_prefix, source_angle)
    write_camera_files(out_root, sample_name, frame_id, K, extr, overwrite=overwrite)


def create_map_dirs(out_root: Path) -> None:
    for d in MAP_DIRS:
        (out_root / d).mkdir(parents=True, exist_ok=True)


def copy_smplx_objs(
    subjects: Iterable[str],
    out_root: Path,
    smplx_obj_template: str | None,
    copy_mode: str,
    overwrite: bool,
) -> None:
    """
    Copy/symlink SMPL-X OBJ meshes into GHG's converted split.

    For your current layout, use:
        --smplx-obj-template "/home/tim/Documents/LemonCode/avatar-benchmark/data/THuman_2.0_smplx/{subject}/mesh_smplx.obj"

    Example:
        source: /home/tim/.../THuman_2.0_smplx/0000/mesh_smplx.obj
        output: datasets/THuman/val/smplx_obj/0000.obj
    """
    if not smplx_obj_template:
        print("[INFO] --smplx-obj-template not provided; skipping OBJ copy.")
        return

    obj_out_dir = out_root / "smplx_obj"
    obj_out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for subject in subjects:
        src = Path(smplx_obj_template.format(subject=subject)).expanduser()
        dst = obj_out_dir / f"{subject}.obj"
        if not src.exists():
            missing.append(str(src))
            continue
        link_or_copy(src, dst, copy_mode, overwrite=overwrite)

    if missing:
        print("[WARN] Missing SMPL-X OBJ files:")
        for m in missing[:20]:
            print(f"       {m}")
        if len(missing) > 20:
            print(f"       ... and {len(missing) - 20} more")
        raise FileNotFoundError("Some SMPL-X OBJ files were missing; see warnings above.")

    print(f"[INFO] Copied/symlinked {len(list(subjects))} SMPL-X OBJ files to {obj_out_dir}")


def patch_file_once(path: Path, replacements: List[Tuple[str, str]]) -> bool:
    if not path.exists():
        print(f"[WARN] Cannot patch missing file: {path}")
        return False

    text = path.read_text(encoding="utf-8")
    original = text

    for old, new in replacements:
        if new in text:
            continue
        if old in text:
            text = text.replace(old, new, 1)
        else:
            print(f"[WARN] Pattern not found in {path}: {old!r}")

    if text != original:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print(f"[PATCHED] {path}  backup={backup}")
        return True

    print(f"[SKIP] No changes needed for {path}")
    return False


def patch_ghg_sparse_eval(ghg_root: Path, anchor_suffix: str = "_000") -> None:
    """
    Best-effort patches:
    1. Filter eval samples to only *_000 so *_006 and *_011 are not treated as separate test samples.
    2. Make get_eval_calib use --test_data_root instead of opt.data_root/val if the old code is present.
    """
    human_loader = ghg_root / "lib" / "ghg" / "human_loader.py"
    utils_py = ghg_root / "lib" / "ghg" / "utils.py"

    patch_file_once(
        human_loader,
        [
            (
                "self.data_list = os.listdir(os.path.join(self.data_root, 'img'))",
                (
                    "self.data_list = os.listdir(os.path.join(self.data_root, 'img'))\n"
                    f"            self.data_list = sorted([x for x in self.data_list if x.endswith('{anchor_suffix}')])"
                ),
            ),
            (
                'self.data_list = os.listdir(os.path.join(self.data_root, "img"))',
                (
                    'self.data_list = os.listdir(os.path.join(self.data_root, "img"))\n'
                    f"            self.data_list = sorted([x for x in self.data_list if x.endswith('{anchor_suffix}')])"
                ),
            ),
        ],
    )

    patch_file_once(
        utils_py,
        [
            (
                "data_root = os.path.join(opt.data_root, 'val')",
                "data_root = getattr(opt, 'test_data_root', os.path.join(opt.data_root, 'val'))",
            ),
            (
                'data_root = os.path.join(opt.data_root, "val")',
                'data_root = getattr(opt, "test_data_root", os.path.join(opt.data_root, "val"))',
            ),
        ],
    )


def run_ghg_preprocess(ghg_root: Path, run_position: bool, run_visibility: bool) -> None:
    cmds = []
    if run_position:
        cmds.append([sys.executable, "process_dataset/render_position_map.py"])
    if run_visibility:
        cmds.append([sys.executable, "process_dataset/render_visibility_map.py"])

    for cmd in cmds:
        print(f"[RUN] cd {ghg_root} && {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(ghg_root), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True, help="Path to processed/, containing subject folders like 0001/")
    parser.add_argument("--camera-root", required=True, help="Path to THuman_cameras/, containing thuman_0.json etc.")
    parser.add_argument("--ghg-root", required=True, help="Path to Generalizable-Human-Gaussians repo")
    parser.add_argument("--dataset-name", default="THuman")
    parser.add_argument("--split", default="val")
    parser.add_argument("--subjects", default=None, help="Optional comma-separated subject ids, e.g. 0001,0002")
    parser.add_argument("--input-map", default="0:0,6:135,11:255",
                        help="Mapping from GHG input view id to source camera angle.")
    parser.add_argument("--target-map", default="0:0",
                        help="Mapping from target subview id under anchor folder to source angle. Use '0:0' for --novel_view_nums 1.")
    parser.add_argument("--anchor-ghg-view", type=int, default=0,
                        help="The anchor input view whose folder is used for target subviews. Usually 0.")
    parser.add_argument("--camera-prefix", default="thuman")
    parser.add_argument("--image-ext", default="png")
    parser.add_argument("--mask-ext", default="png")
    parser.add_argument("--mask-suffix", default="_mask")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument("--smplx-obj-template", default=None,
                        help=(
                            "Optional Python format string for SMPL-X OBJ. "
                            "For your layout: "
                            "'/home/tim/Documents/LemonCode/avatar-benchmark/data/THuman_2.0_smplx/{subject}/mesh_smplx.obj'"
                        ))
    parser.add_argument("--create-map-dirs", action="store_true", help="Create empty GHG position/visibility map dirs.")
    parser.add_argument("--patch-ghg", action="store_true", help="Best-effort patch GHG loader/utils for sparse eval.")
    parser.add_argument("--run-position-map", action="store_true", help="Run GHG process_dataset/render_position_map.py after conversion.")
    parser.add_argument("--run-visibility-map", action="store_true", help="Run GHG process_dataset/render_visibility_map.py after conversion.")
    args = parser.parse_args()

    processed_root = Path(args.processed_root).expanduser().resolve()
    camera_root = Path(args.camera_root).expanduser().resolve()
    ghg_root = Path(args.ghg_root).expanduser().resolve()
    out_root = ghg_root / "datasets" / args.dataset_name / args.split

    if not processed_root.exists():
        raise FileNotFoundError(processed_root)
    if not camera_root.exists():
        raise FileNotFoundError(camera_root)
    if not ghg_root.exists():
        raise FileNotFoundError(ghg_root)

    subjects = list_subjects(processed_root, args.subjects)
    input_map = parse_map(args.input_map)
    target_map = parse_map(args.target_map)

    if args.anchor_ghg_view not in input_map:
        raise ValueError(f"--anchor-ghg-view {args.anchor_ghg_view} must exist in --input-map {args.input_map}")

    print(f"[INFO] subjects: {len(subjects)}")
    print(f"[INFO] output: {out_root}")
    print(f"[INFO] input_map: {input_map}")
    print(f"[INFO] target_map: {target_map}")

    for subdir in ["img", "mask", "parm"]:
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    if args.create_map_dirs:
        create_map_dirs(out_root)

    # Create view-specific folders (e.g., 0001_000, 0001_006, 0001_011) with input view images.
    for subject in subjects:
        for ghg_view, source_angle in input_map.items():
            sample_name = f"{subject}_{ghg_view:03d}"
            copy_view(
                processed_root=processed_root,
                camera_root=camera_root,
                out_root=out_root,
                subject=subject,
                sample_name=sample_name,
                frame_id=0,
                source_angle=source_angle,
                camera_prefix=args.camera_prefix,
                image_ext=args.image_ext,
                mask_ext=args.mask_ext,
                mask_suffix=args.mask_suffix,
                overwrite=args.overwrite,
            )

        # Create target subviews only under the anchor folder, not as extra view folders.
        # For --novel_view_nums 1, target_map '0:0' is enough and usually already exists.
        anchor_sample = f"{subject}_{args.anchor_ghg_view:03d}"
        anchor_source_angle = input_map[args.anchor_ghg_view]
        for subview_id, source_angle in target_map.items():
            # Skip duplicate anchor view that is already produced by input_map.
            if subview_id == 0 and source_angle == anchor_source_angle:
                continue
            copy_view(
                processed_root=processed_root,
                camera_root=camera_root,
                out_root=out_root,
                subject=subject,
                sample_name=anchor_sample,
                frame_id=subview_id,
                source_angle=source_angle,
                camera_prefix=args.camera_prefix,
                image_ext=args.image_ext,
                mask_ext=args.mask_ext,
                mask_suffix=args.mask_suffix,
                overwrite=args.overwrite,
            )

    copy_smplx_objs(
        subjects=subjects,
        out_root=out_root,
        smplx_obj_template=args.smplx_obj_template,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )

    if args.patch_ghg:
        patch_ghg_sparse_eval(ghg_root, anchor_suffix=f"_{args.anchor_ghg_view:03d}")

    if args.run_position_map or args.run_visibility_map:
        run_ghg_preprocess(
            ghg_root=ghg_root,
            run_position=args.run_position_map,
            run_visibility=args.run_visibility_map,
        )

    print("[DONE] Conversion finished.")
    print()
    print("Suggested first GHG test command:")
    print(f"cd {ghg_root}")
    print(
        "CUDA_VISIBLE_DEVICES=0 python eval.py "
        f"--test_data_root datasets/{args.dataset_name}/{args.split} "
        "--regressor_path weights/model_gaussian.pth "
        "--inpaintor_path weights/model_inpaint.pth "
        "--novel_view_nums 1 "
        "--bg_color black"
    )


if __name__ == "__main__":
    main()
