#!/usr/bin/env python3
"""Convert NLF-GS ``processed/`` THuman layout to a SHERF-compatible multi-view tree.

SHERF loader contract (``sherf/training/RenderPeople_dataset.py``)
----------------------------------------------------------------
These paths and dtypes must match what the loader builds:

* ``img/camera{view:04d}/{pose:04d}.jpg`` — RGB JPEG, ``pose`` from
  ``poses_start + k * poses_interval`` (default data uses ``0000`` …).
* ``mask/camera{view:04d}/{pose:04d}.png`` — single-channel mask; loader
  binarizes non-zero to 255.
* ``outputs_re_fitting/refit_smpl_2nd.npz`` — compressed archive with object
  array key ``smpl`` such that ``np.load(...)['smpl'].item()`` is a ``dict``
  with the keys read in ``prepare_smpl_params``:

  - ``betas`` — shape ``(10,)`` float32 (passed to SMPL as ``shapes``).
  - ``global_orient`` — ``(num_poses, 3)``; indexed per frame.
  - ``body_pose`` — ``(num_poses, 69)`` (23 joints × axis-angle).
  - ``transl`` — ``(num_poses, 3)``.

This script reads ``smpl_param.pkl`` and writes the above ``.npz`` layout for SHERF.

SMPL source layout
------------------
The THuman preprocessing step now exports a per-subject ``smpl_param.pkl``
alongside the rendered views. This script consumes that file directly and only
repackages it into SHERF's ``refit_smpl_2nd.npz`` format.

Source layout (matches ``preprocess_thuman.py``):

  processed/<id>/
    <id>_<azimuth>.png   # e.g. ``0001_0.png``, ``0001_15.png``, … (default 24 orbit views)
    <id>_<azimuth>_mask.png
    smpl_param.pkl

Cameras are read from ``data/THuman_cameras/thuman_<azimuth>.json`` (e.g. ``thuman_0.json``),
the same files ``preprocess_thuman.generate_camera_mapping`` writes.

Target layout::

  <output-root>/thuman2.0_24views/
    human_list.txt
    seq_<6d>-thuman_<id>/
      cameras.json
      img/camera0000/0000.jpg ... (num_pose_frames files per camera)
      mask/camera0000/0000.png ...
      outputs_re_fitting/refit_smpl_2nd.npz

Use ``--output-subdir`` to change ``thuman2.0_24views`` (e.g. if you export cardinal mode only).

Notes
-----
* Default export uses the full orbit from ``avatar_utils.view_config`` (24 views at
  15° steps, labels ``"0"`` … ``"345"``). Use ``--views-mode cardinal`` for legacy
  four-view folders (``0``, ``90``, ``180``, ``270`` only).
* SHERF RenderPeople defaults to **36** cameras. ``--pad-cameras N`` repeats each
  exported view evenly; ``N`` must be a multiple of the number of views (24 or 4).
  Example: ``--pad-cameras 72`` with orbit mode tiles each real camera 3× to fill
  72 slots, or set ``camera_view_num=24`` in SHERF and skip padding.
* Poses are **static**: the same RGB/mask is written for every pose index ``0000`` …,
  and SMPL arrays are tiled along the time axis.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Make ``avatar_utils`` importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avatar_utils.config import get_config  # noqa: E402
from avatar_utils.view_config import MODEL_INPUT_VIEW_ORDER, VIEW_ORDER as ORBIT_VIEW_ORDER  # noqa: E402


def _views_tuple(mode: str) -> tuple[str, ...]:
    """Resolve which processed view suffixes to export (must match ``preprocess_thuman`` / THuman_cameras)."""
    m = str(mode).strip().lower()
    if m in ("orbit", "full", "24", "all"):
        return tuple(ORBIT_VIEW_ORDER)
    if m in ("cardinal", "four", "4", "model"):
        return tuple(MODEL_INPUT_VIEW_ORDER)
    raise ValueError(
        f"views_mode must be 'orbit' (default {len(ORBIT_VIEW_ORDER)} azimuth steps) or "
        f"'cardinal' (four views {list(MODEL_INPUT_VIEW_ORDER)}), got {mode!r}"
    )


def _smpl_pkl_to_sherf_smpl_dict(params: dict, num_frames: int, *, verbose: bool = False) -> dict:
    """Build ``smpl`` dict for ``refit_smpl_2nd.npz`` from ``smpl_param.pkl`` (SHERF ``prepare_smpl_params`` layout)."""
    if verbose:
        print("  SMPL params: from smpl_param.pkl → refit_smpl_2nd.npz")

    betas = np.asarray(params.get("betas", np.zeros(10, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if betas.size < 10:
        betas = np.pad(betas, (0, 10 - betas.size), mode="constant")
    betas = betas[:10]

    go = np.asarray(params.get("global_orient", np.zeros(3)), dtype=np.float32).reshape(-1)
    if go.size < 3:
        go = np.pad(go, (0, 3 - go.size), mode="constant")
    go = go[:3]

    body = np.asarray(params.get("body_pose", np.zeros(69)), dtype=np.float32).reshape(-1)
    if body.size >= 69:
        body_smpl = body[:69]
    elif body.size == 63:
        body_smpl = np.concatenate([body, np.zeros(6, dtype=np.float32)], axis=0)
    else:
        body_smpl = np.pad(body, (0, 69 - body.size), mode="constant")[:69]

    if "transl" in params:
        tr = np.asarray(params["transl"], dtype=np.float32).reshape(-1)
    else:
        tr = np.asarray(params.get("translation", np.zeros(3)), dtype=np.float32).reshape(-1)
    if tr.size < 3:
        tr = np.pad(tr, (0, 3 - tr.size), mode="constant")
    tr = tr[:3]

    return {
        "betas": betas,
        "global_orient": np.tile(go[None, :], (num_frames, 1)),
        "body_pose": np.tile(body_smpl[None, :], (num_frames, 1)),
        "transl": np.tile(tr[None, :], (num_frames, 1)),
    }


def _default_paths() -> tuple[Path, Path]:
    cfg = get_config()
    data_cfg = cfg.get("data", {})
    processed = Path(data_cfg.get("processed_root", "processed"))
    project_root = Path(__file__).resolve().parents[2]
    camera_dir = project_root / "data" / "THuman_cameras"
    return processed, camera_dir


def _load_thuman_camera_json(camera_dir: Path, view: str) -> dict:
    path = camera_dir / f"thuman_{view}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing camera JSON {path}. Run ``python src/data/preprocess_thuman.py`` "
            "once (or call generate_camera_mapping) to create THuman_cameras."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_cameras_json(camera_dir: Path, views: tuple[str, ...]) -> dict:
    """SHERF expects top-level keys ``camera0000`` … with ``K``, ``R``, ``T``."""
    out: dict = {}
    for i, view in enumerate(views):
        payload = _load_thuman_camera_json(camera_dir, view)
        key = f"camera{str(i).zfill(4)}"
        k = np.asarray(payload["K"], dtype=np.float64)
        r = np.asarray(payload["R"], dtype=np.float64)
        t = np.asarray(payload["T"], dtype=np.float64).reshape(3)
        out[key] = {
            "K": k.tolist(),
            "R": r.tolist(),
            "T": t.tolist(),
        }
    return out


def _pad_cameras_block(
    base: dict,
    num_target: int,
    num_real: int,
) -> dict:
    """Repeat real cameras evenly until ``num_target`` keys exist."""
    if num_target < num_real:
        raise ValueError(f"pad target {num_target} must be >= real cameras {num_real}")
    if num_target == num_real:
        return base
    if num_target % num_real != 0:
        raise ValueError(
            f"--pad-cameras {num_target} must be a multiple of real view count {num_real}"
        )
    reps = num_target // num_real
    out: dict = {}
    idx = 0
    for _ in range(reps):
        for j in range(num_real):
            src_key = f"camera{str(j).zfill(4)}"
            dst_key = f"camera{str(idx).zfill(4)}"
            out[dst_key] = json.loads(json.dumps(base[src_key]))  # deep copy plain dict
            idx += 1
    return out


def _save_sherf_smpl_npz(out_path: Path, smpl_inner: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = np.empty((), dtype=object)
    blob[()] = smpl_inner
    np.savez_compressed(out_path, smpl=blob)


def _export_one_subject(
    subject_id: str,
    processed_root: Path,
    camera_dir: Path,
    seq_dir: Path,
    views: tuple[str, ...],
    num_pose_frames: int,
    pad_cameras: int | None,
    jpeg_quality: int,
    *,
    verbose: bool,
) -> None:
    src = processed_root / subject_id
    if not src.is_dir():
        print("  Skipping missing subject folder:", src)
        return False

    seq_dir.mkdir(parents=True, exist_ok=True)
    img_root = seq_dir / "img"
    mask_root = seq_dir / "mask"
    fit_root = seq_dir / "outputs_re_fitting"
    fit_root.mkdir(parents=True, exist_ok=True)

    cameras = _build_cameras_json(camera_dir, views)
    n_real = len(views)
    n_cam = pad_cameras if pad_cameras is not None else n_real
    cameras = _pad_cameras_block(cameras, n_cam, n_real)

    with open(seq_dir / "cameras.json", "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)

    reps = n_cam // n_real
    view_for_slot = []
    for _ in range(reps):
        for v in views:
            view_for_slot.append(v)

    for cam_idx in range(n_cam):
        view = view_for_slot[cam_idx]
        cdir = f"camera{str(cam_idx).zfill(4)}"
        (img_root / cdir).mkdir(parents=True, exist_ok=True)
        (mask_root / cdir).mkdir(parents=True, exist_ok=True)

        color_path = src / f"{subject_id}_{view}.png"
        if not color_path.exists():
            color_path = src / f"{subject_id}_{view}.jpg"
        if not color_path.exists():
            raise FileNotFoundError(f"Missing RGB for view {view}: {color_path}")

        mask_path = src / f"{subject_id}_{view}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for view {view}: {mask_path}")

        rgb = Image.open(color_path).convert("RGB")
        msk = Image.open(mask_path).convert("L")

        for fi in range(num_pose_frames):
            stem = str(fi).zfill(4)
            rgb.save(img_root / cdir / f"{stem}.jpg", quality=jpeg_quality, optimize=True)
            msk.save(mask_root / cdir / f"{stem}.png")

    pkl_path = src / "smpl_param.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}")
    with open(pkl_path, "rb") as f:
        smpl_params = pickle.load(f)
    smpl_dict = _smpl_pkl_to_sherf_smpl_dict(smpl_params, num_pose_frames, verbose=verbose)
    _save_sherf_smpl_npz(fit_root / "refit_smpl_2nd.npz", smpl_dict)
    return True


def _parse_subject_range(
    processed_root: Path,
    start: str | None,
    end: str | None,
) -> list[str]:
    ids: list[str] = []
    start_i = int(start) if start is not None else None
    end_i = int(end) if end is not None else None
    for p in sorted(processed_root.iterdir()):
        if not p.is_dir():
            continue
        try:
            sid = int(p.name)
        except ValueError:
            continue
        if start_i is not None and sid < start_i:
            continue
        if end_i is not None and sid > end_i:
            continue
        ids.append(p.name)
    return ids


def main() -> None:
    default_proc, default_cam = _default_paths()
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Convert processed THuman folders to a SHERF-style dataset layout."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=default_proc,
        help="Root with per-subject processed folders (default: config data.processed_root).",
    )
    parser.add_argument(
        "--camera-dir",
        type=Path,
        default=default_cam,
        help="Directory with thuman_<azimuth>.json (default: <project>/data/THuman_cameras).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Destination root; sequences and human_list.txt go under "
        "<output-root>/<output-subdir>/ (default: current directory).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="thuman2.0_24views",
        help="Folder name under --output-root (default: thuman2.0_24views).",
    )
    parser.add_argument(
        "--start-subject",
        type=str,
        default=None,
        help="Numeric subject id lower bound (inclusive), e.g. 0001.",
    )
    parser.add_argument(
        "--end-subject",
        type=str,
        default=None,
        help="Numeric subject id upper bound (inclusive).",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject ids to export (overrides --subject-list and start/end range).",
    )
    parser.add_argument(
        "--subject-list",
        type=Path,
        default="data/split_val.txt",
        help="Path to file with subject ids (one per line). Default: data/split_val.txt.",
    )
    parser.add_argument(
        "--num-pose-frames",
        type=int,
        default=1,
        help="Number of frame indices per camera (0000 …). Static data repeats the same image.",
    )
    parser.add_argument(
        "--pad-cameras",
        type=int,
        default=None,
        help="Repeat K,R,T and images to N cameras; N must be a multiple of the "
        "exported view count (24 in orbit mode, 4 in cardinal). E.g. 72 = 3×24 for SHERF.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for exported RGB (default 95).",
    )
    parser.add_argument(
        "--seq-prefix",
        type=str,
        default="seq",
        help="Sequence folder prefix (default: seq).",
    )
    parser.add_argument(
        "--views-mode",
        type=str,
        choices=("orbit", "cardinal"),
        default="orbit",
        help="orbit: 24 azimuth views (0,15,…,345) matching preprocess_thuman; "
        "cardinal: four views 0/90/180/270 only (legacy processed folders).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-subject messages.",
    )
    args = parser.parse_args()

    export_views = _views_tuple(args.views_mode)

    if args.subjects:
        subject_ids = [s.strip() for s in args.subjects.split(",") if s.strip()]
    elif args.subject_list.exists():
        with open(args.subject_list, "r", encoding="utf-8") as f:
            subject_ids = [s.strip() for s in f.readlines() if s.strip()]
    else:
        subject_ids = _parse_subject_range(
            args.processed_root, args.start_subject, args.end_subject
        )

    if not subject_ids:
        raise SystemExit("No subjects found to export (check --processed-root and filters).")

    out_root = (args.output_root / args.output_subdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet
    lines: list[str] = []
    for sid in subject_ids:
        try:
            n = int(sid)
        except ValueError:
            n = 0
        seq_name = f"{args.seq_prefix}_{n:06d}-thuman_{sid}"
        seq_dir = out_root / seq_name
        print(f"Exporting {sid} -> {seq_dir}")
        sucess_export = _export_one_subject(
            sid,
            args.processed_root,
            args.camera_dir,
            seq_dir,
            export_views,
            args.num_pose_frames,
            args.pad_cameras,
            args.jpeg_quality,
            verbose=verbose,
        )
        lines.append(seq_name) if sucess_export else print(f"  Skipped {sid} due to missing data.")

    list_path = out_root / "human_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} entries to {list_path}")


if __name__ == "__main__":
    main()
