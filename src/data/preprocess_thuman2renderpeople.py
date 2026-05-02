#!/usr/bin/env python3
"""Convert NLF-GS ``processed/`` THuman layout to a SHERF-style RenderPeople tree.

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

This script writes that ``.npz`` layout (not raw ``smpl_param.pkl``).

SMPL source layout
------------------
The THuman preprocessing step now exports a per-subject ``smpl_param.pkl``
alongside the rendered views. This script consumes that file directly and only
repackages it into SHERF's ``refit_smpl_2nd.npz`` format.

Source layout (see README):
  processed/<id>/
    <id>_front.png, ... _mask.png
        smplx_param.pkl
        smpl_param.pkl

Cameras (intrinsics / extrinsics) are read from ``data/THuman_cameras/thuman_<view>.json``
(same JSON as :func:`avatar_utils.camera.load_camera_mapping`).

Target layout::

  <out>/RenderPeople_recon/<date>/
    human_list.txt
    seq_<6d>-thuman_<id>/
      cameras.json
      img/camera0000/0000.jpg ... (num_pose_frames files per camera)
      mask/camera0000/0000.png ...
      outputs_re_fitting/refit_smpl_2nd.npz

Notes
-----
* THuman preprocessing uses **4** views; SHERF RenderPeople defaults to **36**
  cameras. Use ``--pad-cameras 36`` to repeat each real view across 9 slots
  (same ``K,R,T`` and duplicated RGB/mask) so stock ``camera_view_num=36`` runs
  without editing SHERF. Otherwise keep 4 cameras and set
  ``camera_view_num=4`` in the SHERF dataset / shell script.
* Poses are **static** here: the same RGB/mask is copied for every frame index
  ``0000`` … ``num_pose_frames-1``, and SMPL arrays are repeated along the
  time axis (SHERF indexes by ``pose_index``).
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

VIEW_ORDER = ("front", "back", "left", "right")


def _scale_translation_from_pkl(params: dict) -> tuple[float, np.ndarray]:
    scale = float(np.asarray(params.get("scale", 1.0), dtype=np.float64).reshape(-1)[0])
    tr = np.asarray(params.get("translation", np.zeros(3)), dtype=np.float64).reshape(-1)
    if tr.size != 3:
        tr = np.pad(tr[: min(tr.size, 3)], (0, max(0, 3 - tr.size)), mode="constant")[:3]
    return scale, tr.astype(np.float32)


def _smpl_pkl_to_sherf_smpl_dict(params: dict, num_frames: int) -> dict:
    """Convert THuman ``smpl_param.pkl`` into SHERF's repeated-frame SMPL dict."""
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


def _fit_smpl_params_with_smplx(
    params: dict,
    smpl_model_path: Path,
    smplx_model_path: Path,
    *,
    gender: str = "neutral",
    num_pca_comps: int = 12,
    num_steps: int = 300,
    lr: float = 0.05,
) -> dict | None:
    """Fit ``smplx.SMPL`` pose/trans to ``smplx.SMPLX`` joints (first 22), using ``smplx`` + ``torch``.

    Matches the joint frame implied by ``avatar_utils.smplx_loader`` (forward without
    ``transl``, then ``joints * scale + translation``). SMPL and SMPL-X share the same
    kinematic names for indices 0–21 (pelvis through wrists); SMPL hand joints 22–23
    are not in the loss.
    """
    try:
        import torch
        import smplx as _smplx
    except ImportError:
        return None

    if not smpl_model_path.exists():
        return None

    scale, trans_np = _scale_translation_from_pkl(params)
    device = torch.device("cpu")
    dtype = torch.float32

    def _t(name: str, shape_tail: tuple[int, ...], default: float = 0.0) -> torch.Tensor:
        if name not in params:
            return torch.zeros((1,) + shape_tail, device=device, dtype=dtype)
        a = np.asarray(params[name], dtype=np.float32)
        a = np.reshape(a, (1, -1))
        expected = int(np.prod(shape_tail))
        if a.shape[1] != expected:
            if a.shape[1] > expected:
                a = a[:, :expected]
            else:
                a = np.pad(a, ((0, 0), (0, expected - a.shape[1])), mode="constant")
        return torch.from_numpy(a.reshape((1,) + shape_tail)).to(device=device, dtype=dtype)

    betas_x = _t("betas", (10,), 0.0)
    if betas_x.shape[1] < 10:
        betas_x = torch.nn.functional.pad(betas_x, (0, 10 - betas_x.shape[1]))
    elif betas_x.shape[1] > 10:
        betas_x = betas_x[:, :10]

    go_x = _t("global_orient", (3,))
    bp_x = _t("body_pose", (63,))
    lhp = _t("left_hand_pose", (num_pca_comps,))
    rhp = _t("right_hand_pose", (num_pca_comps,))
    jaw = _t("jaw_pose", (3,))
    leye = _t("leye_pose", (3,))
    reye = _t("reye_pose", (3,))

    try:
        smplx_m = _smplx.SMPLX(
            model_path=str(smplx_model_path),
            gender=gender,
            use_pca=True,
            num_pca_comps=num_pca_comps,
            flat_hand_mean=True,
            batch_size=1,
        ).to(device=device, dtype=dtype)
        smpl_dir = smpl_model_path if smpl_model_path.is_dir() else smpl_model_path.parent
        smpl_m = _smplx.SMPL(model_path=str(smpl_dir), gender=gender, batch_size=1).to(
            device=device, dtype=dtype
        )
    except Exception:
        return None

    nec = int(getattr(smplx_m, "num_expression_coeffs", 10))
    expr_np = np.asarray(params.get("expression", np.zeros(nec, dtype=np.float32)), dtype=np.float32).ravel()
    if expr_np.size < nec:
        expr_np = np.pad(expr_np, (0, nec - expr_np.size), mode="constant")
    else:
        expr_np = expr_np[:nec]
    expr = torch.from_numpy(expr_np.reshape(1, -1)).to(device=device, dtype=dtype)

    smplx_m.eval()
    smpl_m.eval()

    with torch.no_grad():
        out_x = smplx_m(
            betas=betas_x,
            global_orient=go_x,
            body_pose=bp_x,
            left_hand_pose=lhp,
            right_hand_pose=rhp,
            jaw_pose=jaw,
            leye_pose=leye,
            reye_pose=reye,
            expression=expr,
            return_verts=False,
        )
        target_j = out_x.joints[:, :22, :].clone() * float(scale) + torch.from_numpy(trans_np).to(
            device=device, dtype=dtype
        ).view(1, 1, 3)

    betas_s = betas_x.detach()

    go = torch.nn.Parameter(go_x.clone())
    bp_flat = bp_x.reshape(1, -1)
    if bp_flat.shape[1] < 69:
        bp_flat = torch.nn.functional.pad(bp_flat, (0, 69 - bp_flat.shape[1]))
    else:
        bp_flat = bp_flat[:, :69]
    bp = torch.nn.Parameter(bp_flat.clone())

    tr_init = np.asarray(params.get("transl", params.get("translation", np.zeros(3))), dtype=np.float32).reshape(-1)
    if tr_init.size < 3:
        tr_init = np.pad(tr_init, (0, 3 - tr_init.size), mode="constant")
    tr = torch.nn.Parameter(torch.from_numpy(tr_init[:3]).view(1, 3).to(device=device, dtype=dtype))

    opt = torch.optim.Adam([go, bp, tr], lr=lr)
    for _ in range(num_steps):
        out_s = smpl_m(
            betas=betas_s,
            global_orient=go,
            body_pose=bp,
            transl=tr,
            return_verts=False,
        )
        pred = out_s.joints[:, :22, :]
        loss = torch.mean((pred - target_j) ** 2)
        if not torch.isfinite(loss):
            return None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        go_np = go.detach().cpu().numpy().astype(np.float32).reshape(3)
        bp_np = bp.detach().cpu().numpy().astype(np.float32).reshape(69)
        tr_np = tr.detach().cpu().numpy().astype(np.float32).reshape(3)

    return {
        "betas": betas_s.detach().cpu().numpy().astype(np.float32).reshape(10),
        "global_orient": np.tile(go_np[None, :], (1, 1)),  # caller expands frames
        "body_pose": np.tile(bp_np[None, :], (1, 1)),
        "transl": np.tile(tr_np[None, :], (1, 1)),
    }


def _smpl_pkl_to_sherf_smpl_dict(
    params: dict,
    num_frames: int,
    *,
    smpl_model_path: Path | None,
    smplx_model_path: Path,
    smpl_fit_steps: int,
    smpl_fit_lr: float,
    gender: str,
    num_pca_comps: int,
    verbose: bool,
) -> dict:
    """Build ``smpl`` dict for ``refit_smpl_2nd.npz`` from ``smpl_param.pkl`` directly."""
    if verbose and (smpl_fit_steps > 0 or smpl_model_path is not None):
        print("  SMPL params: using smpl_param.pkl directly; SMPL-X fitting is disabled.")

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


def _default_smpl_model_path(project_root: Path) -> Path | None:
    for rel in (
        Path("models") / "smpl" / "SMPL_NEUTRAL.pkl",
        Path("assets") / "SMPL_NEUTRAL.pkl",
    ):
        p = project_root / rel
        if p.is_file():
            return p
    return None


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
    smpl_model_path: Path | None,
    smplx_model_path: Path,
    smpl_fit_steps: int,
    smpl_fit_lr: float,
    smpl_gender: str,
    smplx_num_pca_comps: int,
    verbose: bool,
) -> None:
    src = processed_root / subject_id
    if not src.is_dir():
        raise FileNotFoundError(f"Missing processed subject dir: {src}")

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
        smplx_params = pickle.load(f)
    smpl_dict = _smpl_pkl_to_sherf_smpl_dict(
        smplx_params,
        num_pose_frames,
        smpl_model_path=smpl_model_path,
        smplx_model_path=smplx_model_path,
        smpl_fit_steps=smpl_fit_steps,
        smpl_fit_lr=smpl_fit_lr,
        gender=smpl_gender,
        num_pca_comps=smplx_num_pca_comps,
        verbose=verbose,
    )
    _save_sherf_smpl_npz(fit_root / "refit_smpl_2nd.npz", smpl_dict)


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
        description="Convert processed THuman folders to SHERF RenderPeople-style layout."
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
        help="Directory with thuman_<view>.json (default: <project>/data/THuman_cameras).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Destination root; creates RenderPeople_recon/<date>/… under this path (default: current directory).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="20230228",
        help="Subfolder name under RenderPeople_recon/ (default matches SHERF examples).",
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
        help="If set (e.g. 36), repeat K,R,T and images to this many cameras; must be a multiple of 4.",
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
        "--smplx-model-path",
        type=Path,
        default=project_root / "models" / "smplx",
        help="Directory with SMPLX_NEUTRAL.pkl (same as NLF-GS training).",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=Path,
        default=_default_smpl_model_path(project_root),
        help="Path to SMPL_NEUTRAL.pkl (or .npz) or its parent directory. "
        "Default: auto-detect under models/smpl/ or assets/.",
    )
    parser.add_argument(
        "--smpl-fit-steps",
        type=int,
        default=300,
        help="Adam steps for SMPL joint fit (0 = heuristic SMPL-X→SMPL only).",
    )
    parser.add_argument(
        "--smpl-fit-lr",
        type=float,
        default=0.05,
        help="Learning rate for SMPL fit.",
    )
    parser.add_argument(
        "--smpl-gender",
        type=str,
        default="neutral",
        help="Gender string for smplx.SMPL / SMPLX constructors.",
    )
    parser.add_argument(
        "--smplx-num-pca-comps",
        type=int,
        default=12,
        help="Hand PCA components (must match preprocessing / smplx_loader).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-subject SMPL conversion messages.",
    )
    args = parser.parse_args()

    if (
        args.smpl_fit_steps > 0
        and args.smpl_model_path is not None
        and not args.smplx_model_path.exists()
    ):
        raise SystemExit(f"--smplx-model-path does not exist: {args.smplx_model_path}")

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

    recon_root = args.output_root / "RenderPeople_recon" / args.date
    recon_root.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet
    lines: list[str] = []
    for sid in subject_ids:
        try:
            n = int(sid)
        except ValueError:
            n = 0
        seq_name = f"{args.seq_prefix}_{n:06d}-thuman_{sid}"
        seq_dir = recon_root / seq_name
        print(f"Exporting {sid} -> {seq_dir}")
        _export_one_subject(
            sid,
            args.processed_root,
            args.camera_dir,
            seq_dir,
            VIEW_ORDER,
            args.num_pose_frames,
            args.pad_cameras,
            args.jpeg_quality,
            smpl_model_path=args.smpl_model_path,
            smplx_model_path=args.smplx_model_path,
            smpl_fit_steps=args.smpl_fit_steps,
            smpl_fit_lr=args.smpl_fit_lr,
            smpl_gender=args.smpl_gender,
            smplx_num_pca_comps=args.smplx_num_pca_comps,
            verbose=verbose,
        )
        lines.append(seq_name)

    list_path = recon_root / "human_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} entries to {list_path}")


if __name__ == "__main__":
    main()
