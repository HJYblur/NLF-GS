"""
Drive saved NLF-GS Gaussian **appearance** (decoder output from ``inference``) under new SMPL-X poses.

Loads the primary inference ``.pt`` bundle (``save_pt``: linear scales, includes ``offset`` — **not**
``*_view.pt``, which strips offset). Recomputes only fused 3D centers from new vertices + template
geometry; then renders the same four canonical cameras.

Example::

    python anim.py --start-subject 0425 --end-subject 0425 --pose tpose \\
        --config configs/nlgfs_test.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent / "src"))

from src.avatar_utils.anim_replay import gaussian_params_for_render, replay_fused_gaussian_means
from src.avatar_utils.config import load_config
from src.avatar_utils.ply_loader import reconstruct_gaussian_avatar_as_ply
from src.avatar_utils.smplx_loader import load_smplx_coord3d, load_smplx_coord3d_tpose
from src.training.nlfgs_builder import (
    apply_matmul_precision_for_device,
    build_nlf_gaussian_model,
    device_from_cfg,
    gsplat_renderer_if_cuda,
)

from inference import (
    _inference_ply_filename,
    _inference_pt_filename,
    _repo_root,
    _resolve_path,
    _subjects_in_range,
)

PLY_SAVE_LOG_SCALES = True
PLY_SAVE_INCLUDE_PARENT = True


def _resolve_output_root(cfg: dict) -> Path:
    inf_cfg = cfg.get("inference", {})
    out_root = Path(str(inf_cfg.get("output_dir", cfg.get("render", {}).get("save_path", "output"))))
    if not out_root.is_absolute():
        out_root = _repo_root() / out_root
    return out_root


def _default_bundle_path(cfg: dict, subject: str) -> Path:
    inf_cfg = cfg.get("inference", {})
    ply_base = _inference_ply_filename(inf_cfg, subject)
    pt_name = _inference_pt_filename(inf_cfg, ply_base)
    return _resolve_output_root(cfg) / subject / pt_name


def _resolve_smplx_pkl(cfg: dict, subject: str) -> Path:
    smplx_root = Path(cfg.get("data", {}).get("smplx_root", "data/THuman_2.0_smplx_params"))
    if not smplx_root.is_absolute():
        smplx_root = _repo_root() / smplx_root
    return smplx_root / subject / "smplx_param.pkl"


def _load_vertices_for_pose(cfg: dict, subject: str, pose: str, pkl_override: Path | None) -> torch.Tensor:
    pkl = pkl_override if pkl_override is not None else _resolve_smplx_pkl(cfg, subject)
    if not pkl.is_file():
        raise FileNotFoundError(f"Need SMPL-X pickle for pose {pose!r}: {pkl}")
    if pose == "tpose":
        return load_smplx_coord3d_tpose(str(pkl))
    if pose == "subject":
        return load_smplx_coord3d(str(pkl))
    raise ValueError(f"Unknown --pose {pose!r}; use 'tpose' or 'subject'.")


def _template_dict_from_bundle(bundle: dict) -> dict:
    if "parent" in bundle:
        return {"parent": bundle["parent"]}
    raise ValueError(
        "Bundle has no 'parent' key — export inference with the same template so parent indices are saved."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Replay Gaussian appearance from inference .pt under new SMPL-X pose (geometry only)."
    )
    parser.add_argument("--start-subject", type=str, required=True)
    parser.add_argument("--end-subject", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/nlgfs_test.yaml")
    parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Path to inference .pt (default: output/<subject>/<stem>.pt from config naming).",
    )
    parser.add_argument(
        "--pose",
        type=str,
        choices=("tpose", "subject"),
        default="tpose",
        help="tpose: zero body/hand global pose from pkl; subject: full pose from smplx_param.pkl.",
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default=None,
        help="Override path to smplx_param.pkl (default: smplx_root/<subject>/smplx_param.pkl).",
    )
    parser.add_argument(
        "--views-subdir",
        type=str,
        default="canonical_views_anim",
        help="Subfolder under subject output for PNGs.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="anim",
        help="PNG prefix: {save-prefix}_front.png, etc.",
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="Also write anim_{subject}.ply (means fused, gaussian_params without offset).",
    )
    args = parser.parse_args()

    os.environ["NLFGS_CONFIG"] = args.config
    cfg = load_config(args.config)
    device = device_from_cfg(cfg)
    apply_matmul_precision_for_device(cfg, device)

    pkl_override = Path(args.pkl).resolve() if args.pkl else None

    subjects = _subjects_in_range(cfg, args.start_subject, args.end_subject)
    renderer = gsplat_renderer_if_cuda(device)

    model = build_nlf_gaussian_model(cfg, device)
    model.eval()
    estimator = model.avatar_estimator
    out_root = _resolve_output_root(cfg)

    for subject in subjects:
        bundle_path = Path(args.bundle) if args.bundle else _default_bundle_path(cfg, subject)
        bundle_path = _resolve_path(bundle_path) if not bundle_path.is_absolute() else bundle_path
        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"No inference bundle at {bundle_path}. Run inference with inference.save_pt: true, "
                f"or pass --bundle to the primary .pt (not *_view.pt — those omit offset)."
            )

        try:
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(bundle_path, map_location="cpu")

        if "gaussian_params" not in bundle:
            raise KeyError(f"Bundle {bundle_path} has no 'gaussian_params'.")
        gaussian_params = bundle["gaussian_params"]
        if not isinstance(gaussian_params, dict):
            raise TypeError("bundle['gaussian_params'] must be a dict of tensors.")

        vertices_new = _load_vertices_for_pose(cfg, subject, args.pose, pkl_override)

        fused = replay_fused_gaussian_means(
            estimator,
            vertices_new,
            gaussian_params,
            device=device,
        )
        gp_render = gaussian_params_for_render(gaussian_params, device=device)

        sub_dir = out_root / subject
        sub_dir.mkdir(parents=True, exist_ok=True)

        if args.save_ply:
            tpl = _template_dict_from_bundle(bundle)
            ply_path = sub_dir / f"anim_{subject}.ply"
            gp_ply = {k: v.cpu() for k, v in gp_render.items()}
            reconstruct_gaussian_avatar_as_ply(
                fused.detach().cpu(),
                gp_ply,
                tpl,
                str(ply_path),
                log_scales=PLY_SAVE_LOG_SCALES,
                include_parent=PLY_SAVE_INCLUDE_PARENT,
            )
            print(f"[{subject}] Saved PLY {ply_path.resolve()}")

        if device.type != "cuda":
            print(f"[{subject}] Skipping PNG renders (CUDA required). Fused means shape {tuple(fused.shape)}.")
            continue

        assert renderer is not None
        views_dir = sub_dir / args.views_subdir
        renderer.render_canonical_views(
            fused,
            gp_render,
            views_dir,
            save_prefix=args.save_prefix,
        )
        print(
            f"[{subject}] Rendered {args.save_prefix}_*.png from {bundle_path.name} "
            f"under --pose {args.pose} → {views_dir.resolve()}"
        )


if __name__ == "__main__":
    main()
