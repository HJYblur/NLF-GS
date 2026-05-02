"""
Drive saved NLF-GS Gaussian **appearance** (inference ``.pt``) under new SMPL-X poses (geometry only).

**Config** (``animation`` in YAML, e.g. ``configs/nlfgs_gpu.yaml``):

* ``pose``: ``reconstruction`` (pkl as stored) | ``tpose`` (body pose rest) | ``custom`` (``custom_pose_path`` — motion only: pose joints from file, shape/scale/translation from subject ``smplx_param.pkl``)
* ``display_mode``: ``image`` — canonical orbit views into ``reconstruction_subdir``; ``video`` — spin, ``{prefix}_{subject}_{pose}.mp4`` in ``video_subdir``
* ``reconstruction_subdir`` (else ``inference.reconstruction_subdir``; default ``reconstruction``)
* ``fps`` / ``duration_seconds`` (legacy: ``frame`` as fps, ``duration`` as seconds)
* ``video_subdir`` (else ``inference.video_subdir``; default ``anim_video``)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy
import torch

sys.path.append(str(Path(__file__).parent / "src"))

from src.avatar_utils.anim_replay import gaussian_params_for_render, replay_fused_gaussian_means
from src.avatar_utils.config import load_config
from src.avatar_utils.ply_loader import reconstruct_gaussian_avatar_as_ply
from src.avatar_utils.smplx_loader import (
    copy_smplx_params_spin_global_yaw,
    copy_smplx_params_tpose_rest,
    frame_count_for_duration_seconds,
    load_smplx_params_dict,
    load_smplx_params_from_path,
    merge_subject_identity_with_driver_pose,
    vertices_from_smplx_param_dict,
)
from src.training.nlfgs_builder import (
    apply_matmul_precision_for_device,
    build_nlf_gaussian_model,
    device_from_cfg,
    gsplat_renderer_if_cuda,
)

from inference import (
    _inference_pt_filename,
    _repo_root,
    _resolve_path,
    _subjects_in_range,
)

PLY_SAVE_LOG_SCALES = True
PLY_SAVE_INCLUDE_PARENT = True

POSE_CHOICES = frozenset({"reconstruction", "tpose", "custom"})
DISPLAY_CHOICES = frozenset({"image", "video"})


def _anim_section(cfg: dict) -> dict:
    a = cfg.get("animation")
    return a if isinstance(a, dict) else {}


def _animation_video_subdir(cfg: dict) -> str:
    anim = _anim_section(cfg)
    if anim.get("video_subdir"):
        return str(anim["video_subdir"])
    inf = cfg.get("inference") or {}
    if inf.get("video_subdir"):
        return str(inf["video_subdir"])
    return "anim_video"


def _reconstruction_subdir(cfg: dict) -> str:
    anim = _anim_section(cfg)
    if anim.get("reconstruction_subdir"):
        return str(anim["reconstruction_subdir"])
    inf = cfg.get("inference") or {}
    if inf.get("reconstruction_subdir"):
        return str(inf["reconstruction_subdir"])
    if inf.get("canonical_views_subdir"):
        return str(inf["canonical_views_subdir"])
    return "reconstruction"


def _animation_pose_display_mode(
    cfg: dict, pose_arg: str | None, display_arg: str | None
) -> tuple[str, str]:
    anim = _anim_section(cfg)
    pose = (pose_arg or anim.get("pose") or "reconstruction")
    if isinstance(pose, str):
        pose = pose.strip().lower()
    if pose not in POSE_CHOICES:
        raise ValueError(
            f"pose must be one of {sorted(POSE_CHOICES)} (got {pose!r}; set in animation.pose or --pose)"
        )
    display_mode = (display_arg or anim.get("display_mode") or "image")
    if isinstance(display_mode, str):
        display_mode = display_mode.strip().lower()
    if display_mode not in DISPLAY_CHOICES:
        raise ValueError(
            f"display_mode must be one of {sorted(DISPLAY_CHOICES)} (got {display_mode!r})"
        )
    return pose, display_mode


def _animation_fps_duration(cfg: dict) -> tuple[float, float]:
    anim = _anim_section(cfg)
    fps = anim.get("fps")
    if fps is None:
        fps = anim.get("frame")
    if fps is None:
        fps = 30.0
    duration = anim.get("duration_seconds")
    if duration is None:
        duration = anim.get("duration")
    if duration is None:
        duration = 2.0
    return float(fps), float(duration)


def _resolve_output_root(cfg: dict) -> Path:
    inf_cfg = cfg.get("inference", {})
    out_root = Path(str(inf_cfg.get("output_dir", cfg.get("render", {}).get("save_path", "output"))))
    if not out_root.is_absolute():
        out_root = _repo_root() / out_root
    return out_root


def _default_bundle_path(cfg: dict, subject: str) -> Path:
    inf_cfg = cfg.get("inference", {})
    pt_name = _inference_pt_filename(inf_cfg, subject)
    return _resolve_output_root(cfg) / subject / pt_name


def _resolve_smplx_pkl(cfg: dict, subject: str) -> Path:
    data_cfg = cfg.get("data", {})
    processed_root = Path(data_cfg.get("processed_root", "processed"))
    if not processed_root.is_absolute():
        processed_root = _repo_root() / processed_root
    return processed_root / subject / "smplx_param.pkl"


def _base_smplx_params(cfg: dict, subject: str, pose: str, pkl_override: Path | None) -> dict:
    """Static SMPL-X parameter dict before optional per-frame yaw (``display_mode: video``)."""
    anim = _anim_section(cfg)

    if pose == "custom":
        rel = anim.get("custom_pose_path")
        if not rel:
            raise ValueError("animation.custom_pose_path is required when animation.pose is 'custom'")
        motion_path = _resolve_path(str(rel))
        if not motion_path.is_file():
            raise FileNotFoundError(f"custom pose file not found: {motion_path}")
        driver = load_smplx_params_from_path(str(motion_path))
        subj_pkl = pkl_override if pkl_override is not None else _resolve_smplx_pkl(cfg, subject)
        if not subj_pkl.is_file():
            raise FileNotFoundError(
                f"Need subject SMPL-X pickle for identity (betas/scale/translation): {subj_pkl}"
            )
        subject_params = load_smplx_params_dict(str(subj_pkl))
        return merge_subject_identity_with_driver_pose(subject_params, driver)

    pkl = pkl_override if pkl_override is not None else _resolve_smplx_pkl(cfg, subject)
    if not pkl.is_file():
        raise FileNotFoundError(f"Need SMPL-X pickle for subject {subject!r}: {pkl}")

    params = load_smplx_params_dict(str(pkl))
    if pose == "tpose":
        return copy_smplx_params_tpose_rest(params)
    if pose == "reconstruction":
        return params
    raise AssertionError(f"unhandled pose {pose!r}")


def _template_dict_from_bundle(bundle: dict) -> dict:
    if "parent" in bundle:
        return {"parent": bundle["parent"]}
    raise ValueError(
        "Bundle has no 'parent' key — export inference with the same template so parent indices are saved."
    )


def _render_spin_video(
    *,
    renderer,
    estimator,
    gaussian_params: dict,
    device: torch.device,
    static_smplx_params: dict,
    num_frames: int,
    fps: float,
    video_path: Path,
) -> None:
    frames_rgb: list[numpy.ndarray] = []
    gp_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in gaussian_params.items()}

    with torch.inference_mode():
        for i in range(num_frames):
            p_i = copy_smplx_params_spin_global_yaw(static_smplx_params, i, num_frames)
            verts_i = vertices_from_smplx_param_dict(p_i)
            fused = replay_fused_gaussian_means(
                estimator,
                verts_i,
                gp_cpu,
                device=device,
            )
            gp_render = gaussian_params_for_render(gp_cpu, device=device)
            rgb = renderer.render(
                fused,
                gp_render,
                view_name="0",
                save_folder_path=None,
            )
            frames_rgb.append(
                (rgb[0].clamp(0, 1).detach().cpu().numpy() * 255.0).astype(numpy.uint8)
            )

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(video_path), frames_rgb, fps=fps)
    print(f"Saved video ({num_frames} frames @ {fps} fps) → {video_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay Gaussian appearance from inference .pt (anim.yaml: pose + display_mode)."
    )
    parser.add_argument("--start-subject", type=str, required=True)
    parser.add_argument("--end-subject", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/nlfgs_gpu.yaml")
    parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Path to inference .pt (default: output/<subject>/<stem>.pt from config naming).",
    )
    parser.add_argument(
        "--pose",
        type=str,
        choices=sorted(POSE_CHOICES),
        default=None,
        help="Override animation.pose (reconstruction | tpose | custom).",
    )
    parser.add_argument(
        "--display-mode",
        type=str,
        choices=sorted(DISPLAY_CHOICES),
        default=None,
        help="Override animation.display_mode (image | video).",
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default=None,
        help="Override path to subject smplx_param.pkl (also used as identity for pose: custom).",
    )
    parser.add_argument(
        "--views-subdir",
        type=str,
        default=None,
        help="Override animation / inference reconstruction_subdir for image-mode PNGs.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="anim",
        help="Prefix for PNGs; for video, file is {prefix}_{subject}_{pose}.mp4",
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="Also write anim_{subject}.ply (single static pose; image mode or first frame semantics N/A).",
    )
    args = parser.parse_args()

    os.environ["NLFGS_CONFIG"] = args.config
    cfg = load_config(args.config)
    device = device_from_cfg(cfg)
    apply_matmul_precision_for_device(cfg, device)

    pose, display_mode = _animation_pose_display_mode(cfg, args.pose, args.display_mode)
    fps, duration_s = _animation_fps_duration(cfg)
    pkl_override = Path(args.pkl).resolve() if args.pkl else None

    subjects = _subjects_in_range(cfg, args.start_subject, args.end_subject)
    renderer = gsplat_renderer_if_cuda(device)

    model = build_nlf_gaussian_model(cfg, device)
    model.eval()
    estimator = model.avatar_estimator
    out_root = _resolve_output_root(cfg)
    video_rel_subdir = _animation_video_subdir(cfg)

    views_subdir = args.views_subdir or _reconstruction_subdir(cfg)

    for subject in subjects:
        bundle_path = Path(args.bundle) if args.bundle else _default_bundle_path(cfg, subject)
        bundle_path = _resolve_path(bundle_path) if not bundle_path.is_absolute() else bundle_path
        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"No inference bundle at {bundle_path}. Run ``inference.py`` first (saves {subject}.pt), "
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

        static_params = _base_smplx_params(cfg, subject, pose, pkl_override)

        sub_dir = out_root / subject
        sub_dir.mkdir(parents=True, exist_ok=True)

        if display_mode == "video":
            if device.type != "cuda":
                n = frame_count_for_duration_seconds(fps, duration_s)
                print(
                    f"[{subject}] Skipping spin video (CUDA required). Would write {n} frames @ {fps} Hz."
                )
                continue
            assert renderer is not None
            n_frames = frame_count_for_duration_seconds(fps, duration_s)
            vid_dir = sub_dir / video_rel_subdir
            vid_dir.mkdir(parents=True, exist_ok=True)
            vid_path = vid_dir / f"{args.save_prefix}_{subject}_{pose}.mp4"
            _render_spin_video(
                renderer=renderer,
                estimator=estimator,
                gaussian_params=gaussian_params,
                device=device,
                static_smplx_params=static_params,
                num_frames=n_frames,
                fps=fps,
                video_path=vid_path,
            )
            continue

        # display_mode == image — single static pose, four canonical cameras
        verts_new = vertices_from_smplx_param_dict(static_params)
        fused = replay_fused_gaussian_means(
            estimator,
            verts_new,
            gaussian_params,
            device=device,
        )
        gp_render = gaussian_params_for_render(gaussian_params, device=device)

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
        views_dir = sub_dir / views_subdir
        renderer.render_canonical_views(
            fused,
            gp_render,
            views_dir,
            save_prefix=args.save_prefix,
        )
        print(
            f"[{subject}] Rendered {args.save_prefix}_*.png (pose={pose}) → {views_dir.resolve()}"
        )


if __name__ == "__main__":
    main()
