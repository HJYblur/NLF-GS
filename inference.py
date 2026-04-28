"""
Run NLF-GS inference and always save ``{subject}.pt``.

When ``inference.save_reconstruction`` is true, also write ``reconstructed_{subject}.ply`` and four PNGs
(``reconstructed_<view>.png``) under ``reconstruction_subdir``. ``save_test_ply`` writes
``<subject>_view.pt`` / ``.pkl`` (log scales, no offset).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent / "src"))

from src.avatar_utils.config import load_config
from src.training.nlfgs_builder import (
    apply_matmul_precision_for_device,
    build_nlf_gaussian_model,
    device_from_cfg,
    gsplat_renderer_if_cuda,
)
from src.avatar_utils.ply_loader import reconstruct_gaussian_avatar_as_ply
from src.avatar_utils.smplx_loader import load_smplx_coord3d, vertices_3d_to_2d
from src.avatar_utils.camera import load_camera_mapping
from src.data.datasets import VIEW_ORDER, AvatarDataset
from src.encoder.avatar_template import AvatarTemplate
from src.training.nlfgs import NlfGaussianModel

# PLY export (fixed): linear ``scales`` in memory, ``save_ply`` writes ``log(scale)``; include ``parent_*``.
PLY_SAVE_LOG_SCALES = True
PLY_SAVE_INCLUDE_PARENT = True


def _find_subject_index(ds: AvatarDataset, subject: str) -> int:
    for i, rec in enumerate(ds._records):
        if rec["subject"] == subject:
            return i
    raise ValueError(
        f"Subject {subject!r} not found under {ds.root}. "
        f"Expected folders like {ds.root}/<subject>/ with {VIEW_ORDER} views."
    )


def _subject_sort_key(subject: str) -> tuple[int, int | str]:
    if subject.isdigit():
        return (0, int(subject))
    return (1, subject)


def _subjects_in_range(cfg: dict, start_subject: str, end_subject: str) -> list[str]:
    """Return available dataset subjects within the inclusive [start, end] range."""
    data_cfg = cfg.get("data", {})
    data_root = data_cfg.get("processed_root", data_cfg.get("root", "data/processed_test"))
    ds = AvatarDataset(root=data_root)
    available = sorted({str(rec["subject"]) for rec in ds._records}, key=_subject_sort_key)

    if not available:
        raise ValueError(f"No subjects found under {data_root!r}.")

    start = str(start_subject).strip()
    end = str(end_subject).strip()

    if start.isdigit() and end.isdigit():
        start_i, end_i = int(start), int(end)
        if start_i > end_i:
            raise ValueError(f"start-subject ({start}) must be <= end-subject ({end}).")
        selected = [s for s in available if s.isdigit() and start_i <= int(s) <= end_i]
    else:
        if start > end:
            raise ValueError(f"start-subject ({start}) must be <= end-subject ({end}) in lexical order.")
        selected = [s for s in available if start <= s <= end]

    if not selected:
        raise ValueError(
            f"No dataset subjects found in range [{start}, {end}] under {data_root!r}."
        )

    return selected


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _reconstruction_ply_filename(inf_cfg: dict, subject: str) -> str:
    """PLY next to ``{subject}.pt`` when ``save_reconstruction`` is true (default ``reconstructed_{subject}.ply``)."""
    raw = inf_cfg.get("reconstruction_ply_filename")
    if raw is not None and str(raw).strip() != "":
        return str(raw).format(subject=subject)
    return f"reconstructed_{subject}.ply"


def _reconstruction_png_prefix(inf_cfg: dict) -> str:
    """PNG basename prefix for four views (``reconstructed_front.png``, …)."""
    raw = inf_cfg.get("reconstruction_save_prefix")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip()
    return "reconstructed"


def _inference_pt_filename(inf_cfg: dict, subject: str) -> str:
    """Primary ``.pt`` filename (default ``{subject}.pt``). Optional ``pt_filename`` may contain ``{subject}``."""
    raw = inf_cfg.get("pt_filename")
    if raw is not None and str(raw).strip() != "":
        return str(raw).format(subject=subject)
    return f"{subject}.pt"


def _inference_view_pt_filename(inf_cfg: dict, subject: str) -> str:
    """``<subject>_view.pt`` (``save_test_ply``): log ``scales``, no ``offset`` in ``gaussian_params``."""
    base = Path(_inference_pt_filename(inf_cfg, subject))
    stem = base.stem
    suffix = base.suffix if base.suffix else ".pt"
    return f"{stem}_view{suffix}"


def _inference_view_pkl_filename(inf_cfg: dict, subject: str) -> str:
    """``<subject>_view.pkl`` (``save_test_ply``): same dict as ``_view.pt``, pickled."""
    base = Path(_inference_pt_filename(inf_cfg, subject))
    stem = base.stem
    return f"{stem}_view.pkl"


def _gaussian_params_for_view_bundle(
    gp_cpu: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """View export: same keys as render ``gaussian_params`` except ``offset`` omitted; ``scales`` are log(linear)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in gp_cpu.items():
        if k == "offset":
            continue
        if k == "scales":
            out[k] = torch.log(torch.clamp(v, min=1e-10))
        else:
            out[k] = v
    return out


def _inference_pt_shared_meta(
    subject: str,
    vertices3d: torch.Tensor,
    ckpt_path: Path,
    config_path: Path,
) -> dict[str, object]:
    return {
        "subject": subject,
        "vertices3d": vertices3d,
        "checkpoint": str(ckpt_path.resolve()),
        "config": str(config_path.resolve()),
    }


def _inference_pt_bundle(
    *,
    scales_are_log_space: bool,
    gaussian_3d: torch.Tensor,
    gaussian_params: dict[str, torch.Tensor],
    template_avatar: dict,
    meta: dict[str, object],
) -> dict[str, object]:
    """Same top-level schema for render and view bundles (differs only in tensor values + log flag)."""
    out: dict[str, object] = {
        **meta,
        "scales_are_log_space": scales_are_log_space,
        "gaussian_3d": gaussian_3d.detach().cpu(),
        "gaussian_params": gaussian_params,
    }
    if "parent" in template_avatar:
        out["parent"] = template_avatar["parent"].detach().cpu()
    return out


def _vertices3d_for_inference(cfg: dict, subject: str) -> torch.Tensor:
    """SMPL-X vertices aligned with avatar_template parent indices (see configs: inference.smplx_source)."""
    inf = cfg.get("inference", {})
    source = str(inf.get("smplx_source", "subject_params")).lower().strip()
    if source == "canonical_mesh":
        mesh = AvatarTemplate().load_cano_mesh()
        return torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))

    if source != "subject_params":
        raise ValueError(
            f"inference.smplx_source must be 'subject_params' or 'canonical_mesh', got {source!r}"
        )

    data_cfg = cfg.get("data", {})
    smplx_root = Path(
        data_cfg.get(
            "smplx_root",
            data_cfg.get("processed_root", "data/processed_test"),
        )
    )
    raw_smplx_root = Path(data_cfg.get("raw_smplx_root", "data/THuman_2.0_smplx_paras"))
    candidates = [
        smplx_root / subject / f"{subject}_smplx.pkl",
        smplx_root / subject / "smplx_param.pkl",
        smplx_root / subject / "mesh_smplx.obj",
        raw_smplx_root / subject / "smplx_param.pkl",
        raw_smplx_root / subject / "mesh_smplx.obj",
    ]
    for p in candidates:
        if p.exists():
            return load_smplx_coord3d(str(p))
    raise FileNotFoundError(
        f"SMPL-X params not found for subject {subject!r} under {smplx_root} or {raw_smplx_root} "
        f"(expected {subject}_smplx.pkl, smplx_param.pkl, or mesh_smplx.obj). "
        f"Set inference.smplx_source: canonical_mesh to use avatar_template.cano_mesh_path instead."
    )


def _load_checkpoint(model: NlfGaussianModel, ckpt_path: str, device: torch.device) -> None:
    ckpt_path = str(ckpt_path)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    sd = blob.get("state_dict", blob)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()


def run_inference(
    cfg: dict,
    subject: str,
    checkpoint: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, str, dict]:
    """Run the forward pass for one subject (4 views). Returns fused Gaussians, vertices3d, subject, template avatar dict."""
    data_cfg = cfg.get("data", {})
    data_root = data_cfg.get("processed_root", data_cfg.get("root", "data/processed_test"))
    ds = AvatarDataset(root=data_root)
    idx = _find_subject_index(ds, subject)
    batch = ds[idx]
    img_float = batch["images_float"].to(device)
    view_names = batch["view_names"]
    B = img_float.shape[0]
    assert B == 4 and list(view_names) == VIEW_ORDER, "Inference expects 4 views in canonical order."

    vertices3d = _vertices3d_for_inference(cfg, subject)
    vertices3d = vertices3d.to(device=device, dtype=torch.float32)
    if vertices3d.shape[0] > 0:
        viewmats, Ks = load_camera_mapping(view_names)
        verts2d_list = []
        for v_idx in range(viewmats.shape[0]):
            v2d = vertices_3d_to_2d(vertices3d, Ks[v_idx], viewmats[v_idx])
            verts2d_list.append(v2d)
        vertices2d = torch.stack(verts2d_list, dim=0).to(device=device, dtype=torch.float32)
    else:
        vertices2d = torch.empty(B, 0, 2, device=device, dtype=torch.float32)

    model = build_nlf_gaussian_model(cfg, device)
    _load_checkpoint(model, checkpoint, device)

    H, W = img_float.shape[-2:]
    grad_ctx = torch.inference_mode()
    with grad_ctx:
        feat_list = []
        for v_idx in range(B):
            f_v = model.backbone.extract_feature_map(img_float[v_idx : v_idx + 1])
            feat_list.append(f_v)
        if isinstance(feat_list[0], dict):
            feats = {
                level: torch.cat([fv[level] for fv in feat_list], dim=0)
                for level in feat_list[0].keys()
            }
        else:
            feats = torch.cat(feat_list, dim=0)

        local_feats, view_weights, gaussian_3d, _centers2d = (
            model.avatar_estimator.feature_sample_with_visibility(
                feats,
                vertices3d,
                vertices2d,
                img_shape=(H, W),
                view_names=view_names,
            )
        )
        local_frames = model.avatar_estimator.compute_gaussian_local_frames(
            vertices3d, device=gaussian_3d.device, batch_size=B
        )
        if local_frames.shape[0] == 1 and B > 1:
            local_frames = local_frames.expand(B, -1, -1, -1)

        if model.num_views == 4:
            if model.view_fusion is not None:
                local_feats = model.view_fusion(local_feats, view_weights)
            else:
                weights = view_weights.clamp_min(0.0)
                weights_sum = weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
                weights = weights / weights_sum
                local_feats = (local_feats * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
            gaussian_3d_decode = gaussian_3d[0:1]
            local_frames_decode = local_frames[0:1]
        else:
            gaussian_3d_decode = gaussian_3d
            local_frames_decode = local_frames

        gaussian_params_fused = model.decoder(local_feats)
        gaussian_3d_fused = gaussian_3d_decode[0]
        offset_local = gaussian_params_fused.get("offset", None)
        if offset_local is not None:
            gaussian_3d_fused = gaussian_3d_fused + torch.einsum(
                "nij,nj->ni", local_frames_decode[0], offset_local
            )

    template_avatar = model.template.avatar
    return gaussian_3d_fused, gaussian_params_fused, vertices3d.cpu(), subject, template_avatar


def main():
    parser = argparse.ArgumentParser(description="NLF-GS inference (single subject, 4 views)")
    parser.add_argument(
        "--start-subject",
        type=str,
        required=True,
        help="Start subject id (inclusive), e.g. 0001",
    )
    parser.add_argument(
        "--end-subject",
        type=str,
        required=True,
        help="End subject id (inclusive), e.g. 0020",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nlgfs_test.yaml",
        help="YAML config (default: configs/nlgfs_test.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Lightning .ckpt (default: inference.checkpoint in config)",
    )
    args = parser.parse_args()

    os.environ["NLFGS_CONFIG"] = args.config
    cfg = load_config(args.config)
    device = device_from_cfg(cfg)
    apply_matmul_precision_for_device(cfg, device)

    inf_cfg = cfg.get("inference", {})
    ckpt_arg = args.checkpoint or inf_cfg.get("checkpoint")
    if not ckpt_arg:
        raise ValueError("Pass --checkpoint or set inference.checkpoint in the YAML config.")
    ckpt_path = _resolve_path(str(ckpt_arg))

    subjects = _subjects_in_range(cfg, args.start_subject, args.end_subject)
    print(
        f"Running inference for {len(subjects)} subject(s): "
        f"{subjects[0]} -> {subjects[-1]}"
    )

    renderer = gsplat_renderer_if_cuda(device)

    out_root = Path(str(inf_cfg.get("output_dir", cfg.get("render", {}).get("save_path", "output"))))

    for subject in subjects:
        gaussian_3d, gaussian_params, vertices3d, _subject, template_avatar = run_inference(
            cfg, subject, str(ckpt_path), device
        )

        sub_dir = out_root / subject
        sub_dir.mkdir(parents=True, exist_ok=True)

        gp_cpu = {k: v.detach().cpu() for k, v in gaussian_params.items()}
        gp_ply = {k: v for k, v in gp_cpu.items() if k != "offset"}

        cfg_path = Path(args.config).resolve()
        pt_meta = _inference_pt_shared_meta(subject, vertices3d, ckpt_path, cfg_path)

        render_path = sub_dir / _inference_pt_filename(inf_cfg, subject)
        render_bundle = _inference_pt_bundle(
            scales_are_log_space=False,
            gaussian_3d=gaussian_3d,
            gaussian_params=gp_cpu,
            template_avatar=template_avatar,
            meta=pt_meta,
        )
        torch.save(render_bundle, render_path)
        print(f"[{subject}] Saved {render_path.name} → {render_path.resolve()}")

        save_reconstruction = bool(inf_cfg.get("save_reconstruction", False))

        if save_reconstruction:
            ply_path = sub_dir / _reconstruction_ply_filename(inf_cfg, subject)
            reconstruct_gaussian_avatar_as_ply(
                gaussian_3d.detach().cpu(),
                gp_ply,
                template_avatar,
                str(ply_path),
                log_scales=PLY_SAVE_LOG_SCALES,
                include_parent=PLY_SAVE_INCLUDE_PARENT,
            )
            print(f"[{subject}] Saved {ply_path.name} → {ply_path.resolve()}")

        if bool(inf_cfg.get("save_test_ply", False)):
            gp_view = _gaussian_params_for_view_bundle(gp_cpu)
            view_bundle = _inference_pt_bundle(
                scales_are_log_space=True,
                gaussian_3d=gaussian_3d,
                gaussian_params=gp_view,
                template_avatar=template_avatar,
                meta=pt_meta,
            )
            view_pt = sub_dir / _inference_view_pt_filename(inf_cfg, subject)
            view_pkl = sub_dir / _inference_view_pkl_filename(inf_cfg, subject)
            torch.save(view_bundle, view_pt)
            with open(view_pkl, "wb") as f:
                pickle.dump(view_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[{subject}] Saved view tensor bundle to {view_pt.resolve()}")
            print(f"[{subject}] Saved view pickle bundle to {view_pkl.resolve()}")

        if not save_reconstruction:
            continue

        if device.type != "cuda":
            print(
                f"[{subject}] Skipping reconstruction PNGs (CUDA required for gsplat). "
                f"PLY already saved under {sub_dir.resolve()}."
            )
            continue

        gaussian_3d_gpu = gaussian_3d.to(device)
        gaussian_params_gpu = {k: v.to(device) for k, v in gaussian_params.items()}

        recon_subdir = inf_cfg.get("reconstruction_subdir") or inf_cfg.get("canonical_views_subdir") or "reconstruction"
        views_dir = sub_dir / str(recon_subdir)
        recon_prefix = _reconstruction_png_prefix(inf_cfg)
        assert renderer is not None
        renderer.render_canonical_views(
            gaussian_3d_gpu,
            gaussian_params_gpu,
            views_dir,
            save_prefix=recon_prefix,
        )

        print(f"[{subject}] Saved {recon_prefix}_*.png under {views_dir.resolve()}")


if __name__ == "__main__":
    main()
