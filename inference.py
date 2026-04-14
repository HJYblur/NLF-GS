"""
Run NLF-GS inference for a single subject (4 canonical views) and save PLY / rendered PNGs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent / "src"))

from src.avatar_utils.config import load_config
from src.avatar_utils.ply_loader import reconstruct_gaussian_avatar_as_ply
from src.avatar_utils.smplx_loader import load_smplx_coord3d, vertices_3d_to_2d
from src.avatar_utils.camera import load_camera_mapping
from src.data.datasets import VIEW_ORDER, AvatarDataset
from src.decoder.gaussian_decoder import GaussianDecoder
from src.encoder.avatar_template import AvatarTemplate
from src.encoder.feature_extractor import FeatureExtractor
from src.encoder.identity_encoder import IdentityEncoder
from src.render.gaussian_renderer import GsplatRenderer
from src.training.trainer import NlfGaussianModel


def _find_subject_index(ds: AvatarDataset, subject: str) -> int:
    for i, rec in enumerate(ds._records):
        if rec["subject"] == subject:
            return i
    raise ValueError(
        f"Subject {subject!r} not found under {ds.root}. "
        f"Expected folders like {ds.root}/<subject>/ with {VIEW_ORDER} views."
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _inference_ply_filename(inf_cfg: dict, subject: str) -> str:
    """Default PLY name from ``smplx_source`` if ``ply_filename`` is unset."""
    raw = inf_cfg.get("ply_filename")
    if raw is not None and str(raw).strip() != "":
        return str(raw)
    src = str(inf_cfg.get("smplx_source", "subject_params")).lower().strip()
    if src == "subject_params":
        return f"paramed_{subject}.ply"
    return f"canonical_{subject}.ply"


def _inference_save_prefix(inf_cfg: dict) -> str:
    """Default canonical-render PNG prefix from ``smplx_source`` if ``save_prefix`` is unset."""
    raw = inf_cfg.get("save_prefix")
    if raw is not None and str(raw).strip() != "":
        return str(raw)
    src = str(inf_cfg.get("smplx_source", "subject_params")).lower().strip()
    if src == "subject_params":
        return "paramed"
    return "canonical"


def _inference_pt_filename(inf_cfg: dict, ply_basename: str) -> str:
    """Tensor bundle name: same stem as PLY unless ``pt_filename`` is set."""
    raw = inf_cfg.get("pt_filename")
    if raw is not None and str(raw).strip() != "":
        return str(raw)
    return f"{Path(ply_basename).stem}.pt"


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

    smplx_root = Path(cfg.get("data", {}).get("smplx_root", "data/THuman_2.0_smplx_params"))
    pkl = smplx_root / subject / "smplx_param.pkl"
    obj = smplx_root / subject / "mesh_smplx.obj"
    if pkl.exists():
        return load_smplx_coord3d(str(pkl))
    if obj.exists():
        return load_smplx_coord3d(str(obj))
    raise FileNotFoundError(
        f"SMPL-X params not found for subject {subject!r} under {smplx_root} "
        f"(expected smplx_param.pkl or mesh_smplx.obj). "
        f"Set inference.smplx_source: canonical_mesh to use avatar.template.cano_mesh_path instead."
    )


def _build_model(cfg: dict, device: torch.device) -> NlfGaussianModel:
    backbone_cfg = cfg.get("backbone", {})
    train_cfg = cfg.get("train", {})
    train_decoder_only = bool(train_cfg.get("train_decoder_only", True))
    fpn_levels = tuple(backbone_cfg.get("fpn_levels", ["p2", "p3", "p4"]))
    backbone = FeatureExtractor(
        fpn_levels=fpn_levels,
        resnet_weights_path=backbone_cfg.get("resnet50_weights_path"),
        freeze_resnet_fpn=train_decoder_only,
    )
    fpn_out_channels = int(backbone_cfg.get("fpn_out_channels", 256))
    c_local = fpn_out_channels * len(fpn_levels)
    id_latent_dim = int(cfg["identity_encoder"].get("latent_dim", 64))
    id_encoder = IdentityEncoder(backbone_feat_dim=c_local, latent_dim=id_latent_dim)
    decoder = GaussianDecoder()
    renderer = GsplatRenderer() if device.type == "cuda" else None
    return NlfGaussianModel(
        backbone_adapter=backbone,
        identity_encoder=id_encoder,
        decoder=decoder,
        renderer=renderer,
        train_decoder_only=train_decoder_only,
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
    data_root = cfg.get("data", {}).get("root", "data/processed_test")
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

    model = _build_model(cfg, device)
    _load_checkpoint(model, checkpoint, device)

    H, W = img_float.shape[-2:]
    grad_ctx = torch.inference_mode()
    with grad_ctx:
        feat_list = []
        for v_idx in range(B):
            f_v = model.backbone.extract_feature_map(
                image=img_float[v_idx : v_idx + 1], use_half=True
            )
            feat_list.append(f_v)
        if isinstance(feat_list[0], dict):
            feats = {
                level: torch.cat([fv[level] for fv in feat_list], dim=0)
                for level in feat_list[0].keys()
            }
        else:
            feats = torch.cat(feat_list, dim=0)

        if isinstance(feats, dict):
            feat_for_id = next(iter(feats.values()))
        else:
            feat_for_id = feats

        if model.use_identity_encoder:
            z_id = model.identity_encoder(feature_map=feat_for_id)
        else:
            z_id = None

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
            z_id_decode = None if z_id is None else z_id.mean(dim=0, keepdim=True)
        else:
            gaussian_3d_decode = gaussian_3d
            local_frames_decode = local_frames
            z_id_decode = z_id

        gaussian_params_fused = model.decoder(local_feats, z_id_decode)
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
        "--subject",
        type=str,
        required=True,
        help="Subject folder name under data/processed_test/<subject>/",
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
    device_str = cfg.get("sys", {}).get("device", "cpu")
    device = torch.device(device_str)

    try:
        if device.type == "cuda":
            matmul_prec = cfg.get("sys", {}).get("matmul_precision")
            torch.set_float32_matmul_precision(matmul_prec or "high")
    except Exception:
        pass

    inf_cfg = cfg.get("inference", {})
    ckpt_arg = args.checkpoint or inf_cfg.get("checkpoint")
    if not ckpt_arg:
        raise ValueError("Pass --checkpoint or set inference.checkpoint in the YAML config.")
    ckpt_path = _resolve_path(str(ckpt_arg))

    gaussian_3d, gaussian_params, vertices3d, subject, template_avatar = run_inference(
        cfg, args.subject, str(ckpt_path), device
    )

    out_root = Path(str(inf_cfg.get("output_dir", cfg.get("render", {}).get("save_path", "output"))))
    sub_dir = out_root / subject
    sub_dir.mkdir(parents=True, exist_ok=True)

    ply_name = _inference_ply_filename(inf_cfg, args.subject)
    ply_path = sub_dir / ply_name
    gp_cpu = {k: v.detach().cpu() for k, v in gaussian_params.items()}
    reconstruct_gaussian_avatar_as_ply(
        gaussian_3d.detach().cpu(),
        gp_cpu,
        template_avatar,
        str(ply_path),
        log_scales=bool(inf_cfg.get("ply_log_scales", True)),
        include_parent=bool(inf_cfg.get("ply_include_parent", True)),
    )

    if bool(inf_cfg.get("save_pt", False)):
        save_path = sub_dir / _inference_pt_filename(inf_cfg, ply_name)
        torch.save(
            {
                "subject": subject,
                "gaussian_3d": gaussian_3d.detach().cpu(),
                "gaussian_params": gp_cpu,
                "vertices3d": vertices3d,
                "checkpoint": str(ckpt_path.resolve()),
                "config": str(Path(args.config).resolve()),
            },
            save_path,
        )
        print(f"Saved tensor bundle to {save_path.resolve()}")

    if device.type != "cuda":
        print(
            "Skipping canonical-view PNG renders (CUDA required for gsplat). "
            f"Gaussian PLY saved to {ply_path.resolve()}."
        )
        return

    renderer = GsplatRenderer()
    gaussian_3d_gpu = gaussian_3d.to(device)
    gaussian_params_gpu = {k: v.to(device) for k, v in gaussian_params.items()}

    views_dir = sub_dir / str(inf_cfg.get("canonical_views_subdir", "canonical_views"))
    train_prefix = _inference_save_prefix(inf_cfg)
    renderer.render_canonical_views(
        gaussian_3d_gpu,
        gaussian_params_gpu,
        views_dir,
        save_prefix=train_prefix,
    )

    print(f"Saved Gaussian PLY to {ply_path.resolve()}")
    print(f"Saved {train_prefix}_*.png under {views_dir.resolve()}")


if __name__ == "__main__":
    main()
