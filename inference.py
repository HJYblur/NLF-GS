import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

# Make 'src' importable when running as a script
sys.path.append(str(Path(__file__).parent / "src"))

from src.avatar_utils.camera import intrinsic_matrix_from_field_of_view, look_at_viewmatrix
from src.avatar_utils.config import load_config
from src.avatar_utils.ply_loader import save_ply
from src.avatar_utils.smplx_loader import load_smplx_coord3d, vertices_3d_to_2d
from src.backbone.resnet_fpn import FrozenResNet50FPNExtractor
from src.decoder.gaussian_decoder import GaussianDecoder
from src.encoder.avatar_template import AvatarTemplate
from src.encoder.gaussian_estimator import AvatarGaussianEstimator
from src.encoder.identity_encoder import IdentityEncoder


VIEW_ORDER = ["front", "back", "left", "right"]
IMG_EXTS = (".png", ".jpg", ".jpeg")


def _find_view_image(subject_dir: Path, subject_name: str, view_name: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = subject_dir / f"{subject_name}_{view_name}{ext}"
        if p.exists():
            return p
    return None


def _load_subject_images(subject_dir: Path, image_size: Tuple[int, int]) -> torch.Tensor:
    images: List[torch.Tensor] = []
    for view_name in VIEW_ORDER:
        p = _find_view_image(subject_dir, subject_dir.name, view_name)
        if p is None:
            raise FileNotFoundError(
                f"Missing required view '{view_name}' under {subject_dir}. "
                f"Expected files like {subject_dir.name}_{view_name}.png"
            )
        img = Image.open(p).convert("RGB")
        if img.size != image_size:
            img = img.resize(image_size, Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        images.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(images, dim=0)  # [4,3,H,W]


def _canonical_vertices(cfg: Dict) -> torch.Tensor:
    cano_mesh_path = cfg.get("avatar", {}).get("template", {}).get("cano_mesh_path")
    if not cano_mesh_path:
        raise FileNotFoundError("No avatar.template.cano_mesh_path found in config for canonical fallback.")
    return load_smplx_coord3d(cano_mesh_path)


def _load_subject_vertices(subject_name: str, cfg: Dict) -> torch.Tensor:
    smplx_root = Path(cfg.get("data", {}).get("smplx_root", "data/THuman_2.0_smplx_params"))
    pkl_path = smplx_root / subject_name / "smplx_param.pkl"
    obj_path = smplx_root / subject_name / "mesh_smplx.obj"

    if pkl_path.exists():
        return load_smplx_coord3d(str(pkl_path))
    if obj_path.exists():
        return load_smplx_coord3d(str(obj_path))
    print(f"[WARN] No SMPL-X params for '{subject_name}'. Falling back to canonical pose mesh.")
    return _canonical_vertices(cfg)


def _project_vertices_for_views(vertices3d: torch.Tensor) -> torch.Tensor:
    from src.avatar_utils.camera import load_camera_mapping

    viewmats, Ks = load_camera_mapping(VIEW_ORDER)
    verts2d = [vertices_3d_to_2d(vertices3d, Ks[i], viewmats[i]) for i in range(len(VIEW_ORDER))]
    return torch.stack(verts2d, dim=0)


def _resolve_checkpoint(cli_checkpoint: Optional[str]) -> Path:
    if cli_checkpoint:
        p = Path(cli_checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    candidates: List[Path] = []
    for root in (Path("output"), Path("checkpoints"), Path("lightning_logs")):
        if root.exists():
            candidates.extend(root.rglob("*.ckpt"))

    if not candidates:
        raise FileNotFoundError(
            "No .ckpt file found. Pass --checkpoint explicitly or place checkpoints under "
            "output/, checkpoints/, or lightning_logs/."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_prefixed_state_dict(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], prefixes: Sequence[str]) -> None:
    matched: Dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        for k, v in state_dict.items():
            if k.startswith(prefix):
                matched[k[len(prefix) :]] = v
        if matched:
            break

    if not matched:
        print(f"[WARN] No weights found for prefixes: {list(prefixes)}")
        return

    missing, unexpected = module.load_state_dict(matched, strict=False)
    if missing:
        print(f"[WARN] Missing keys for {module.__class__.__name__}: {missing[:8]}")
    if unexpected:
        print(f"[WARN] Unexpected keys for {module.__class__.__name__}: {unexpected[:8]}")


def _build_inference_modules(cfg: Dict, device: torch.device):
    backbone_cfg = cfg.get("backbone", {})
    fpn_levels = tuple(backbone_cfg.get("fpn_levels", ["p2", "p3", "p4"]))

    feature_extractor = FrozenResNet50FPNExtractor(
        selected_levels=fpn_levels,
        backbone_weights_path=backbone_cfg.get("resnet50_weights_path"),
        frozen=True,
    ).to(device)

    c_local = int(backbone_cfg.get("fpn_out_channels", 256)) * len(fpn_levels)
    identity_encoder = IdentityEncoder(
        backbone_feat_dim=c_local,
        latent_dim=int(cfg.get("identity_encoder", {}).get("latent_dim", 64)),
    ).to(device)

    decoder = GaussianDecoder().to(device)
    template = AvatarTemplate(
        avatar_path=cfg.get("avatar", {}).get("template", {}).get("path"),
        cano_mesh_path=cfg.get("avatar", {}).get("template", {}).get("cano_mesh_path"),
    )
    estimator = AvatarGaussianEstimator(template).to(device)

    feature_extractor.eval()
    identity_encoder.eval()
    decoder.eval()
    estimator.eval()
    return feature_extractor, identity_encoder, decoder, estimator, template


def _predict_gaussians(
    images_float: torch.Tensor,
    vertices3d: torch.Tensor,
    vertices2d: torch.Tensor,
    feature_extractor: FrozenResNet50FPNExtractor,
    identity_encoder: IdentityEncoder,
    decoder: GaussianDecoder,
    estimator: AvatarGaussianEstimator,
    use_identity_encoder: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    with torch.inference_mode():
        img = images_float.to(device)
        verts3d = vertices3d.to(device)
        verts2d = vertices2d.to(device)

        feat_list = [feature_extractor(img[i : i + 1]) for i in range(img.shape[0])]
        feats = {k: torch.cat([fv[k] for fv in feat_list], dim=0) for k in feat_list[0].keys()}
        feat_for_id = next(iter(feats.values()))

        z_id = identity_encoder(feat_for_id) if use_identity_encoder else None
        H, W = img.shape[-2:]
        local_feats, view_weights, gaussian_3d, _ = estimator.feature_sample_with_visibility(
            feats, verts3d, verts2d, img_shape=(H, W)
        )
        local_frames = estimator.compute_gaussian_local_frames(verts3d, device=device, batch_size=img.shape[0])

        weights = view_weights.clamp_min(0.0)
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
        fused_feats = (local_feats * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        z_id_decode = None if z_id is None else z_id.mean(dim=0, keepdim=True)

        gaussian_params = decoder(fused_feats, z_id_decode)
        gaussian_xyz = gaussian_3d[0]
        if gaussian_params.get("offset", None) is not None:
            gaussian_xyz = gaussian_xyz + torch.einsum("nij,nj->ni", local_frames[0], gaussian_params["offset"])

    return gaussian_xyz, gaussian_params


def _save_avatar_ply(
    subject: str,
    gaussian_xyz: torch.Tensor,
    gaussian_params: Dict[str, torch.Tensor],
    template: AvatarTemplate,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_avatar_gaussian_smplx.ply"

    save_ply(
        {
            "xyz": gaussian_xyz.detach().cpu(),
            "shs": gaussian_params["sh"].detach().cpu(),
            "opacities": gaussian_params["alpha"].detach().cpu().unsqueeze(-1),
            "scales": gaussian_params["scales"].detach().cpu(),
            "rots": gaussian_params["rotation"].detach().cpu(),
            "parent": template.parents.detach().cpu(),
        },
        str(out_path),
    )
    return out_path


def _render_with_custom_cameras(
    subject: str,
    gaussian_xyz: torch.Tensor,
    gaussian_params: Dict[str, torch.Tensor],
    out_dir: Path,
    image_size: Tuple[int, int],
    yfov_deg: float,
    views: Sequence[str] = ("front", "left", "top", "back"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = image_size

    center = gaussian_xyz.mean(dim=0)
    extent = (gaussian_xyz.max(dim=0).values - gaussian_xyz.min(dim=0).values).max().clamp_min(1e-3)
    dist = float(extent * 3.0)

    eye_map = {
        "front": center + torch.tensor([0.0, 0.0, dist], device=gaussian_xyz.device),
        "back": center + torch.tensor([0.0, 0.0, -dist], device=gaussian_xyz.device),
        "left": center + torch.tensor([-dist, 0.0, 0.0], device=gaussian_xyz.device),
        "right": center + torch.tensor([dist, 0.0, 0.0], device=gaussian_xyz.device),
        "top": center + torch.tensor([0.0, dist, 0.0], device=gaussian_xyz.device),
    }

    sh = gaussian_params["sh"]
    sh_degree = int((int(sh.shape[1] // 3) ** 0.5) - 1)
    colors = sh.view(sh.shape[0], -1, 3).float().contiguous()

    K = intrinsic_matrix_from_field_of_view(
        fov_degrees=float(yfov_deg),
        imshape=[H, W],
        device=gaussian_xyz.device,
    ).squeeze(0)

    from torchvision.io import write_png
    from torchvision.transforms.functional import convert_image_dtype

    for view_name in views:
        if view_name not in eye_map:
            continue

        up = (0.0, 0.0, -1.0) if view_name == "top" else (0.0, 1.0, 0.0)
        viewmat, _ = look_at_viewmatrix(
            eye=eye_map[view_name],
            target=center,
            up=up,
            device=gaussian_xyz.device,
            forward="+z",
        )

        rendered, _, _ = rasterization(
            means=gaussian_xyz.float().contiguous(),
            quats=gaussian_params["rotation"].float().contiguous(),
            scales=gaussian_params["scales"].float().contiguous(),
            opacities=gaussian_params["alpha"].float().contiguous(),
            sh_degree=sh_degree,
            colors=colors,
            viewmats=viewmat.unsqueeze(0).contiguous(),
            Ks=K.unsqueeze(0).contiguous(),
            width=int(W),
            height=int(H),
            render_mode="RGB",
        )

        img = rendered[0].permute(2, 0, 1).detach().cpu().clamp(0, 1)
        write_png(convert_image_dtype(img, dtype=torch.uint8), str(out_dir / f"{subject}_{view_name}.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference to generate SMPL-X attached Gaussian avatars.")
    parser.add_argument("--config", type=str, default="configs/nlfgs_gpu.yaml")
    parser.add_argument(
        "--input_root",
        type=str,
        default="processed",
        help="Folder with inference subjects in processed-like 4-view layout.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Lightning checkpoint (.ckpt). If omitted, latest checkpoint is auto-selected.",
    )
    parser.add_argument("--output_root", type=str, default="output", help="Output folder.")
    args = parser.parse_args()

    os.environ["NLFGS_CONFIG"] = args.config
    cfg = load_config(args.config)

    device = torch.device(cfg.get("sys", {}).get("device", "cpu"))
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {ckpt_path}")

    feature_extractor, identity_encoder, decoder, estimator, template = _build_inference_modules(cfg, device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    _load_prefixed_state_dict(feature_extractor, state_dict, ["backbone.fpn_extractor.", "feature_extractor."])
    _load_prefixed_state_dict(identity_encoder, state_dict, ["identity_encoder."])
    _load_prefixed_state_dict(decoder, state_dict, ["decoder."])

    width, height = cfg.get("data", {}).get("image_size", [512, 512])
    out_root = Path(args.output_root)
    avatars_dir = out_root / "avatars"
    renders_dir = out_root / "renders"

    subjects = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not subjects:
        raise RuntimeError(f"No subject folders found under {input_root}")

    use_identity_encoder = bool(cfg.get("identity_encoder", {}).get("use_flag", True))

    for subj_dir in subjects:
        subject = subj_dir.name
        print(f"[Inference] Subject: {subject}")

        images_float = _load_subject_images(subj_dir, image_size=(int(width), int(height)))
        vertices3d = _load_subject_vertices(subject, cfg)
        vertices2d = _project_vertices_for_views(vertices3d)

        gaussian_xyz, gaussian_params = _predict_gaussians(
            images_float=images_float,
            vertices3d=vertices3d,
            vertices2d=vertices2d,
            feature_extractor=feature_extractor,
            identity_encoder=identity_encoder,
            decoder=decoder,
            estimator=estimator,
            use_identity_encoder=use_identity_encoder,
            device=device,
        )

        ply_path = _save_avatar_ply(subject, gaussian_xyz, gaussian_params, template, avatars_dir)
        print(f"  Saved Gaussian avatar: {ply_path}")

        _render_with_custom_cameras(
            subject=subject,
            gaussian_xyz=gaussian_xyz,
            gaussian_params=gaussian_params,
            out_dir=renders_dir / subject,
            image_size=(int(width), int(height)),
            yfov_deg=float(cfg.get("camera", {}).get("yfov_deg", 45.0)),
            views=("front", "left", "top", "back"),
        )
        print(f"  Saved renders under: {renders_dir / subject}")


if __name__ == "__main__":
    main()
