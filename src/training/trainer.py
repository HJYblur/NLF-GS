from typing import Any, Dict, Optional
from contextlib import nullcontext
from pathlib import Path
import math
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
from PIL import Image
from encoder.feature_extractor import FeatureExtractor
from encoder.gaussian_estimator import AvatarGaussianEstimator
from encoder.learned_view_fusion import LearnedViewFusion
from encoder.identity_encoder import IdentityEncoder
from encoder.avatar_template import AvatarTemplate
from decoder.gaussian_decoder import GaussianDecoder
from render.gaussian_renderer import GsplatRenderer
from training.losses import LossFunctions
from avatar_utils.config import get_config


class NlfGaussianModel(L.LightningModule):
    def __init__(
        self,
        backbone_adapter: FeatureExtractor,
        identity_encoder: IdentityEncoder,
        decoder: GaussianDecoder,
        renderer: GsplatRenderer,
        train_decoder_only: bool = True,
    ):
        super().__init__()
        self._profile_gpu = bool(get_config().get("train", {}).get("profile_gpu", False))
        self.use_identity_encoder = bool(
            get_config().get("identity_encoder", {}).get("use_flag", True)
        )
        self.num_views = int(get_config().get("data", {}).get("num_views", 1))
        if self.num_views not in (1, 4):
            raise ValueError(f"data.num_views must be 1 or 4, got {self.num_views}")
        fusion_cfg = get_config().get("fusion", {})
        self.fusion_mode = str(fusion_cfg.get("mode", "fixed"))
        if self.num_views == 4 and self.fusion_mode not in ("fixed", "learned"):
            raise ValueError(f"fusion.mode must be 'fixed' or 'learned', got {self.fusion_mode!r}")
        dec_cfg = get_config().get("decoder", {})
        feat_dim = int(dec_cfg.get("in_dim", get_config().get("model", {}).get("local_feature_dim", 768)))
        fusion_hidden = int(fusion_cfg.get("hidden_dim", dec_cfg.get("hidden", 256)))
        self.view_fusion = (
            LearnedViewFusion(feat_dim=feat_dim, hidden_dim=fusion_hidden)
            if (self.num_views == 4 and self.fusion_mode == "learned")
            else None
        )
        self.template = AvatarTemplate()
        self.backbone = backbone_adapter
        self.avatar_estimator = AvatarGaussianEstimator(self.template)
        self.identity_encoder = identity_encoder
        self.decoder = decoder
        self.renderer = renderer
        self.loss_fn = LossFunctions()    
        self.train_decoder_only = train_decoder_only

        # Keep backbone frozen only in decoder-only mode.
        if hasattr(self.backbone, "set_resnet_fpn_frozen"):
            self.backbone.set_resnet_fpn_frozen(self.train_decoder_only)

        # Read optimizer & scheduler settings from config and save as hyperparameters
        train_cfg = get_config().get("train", {})
        lr = float(train_cfg.get("lr", 1e-4))
        wd = float(train_cfg.get("wd", train_cfg.get("weight_decay", 0.01)))
        betas = train_cfg.get("betas", [0.9, 0.99])
        eps = float(train_cfg.get("eps", 1e-8))
        warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
        bb_lr_mult = float(train_cfg.get("bb_lr_mult", 0.1))
        min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))
        scheduler_name = train_cfg.get("scheduler", "cosine")
        # Persist to hparams for use in configure_optimizers
        self.save_hyperparameters({
            "lr": lr,
            "wd": wd,
            "betas": betas,
            "eps": eps,
            "warmup_ratio": warmup_ratio,
            "bb_lr_mult": bb_lr_mult,
            "min_lr_ratio": min_lr_ratio,
            "scheduler": scheduler_name,
        })

        # If True, freeze all parameters except the decoder's so only decoder gets updated.
        if self.train_decoder_only:
            self.freeze_encoder()

        # Feature-dump settings for PCA analysis
        analysis_cfg = get_config().get("analysis", {})
        self._dump_local_feats = bool(analysis_cfg.get("dump_local_feats", False))
        self._dump_subject = analysis_cfg.get("dump_subject", None)  # None → dump all
        self._dump_dir = Path(str(analysis_cfg.get("dump_dir", "output/analysis")))
            
        self._shape_debug_logged = False
        render_cfg = get_config().get("render", {})
        self._log_render_outputs_online = bool(
            render_cfg.get("log_online", True)
        )
        self._render_log_stages = set(render_cfg.get("log_stages", ["train", "val"]))
        train_cfg = get_config().get("train", {})
        self._log_train_losses = bool(train_cfg.get("log_train_losses", True))
        self._log_val_losses = bool(train_cfg.get("log_val_losses", True))
        self._tracked_loss_names = train_cfg.get("tracked_losses", None)
        raw_test_renders = train_cfg.get("test_renders", [7, 100, 350])
        self._test_renders = {int(subject_id) for subject_id in raw_test_renders}
        debug_cfg = get_config().get("debug", {})
        self._dump_view_weight_images = bool(debug_cfg.get("dump_view_weight_images", False))
        self._view_weight_debug_dir = Path(str(debug_cfg.get("view_weight_dir", "debug")))

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss_dict = self.shared_step(batch=batch, batch_idx=batch_idx, stage="train")
        self._log_loss_dict(loss_dict, stage="train")
        return loss_dict["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        val_loss_dict = self.shared_step(batch=batch, batch_idx=batch_idx, stage="val")
        self._log_loss_dict(val_loss_dict, stage="val")
        return val_loss_dict["loss"]

    def shared_step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> torch.Tensor:
        # Extract data from batch
        img_float, img_uint8, masks_float, (B, H, W), subject, view_names, vertices3d, vertices2d = self.process_input(batch)

        grad_ctx = torch.inference_mode() if self.train_decoder_only else nullcontext()
        with grad_ctx:
            # Process backbone one view at a time to avoid holding B full-res
            # feature maps on GPU simultaneously (saves ~(B-1)/B of backbone VRAM).
            feat_list = []
            for v_idx in range(B):
                f_v = self.backbone.extract_feature_map(
                    image=img_float[v_idx : v_idx + 1], use_half=True
                )
                feat_list.append(f_v)
            if isinstance(feat_list[0], dict):
                feats = {
                    level: torch.cat([fv[level] for fv in feat_list], dim=0)
                    for level in feat_list[0].keys()
                }
            else:
                feats = torch.cat(feat_list, dim=0)  # (B, C, Hf, Wf)
            del feat_list
            gt_images = img_float
            del img_float

        # Keep raw backbone feature magnitudes.
        if isinstance(feats, dict):
            feat_for_id = next(iter(feats.values()))
        else:
            feat_for_id = feats

        """
        Encode:
        z_id: Identity Latent Vector (B, D)
        local_feats: Local Features sampled at Gaussian centers (B, N, C_local)
        gaussian_3d: Gaussian 3D Coordinates (B, N, 3)
        """
        B_feats = feat_for_id.shape[0]
        assert B == B_feats, "Batch size mismatch between image and features"

        with grad_ctx:
            if self.use_identity_encoder:
                z_id = self.identity_encoder(feature_map=feat_for_id)  # (1, D)
            else:
                z_id = None

            local_feats, view_weights, gaussian_3d, centers2d = (
                self.avatar_estimator.feature_sample_with_visibility(
                    feats,
                    vertices3d,
                    vertices2d,
                    img_shape=(H, W),
                    view_names=view_names,
                )
            )  # (B, N, C_local), (B, N), (B, N, 3), (B, N, 2)
            local_frames = self.avatar_estimator.compute_gaussian_local_frames(
                vertices3d, device=gaussian_3d.device, batch_size=B
            )  # (B, N, 3, 3)
            if local_frames.shape[0] == 1 and B > 1:
                local_frames = local_frames.expand(B, -1, -1, -1)
        local_feats_prefusion = local_feats
        # Use all views for supervision; `data.num_views` only controls
        # whether decoder input is per-view (1) or fused (4).
        if self.num_views == 4:
            if self.view_fusion is not None:
                local_feats = self.view_fusion(local_feats, view_weights)
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


        feat_size = feat_for_id.shape[-2:]  # (Hf, Wf)
        self._maybe_dump_local_feats(
            subject,
            local_feats,
            centers2d,
            (H, W),
            feat_size,
            local_feats_prefusion,
            view_weights,
        )
        self._maybe_dump_view_weight_images(
            subject=subject,
            view_names=view_names,
            stage=stage,
            batch_idx=batch_idx,
            img_size=(H, W),
        )

        if not self._shape_debug_logged:
            assert local_feats.shape[-1] == self.decoder.in_dim, (
                f"Local feature dim {local_feats.shape[-1]} != decoder.in_dim {self.decoder.in_dim}"
            )
            self._shape_debug_logged = True

        # Free large intermediates early to reduce peak VRAM before decoding
        del feats
        del feat_for_id
        del img_uint8

        assert gaussian_3d.shape[0] == B, "Mismatch between gaussian_3d and number of views"

        per_view_losses = []

        if self.num_views == 4:
            gaussian_params_fused = self.decoder(local_feats, z_id_decode)
            gaussian_3d_fused = gaussian_3d_decode[0]
            offset_local = gaussian_params_fused.get("offset", None)
            if offset_local is not None:
                gaussian_3d_fused = gaussian_3d_fused + torch.einsum(
                    "nij,nj->ni", local_frames_decode[0], offset_local
                )

            for view_idx, view_name in enumerate(view_names):
                rendered_imgs = self.renderer.render(
                    gaussian_3d=gaussian_3d_fused,
                    gaussian_params=gaussian_params_fused,
                    view_name=view_name,
                )
                self._log_render_output(
                    rendered_imgs=rendered_imgs,
                    subject=subject,
                    view_name=view_name,
                    stage=stage,
                )

                pred = rendered_imgs.permute(0, 3, 1, 2)
                gt = gt_images[view_idx : view_idx + 1]
                mask = None if masks_float is None else masks_float[view_idx : view_idx + 1]
                per_view_losses.append(
                    self.loss_fn(
                        pred,
                        gt,
                        mask,
                        gaussian_params=gaussian_params_fused,
                        gaussian_3d=gaussian_3d_fused.unsqueeze(0),
                    )
                )
        else:
            for view_idx, view_name in enumerate(view_names):
                local_feats_view = local_feats[view_idx : view_idx + 1]
                z_id_view = None if z_id_decode is None else z_id_decode
                if z_id_decode is not None and z_id_decode.shape[0] == B:
                    z_id_view = z_id_decode[view_idx : view_idx + 1]

                gaussian_params_view = self.decoder(local_feats_view, z_id_view)
                gaussian_3d_view = gaussian_3d_decode[view_idx]
                offset_local = gaussian_params_view.get("offset", None)
                if offset_local is not None:
                    frame_idx = view_idx if local_frames_decode.shape[0] > 1 else 0
                    gaussian_3d_view = gaussian_3d_view + torch.einsum(
                        "nij,nj->ni", local_frames_decode[frame_idx], offset_local
                    )

                rendered_imgs = self.renderer.render(
                    gaussian_3d=gaussian_3d_view,
                    gaussian_params=gaussian_params_view,
                    view_name=view_name,
                )
                self._log_render_output(
                    rendered_imgs=rendered_imgs,
                    subject=subject,
                    view_name=view_name,
                    stage=stage,
                )

                pred = rendered_imgs.permute(0, 3, 1, 2)
                gt = gt_images[view_idx : view_idx + 1]
                mask = None if masks_float is None else masks_float[view_idx : view_idx + 1]
                per_view_losses.append(
                    self.loss_fn(
                        pred,
                        gt,
                        mask,
                        gaussian_params=gaussian_params_view,
                        gaussian_3d=gaussian_3d_view.unsqueeze(0),
                    )
                )
        del local_feats
        del local_frames
        del z_id

        loss_dict = {
            key: torch.stack([ld[key] for ld in per_view_losses]).mean()
            for key in per_view_losses[0]
        }

        del gaussian_3d
        return loss_dict

    def _maybe_dump_local_feats(
        self,
        subject: str,
        local_feats: torch.Tensor,
        centers2d: torch.Tensor,
        img_size: tuple,
        feat_size,
        local_feats_prefusion: torch.Tensor,
        view_weights: torch.Tensor,
    ) -> None:
        """Save local_feats (post-mode processing) to disk for offline PCA analysis.

        Also saves a companion collision-data file containing per-view Gaussian
        2-D centers in pixel coordinates plus the image/feature-map sizes, used
        by the projection-collision analysis notebook section.

        Enabled only when ``analysis.dump_local_feats: true`` in the config.
        If ``analysis.dump_subject`` is set, only that subject is dumped.
        """
        if not self._dump_local_feats:
            return
        if self._dump_subject is not None and str(subject) != str(self._dump_subject):
            return

        self._dump_dir.mkdir(parents=True, exist_ok=True)

        out_path = self._dump_dir / f"local_feats_{subject}.pt"
        torch.save(local_feats.detach().cpu(), out_path)

        # Companion file for pre/post fusion quality analysis
        fusion_path = self._dump_dir / f"fusion_data_{subject}.pt"
        torch.save(
            {
                "local_feats_prefusion": local_feats_prefusion.detach().cpu(),  # (B, N, C)
                "view_weights": view_weights.detach().cpu(),  # (B, N)
                "local_feats_postfusion": local_feats.detach().cpu(),  # (1, N, C) fused or (B, N, C) per-view
                "centers2d": centers2d.detach().cpu(),  # (B, N, 2)
            },
            fusion_path,
        )

        # Companion file for projection-collision analysis
        H, W = int(img_size[0]), int(img_size[1])
        Hf, Wf = int(feat_size[-2]), int(feat_size[-1])
        collision_path = self._dump_dir / f"collision_data_{subject}.pt"
        torch.save(
            {
                "centers2d": centers2d.detach().cpu(),  # (B, N, 2) pixel coords
                "img_size": (H, W),
                "feat_size": (Hf, Wf),
            },
            collision_path,
        )

    def freeze_encoder(self):
        for p in self.identity_encoder.parameters():
            p.requires_grad = False

    def configure_optimizers(self):
        base_lr = float(self.hparams.lr)              # decoder lr
        bb_mult = float(getattr(self.hparams, "bb_lr_mult", 0.1))
        wd = float(self.hparams.wd)

        def is_no_decay(name, param):
            if param.ndim == 1:
                return True
            if name.endswith(".bias"):
                return True
            n = name.lower()
            if "bn" in n or "norm" in n:
                return True
            return False

        dec_decay, dec_no_decay = [], []
        for mod in (self.decoder, self.view_fusion):
            if mod is None:
                continue
            for n, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                (dec_no_decay if is_no_decay(n, p) else dec_decay).append(p)

        bb_decay, bb_no_decay = [], []
        for n, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            (bb_no_decay if is_no_decay(n, p) else bb_decay).append(p)

        optimizer = torch.optim.AdamW(
            [
                {
                    "name": "decoder_decay",
                    "params": dec_decay,
                    "lr": base_lr,
                    "weight_decay": wd,
                },
                {
                    "name": "decoder_no_decay",
                    "params": dec_no_decay,
                    "lr": base_lr,
                    "weight_decay": 0.0,
                },
                {
                    "name": "backbone_decay",
                    "params": bb_decay,
                    "lr": base_lr * bb_mult,
                    "weight_decay": wd,
                },
                {
                    "name": "backbone_no_decay",
                    "params": bb_no_decay,
                    "lr": base_lr * bb_mult,
                    "weight_decay": 0.0,
                },
            ],
            betas=tuple(self.hparams.betas) if isinstance(self.hparams.betas, (list, tuple)) else (0.9, 0.99),
            eps=float(self.hparams.eps),
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_ratio = float(self.hparams.warmup_ratio)
        warmup_ratio = min(1.0, max(0.0, warmup_ratio))
        warmup_steps = int(warmup_ratio * max(1, total_steps))
        min_lr_ratio = float(getattr(self.hparams, "min_lr_ratio", 0.05))

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            denom = max(1, total_steps - warmup_steps)
            progress = (step - warmup_steps) / denom
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        param_groups = self.trainer.optimizers[0].param_groups

        decoder_lrs = [
            group["lr"]
            for group in param_groups
            if str(group.get("name", "")).startswith("decoder")
        ]
        backbone_lrs = [
            group["lr"]
            for group in param_groups
            if str(group.get("name", "")).startswith("backbone")
        ]

        decoder_lr = float(decoder_lrs[0]) if decoder_lrs else float(param_groups[0]["lr"])
        backbone_lr = (
            float(backbone_lrs[0])
            if backbone_lrs
            else float(param_groups[min(2, len(param_groups) - 1)]["lr"])
        )

        self.log("train/lr_decoder", decoder_lr, prog_bar=True)
        self.log("train/lr_backbone", backbone_lr, prog_bar=False)

    def process_input(self, batch):
        """Extract tensors from the dataset batch and normalize shape/device.

        Batch : Dict[str, Any]
            "images_float": images_float,
            "images_uint8": images_uint8,
            "masks_float": masks_float,
            "subject": str,
            "view_names": List[str]
            "vertices3d": Optional[torch.FloatTensor] [Nv, 3]
            "vertices2d": Optional[torch.FloatTensor] [V, Nv, 2]

        Returns
        -------
        img_float : torch.Tensor
            Float image tensor used for feature extraction, shape (B,C,H,W) on self.device.
        img_uint8 : torch.Tensor
            The original uint8 image used by the detector, shape (B,C,H,W) on self.device.
        masks_float : torch.Tensor
            Float mask tensor, shape (B,1,H,W) on self.device.
        (B, H, W) : tuple[int,int,int]
            Spatial dims extracted from `img_float`.
        subject : str
        view_names : List[str]
        vertices3d : torch.Tensor
            SMPLX 3D vertices, shape (Nv, 3) on self.device. Empty (0,3) if unavailable.
        vertices2d : torch.Tensor
            SMPLX 2D projections per view, shape (B, Nv, 2) on self.device. Empty (B,0,2) if unavailable.
        """
        assert (
            "images_float" in batch and "images_uint8" in batch
        ), "Batch missing 'images_float' or 'images_uint8' key"

        img_float = batch["images_float"]
        # If dataset wrapped a singleton batch dim, unwrap it
        if img_float.ndim == 5 and img_float.shape[0] == 1:
            img_float = img_float[0]

        # Optional uint8 input for detectors
        img_uint8 = batch["images_uint8"]
        if img_uint8.ndim == 5 and img_uint8.shape[0] == 1:
            img_uint8 = img_uint8[0]

        # Extract masks
        masks_float = batch.get("masks_float", None)
        if masks_float is not None:
            if masks_float.ndim == 5 and masks_float.shape[0] == 1:
                masks_float = masks_float[0]
        else:
            # If masks are not in batch, return None and let the loss function handle it
            B, _, H, W = img_float.shape
            masks_float = None

        B, _, H, W = img_float.shape

        # Normalize subject (may be a str or a singleton list/tuple)
        subject = batch.get("subject", None)
        if isinstance(subject, (list, tuple)):
            subject = subject[0]

        # Normalize view_names to a flat List[str]
        view_names = batch.get("view_names", None)
        if isinstance(view_names, (list, tuple)):
            # DataLoader with batch_size=1 may wrap as [List[str]]
            if len(view_names) == 1 and isinstance(view_names[0], (list, tuple)):
                view_names = list(view_names[0])
            else:
                # Flatten possible tuples like ('front',) and ensure str
                view_names = [
                    vn[0] if isinstance(vn, (list, tuple)) else vn for vn in view_names
                ]
        # Else leave None as-is

        # Extract SMPLX 3D vertices (shape [Nv, 3] or empty [0, 3])
        vertices3d = batch.get("vertices3d", None)
        if vertices3d is not None:
            if vertices3d.ndim == 3 and vertices3d.shape[0] == 1:
                vertices3d = vertices3d[0]
            # vertices3d = vertices3d.to(self.device)
        else:
            vertices3d = torch.empty(0, 3) #, dtype=torch.float32, device=self.device)

        # Extract SMPLX 2D projections (shape [B, Nv, 2] or empty [B, 0, 2])
        vertices2d = batch.get("vertices2d", None)
        if vertices2d is not None:
            if vertices2d.ndim == 4 and vertices2d.shape[0] == 1:
                vertices2d = vertices2d[0]
            # vertices2d = vertices2d.to(self.device)
        else:
            vertices2d = torch.empty(B, 0, 2) #, dtype=torch.float32, device=self.device)

        return img_float, img_uint8, masks_float, (B, H, W), subject, view_names, vertices3d, vertices2d

    def _is_test_render_batch(self, subject: str) -> bool:
        try:
            return int(subject) in self._test_renders
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _sparse_values_to_image(values: torch.Tensor, centers2d: torch.Tensor, img_h: int, img_w: int) -> np.ndarray:
        vals = values.detach().float()
        ctr = centers2d.detach().float()
        x = torch.round(ctr[:, 0]).long().clamp(0, img_w - 1)
        y = torch.round(ctr[:, 1]).long().clamp(0, img_h - 1)
        idx = y * img_w + x
        flat = torch.zeros(img_h * img_w, dtype=vals.dtype, device=vals.device)
        if hasattr(torch.Tensor, "scatter_reduce_"):
            flat.scatter_reduce_(0, idx, vals, reduce="amax", include_self=True)
        else:
            for i in range(idx.numel()):
                flat[idx[i]] = torch.maximum(flat[idx[i]], vals[i])
        image = flat.view(img_h, img_w)
        vmax = image.max()
        if torch.isfinite(vmax) and float(vmax) > 0.0:
            image = image / vmax
        image_u8 = (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        return image_u8

    def _maybe_dump_view_weight_images(
        self,
        subject: str,
        view_names,
        stage: str,
        batch_idx: int,
        img_size: tuple[int, int],
    ) -> None:
        if not self._dump_view_weight_images:
            return
        if not self._is_test_render_batch(subject):
            return
        debug_data = getattr(self.avatar_estimator, "last_weight_debug", None)
        if not debug_data:
            return
        angle_weight = debug_data.get("angle_weight", None)
        visibility = debug_data.get("visibility", None)
        centers2d = debug_data.get("centers2d", None)
        if angle_weight is None or visibility is None or centers2d is None:
            return

        img_h, img_w = int(img_size[0]), int(img_size[1])
        view_names_list = view_names if isinstance(view_names, (list, tuple)) else []
        out_dir = self._view_weight_debug_dir / str(subject) / stage
        out_dir.mkdir(parents=True, exist_ok=True)

        num_views = min(angle_weight.shape[0], centers2d.shape[0])
        for view_idx in range(num_views):
            view_name = str(view_names_list[view_idx]) if view_idx < len(view_names_list) else f"view{view_idx}"
            angle_img = self._sparse_values_to_image(
                angle_weight[view_idx], centers2d[view_idx], img_h=img_h, img_w=img_w
            )
            vis_img = self._sparse_values_to_image(
                visibility[view_idx], centers2d[view_idx], img_h=img_h, img_w=img_w
            )
            angle_path = out_dir / f"step{self.global_step:07d}_batch{batch_idx:05d}_{view_name}_angle.png"
            vis_path = out_dir / f"step{self.global_step:07d}_batch{batch_idx:05d}_{view_name}_visibility.png"
            Image.fromarray(angle_img, mode="L").save(angle_path)
            Image.fromarray(vis_img, mode="L").save(vis_path)

    def _log_loss_dict(self, loss_dict: Dict[str, torch.Tensor], stage: str) -> None:
        should_log = (stage == "train" and self._log_train_losses) or (
            stage == "val" and self._log_val_losses
        )
        if not should_log:
            return

        tracked = set(self._tracked_loss_names) if self._tracked_loss_names else None
        for key, value in loss_dict.items():
            if tracked is not None and key not in tracked:
                continue
            self.log(f"{stage}/{key}", value, prog_bar=(stage == "val" and key == "loss"))

    def _log_render_output(
        self,
        rendered_imgs: torch.Tensor,
        subject: str,
        view_name: str,
        stage: str,
    ) -> None:
        """Log rendered images online instead of writing to the local output/ folder."""
        if not self._log_render_outputs_online:
            return
        if stage not in self._render_log_stages:
            return
        if not self._is_test_render_batch(subject):
            return
        if not hasattr(self.logger, "experiment") or self.logger.experiment is None:
            return

        try:
            import wandb
        except Exception:
            return

        render_np = rendered_imgs[0].detach().clamp(0, 1).cpu().numpy()
        safe_subject = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(subject))
        safe_view = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(view_name))
        key = f"render/{stage}/{safe_subject}_{safe_view}"
        self.logger.experiment.log(
            {key: wandb.Image(render_np), "global_step": int(self.global_step)}
        )




    
