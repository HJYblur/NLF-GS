"""Lightning module for NLF-GS: multi-view image → Gaussians → render → loss.

Pipeline stages in ``shared_step`` are marked with section headers:
batch unpacking, backbone features, surface sampling, view fusion,
Gaussian decode, rasterization, and supervision.
"""

from typing import Any, Dict
from contextlib import nullcontext
import re
import torch
import lightning as L
from encoder.feature_extractor import FeatureExtractor
from encoder.gaussian_estimator import AvatarGaussianEstimator
from encoder.learned_view_fusion import LearnedViewFusion
from encoder.avatar_template import AvatarTemplate
from decoder.gaussian_decoder import GaussianDecoder
from render.gaussian_renderer import GsplatRenderer
from training.losses import LossFunctions
from training.nlfgs_training_utils import configure_nlf_gaussian_optimizers, unpack_training_batch
from avatar_utils.config import get_config


class NlfGaussianModel(L.LightningModule):
    """End-to-end NLF-GS training / validation (see module docstring for stage flow)."""

    # -------------------------------------------------------------------------
    # Construction: config, submodules, optimizer hyperparameters, logging flags
    # -------------------------------------------------------------------------
    def __init__(
        self,
        backbone: FeatureExtractor,
        decoder: GaussianDecoder,
        renderer: GsplatRenderer,
        train_decoder_only: bool = True,
    ):
        super().__init__()
        cfg = get_config()
        self.num_views = int(cfg.get("data", {}).get("num_views", 1))
        if not (1 <= self.num_views <= 4):
            raise ValueError(f"data.num_views must be in 1..4, got {self.num_views}")
        fusion_cfg = cfg.get("fusion", {})
        self.fusion_mode = str(fusion_cfg.get("mode", "fixed"))
        if self.num_views > 1 and self.fusion_mode not in ("fixed", "learned"):
            raise ValueError(f"fusion.mode must be 'fixed' or 'learned', got {self.fusion_mode!r}")
        dec_cfg = cfg.get("decoder", {})
        bb = cfg.get("backbone") or {}
        if not isinstance(bb, dict):
            bb = {}
        if bb.get("local_feature_dim") is not None:
            default_in_dim = int(bb["local_feature_dim"])
        else:
            levels = bb.get("fpn_levels") or ["p2", "p3", "p4"]
            ch = int(bb.get("fpn_out_channels", 256))
            default_in_dim = (
                ch * len(levels) if isinstance(levels, (list, tuple)) and len(levels) > 0 else 768
            )
        feat_dim = int(dec_cfg.get("in_dim", default_in_dim))
        fusion_hidden = int(fusion_cfg.get("hidden_dim", dec_cfg.get("hidden", 256)))
        self.view_fusion = (
            LearnedViewFusion(feat_dim=feat_dim, hidden_dim=fusion_hidden)
            if (self.num_views > 1 and self.fusion_mode == "learned")
            else None
        )
        self.template = AvatarTemplate()
        self.backbone = backbone
        self.avatar_estimator = AvatarGaussianEstimator(self.template)
        self.decoder = decoder
        self.renderer = renderer
        self.loss_fn = LossFunctions()
        self.train_decoder_only = train_decoder_only

        # Backbone freeze policy matches decoder-only training (ResNet-FPN).
        if hasattr(self.backbone, "set_resnet_fpn_frozen"):
            self.backbone.set_resnet_fpn_frozen(self.train_decoder_only)

        # Optimizer / scheduler hyperparameters (read in configure_nlf_gaussian_optimizers).
        train_cfg = cfg.get("train", {})
        ablation_epochs = int(train_cfg.get("ablation_epochs", 0))
        lr = float(train_cfg.get("lr", 1e-4))
        wd = float(train_cfg.get("wd", train_cfg.get("weight_decay", 0.01)))
        betas = train_cfg.get("betas", [0.9, 0.99])
        eps = float(train_cfg.get("eps", 1e-8))
        warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
        bb_lr_mult = float(train_cfg.get("bb_lr_mult", 0.1))
        min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))
        scheduler_name = train_cfg.get("scheduler", "cosine")
        self.save_hyperparameters(
            {
                "lr": lr,
                "wd": wd,
                "betas": betas,
                "eps": eps,
                "warmup_ratio": warmup_ratio,
                "bb_lr_mult": bb_lr_mult,
                "min_lr_ratio": min_lr_ratio,
                "scheduler": scheduler_name,
                "ablation_epochs": ablation_epochs,
            }
        )

        # Logging: scalar losses (Lightning) and optional WandB render images.
        self._shape_debug_logged = False
        render_cfg = cfg.get("render", {})
        self._log_render_outputs_online = bool(render_cfg.get("log_online", True))
        self._render_log_stages = set(render_cfg.get("log_stages", ["train", "val"]))
        # WandB RGB logs (`_log_render_output`): `train.test_renders` = subject int IDs, or ``[-1]`` = all subjects.
        raw_tr = train_cfg.get("test_renders")
        self._test_renders_log_all = False
        self._test_renders = set()
        if isinstance(raw_tr, (list, tuple)) and len(raw_tr) > 0:
            ints = [int(x) for x in raw_tr]
            if len(ints) == 1 and ints[0] == -1:
                self._test_renders_log_all = True
            else:
                self._test_renders = set(ints)

    # -------------------------------------------------------------------------
    # Lightning train / val steps
    # -------------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss_dict = self.shared_step(batch=batch, stage="train")
        self._log_loss_dict(loss_dict, stage="train")
        return loss_dict["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        val_loss_dict = self.shared_step(batch=batch, stage="val")
        self._log_loss_dict(val_loss_dict, stage="val")
        return val_loss_dict["loss"]

    def shared_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, torch.Tensor]:
        # --- Batch / data loading (dataset → tensors, one chunk of views = B) ---
        img_float, img_uint8, masks_float, (B, H, W), subject, view_names, vertices3d, vertices2d = (
            unpack_training_batch(batch)
        )

        # Decoder-only mode: no grad through backbone / sampling path (saves VRAM).
        grad_ctx = torch.inference_mode() if self.train_decoder_only else nullcontext()

        # --- Backbone: per-view image → FPN feature maps (concatenated over views) ---
        with grad_ctx:
            feat_list = []
            for v_idx in range(B):
                f_v = self.backbone.extract_feature_map(img_float[v_idx : v_idx + 1])
                feat_list.append(f_v)
            if isinstance(feat_list[0], dict):
                feats = {
                    level: torch.cat([fv[level] for fv in feat_list], dim=0)
                    for level in feat_list[0].keys()
                }
            else:
                feats = torch.cat(feat_list, dim=0)
            del feat_list
            gt_images = img_float
            del img_float

        feat_ref = next(iter(feats.values())) if isinstance(feats, dict) else feats
        assert B == feat_ref.shape[0], "Batch size mismatch between image and features"

        with grad_ctx:
            # --- Sampling: template Gaussians + grid_sample features + view weights / 3D centers ---
            local_feats, view_weights, gaussian_3d, centers2d = (
                self.avatar_estimator.feature_sample_with_visibility(
                    feats,
                    vertices3d,
                    vertices2d,
                    img_shape=(H, W),
                    view_names=view_names,
                )
            )
            local_frames = self.avatar_estimator.compute_gaussian_local_frames(
                vertices3d, device=gaussian_3d.device, batch_size=B
            )
            if local_frames.shape[0] == 1 and B > 1:
                local_frames = local_frames.expand(B, -1, -1, -1)

        # --- View fusion (multi-view): learned MLP or fixed weights → single fused feature row ---
        if self.num_views > 1:
            if self.view_fusion is not None:
                local_feats = self.view_fusion(local_feats, view_weights)
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

        # One-time shape check: sampled feature dim must match decoder MLP input.
        if not self._shape_debug_logged:
            assert local_feats.shape[-1] == self.decoder.in_dim, (
                f"Local feature dim {local_feats.shape[-1]} != decoder.in_dim {self.decoder.in_dim}"
            )
            self._shape_debug_logged = True

        del feats
        del feat_ref
        del img_uint8

        assert gaussian_3d.shape[0] == B, "Mismatch between gaussian_3d and number of views"

        per_view_losses = []

        # --- Decode + render + loss: fused multi-view path vs single-view (no fusion) ---
        if self.num_views > 1:
            # Decode once from fused features; apply local offset in template frame.
            gaussian_params_fused = self.decoder(local_feats)
            gaussian_3d_fused = gaussian_3d_decode[0]
            offset_local = gaussian_params_fused.get("offset", None)
            if offset_local is not None:
                gaussian_3d_fused = gaussian_3d_fused + torch.einsum(
                    "nij,nj->ni", local_frames_decode[0], offset_local
                )

            # Supervise each camera with the same fused Gaussians (different viewmat / K).
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
            # num_views == 1: single input view — no fusion; decode and supervise that view only.
            for view_idx, view_name in enumerate(view_names):
                local_feats_view = local_feats[view_idx : view_idx + 1]
                gaussian_params_view = self.decoder(local_feats_view)
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

        # --- Loss aggregation: mean over views for each loss key ---
        loss_dict = {
            key: torch.stack([ld[key] for ld in per_view_losses]).mean()
            for key in per_view_losses[0]
        }

        del gaussian_3d
        return loss_dict

    # -------------------------------------------------------------------------
    # Training utilities: optimizer, LR logging
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        return configure_nlf_gaussian_optimizers(self)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Log effective LRs for decoder vs backbone param groups (see nlfgs_training_utils).
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

    # -------------------------------------------------------------------------
    # WandB / Lightning logging helpers (scalars + optional render images)
    # -------------------------------------------------------------------------
    def _is_test_render_batch(self, subject: str) -> bool:
        if self._test_renders_log_all:
            return True
        try:
            return int(subject) in self._test_renders
        except (TypeError, ValueError):
            return False

    def _log_loss_dict(self, loss_dict: Dict[str, torch.Tensor], stage: str) -> None:
        """Log every scalar returned in ``loss_dict`` (train and val)."""
        for key, value in loss_dict.items():
            self.log(f"{stage}/{key}", value, prog_bar=(stage == "val" and key == "loss"))

    def _log_render_output(
        self,
        rendered_imgs: torch.Tensor,
        subject: str,
        view_name: str,
        stage: str,
    ) -> None:
        """Push RGB renders to WandB when ``train.test_renders`` lists subject IDs, or ``[-1]`` for every subject."""
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
        # Use WandB step= so we do not log a redundant scalar metric named "global_step".
        self.logger.experiment.log(
            {key: wandb.Image(render_np)}, step=int(self.global_step)
        )

    def on_train_epoch_end(self):
        ablation_epochs = int(getattr(self.hparams, "ablation_epochs", 0))
        if ablation_epochs > 0 and self.current_epoch >= ablation_epochs:
            print(f"Reached epoch {self.current_epoch}, stopping training.")
            import sys

            sys.exit(0)