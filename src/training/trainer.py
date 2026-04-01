from typing import Any, Dict, Optional
from contextlib import nullcontext
from pathlib import Path
import math
import random
import re
import torch
from torch.utils.data import DataLoader
import lightning as L
from encoder.feature_extractor import FeatureExtractor
from encoder.gaussian_estimator import AvatarGaussianEstimator
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
        self.template = AvatarTemplate()
        self.backbone = backbone_adapter
        self.avatar_estimator = AvatarGaussianEstimator(self.template)
        self.identity_encoder = identity_encoder
        self.decoder = decoder
        self.renderer = renderer
        self.loss_fn = LossFunctions()    
        self.train_decoder_only = train_decoder_only
        train_cfg = get_config().get("train", {})

        two_phase_cfg = train_cfg.get("two_phase", {})
        self.total_steps_cap = int(train_cfg.get("max_steps", 25000))
        self.phase1_steps = int(two_phase_cfg.get("phase1_steps", 6000))
        self.phase1_steps = max(0, min(self.phase1_steps, self.total_steps_cap))
        self.phase2_target_views = int(two_phase_cfg.get("phase2_target_views", 2))
        self.phase2_target_views = max(1, self.phase2_target_views)
        self.phase2_target_weight_start = float(
            two_phase_cfg.get("phase2_target_weight_start", 0.3)
        )
        self.phase2_target_weight_end = float(
            two_phase_cfg.get("phase2_target_weight_end", 0.8)
        )
        self.phase2_curriculum_start = float(
            two_phase_cfg.get("phase2_curriculum_start", 0.35)
        )
        self.warmup_scale_reg_mult = float(
            two_phase_cfg.get("warmup_scale_reg_multiplier", 2.5)
        )
        self.early_phase2_scale_reg_mult = float(
            two_phase_cfg.get("early_phase2_scale_reg_multiplier", 1.5)
        )
        self.phase2_scale_reg_relax_progress = float(
            two_phase_cfg.get("phase2_scale_reg_relax_progress", 0.5)
        )
        self.backbone_lr_phase2_mult = float(
            two_phase_cfg.get("phase2_backbone_lr_mult", 0.1)
        )

        # Keep backbone frozen during warm-up and decoder-only mode.
        if hasattr(self.backbone, "set_resnet_fpn_frozen"):
            self.backbone.set_resnet_fpn_frozen(self.train_decoder_only or self.phase1_steps > 0)

        # Read optimizer & scheduler settings from config and save as hyperparameters
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
        self._phase2_backbone_activated = False

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss_dict = self.shared_step(batch=batch, batch_idx=batch_idx, stage="train")
        self._log_loss_dict(loss_dict, stage="train")
        return loss_dict["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        val_loss_dict = self.shared_step(batch=batch, batch_idx=batch_idx, stage="val")
        self._log_loss_dict(val_loss_dict, stage="val")
        return val_loss_dict["loss"]

    def _in_phase2(self) -> bool:
        return int(self.global_step) >= int(self.phase1_steps)

    def _phase2_progress(self) -> float:
        if not self._in_phase2():
            return 0.0
        denom = max(1, int(self.total_steps_cap) - int(self.phase1_steps))
        return min(1.0, max(0.0, (int(self.global_step) - int(self.phase1_steps)) / denom))

    def _current_target_weight(self) -> float:
        p = self._phase2_progress()
        return self.phase2_target_weight_start + (
            self.phase2_target_weight_end - self.phase2_target_weight_start
        ) * p

    def _current_scale_reg_weight(self) -> float:
        base = float(self.loss_fn.weight_scale_reg)
        if not self._in_phase2():
            return base * self.warmup_scale_reg_mult
        relax_gate = min(1.0, self._phase2_progress() / max(1e-6, self.phase2_scale_reg_relax_progress))
        scale_mult = self.early_phase2_scale_reg_mult + (1.0 - self.early_phase2_scale_reg_mult) * relax_gate
        return base * scale_mult

    def _select_source_and_targets(self, view_names, stage: str):
        if view_names is None:
            return 0, []
        n_views = len(view_names)
        if n_views == 0:
            return 0, []
        source_idx = 0 if stage != "train" else random.randrange(n_views)
        candidates = [i for i in range(n_views) if i != source_idx]
        if not candidates:
            return source_idx, []

        if self._in_phase2():
            progress = self._phase2_progress()
            source_name = view_names[source_idx]
            index_map = {name: idx for idx, name in enumerate(["front", "left", "back", "right"])}
            src_ring = index_map.get(str(source_name).lower(), None)
            if src_ring is not None:
                def ring_dist(idx):
                    tgt_ring = index_map.get(str(view_names[idx]).lower(), src_ring)
                    d = abs(tgt_ring - src_ring)
                    return min(d, 4 - d)

                near = [i for i in candidates if ring_dist(i) <= 1]
                far = [i for i in candidates if ring_dist(i) > 1]
                if progress < self.phase2_curriculum_start and near:
                    pool = near
                elif far:
                    near_quota = max(0, self.phase2_target_views - 1)
                    if stage == "train":
                        random.shuffle(near)
                        random.shuffle(far)
                    pool = near[:near_quota] + far
                else:
                    pool = candidates
            else:
                pool = candidates
            if stage == "train":
                random.shuffle(pool)
            targets = pool[: min(self.phase2_target_views, len(pool))]
            return source_idx, targets

        return source_idx, []

    def shared_step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> torch.Tensor:
        # Extract data from batch
        img_float, img_uint8, masks_float, (B, H, W), subject, view_names, vertices3d, vertices2d = self.process_input(batch)
        source_idx, target_indices = self._select_source_and_targets(
            view_names=view_names, stage=stage
        )
        supervise_indices = [source_idx]
        if self._in_phase2():
            supervise_indices.extend(target_indices)

        gt_images = img_float

        grad_ctx = torch.inference_mode() if self.train_decoder_only else nullcontext()
        with grad_ctx:
            feats = self.backbone.extract_feature_map(
                image=img_float[source_idx : source_idx + 1], use_half=True
            )
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
        assert B_feats == 1, "Single-view training expects one source feature map."

        with grad_ctx:
            if self.use_identity_encoder:
                z_id = self.identity_encoder(feature_map=feat_for_id)  # (1, D)
            else:
                z_id = None

            vertices2d_source = vertices2d[source_idx : source_idx + 1]
            local_feats, view_weights, gaussian_3d, centers2d = (
                self.avatar_estimator.feature_sample_with_visibility(
                    feats, vertices3d, vertices2d_source, img_shape=(H, W)
                )
            )  # (1, N, C_local), (1, N), (1, N, 3), (1, N, 2)
            local_frames = self.avatar_estimator.compute_gaussian_local_frames(
                vertices3d, device=gaussian_3d.device, batch_size=1
            )  # (1, N, 3, 3)
        local_feats_prefusion = local_feats
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

        if not self._shape_debug_logged:
            assert local_feats.shape[-1] == self.decoder.in_dim, (
                f"Local feature dim {local_feats.shape[-1]} != decoder.in_dim {self.decoder.in_dim}"
            )
            self._shape_debug_logged = True

        # Free large intermediates early to reduce peak VRAM before decoding
        del feats
        del feat_for_id
        del img_uint8

        local_feats_source = local_feats
        z_id_source = None
        if z_id is not None:
            z_id_source = z_id

        gaussian_params = self.decoder(local_feats_source, z_id_source)
        gaussian_3d_shared = gaussian_3d[0]
        frame_idx = 0
        offset_local = gaussian_params.get("offset", None)
        if offset_local is not None:
            gaussian_3d_shared = gaussian_3d_shared + torch.einsum(
                "nij,nj->ni", local_frames[frame_idx], offset_local
            )

        weight_overrides = {"weight_scale_reg": self._current_scale_reg_weight()}
        per_view_losses = []
        per_view_psnr = []
        for view_idx in supervise_indices:
            view_name = view_names[view_idx]
            rendered_imgs = self.renderer.render(
                gaussian_3d=gaussian_3d_shared,
                gaussian_params=gaussian_params,
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
            mse = torch.mean((pred - gt) ** 2).clamp_min(1e-8)
            per_view_psnr.append(10.0 * torch.log10(1.0 / mse))
            per_view_losses.append(
                self.loss_fn(
                    pred,
                    gt,
                    mask,
                    gaussian_params=gaussian_params,
                    gaussian_3d=gaussian_3d_shared.unsqueeze(0),
                    weight_overrides=weight_overrides,
                )
            )

        src_loss_dict = per_view_losses[0]
        if len(per_view_losses) > 1:
            tgt_loss_dict = {
                key: torch.stack([ld[key] for ld in per_view_losses[1:]]).mean()
                for key in per_view_losses[0]
            }
        else:
            tgt_loss_dict = {key: torch.zeros_like(val) for key, val in src_loss_dict.items()}

        w_tgt = self._current_target_weight() if self._in_phase2() else 0.0
        w_src = 1.0

        loss_dict = {
            key: (w_src * src_loss_dict[key] + w_tgt * tgt_loss_dict[key]) / max(1.0, w_src + w_tgt)
            for key in src_loss_dict
        }
        loss_dict["src_loss"] = src_loss_dict["loss"]
        loss_dict["tgt_loss"] = tgt_loss_dict["loss"]
        loss_dict["src_psnr"] = per_view_psnr[0]
        if len(per_view_psnr) > 1:
            loss_dict["tgt_psnr"] = torch.stack(per_view_psnr[1:]).mean()
        else:
            loss_dict["tgt_psnr"] = torch.zeros_like(per_view_psnr[0])
        loss_dict["w_tgt"] = torch.tensor(w_tgt, device=loss_dict["loss"].device)
        loss_dict["phase"] = torch.tensor(2 if self._in_phase2() else 1, device=loss_dict["loss"].device)
        del local_feats
        del local_frames
        del z_id

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
        for n, p in self.decoder.named_parameters():
            if not p.requires_grad:
                continue
            (dec_no_decay if is_no_decay(n, p) else dec_decay).append(p)

        bb_decay, bb_no_decay = [], []
        for n, p in self.backbone.named_parameters():
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
        self.log("train/w_tgt", float(self._current_target_weight()) if self._in_phase2() else 0.0, prog_bar=False)

    def on_train_batch_start(self, batch, batch_idx):
        if self._phase2_backbone_activated:
            return
        if not self._in_phase2():
            return
        if hasattr(self.backbone, "set_resnet_fpn_frozen"):
            self.backbone.set_resnet_fpn_frozen(False)
        optimizer = self.trainer.optimizers[0]
        decoder_lr = None
        for group in optimizer.param_groups:
            group_name = str(group.get("name", ""))
            if group_name.startswith("decoder") and decoder_lr is None:
                decoder_lr = float(group["lr"])
        if decoder_lr is None:
            decoder_lr = float(self.hparams.lr)
        for group in optimizer.param_groups:
            group_name = str(group.get("name", ""))
            if group_name.startswith("backbone"):
                group["lr"] = decoder_lr * self.backbone_lr_phase2_mult
        self._phase2_backbone_activated = True

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
        return int(subject) in self._test_renders

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




    
