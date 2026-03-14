from typing import Any, Dict, Optional
from contextlib import nullcontext
import json
import logging
from pathlib import Path
import math
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
        self._logger = logging.getLogger("train")
        self._profile_gpu = bool(get_config().get("train", {}).get("profile_gpu", False))
        self.use_identity_encoder = bool(
            get_config().get("identity_encoder", {}).get("use_flag", True)
        )
        self.num_views = int(get_config().get("data", {}).get("num_views", 1))
        if self.num_views not in (1, 4):
            raise ValueError(f"data.num_views must be 1 or 4, got {self.num_views}")
        aug_cfg = get_config().get("data", {}).get("augmentation", {})
        self._save_augmented_inputs = bool(aug_cfg.get("save_preview", False))
        self._save_augmented_once_per_subject = bool(aug_cfg.get("save_once_per_subject", True))
        self._saved_augmented_subjects = set()
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
        wd = float(train_cfg.get("weight_decay", 0.0))
        betas = train_cfg.get("betas", [0.9, 0.99])
        eps = float(train_cfg.get("eps", 1e-8))
        warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
        scheduler_name = train_cfg.get("scheduler", "cosine")
        # Persist to hparams for use in configure_optimizers
        self.save_hyperparameters({
            "lr": lr,
            "wd": wd,
            "betas": betas,
            "eps": eps,
            "warmup_ratio": warmup_ratio,
            "scheduler": scheduler_name,
        })

        # If True, freeze all parameters except the decoder's so only decoder gets updated.
        if self.train_decoder_only:
            self.freeze_encoder()
            self._logger.info("Frozen encoder parameters.")

        # Feature-dump settings for PCA analysis
        analysis_cfg = get_config().get("analysis", {})
        self._dump_local_feats = bool(analysis_cfg.get("dump_local_feats", False))
        self._dump_subject = analysis_cfg.get("dump_subject", None)  # None → dump all
        self._dump_dir = Path(str(analysis_cfg.get("dump_dir", "output/analysis")))
            
        self._shape_debug_logged = False

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss_dict = self.shared_step(batch=batch, batch_idx=batch_idx, stage="train")
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v)
        return loss_dict["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        val_loss_dict = self.shared_step(batch=batch, batch_idx=batch_idx, stage="val")
        for k,v in val_loss_dict.items():
            self.log(f"val/{k}", v, prog_bar=(k=="loss"))
        return val_loss_dict["loss"]

    def shared_step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> torch.Tensor:
        # Extract data from batch
        img_float, img_uint8, masks_float, (B, H, W), subject, view_names, vertices3d, vertices2d, augmentation_info = self.process_input(batch)
        if stage == "train":
            self._logger.info(f"Processing subject: {subject}, views: {view_names}")
            if self._is_test_render_batch(subject):
                self._maybe_save_augmented_inputs(subject, view_names, img_float, augmentation_info)

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
                    feats, vertices3d, vertices2d, img_shape=(H, W)
                )
            )  # (B, N, C_local), (B, N), (B, N, 3), (B, N, 2)
        local_feats_prefusion = local_feats
        # Use all views for supervision; `data.num_views` only controls
        # whether decoder input is per-view (1) or fused (4).
        if self.num_views == 4:
            weights = view_weights.clamp_min(0.0)
            weights_sum = weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
            weights = weights / weights_sum
            local_feats = (local_feats * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
            gaussian_3d_decode = gaussian_3d[0:1]
            z_id_decode = None if z_id is None else z_id.mean(dim=0, keepdim=True)
        else:
            gaussian_3d_decode = gaussian_3d
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

        if not self._shape_debug_logged:
            if isinstance(feats, dict):
                fmap_shapes = {k: tuple(v.shape) for k, v in feats.items()}
                self._logger.info(f"FPN feature map shapes: {fmap_shapes}")
                self._logger.info(
                    f"Per-level sampled feature shapes: {self.avatar_estimator.last_sampled_level_shapes}"
                )
            self._logger.info(f"Concatenated local_feats shape: {tuple(local_feats.shape)}")
            self._logger.info(f"Decoder expected input dim: {self.decoder.in_dim}")
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
        save_path = (
            Path(get_config().get("render", {}).get("save_path", "output"))
            / subject
        ) if self._is_test_render_batch(subject) else None

        if self.num_views == 4:
            gaussian_params_fused = self.decoder(local_feats, z_id_decode)
            gaussian_3d_fused = gaussian_3d_decode[0]

            for view_idx, view_name in enumerate(view_names):
                rendered_imgs = self.renderer.render(
                    gaussian_3d=gaussian_3d_fused,
                    gaussian_params=gaussian_params_fused,
                    view_name=view_name,
                    save_folder_path=save_path,
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

                rendered_imgs = self.renderer.render(
                    gaussian_3d=gaussian_3d_view,
                    gaussian_params=gaussian_params_view,
                    view_name=view_name,
                    save_folder_path=save_path,
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
        self._logger.info(f"Dumped local_feats for subject {subject} → {out_path}")

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
        self._logger.info(f"Dumped fusion_data for subject {subject} → {fusion_path}")

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
        self._logger.info(f"Dumped collision_data for subject {subject} → {collision_path}")

    def freeze_encoder(self):
        for p in self.identity_encoder.parameters():
            p.requires_grad = False

    def configure_optimizers(self):
        trainable_params = list(self.decoder.parameters())
        if not self.train_decoder_only:
            trainable_params += [
                p for p in self.backbone.parameters() if p.requires_grad
            ]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.wd),
            betas=tuple(self.hparams.betas) if isinstance(self.hparams.betas, (list, tuple)) else (0.9, 0.99),
            eps=float(self.hparams.eps),
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(float(self.hparams.warmup_ratio) * max(1, total_steps))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            # cosine decay to zero
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=True)

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
        augmentation_info : Dict[str, Any]
            Metadata describing sampled augmentations (if enabled by dataset).
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

        augmentation_info = batch.get("augmentation_info", {})
        if isinstance(augmentation_info, (list, tuple)) and len(augmentation_info) == 1 and isinstance(augmentation_info[0], dict):
            augmentation_info = augmentation_info[0]

        return img_float, img_uint8, masks_float, (B, H, W), subject, view_names, vertices3d, vertices2d, augmentation_info

    def _is_test_render_batch(self, subject: str) -> bool:
        test_renders = [0, 7, 50, 100, 200, 350]
        return int(subject) in test_renders

    @staticmethod
    def _to_json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): NlfGaussianModel._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [NlfGaussianModel._to_json_safe(v) for v in value]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        try:
            import numpy as _np
            if isinstance(value, (_np.generic,)):
                return value.item()
            if isinstance(value, _np.ndarray):
                return value.tolist()
        except Exception:
            pass
        if isinstance(value, Path):
            return str(value)
        return value

    def _maybe_save_augmented_inputs(
        self,
        subject: Any,
        view_names: Any,
        images_float: torch.Tensor,
        augmentation_info: Dict[str, Any],
    ) -> None:
        if not self._save_augmented_inputs:
            return

        subject_str = str(subject)
        if self._save_augmented_once_per_subject and subject_str in self._saved_augmented_subjects:
            return

        output_root = Path(get_config().get("render", {}).get("save_path", "output"))
        out_dir = output_root / "augmented_inputs" / subject_str
        out_dir.mkdir(parents=True, exist_ok=True)

        names = view_names if isinstance(view_names, list) else []
        for i in range(images_float.shape[0]):
            vn = names[i] if i < len(names) else f"view{i}"
            arr = (
                images_float[i].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
            ).round().astype("uint8")
            from PIL import Image
            Image.fromarray(arr, mode="RGB").save(out_dir / f"{subject_str}_{vn}_aug.png")

        payload = {
            "subject": subject_str,
            "view_names": self._to_json_safe(names),
            "augmentation": self._to_json_safe(augmentation_info if isinstance(augmentation_info, dict) else {}),
        }
        with open(out_dir / "augmentation_info.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        self._saved_augmented_subjects.add(subject_str)
        self._logger.info(f"Saved augmented input previews to {out_dir}")




    
