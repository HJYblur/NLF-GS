from typing import Any, Dict, Optional
from contextlib import nullcontext
import json
import logging
from pathlib import Path
import os
import math
import torch
from torch.utils.data import DataLoader
import lightning as L
from encoder.nlf_backbone_adapter import NLFBackboneAdapter
from encoder.gaussian_estimator import AvatarGaussianEstimator
from encoder.identity_encoder import IdentityEncoder
from encoder.avatar_template import AvatarTemplate
from decoder.gaussian_decoder import GaussianDecoder
from render.gaussian_renderer import GsplatRenderer
from training.losses import LossFunctions
from avatar_utils.ply_loader import reconstruct_gaussian_avatar_as_ply
from avatar_utils.config import get_config


class NlfGaussianModel(L.LightningModule):
    def __init__(
        self,
        backbone_adapter: NLFBackboneAdapter,
        identity_encoder: IdentityEncoder,
        decoder: GaussianDecoder,
        renderer: GsplatRenderer,
        train_decoder_only: bool = True,
    ):
        super().__init__()
        self._logger = logging.getLogger("train")
        self.debug = bool(get_config().get("sys", {}).get("debug", False))
        self._profile_gpu = bool(get_config().get("train", {}).get("profile_gpu", False))
        self.use_identity_encoder = bool(
            get_config().get("identity_encoder", {}).get("use_flag", True)
        )
        self.num_views = int(get_config().get("data", {}).get("num_views", 1))
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
            
        self._logger.info(f"Debug mode: {self.debug}")

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
            if self.debug:
                feats = self.load_debug_feats(img_float, img_uint8)[0]
            else:
                # Process backbone one view at a time to avoid holding B full-res
                # feature maps on GPU simultaneously (saves ~(B-1)/B of backbone VRAM).
                feat_list = []
                for v_idx in range(B):
                    f_v = self.backbone.extract_feature_map(
                        image=img_float[v_idx : v_idx + 1], use_half=True
                    )
                    feat_list.append(f_v)
                feats = torch.cat(feat_list, dim=0)  # (B, C, Hf, Wf)
                del feat_list
            # # Stash GT images on CPU while encoder/decoder run; bring back for loss later
            gt_images = img_float
            del img_float
            
        # Normalize backbone features per spatial location across channels.
        feats = torch.nn.functional.normalize(feats.float(), dim=1, eps=1e-6).to(feats.dtype)

        """
        Encode:
        z_id: Identity Latent Vector (B, D)
        local_feats: Local Features sampled at Gaussian centers (B, N, C_local)
        gaussian_3d: Gaussian 3D Coordinates (B, N, 3)
        """
        B_feats, C_local, Hf, Wf = feats.shape
        assert B == B_feats, "Batch size mismatch between image and features"

        with grad_ctx:
            if self.use_identity_encoder:
                z_id = self.identity_encoder(feature_map=feats)  # (1, D)
            else:
                z_id = None

            local_feats, view_weights, gaussian_3d = (
                self.avatar_estimator.feature_sample_with_visibility(
                    feats, vertices3d, vertices2d, img_shape=(H, W)
                )
            )  # (B, N, C_local), (B, N), (B, N, 3)
            
        # self.debug3d(gaussian_3d[0], subject)

        if local_feats.shape[0] > 1:
            weight_sum = view_weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
            local_feats = (local_feats * view_weights.unsqueeze(-1)).sum(
                dim=0, keepdim=True
            ) / weight_sum.unsqueeze(
                -1
            )  # (1, N, C_local)

        # Free large intermediates early to reduce peak VRAM before decoding
        del feats
        del img_uint8

        """
        Decode:
        gaussian_params: Fused gaussian Params(N, C_params)
        """

        gaussian_params = self.decoder(local_feats, z_id)
        # # overwrite randomly with either (1,0,0), (0,1,0), (0,0,1)
        # N, k = gaussian_params['sh'].shape
        # idx = torch.randint(0, k, (N,), device=gaussian_params['sh'].device)
        # gaussian_params['sh'] = torch.nn.functional.one_hot(idx, num_classes=k).float()
        # gaussian_params['alpha'][:] = 1.0

        # Log Gaussian parameter statistics to track which dimensions are active
        if stage == "train" and batch_idx % 10 == 0:
            self._log_gaussian_param_stats(gaussian_params)

        # Debug check:
        if self.debug:
            for k, v in gaussian_params.items():
                self._logger.debug(f"Decoded gaussian_params[{k}] shape: {v.shape}")

        """
        Render and Loss Computation:
        1. For every view, use the gaussian_params and gaussian_3d to reconstruct an avatar
        2. Render from gaussian_params and compute losses
        3. Return a loss with gradient graph for optimizer step
        """

        assert (
            gaussian_3d.shape[0] == self.num_views
        ), "Mismatch between gaussian_3d and num_views"

        if self.debug and stage == "train":
            # Use gaussian_params and gaussian_3ds to generate a .ply file as the reconstruction.
            new_avatar = reconstruct_gaussian_avatar_as_ply(
                xyz=gaussian_3d[0],
                gaussian_params=gaussian_params,
                template=self.template.load_avatar_template(mode="test"),
                output_path=f"output/{subject}/{subject}_debug.ply",
            )

        save_path = (
            Path(get_config().get("render", {}).get("save_path", "output"))
            / subject
        ) if self._is_test_render_batch(subject) else None
        rendered_imgs = self.renderer.render(
            gaussian_3d=gaussian_3d[0],
            gaussian_params=gaussian_params,
            view_name=view_names,
            save_folder_path=save_path,
        )  # (V, H, W, 3)
        
        # Updated condition for proxy loss, we should use real loss for validation as well.
        if stage == "train" and torch.is_grad_enabled() and not rendered_imgs.requires_grad:
            # Renderer returned a non-differentiable tensor while gradients are expected;
            # fall back to proxy loss to keep optimization stable.
            rendered_imgs = None

        # Free combined inputs post-decoding
        del local_feats
        del z_id

        if rendered_imgs is not None:
            pred = rendered_imgs.permute(0, 3, 1, 2)  # (B, 3, H, W)
            gt = gt_images # .to(self.device)  # Move GT back to GPU for loss
            
            # Register hooks to capture gradients on gaussian_params to see which are most affected by loss
            if stage == "train" and batch_idx % 10 == 0:
                self._register_gaussian_param_grad_hooks(gaussian_params, batch_idx)
            
            loss_dict = self.loss_fn(
                pred,
                gt,
                masks_float,
                gaussian_params=gaussian_params,
                gaussian_3d=gaussian_3d,
            )
        else:
            print("ERROROROROROROR SHOULD NEVER APPEAR!!!")
            loss_dict = self._proxy_regularization_loss(gaussian_params)

        del gaussian_3d
        return loss_dict

    def freeze_encoder(self):
        for p in self.identity_encoder.parameters():
            p.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
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

        # # Only move uint8 images to GPU when needed (debug mode); saves ~48 MB
        # debug = bool(get_config().get("sys", {}).get("debug", False))
        # if debug:
        #     img_uint8 = img_uint8.to(self.device)
        # # else: keep on CPU

        # # Move float images to GPU
        # img_float = img_float.to(self.device)

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

        # Pre-check: Set the avatar to be all red
        # mask = (img_float == 0).all(dim=1, keepdim=True)   # (B, 1, H, W)

        # # set masked pixels to red
        # img_float = torch.where(
        #     mask,
        #     torch.tensor([1.0, 0.0, 0.0], device=img_float.device, dtype=img_float.dtype)
        #         .view(1, 3, 1, 1),
        #     img_float
        # )

        # from torchvision.utils import save_image
        # save_image(img_float.clamp(0.0, 1.0), f"debug_input_{subject}.png")
        
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

    def load_debug_feats(self, img_float, img_uint8):
        feats_path = "debug_backbone_features.pt"
        preds_path = "debug_backbone_preds.pt"
        if os.path.exists(feats_path) and os.path.exists(preds_path):
            # Try to load precomputed features and preds for faster debugging
            preds = torch.load(preds_path, map_location=self.device, weights_only=True)
            feats = torch.load(feats_path, map_location=self.device, weights_only=True)
            self._logger.info(
                f"Loaded backbone features from {feats_path} and preds from {preds_path}"
            )
        else:
            feats, preds = self.backbone.detect_with_features(
                image_feature=img_float, frame_batch=img_uint8, use_half=True
            )
            torch.save(feats, feats_path)
            torch.save(preds, preds_path)
            self._logger.info(
                f"Saved backbone features to {feats_path} and preds to {preds_path}"
            )

        try:
            # Save the first sample image to disk for visual inspection.
            from torchvision.utils import save_image

            sample_img = img_float[0].detach().cpu()
            # Clamp in case image values are slightly out of [0,1]
            save_image(sample_img.clamp(0.0, 1.0), "debug_sample.png")
            self._logger.info("Saved input sample to debug_sample.png")
        except Exception as exc:  # pragma: no cover - debugging helper
            self._logger.warning(f"Unable to save sample image: {exc}")

        return feats, preds

    def _proxy_regularization_loss(
        self, gaussian_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """A simple differentiable loss on decoded gaussian parameters.

        This is used when a differentiable renderer is not available (e.g., CPU).
        It regularizes scales and opacities to small values while keeping rotation
        quaternions bounded. Adjust weights as needed.
        """
        loss = torch.tensor(0.0, device=self.device)
        if "scales" in gaussian_params:
            loss = loss + gaussian_params["scales"].pow(2).mean()
        if "alpha" in gaussian_params:
            loss = loss + 0.1 * gaussian_params["alpha"].pow(2).mean()
        if "rotation" in gaussian_params:
            # Encourage unit quaternions (norm ~ 1)
            q = gaussian_params["rotation"]
            loss = loss + 0.1 * (q.norm(dim=-1) - 1.0).pow(2).mean()
        return {"loss": loss}
    
    def _log_gaussian_param_stats(self, gaussian_params: Dict[str, torch.Tensor]):
        """Log stats aligned with decoder output dict: scales, rotation, alpha, sh."""
        stats = {}
        for key in ("scales", "rotation", "alpha", "sh"):
            tensor = gaussian_params.get(key, None)
            if tensor is None or tensor.numel() == 0:
                continue

            stats[f"gaussian/{key}_mean"] = tensor.mean().item()
            stats[f"gaussian/{key}_min"] = tensor.min().item()
            stats[f"gaussian/{key}_max"] = tensor.max().item()

        if stats:
            self.log_dict(stats, on_step=True, on_epoch=False)
    
    def _register_gaussian_param_grad_hooks(self, gaussian_params: Dict[str, torch.Tensor], batch_idx: int):
        """Register backward hooks and log per-output-dict gradient stats (scales/rotation/alpha/sh)."""

        def make_hook(param_name: str):
            def hook(grad: torch.Tensor):
                if grad is None or grad.numel() == 0:
                    return grad

                grad_stats = {
                    f"gaussian_grads/{param_name}_mean": grad.mean().item(),
                    f"gaussian_grads/{param_name}_min": grad.min().item(),
                    f"gaussian_grads/{param_name}_max": grad.max().item(),
                }

                self.log_dict(grad_stats, on_step=True, on_epoch=False)
                return grad

            return hook

        for name in ("scales", "rotation", "alpha", "sh"):
            tensor = gaussian_params.get(name, None)
            if tensor is not None and tensor.requires_grad and tensor.numel() > 0:
                tensor.register_hook(make_hook(name))


    
    def debug3d(self, vertices3d: torch.Tensor, subject:str):
        # Show sample vertices3d images
        from matplotlib import pyplot as plt
        import numpy as np
        pts = np.asarray(vertices3d.cpu())
        assert pts.ndim == 2 and pts.shape[1] == 3, f"Expected (Nv, 3), got {pts.shape}"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="blue", s=10)  # blue dots
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.savefig(f"output/{subject}_vertices3d.png", dpi=200, bbox_inches="tight")
        plt.close()
    
    def debug2d(self, vertices2d:torch.Tensor, subject:str):
        # Show sample vertices2d images
        from matplotlib import pyplot as plt
        import numpy as np
        pts = np.asarray(vertices2d[0].cpu())
        assert pts.ndim == 2 and pts.shape[1] == 2, f"Expected (Nv, 2), got {pts.shape}"

        plt.figure()
        plt.scatter(pts[:, 0], pts[:, 1], c="red", s=10)  # red dots
        plt.axis("equal")  # keep x/y scale the same
        plt.savefig(f"output/{subject}_vertices2d.png", dpi=200, bbox_inches="tight")
        plt.close()