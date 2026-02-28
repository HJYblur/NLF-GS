from typing import Any, Dict, Optional
from contextlib import nullcontext
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
        img_float, img_uint8, (B, H, W), subject, view_names, vertices3d, vertices2d = self.process_input(batch)
        if stage == "train":
            self._logger.info(f"Processing subject: {subject}, views: {view_names}")

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

        num_train_batches = len(self.trainer.datamodule.train_dataloader())
        test_renders = set(
            torch.linspace(0, num_train_batches - 1, steps=10, dtype=torch.long).tolist()
        )
        save_path = (
            Path(get_config().get("render", {}).get("save_path", "output"))
            / subject
        ) if int(subject) in test_renders else None
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
            loss_dict = self.loss_fn(
                pred,
                gt,
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

        return img_float, img_uint8, (B, H, W), subject, view_names, vertices3d, vertices2d

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