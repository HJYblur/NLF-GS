import torch
import torch.nn as nn
import torch.nn.functional as F
from avatar_utils.config import load_config


class LossFunctions(nn.Module):
    def __init__(self, weight_rgb=None):
        super().__init__()
        train_cfg = load_config().get("train", {})
        self.weight_rgb = (
            weight_rgb if weight_rgb is not None else float(train_cfg.get("weight_rgb", 1.0))
        )
        self.weight_l1 = float(train_cfg.get("weight_l1", 0.0))
        self.weight_masked_ssim = float(train_cfg.get("weight_masked_ssim", 0.0))
        self.weight_perceptual = float(train_cfg.get("weight_perceptual", 0.0))
        self.weight_scale_reg = float(train_cfg.get("weight_scale_reg", 0.0))
        self.weight_opacity_reg = float(train_cfg.get("weight_opacity_reg", 0.0))
        self.weight_silhouette = float(train_cfg.get("weight_silhouette", 0.0))
        self.weight_offset_reg = float(train_cfg.get("weight_offset_reg", 0.0))
        self.weight_multiview_consistency = float(
            train_cfg.get("weight_multiview_consistency", 0.0)
        )
        # mean(1 - alpha): encourages all Gaussians to be less transparent (0 = off)
        self.weight_opacity_nontransparent = float(
            train_cfg.get("weight_opacity_nontransparent", 0.0)
        )

    def _foreground_mask(self, gt_imgs: torch.Tensor) -> torch.Tensor:
        """Return foreground mask from non-black GT pixels, shape [B,1,H,W]."""
        return (gt_imgs.abs().sum(dim=1, keepdim=True) > 0.0).float()

    def _masked_l1(
        self, pred_imgs: torch.Tensor, gt_imgs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask3 = mask.expand_as(pred_imgs)
        valid = mask3.sum()
        if valid <= 0:
            return torch.zeros((), device=pred_imgs.device)
        return (pred_imgs - gt_imgs).abs().mul(mask3).sum() / valid

    def _masked_ssim(
        self,
        pred_imgs: torch.Tensor,
        gt_imgs: torch.Tensor,
        mask: torch.Tensor,
        kernel_size: int = 11,
    ) -> torch.Tensor:
        """Compute mask-weighted SSIM using a local-window SSIM map."""
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        pad = kernel_size // 2

        mu_x = F.avg_pool2d(pred_imgs, kernel_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(gt_imgs, kernel_size, stride=1, padding=pad)

        sigma_x = (
            F.avg_pool2d(pred_imgs * pred_imgs, kernel_size, stride=1, padding=pad)
            - mu_x * mu_x
        )
        sigma_y = (
            F.avg_pool2d(gt_imgs * gt_imgs, kernel_size, stride=1, padding=pad)
            - mu_y * mu_y
        )
        sigma_xy = (
            F.avg_pool2d(pred_imgs * gt_imgs, kernel_size, stride=1, padding=pad)
            - mu_x * mu_y
        )

        ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        ssim_map = ssim_map.mean(dim=1, keepdim=True)

        masked_sum = (ssim_map * mask).sum()
        valid = mask.sum()
        if valid <= 0:
            return ssim_map.mean()
        return masked_sum / valid
    
    def silhouette_loss(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """L2 loss comparing predicted silhouette with ground truth mask."""
        return F.mse_loss(pred_masks, gt_masks)

    def _masked_multiscale_perceptual(
        self,
        pred_imgs: torch.Tensor,
        gt_imgs: torch.Tensor,
        mask: torch.Tensor,
        scales=(1, 2, 4),
    ) -> torch.Tensor:
        """Lightweight perceptual loss via masked multi-scale image + gradient differences."""
        loss = torch.zeros((), device=pred_imgs.device)
        for s in scales:
            if s > 1:
                pred_s = F.avg_pool2d(pred_imgs, s, s)
                gt_s = F.avg_pool2d(gt_imgs, s, s)
                mask_s = F.avg_pool2d(mask, s, s)
                mask_s = (mask_s > 0.5).float()
            else:
                pred_s, gt_s, mask_s = pred_imgs, gt_imgs, mask

            # Intensity difference
            l_int = self._masked_l1(pred_s, gt_s, mask_s)

            # Edge/structure difference (Sobel-like finite differences)
            pred_dx = pred_s[:, :, :, 1:] - pred_s[:, :, :, :-1]
            gt_dx = gt_s[:, :, :, 1:] - gt_s[:, :, :, :-1]
            mask_dx = mask_s[:, :, :, 1:] * mask_s[:, :, :, :-1]
            l_dx = self._masked_l1(pred_dx, gt_dx, mask_dx)

            pred_dy = pred_s[:, :, 1:, :] - pred_s[:, :, :-1, :]
            gt_dy = gt_s[:, :, 1:, :] - gt_s[:, :, :-1, :]
            mask_dy = mask_s[:, :, 1:, :] * mask_s[:, :, :-1, :]
            l_dy = self._masked_l1(pred_dy, gt_dy, mask_dy)

            loss = loss + l_int + 0.5 * (l_dx + l_dy)

        return loss / float(len(scales))

    def regularization_loss(self, gaussian_params, device):
        if gaussian_params is None:
            zero = torch.zeros((), device=device)
            return zero, zero, zero
        scales = gaussian_params.get("scales", None)
        alpha = gaussian_params.get("alpha", None)
        offset = gaussian_params.get("offset", None)
        scale_reg = scales.mean() if scales is not None else torch.zeros((), device=device)
        opacity_reg = alpha.mean() if alpha is not None else torch.zeros((), device=device)
        offset_reg = offset.pow(2).mean() if offset is not None else torch.zeros((), device=device)
        return scale_reg, opacity_reg, offset_reg

    def multiview_consistency_loss(self, gaussian_3d, device):
        """Penalize disagreement of per-view Gaussian 3D positions.

        gaussian_3d expected shape: [V, N, 3].
        """
        if gaussian_3d is None:
            return torch.zeros((), device=device)
        if gaussian_3d.ndim != 3 or gaussian_3d.shape[0] <= 1:
            return torch.zeros((), device=device)
        return gaussian_3d.var(dim=0, unbiased=False).mean()

    def mean_transparency_penalty(self, gaussian_params, device: torch.device) -> torch.Tensor:
        """mean(1 - alpha): lower is more opaque. Minimizing this pushes alpha toward 1."""
        if gaussian_params is None:
            return torch.zeros((), device=device)
        alpha = gaussian_params.get("alpha")
        if alpha is None:
            return torch.zeros((), device=device)
        a = alpha.reshape(-1).clamp(0.0, 1.0)
        return (1.0 - a).mean()

    def forward(self, pred_imgs, gt_imgs, gt_masks, gaussian_params=None, gaussian_3d=None):
        fg_mask = self._foreground_mask(gt_imgs)

        l1_loss = self._masked_l1(pred_imgs, gt_imgs, fg_mask)
        masked_ssim_val = self._masked_ssim(pred_imgs, gt_imgs, fg_mask)
        perceptual_loss = self._masked_multiscale_perceptual(pred_imgs, gt_imgs, fg_mask)

        # Silhouette loss: extract predicted mask from rendered images
        sil_loss = torch.zeros((), device=pred_imgs.device)
        if self.weight_silhouette > 0 and gt_masks is not None:
            pred_sil = self._foreground_mask(pred_imgs)  # Extract from PREDICTED images
            sil_loss = self.silhouette_loss(pred_sil, gt_masks)

        scale_reg, opacity_reg, offset_reg = self.regularization_loss(
            gaussian_params, device=pred_imgs.device
        )
        multiview_consistency = self.multiview_consistency_loss(
            gaussian_3d, device=pred_imgs.device
        )

        if self.weight_opacity_nontransparent > 0:
            transparency_penalty = self.mean_transparency_penalty(
                gaussian_params, device=pred_imgs.device
            )
        else:
            transparency_penalty = torch.zeros((), device=pred_imgs.device)

        ssim_loss = 1 - masked_ssim_val
        final_loss = (
            self.weight_l1 * l1_loss
            + self.weight_masked_ssim * ssim_loss
            + self.weight_perceptual * perceptual_loss
            + self.weight_silhouette * sil_loss
            + self.weight_scale_reg * scale_reg
            + self.weight_offset_reg * offset_reg
            + self.weight_opacity_nontransparent * transparency_penalty
            # self.weight_opacity_reg * opacity_reg
            # self.weight_multiview_consistency * multiview_consistency
        )

        return {
            "loss": final_loss,
            "l1": l1_loss,
            "silhouette": sil_loss,
            "masked_ssim": ssim_loss,
            "perceptual": perceptual_loss,
            "scale_reg": scale_reg,
            "offset_reg": offset_reg,
            "transparency_penalty": transparency_penalty,
            # "opacity_reg": opacity_reg,
            # "multiview_consistency": multiview_consistency,
        }
