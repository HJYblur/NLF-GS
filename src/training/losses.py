import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from avatar_utils.config import load_config

class LossFunctions(nn.Module):
    def __init__(self, weight_rgb=None):
        super().__init__()
        self.weight_rgb = weight_rgb if weight_rgb is not None else float(load_config().get("train", {}).get("weight_rgb", 1.0))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def rgb_loss(self, pred_imgs, gt_imgs):
        mask = (gt_imgs.abs().sum(dim=1) > 0.0)
        pred_imgs_masked = pred_imgs.permute(0, 2, 3, 1)[mask]
        gt_imgs_masked = gt_imgs.permute(0, 2, 3, 1)[mask]
        
        l1_loss = nn.functional.l1_loss(pred_imgs_masked, gt_imgs_masked)
        # TODO: Also a masked version fo ssim?
        ssim_val = self.ssim(pred_imgs, gt_imgs)
        
        return l1_loss, ssim_val

    def forward(self, pred_imgs, gt_imgs):
        # TODO: Add more loss components (e.g., regularization on Gaussian parameters) and corresponding weights from config
        l1_loss, ssim_val = self.rgb_loss(pred_imgs, gt_imgs)
        final_loss = self.weight_rgb * (0.8 * l1_loss + 0.2 * (1 - ssim_val))
        
        return {"loss": final_loss, "l1": l1_loss, "ssim": ssim_val}
