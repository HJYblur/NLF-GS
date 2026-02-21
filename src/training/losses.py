import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from avatar_utils.config import load_config

class LossFunctions(nn.Module):
    def __init__(self, weight_rgb=None):
        super().__init__()
        self.weight_rgb = weight_rgb if weight_rgb is not None else float(load_config().get("train", {}).get("weight_rgb", 1.0))
        # Create SSIM once and keep it as a sub-module (avoids repeated GPU allocations)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def rgb_loss(self, pred_imgs, gt_imgs):
        l1_loss = nn.functional.l1_loss(pred_imgs, gt_imgs)
        ssim_val = self.ssim(pred_imgs, gt_imgs)
        return 0.8 * l1_loss + 0.2 * (1 - ssim_val)

    def forward(self, pred_imgs, gt_imgs):
        loss = self.weight_rgb * self.rgb_loss(pred_imgs, gt_imgs)
        # TODO: Add more loss components (e.g., regularization on Gaussian parameters) and corresponding weights from config
        return loss
