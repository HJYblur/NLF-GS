from typing import Optional

import torch
import torch.nn as nn

from backbone.resnet_fpn import FrozenResNet50FPNExtractor


class FeatureExtractor(nn.Module):
    """Feature extractor backed by a ResNet50-FPN encoder."""

    def __init__(
        self,
        fpn_levels=None,
        resnet_weights_path: Optional[str] = None,
        freeze_resnet_fpn: bool = True,
    ):
        super().__init__()
        self.fpn_extractor = FrozenResNet50FPNExtractor(
            selected_levels=fpn_levels or ("p2", "p3", "p4"),
            backbone_weights_path=resnet_weights_path,
            frozen=freeze_resnet_fpn,
        )

    def set_resnet_fpn_frozen(self, frozen: bool) -> None:
        self.fpn_extractor.set_frozen(frozen)


    def forward(
        self, image: torch.Tensor, use_half: bool = True, use_heatmap_head: bool = True
    ):
        return self.extract_feature_map(
            image=image, use_half=use_half, use_heatmap_head=use_heatmap_head
        )

    def extract_feature_map(
        self, image: torch.Tensor, use_half: bool = True, use_heatmap_head: bool = True
    ):
        """Return either a single feature map or an OrderedDict of FPN maps."""
        return self.fpn_extractor(image.float())
