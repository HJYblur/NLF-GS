from collections import OrderedDict
from typing import Iterable, List

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FrozenResNet50FPNExtractor(nn.Module):
    """Frozen ResNet50+FPN extractor returning selected pyramid levels."""

    def __init__(self, selected_levels: Iterable[str] = ("p2", "p3", "p4")):
        super().__init__()
        self.selected_levels: List[str] = list(selected_levels)
        self._level_to_fpn_key = {"p2": "0", "p3": "1", "p4": "2", "p5": "3"}

        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=ResNet50_Weights.DEFAULT,
            trainable_layers=0,
        )

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        # Keep this extractor permanently frozen in eval mode.
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> OrderedDict:
        with torch.no_grad():
            feats = self.backbone(image)

        selected = OrderedDict()
        for level in self.selected_levels:
            key = self._level_to_fpn_key[level]
            if key not in feats:
                raise KeyError(f"FPN level '{level}' (key='{key}') not found in backbone output")
            selected[level] = feats[key]
        return selected

