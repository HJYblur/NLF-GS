from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FrozenResNet50FPNExtractor(nn.Module):
    """Frozen ResNet50+FPN extractor returning selected pyramid levels."""

    def __init__(
        self,
        selected_levels: Iterable[str] = ("p2", "p3", "p4"),
        backbone_weights_path: Optional[str] = None,
    ):
        super().__init__()
        self.selected_levels: List[str] = list(selected_levels)
        self._level_to_fpn_key = {"p2": "0", "p3": "1", "p4": "2", "p5": "3"}

        self.backbone = self._build_backbone(backbone_weights_path)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _build_backbone(self, backbone_weights_path: Optional[str]) -> nn.Module:
        if backbone_weights_path:
            weights_path = Path(backbone_weights_path)
            if weights_path.is_file():
                backbone = resnet_fpn_backbone(
                    backbone_name="resnet50",
                    weights=None,
                    trainable_layers=0,
                )
                state_dict = torch.load(str(weights_path), map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                if not isinstance(state_dict, dict):
                    raise TypeError(
                        f"Unsupported weight file format at {weights_path}; expected a state_dict"
                    )
                backbone.body.load_state_dict(state_dict, strict=False)
                return backbone

        return resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=ResNet50_Weights.DEFAULT,
            trainable_layers=0,
        )

    def _ensure_backbone_device(self, device: torch.device) -> None:
        current_device = next(self.backbone.parameters()).device
        if current_device != device:
            self.backbone.to(device)

    def train(self, mode: bool = True):
        # Keep this extractor permanently frozen in eval mode.
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> OrderedDict:
        # FeatureExtractor is not an nn.Module, so Lightning will not auto-move this
        # frozen backbone to the batch device. Keep device and dtype aligned manually.
        self._ensure_backbone_device(image.device)

        # Keep the frozen extractor in fp32 to avoid AMP/mixed-precision dtype mismatches.
        with torch.no_grad(), torch.autocast(device_type=image.device.type, enabled=False):
            feats = self.backbone(image.float())

        selected = OrderedDict()
        for level in self.selected_levels:
            key = self._level_to_fpn_key[level]
            if key not in feats:
                raise KeyError(f"FPN level '{level}' (key='{key}') not found in backbone output")
            selected[level] = feats[key]
        return selected

