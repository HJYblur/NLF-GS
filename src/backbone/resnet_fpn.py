from collections import OrderedDict
from contextlib import nullcontext
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
        frozen: bool = True,
    ):
        super().__init__()
        self.selected_levels: List[str] = list(selected_levels)
        self._level_to_fpn_key = {"p2": "0", "p3": "1", "p4": "2", "p5": "3"}
        self.frozen = bool(frozen)

        self.backbone = self._build_backbone(backbone_weights_path)
        self.set_frozen(self.frozen)

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        for param in self.backbone.parameters():
            param.requires_grad = not self.frozen
        if self.frozen:
            self.backbone.eval()

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

    def train(self, mode: bool = True):
        if self.frozen:
            # Keep this extractor permanently frozen in eval mode.
            super().train(False)
            self.backbone.eval()
            return self

        super().train(mode)
        self.backbone.train(mode)
        return self

    def forward(self, image: torch.Tensor) -> OrderedDict:
        # Keep extractor in fp32 to avoid dtype mismatches with downstream consumers.
        grad_ctx = torch.no_grad() if self.frozen else nullcontext()
        with grad_ctx, torch.autocast(device_type=image.device.type, enabled=False):
            feats = self.backbone(image.float())

        selected = OrderedDict()
        for level in self.selected_levels:
            key = self._level_to_fpn_key[level]
            if key not in feats:
                raise KeyError(f"FPN level '{level}' (key='{key}') not found in backbone output")
            selected[level] = feats[key]
        return selected

