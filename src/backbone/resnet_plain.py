"""Single-map ResNet50 encoder (no FPN): ``layer4`` features, optional 1×1 projection.

Used when ``backbone.encoder: plain`` in config. Implements the same freeze/extract API as
:class:`encoder.feature_extractor.FeatureExtractor` so training and inference stay unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class PlainResNet50FeatureExtractor(nn.Module):
    """ResNet50 through ``layer4`` → ``[B, C, H/32, W/32]`` with optional channel projection."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        frozen: bool = True,
        proj_channels: Optional[int] = 256,
        *,
        pretrained_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.proj_channels = proj_channels
        self._frozen = bool(frozen)
        self._pretrained_fallback = bool(pretrained_fallback)

        net = resnet50(weights=None)
        # torchvision layout: stem → layer1–4 → avgpool → fc
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        out_ch = 2048
        self.proj: Optional[nn.Module]
        if proj_channels is not None and int(proj_channels) > 0:
            self.proj = nn.Conv2d(out_ch, int(proj_channels), kernel_size=1, bias=True)
            self.out_channels = int(proj_channels)
        else:
            self.proj = None
            self.out_channels = out_ch

        self._load_backbone_weights(weights_path)
        self.set_resnet_fpn_frozen(self._frozen)

    def _load_backbone_weights(self, weights_path: Optional[str]) -> None:
        if weights_path:
            path = Path(weights_path)
            if path.is_file():
                blob = torch.load(str(path), map_location="cpu")
                state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
                if isinstance(state, dict):
                    self.load_state_dict(state, strict=False)
                    return
        if self._pretrained_fallback:
            net = resnet50(weights=ResNet50_Weights.DEFAULT)
            self._copy_loaded(net)

    def _copy_loaded(self, net: nn.Module) -> None:
        self.conv1.load_state_dict(net.conv1.state_dict())
        self.bn1.load_state_dict(net.bn1.state_dict())
        self.relu.load_state_dict(net.relu.state_dict())
        self.maxpool.load_state_dict(net.maxpool.state_dict())
        self.layer1.load_state_dict(net.layer1.state_dict())
        self.layer2.load_state_dict(net.layer2.state_dict())
        self.layer3.load_state_dict(net.layer3.state_dict())
        self.layer4.load_state_dict(net.layer4.state_dict())

    def set_resnet_fpn_frozen(self, frozen: bool) -> None:
        """Match :meth:`FeatureExtractor.set_resnet_fpn_frozen` name for NLF-GS trainer."""
        self._frozen = bool(frozen)
        for p in self.parameters():
            p.requires_grad = not self._frozen
        if self._frozen:
            self.eval()

    def train(self, mode: bool = True):
        if self._frozen:
            super().train(False)
            self.eval()
            return self
        return super().train(mode)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image.float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.proj is not None:
            x = self.proj(x)
        return x

    def extract_feature_map(self, image: torch.Tensor) -> torch.Tensor:
        grad_ctx = torch.no_grad() if self._frozen else nullcontext()
        with grad_ctx, torch.autocast(device_type=image.device.type, enabled=False):
            return self.forward(image)
