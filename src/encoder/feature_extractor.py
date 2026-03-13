from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from backbone.resnet_fpn import FrozenResNet50FPNExtractor


class FeatureExtractor(nn.Module):
    """Feature extractor exposing either NLF features or frozen ResNet50-FPN features."""

    def __init__(
        self,
        nlf_model,
        use_resnet_fpn: bool = False,
        fpn_levels=None,
        resnet_weights_path: Optional[str] = None,
    ):
        super().__init__()
        if not hasattr(nlf_model, "detector"):
            raise AttributeError("nlf_model must expose detector")

        self.nlf_model = nlf_model
        self.use_resnet_fpn = bool(use_resnet_fpn)
        self.fpn_extractor: Optional[FrozenResNet50FPNExtractor] = None

        if self.use_resnet_fpn:
            self.fpn_extractor = FrozenResNet50FPNExtractor(
                selected_levels=fpn_levels or ("p2", "p3", "p4"),
                backbone_weights_path=resnet_weights_path,
            )
        else:
            if not hasattr(nlf_model, "crop_model") or not hasattr(
                nlf_model.crop_model, "backbone"
            ):
                raise AttributeError("nlf_model must expose crop_model.backbone")


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
        if self.use_resnet_fpn:
            assert self.fpn_extractor is not None
            return self.fpn_extractor(image.float())

        x = image.half() if use_half else image
        x = self.nlf_model.crop_model.backbone(x)
        if use_heatmap_head and hasattr(self.nlf_model.crop_model, "heatmap_head"):
            head = self.nlf_model.crop_model.heatmap_head
            x = getattr(head, "layer", head)(x)
        return x

    def detect_with_features(
        self,
        image_feature: torch.Tensor,
        frame_batch: torch.Tensor,
        model_name: str = "smplx",
        use_half: bool = True,
        use_heatmap_head: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run detection while exposing intermediate backbone features."""

        feature_map = self.extract_feature_map(
            image=image_feature, use_half=use_half, use_heatmap_head=use_heatmap_head
        )

        if frame_batch.dtype.is_floating_point:
            frame_batch = (frame_batch * 255.0).round().clamp(0, 255).to(torch.uint8)

        preds = self.nlf_model.detect_smpl_batched(
            frame_batch, model_name=model_name, **kwargs
        )
        return feature_map, preds
