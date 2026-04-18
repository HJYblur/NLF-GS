import torch
import torch.nn as nn


class IdentityEncoder(nn.Module):
    """Identity encoder that takes the feature map from the backbone and
    applies a linear layer after global average pooling.
    The output is a latent vector of specified dimension.
    """

    def __init__(self, backbone_feat_dim, latent_dim):
        super().__init__()
        self.backbone_feat_dim = backbone_feat_dim
        self.latent_dim = latent_dim
        self.fc = nn.Linear(backbone_feat_dim, latent_dim)

    def forward(self, feature_map):
        # Pool feature map to get global features
        B, C, Hf, Wf = feature_map.shape
        pooled_feats = torch.mean(feature_map.float(), dim=(2, 3))  # (B, C)

        # Match the linear layer dtype to avoid precision/dtype asserts when using fp16 backbones
        fc_dtype = self.fc.weight.dtype
        if pooled_feats.dtype != fc_dtype:
            pooled_feats = pooled_feats.to(fc_dtype)

        batched_z_id = self.fc(pooled_feats)  # (B, latent_dim)

        # TODO(Optional): Use EMA to stabilize identity encoding across batches

        # Use the mean of the batched_z_id as the final identity latent vector per identity,
        # but still returns as a batch of size B for compatibility
        z_id = batched_z_id.mean(dim=0)  # (latent_dim,)
        z_id = z_id.unsqueeze(0)  # (1, latent_dim)

        return z_id
