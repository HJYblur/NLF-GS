import torch
import torch.nn as nn


class LearnedViewFusion(nn.Module):
    """Fuse multi-view Gaussian features: h_v = MLP([f_v, w_v]), α = softmax(MLP_gate(h_v)), out = Σ α_v h_v."""

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__()
        d_in = feat_dim + 1
        self.proj = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, local_feats: torch.Tensor, view_weights: torch.Tensor) -> torch.Tensor:
        # local_feats: (V, N, C), view_weights: (V, N) — V = batch of views
        w = view_weights.unsqueeze(-1).clamp_min(0.0)
        h = self.proj(torch.cat([local_feats, w], dim=-1))
        s = self.gate(h).squeeze(-1)
        alpha = torch.softmax(s, dim=0)
        return (alpha.unsqueeze(-1) * h).sum(dim=0, keepdim=True)
