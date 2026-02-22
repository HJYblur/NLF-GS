import torch
import torch.nn as nn
import torch.nn.functional as F
from avatar_utils.config import load_config


class GaussianDecoder(nn.Module):
    """
    MLP head to predict Gaussian parameters taking in the encoding of pose info + appearance features,
    while incorporating identity latent code as a FiLM layer to modulate the features.

    Output parameterization (per Gaussian):
      - scales: 3 values (positive radii)
      - rotation: quaternion (4)
      - alpha: opacity (1)
      - sh: K spherical-harmonics coeffs (remaining dims)
    """

    def __init__(self, debug = False):
        super().__init__()
        cfg = load_config() or {}
        dec_cfg = cfg.get("decoder", {})

        self.debug = debug
        self.in_dim = int(dec_cfg.get("in_dim", 512))
        self.hidden = int(dec_cfg.get("hidden", 256))
        self.out_dim = int(dec_cfg.get("out_dim", 56))
        self.z_dim = int(cfg.get("identity_encoder", {}).get("latent_dim", 64))

        # first local block
        self.fc1 = nn.Linear(self.in_dim, self.hidden)
        self.activation1 = nn.ReLU(inplace=True)

        # FiLM film_net
        self.film_net = nn.Linear(self.z_dim, 2 * self.hidden)

        # remaining MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.out_dim),
        )

    def forward(self, combined_feats, z_id=None):
        """
        combined_feats: (1, N, in_dim)
        z_id: Optional (1, z_dim). If provided, FiLM modulation is applied.

        Returns a dict of parameterized Gaussian fields without batch fusion:
            scales: (N,3), rotation: (N,4), alpha: (N,), sh: (N,K)

        Assumption: inputs have been aggregated across batch already (B==1).
        """

        # Support chunked decoding over the Gaussian dimension to reduce peak VRAM
        cfg = load_config() or {}
        dec_cfg = cfg.get("decoder", {})
        chunk_size = int(dec_cfg.get("chunk_size", 8192))

        B, N, _ = combined_feats.shape

        if B != 1:
            raise ValueError(
                f"Decoder expects aggregated inputs with batch size 1, got B={B}"
            )

        # Precompute FiLM gamma/beta once per batch and reuse for chunks if z_id is provided
        if z_id is not None:
            gamma_beta = self.film_net(z_id)  # (B, 2H)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)  # (B,1,H)
            beta = beta.unsqueeze(1)  # (B,1,H)
        else:
            gamma = None
            beta = None

        parts = {"scales": [], "rotation": [], "alpha": [], "sh": []}
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            feats_chunk = combined_feats[:, start:end, :]  # (B,nc,in_dim)

            # First block + FiLM
            h = self.fc1(feats_chunk)  # (B,nc,H)
            if gamma is not None and beta is not None:
                h = (1.0 + gamma) * h + beta
            h = self.activation1(h)

            # Remaining MLP
            out = self.mlp(h)  # (B,nc,out_dim)

            # Parameterize per chunk without batch fusion
            split_out = self.split_and_parameterize(out)  # dict of (B,nc,*)
            # Squeeze batch dimension (B==1)
            scales_nc = split_out["scales"].squeeze(0)  # (nc,3)
            rot_nc = split_out["rotation"].squeeze(0)  # (nc,4)
            alpha_nc = split_out["alpha"].squeeze(0)  # (nc,1)
            sh_nc = split_out.get("sh", None)
            sh_nc = (
                None if (sh_nc is None or sh_nc.numel() == 0) else sh_nc.squeeze(0)
            )  # (nc,K)

            parts["scales"].append(scales_nc)
            parts["rotation"].append(rot_nc)
            parts["alpha"].append(alpha_nc)
            parts["sh"].append(sh_nc)

            # Free chunk temporaries ASAP
            del feats_chunk, h, out, split_out

        # Concatenate chunk results
        scales = torch.cat([x for x in parts["scales"]], dim=0)  # (N,3)
        rotation = torch.cat([x for x in parts["rotation"]], dim=0)  # (N,4)
        alpha = torch.cat([x.squeeze(-1) for x in parts["alpha"]], dim=0)  # (N,)

        # If any chunk had SH, stack; else set to None
        if any(x is not None for x in parts["sh"]):
            sh = torch.cat([x for x in parts["sh"] if x is not None], dim=0)  # (N,K)
        else:
            sh = None

        return {"scales": scales, "rotation": rotation, "alpha": alpha, "sh": sh}

    def split_and_parameterize(self, out):
        """
        Split raw MLP outputs into parameter fields and apply stable parameterizations.
        """
        D = out.shape[-1]
        min_header = 3 + 4 + 1
        if D < min_header:
            raise ValueError(f"Output dim must be >= {min_header}, got {D}")

        sh_dim = D - min_header
        i = 0
        scales_raw = out[..., i : i + 3]
        i += 3
        rot_raw = out[..., i : i + 4]
        i += 4
        alpha_raw = out[..., i : i + 1]
        i += 1
        sh_raw = (
            out[..., i : i + sh_dim]
            if sh_dim > 0
            else out.new_zeros((*out.shape[:-1], 0))
        )

        eps = 1e-6
        scales = F.softplus(scales_raw) + eps
        scales = torch.clamp(scales, min=1e-6, max=0.005)

        rot_norm = torch.linalg.norm(rot_raw, dim=-1, keepdim=True)
        rot = rot_raw / (rot_norm + 1e-8)

        alpha = torch.sigmoid(alpha_raw)
        alpha = torch.clamp(alpha, min=1e-6, max=1.0)

        sh = torch.tanh(sh_raw) * 0.5

        self.investigate(scales, rot, alpha, sh)
        
        return {
            "scales": scales,
            "rotation": rot,
            "alpha": alpha,
            "sh": sh,
        }

    def investigate(self, scales, rotation, alpha, sh):
        if not self.debug:
            return
        print(f"Scale min/max: {scales.min().item()},{scales.max().item()}, mean/std: {scales.mean().item()},{scales.std().item()} ")
        print(f"Rotation min/max: {rotation.min().item()},{rotation.max().item()}, mean/std: {rotation.mean().item()},{rotation.std().item()} ")
        print(f"Alpha min/max: {alpha.min().item()},{alpha.max().item()}, mean/std: {alpha.mean().item()},{alpha.std().item()} ")
        if sh is not None:
            print(f"SH min/max: {sh.min().item()},{sh.max().item()}, mean/std: {sh.mean().item()},{sh.std().item()} ")
            