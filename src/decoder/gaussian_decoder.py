import torch
import torch.nn as nn
import torch.nn.functional as F
from avatar_utils.config import get_config


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
        cfg = get_config() or {}
        dec_cfg = cfg.get("decoder", {})

        self.debug = debug
        self.in_dim = int(dec_cfg.get("in_dim", cfg.get("model", {}).get("local_feature_dim", 512)))
        self.hidden = int(dec_cfg.get("hidden", 256))
        self.out_dim = int(dec_cfg.get("out_dim", 56))
        self.z_dim = int(cfg.get("identity_encoder", {}).get("latent_dim", 64))

        # Scale parameterization bounds (sigmoid-based, always differentiable)
        self.scale_min = float(dec_cfg.get("scale_min", 1e-6))
        self.scale_max = float(dec_cfg.get("scale_max", 0.001))
        self.alpha_min = float(dec_cfg.get("alpha_min", 0.0))
        self.alpha_max = float(dec_cfg.get("alpha_max", 1.0))
        self.rot_min = float(dec_cfg.get("rot_min", -1.0))
        self.rot_max = float(dec_cfg.get("rot_max", 1.0))
        self.sh_min = float(dec_cfg.get("sh_min", -1.0))
        self.sh_max = float(dec_cfg.get("sh_max", 1.0))
        self.offset_scale = float(dec_cfg.get("offset_scale", 0.01))

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

        # self._init_output_bias()

    def _init_output_bias(self):
        """Set the bias of the last MLP layer to produce sensible initial Gaussian params.

        At raw=0 with default init the mapping is:
          scales:   sigmoid(0)=0.5 → mid-range  (start smaller to avoid blobs)
          rotation: ~random unit quat             (start at identity [1,0,0,0])
          opacity:  sigmoid(0)=0.5                (start more opaque for visibility)
          SH:       random raw bias         (breaks symmetry at startup)
        """
        last_layer = self.mlp[-1]  # nn.Linear(hidden, out_dim)
        with torch.no_grad():
            last_layer.bias.zero_()
            # Scales (indices 0-2): bias=-2 → sigmoid(-2)≈0.12 → small initial Gaussians
            last_layer.bias[0:3] = -2.0
            # Rotation (indices 3-6): bias toward identity quaternion [w=1, x=0, y=0, z=0]
            last_layer.bias[3] = 1.0   # w component (dominant)
            last_layer.bias[4:7] = 0.0  # x, y, z near zero
            # Opacity (index 7): bias=2 → sigmoid(2)≈0.88, clearly visible
            last_layer.bias[7] = 2.0
            # SH (indices 8+): initialize DC and higher-order terms separately.
            # First SH RGB triplet (DC) controls base color most strongly, so keep it
            # zero-centered to avoid washed-out white startup renders.
            if self.out_dim > 8:
                sh_bias = last_layer.bias[8:]
                if sh_bias.numel() >= 3:
                    sh_bias[:3].uniform_(-1, 1)
                    if sh_bias.numel() > 3:
                        sh_bias[3:].uniform_(-0.12, 0.12)
                else:
                    sh_bias.uniform_(-0.05, 0.05)

    def forward(self, combined_feats, z_id=None):
        """
        combined_feats: (1, N, in_dim)
        z_id: Optional (1, z_dim). If provided, FiLM modulation is applied.

        Returns a dict of parameterized Gaussian fields without batch fusion:
            scales: (N,3), rotation: (N,4), alpha: (N,), sh: (N,K)

        Assumption: inputs have been aggregated across batch already (B==1).
        """

        # Support chunked decoding over the Gaussian dimension to reduce peak VRAM
        cfg = get_config() or {}
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

        parts = {"scales": [], "rotation": [], "alpha": [], "offset": [], "sh": []}
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
            offset_nc = split_out["offset"].squeeze(0)  # (nc,3)
            sh_nc = split_out.get("sh", None)
            sh_nc = (
                None if (sh_nc is None or sh_nc.numel() == 0) else sh_nc.squeeze(0)
            )  # (nc,K)

            parts["scales"].append(scales_nc)
            parts["rotation"].append(rot_nc)
            parts["alpha"].append(alpha_nc)
            parts["offset"].append(offset_nc)
            parts["sh"].append(sh_nc)

            # Free chunk temporaries ASAP
            del feats_chunk, h, out, split_out

        # Concatenate chunk results
        scales = torch.cat([x for x in parts["scales"]], dim=0)  # (N,3)
        rotation = torch.cat([x for x in parts["rotation"]], dim=0)  # (N,4)
        alpha = torch.cat([x.squeeze(-1) for x in parts["alpha"]], dim=0)  # (N,)
        offset = torch.cat([x for x in parts["offset"]], dim=0)  # (N,3)

        # If any chunk had SH, stack; else set to None
        if any(x is not None for x in parts["sh"]):
            sh = torch.cat([x for x in parts["sh"] if x is not None], dim=0)  # (N,K)
        else:
            sh = None

        return {
            "scales": scales,
            "rotation": rotation,
            "alpha": alpha,
            "offset": offset,
            "sh": sh,
        }

    def split_and_parameterize(self, out):
        """
        Split raw MLP outputs into parameter fields and apply stable parameterizations.
        """
        D = out.shape[-1]
        min_header = 3 + 4 + 1 + 3
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
        offset_raw = out[..., i : i + 3]
        i += 3
        sh_raw = (
            out[..., i : i + sh_dim]
            if sh_dim > 0
            else out.new_zeros((*out.shape[:-1], 0))
        )

        # Normalize each output head to [0, 1] in latent prediction space,
        # then map to its physical domain.
        scales_01 = torch.sigmoid(scales_raw)
        rot_01 = torch.sigmoid(rot_raw)
        alpha_01 = torch.sigmoid(alpha_raw)
        sh_01 = torch.sigmoid(sh_raw)

        scales = self.scale_min + (self.scale_max - self.scale_min) * scales_01

        rot_mapped = self.rot_min + (self.rot_max - self.rot_min) * rot_01
        rot_norm = torch.linalg.norm(rot_mapped, dim=-1, keepdim=True)
        rot = rot_mapped / (rot_norm + 1e-8)

        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_01
        offset = self.offset_scale * torch.tanh(offset_raw)

        sh = self.sh_min + (self.sh_max - self.sh_min) * sh_01

        self.investigate(scales, rot, alpha, offset, sh)
        
        return {
            "scales": scales,
            "rotation": rot,
            "alpha": alpha,
            "offset": offset,
            "sh": sh,
        }

    def investigate(self, scales, rotation, alpha, offset, sh):
        if not self.debug:
            return
        print(f"Scale min/max: {scales.min().item()},{scales.max().item()}, mean/std: {scales.mean().item()},{scales.std().item()} ")
        print(f"Rotation min/max: {rotation.min().item()},{rotation.max().item()}, mean/std: {rotation.mean().item()},{rotation.std().item()} ")
        print(f"Alpha min/max: {alpha.min().item()},{alpha.max().item()}, mean/std: {alpha.mean().item()},{alpha.std().item()} ")
        print(f"Offset min/max: {offset.min().item()},{offset.max().item()}, mean/std: {offset.mean().item()},{offset.std().item()} ")
        if sh is not None:
            print(f"SH min/max: {sh.min().item()},{sh.max().item()}, mean/std: {sh.mean().item()},{sh.std().item()} ")
            
