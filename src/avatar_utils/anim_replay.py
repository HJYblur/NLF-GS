"""Replay Gaussian **positions** under new SMPL-X vertices without re-running backbone/decoder."""

from __future__ import annotations

import torch

from encoder.gaussian_estimator import AvatarGaussianEstimator


def replay_fused_gaussian_means(
    estimator: AvatarGaussianEstimator,
    vertices3d: torch.Tensor,
    gaussian_params: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Fused 3D centers for ``vertices3d`` using **saved** local offsets from inference.

    Matches training / ``run_inference``:

        fused = base_centers(vertices3d) + einsum(local_frames(vertices3d), offset)

    where ``offset`` is the decoder's per-Gaussian offset in the template local frame.
    Appearance tensors (``scales``, ``rotation``, ``sh``, ``alpha``) are **not** recomputed.

    Args:
        estimator: Same template topology as training (from ``NlfGaussianModel.avatar_estimator``).
        vertices3d: ``(N_verts, 3)`` or ``(1, N_verts, 3)`` SMPL-X vertices (world space).
        gaussian_params: Saved dict from inference; must contain ``offset`` if offsets were used;
            otherwise only barycentric bases are used.

    Returns:
        ``(N_gauss, 3)`` float tensor on ``device``.
    """
    verts = vertices3d.to(device=device, dtype=torch.float32)
    if verts.ndim == 2:
        verts_b = verts.unsqueeze(0)
    else:
        verts_b = verts
    B = verts_b.shape[0]
    if B != 1:
        raise ValueError(
            f"anim replay expects a single pose (batch 1); got vertices batch size {B}."
        )

    dummy_fmap = torch.zeros(1, 1, 2, 2, device=device, dtype=torch.float32)
    centers3d = estimator.compute_gaussian_coord3d(dummy_fmap, verts_b)
    local_frames = estimator.compute_gaussian_local_frames(
        verts_b[0], device=device, batch_size=1
    )

    base = centers3d[0]
    offset = gaussian_params.get("offset", None)
    if offset is None:
        return base

    off = offset.to(device=device, dtype=torch.float32)
    return base + torch.einsum("nij,nj->ni", local_frames[0], off)


def gaussian_params_for_render(
    gaussian_params: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Strip ``offset`` (already fused into means) and move tensors to ``device`` for gsplat."""
    return {
        k: v.to(device=device)
        for k, v in gaussian_params.items()
        if k != "offset"
    }
