import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from encoder.avatar_template import AvatarTemplate


class AvatarGaussianEstimator(nn.Module):
    """
    Feature Sampler features and predicts Gaussian parameters
    """

    def __init__(self, template: AvatarTemplate):
        super().__init__()
        self._avatar = template
        self.last_sampled_level_shapes = None

    @property
    def template(self) -> AvatarTemplate:
        return self._avatar

    @staticmethod
    def _ordered_levels(feature_maps):
        if isinstance(feature_maps, OrderedDict):
            return list(feature_maps.keys())
        return sorted(feature_maps.keys())

    def compute_gaussian_coord2d(self, feature_map, vertices2d, img_shape: tuple = None):
        B = feature_map.shape[0]
        N = int(self._avatar.total_gaussians_num)
        K = int(self._avatar.barycentric_coords.shape[0])

        parents = self._avatar.parents
        bary = self._avatar.barycentric_coords

        device = feature_map.device

        parents = parents.to(device=device, dtype=torch.long)
        bary = bary.to(device=device, dtype=torch.float32)
        verts2d = vertices2d.to(device=device, dtype=torch.float32)
        assert verts2d.shape[0] == B, "Batch size mismatch in vertices2d"

        idx = torch.arange(N, device=device) % K
        bary_per_gauss = bary[idx]

        flat_idx = parents.reshape(-1)
        verts_sel = verts2d.index_select(1, flat_idx)
        face_verts = verts_sel.reshape(B, N, 3, 2)

        centers2d = torch.einsum("bnvc, nv->bnc", face_verts, bary_per_gauss)
        return centers2d

    def compute_gaussian_coord3d(self, feature_map, vertices3d):
        B = feature_map.shape[0]
        N = int(self._avatar.total_gaussians_num)
        K = int(self._avatar.barycentric_coords.shape[0])

        parents = self._avatar.parents
        bary = self._avatar.barycentric_coords

        device = feature_map.device

        parents = parents.to(device=device, dtype=torch.long)
        bary = bary.to(device=device, dtype=torch.float32)
        verts3d = vertices3d.to(device=device, dtype=torch.float32)
        if verts3d.ndim == 2:
            verts3d = verts3d.unsqueeze(0).expand(B, -1, -1)
        assert verts3d.shape[0] == B, "Batch size mismatch in vertices3d"

        idx = torch.arange(N, device=device) % K
        bary_per_gauss = bary[idx]

        flat_idx = parents.reshape(-1)
        verts_sel = verts3d.index_select(1, flat_idx)
        face_verts = verts_sel.reshape(B, N, 3, 3)

        centers3d = torch.einsum("bnpc,np->bnc", face_verts, bary_per_gauss)
        return centers3d

    def compute_gaussian_normals(self, vertices3d, device):
        verts3d = vertices3d.to(device=device, dtype=torch.float32)
        if verts3d.ndim == 2:
            verts3d = verts3d.unsqueeze(0)
        B = verts3d.shape[0]
        N = int(self._avatar.total_gaussians_num)
        parents = self._avatar.parents.to(device=device, dtype=torch.long)

        flat_idx = parents.reshape(-1)
        verts_sel = verts3d.index_select(dim=1, index=flat_idx)
        face_verts = verts_sel.reshape(B, N, 3, 3)

        e1 = face_verts[:, :, 1] - face_verts[:, :, 0]
        e2 = face_verts[:, :, 2] - face_verts[:, :, 0]
        normals = torch.linalg.cross(e1, e2, dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        return normals

    def compute_gaussian_local_frames(self, vertices3d, device):
        """Build per-Gaussian orthonormal local frames from parent face geometry.

        Returns:
            Tensor[B, N, 3, 3] where columns are (tangent_u, tangent_v, normal).
        """
        verts3d = vertices3d.to(device=device, dtype=torch.float32)
        if verts3d.ndim == 2:
            verts3d = verts3d.unsqueeze(0)
        B = verts3d.shape[0]
        N = int(self._avatar.total_gaussians_num)
        parents = self._avatar.parents.to(device=device, dtype=torch.long)

        flat_idx = parents.reshape(-1)
        verts_sel = verts3d.index_select(dim=1, index=flat_idx)
        face_verts = verts_sel.reshape(B, N, 3, 3)

        e1 = face_verts[:, :, 1] - face_verts[:, :, 0]
        e2 = face_verts[:, :, 2] - face_verts[:, :, 0]

        tangent_u = e1 / (torch.norm(e1, dim=-1, keepdim=True) + 1e-8)
        normal = torch.linalg.cross(e1, e2, dim=-1)
        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-8)
        tangent_v = torch.linalg.cross(normal, tangent_u, dim=-1)
        tangent_v = tangent_v / (torch.norm(tangent_v, dim=-1, keepdim=True) + 1e-8)

        return torch.stack((tangent_u, tangent_v, normal), dim=-1)

    def build_depth_map(self, vertices3d, vertices2d, img_shape: tuple, device):
        H_img, W_img = int(img_shape[0]), int(img_shape[1])
        v2d = vertices2d.to(device=device, dtype=torch.float32)
        v3d = vertices3d.to(device=device, dtype=torch.float32)
        if v3d.ndim == 2:
            v3d = v3d.unsqueeze(0).expand(v2d.shape[0], -1, -1)

        B, Nv, _ = v2d.shape
        depth_maps = torch.full((B, H_img * W_img), float("inf"), device=device, dtype=v3d.dtype)

        x = torch.round(v2d[..., 0]).to(torch.long)
        y = torch.round(v2d[..., 1]).to(torch.long)
        z = v3d[..., 2]

        valid = (x >= 0) & (x < W_img) & (y >= 0) & (y < H_img)
        for b in range(B):
            if not valid[b].any():
                continue
            idx = (y[b, valid[b]] * W_img + x[b, valid[b]]).to(torch.long)
            depth_vals = z[b, valid[b]]
            if hasattr(torch.Tensor, "scatter_reduce_"):
                depth_maps[b].scatter_reduce_(0, idx, depth_vals, reduce="amin", include_self=True)
            else:
                for i in range(idx.numel()):
                    depth_maps[b, idx[i]] = torch.minimum(depth_maps[b, idx[i]], depth_vals[i])

        return depth_maps.view(B, H_img, W_img)

    def compute_view_weights(
        self,
        feature_map,
        vertices3d: torch.Tensor,
        vertices2d: torch.Tensor,
        centers2d: torch.Tensor,
        centers3d: torch.Tensor,
        img_shape: tuple,
        depth_eps: float = 1e-3,
    ):
        device = feature_map.device
        normals = self.compute_gaussian_normals(vertices3d, device=device)

        view_dir = -centers3d
        view_dir = view_dir / (torch.norm(view_dir, dim=-1, keepdim=True) + 1e-8)
        angle_weight = torch.sum(normals * view_dir, dim=-1).clamp_min(0.0)

        depth_map = self.build_depth_map(vertices3d, vertices2d, img_shape=img_shape, device=device)
        H_img, W_img = int(img_shape[0]), int(img_shape[1])

        x = torch.round(centers2d[..., 0]).to(torch.long)
        y = torch.round(centers2d[..., 1]).to(torch.long)
        x = x.clamp(0, W_img - 1)
        y = y.clamp(0, H_img - 1)

        batch_idx = torch.arange(depth_map.shape[0], device=device)[:, None]
        depth_samples = depth_map[batch_idx, y, x]
        center_depth = centers3d[..., 2]
        visible = center_depth <= (depth_samples + depth_eps)
        visible = visible | torch.isinf(depth_samples)
        visibility = visible.to(dtype=feature_map.dtype)

        return angle_weight * visibility

    def feature_sample(self, feature_map, vertices2d, img_shape: tuple = None, centers2d: torch.Tensor = None):
        if centers2d is None:
            centers2d = self.compute_gaussian_coord2d(feature_map, vertices2d, img_shape=img_shape)

        _, _, Hf, Wf = feature_map.shape
        centers2d = centers2d.to(device=feature_map.device, dtype=feature_map.dtype)

        if img_shape is not None:
            H_img, W_img = int(img_shape[0]), int(img_shape[1])
        else:
            H_img, W_img = Hf, Wf

        x = centers2d[..., 0] / max(1, (W_img - 1)) * 2 - 1
        y = centers2d[..., 1] / max(1, (H_img - 1)) * 2 - 1
        grid = torch.stack([x, y], dim=-1).unsqueeze(1)

        if grid.dtype != feature_map.dtype:
            grid = grid.to(dtype=feature_map.dtype)

        sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True)
        return sampled[:, :, 0, :].permute(0, 2, 1)

    def feature_sample_multiscale(self, feature_maps, vertices2d, img_shape: tuple = None, centers2d: torch.Tensor = None):
        per_level = []
        sampled_shapes = {}
        for level in self._ordered_levels(feature_maps):
            fmap = feature_maps[level]
            sampled = self.feature_sample(fmap, vertices2d, img_shape=img_shape, centers2d=centers2d)
            sampled_shapes[level] = tuple(sampled.shape)
            per_level.append(sampled)
        self.last_sampled_level_shapes = sampled_shapes
        return torch.cat(per_level, dim=-1)

    def feature_sample_with_visibility(
        self,
        feature_map,
        vertices3d,
        vertices2d,
        img_shape: tuple = None,
        depth_eps: float = 1e-3,
    ):
        fmap_for_coords = next(iter(feature_map.values())) if isinstance(feature_map, dict) else feature_map
        centers2d = self.compute_gaussian_coord2d(fmap_for_coords, vertices2d, img_shape=img_shape)
        centers3d = self.compute_gaussian_coord3d(fmap_for_coords, vertices3d)

        if isinstance(feature_map, dict):
            local_feats = self.feature_sample_multiscale(feature_map, vertices2d, img_shape=img_shape, centers2d=centers2d)
        else:
            local_feats = self.feature_sample(feature_map, vertices2d, img_shape=img_shape, centers2d=centers2d)

        if img_shape is None:
            img_shape = fmap_for_coords.shape[-2:]

        view_weights = self.compute_view_weights(
            fmap_for_coords,
            vertices3d,
            vertices2d,
            centers2d=centers2d,
            centers3d=centers3d,
            img_shape=img_shape,
            depth_eps=depth_eps,
        )
        return local_feats, view_weights, centers3d, centers2d
