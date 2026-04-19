import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

from avatar_utils.config import get as get_cfg
from avatar_utils.ply_loader import load_ply, matrix_to_quaternion, save_ply


def _cfg_req(path: str) -> Any:
    v = get_cfg(path)
    if v is None:
        raise ValueError(f"Missing required config key: {path}")
    return v


class AvatarTemplate:
    """Gaussian primitives rigidly attached to each face of the canonical SMPL-X UV mesh.

    The template is built from ``cano_mesh_path`` with a fixed count of Gaussians per face.
    If ``avatar_path`` already exists it is loaded; otherwise the template is generated
    from the canonical mesh and saved for reuse.

    PLY schema per Gaussian:
        ``xyz``, normals, SH DC / rest, opacity, scales, quaternion rotation, and
        ``parent_*``: the three mesh vertex indices of the supporting triangle.
    """

    def __init__(
        self,
        avatar_path: Optional[str] = None,
        cano_mesh_path: Optional[str] = None,
        k_num_gaussians: Optional[int] = None,
    ) -> None:
        self.cano_mesh_path = (
            str(cano_mesh_path)
            if cano_mesh_path is not None
            else str(_cfg_req("avatar_template.cano_mesh_path"))
        )
        self.avatar_path = (
            str(avatar_path)
            if avatar_path is not None
            else str(_cfg_req("avatar_template.path"))
        )
        self.k_num_gaussians = int(
            k_num_gaussians
            if k_num_gaussians is not None
            else int(_cfg_req("avatar_template.k_num_gaussians"))
        )
        self._barycentric_coords = self.get_barycentric_coords()
        self._avatar = self.load_avatar_template(mode="default")
        self._mesh_faces = self.mesh_faces  # Preload mesh faces property

    def load_cano_mesh(self):
        if not os.path.exists(self.cano_mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {self.cano_mesh_path}")
        return trimesh.load(self.cano_mesh_path, process=False, maintain_order=True)

    def load_avatar_template(self, mode: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Load or (re)build the avatar template PLY.

        ``mode``:
            ``default``: load existing ``avatar_path``.
            ``generate``: rebuild from the canonical mesh and write ``avatar_path``.
            ``test``: load PLY, map locals to world using the canonical mesh, save ``*_test.ply``.
            ``anim``: same as ``test`` but using ``avatar_template.anim_mesh_path`` as geometry.

        Returns:
            Dict with keys ``xyz``, ``shs``, ``opacities``, ``scales``, ``rots``, ``parent``.
        """
        if mode is None:
            mode = str(_cfg_req("avatar_template.mode"))
        if not os.path.exists(self.avatar_path):
            mode = "generate"

        if mode == "default":
            assert os.path.exists(self.avatar_path), (
                f"Avatar template file not found: {self.avatar_path}"
            )
            return load_ply(self.avatar_path, mode="default")

        if mode == "generate":
            avatar = self.generate_avatar_template()
            save_ply(avatar, self.avatar_path)
            return avatar

        if mode == "test":
            cano_mesh = self.load_cano_mesh()
            avatar = load_ply(self.avatar_path, mode="test", cano_mesh=cano_mesh)
            test_path = self.avatar_path.replace(".ply", "_test.ply")
            save_ply(avatar, test_path)
            return avatar

        if mode == "anim":
            mesh_path = str(_cfg_req("avatar_template.anim_mesh_path"))
            assert os.path.exists(mesh_path), f"Animated mesh file not found: {mesh_path}"
            animated_mesh = trimesh.load(mesh_path)
            avatar = load_ply(self.avatar_path, mode="test", cano_mesh=animated_mesh)
            out_path = self.avatar_path.replace(".ply", "_anim.ply")
            save_ply(avatar, out_path)
            return avatar

        raise ValueError(f"Unknown mode: {mode}")

    def generate_avatar_template(self) -> Dict[str, torch.Tensor]:
        mesh = self.load_cano_mesh()
        vertices = mesh.vertices
        faces = mesh.faces

        all_xyz: List[torch.Tensor] = []
        all_shs: List[torch.Tensor] = []
        all_opacities: List[torch.Tensor] = []
        all_scales: List[torch.Tensor] = []
        all_rots: List[torch.Tensor] = []
        all_parents: List[torch.Tensor] = []

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            xyzs, shs, opacity, scales, rots = self.generate_gaussians_per_face(v0, v1, v2)

            all_xyz.append(xyzs)
            all_shs.append(shs)
            all_opacities.append(opacity)
            all_scales.append(scales)
            all_rots.append(rots)

            parent_triplet = torch.tensor(
                [int(face[0]), int(face[1]), int(face[2])], dtype=torch.int32
            )
            parents_for_gaussians = parent_triplet.unsqueeze(0).repeat(self.k_num_gaussians, 1)
            all_parents.append(parents_for_gaussians)

        if len(all_xyz) == 0:
            raise RuntimeError("No gaussians generated from mesh")

        return {
            "xyz": torch.cat(all_xyz, dim=0),
            "shs": torch.cat(all_shs, dim=0),
            "opacities": torch.cat(all_opacities, dim=0).reshape(-1, 1),
            "scales": torch.cat(all_scales, dim=0),
            "rots": torch.cat(all_rots, dim=0),
            "parent": torch.cat(all_parents, dim=0),
        }

    def generate_gaussians_per_face(
        self, v0, v1, v2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_gaussians = self.k_num_gaussians

        v0_t = torch.as_tensor(v0, dtype=torch.float32)
        v1_t = torch.as_tensor(v1, dtype=torch.float32)
        v2_t = torch.as_tensor(v2, dtype=torch.float32)

        center = (v0_t + v1_t + v2_t) / 3.0

        e1 = v1_t - v0_t
        normal = torch.linalg.cross(e1, v2_t - v0_t)
        e2 = torch.linalg.cross(normal, e1)
        e2 = e2 / (torch.norm(e2) + 1e-9)
        R_t = torch.stack([e1, e2, normal], dim=1)

        face_area = torch.norm(torch.linalg.cross(v1_t - v0_t, v2_t - v0_t)) / 2.0
        gaussian_area = face_area / float(num_gaussians)
        r = torch.sqrt(gaussian_area / float(np.pi))
        r = torch.log(r)
        std_scale = torch.tensor([r, r, r], dtype=torch.float32)

        xyzs = torch.zeros((num_gaussians, 3), dtype=torch.float32)
        shs = torch.zeros((num_gaussians, 3), dtype=torch.float32)
        opacity = torch.zeros((num_gaussians, 1), dtype=torch.float32)
        scales = torch.zeros((num_gaussians, 3), dtype=torch.float32)
        rots = torch.zeros((num_gaussians, 4), dtype=torch.float32)

        R_np = R_t.cpu().numpy()
        quat = matrix_to_quaternion(R_np)

        B4 = self.get_barycentric_coords()
        for i in range(num_gaussians):
            bary = B4[i]
            xyzs[i, :] = bary[0] * v0_t + bary[1] * v1_t + bary[2] * v2_t - center
            shs[i, :] = 0.5
            opacity[i, 0] = 0.6
            scales[i, :] = std_scale
            rots[i, :] = torch.from_numpy(quat).to(torch.float32)

        return xyzs, shs, opacity, scales, rots

    def get_barycentric_coords(self) -> torch.Tensor:
        if not hasattr(self, "_barycentric_coords"):
            raw = _cfg_req("avatar_template.barycentric_coords")
            rows = [[float(x) for x in row] for row in raw]
            self._barycentric_coords = torch.tensor(rows, dtype=torch.float32)
        return self._barycentric_coords

    @property
    def barycentric_coords(self) -> torch.Tensor:
        return self._barycentric_coords

    @property
    def total_gaussians_num(self) -> int:
        return self.avatar["xyz"].shape[0]

    @property
    def avatar(self) -> Dict[str, torch.Tensor]:
        return self._avatar

    @property
    def parents(self) -> torch.Tensor:
        return self.avatar["parent"]

    @property
    def mesh_faces(self) -> torch.Tensor:
        if not hasattr(self, "_mesh_faces"):
            mesh = self.load_cano_mesh()
            self._mesh_faces = torch.as_tensor(mesh.faces, dtype=torch.int32)
        return self._mesh_faces
