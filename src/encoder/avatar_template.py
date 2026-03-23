import os
import trimesh
import numpy as np
import torch

# Import from the shared utils under the src root
from avatar_utils.ply_loader import load_ply, save_ply, matrix_to_quaternion
from avatar_utils.config import get as get_cfg


class GaussianData:
    def __init__(self, idx, xyz, rots, scales, opacities, shs):
        self.xyz = xyz
        self.rots = rots
        self.scales = scales
        self.opacities = opacities
        self.shs = shs
        self.parent = idx


class AvatarTemplate:
    """
    AvatarTemplate class.

    Manages a template of Gaussian primitives attached to each face of a source mesh.
    The template is initialized from the spmlx_uv mesh (self.cano_mesh_path) with a fixed
    number of Gaussians per face. On construction, if a cached avatar file (.ply) exists
    at avatar_path it is loaded; otherwise the template is generated from the mesh and
    saved for future use.

    PLY schema per vertex (Gaussian):
        x, y, z -> position
        nx, ny, nz -> normal
        f_dc_0, f_dc_1, f_dc_2 -> SH DC terms
        f_rest_0 .. f_rest_44 -> SH rest terms, Ignore for now
        opacity -> opacity
        scale_0, scale_1, scale_2 -> scale
        rot_0, rot_1, rot_2, rot_3 -> rotation
        parent_0, parent_1, parent_2 -> the 3 vertex indices that define the face

    Notes:
    - The number of Gaussians per face is currently static.
    - Generation is intended to be done once; subsequent uses load the cached file.
    """

    def __init__(self, avatar_path=None, cano_mesh_path=None, _k_num_gaussians=None):
        # Read defaults from config if not provided
        self.cano_mesh_path = (
            cano_mesh_path
            if cano_mesh_path is not None
            else get_cfg("avatar.template.cano_mesh_path", "models/smplx/smplx_uv.obj")
        )
        self.avatar_path = (
            avatar_path
            if avatar_path is not None
            else get_cfg("avatar.template.path", "./models/avatar_template.ply")
        )
        self._k_num_gaussians = int(
            _k_num_gaussians
            if _k_num_gaussians is not None
            else get_cfg("avatar.template._k_num_gaussians", 4)
        )
        self._barycentric_coords = self.get_barycentric_coords()
        self._avatar = self.load_avatar_template(mode="default")
        self._mesh_faces = self.mesh_faces  # Preload mesh faces property
        self._posenc_levels = int(get_cfg("avatar.template.posenc_levels", 4))
        self._gaussian_posenc = self._build_gaussian_positional_encoding(
            self.avatar["xyz"].to(torch.float32), self._posenc_levels
        )

    def load_cano_mesh(self):
        if not os.path.exists(self.cano_mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {self.cano_mesh_path}")
        mesh = trimesh.load(self.cano_mesh_path, process=False, maintain_order=True)
        print(
            f"Loaded mesh from {self.cano_mesh_path}, with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces."
        )
        return mesh

    def load_avatar_template(self, mode=None):
        """
        Load or generate the avatar template.

        Args:
        mode:
          1. "default": load the saved template as is.
          2. "generate": regenerate the template from the canonical mesh even if a saved file exists.
          3. "test": load the saved template but convert local coords to world coords for visualization.
          4. "anim": load the saved template but convert local coords to world coords for animated mesh visualization.

        Returns:
        _avatar: dict with keys 'xyz', 'shs', 'opacities', 'scales', 'rots', 'parents'
        """

        # Decide mode flag
        if mode is None:
            mode = get_cfg("avatar.template.mode", "default")
        if not os.path.exists(self.avatar_path):
            mode = "generate"

        # Load or generate avatar based on mode
        if mode == "default":
            assert os.path.exists(
                self.avatar_path
            ), f"Avatar template file not found: {self.avatar_path}"
            print(f"Loaded avatar template from: {self.avatar_path}")
            _avatar = load_ply(self.avatar_path, mode="default")
            return _avatar
        elif mode == "generate":
            print(f"Generating avatar template...")
            _avatar = self.generate_avatar_template()
            save_ply(_avatar, self.avatar_path)
            print("Generated and saved avatar template to:", self.avatar_path)
            return _avatar
        elif mode == "test":
            # In test mode, we reload the xyz from face based local coords to world coords for visualization
            print("Reloading avatar template in test mode for visualization...")
            cano_mesh = self.load_cano_mesh()
            _avatar = load_ply(self.avatar_path, mode="test", cano_mesh=cano_mesh)
            test_path = self.avatar_path.replace(".ply", "_test.ply")
            save_ply(_avatar, test_path)
            print("Saved test avatar template to:", test_path)
            return _avatar
        elif mode == "anim":
            # In anim mode, we reload the xyz from face based local coords to world coords for animated mesh visualization
            print(
                "Reloading avatar template in anim mode for animated mesh visualization..."
            )
            mesh_path = get_cfg(
                "avatar.template.anim_mesh_path", "tmp/output_smplx_mesh.ply"
            )
            assert os.path.exists(
                mesh_path
            ), f"Animated mesh file not found: {mesh_path}"
            animated_mesh = trimesh.load(mesh_path)
            _avatar = load_ply(self.avatar_path, mode="test", cano_mesh=animated_mesh)
            test_path = self.avatar_path.replace(".ply", "_anim.ply")
            save_ply(_avatar, test_path)
            return _avatar
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate_avatar_template(self):
        mesh = self.load_cano_mesh()
        # Example vertex/face data:
        # vertices[0]: [ 0.062714  0.2885   -0.009561]
        # face[0]: [3 1 0]
        vertices = mesh.vertices
        faces = mesh.faces

        all_xyz = []
        all_shs = []
        all_opacities = []
        all_scales = []
        all_rots = []
        all_parents = []

        for idx, face in enumerate(faces):
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # generate gaussians for each face (returns tensors)
            xyzs, shs, opacity, scales, rots = self.generate_gaussians_per_face(
                v0, v1, v2
            )

            all_xyz.append(xyzs)
            all_shs.append(shs)
            all_opacities.append(opacity)
            all_scales.append(scales)
            all_rots.append(rots)
            # Record the 3 vertex indices that define the face as parents for each gaussian
            # Shape: (_k_num_gaussians, 3)
            parent_triplet = torch.tensor(
                [int(face[0]), int(face[1]), int(face[2])], dtype=torch.int32
            )
            parents_for_gaussians = parent_triplet.unsqueeze(0).repeat(
                self._k_num_gaussians, 1
            )
            all_parents.append(parents_for_gaussians)

        if len(all_xyz) == 0:
            raise RuntimeError("No gaussians generated from mesh")

        # Concatenate lists of torch tensors into single tensors
        all_xyz = torch.cat(all_xyz, dim=0)
        all_shs = torch.cat(all_shs, dim=0)
        all_opacities = torch.cat(all_opacities, dim=0).reshape(-1, 1)
        all_scales = torch.cat(all_scales, dim=0)
        all_rots = torch.cat(all_rots, dim=0)
        all_parents = torch.cat(all_parents, dim=0)  # [N_gaussians, 3]

        data = {
            "xyz": all_xyz,
            "shs": all_shs,
            "opacities": all_opacities,
            "scales": all_scales,
            "rots": all_rots,
            "parent": all_parents,
        }

        return data

    def generate_gaussians_per_face(self, v0, v1, v2):
        num_gaussians = self._k_num_gaussians

        v0_t = torch.as_tensor(v0, dtype=torch.float32)
        v1_t = torch.as_tensor(v1, dtype=torch.float32)
        v2_t = torch.as_tensor(v2, dtype=torch.float32)

        center = (v0_t + v1_t + v2_t) / 3.0

        # Construct a local coordinate system for the face
        e1 = v1_t - v0_t
        normal = torch.linalg.cross(e1, v2_t - v0_t)
        e2 = torch.linalg.cross(normal, e1)
        e2 = e2 / (torch.norm(e2) + 1e-9)
        R_t = torch.stack([e1, e2, normal], dim=1)  # 3×3 rotation matrix (columns)

        # Calculate face area to determine Gaussian scale
        face_area = torch.norm(torch.linalg.cross(v1_t - v0_t, v2_t - v0_t)) / 2.0
        gaussian_area = face_area / float(num_gaussians)
        r = torch.sqrt(gaussian_area / float(np.pi))
        std_scale = torch.tensor([r, r, r], dtype=torch.float32)

        # Initialize tensors
        xyzs = torch.zeros((num_gaussians, 3), dtype=torch.float32)
        shs = torch.zeros((num_gaussians, 3), dtype=torch.float32)
        opacity = torch.zeros((num_gaussians, 1), dtype=torch.float32)
        scales = torch.zeros((num_gaussians, 3), dtype=torch.float32)
        rots = torch.zeros((num_gaussians, 4), dtype=torch.float32)

        # We still use the existing matrix_to_quaternion helper which expects numpy input;
        # convert R_t to numpy just for quaternion computation (cheap for small matrices).
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

        # Shape: (num_gaussians, 3), (num_gaussians, 3), (num_gaussians, 1), (num_gaussians, 3), (num_gaussians, 4)
        return xyzs, shs, opacity, scales, rots

    def get_barycentric_coords(self):
        if not hasattr(self, "_barycentric_coords"):
            B4_list = get_cfg(
                "avatar.template.barycentric_coords",
                [
                    (0.6, 0.2, 0.2),
                    (0.2, 0.6, 0.2),
                    (0.2, 0.2, 0.6),
                    (1 / 3, 1 / 3, 1 / 3),
                ],
            )
            B4_list = [[float(x) for x in row] for row in B4_list]
            self._barycentric_coords = torch.tensor(B4_list, dtype=torch.float32)
        return self._barycentric_coords

    def _build_gaussian_positional_encoding(
        self, xyz: torch.Tensor, levels: int
    ) -> torch.Tensor:
        """Build sinusoidal PE per Gaussian using template coordinates."""
        if levels <= 0:
            return xyz.new_zeros((xyz.shape[0], 0))
        freqs = (
            (2.0 ** torch.arange(levels, dtype=xyz.dtype, device=xyz.device))
            * torch.pi
        )
        xyz_f = xyz.unsqueeze(-1) * freqs.view(1, 1, levels)  # (N,3,L)
        sin_part = torch.sin(xyz_f).reshape(xyz.shape[0], -1)
        cos_part = torch.cos(xyz_f).reshape(xyz.shape[0], -1)
        return torch.cat([xyz, sin_part, cos_part], dim=-1)

    @property
    def gaussian_positional_encoding(self):
        return self._gaussian_posenc

    @property
    def barycentric_coords(self):
        return self._barycentric_coords

    @property
    def total_gaussians_num(self):
        return self.avatar["xyz"].shape[0]

    @property
    def avatar(self):
        return self._avatar

    @property
    def parents(self):
        return self.avatar["parent"]

    @property
    def mesh_faces(self):
        if not hasattr(self, "_mesh_faces"):
            mesh = self.load_cano_mesh()
            self._mesh_faces = torch.as_tensor(mesh.faces, dtype=torch.int32)
        return self._mesh_faces


# if __name__ == "__main__":
#     avatar_template = AvatarTemplate()
#     avatar_generated = avatar_template.load_avatar_template(mode="generate")
#     print("Avatar template generated and saved.")
#     avatar_test = avatar_template.load_avatar_template(mode="test")
#     print("Avatar template test file generated.")
#     avatar_anim = avatar_template.load_avatar_template(mode="anim")
#     print("Avatar template anim file generated.")
