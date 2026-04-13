import math
import os
from collections import namedtuple
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import trimesh
from plyfile import PlyData, PlyElement


# Simple container for loaded Gaussian data (now includes parent index)
GaussianData = namedtuple(
    "GaussianData", ["xyz", "rots", "scales", "opacities", "shs", "parent"]
)


def matrix_to_quaternion(matrix):
    # Convert a rotation matrix to a quaternion.
    m = matrix.reshape(3, 3)
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def load_cano_mesh_face_center(self):
    if not os.path.exists(self.cano_mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {self.cano_mesh_path}")
    mesh = trimesh.load(self.cano_mesh_path)
    vertices, faces = mesh.vertices, mesh.faces

    face_center = np.zeros((len(faces), 3), dtype=np.float32)
    for idx, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        center = (v0 + v1 + v2) / 3.0
        face_center[idx, :] = center
    return face_center


def load_ply(path, mode="default", cano_mesh=None, return_torch: bool = True):
    """Load Gaussian template PLY saved by save_ply.

    This loader is tolerant: it reads whatever fields are present (DC SH, optional
    extra SH coefficients, scale_*, rot_*, optional parent) and returns a
    GaussianData object. Coordinates are negated back to the original sign
    because save_ply stores x,y,z with flipped signs.
    """
    plydata = PlyData.read(path)
    elem = plydata.elements[0]
    names = elem.data.dtype.names

    # Positions:
    # In "test" mode, reload xyz from face-based local coords to world coords for visualization.
    # We require that the PLY contains explicit parent triplets: parent_0,parent_1,parent_2.
    if mode == "test":
        if cano_mesh is None:
            raise ValueError("cano_mesh must be provided in test mode to reload xyz.")
        # Convert stored local offsets back to world coords per-gaussian using its 3 parent vertices
        vertices = cano_mesh.vertices
        names = elem.data.dtype.names
        if not all(k in names for k in ("parent_0", "parent_1", "parent_2")):
            raise ValueError(
                "PLY must contain parent_0,parent_1,parent_2 fields to reconstruct world coords."
            )
        for idx in range(len(elem.data)):
            i0 = int(elem.data["parent_0"][idx])
            i1 = int(elem.data["parent_1"][idx])
            i2 = int(elem.data["parent_2"][idx])
            center = (vertices[i0] + vertices[i1] + vertices[i2]) / 3.0
            elem.data["x"][idx] = elem.data["x"][idx] + center[0]
            elem.data["y"][idx] = elem.data["y"][idx] + center[1]
            elem.data["z"][idx] = elem.data["z"][idx] + center[2]

    # Else in "default" mode, just load it. We need it in local coords for further processing
    xyz = np.stack(
        (np.asarray(elem["x"]), np.asarray(elem["y"]), np.asarray(elem["z"])),
        axis=1,
    ).astype(np.float32)

    # Opacity stored directly (0..1), keep as column vector
    opacities = np.asarray(elem["opacity"])[..., np.newaxis].astype(np.float32)

    # SH DC (f_dc_0..2) if present, otherwise zeros
    dc = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    for i in range(3):
        key = f"f_dc_{i}"
        if key in names:
            dc[:, i] = np.asarray(elem[key]).astype(np.float32)

    # Optional extra SH coefficients (f_rest_*) – read if present and keep order
    extra_f_names = [n for n in names if n.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    if len(extra_f_names) > 0:
        extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
        for idx, k in enumerate(extra_f_names):
            extra[:, idx] = np.asarray(elem[k]).astype(np.float32)
        shs = np.concatenate([dc, extra], axis=1)
    else:
        shs = dc

    # scales (scale_0.. ) if present; default to ones
    scale_names = [n for n in names if n.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    if len(scale_names) > 0:
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, k in enumerate(scale_names):
            scales[:, idx] = np.asarray(elem[k]).astype(np.float32)
    else:
        scales = np.ones((xyz.shape[0], 3), dtype=np.float32)

    # rotations (rot_0..rot_M) if present
    rot_names = [n for n in names if n.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    if len(rot_names) > 0:
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, k in enumerate(rot_names):
            rots[:, idx] = np.asarray(elem[k]).astype(np.float32)
        # normalize
        nrm = np.linalg.norm(rots, axis=-1, keepdims=True)
        nrm[nrm == 0] = 1.0
        rots = (rots / nrm).astype(np.float32)
    else:
        rots = np.zeros((xyz.shape[0], 4), dtype=np.float32)

    # parent indices: require triplet (parent_0, parent_1, parent_2)
    if not all(n in names for n in ("parent_0", "parent_1", "parent_2")):
        raise ValueError("PLY missing required parent_0,parent_1,parent_2 fields")
    parent = np.stack(
        [
            np.asarray(elem["parent_0"]).astype(np.int32),
            np.asarray(elem["parent_1"]).astype(np.int32),
            np.asarray(elem["parent_2"]).astype(np.int32),
        ],
        axis=1,
    )

    # Build result as GaussianData namedtuple for attribute access (xyz, rots, scales, opacities, shs, parent)
    if return_torch:
        return {
            "xyz": torch.from_numpy(xyz).to(torch.float32),
            "shs": torch.from_numpy(shs).to(torch.float32),
            "opacities": torch.from_numpy(opacities).to(torch.float32),
            "scales": torch.from_numpy(scales).to(torch.float32),
            "rots": torch.from_numpy(rots).to(torch.float32),
            "parent": torch.from_numpy(parent).to(torch.int32),
        }

    return {
        "xyz": xyz,
        "shs": shs,
        "opacities": opacities,
        "scales": scales,
        "rots": rots,
        "parent": parent,
    }


def _infer_sh_degree_from_flat_dim(K: int) -> int:
    """Infer SH degree d from flat width K = (d+1)^2 * 3 (RGB per spherical-harmonic basis)."""
    if K < 3 or K % 3 != 0:
        raise ValueError(f"SH flat dim must be >= 3 and divisible by 3, got {K}")
    n_basis = K // 3
    r = int(round(math.sqrt(n_basis)))
    if r * r != n_basis:
        raise ValueError(
            f"SH flat dim implies non-square basis count: K={K} -> n_basis={n_basis}"
        )
    return r - 1


def _split_sh_dc_rest(sh_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split flat (N, K) SH into DC (N, 3) and rest (N, (K-3)) in gsplat / Inria layout."""
    N, K = sh_flat.shape
    d = _infer_sh_degree_from_flat_dim(K)
    n_basis = (d + 1) ** 2
    sh = sh_flat.reshape(N, n_basis, 3)
    dc = sh[:, 0, :].astype(np.float32)
    rest = sh[:, 1:, :].reshape(N, -1).astype(np.float32)
    return dc, rest


def save_ply(
    data,
    path: str,
    *,
    log_scales: bool = False,
    include_parent: Optional[bool] = None,
):
    """Save Gaussian template to a PLY file (DC-only ``shs``), or full-SH export when ``shs`` is wider.

    Original behavior: ``shs`` shape (N, 3) — opacity, ``f_dc_*``, scales, rot, parent.

    Extended: ``shs`` shape ``(N, (d+1)^2 * 3)`` for any valid degree ``d`` — writes
    ``x,y,z``, ``nx,ny,nz``, ``f_dc_*``, ``f_rest_*``, ``opacity``, ``scale_*``, ``rot_*``,
    and optional ``parent_*`` (Inria / 3DGS-friendly). Use ``log_scales=True`` for log-space
    scales in the file.
    """

    # Normalize accessors (accept numpy arrays or torch tensors)
    def to_numpy(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        return np.asarray(v)

    if hasattr(data, "xyz"):
        xyz = to_numpy(getattr(data, "xyz"))
    elif isinstance(data, dict):
        xyz = to_numpy(data["xyz"])
    else:
        raise ValueError("Unsupported data type for save_ply")

    N = xyz.shape[0]

    def get_field(name, default, shape=None):
        if hasattr(data, name):
            v = to_numpy(getattr(data, name))
        elif isinstance(data, dict) and name in data:
            v = to_numpy(data[name])
        else:
            v = np.full(
                (N,) + (() if shape is None else shape), default, dtype=np.float32
            )
        return v

    shs = get_field("shs", 0.5, shape=(3,)).reshape(N, -1)  # DC RGB or full flat SH
    opacities = get_field("opacities", 1.0).reshape(N, -1)
    scales = get_field("scales", 1.0, shape=(3,)).reshape(N, -1)
    rots = get_field("rots", 0.0, shape=(4,)).reshape(N, -1)

    parents = None
    if hasattr(data, "parent"):
        parents = to_numpy(getattr(data, "parent"))
    elif isinstance(data, dict) and "parent" in data:
        parents = to_numpy(data["parent"])

    # --- Full spherical harmonics (decoded Gaussians): any degree d with K = (d+1)^2 * 3 ---
    if shs.shape[1] > 3:
        if include_parent is None:
            include_parent = parents is not None
        if include_parent and parents is None:
            raise ValueError(
                "include_parent=True requires a 'parent' field with mesh vertex indices"
            )
        dc, rest = _split_sh_dc_rest(shs.astype(np.float32))
        scales_out = (
            np.log(np.maximum(scales.astype(np.float64), 1e-10)).astype(np.float32)
            if log_scales
            else scales.astype(np.float32)
        )
        parent_count = int(parents.shape[1]) if (include_parent and parents is not None) else 0

        vertex_dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
        ]
        vertex_dtype += [(f"f_dc_{i}", "f4") for i in range(3)]
        for i in range(rest.shape[1]):
            vertex_dtype.append((f"f_rest_{i}", "f4"))
        vertex_dtype.append(("opacity", "f4"))
        for i in range(scales_out.shape[1]):
            vertex_dtype.append((f"scale_{i}", "f4"))
        for i in range(rots.shape[1]):
            vertex_dtype.append((f"rot_{i}", "f4"))
        if parent_count > 0:
            vertex_dtype += [(f"parent_{i}", "i4") for i in range(parent_count)]

        vertices = np.empty(N, dtype=vertex_dtype)
        vertices["x"] = xyz[:, 0].astype(np.float32)
        vertices["y"] = xyz[:, 1].astype(np.float32)
        vertices["z"] = xyz[:, 2].astype(np.float32)
        vertices["nx"] = 0.0
        vertices["ny"] = 0.0
        vertices["nz"] = 0.0
        vertices["f_dc_0"] = dc[:, 0]
        vertices["f_dc_1"] = dc[:, 1]
        vertices["f_dc_2"] = dc[:, 2]
        for i in range(rest.shape[1]):
            vertices[f"f_rest_{i}"] = rest[:, i]
        vertices["opacity"] = opacities.reshape(-1).astype(np.float32)
        for i in range(scales_out.shape[1]):
            vertices[f"scale_{i}"] = scales_out[:, i]
        for i in range(rots.shape[1]):
            vertices[f"rot_{i}"] = rots[:, i]
        if parent_count > 0:
            for i in range(parent_count):
                vertices[f"parent_{i}"] = parents[:, i].astype(np.int32)

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        ply_el = PlyElement.describe(vertices, "vertex")
        PlyData([ply_el], text=False).write(path)
        return

    # --- Original template path: DC-only SH (N,3) ---
    if parents is None:
        raise ValueError(
            "save_ply requires a 'parent' field with shape (N,3) containing vertex indices"
        )

    parents = np.asarray(parents)
    if parents.ndim != 2:
        raise ValueError(
            "'parent' must be a 2D array of shape (N,P) with vertex indices"
        )
    parent_count = int(parents.shape[1])

    # Build structured dtype: position, opacity, SH DC, scales, rot, parent_{i}
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
    ]
    # SH DC
    vertex_dtype += [(f"f_dc_{i}", "f4") for i in range(3)]
    # scales (scale_0..)
    for i in range(scales.shape[1]):
        vertex_dtype.append((f"scale_{i}", "f4"))
    # rotation (rot_0..rot_M)
    for i in range(rots.shape[1]):
        vertex_dtype.append((f"rot_{i}", "f4"))
    # parents
    vertex_dtype += [(f"parent_{i}", "i4") for i in range(parent_count)]

    # Create structured array with final dtype and fill fields directly (no intermediate copy)
    vertices = np.empty(N, dtype=vertex_dtype)
    vertices["x"] = xyz[:, 0].astype(np.float32)
    vertices["y"] = xyz[:, 1].astype(np.float32)
    vertices["z"] = xyz[:, 2].astype(np.float32)
    vertices["opacity"] = opacities.reshape(-1).astype(np.float32)

    # SH DC
    vertices["f_dc_0"] = shs[:, 0].astype(np.float32)
    vertices["f_dc_1"] = shs[:, 1].astype(np.float32)
    vertices["f_dc_2"] = shs[:, 2].astype(np.float32)

    # scales
    for i in range(scales.shape[1]):
        vertices[f"scale_{i}"] = scales[:, i].astype(np.float32)

    # rotation
    for i in range(rots.shape[1]):
        vertices[f"rot_{i}"] = rots[:, i].astype(np.float32)

    # parents
    for i in range(parent_count):
        vertices[f"parent_{i}"] = parents[:, i].astype(np.int32)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ply_el = PlyElement.describe(vertices, "vertex")
    PlyData([ply_el], text=False).write(path)


def reconstruct_gaussian_avatar_as_ply(
    xyz: Union[torch.Tensor, np.ndarray],
    gaussian_params: Dict[str, Any],
    template: Optional[Dict[str, Any]],
    output_path: str,
    *,
    log_scales: bool = True,
    include_parent: bool = True,
) -> Dict[str, Any]:
    """
    Reconstruct a Gaussian avatar from the given parameters and return (and save) as a PLY file.
    Full ``sh`` tensors (any supported SH degree) are written as ``f_dc_*`` + ``f_rest_*``.
    """
    _out_dir = os.path.dirname(output_path)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    scales = gaussian_params["scales"]
    rots = gaussian_params["rotation"]
    alphas = gaussian_params["alpha"]
    shs = gaussian_params["sh"]

    ply_data: Dict[str, Any] = {
        "xyz": xyz,
        "scales": scales,
        "rots": rots,
        "opacities": alphas,
        "shs": shs,
    }
    if include_parent:
        if template is None or "parent" not in template:
            raise ValueError("include_parent=True requires template['parent'] (face vertex indices).")
        ply_data["parent"] = template["parent"]

    save_ply(
        ply_data,
        output_path,
        log_scales=log_scales,
        include_parent=include_parent,
    )
    return ply_data
