import logging
import pickle
from pathlib import Path

import numpy
import torch
import trimesh

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached SMPLX body model (loaded once per model_path + gender + num_pca_comps)
# ---------------------------------------------------------------------------
_SMPLX_MODEL_CACHE: dict = {}


def _get_smplx_model(
    model_path: str = "models/smplx",
    gender: str = "neutral",
    num_pca_comps: int = 12,
):
    """Return a cached :class:`smplx.SMPLX` body-model instance."""
    key = (model_path, gender, num_pca_comps)
    if key not in _SMPLX_MODEL_CACHE:
        import smplx as _smplx

        _SMPLX_MODEL_CACHE[key] = _smplx.SMPLX(
            model_path=model_path,
            gender=gender,
            use_pca=True,
            num_pca_comps=num_pca_comps,
            flat_hand_mean=True,
        )
        _logger.info("Loaded SMPLX model from %s (gender=%s, pca=%d)", model_path, gender, num_pca_comps)
    return _SMPLX_MODEL_CACHE[key]


def load_smplx_vertices(path: str, camera_intrinsics, w2c):
    """Load 3D vertices and its 2D projections from a SMPLX file."""
    vertices = load_smplx_coord3d(path)
    # Check if camera_intrinsics and w2c are tensor
    if isinstance(camera_intrinsics, numpy.ndarray):
        camera_intrinsics = torch.from_numpy(camera_intrinsics).float()
    if isinstance(w2c, numpy.ndarray):
        w2c = torch.from_numpy(w2c).float()
    vertices_2d = vertices_3d_to_2d(vertices, camera_intrinsics, w2c)
    return vertices, vertices_2d


def load_smplx_coord3d(path: str):
    """Load 3D SMPLX vertices in **standard SMPLX vertex ordering**.

    Accepts either:
    * A path to ``smplx_param.pkl`` – runs the SMPLX body model forward pass
      and returns vertices in the canonical SMPLX vertex ordering (matching the
      face indices used by the avatar template).
    * A path to an ``.obj`` mesh file – falls back to loading via trimesh.
      **Warning**: trimesh may reorder vertices, breaking the correspondence
      with the avatar template's parent indices.

    The preferred input is the ``.pkl`` parameter file.
    """
    p = Path(path)

    if p.suffix == ".pkl":
        return _load_smplx_coord3d_from_params(str(p))

    # Legacy / fallback: load directly from mesh file
    _logger.warning(
        "Loading SMPLX vertices from %s via trimesh – vertex ordering may "
        "not match the avatar template.  Prefer passing smplx_param.pkl.",
        path,
    )
    mesh = trimesh.load(path)
    vertices = torch.from_numpy(mesh.vertices).float()
    return vertices


def _load_smplx_coord3d_from_params(
    pkl_path: str,
    model_path: str = "models/smplx",
    gender: str = "neutral",
    num_pca_comps: int = 12,
) -> torch.Tensor:
    """Generate 3D vertices from SMPLX parameters in standard ordering.

    This guarantees the returned vertex indices are consistent with the
    standard SMPLX topology (and therefore with the avatar template's
    ``parents`` tensor).
    """
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)

    model = _get_smplx_model(model_path, gender, num_pca_comps)

    with torch.no_grad():
        output = model(
            betas=torch.from_numpy(params["betas"]).float(),
            global_orient=torch.from_numpy(params["global_orient"]).float(),
            body_pose=torch.from_numpy(params["body_pose"]).float(),
            left_hand_pose=torch.from_numpy(params["left_hand_pose"]).float(),
            right_hand_pose=torch.from_numpy(params["right_hand_pose"]).float(),
            jaw_pose=torch.from_numpy(params["jaw_pose"]).float(),
            leye_pose=torch.from_numpy(params["leye_pose"]).float(),
            reye_pose=torch.from_numpy(params["reye_pose"]).float(),
            expression=torch.from_numpy(params["expression"]).float(),
        )

    verts = output.vertices[0]  # (Nv, 3), standard ordering

    # Apply scale and translation stored alongside the SMPLX params
    if "scale" in params:
        scale = float(params["scale"].squeeze())
        verts = verts * scale
    if "translation" in params:
        translation = torch.from_numpy(
            numpy.asarray(params["translation"], dtype=numpy.float32)
        ).reshape(1, 3)
        verts = verts + translation

    return verts  # (Nv, 3) float32


def vertices_3d_to_2d(
        vertices3d: torch.FloatTensor, 
        camera_intrinsics: torch.FloatTensor, 
        w2c: torch.FloatTensor
    ) -> torch.FloatTensor:
    """Project 3D vertices to 2D using camera intrinsics and extrinsics.

    Args:
        vertices3d: [N, 3] tensor of 3D vertices in world space
        camera_intrinsics: [3, 3] camera intrinsics matrix K
        w2c: [4, 4] world-to-camera transformation matrix
             This is the 'viewmat' from gsplat which already has OpenGL convention
             baked into the rotation matrix (R[2,2]=-1 for front camera)

    Returns:
        vertices2d: [N, 2] tensor of 2D projected pixel coordinates

    """
    # Align all tensors to the same device/dtype as vertices to avoid CPU/GPU mismatch
    camera_intrinsics = camera_intrinsics.to(device=vertices3d.device, dtype=vertices3d.dtype)
    w2c = w2c.to(device=vertices3d.device, dtype=vertices3d.dtype)

    # Step 1: Transform vertices from world space to camera space
    # Using standard matrix multiplication: P_cam = R @ P_world + t
    R = w2c[:3, :3]  # [3, 3] rotation (already has OpenGL Z-flip baked in)
    t = w2c[:3, 3:4]  # [3, 1] translation

    # Transform: P_cam = (P_world @ R.T) + t.T
    vertices_cam = (vertices3d @ R.T) + t.T  # [N, 3]

    # Step 2: Project 3D points in camera space to 2D: p = K @ P_cam
    # NO Z-flip needed - the w2c matrix from gsplat already handles the convention
    projected = (camera_intrinsics @ vertices_cam.T).T  # [N, 3]

    # Step 3: Normalize by depth (z-coordinate) to get pixel coordinates
    vertices2d = projected[:, :2] / projected[:, 2:3]  # [N, 2]

    return vertices2d