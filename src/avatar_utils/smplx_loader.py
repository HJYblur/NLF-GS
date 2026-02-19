import torch
import trimesh
import numpy

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
    """Load 3D coordinates from a SMPLX file."""
    mesh = trimesh.load(path)
    vertices = torch.from_numpy(mesh.vertices).float()
    return vertices


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