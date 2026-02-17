import os
import json
import trimesh
import torch
from typing import List, Optional, Sequence, Union, Dict, Tuple
from avatar_utils.config import get_config


def load_camera_mapping(
    view_name: Union[str, Sequence[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load camera matrices from project-local cache with optional batching.

    Accepts a single view name or a list/sequence of view names. Returns
    batched tensors for viewmats (B,4,4) and Ks (B,3,3). If a cached JSON file
    is missing or unreadable, falls back to computed values for that view.

    Expects JSON files under project-root/data/THuman_cameras named
    thuman_<view>.json with keys: K (3x3), viewmat (4x4), and image_size.
    """
    # Resolve project root as two levels up from this file (src/...)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cache_dir = os.path.join(project_root, "data", "THuman_cameras")
    
    # Define device
    device = torch.device(get_config().get("sys", {}).get("device", "cpu"))

    def _load_one(vname: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = os.path.join(cache_dir, f"thuman_{vname}.json")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            K = torch.tensor(payload["K"], dtype=torch.float32)
            viewmat = torch.tensor(payload["viewmat"], dtype=torch.float32)
            # If JSON stores camera-to-world, convert to world-to-camera
            t = payload.get("type")
            if isinstance(t, str) and t.lower() == "c2w":
                try:
                    viewmat = torch.linalg.inv(viewmat)
                except Exception:
                    pass
            # Adjust intrinsics if current image size differs from cached
            try:
                W0, H0 = payload.get("image_size", [None, None])
                W1, H1 = get_config().get("data", {}).get("image_size", (W0, H0))
                if W0 and H0 and W1 and H1 and (W0 != W1 or H0 != H1):
                    sx = float(W1) / float(W0)
                    sy = float(H1) / float(H0)
                    # fx, fy scale with sy (derived from vertical FOV), cx scales with sx, cy with sy
                    K[0, 0] = K[0, 0] * sy
                    K[1, 1] = K[1, 1] * sy
                    K[0, 2] = K[0, 2] * sx
                    K[1, 2] = K[1, 2] * sy
            except Exception:
                pass
            return viewmat, K
        except Exception:
            # Fallback to on-the-fly computation when cache is missing
            vm, k = camera_mapping(vname)
            # camera_mapping already returns batched (1,...), squeeze for stacking
            return vm.squeeze(0), k.squeeze(0)

    if isinstance(view_name, str):
        vm, k = _load_one(view_name)
        return vm.unsqueeze(0).to(device), k.unsqueeze(0).to(device)
    else:
        vms = []
        ks = []
        for v in view_name:
            vm, k = _load_one(str(v))
            vms.append(vm)
            ks.append(k)
        return torch.stack(vms, dim=0).to(device), torch.stack(ks, dim=0).to(device)


def camera_mapping(view_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get camera intrinsics and extrinsics for a given view name.

    Args:
        view_name: Name of the view (e.g., 'front', 'back', 'left', 'right').

    Returns:
        viewmats: Tensor of shape (B, 4, 4) representing the batched camera extrinsic matrix.
        Ks: Tensor of shape (B, 3, 3) representing the batched camera intrinsic matrix.
    """
    # Get camera configuration from config
    cfg = get_config()
    camera_cfg = cfg.get("camera", {})
    
    # Image size used to derive intrinsics (principal point & focal length)
    width, height = cfg.get("data", {}).get("image_size", (1024, 1024))
    W, H = int(width), int(height)
    
    # Camera parameters from config
    yfov_deg = float(camera_cfg.get("yfov_deg", 45.0))
    yfov_rad = torch.tensor(yfov_deg * 3.141592653589793 / 180.0, dtype=torch.float32)
    # Focal length from vertical FOV: fy = H / (2 * tan(yfov/2)); fx = fy (square pixels)
    fy = H / (2.0 * torch.tan(yfov_rad / 2.0))
    fx = fy
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    K = torch.tensor(
        [[fx.item(), 0.0, cx], [0.0, fy.item(), cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).unsqueeze(
        0
    )  # (1,3,3)

    # Map view name to direction vector from config
    viewpoints_cfg = camera_cfg.get("viewpoints", {
        "front": [0.0, 0.0, 1.0],
        "back": [0.0, 0.0, -1.0],
        "left": [-1.0, 0.0, 0.0],
        "right": [1.0, 0.0, 0.0],
    })
    directions = {
        k: torch.tensor(v, dtype=torch.float32) 
        for k, v in viewpoints_cfg.items()
    }
    if view_name not in directions:
        raise ValueError(f"Unsupported view_name: {view_name}")

    center = torch.zeros(3, dtype=torch.float32)
    direction = directions[view_name]
    # Use canonical distance from config
    distance = float(camera_cfg.get("distance", 1.2))
    eye = center + direction * distance

    # Get up vector from config
    up_vec = camera_cfg.get("up", [0.0, 1.0, 0.0])
    up = torch.tensor(up_vec, dtype=torch.float32)
    # If up is parallel to direction, use Z-up
    if torch.allclose(torch.linalg.cross(up, direction), torch.zeros(3, dtype=torch.float32)):
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    # Build camera-to-world pose (match look_at in preprocess)
    z = eye - center
    z = z / (torch.norm(z) + 1e-8)
    x = torch.linalg.cross(up, z)
    x = x / (torch.norm(x) + 1e-8)
    y = torch.linalg.cross(z, x)

    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye

    # Extrinsics expected by rasterizer are usually world-to-camera: inverse of c2w
    w2c = torch.linalg.inv(c2w).unsqueeze(0)  # (1,4,4)
    return w2c, K


def intrinsic_matrix_from_field_of_view(
    fov_degrees: float, imshape: List[int], device: Optional[torch.device] = None
):
    imshape = torch.tensor(imshape, dtype=torch.float32, device=device)
    fov_radians = fov_degrees * torch.tensor(
        torch.pi / 180, dtype=torch.float32, device=device
    )
    larger_side = torch.max(imshape)
    focal_length = larger_side / (torch.tan(fov_radians / 2) * 2)
    _0 = torch.tensor(0, dtype=torch.float32, device=device)
    _1 = torch.tensor(1, dtype=torch.float32, device=device)

    # print(torch.stack([focal_length, _0, imshape[1] / 2], dim=-1))
    return (
        torch.stack(
            [
                focal_length,
                _0,
                (imshape[1] - 1) / 2,
                _0,
                focal_length,
                (imshape[0] - 1) / 2,
                _0,
                _0,
                _1,
            ],
            dim=-1,
        )
        .unflatten(-1, (3, 3))
        .unsqueeze(0)
    )

def look_at_viewmatrix(
    eye,          # (3,) camera position in world: [x1,y1,z1]
    target,       # (3,) point camera looks at:   [x2,y2,z2]
    up=(0.0, 1.0, 0.0),
    device=None,
    dtype=torch.float32,
    forward="-z",  # "-z" (OpenGL-style) or "+z" (some CV pipelines)
):
    """
    Returns:
      w2c: (4,4) world-to-camera view matrix
      c2w: (4,4) camera-to-world (inverse pose)
    Convention:
      - If forward == "-z": camera looks along -Z in camera space (common in graphics).
      - If forward == "+z": camera looks along +Z in camera space (common in some CV).
    """
    eye = torch.as_tensor(eye, dtype=dtype, device=device)
    target = torch.as_tensor(target, dtype=dtype, device=device)
    up = torch.as_tensor(up, dtype=dtype, device=device)

    # Forward direction in world space
    f = target - eye
    f = f / (torch.norm(f) + 1e-8)

    # Handle degenerate up (parallel to forward)
    if torch.norm(torch.linalg.cross(f, up)) < 1e-6:
        # pick an alternate up that's not parallel
        up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        if torch.norm(torch.linalg.cross(f, up)) < 1e-6:
            up = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)

    # Build an orthonormal basis
    # right
    r = torch.linalg.cross(f, up)
    r = r / (torch.norm(r) + 1e-8)
    # true up
    u = torch.linalg.cross(r, f)

    # Camera's +Z axis in world space depends on convention
    if forward == "-z":
        z_axis = -f  # camera +Z points backward
    elif forward == "+z":
        z_axis = f   # camera +Z points forward
    else:
        raise ValueError("forward must be '-z' or '+z'")

    # Camera-to-world: columns are camera axes in world coords
    c2w = torch.eye(4, dtype=dtype, device=device)
    c2w[:3, 0] = r
    c2w[:3, 1] = u
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = eye

    # World-to-camera
    w2c = torch.linalg.inv(c2w)
    return w2c, c2w

def bbox_and_4_viewmats(
    gaussian_3d: torch.Tensor,          # (N,3)
    forward: str = "-z",
    up=(0.0, 1.0, 0.0),
    margin_factor: float = 4.0,         # “4 times outside” the bbox size
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Computes 3D AABB + 4 canonical view matrices (front/back/left/right).

    AABB:
      - min_xyz (bottom-left-near-ish): (3,)
      - max_xyz (top-right-far-ish):    (3,)
      - center: (3,)
      - extent: (3,) = max-min

    Cameras:
      - target is bbox center
      - each eye starts at the center of the corresponding face of the bbox
      - then moved outward along that face normal by: margin_factor * (extent along that axis)
        (with a small epsilon fallback if that extent is ~0)

    Returns:
      viewmats: dict view->(4,4) world-to-camera
      eyes:     dict view->(3,)
      bbox:     dict with min_xyz, max_xyz, center, extent
    """
    assert gaussian_3d.ndim == 2 and gaussian_3d.shape[1] == 3, f"Expected (N,3), got {gaussian_3d.shape}"
    device = gaussian_3d.device
    dtype = gaussian_3d.dtype

    # AABB
    min_xyz = gaussian_3d.min(dim=0).values
    max_xyz = gaussian_3d.max(dim=0).values
    center = (min_xyz + max_xyz) * 0.5
    extent = (max_xyz - min_xyz)

    # Axis-aligned face centers
    # x faces
    left_face_center  = torch.stack([min_xyz[0], center[1], center[2]])
    right_face_center = torch.stack([max_xyz[0], center[1], center[2]])
    # z faces
    back_face_center  = torch.stack([center[0], center[1], min_xyz[2]])
    front_face_center = torch.stack([center[0], center[1], max_xyz[2]])

    # How far to move outside each face (use axis extent; avoid zero extent)
    eps = torch.tensor(1e-4, device=device, dtype=dtype)
    dx = torch.maximum(extent[0], eps)
    dz = torch.maximum(extent[2], eps)

    # Outward normals for faces:
    # left  face normal: -X
    # right face normal: +X
    # back  face normal: -Z
    # front face normal: +Z
    eyes = {
        "left":  left_face_center  + torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype) * (margin_factor * dx),
        "right": right_face_center + torch.tensor([ 1.0, 0.0, 0.0], device=device, dtype=dtype) * (margin_factor * dx),
        "back":  back_face_center  + torch.tensor([ 0.0, 0.0,-1.0], device=device, dtype=dtype) * (margin_factor * dz),
        "front": front_face_center + torch.tensor([ 0.0, 0.0, 1.0], device=device, dtype=dtype) * (margin_factor * dz),
    }

    # Build view matrices using the provided look_at_viewmatrix
    viewmats = {}
    for name, eye in eyes.items():
        w2c, _c2w = look_at_viewmatrix(
            eye=eye,
            target=center,
            up=up,
            device=device,
            dtype=dtype,
            forward=forward,
        )
        viewmats[name] = w2c

    bbox = {
        "min_xyz": min_xyz,
        "max_xyz": max_xyz,
        "center": center,
        "extent": extent,
    }

    return viewmats, eyes, bbox