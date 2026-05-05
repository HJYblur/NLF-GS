import copy
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy
import torch
import trimesh

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
        with open(p, "rb") as f:
            params = pickle.load(f)
        return _vertices_from_smplx_param_dict(params)

    # Legacy / fallback: load directly from mesh file
    mesh = trimesh.load(path)
    vertices = torch.from_numpy(mesh.vertices).float()
    return vertices


# Zeroed for ``load_smplx_coord3d_tpose`` / ``copy_smplx_params_tpose_rest`` (body/hand; jaw/eyes from pickle).
_SMPLX_POSE_KEYS = (
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
)


def load_smplx_params_dict(pkl_path: str) -> dict:
    """Load raw ``smplx_param.pkl`` dict (numpy arrays)."""
    with open(Path(pkl_path), "rb") as f:
        return pickle.load(f)


def _coerce_smplx_params_dict(raw: dict) -> dict:
    """Normalize a mapping (e.g. from JSON) to numpy float32 arrays like ``smplx_param.pkl``."""
    out: dict = {}
    for key, val in raw.items():
        if key == "transl":
            continue
        if isinstance(val, (list, tuple)):
            out[key] = numpy.asarray(val, dtype=numpy.float32)
        elif isinstance(val, numpy.ndarray):
            out[key] = val.astype(numpy.float32, copy=False)
        elif numpy.isscalar(val):
            out[key] = numpy.asarray([val], dtype=numpy.float32)
    if "translation" not in out and "transl" in raw:
        t = numpy.asarray(raw["transl"], dtype=numpy.float32).reshape(-1)
        if t.size >= 3:
            out["translation"] = t[:3].reshape(1, 3)
    return out


def load_smplx_params_from_path(path: str) -> dict:
    """Load SMPL-X params from ``.pkl`` or ``.json`` (same semantic keys as ``smplx_param.pkl``).

    JSON exports often use ``transl`` instead of ``translation``; that is normalized here so
    :func:`vertices_from_smplx_param_dict` can run on the dict alone. For animation retargeting,
    use :func:`merge_subject_identity_with_driver_pose` so world ``translation`` / ``scale`` come
    from the subject, not the motion file.
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise TypeError(f"Expected JSON object in {path}, got {type(raw)}")
        return _coerce_smplx_params_dict(raw)
    return load_smplx_params_dict(str(p))


def _motion_file_sort_key(path: Path) -> tuple:
    """Stable ordering for motion frames: digit runs in the stem (left→right), then lexical.

    Works for arbitrary layouts (``frame_00012.json``, ``seq-f00210.pkl``, ``clip03_part02.json``, …).
    """
    stem = path.stem
    nums = tuple(int(m) for m in re.findall(r"\d+", stem))
    if nums:
        return (0, nums, path.name.lower())
    return (1, path.name.lower())


def smplx_motion_sequence_paths(motion_path: str | Path) -> list[Path]:
    """Resolve SMPL-X motion inputs: one ``.json`` / ``.pkl`` file, or a directory of per-frame files.

    For a **directory**, collects every non-hidden ``*.json`` and ``*.pkl`` at the top level (no subfolders).
    Files are sorted by **integer tuples** extracted from digit runs in the filename stem (e.g.
    ``pose_12.json`` → ``(12,)``, ``clip_3_frame_004.json`` → ``(3, 4)``). Names with no digits sort
    alphabetically after digit-based names.

    Returns:
        Non-empty list of absolute paths.
    """
    root = Path(motion_path).expanduser()
    if root.is_file():
        if root.suffix.lower() not in (".json", ".pkl"):
            raise ValueError(f"SMPL-X motion file must be .json or .pkl, got {root}")
        return [root.resolve()]

    if not root.is_dir():
        raise FileNotFoundError(f"SMPL-X motion path not found: {root}")

    seen: set[str] = set()
    uniq: list[Path] = []
    for pattern in ("*.json", "*.pkl"):
        for p in root.glob(pattern):
            if not p.is_file() or p.name.startswith("."):
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(Path(key))

    if not uniq:
        raise FileNotFoundError(
            f"No SMPL-X motion files (*.json or *.pkl) under {root}"
        )

    uniq.sort(key=_motion_file_sort_key)
    return uniq


# Pose tensors copied from a motion / driver pickle when retargeting onto a subject identity.
_SMPLX_DRIVER_POSE_KEYS = (
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
)


def merge_subject_identity_with_driver_pose(subject: dict, driver: dict) -> dict:
    """Keep shape and normalization from ``subject``; apply only pose articulation from ``driver``.

    Identity (shape, scale, world placement, facial identity in the SMPL-X sense) is taken from
    ``subject`` — typically processed ``smplx_param.pkl`` for the avatar. ``driver`` may be raw
    animation parameters (often different ``betas``, ``scale``, ``translation`` / ``transl``);
    only :data:`_SMPLX_DRIVER_POSE_KEYS` are overlaid so Gaussians stay in the same normalized
    frame as training while following the driver's motion.

    Args:
        subject: Dict with ``betas``, ``expression``, optional ``scale`` / ``translation``, and pose keys.
        driver: Same schema; pose arrays must match ``subject`` shapes (same PCA layout).

    Returns:
        New dict suitable for :func:`vertices_from_smplx_param_dict`.
    """
    out = copy.deepcopy(subject)
    for key in _SMPLX_DRIVER_POSE_KEYS:
        if key not in driver:
            continue
        d = numpy.asarray(driver[key], dtype=numpy.float32)
        if key not in out:
            out[key] = d.copy()
            continue
        s = numpy.asarray(out[key], dtype=numpy.float32)
        if d.shape != s.shape:
            raise ValueError(
                f"Driver pose '{key}' has shape {d.shape}, subject has {s.shape}. "
                "Use the same SMPL-X model settings (e.g. num_pca_comps) for both."
            )
        out[key] = d.copy()
    return out


def copy_smplx_params_tpose_rest(params: dict) -> dict:
    """Deep copy with :data:`_SMPLX_POSE_KEYS` zeroed (same preprocessing as :func:`load_smplx_coord3d_tpose`)."""
    out = copy.deepcopy(params)
    for key in _SMPLX_POSE_KEYS:
        if key not in out:
            continue
        arr = numpy.asarray(out[key], dtype=numpy.float32)
        out[key] = numpy.zeros_like(arr)
    return out


def _vertices_from_smplx_param_dict(
    params: dict,
    model_path: str = "models/smplx",
    gender: str = "neutral",
    num_pca_comps: int = 12,
) -> torch.Tensor:
    """Run SMPL-X forward from a parameter dict (same keys as ``smplx_param.pkl``)."""
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

    if "scale" in params:
        scale = float(numpy.asarray(params["scale"]).squeeze())
        verts = verts * scale
    if "translation" in params:
        translation = torch.from_numpy(
            numpy.asarray(params["translation"], dtype=numpy.float32)
        ).reshape(1, 3)
        verts = verts + translation

    return verts  # (Nv, 3) float32


def vertices_from_smplx_param_dict(
    params: dict,
    model_path: str = "models/smplx",
    gender: str = "neutral",
    num_pca_comps: int = 12,
) -> torch.Tensor:
    """SMPL-X mesh vertices from a parameter dict (same schema as ``smplx_param.pkl``)."""
    return _vertices_from_smplx_param_dict(
        params, model_path=model_path, gender=gender, num_pca_comps=num_pca_comps
    )


def load_smplx_coord3d_tpose(
    pkl_path: str,
    model_path: str = "models/smplx",
    gender: str = "neutral",
    num_pca_comps: int = 12,
) -> torch.Tensor:
    """Same as ``load_smplx_coord3d`` for a ``.pkl``, but axis-angle / PCA poses are zeroed (T-pose).

    Shape coefficients (``betas``), ``expression``, and optional ``scale`` / ``translation`` are kept.
    """
    params = load_smplx_params_dict(pkl_path)
    tpose = copy_smplx_params_tpose_rest(params)
    return _vertices_from_smplx_param_dict(
        tpose, model_path=model_path, gender=gender, num_pca_comps=num_pca_comps
    )


def frame_count_for_duration_seconds(
    fps: float,
    duration_seconds: float = 2.0,
) -> int:
    """Number of frames for a clip at ``fps`` (e.g. ``30`` Hz × ``2`` s → ``60`` frames)."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if duration_seconds < 0:
        raise ValueError(f"duration_seconds must be non-negative, got {duration_seconds}")
    n = int(round(float(fps) * float(duration_seconds)))
    return max(1, n)


def _rotation_matrix_about_y(angle_rad: float) -> numpy.ndarray:
    """Right-handed rotation about +Y (SMPL vertical axis), shape ``(3, 3)``."""
    c = float(numpy.cos(angle_rad))
    s = float(numpy.sin(angle_rad))
    return numpy.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=numpy.float64)


def _axis_angle_to_rotation_matrix(axis_angle: numpy.ndarray) -> numpy.ndarray:
    rvec = numpy.asarray(axis_angle, dtype=numpy.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R


def _rotation_matrix_to_axis_angle(R: numpy.ndarray) -> numpy.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(numpy.float64))
    return rvec.reshape(3)


def copy_smplx_params_spin_global_yaw(
    params: dict,
    frame_index: int,
    num_frames: int,
    *,
    compose_with_original: bool = True,
    full_turn_rad: float = 2.0 * numpy.pi,
) -> dict:
    """Copy SMPL-X params with root rotation adjusted for a **self-spin** (yaw about +Y).

    One full turn is spread across ``num_frames`` (frame ``0`` → angle ``0``, last frame → just
    below one full revolution). This **composes** a world-space yaw with the existing
    ``global_orient`` (axis-angle), so the subject keeps their original facing offset while
    spinning; set ``compose_with_original=False`` for pure yaw ``R_y(angle)`` only.

    Args:
        params: Loaded ``smplx_param.pkl``-style dict (must contain ``global_orient``).
        frame_index: ``0 .. num_frames - 1``.
        num_frames: Total frames in the clip (use :func:`frame_count_for_duration_seconds`).
        compose_with_original: If True, ``R_new = R_y(angle) @ R_orig``; if False, ``R_new = R_y(angle)``.
        full_turn_rad: Rotation range over the whole clip (default one full turn).

    Returns:
        Deep copy of ``params`` with ``global_orient`` updated (same dtype/shape as input).
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if not (0 <= frame_index < num_frames):
        raise ValueError(
            f"frame_index must be in [0, num_frames), got frame_index={frame_index}, num_frames={num_frames}"
        )

    out = copy.deepcopy(params)
    if "global_orient" not in out:
        raise KeyError("params must contain 'global_orient'")

    go = numpy.asarray(params["global_orient"], dtype=numpy.float64).reshape(-1)
    orig_dtype = numpy.asarray(params["global_orient"]).dtype
    orig_shape = numpy.asarray(params["global_orient"]).shape

    angle = full_turn_rad * (float(frame_index) / float(num_frames))
    R_y = _rotation_matrix_about_y(angle)

    if compose_with_original:
        R_orig = _axis_angle_to_rotation_matrix(go)
        R_new = R_y @ R_orig
    else:
        R_new = R_y

    aa = _rotation_matrix_to_axis_angle(R_new).astype(orig_dtype, copy=False).reshape(orig_shape)
    out["global_orient"] = aa
    return out


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