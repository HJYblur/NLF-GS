import os
import sys
from pathlib import Path
import argparse
import pickle

# Make 'avatar_utils' importable when running as a script
# Add the 'src' directory to sys.path so imports like 'from avatar_utils.x import y' work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import json
import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image
from avatar_utils.camera import look_at_viewmatrix
from avatar_utils.config import get_config


def configure_pyopengl_platform(prefer_gpu=True):
    # Don't override user choice
    if "PYOPENGL_PLATFORM" in os.environ:
        return

    is_macos = (sys.platform == "darwin")
    is_linux = sys.platform.startswith("linux")
    is_headless = (not os.environ.get("DISPLAY")) and (not os.environ.get("WAYLAND_DISPLAY"))

    if is_macos:
        return

    if is_linux and is_headless:
        # Linux headless: prefer EGL for GPU offscreen; fallback to OSMesa (CPU)
        if prefer_gpu:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        else:
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        return

    # Linux/others with a display: don't force anything (GLX/WGL/whatever works)
    # If you *must* force on Linux desktop, use "glx", but usually best to do nothing.


configure_pyopengl_platform()

# Load camera configuration from YAML
def get_camera_config():
    """Get camera configuration from YAML config file."""
    cfg = get_config()
    camera_cfg = cfg.get("camera", {})
    data_cfg = cfg.get("data", {})
    
    return {
        "distance": float(camera_cfg.get("distance", 1.2)),
        "yfov_deg": float(camera_cfg.get("yfov_deg", 45.0)),
        "up": camera_cfg.get("up", [0.0, 1.0, 0.0]),
        "viewpoints": camera_cfg.get("viewpoints", {
            "front": [0.0, 0.0, 1.0],
            "back": [0.0, 0.0, -1.0],
            "left": [-1.0, 0.0, 0.0],
            "right": [1.0, 0.0, 0.0],
        }),
        "image_size": tuple(data_cfg.get("image_size", [1024, 1024])),
    }

# Global constants from config
CAMERA_CONFIG = get_camera_config()
IMAGE_SIZE = CAMERA_CONFIG["image_size"]
VIEWPOINTS = {k: np.array(v) for k, v in CAMERA_CONFIG["viewpoints"].items()}
CAMERA_MAP_ROOT = Path(__file__).resolve().parents[2] / "data" / "THuman_cameras"
TARGET_SUBJECT_HEIGHT_M = 1.80


def get_data_paths() -> tuple[Path, Path, Path]:
    cfg = get_config()
    data_cfg = cfg.get("data", {})
    raw_obj_root = Path(data_cfg.get("raw_obj_root", data_cfg.get("raw", "data/THuman_2.0")))
    raw_smplx_root = Path(
        data_cfg.get("raw_smplx_root", "data/THuman_2.0_smplx_paras")
    )
    processed_root = Path(data_cfg.get("processed_root", data_cfg.get("root", "processed")))
    return raw_obj_root, raw_smplx_root, processed_root


DATA_ROOT, SMPLX_SOURCE_ROOT, OUT_ROOT = get_data_paths()


def _iter_identities(
    root: Path,
    start_subject: str | None = None,
    end_subject: str | None = None,
):
    start_id = int(start_subject) if start_subject is not None else None
    end_id = int(end_subject) if end_subject is not None else None

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            try:
                subject_id = int(entry.name)
            except ValueError:
                continue

            if start_id is not None and subject_id < start_id:
                continue
            if end_id is not None and subject_id > end_id:
                continue

            # Filter macOS AppleDouble sidecar files (e.g. ._0001.obj).
            obj_files = sorted(
                p
                for p in entry.glob("*.obj")
                if not p.name.startswith(".") and not p.name.startswith("._")
            )
            if not obj_files:
                continue

            # Prefer canonical subject-named OBJ when present.
            canonical_obj = entry / f"{entry.name}.obj"
            yield entry.name, canonical_obj if canonical_obj in obj_files else obj_files[0]


def _find_texture_for_obj(obj_path: Path) -> Path | None:
    """Find the per-identity texture image, preferring material0.* if present.

    THuman subjects typically ship with `material0.mtl` and `material0.jpeg`.
    Prefer that, but fall back to any adjacent .jpeg/.jpg/.png.
    """
    # Prefer the canonical material0.* texture name
    for ext in (".jpeg", ".jpg", ".png"):
        p = obj_path.parent / f"material0{ext}"
        if p.exists():
            return p
    # Fallback: any adjacent image next to the OBJ
    for pattern in ("*.jpeg", "*.jpg", "*.png"):
        for p in sorted(obj_path.parent.glob(pattern)):
            if p.name.lower().startswith("material0"):
                return p
            # return the first match if no material0 was found
            return p
    return None


def _load_meshes(obj_path: Path) -> list[trimesh.Trimesh]:
    """Load an OBJ and return a list of geometry meshes with visuals/materials.

    Use process=True so trimesh parses the associated MTL and texture maps.
    """
    loaded = trimesh.load(obj_path, process=True)
    if isinstance(loaded, trimesh.Scene):
        # Collect all geometry as trimesh.Trimesh objects
        geoms = []
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                geoms.append(geom)
        return geoms
    elif isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    else:
        return []


def _mesh_to_pyrender(
    mesh: trimesh.Trimesh, texture_path: Path | None
) -> pyrender.Mesh:
    """Convert trimesh to pyrender mesh.

    If trimesh visuals/materials are present (from MTL), let pyrender build its
    own material from the mesh visuals. Otherwise, if a texture_path was found,
    create a simple PBR material with that texture as baseColor.
    """
    # Prefer using existing visuals/materials parsed from MTL
    if (
        getattr(mesh, "visual", None) is not None
        and getattr(mesh.visual, "material", None) is not None
    ):
        return pyrender.Mesh.from_trimesh(mesh, smooth=True)

    # Fallback: apply explicit texture if provided
    if texture_path is not None and texture_path.exists():
        tex_img = Image.open(texture_path).convert("RGB")
        tex_data = np.asarray(tex_img)
        tex = pyrender.Texture(source=tex_data, source_channels="RGB")
        material = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=tex,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
        )
        return pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    # Last resort: no visuals or texture found
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def _compute_similarity_transform_from_vertices(
    vertices: np.ndarray,
    target_height_m: float = TARGET_SUBJECT_HEIGHT_M,
) -> tuple[float, np.ndarray]:
    """Compute canonical scale+translation from stacked vertices."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    height = float(vmax[1] - vmin[1])
    if height <= 1e-8:
        raise ValueError(f"Invalid mesh height for normalization: {height}")

    scale = float(target_height_m / height)
    scaled_vertices = vertices * scale
    svmin = scaled_vertices.min(axis=0)
    svmax = scaled_vertices.max(axis=0)

    # Center body around the origin for fixed camera rendering.
    center_x = 0.5 * (svmin[0] + svmax[0])
    center_z = 0.5 * (svmin[2] + svmax[2])
    center_y = 0.5 * (svmin[1] + svmax[1])
    translation = np.array([-center_x, -center_y, -center_z], dtype=np.float64)
    return scale, translation


def _normalize_meshes_for_rendering(
    meshes: list[trimesh.Trimesh],
    target_height_m: float = TARGET_SUBJECT_HEIGHT_M,
) -> tuple[list[trimesh.Trimesh], float, np.ndarray]:
    """Normalize subject scale and position before rendering.

    This follows GHG-style preprocessing by:
      1) scaling the subject to a canonical height (default: 1.80m),
      2) recentering horizontally (X/Z) to the origin,
      3) vertically centering the subject around Y=0.
    """
    if not meshes:
        return meshes, 1.0, np.zeros(3, dtype=np.float64)

    all_vertices = np.vstack([m.vertices for m in meshes])
    scale, translation = _compute_similarity_transform_from_vertices(
        all_vertices,
        target_height_m=target_height_m,
    )

    normalized_meshes: list[trimesh.Trimesh] = []
    for mesh in meshes:
        m = mesh.copy()
        m.vertices = (m.vertices.astype(np.float64) * scale) + translation
        normalized_meshes.append(m)

    return normalized_meshes, scale, translation


def _export_subject_smplx(
    identity: str,
    out_dir: Path,
    scale: float,
    translation: np.ndarray,
) -> None:
    """Apply mesh normalization transform to source smplx_param.pkl and export it."""
    src_pkl = SMPLX_SOURCE_ROOT / identity / "smplx_param.pkl"
    if not src_pkl.exists():
        print(f"Skipping SMPL-X export for {identity}: missing {src_pkl}")
        return

    with open(src_pkl, "rb") as f:
        params = pickle.load(f)

    old_scale = float(np.asarray(params.get("scale", 1.0)).reshape(-1)[0])
    old_translation = np.asarray(params.get("translation", np.zeros(3)), dtype=np.float64).reshape(-1)
    if old_translation.shape[0] != 3:
        old_translation = old_translation[:3]
        if old_translation.shape[0] < 3:
            old_translation = np.pad(old_translation, (0, 3 - old_translation.shape[0]), mode="constant")

    # Preserve smplx_loader convention: vertices = base_vertices * scale + translation.
    new_scale = old_scale * scale
    new_translation = old_translation * scale + translation
    params["scale"] = np.array([new_scale], dtype=np.float32)
    params["translation"] = new_translation.astype(np.float32).reshape(1, 3)

    out_path = out_dir / f"{identity}_smplx.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Wrote normalized SMPL-X params: {out_path}")


def _render_views(
    meshes: list[trimesh.Trimesh],
    out_dir: Path,
    texture_path: Path | None,
    identity: str,
    bg_color: list[int] | tuple[int, int, int] = (0, 0, 0),
):
    """Render THuman meshes from canonical camera positions.
    
    """
    if not meshes:
        raise ValueError(f"No renderable meshes loaded for identity {identity}")

    # print min and max location of all meshes for debugging
    all_vertices = np.vstack([m.vertices for m in meshes])
    print(
        f"Meshes vertex bounds: min {all_vertices.min(axis=0)}, max {all_vertices.max(axis=0)}"
    )

    # Build scene
    bg_rgb = list(bg_color)
    if len(bg_rgb) != 3:
        raise ValueError(f"bg_color must be RGB (len=3), got: {bg_color}")
    scene = pyrender.Scene(bg_color=[bg_rgb[0], bg_rgb[1], bg_rgb[2], 0], ambient_light=[0.3, 0.3, 0.3])
    for m in meshes:
        scene.add(_mesh_to_pyrender(m, texture_path))

    # Canonical global cameras: look at origin with fixed distance
    yfov_deg = CAMERA_CONFIG["yfov_deg"]
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(yfov_deg))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    # Camera intrinsics/extrinsics are generated once globally by generate_camera_mapping()
    # and consumed by gsplat via avatar_utils.camera.load_camera_mapping().
    # We intentionally do NOT write per-identity camera JSONs here to avoid duplication.

    renderer = pyrender.OffscreenRenderer(*IMAGE_SIZE)
    try:
        origin = np.zeros(3, dtype=float)
        up_default = np.array(CAMERA_CONFIG["up"], dtype=float)
        distance = CAMERA_CONFIG["distance"]
        
        for name, direction in VIEWPOINTS.items():
            # Ensure up is not parallel to view direction
            up = up_default.copy()
            if np.allclose(np.cross(up, direction), 0.0):
                up = np.array([0.0, 0.0, 1.0], dtype=float)
            
            # --- Camera conventions ---
            # pyrender (OpenGL-style) uses a camera that looks along -Z in camera space.
            # gsplat uses +Z forward.
            # We'll:
            #   (1) build a pyrender camera pose using forward='-z'
            #   (2) save a gsplat-compatible w2c using forward='+z'
            eye = origin + direction * distance

            # Build pyrender camera-to-world pose (-Z forward)
            _w2c_pyr_t, c2w_pyr_t = look_at_viewmatrix(
                eye=eye,
                target=origin,
                up=up,
                device=None,
                dtype=torch.float32,
                forward="-z",
            )
            c2w_pyr = c2w_pyr_t.detach().cpu().numpy().astype(float)
            cam_node = scene.add(camera, pose=c2w_pyr)
            light_node = scene.add(light, pose=c2w_pyr)

            color, depth = renderer.render(scene)
            # Save color image with identity + view in the filename
            Image.fromarray(color).save(out_dir / f"{identity}_{name}.png")

            # Generate a foreground mask from depth (valid, >0)
            if depth is not None:
                mask_bool = np.isfinite(depth) & (depth > 0)
                mask_img = mask_bool.astype(np.uint8) * 255
                Image.fromarray(mask_img).save(out_dir / f"{identity}_{name}_mask.png")

            scene.remove_node(cam_node)
            scene.remove_node(light_node)
    finally:
        renderer.delete()


def generate_camera_mapping(
    output_dir: Path | None = None,
    image_size: tuple[int, int] | None = None,
    yfov_deg: float | None = None,
    distance: float | None = None,
) -> None:
    """Generate and store camera intrinsics & extrinsics for THuman views.

    This writes one JSON per view under ``.data/`` by default, named
    ``thuman_<view>.json``. Each JSON contains:
      - "K": 3x3 intrinsics matrix
      - "viewmat": 4x4 world-to-camera matrix
      - "image_size": [W, H]
      - "yfov_deg": vertical field of view in degrees

    Args:
        output_dir: Destination directory (default: project-root/.data).
        image_size: (W, H) used to derive principal point and focal length. If None, read from config.
        yfov_deg: Vertical field of view in degrees. If None, read from config.
        distance: Canonical camera distance from origin for all views. If None, read from config.
    """
    # Use config values if not provided
    if image_size is None:
        image_size = CAMERA_CONFIG["image_size"]
    if yfov_deg is None:
        yfov_deg = CAMERA_CONFIG["yfov_deg"]
    if distance is None:
        distance = CAMERA_CONFIG["distance"]
    
    if output_dir is None:
        output_dir = CAMERA_MAP_ROOT
    os.makedirs(output_dir, exist_ok=True)

    W, H = image_size
    yfov_rad = np.deg2rad(yfov_deg)
    # Focal length from vertical FOV: fy = H / (2 * tan(yfov/2)); fx = fy
    fy = H / (2.0 * np.tan(yfov_rad / 2.0))
    fx = fy
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

    center = np.zeros(3, dtype=float)
    up_world = np.array(CAMERA_CONFIG["up"], dtype=float)

    for view_name, direction in VIEWPOINTS.items():
        eye = center + direction * distance
        up = up_world.copy()
        if np.allclose(np.cross(up, direction), 0.0):
            up = np.array([0.0, 0.0, 1.0], dtype=float)

        # Save gsplat-compatible world-to-camera (+Z forward)
        w2c_gs_t, _c2w_gs_t = look_at_viewmatrix(
            eye=eye,
            target=center,
            up=up,
            device=None,
            dtype=torch.float32,
            forward="+z",
        )
        w2c = w2c_gs_t.detach().cpu().numpy().astype(float)

        # Also keep the pyrender (-Z forward) camera-to-world pose for debugging/reference
        _w2c_pyr_t, c2w_pyr_t = look_at_viewmatrix(
            eye=eye,
            target=center,
            up=up,
            device=None,
            dtype=torch.float32,
            forward="-z",
        )
        c2w_pyr = c2w_pyr_t.detach().cpu().numpy().astype(float)

        payload = {
            "K": K.tolist(),
            "viewmat": w2c.tolist(),
            "coords": "+z",
            "type": "w2c",
            "c2w_pyrender": c2w_pyr.tolist(),
            "image_size": [int(W), int(H)],
            "yfov_deg": float(yfov_deg),
        }
        out_path = output_dir / f"thuman_{view_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Camera mappings written to: {output_dir}")


def preprocess_thuman(
    data_root: Path = DATA_ROOT,
    out_root: Path = OUT_ROOT,
    start_subject: str | None = None,
    end_subject: str | None = None,
):
    os.makedirs(out_root, exist_ok=True)
    # Proactively generate camera mappings if not present
    try:
        generate_camera_mapping(output_dir=CAMERA_MAP_ROOT)
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate camera mapping. Ensure that pyrender and its dependencies are properly installed and that a compatible OpenGL context is available. You may try setting PYOPENGL_PLATFORM=egl or PYOPENGL_PLATFORM=osmesa in your environment variables. Original error: {e}"
        )
    for identity, obj_path in _iter_identities(data_root, start_subject, end_subject):
        target_dir = out_root / identity
        target_dir.mkdir(parents=True, exist_ok=True)
        meshes = _load_meshes(obj_path)
        if not meshes:
            print(f"Skipping {identity}: failed to load meshes from {obj_path}")
            continue
        meshes, scale, translation = _normalize_meshes_for_rendering(meshes)
        _export_subject_smplx(identity, target_dir, scale, translation)
        texture_path = _find_texture_for_obj(obj_path)
        _render_views(meshes, target_dir, texture_path, identity)
        print(f"Rendered {identity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess THuman subjects by rendering views.")
    parser.add_argument(
        "--start-subject",
        type=str,
        default=None,
        help="Start subject id (inclusive), e.g. 0054",
    )
    parser.add_argument(
        "--end-subject",
        type=str,
        default=None,
        help="End subject id (inclusive), e.g. 0100",
    )
    args = parser.parse_args()

    preprocess_thuman(
        start_subject=args.start_subject,
        end_subject=args.end_subject,
    )
