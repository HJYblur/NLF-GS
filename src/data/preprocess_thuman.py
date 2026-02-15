import os

# Headless Linux often lacks an X11 display; force an offscreen GL backend early
# before importing pyrender so it can pick a headless platform (EGL/OSMesa).
if not os.environ.get("DISPLAY") and "PYOPENGL_PLATFORM" not in os.environ:
    # Prefer EGL on GPU servers; if unavailable, users can switch to 'osmesa'
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import json
from pathlib import Path
import numpy as np
import trimesh
import pyrender
from PIL import Image


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a 4x4 camera-to-world pose matrix.

    This matches the common OpenGL/pyrender convention where the camera looks
    towards the negative Z axis. The returned matrix places the camera at
    `eye` and orients it to look at `target` with the given `up` vector.

    Args:
        eye: (3,) camera position in world coordinates.
        target: (3,) point the camera looks at.
        up: (3,) up direction for the camera.

    Returns:
        (4,4) homogeneous transformation matrix (camera -> world).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    # forward vector (camera's local Z) points from target to eye
    z = eye - target
    z = z / np.linalg.norm(z)

    # right vector (camera's local X)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        # up was parallel to viewing direction; pick a different up
        up_alt = np.array([0.0, 0.0, 1.0])
        x = np.cross(up_alt, z)
    x = x / np.linalg.norm(x)

    # true up (camera's local Y)
    y = np.cross(z, x)

    mat = np.eye(4, dtype=float)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return mat


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "THuman_2.0"
OUT_ROOT = Path(__file__).resolve().parents[2] / "processed"
IMAGE_SIZE = (1024, 1024)
MESH_SCALE = 1.5
VIEWPOINTS = {
    "front": np.array([0.0, 0.0, 1.0]),
    "back": np.array([0.0, 0.0, -1.0]),
    "left": np.array([-1.0, 0.0, 0.0]),
    "right": np.array([1.0, 0.0, 0.0]),
}
CAMERA_MAP_ROOT = Path(__file__).resolve().parents[2] / "data" / "THuman_cameras"


def _iter_identities(root: Path):
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            obj_files = list(entry.glob("*.obj"))
            if obj_files:
                yield entry.name, obj_files[0]


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


def _render_views(
    meshes: list[trimesh.Trimesh],
    out_dir: Path,
    texture_path: Path | None,
    identity: str,
):
    """Render THuman meshes from canonical camera positions.
    
    Scale meshes to match SMPL-X size and render from fixed camera positions
    to ensure consistency with training data.
    """
    # Scale meshes to reasonable size for rendering
    # THuman scans are physically small (~0.6-0.9m), scale them up for better framing
    scaled_meshes = []
    for m in meshes:
        m_scaled = m.copy()
        m_scaled.apply_scale(MESH_SCALE)
        scaled_meshes.append(m_scaled)
        
    # print min and max location of all scaled meshes for debugging
    all_vertices = np.vstack([m.vertices for m in scaled_meshes])
    print(f"Scaled meshes vertex bounds: min {all_vertices.min(axis=0)}, max{all_vertices.max(axis=0)}")
    
    # Build scene with scaled meshes
    scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3, 0.3, 0.3])
    for m in scaled_meshes:
        scene.add(_mesh_to_pyrender(m, texture_path))

    # Canonical global cameras: look at origin with fixed distance
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    renderer = pyrender.OffscreenRenderer(*IMAGE_SIZE)
    try:
        origin = np.zeros(3, dtype=float)
        up_default = np.array([0.0, 1.0, 0.0], dtype=float)
        distance = 2.0  # Closer camera for better framing
        
        for name, direction in VIEWPOINTS.items():
            # Ensure up is not parallel to view direction
            up = up_default.copy()
            if np.allclose(np.cross(up, direction), 0.0):
                up = np.array([0.0, 0.0, 1.0], dtype=float)
            
            eye = origin + direction * distance
            c2w = look_at(eye, origin, up)
            # pyrender follows an OpenGL-style camera convention (camera looks along -Z).
            # For gsplat we want a world-to-camera matrix. We save w2c (= inv(c2w)).
            # In this project we treat the saved viewmat as "+Z forward" when consumed
            # by gsplat.
            w2c = np.linalg.inv(c2w)
            cam_node = scene.add(camera, pose=c2w)
            light_node = scene.add(light, pose=c2w)

            color, depth = renderer.render(scene)
            # Save color image with identity + view in the filename
            Image.fromarray(color).save(out_dir / f"{identity}_{name}.png")

            # Save the per-view world-to-camera matrix for gsplat (alongside images)
            cam_payload = {
                "viewmat": w2c.tolist(),
                "type": "w2c",
                "coords": "+z",
            }
            with open(out_dir / f"{identity}_{name}_w2c.json", "w", encoding="utf-8") as f:
                json.dump(cam_payload, f, ensure_ascii=False, indent=2)

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
    image_size: tuple[int, int] = IMAGE_SIZE,
    yfov_deg: float = 45.0,
    distance: float = 2.0,  # Changed to 2.0 to match rendering
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
        image_size: (W, H) used to derive principal point and focal length.
        yfov_deg: Vertical field of view in degrees.
        distance: Canonical camera distance from origin for all views.
    """
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
    up_world = np.array([0.0, 1.0, 0.0], dtype=float)

    for view_name, direction in VIEWPOINTS.items():
        eye = center + direction * distance
        up = up_world.copy()
        if np.allclose(np.cross(up, direction), 0.0):
            up = np.array([0.0, 0.0, 1.0], dtype=float)

        c2w = look_at(eye, center, up)
        w2c = np.linalg.inv(c2w)

        payload = {
            "K": K.tolist(),
            "viewmat": w2c.tolist(),
            "image_size": [int(W), int(H)],
            "yfov_deg": float(yfov_deg),
        }
        out_path = output_dir / f"thuman_{view_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Camera mappings written to: {output_dir}")


def preprocess_thuman(data_root: Path = DATA_ROOT, out_root: Path = OUT_ROOT):
    os.makedirs(out_root, exist_ok=True)
    # Proactively generate camera mappings if not present
    try:
        generate_camera_mapping(output_dir=CAMERA_MAP_ROOT)
    except Exception:
        # Non-fatal; continue preprocessing even if mapping generation fails
        pass
    for identity, obj_path in _iter_identities(data_root):
        target_dir = out_root / identity
        target_dir.mkdir(parents=True, exist_ok=True)
        meshes = _load_meshes(obj_path)
        texture_path = _find_texture_for_obj(obj_path)
        _render_views(meshes, target_dir, texture_path, identity)
        print(f"Rendered {identity}")


if __name__ == "__main__":
    preprocess_thuman()
