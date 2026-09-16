"""
camera.py

DTU (Aanaes et al. 2016) camera loading + ray generation, following the
convention used by IDR/NeuS-style preprocessing of DTU, which the paper's
baselines (NeuS, Wang et al. 2023) also build on:
    cameras_sphere.npz  with per-view keys:
        world_mat_%d  (4x4)  = K @ [R|t]   (world-to-image projection)
        scale_mat_%d  (4x4)  = normalization transform to unit sphere

We AUTO-DISCOVER this file rather than hard-coding a path. If it is not
present (as in this environment -- no raw DTU images/calibration were
uploaded, only the paper), we do NOT fabricate camera poses. Per Section 5
of the task brief ("DATASET REALITY"), we instead expose a clearly-labeled
SYNTHETIC camera ring so the rest of the pipeline (transmittance-crossing,
SDF refinement, mesh extraction) can be smoke-tested end-to-end. This is
declared explicitly wherever it is used -- never silently substituted.
"""
import glob
import os
import numpy as np
import torch


def discover_camera_file(root_dir: str):
    """Look for common DTU/IDR/NeuS camera file names anywhere under root_dir."""
    candidates = []
    for pattern in ("cameras_sphere.npz", "cameras.npz", "cameras_large.npz"):
        candidates += glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
    return sorted(candidates)[0] if candidates else None


def discover_official_3dgs_cameras_json(root_dir: str):
    """Look for the official graphdeco-inria/gaussian-splatting `cameras.json`
    export (written next to point_cloud/ in the training output folder)."""
    candidates = glob.glob(os.path.join(root_dir, "**", "cameras.json"), recursive=True)
    return sorted(candidates)[0] if candidates else None


def discover_images_dir(root_dir: str):
    for pattern in ("image", "images", "rgb"):
        hits = glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            return sorted(hits)[0]
    return None


class Camera:
    """A single calibrated pinhole camera: intrinsics K, world-to-cam [R|t]."""

    def __init__(self, K, R, t, H, W, img_name=None):
        self.K = K.astype(np.float32)          # (3,3)
        self.R = R.astype(np.float32)          # (3,3) world->cam rotation
        self.t = t.astype(np.float32)          # (3,)  world->cam translation
        self.H, self.W = H, W
        self.img_name = img_name                # original filename, if known (official 3DGS cameras.json)
        self.center = (-R.T @ t).astype(np.float32)   # camera center in world coords

    def rays(self, device, stride: int = 1):
        """Generate one ray per pixel (subsampled by `stride`). Returns (origins, dirs)."""
        ys, xs = np.meshgrid(np.arange(0, self.H, stride), np.arange(0, self.W, stride), indexing="ij")
        pix = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs).ravel()], axis=1).astype(np.float32)
        Kinv = np.linalg.inv(self.K)
        cam_dirs = (Kinv @ pix.T).T                      # (M,3) direction in camera frame
        world_dirs = (self.R.T @ cam_dirs.T).T            # rotate into world frame
        world_dirs = world_dirs / np.linalg.norm(world_dirs, axis=1, keepdims=True)
        origins = np.tile(self.center[None, :], (world_dirs.shape[0], 1))
        return (torch.tensor(origins, device=device),
                torch.tensor(world_dirs.astype(np.float32), device=device),
                (ys.ravel(), xs.ravel()))


def load_dtu_cameras(npz_path: str, H: int, W: int):
    """Parse an IDR/NeuS-style cameras_sphere.npz into a list of Camera objects."""
    data = np.load(npz_path)
    n_cams = len([k for k in data.files if k.startswith("world_mat_")])
    cams = []
    for i in range(n_cams):
        world_mat = data[f"world_mat_{i}"].astype(np.float64)
        scale_mat = data[f"scale_mat_{i}"].astype(np.float64) if f"scale_mat_{i}" in data.files \
            else np.eye(4)
        P = (world_mat @ scale_mat)[:3, :4]
        K, R, t = decompose_projection(P)
        cams.append(Camera(K, R, t, H, W))
    return cams


def load_official_3dgs_cameras(json_path: str, cx_cy: str = "center"):
    """
    Parse the `cameras.json` written by the OFFICIAL graphdeco-inria/
    gaussian-splatting training script (utils/camera_utils.py:camera_to_JSON,
    verified against the actual source on GitHub, not from memory):

        Rt = [[R.T | T], [0,0,0,1]]     # R,T = COLMAP-derived world<->cam pair
                                          # stored by the repo such that
                                          # R = cam2world rotation,
                                          # T = world2cam translation
        W2C_field_confusingly_named = inv(Rt)   # actually camera-to-world!
        position = (cam2world)[:3, 3]    # camera CENTER in world coords
        rotation = (cam2world)[:3, :3]   # camera-to-world ROTATION matrix
        fx, fy   = focal lengths in pixels (from FoVx/FoVy, no cx/cy stored)

    So per entry: world2cam rotation R_w2c = rotation.T,
                  world2cam translation t_w2c = -R_w2c @ position.

    `cx_cy="center"` assumes the principal point is at (W/2, H/2), since the
    official cameras.json does NOT store cx/cy explicitly -- this is the
    standard assumption for COLMAP/3DGS pinhole cameras without documented
    principal-point offset, flagged here rather than silently guessed.
    """
    import json as _json
    with open(json_path, "r") as f:
        entries = _json.load(f)

    cams = []
    for e in entries:
        W, H = int(e["width"]), int(e["height"])
        fx, fy = float(e["fx"]), float(e["fy"])
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        R_c2w = np.array(e["rotation"], dtype=np.float64)   # (3,3)
        position = np.array(e["position"], dtype=np.float64)  # (3,) camera center, world coords

        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ position

        cams.append(Camera(K, R_w2c, t_w2c, H, W, img_name=e.get("img_name")))
    return cams


def load_image_for_camera(cam: "Camera", images_root: str, resize_to_camera=True):
    """
    Load the real observed image referenced by `cam.img_name`, searching
    common subfolder names/extensions. Returns a float32 (H,W,3) array in
    [0,1], resized (area-interpolated) to (cam.H, cam.W) if requested so it
    matches the ray grid built by `Camera.rays()`.
    """
    from PIL import Image as _Image

    if cam.img_name is None:
        raise ValueError("Camera has no img_name -- cannot locate its source image.")

    candidates = []
    for sub in ("images", "image", "input", ""):
        base = os.path.join(images_root, sub) if sub else images_root
        for ext in ("", ".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            candidates.append(os.path.join(base, cam.img_name + ext))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        # last resort: recursive search by basename
        base_name = os.path.splitext(os.path.basename(cam.img_name))[0]
        hits = glob.glob(os.path.join(images_root, "**", base_name + ".*"), recursive=True)
        candidates = hits

    if not candidates:
        raise FileNotFoundError(
            f"Could not find source image for '{cam.img_name}' under '{images_root}'. "
            f"Checked images/, image/, input/ with common extensions, and a recursive "
            f"basename search. Please confirm your images folder layout."
        )

    img = _Image.open(candidates[0]).convert("RGB")
    if resize_to_camera and img.size != (cam.W, cam.H):
        img = img.resize((cam.W, cam.H), _Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def decompose_projection(P):
    """Standard RQ-decomposition of a 3x4 camera matrix P = K[R|t]."""
    M = P[:, :3]
    K, R = rq3(M)
    t = np.linalg.inv(K) @ P[:, 3]
    K = K / K[2, 2]
    return K, R, t


def rq3(A):
    """RQ decomposition of a 3x3 matrix (used to split K and R from P[:, :3])."""
    Q, R = np.linalg.qr(np.flipud(A).T)
    R = np.flipud(R.T)
    Q = Q.T
    R = np.fliplr(R)
    Q = np.flipud(Q)
    # fix signs so K has positive diagonal
    T = np.diag(np.sign(np.diag(R)))
    R = R @ T
    Q = T @ Q
    return R, Q


def synthetic_camera_ring(n_views: int, radius: float, H: int, W: int, focal: float, center=(0, 0, 0)):
    """
    SYNTHETIC fallback camera ring, used ONLY when no real calibration data
    is discoverable (Section 5 -- explicit, labeled fallback, not a silent
    fabrication of DTU ground truth). Cameras look at `center` from a ring
    of radius `radius` in the XZ plane.
    """
    cams = []
    center = np.array(center, dtype=np.float64)
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float64)
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        eye = center + radius * np.array([np.cos(theta), 0.3, np.sin(theta)])
        R, t = look_at(eye, center, np.array([0, 1, 0]))
        cams.append(Camera(K, R, t, H, W))
    return cams


def sanity_check_cameras(cams, scene_center: np.ndarray, verbose=True):
    """
    Validation for a REAL parsed camera set (Section 12/13 requirement --
    never trust a parser silently). For each camera, checks that the
    principal ray (through the image center) points roughly TOWARD
    `scene_center` rather than away from it -- the #1 symptom of a
    rotation/translation convention mixed up while parsing an external
    camera format.
    Returns the fraction of cameras that pass (cos(angle) > 0), and prints
    a per-camera table if verbose.
    """
    ok = 0
    for i, cam in enumerate(cams):
        cx, cy = cam.W / 2.0, cam.H / 2.0
        ray_cam = np.linalg.inv(cam.K) @ np.array([cx, cy, 1.0])
        ray_world = cam.R.T @ ray_cam
        ray_world = ray_world / (np.linalg.norm(ray_world) + 1e-8)
        to_center = scene_center - cam.center
        to_center = to_center / (np.linalg.norm(to_center) + 1e-8)
        cos = float(np.dot(ray_world, to_center))
        passed = cos > 0
        ok += int(passed)
        if verbose:
            print(f"  cam {i:3d}  img_name={getattr(cam, 'img_name', None)!s:20s}  "
                  f"center={np.round(cam.center, 3)}  cos(principal_ray, ->scene_center)={cos:+.3f}  "
                  f"{'OK' if passed else 'SUSPECT (points away from scene!)'}")
    frac = ok / max(1, len(cams))
    if verbose:
        print(f"-> {ok}/{len(cams)} cameras point toward the scene center ({frac*100:.1f}%). "
              f"If this is low, the R/T convention of the parser likely does not match your "
              f"actual camera export -- do not proceed to training until this is high.")
    return frac


def look_at(eye, target, up):
    """
    World->camera [R|t] for a camera at `eye` looking at `target`, using the
    OpenCV pinhole convention (camera looks down its own +z axis, matching
    `Camera.rays()` which builds direction vectors with z=1).
    """
    z = (target - eye)                     # forward: +z points INTO the scene
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    R_c2w = np.stack([x, y, z], axis=1)    # camera axes expressed in world coords
    R = R_c2w.T                            # world -> camera
    t = -R @ eye
    return R, t
