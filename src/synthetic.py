"""
synthetic.py

*** THIS MODULE IS A FALLBACK, NOT PART OF THE PAPER'S METHOD. ***

The task brief requires starting from an already-trained 3DGS
`point_cloud.ply`. No such file (and no DTU multi-view images / camera
calibration) was uploaded alongside the paper -- only the thesis document
itself. Per Section 5 of the brief ("DATASET REALITY"): we do not fabricate
DTU ground truth. Instead we generate a small, clearly-labeled SYNTHETIC
"pretrained" Gaussian-splat scene (Gaussians flattened and oriented on the
surface of a unit sphere, exactly matching the real 3DGS PLY schema) purely
so the REAL pipeline code in gaussian.py / transmittance.py / sdf.py /
losses.py / mesh.py / evaluation.py can be smoke-tested end-to-end and its
correctness verified against a known analytic ground truth (a sphere).

When the user supplies a real `point_cloud.ply` under
`3DGS_Pretrained_SDFGS/<scan>/point_cloud/iteration_XXXXX/point_cloud.ply`
(and, for the photometric term, a `cameras_sphere.npz` + images/), the
notebook's auto-discovery (gaussian.discover_point_clouds /
camera.discover_camera_file) will pick that up automatically INSTEAD of
this synthetic fallback.
"""
import os
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh


def _quat_from_normal(n):
    """Quaternion (w,x,y,z) rotating local z-axis (0,0,1) onto normal n."""
    z = np.array([0.0, 0.0, 1.0])
    n = n / (np.linalg.norm(n) + 1e-8)
    v = np.cross(z, n)
    c = np.dot(z, n)
    if c < -0.9999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    s = np.sqrt((1 + c) * 2)
    invs = 1.0 / s
    x, y, zz = v[0] * invs, v[1] * invs, v[2] * invs
    w = s * 0.5
    q = np.array([w, x, y, zz])
    return q / np.linalg.norm(q)


def make_synthetic_sphere_gaussians(n_points=4000, radius=1.0, seed=42,
                                     flat_scale=0.012, tangent_scale=0.045,
                                     opacity_logit=4.0, color=(0.7, 0.35, 0.2)):
    """
    Build Gaussians flattened along the surface normal of a unit sphere
    (radial axis = shortest scale -> matches estimated_normals() convention
    in gaussian.py) so the transmittance-crossing estimator has a well
    defined T(t)=0.5 surface, analogous to a converged real 3DGS scene of a
    solid object.
    """
    rng = np.random.default_rng(seed)
    # Fibonacci sphere sampling for uniform coverage
    i = np.arange(0, n_points, dtype=np.float64)
    phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
    golden = np.pi * (1 + 5 ** 0.5)
    theta = golden * i
    xyz = np.stack([
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi),
    ], axis=1)
    normals = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

    scale_raw = np.zeros((n_points, 3), dtype=np.float32)
    scale_raw[:, 0] = np.log(tangent_scale)
    scale_raw[:, 1] = np.log(tangent_scale)
    scale_raw[:, 2] = np.log(flat_scale)  # local z (flat axis) -> becomes radial normal after rotation

    rot = np.stack([_quat_from_normal(n) for n in normals], axis=0).astype(np.float32)

    opacity_raw = np.full((n_points,), opacity_logit, dtype=np.float32)  # sigmoid(4.0) ~= 0.98

    SH_C0 = 0.28209479177387814
    base_color = np.array(color, dtype=np.float32)
    f_dc = ((base_color - 0.5) / SH_C0).astype(np.float32)
    f_dc = np.tile(f_dc[None, :], (n_points, 1))
    f_dc += rng.normal(0, 0.02, size=f_dc.shape).astype(np.float32)  # mild per-point variation

    return dict(xyz=xyz.astype(np.float32), scale_raw=scale_raw, rot_raw=rot,
                opacity_raw=opacity_raw, f_dc=f_dc)


def write_synthetic_ply(path: str, n_points=4000, radius=1.0, seed=42):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g = make_synthetic_sphere_gaussians(n_points=n_points, radius=radius, seed=seed)
    n = g["xyz"].shape[0]

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
             ("opacity", "f4"),
             ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    data = np.zeros(n, dtype=dtype)
    data["x"], data["y"], data["z"] = g["xyz"][:, 0], g["xyz"][:, 1], g["xyz"][:, 2]
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = g["f_dc"][:, 0], g["f_dc"][:, 1], g["f_dc"][:, 2]
    data["opacity"] = g["opacity_raw"]
    data["scale_0"], data["scale_1"], data["scale_2"] = (g["scale_raw"][:, 0], g["scale_raw"][:, 1],
                                                           g["scale_raw"][:, 2])
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = (g["rot_raw"][:, 0], g["rot_raw"][:, 1],
                                                                    g["rot_raw"][:, 2], g["rot_raw"][:, 3])

    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(path)
    return path


def ground_truth_sphere_mesh(radius=1.0, subdivisions=4):
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
