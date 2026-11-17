"""DTU MVS evaluation, following the official ObsMask + Plane protocol.

The metric definitions and filtering rules mirror the official MATLAB code
(``BaseEvalMain_web.m``, ``PointCompareMain.m``, ``MaxDistCP.m``,
``reducePts_haa.m``):

* the predicted point cloud is reduced to a minimum inter-point distance of
  0.2 mm with the stochastic greedy procedure of ``reducePts_haa``;
* accuracy is the mean predicted -> reference distance for predicted points
  inside the ObsMask, after discarding distances >= 20 mm;
* completeness is the mean reference -> predicted distance for reference
  points above the supplied ground plane, after discarding distances >= 20 mm;
* overall is the mean of accuracy and completeness.

Ground truth is read only inside this module, after reconstruction, mesh
extraction, transform selection, and model selection are complete. No function
here is allowed to influence training or choose a transform. Transforms must be
camera-derived (``scale_mat`` from ``cameras_sphere.npz`` or an Umeyama fit of
calibrated camera centres).

The additional F-score @1/2/5 mm is computed with the same point sets, but it is
our own metric and is explicitly labelled non-official.
"""
from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import trimesh
from scipy.io import loadmat
from scipy.spatial import cKDTree

from .config import DTUConfig


# --------------------------------------------------------------------------- #
# Small utility types and validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObsMaskData:
    mask: np.ndarray
    bb: np.ndarray
    res: float
    margin: float
    path: Path


def _as_points(points, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_transform(matrix, name: str = "transform") -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def apply_transform(points, matrix) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to an ``(N, 3)`` point array."""
    pts = _as_points(points, "points")
    mat = _as_transform(matrix)
    hom = np.c_[pts, np.ones(len(pts))]
    out = hom @ mat.T
    if not np.allclose(out[:, 3], 1.0, atol=1e-8):
        denom = out[:, 3]
        if np.any(np.abs(denom) < 1e-12):
            raise ValueError("transform produced an invalid homogeneous coordinate")
        out = out[:, :3] / denom[:, None]
    else:
        out = out[:, :3]
    if not np.isfinite(out).all():
        raise ValueError("transform produced non-finite points")
    return out


def matlab_round(values) -> np.ndarray:
    """MATLAB ``round``: ties are rounded away from zero."""
    arr = np.asarray(values, dtype=np.float64)
    return np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def dataset_roots(root: Union[str, Path]) -> List[Path]:
    """Return plausible dataset roots without requiring a specific layout."""
    root = Path(root)
    candidates = [
        root,
        root / "SampleSet" / "MVS Data",
        root / "SampleSet" / "MVS_Data",
        root / "MVS Data",
        root / "MVS_Data",
        root / "SampleSet",
    ]
    seen = set()
    out: List[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_dir():
            out.append(candidate)
    if not out and root.exists() and root.is_dir():
        out.append(root)
    return out


def _format_names(template: Optional[str], scan: int,
                  fallbacks: Sequence[str]) -> List[str]:
    names: List[str] = []
    if template:
        try:
            names.append(template.format(scan=scan, scan3=f"{scan:03d}"))
        except (KeyError, ValueError):
            pass
    names.extend(fallbacks)
    out: List[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def _find_file(root: Union[str, Path], names: Sequence[str],
               subdirs: Sequence[str], what: str) -> Path:
    tried: List[Path] = []
    for base in dataset_roots(root):
        for subdir in subdirs:
            for name in names:
                candidate = base / subdir / name if subdir else base / name
                tried.append(candidate)
                if candidate.is_file():
                    return candidate
    preview = "\n  ".join(str(path) for path in tried[:16])
    raise FileNotFoundError(f"{what} not found. Tried:\n  {preview}")


def _mask_names(scan: int, cfg: Optional[DTUConfig]) -> List[str]:
    template = cfg.mask_name_template if cfg is not None else None
    return _format_names(template, scan, [
        f"ObsMask{scan}_10.mat",
        f"ObsMask{scan:03d}_10.mat",
        f"ObsMask10_{scan}.mat",
        f"ObsMask10_{scan:03d}.mat",
        f"ObsMask{scan}_10.MAT",
    ])


def _plane_names(scan: int, cfg: Optional[DTUConfig]) -> List[str]:
    template = cfg.plane_name_template if cfg is not None else None
    return _format_names(template, scan, [
        f"Plane{scan}.mat",
        f"Plane{scan:03d}.mat",
        f"Plane_{scan}.mat",
        f"Plane_{scan:03d}.mat",
        f"Plane{scan}.MAT",
    ])


def _gt_names(scan: int) -> List[str]:
    return [
        f"stl{scan:03d}_total.ply",
        f"stl{scan:03d}_total2.ply",
        f"stl{scan}_total.ply",
        f"stl{scan}_total2.ply",
    ]


# --------------------------------------------------------------------------- #
# Official files
# --------------------------------------------------------------------------- #
def load_obs_mask(scan: int, root: Union[str, Path],
                  cfg: Optional[DTUConfig] = None) -> ObsMaskData:
    path = _find_file(
        root, _mask_names(int(scan), cfg),
        ["ObsMask", "obs_mask", ""], "ObsMask",
    )
    data = loadmat(path)
    if "ObsMask" not in data:
        raise ValueError(f"ObsMask field missing in {path}")
    mask = np.asarray(data["ObsMask"]).astype(bool)
    if mask.ndim != 3 or mask.size == 0:
        raise ValueError(f"ObsMask in {path} must be a non-empty 3-D array")
    if "BB" not in data:
        raise ValueError(f"BB field missing in {path}")
    bb = np.asarray(data["BB"], dtype=np.float64)
    if bb.size != 6:
        raise ValueError(f"BB in {path} must contain six values")
    bb = bb.reshape(2, 3)
    if not (np.isfinite(bb).all() and (bb[1] > bb[0]).all()):
        raise ValueError(f"invalid BB in {path}")
    if "Res" not in data:
        raise ValueError(f"Res field missing in {path}")
    res = float(np.asarray(data["Res"]).ravel()[0])
    if not math.isfinite(res) or res <= 0:
        raise ValueError(f"invalid Res in {path}")
    margin = 10.0
    if "Margin" in data:
        margin = float(np.asarray(data["Margin"]).ravel()[0])
        if not math.isfinite(margin):
            raise ValueError(f"invalid Margin in {path}")
    if "cSet" in data:
        actual = int(np.asarray(data["cSet"]).ravel()[0])
        if actual != int(scan):
            raise ValueError(
                f"ObsMask cSet={actual} does not match requested scan={scan}: {path}"
            )
    return ObsMaskData(mask=mask, bb=bb, res=res, margin=margin, path=path)


def load_plane(scan: int, root: Union[str, Path],
               cfg: Optional[DTUConfig] = None) -> np.ndarray:
    path = _find_file(
        root, _plane_names(int(scan), cfg),
        ["ObsMask", "obs_mask", ""], "Plane",
    )
    data = loadmat(path)
    if "P" not in data:
        raise ValueError(f"Plane field P missing in {path}")
    plane = np.asarray(data["P"], dtype=np.float64).reshape(-1)
    if plane.size != 4 or not np.isfinite(plane).all():
        raise ValueError(f"Plane P in {path} must contain four finite values")
    if np.allclose(plane, 0.0):
        raise ValueError(f"Plane P in {path} is degenerate")
    return plane


def load_gt_points(scan: int, root: Union[str, Path]) -> Tuple[np.ndarray, Path]:
    path = _find_file(
        root, _gt_names(int(scan)),
        ["Points/stl", "points/stl", "stl", ""], "DTU reference STL points",
    )
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"empty scene in {path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError(f"reference point cloud is empty: {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"reference point cloud has non-finite values: {path}")
    # The official stl*_total.ply files are already reduced to 0.2 mm.
    return vertices, path


def point_in_obs_mask(points, obs: ObsMaskData) -> np.ndarray:
    pts = _as_points(points, "points")
    # Official MATLAB uses 1-based ``round((p - BB(1,:))/Res)+1``. The
    # equivalent 0-based index is ``round((p - BB(1,:))/Res)``.
    grid = matlab_round((pts - obs.bb[0][None, :]) / obs.res)
    valid = np.all((grid >= 0) & (grid < np.asarray(obs.mask.shape)[None, :]),
                   axis=1)
    inside = np.zeros(len(pts), dtype=bool)
    if np.any(valid):
        idx = grid[valid].astype(np.int64)
        inside[valid] = obs.mask[idx[:, 0], idx[:, 1], idx[:, 2]]
    return inside


# --------------------------------------------------------------------------- #
# Downsampling and distances
# --------------------------------------------------------------------------- #
def downsample_official(points, min_dist_mm: float = 0.2,
                        seed: int = 42, chunk: int = 100_000) -> np.ndarray:
    """Greedy ``reducePts_haa`` reimplementation.

    Points are visited in a random order. A point that has not already been
    removed is kept and removes every not-yet-visited point within
    ``min_dist_mm``. This matches the MATLAB procedure apart from the random
    permutation, which MATLAB leaves implicit. A seed is exposed so all methods
    are evaluated with the same reproducible order.
    """
    pts = _as_points(points, "points")
    if min_dist_mm <= 0 or not math.isfinite(min_dist_mm):
        raise ValueError("min_dist_mm must be finite and positive")
    if len(pts) <= 1:
        return pts.copy()
    if int(chunk) <= 0:
        raise ValueError("chunk must be positive")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(pts))
    active = np.ones(len(pts), dtype=bool)
    tree = cKDTree(pts)
    chunk = int(chunk)
    for start in range(0, len(order), chunk):
        ids = order[start:start + chunk]
        neighbours = tree.query_ball_point(pts[ids], r=float(min_dist_mm),
                                           workers=-1)
        for point_id, near in zip(ids, neighbours):
            point_id = int(point_id)
            if not active[point_id]:
                continue
            active[np.asarray(near, dtype=np.int64)] = False
            active[point_id] = True
    return pts[active]


def downsample_voxel(points, voxel_mm: float = 0.2) -> np.ndarray:
    """Deterministic voxel downsampling (convenience only, not official)."""
    pts = _as_points(points, "points")
    if voxel_mm <= 0 or not math.isfinite(voxel_mm):
        raise ValueError("voxel_mm must be finite and positive")
    keys = np.floor(pts / float(voxel_mm)).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(keep)]


def nearest_distances(source, target, max_dist_mm: float = 20.0,
                      workers: int = -1) -> np.ndarray:
    """Distance from each source point to the nearest target point.

    Distances at or beyond ``max_dist_mm`` are returned as the cap, matching
    the official ``MaxDistCP`` behaviour before the ``< MaxDist`` filter.
    """
    src = _as_points(source, "source points")
    tgt = _as_points(target, "target points")
    if max_dist_mm <= 0 or not math.isfinite(max_dist_mm):
        raise ValueError("max_dist_mm must be finite and positive")
    tree = cKDTree(tgt)
    dist, _ = tree.query(src, k=1, distance_upper_bound=float(max_dist_mm),
                         workers=workers)
    dist = np.asarray(dist, dtype=np.float64)
    dist[~np.isfinite(dist)] = float(max_dist_mm)
    dist[dist > float(max_dist_mm)] = float(max_dist_mm)
    return dist


# --------------------------------------------------------------------------- #
# Camera-only alignment
# --------------------------------------------------------------------------- #
def umeyama_alignment(source, target, with_scale: bool = True
                      ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Least-squares similarity transform mapping source to target.

    This is the camera-centre alignment used by the protocol. It must never be
    fitted to ground-truth surface points.
    """
    src = _as_points(source, "source camera centres")
    tgt = _as_points(target, "target camera centres")
    if len(src) != len(tgt):
        raise ValueError("source and target camera centres must have equal length")
    if len(src) < 3:
        raise ValueError("at least three camera centres are required")
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    cov = (tgt_c.T @ src_c) / float(len(src))
    u, singular, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    if with_scale:
        var_src = float(np.mean(np.sum(src_c ** 2, axis=1)))
        if var_src <= 1e-18:
            raise ValueError("source camera centres are degenerate")
        scale = float(np.sum(singular * d) / var_src)
    else:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("estimated similarity scale is invalid")
    translation = tgt_mean - scale * (rotation @ src_mean)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    transformed = apply_transform(src, matrix)
    residual = float(np.sqrt(np.mean(np.sum((transformed - tgt) ** 2, axis=1))))
    mean_camera_distance = float(np.mean(np.linalg.norm(tgt - tgt_mean, axis=1)))
    return matrix, {
        "scale": scale,
        "residual_rms": residual,
        "mean_camera_distance": mean_camera_distance,
        "residual_fraction": (residual / mean_camera_distance
                              if mean_camera_distance > 0 else float("inf")),
    }


def camera_center_from_projection(matrix) -> np.ndarray:
    """Camera centre from a 3x4 projection matrix or a 4x4 pose."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape == (4, 4):
        rotation = arr[:3, :3]
        translation = arr[:3, 3]
        if abs(np.linalg.det(rotation)) < 1e-12:
            raise ValueError("4x4 transform has a singular rotation block")
        return -np.linalg.inv(rotation) @ translation
    if arr.shape != (3, 4):
        raise ValueError("camera matrix must be 3x4 or 4x4")
    # Null space of P: P [C; 1] = 0.
    _, _, vh = np.linalg.svd(arr)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        raise ValueError("projection matrix has a camera centre at infinity")
    return homogeneous[:3] / homogeneous[3]


def load_camera_centers_from_pos(calib_dir: Union[str, Path],
                                 indices: Optional[Iterable[int]] = None
                                 ) -> np.ndarray:
    """Load DTU ``pos_XXX.txt`` projection matrices and return camera centres."""
    directory = Path(calib_dir)
    files = sorted(glob.glob(str(directory / "pos_*.txt")))
    if not files:
        files = sorted(glob.glob(str(directory / "*.txt")))
    if indices is not None:
        wanted = [int(i) for i in indices]
        by_number: Dict[int, Path] = {}
        for path in files:
            stem = Path(path).stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            if digits:
                by_number[int(digits)] = Path(path)
        missing = [i for i in wanted if i not in by_number]
        if missing:
            raise FileNotFoundError(f"missing camera calibration indices: {missing}")
        files = [str(by_number[i]) for i in wanted]
    if not files:
        raise FileNotFoundError(f"no pos_*.txt camera files under {directory}")
    centers: List[np.ndarray] = []
    for path in files:
        try:
            matrix = np.loadtxt(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot parse camera file {path}: {exc}") from exc
        centers.append(camera_center_from_projection(matrix))
    return np.stack(centers, axis=0)


def load_scale_mat(npz_path: Union[str, Path], index: int = 0) -> np.ndarray:
    """Load one ``scale_mat_i`` from an IDR/NeuS ``cameras_sphere.npz``."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"camera archive not found: {path}")
    with np.load(path) as data:
        keys = list(data.keys())
        candidates = [f"scale_mat_{index}", f"scale_mat_{index + 1}",
                      "scale_mat", "scale_mat_0"]
        for key in candidates:
            if key in data:
                matrix = np.asarray(data[key], dtype=np.float64)
                return _as_transform(matrix, key)
    raise KeyError(f"no scale_mat* key in {path}; keys={keys}")


def validate_camera_transform(method_centres, dtu_centres, matrix,
                              max_residual_fraction: float = 0.01
                              ) -> Dict[str, Any]:
    src = _as_points(method_centres, "method camera centres")
    tgt = _as_points(dtu_centres, "DTU camera centres")
    if len(src) != len(tgt):
        raise ValueError("camera-centre arrays must have equal length")
    transformed = apply_transform(src, matrix)
    residual = float(np.sqrt(np.mean(np.sum((transformed - tgt) ** 2, axis=1))))
    tgt_mean = tgt.mean(axis=0)
    mean_distance = float(np.mean(np.linalg.norm(tgt - tgt_mean, axis=1)))
    fraction = residual / mean_distance if mean_distance > 0 else float("inf")
    if fraction > float(max_residual_fraction):
        raise ValueError(
            f"camera-only transform residual is {fraction:.4%} of the mean "
            f"camera distance, above {max_residual_fraction:.4%}"
        )
    return {
        "residual_rms": residual,
        "mean_camera_distance": mean_distance,
        "residual_fraction": fraction,
        "max_residual_fraction": float(max_residual_fraction),
    }


def resolve_transform(
    method_camera_centres=None,
    dtu_camera_centres=None,
    scale_mat=None,
    mode: str = "auto",
    max_residual_fraction: float = 0.01,
) -> Dict[str, Any]:
    """Choose a camera-only transform; never uses ground-truth geometry."""
    mode = str(mode).lower()
    if mode not in {"auto", "scale_mat", "calibration", "none"}:
        raise ValueError("mode must be auto|scale_mat|calibration|none")
    if mode in {"auto", "scale_mat"} and scale_mat is not None:
        matrix = _as_transform(scale_mat, "scale_mat")
        info = {"method": "scale_mat", "matrix": matrix}
        if method_camera_centres is not None and dtu_camera_centres is not None:
            info.update(validate_camera_transform(
                method_camera_centres, dtu_camera_centres, matrix,
                max_residual_fraction))
        info["used_ground_truth"] = False
        return info
    if mode == "scale_mat":
        raise ValueError("scale_mat requested but not supplied")
    if mode in {"auto", "calibration"}:
        if method_camera_centres is None or dtu_camera_centres is None:
            raise ValueError("camera centres are required for calibration alignment")
        matrix, stats = umeyama_alignment(method_camera_centres, dtu_camera_centres)
        if stats["residual_fraction"] > float(max_residual_fraction):
            raise ValueError(
                f"Umeyama residual is {stats['residual_fraction']:.4%} of the "
                f"mean camera distance, above {max_residual_fraction:.4%}"
            )
        info = {"method": "calibration_umeyama", "matrix": matrix}
        info.update(stats)
        info["max_residual_fraction"] = float(max_residual_fraction)
        info["used_ground_truth"] = False
        return info
    # mode == "none": caller asserts the input is already in DTU frame.
    matrix = np.eye(4)
    info = {"method": "none", "matrix": matrix,
            "residual_fraction": None, "used_ground_truth": False}
    if method_camera_centres is not None and dtu_camera_centres is not None:
        info.update(validate_camera_transform(
            method_camera_centres, dtu_camera_centres, matrix,
            max_residual_fraction))
    return info


# --------------------------------------------------------------------------- #
# Mesh sampling
# --------------------------------------------------------------------------- #
def _load_mesh(mesh_or_path: Union[str, Path, trimesh.Trimesh]) -> trimesh.Trimesh:
    if isinstance(mesh_or_path, trimesh.Trimesh):
        mesh = mesh_or_path
    else:
        path = Path(mesh_or_path)
        if not path.is_file():
            raise FileNotFoundError(f"mesh not found: {path}")
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                raise ValueError(f"empty scene in {path}")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("input does not contain a triangle mesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh has no vertices or faces")
    return mesh


def sample_mesh_points(
    mesh_or_path: Union[str, Path, trimesh.Trimesh],
    area_per_point_mm2: float = 0.04,
    min_points: int = 200_000,
    max_points: int = 4_000_000,
    seed: int = 42,
    oversample: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample a triangle mesh uniformly by area.

    ``area_per_point_mm2`` is the historical ``density_points_per_mm2`` field in
    :class:`DTUConfig`: one point per 0.04 mm^2. The official MATLAB surface
    path uses ``MeshSupSamp`` instead; this area sampler is a documented,
    reproducible Python approximation, followed by the official greedy 0.2 mm
    reduction. It is not bit-identical to the MATLAB upsampling.
    """
    mesh = _load_mesh(mesh_or_path)
    if area_per_point_mm2 <= 0 or not math.isfinite(area_per_point_mm2):
        raise ValueError("area_per_point_mm2 must be finite and positive")
    if min_points < 0 or max_points < 1 or max_points < min_points:
        raise ValueError("invalid min_points/max_points")
    if oversample <= 0 or not math.isfinite(oversample):
        raise ValueError("oversample must be finite and positive")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not np.isfinite(vertices).all():
        raise ValueError("mesh vertices contain non-finite values")
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = float(areas.sum())
    if not math.isfinite(total_area) or total_area <= 0:
        raise ValueError("mesh has zero or invalid surface area")
    target = int(round((total_area / float(area_per_point_mm2)) * float(oversample)))
    n = max(int(min_points), min(int(max_points), target))
    n = max(n, 3)
    rng = np.random.default_rng(int(seed))
    probabilities = areas / total_area
    chosen = rng.choice(len(faces), size=n, p=probabilities)
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    tri_chosen = tri[chosen]
    points = (tri_chosen[:, 0] * (1.0 - u - v)[:, None]
              + tri_chosen[:, 1] * u[:, None]
              + tri_chosen[:, 2] * v[:, None])
    info = {
        "method": "uniform_triangle_area_sampling",
        "area_mm2": total_area,
        "area_per_point_mm2": float(area_per_point_mm2),
        "oversample": float(oversample),
        "n_sampled": int(n),
        "target_n": int(target),
        "min_points": int(min_points),
        "max_points": int(max_points),
        "seed": int(seed),
        "official_mesh_upsampler": "MATLAB MeshSupSamp (not bit-identical)",
    }
    return points, info


# --------------------------------------------------------------------------- #
# Core metric
# --------------------------------------------------------------------------- #
def evaluate_points(
    pred_points,
    scan: int,
    root: Union[str, Path],
    cfg: Optional[DTUConfig] = None,
    transform=None,
    return_sets: bool = False,
) -> Dict[str, Any]:
    """Evaluate predicted points with the official ObsMask + Plane protocol."""
    cfg = cfg or DTUConfig()
    pred = _as_points(pred_points, "predicted points")
    if transform is not None:
        pred = apply_transform(pred, transform)
    obs = load_obs_mask(scan, root, cfg)
    plane = load_plane(scan, root, cfg)
    gt, gt_path = load_gt_points(scan, root)

    pred_down = downsample_official(
        pred, min_dist_mm=float(cfg.downsample_mm), seed=int(cfg.seed))
    if len(pred_down) == 0:
        raise ValueError("empty predicted set after downsampling")
    in_mask = point_in_obs_mask(pred_down, obs)
    pred_mask = pred_down[in_mask]
    if len(pred_mask) == 0:
        raise ValueError(
            f"empty predicted set after ObsMask (scan {scan}); no metric computed"
        )

    max_d = float(cfg.max_dist_mm)
    d_pred_gt = nearest_distances(pred_mask, gt, max_d)
    d_pred_used = d_pred_gt[d_pred_gt < max_d]
    if d_pred_used.size == 0:
        raise ValueError(
            f"all predicted points are beyond the {max_d:.1f} mm cap (scan {scan})"
        )

    gt_h = np.c_[gt, np.ones(len(gt))]
    above = (gt_h @ plane) > 0.0
    gt_above = gt[above]
    if len(gt_above) == 0:
        raise ValueError(f"no reference points above the ground plane (scan {scan})")
    d_gt_pred = nearest_distances(gt_above, pred_down, max_d)
    d_gt_used = d_gt_pred[d_gt_pred < max_d]
    if d_gt_used.size == 0:
        raise ValueError(
            f"all reference points are beyond the {max_d:.1f} mm cap (scan {scan})"
        )

    accuracy = float(d_pred_used.mean())
    completeness = float(d_gt_used.mean())
    overall = 0.5 * (accuracy + completeness)
    result: Dict[str, Any] = {
        "scan": int(scan),
        "protocol": (
            "dtu_official(ObsMask+Plane,reducePts_haa,"
            f"ds={cfg.downsample_mm:g}mm,cap={cfg.max_dist_mm:g}mm)"
        ),
        "accuracy": accuracy,
        "completeness": completeness,
        "overall": overall,
        "median_accuracy": float(np.median(d_pred_used)),
        "median_completeness": float(np.median(d_gt_used)),
        "n_pred_raw": int(len(pred)),
        "n_pred_down": int(len(pred_down)),
        "n_pred_in_mask": int(len(pred_mask)),
        "n_pred_accuracy_used": int(len(d_pred_used)),
        "n_gt": int(len(gt)),
        "n_gt_above_plane": int(len(gt_above)),
        "n_completeness_used": int(len(d_gt_used)),
        "downsample_mm": float(cfg.downsample_mm),
        "max_dist_mm": max_d,
        "downsample_method": "reducePts_haa_greedy_seeded",
        "seed": int(cfg.seed),
        "obs_mask_path": str(obs.path),
        "gt_path": str(gt_path),
        "used_ground_truth_for_alignment": False,
        "official_metric": True,
        # Explicitly non-official supplementary metrics.
        "fscore_official": False,
    }
    for threshold in cfg.fscore_thresholds_mm:
        t = float(threshold)
        precision = float((d_pred_gt < t).mean())
        recall = float((d_gt_pred < t).mean())
        fscore = (0.0 if precision + recall == 0.0
                  else 2.0 * precision * recall / (precision + recall))
        result[f"P@{t:g}mm"] = precision
        result[f"R@{t:g}mm"] = recall
        result[f"F@{t:g}mm"] = fscore
    result["fscore_note"] = (
        "F-score @1/2/5 mm is our own metric, computed with the same point "
        "sets; it is not part of the official DTU protocol."
    )
    if return_sets:
        result["_d_pred_gt"] = d_pred_gt
        result["_d_gt_pred"] = d_gt_pred
        result["_pred_down"] = pred_down
        result["_pred_in_mask"] = pred_mask
        result["_gt_above_plane"] = gt_above
    return result


def evaluate_mesh(
    mesh_or_path: Union[str, Path, trimesh.Trimesh],
    scan: int,
    root: Union[str, Path],
    cfg: Optional[DTUConfig] = None,
    transform=None,
    return_sets: bool = False,
) -> Dict[str, Any]:
    """Sample a mesh, transform it with a camera-only matrix, then evaluate."""
    cfg = cfg or DTUConfig()
    points, sample_info = sample_mesh_points(
        mesh_or_path,
        area_per_point_mm2=float(cfg.density_points_per_mm2),
        min_points=int(cfg.min_points),
        max_points=int(cfg.max_points),
        seed=int(cfg.seed),
    )
    result = evaluate_points(
        points, scan, root, cfg=cfg, transform=transform,
        return_sets=return_sets,
    )
    result["sampling"] = sample_info
    result["mesh"] = str(mesh_or_path) if not isinstance(
        mesh_or_path, trimesh.Trimesh) else "<trimesh.Trimesh>"
    result["transform_used"] = transform is not None
    return result


def validate_official_assets(scan: int, root: Union[str, Path],
                             cfg: Optional[DTUConfig] = None) -> Dict[str, Any]:
    """Check that the three official inputs exist and can be parsed."""
    obs = load_obs_mask(scan, root, cfg)
    plane = load_plane(scan, root, cfg)
    gt, gt_path = load_gt_points(scan, root)
    return {
        "scan": int(scan),
        "obs_mask_path": str(obs.path),
        "plane_path": str(_find_file(
            root, _plane_names(int(scan), cfg),
            ["ObsMask", "obs_mask", ""], "Plane")),
        "gt_path": str(gt_path),
        "mask_shape": list(obs.mask.shape),
        "bb": obs.bb.tolist(),
        "res": obs.res,
        "margin": obs.margin,
        "plane": plane.tolist(),
        "n_gt": int(len(gt)),
    }
