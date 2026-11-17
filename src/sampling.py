"""Sampling of surface, free-space, near-surface, and uniform points.

The surface set is produced by the thesis Gaussian-transmittance estimator:
one point per ray whose first transmittance crossing is at or below 0.5.
Free space is represented by rays up to their first ``T <= 0.9`` crossing.

Nothing in this module reads a ground-truth mesh or point cloud. Cleaning is
purely geometric and camera-derived.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .cameras import Camera
from .config import GaussianConfig, SamplingConfig, TransmittanceConfig
from .gaussian import GaussianModel
from .io_utils import ensure_dir, sha256_text
from .transmittance import cull_candidates, transmittance_on_rays


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class SimilarityTransform:
    """``x' = (x - center) * scale``."""
    center: np.ndarray
    scale: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, np.float32).reshape(3)
        self.scale = float(self.scale)
        if not np.isfinite(self.center).all() or not np.isfinite(self.scale):
            raise ValueError("transform must be finite")
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def apply(self, points: np.ndarray | torch.Tensor):
        if isinstance(points, torch.Tensor):
            c = torch.as_tensor(self.center, dtype=points.dtype, device=points.device)
            return (points - c) * self.scale
        return (np.asarray(points, np.float32) - self.center) * self.scale

    def inverse(self, points: np.ndarray | torch.Tensor):
        if isinstance(points, torch.Tensor):
            c = torch.as_tensor(self.center, dtype=points.dtype, device=points.device)
            return points / self.scale + c
        return np.asarray(points, np.float32) / self.scale + self.center

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale}

    @classmethod
    def from_dict(cls, d: dict) -> "SimilarityTransform":
        return cls(np.asarray(d["center"], np.float32), float(d["scale"]))


@dataclass
class SurfaceSamples:
    points: torch.Tensor            # (N,3) normalized frame after normalize()
    normals: torch.Tensor           # (N,3), oriented toward the source camera
    weights: torch.Tensor           # (N,) flatness weight in [0,1]
    flatness: torch.Tensor          # (N,) same proxy retained separately
    gaussian_index: torch.Tensor    # (N,) source Gaussian
    camera_index: torch.Tensor      # (N,) source camera
    pixel_x: torch.Tensor           # (N,)
    pixel_y: torch.Tensor           # (N,)
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def subset(self, idx: torch.Tensor) -> "SurfaceSamples":
        return SurfaceSamples(
            self.points[idx], self.normals[idx], self.weights[idx],
            self.flatness[idx], self.gaussian_index[idx], self.camera_index[idx],
            self.pixel_x[idx], self.pixel_y[idx], dict(self.stats),
        )

    def to_numpy(self) -> Dict[str, np.ndarray]:
        return {
            "points": self.points.detach().cpu().numpy(),
            "normals": self.normals.detach().cpu().numpy(),
            "weights": self.weights.detach().cpu().numpy(),
            "flatness": self.flatness.detach().cpu().numpy(),
            "gaussian_index": self.gaussian_index.detach().cpu().numpy(),
            "camera_index": self.camera_index.detach().cpu().numpy(),
            "pixel_x": self.pixel_x.detach().cpu().numpy(),
            "pixel_y": self.pixel_y.detach().cpu().numpy(),
        }

    @classmethod
    def from_numpy(cls, d: Dict[str, np.ndarray], device: str = "cpu") -> "SurfaceSamples":
        def t(k, dtype=torch.float32):
            return torch.as_tensor(np.asarray(d[k]), dtype=dtype, device=device)
        return cls(t("points"), t("normals"), t("weights"), t("flatness"),
                   t("gaussian_index", torch.long), t("camera_index", torch.long),
                   t("pixel_x", torch.long), t("pixel_y", torch.long))


@dataclass
class FreeSpaceRays:
    origins: torch.Tensor      # (M,3)
    dirs: torch.Tensor         # (M,3)
    t_free: torch.Tensor       # (M,)
    camera_index: torch.Tensor # (M,)

    def __len__(self) -> int:
        return int(self.origins.shape[0])

    def subset(self, idx: torch.Tensor) -> "FreeSpaceRays":
        return FreeSpaceRays(self.origins[idx], self.dirs[idx], self.t_free[idx],
                             self.camera_index[idx])

    def to_numpy(self) -> Dict[str, np.ndarray]:
        return {
            "free_origins": self.origins.detach().cpu().numpy(),
            "free_dirs": self.dirs.detach().cpu().numpy(),
            "free_t": self.t_free.detach().cpu().numpy(),
            "free_camera": self.camera_index.detach().cpu().numpy(),
        }


@dataclass
class SampleBundle:
    surface: SurfaceSamples
    free: FreeSpaceRays
    transform: SimilarityTransform
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def statistical_outlier_removal(points: np.ndarray, k: int = 16,
                                std_ratio: float = 2.5) -> np.ndarray:
    """Return a boolean keep-mask using mean distance to the ``k`` neighbours.

    ``k`` is clipped to ``n-1``. Sets with fewer than 3 points are returned
    unchanged (nothing can be judged an outlier reliably).
    """
    points = np.asarray(points, np.float32)
    n = points.shape[0]
    if n < 3:
        return np.ones(n, dtype=bool)
    from scipy.spatial import cKDTree
    kk = int(min(max(1, k), n - 1))
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=kk + 1)
    mean_dist = dist[:, 1:].mean(axis=1)
    mu = float(mean_dist.mean())
    sigma = float(mean_dist.std())
    if sigma < 1e-12:
        return np.ones(n, dtype=bool)
    return mean_dist <= mu + std_ratio * sigma


def connected_component_filter(points: np.ndarray,
                               radius: Optional[float] = None,
                               radius_factor: float = 3.0,
                               min_fraction: float = 0.1) -> np.ndarray:
    """Keep components whose point count is at least ``min_fraction`` of largest.

    Components are connected under a Euclidean radius graph. When ``radius`` is
    omitted it is ``radius_factor * median nearest-neighbour distance``. This is
    an implementation detail of the thesis's "remove small connected components"
    rule, which does not specify the radius.
    """
    points = np.asarray(points, np.float32)
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.ones(1, dtype=bool)
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    tree = cKDTree(points)
    if radius is None:
        dist, _ = tree.query(points, k=2)
        nn = float(np.median(dist[:, 1]))
        radius = radius_factor * max(nn, 1e-8)
    pairs = tree.query_pairs(r=float(radius), output_type="ndarray")
    if pairs.size == 0:
        # every point is its own component
        labels = np.arange(n)
        n_comp = n
    else:
        data = np.ones(len(pairs), np.int8)
        graph = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        n_comp, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, minlength=n_comp)
    if counts.size == 0:
        return np.zeros(n, dtype=bool)
    keep_labels = counts >= max(1.0, float(min_fraction) * counts.max())
    return keep_labels[labels]


# --------------------------------------------------------------------------- #
# Geometry sampling
# --------------------------------------------------------------------------- #
def _surface_from_rays(
    model: GaussianModel,
    normals_all: torch.Tensor,
    flatness_all: torch.Tensor,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    camera_center: torch.Tensor,
    camera_index: int,
    pixels_xy: Tuple[np.ndarray, np.ndarray],
    gcfg: GaussianConfig,
    tcfg: TransmittanceConfig,
) -> Tuple[SurfaceSamples, FreeSpaceRays, dict]:
    """Extract crossing samples for one camera's rays."""
    candidates = cull_candidates(
        model, origins, dirs,
        cull_sigma=gcfg.cull_sigma,
        max_candidates=gcfg.max_candidates,
        chunk_rays=gcfg.chunk_rays,
    )
    # Process transmittance in chunks to bound padded-tensor memory.
    pts: List[torch.Tensor] = []
    nrm: List[torch.Tensor] = []
    wts: List[torch.Tensor] = []
    flt: List[torch.Tensor] = []
    gidx: List[torch.Tensor] = []
    cidx: List[torch.Tensor] = []
    pxs: List[torch.Tensor] = []
    pys: List[torch.Tensor] = []
    fo: List[torch.Tensor] = []
    fd: List[torch.Tensor] = []
    ft: List[torch.Tensor] = []
    fc: List[torch.Tensor] = []
    n_surface = 0
    n_free = 0
    xs_all, ys_all = pixels_xy
    chunk = max(1, int(gcfg.chunk_rays) * 8)
    with torch.no_grad():
        cov_inv = model.covariance_inv()
        for start in range(0, origins.shape[0], chunk):
            end = min(start + chunk, origins.shape[0])
            res = transmittance_on_rays(
                model, origins[start:end], dirs[start:end], candidates[start:end],
                surface_level=tcfg.level_surface, free_level=tcfg.level_free,
                cov_inv=cov_inv,
            )
            o = origins[start:end]
            d = dirs[start:end]
            sv = res.surface_valid
            if sv.any():
                loc = torch.nonzero(sv, as_tuple=False).squeeze(1)
                depth = res.surface_depth[loc]
                p = o[loc] + depth[:, None] * d[loc]
                gi = res.surface_index[loc]
                safe = gi.clamp_min(0)
                n = normals_all[safe]
                flip = ((camera_center[None, :] - p) * n).sum(-1) < 0
                n = torch.where(flip[:, None], -n, n)
                pts.append(p)
                nrm.append(n)
                flt.append(flatness_all[safe])
                wts.append(flatness_all[safe])
                gidx.append(gi)
                cidx.append(torch.full_like(gi, camera_index))
                pxs.append(torch.as_tensor(xs_all[start:end], dtype=torch.long,
                                           device=o.device)[loc])
                pys.append(torch.as_tensor(ys_all[start:end], dtype=torch.long,
                                           device=o.device)[loc])
                n_surface += int(loc.numel())
            fv = res.free_valid
            if fv.any():
                loc = torch.nonzero(fv, as_tuple=False).squeeze(1)
                fo.append(o[loc])
                fd.append(d[loc])
                ft.append(res.free_depth[loc])
                fc.append(torch.full((loc.numel(),), camera_index, dtype=torch.long,
                                     device=o.device))
                n_free += int(loc.numel())
    device = model.positions.device
    if not pts:
        surface = SurfaceSamples(
            torch.zeros(0, 3, device=device), torch.zeros(0, 3, device=device),
            torch.zeros(0, device=device), torch.zeros(0, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )
    else:
        surface = SurfaceSamples(
            torch.cat(pts), torch.cat(nrm), torch.cat(wts), torch.cat(flt),
            torch.cat(gidx), torch.cat(cidx), torch.cat(pxs), torch.cat(pys),
        )
    if not fo:
        free = FreeSpaceRays(torch.zeros(0, 3, device=device),
                             torch.zeros(0, 3, device=device),
                             torch.zeros(0, device=device),
                             torch.zeros(0, dtype=torch.long, device=device))
    else:
        free = FreeSpaceRays(torch.cat(fo), torch.cat(fd), torch.cat(ft),
                             torch.cat(fc))
    stats = {"n_surface_raw": len(surface), "n_free_raw": len(free),
             "n_rays": int(origins.shape[0])}
    return surface, free, stats


def build_samples(
    model: GaussianModel,
    cameras: Sequence[Camera],
    gcfg: GaussianConfig,
    tcfg: TransmittanceConfig,
    scfg: SamplingConfig,
    device: str = "cpu",
    cache_path: Optional[str | Path] = None,
    max_rays_per_camera: Optional[int] = None,
) -> SampleBundle:
    """Run the transmittance estimator over the selected cameras and clean.

    The returned tensors live on ``device``. The Gaussian model is transformed
    in place only by :func:`normalize_bundle`, not here.
    """
    if len(cameras) == 0:
        raise ValueError("no cameras supplied")
    if len(model) == 0:
        raise ValueError("empty Gaussian model")
    use = list(cameras[: gcfg.n_cameras])
    rng = np.random.default_rng(scfg.seed)
    device_t = torch.device(device)
    if model.positions.device != device_t:
        raise ValueError(
            f"Gaussian model is on {model.positions.device} but sampling device is "
            f"{device_t}; load the model on the requested device"
        )
    with torch.no_grad():
        normals_all = model.estimated_normals()
        flatness_all = model.flatness()
    surfaces: List[SurfaceSamples] = []
    frees: List[FreeSpaceRays] = []
    per_camera = []
    for ci, cam in enumerate(use):
        o_np, d_np, xs, ys = cam.rays(
            stride=gcfg.pixel_stride, max_rays=max_rays_per_camera, generator=rng)
        o = torch.as_tensor(o_np, dtype=torch.float32, device=device_t)
        d = torch.as_tensor(d_np, dtype=torch.float32, device=device_t)
        center = torch.as_tensor(cam.center, dtype=torch.float32, device=device_t)
        s, f, st = _surface_from_rays(
            model, normals_all, flatness_all, o, d, center, ci, (xs, ys), gcfg, tcfg)
        surfaces.append(s)
        frees.append(f)
        per_camera.append({"camera": ci, "img_name": cam.img_name, **st})
    points = torch.cat([s.points for s in surfaces]) if surfaces else torch.zeros(0, 3)
    normals = torch.cat([s.normals for s in surfaces])
    weights = torch.cat([s.weights for s in surfaces])
    flat = torch.cat([s.flatness for s in surfaces])
    gidx = torch.cat([s.gaussian_index for s in surfaces])
    cidx = torch.cat([s.camera_index for s in surfaces])
    px = torch.cat([s.pixel_x for s in surfaces])
    py = torch.cat([s.pixel_y for s in surfaces])
    if len(points) == 0:
        raise RuntimeError(
            "transmittance sampling produced zero surface points. The Gaussian "
            "model may be empty, the cameras may not face the scene, or the "
            "surface level may never be reached. This is a hard failure, not a "
            "synthetic fallback."
        )
    free_o = torch.cat([f.origins for f in frees]) if frees else torch.zeros(0, 3)
    free_d = torch.cat([f.dirs for f in frees]) if frees else torch.zeros(0, 3)
    free_t = torch.cat([f.t_free for f in frees]) if frees else torch.zeros(0)
    free_c = torch.cat([f.camera_index for f in frees]) if frees else torch.zeros(0, dtype=torch.long)

    # Cleaning in numpy (CPU), then back to device.
    p_np = points.detach().cpu().numpy()
    stats: dict = {
        "n_cameras_used": len(use),
        "n_surface_before_cleaning": int(p_np.shape[0]),
        "n_free_rays": int(free_t.numel()),
        "per_camera": per_camera,
        "pixel_stride": gcfg.pixel_stride,
        "pixel_stride_original": gcfg.pixel_stride,
    }
    keep = statistical_outlier_removal(p_np, scfg.sor_k, scfg.sor_std)
    stats["n_removed_sor"] = int((~keep).sum())
    p_np = p_np[keep]
    if p_np.shape[0] == 0:
        raise RuntimeError("statistical outlier removal removed every surface point")
    n_before_cc = p_np.shape[0]
    cc = connected_component_filter(
        p_np, radius_factor=scfg.component_radius_factor,
        min_fraction=scfg.component_min_fraction)
    stats["n_removed_component"] = int((~cc).sum())
    stats["n_surface_clean"] = int(cc.sum())
    stats["n_removed_total"] = int((~keep).sum() + (~cc).sum())
    keep_idx = torch.nonzero(torch.from_numpy(keep), as_tuple=False).squeeze(1)
    keep_idx = keep_idx[torch.from_numpy(cc)]
    keep_idx = keep_idx.to(points.device)
    surface_clean = SurfaceSamples(
        points[keep_idx], normals[keep_idx], weights[keep_idx], flat[keep_idx],
        gidx[keep_idx], cidx[keep_idx], px[keep_idx], py[keep_idx], stats,
    )
    stats["normal_flip_fraction"] = _orientation_flip_fraction(surface_clean, use)
    stats["flatness_mean"] = float(surface_clean.flatness.mean())
    stats["n_kept_fraction"] = float(surface_clean.points.shape[0] / max(1, n_before_cc))
    free = FreeSpaceRays(free_o, free_d, free_t, free_c)
    bounds_min = surface_clean.points.min(0).values
    bounds_max = surface_clean.points.max(0).values
    bundle = SampleBundle(surface_clean, free, SimilarityTransform(
        np.zeros(3, np.float32), 1.0), bounds_min, bounds_max, stats)
    if cache_path is not None:
        save_bundle(bundle, cache_path)
    return bundle


def _orientation_flip_fraction(surface: SurfaceSamples,
                               cameras: Sequence[Camera]) -> float:
    if len(surface) == 0:
        return 0.0
    centers = torch.stack([
        torch.as_tensor(cameras[int(i)].center, dtype=torch.float32,
                        device=surface.points.device)
        for i in surface.camera_index
    ])
    view = centers - surface.points
    # Normals already oriented; a negative dot would indicate an error.
    flipped = ((surface.normals * view).sum(-1) < 0).float().mean()
    return float(flipped)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize_bundle(bundle: SampleBundle, model: GaussianModel,
                     radius: float = 0.8) -> SampleBundle:
    """Map cleaned surface samples to a sphere of ``radius`` and move Gaussians.

    The transform is derived only from the Gaussian-derived surface samples and
    the camera-derived free-space rays. It is recorded so the mesh can be mapped
    back to the original frame for evaluation.
    """
    if len(bundle.surface) < 2:
        raise ValueError("need at least two surface samples for normalization")
    pts = bundle.surface.points.detach().cpu().numpy()
    center = ((pts.min(0) + pts.max(0)) / 2.0).astype(np.float32)
    max_dist = float(np.linalg.norm(pts - center, axis=1).max())
    if not np.isfinite(max_dist) or max_dist <= 1e-9:
        raise ValueError("degenerate surface sample cloud")
    scale = float(radius) / max_dist
    transform = SimilarityTransform(center, scale)
    model.apply_similarity(scale, torch.as_tensor(center, dtype=model.positions.dtype,
                                                 device=model.positions.device))
    dev = bundle.surface.points.device
    surface = SurfaceSamples(
        transform.apply(bundle.surface.points),
        bundle.surface.normals.clone(),  # normals are translation/scale invariant
        bundle.surface.weights.clone(),
        bundle.surface.flatness.clone(),
        bundle.surface.gaussian_index.clone(),
        bundle.surface.camera_index.clone(),
        bundle.surface.pixel_x.clone(),
        bundle.surface.pixel_y.clone(),
        dict(bundle.surface.stats),
    )
    free = FreeSpaceRays(
        transform.apply(bundle.free.origins),
        bundle.free.dirs.clone(),
        bundle.free.t_free * scale,
        bundle.free.camera_index.clone(),
    )
    pts_np = surface.points.detach().cpu().numpy()
    stats = dict(bundle.stats)
    stats.update({
        "normalization_center": center.tolist(),
        "normalization_scale": scale,
        "normalized_radius": float(np.linalg.norm(pts_np, axis=1).max()),
        "bounds_min": pts_np.min(0).tolist(),
        "bounds_max": pts_np.max(0).tolist(),
    })
    return SampleBundle(
        surface, free, transform,
        surface.points.min(0).values, surface.points.max(0).values, stats,
    )


def transform_cameras(cameras: Sequence[Camera],
                      transform: SimilarityTransform) -> List[Camera]:
    return [c.transformed(transform.scale, transform.center) for c in cameras]


# --------------------------------------------------------------------------- #
# Batch samplers (deterministic given a generator)
# --------------------------------------------------------------------------- #
def sample_indices(n: int, batch: int, generator: torch.Generator,
                   replace: Optional[bool] = None) -> torch.Tensor:
    if n <= 0:
        raise ValueError("cannot sample from an empty set")
    if batch <= 0:
        raise ValueError("batch must be positive")
    if replace is None:
        replace = batch > n
    if replace:
        return torch.randint(0, n, (batch,), generator=generator, dtype=torch.long)
    return torch.randperm(n, generator=generator, dtype=torch.long)[:batch]


def sample_surface_batch(surface: SurfaceSamples, batch: int,
                         generator: torch.Generator) -> Tuple[torch.Tensor, ...]:
    idx = sample_indices(len(surface), batch, generator).to(surface.points.device)
    return surface.points[idx], surface.normals[idx], surface.weights[idx]


def sample_uniform_points(batch: int, bounds_min: torch.Tensor,
                          bounds_max: torch.Tensor,
                          generator: torch.Generator) -> torch.Tensor:
    if bounds_min.shape != (3,) or bounds_max.shape != (3,):
        raise ValueError("bounds must be (3,)")
    if (bounds_max <= bounds_min).any():
        raise ValueError("bounds_max must exceed bounds_min")
    u = torch.rand((batch, 3), generator=generator,
                   dtype=bounds_min.dtype, device=bounds_min.device)
    return bounds_min + u * (bounds_max - bounds_min)


def sample_near_surface(points: torch.Tensor, batch: int, jitter_sigma: float,
                        generator: torch.Generator) -> torch.Tensor:
    if points.shape[0] == 0:
        raise ValueError("no surface points for near-surface sampling")
    idx = sample_indices(points.shape[0], batch, generator).to(points.device)
    noise = torch.randn((batch, 3), generator=generator, dtype=points.dtype,
                        device=points.device) * float(jitter_sigma)
    return points[idx] + noise


def sample_free_space(free: FreeSpaceRays, batch: int,
                      generator: torch.Generator,
                      margin_fraction: float = 0.05) -> torch.Tensor:
    """Uniform point on each selected ray before ``(1-margin) * t_free``."""
    if len(free) == 0:
        raise ValueError(
            "no free-space rays with a T<=0.9 crossing; free-space loss "
            "cannot be evaluated"
        )
    if not (0.0 <= margin_fraction < 1.0):
        raise ValueError("margin_fraction must be in [0,1)")
    idx = sample_indices(len(free), batch, generator).to(free.t_free.device)
    t_max = free.t_free[idx] * (1.0 - float(margin_fraction))
    u = torch.rand((batch,), generator=generator, dtype=free.t_free.dtype,
                   device=free.t_free.device)
    t = u * t_max
    return free.origins[idx] + t[:, None] * free.dirs[idx]


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def sample_cache_key(model: GaussianModel, cameras: Sequence[Camera],
                     gcfg: GaussianConfig, tcfg: TransmittanceConfig,
                     scfg: SamplingConfig) -> str:
    """Hash the inputs that determine the sample set (no ground truth)."""
    h = hashlib.sha256()
    h.update(np.asarray([len(model), gcfg.pixel_stride, gcfg.max_candidates,
                         gcfg.cull_sigma, gcfg.min_opacity, tcfg.level_surface,
                         tcfg.level_free, scfg.sor_k, scfg.sor_std,
                         scfg.component_radius_factor,
                         scfg.component_min_fraction, scfg.seed], np.float64).tobytes())
    # Cheaper and more stable than hashing hundreds of thousands of floats:
    pos = model.positions.detach().cpu().numpy()
    if pos.size:
        h.update(np.asarray(pos.shape, np.int64).tobytes())
        h.update(np.asarray(pos.mean(0), np.float64).tobytes())
        h.update(np.asarray(pos.std(0), np.float64).tobytes())
        h.update(str(float(pos.sum())).encode())
    for cam in cameras:
        h.update(np.asarray(cam.center, np.float64).tobytes())
        h.update(np.asarray(cam.K, np.float64).tobytes())
        h.update(np.asarray([cam.H, cam.W], np.int64).tobytes())
        h.update(str(cam.img_name).encode())
    h.update(str(model.source_path).encode())
    return h.hexdigest()


def save_bundle(bundle: SampleBundle, path: str | Path) -> Path:
    """Save the sample bundle as a compressed ``.npz`` (with the transform)."""
    path = Path(path)
    ensure_dir(path.parent)
    arrays = dict(bundle.surface.to_numpy())
    arrays.update(bundle.free.to_numpy())
    arrays["transform_center"] = bundle.transform.center
    arrays["transform_scale"] = np.asarray([bundle.transform.scale])
    arrays["bounds_min"] = bundle.bounds_min.detach().cpu().numpy()
    arrays["bounds_max"] = bundle.bounds_max.detach().cpu().numpy()
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_bundle(path: str | Path, device: str = "cpu") -> SampleBundle:
    data = np.load(str(path))
    surface = SurfaceSamples.from_numpy({k: data[k] for k in (
        "points", "normals", "weights", "flatness", "gaussian_index",
        "camera_index", "pixel_x", "pixel_y")}, device=device)
    free = FreeSpaceRays(
        torch.as_tensor(data["free_origins"], dtype=torch.float32, device=device),
        torch.as_tensor(data["free_dirs"], dtype=torch.float32, device=device),
        torch.as_tensor(data["free_t"], dtype=torch.float32, device=device),
        torch.as_tensor(data["free_camera"], dtype=torch.long, device=device),
    )
    tr = SimilarityTransform(data["transform_center"], float(data["transform_scale"][0]))
    return SampleBundle(
        surface, free, tr,
        torch.as_tensor(data["bounds_min"], dtype=torch.float32, device=device),
        torch.as_tensor(data["bounds_max"], dtype=torch.float32, device=device),
        {},
    )
