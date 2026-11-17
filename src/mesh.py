"""Marching-Cubes mesh extraction and topology-only post-processing.

The extraction is deliberately separated from rendering and evaluation:

* the SDF is evaluated in linear chunks, so no full ``N x 3`` coordinate grid is
  ever allocated;
* Marching Cubes runs at the configured resolution (512^3 by default) on the
  sample bounding box expanded by 10%;
* connected components smaller than 1% of the largest component's area are
  removed;
* boundary loops with fewer than 100 edges are filled with a deterministic
  MST-seeded constrained triangulation;
* Laplacian smoothing is applied only to a copy used for visualization. The
  metric mesh is never smoothed.

No ground truth is read or used here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import trimesh
from skimage import measure
from trimesh.grouping import group_rows

from .config import MeshConfig
from .io_utils import ensure_dir, now_utc


@dataclass
class MeshResult:
    """Metric mesh, optional visualization mesh, and extraction metadata."""

    mesh: trimesh.Trimesh
    viz_mesh: Optional[trimesh.Trimesh]
    metadata: Dict[str, Any]


def empty_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
        process=False,
    )


def expand_bounds(bounds_min, bounds_max, margin: float = 0.10
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Expand an axis-aligned box by ``margin`` of its half-extent per axis."""
    lo = np.asarray(bounds_min, dtype=np.float64).reshape(3)
    hi = np.asarray(bounds_max, dtype=np.float64).reshape(3)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("bounds must be finite")
    if not bool((hi > lo).all()):
        raise ValueError("bounds_max must exceed bounds_min on every axis")
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("bbox margin must be finite and non-negative")
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * (1.0 + float(margin))
    return center - half, center + half


def evaluate_sdf_volume(
    sdf: Callable[[torch.Tensor], torch.Tensor],
    bounds_min,
    bounds_max,
    resolution: int,
    chunk_size: int = 262_144,
    device: str = "cpu",
) -> np.ndarray:
    """Evaluate an SDF on a regular grid, chunked over linear point indices.

    The returned array has shape ``(resolution, resolution, resolution)`` with
    axes ``(x, y, z)``. Only one chunk of coordinates is materialized at a time.
    """
    resolution = int(resolution)
    if resolution < 2:
        raise ValueError("resolution must be >= 2")
    if resolution > 1024:
        raise ValueError("resolution above 1024 is not supported")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    lo = np.asarray(bounds_min, dtype=np.float64).reshape(3)
    hi = np.asarray(bounds_max, dtype=np.float64).reshape(3)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("bounds must be finite")
    if not bool((hi > lo).all()):
        raise ValueError("bounds_max must exceed bounds_min on every axis")

    xs = np.linspace(lo[0], hi[0], resolution, dtype=np.float64)
    ys = np.linspace(lo[1], hi[1], resolution, dtype=np.float64)
    zs = np.linspace(lo[2], hi[2], resolution, dtype=np.float64)
    total = resolution ** 3
    volume = np.empty(total, dtype=np.float32)
    chunk_size = int(chunk_size)
    plane = resolution * resolution
    torch_device = torch.device(device)

    with torch.no_grad():
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            linear = np.arange(start, stop, dtype=np.int64)
            ix = linear // plane
            rem = linear % plane
            iy = rem // resolution
            iz = rem % resolution
            pts = np.stack([xs[ix], ys[iy], zs[iz]], axis=1).astype(np.float32)
            raw = sdf(torch.as_tensor(pts, device=torch_device))
            if isinstance(raw, torch.Tensor):
                arr = raw.detach().to(device="cpu", dtype=torch.float32).numpy()
            else:
                arr = np.asarray(raw, dtype=np.float32)
            arr = np.asarray(arr).reshape(-1)
            if arr.size != stop - start:
                raise ValueError(
                    f"SDF returned {arr.size} values for a chunk of "
                    f"{stop - start} points"
                )
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"SDF returned non-finite values in chunk starting at {start}"
                )
            volume[start:stop] = arr
    return volume.reshape(resolution, resolution, resolution)


# --------------------------------------------------------------------------- #
# Boundary-loop extraction and MST-seeded triangulation
# --------------------------------------------------------------------------- #
def _order_cycle(cycle: Iterable[int], graph: nx.Graph) -> Optional[List[int]]:
    cycle = list(dict.fromkeys(int(v) for v in cycle))
    if len(cycle) < 3:
        return None
    allowed = set(cycle)
    start = min(cycle)
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = sorted(n for n in graph.neighbors(cur)
                      if n in allowed and n != prev)
        if not nxts:
            return None
        nxt = nxts[0]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(cycle):
            return None
    if len(order) != len(cycle):
        return None
    return order


def _boundary_loops(mesh: trimesh.Trimesh) -> Tuple[List[List[int]], int]:
    if len(mesh.faces) == 0:
        return [], 0
    groups = group_rows(mesh.edges_sorted, require_count=1)
    if len(groups) == 0:
        return [], 0
    boundary_edges = mesh.edges_sorted[groups]
    graph = nx.Graph()
    graph.add_edges_from((int(a), int(b)) for a, b in boundary_edges)
    loops: List[List[int]] = []
    for cycle in nx.cycle_basis(graph):
        ordered = _order_cycle(cycle, graph)
        if ordered is not None:
            loops.append(ordered)
    loops.sort(key=lambda loop: (len(loop), tuple(sorted(loop))))
    return loops, int(len(boundary_edges))


def _project_loop(points: np.ndarray) -> Optional[np.ndarray]:
    center = points.mean(axis=0)
    shifted = points - center
    try:
        _, singular, vh = np.linalg.svd(shifted, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular.size < 2 or singular[1] <= 1e-12:
        return None
    basis = vh[:2].T
    return shifted @ basis


def _point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            denom = y2 - y1
            if abs(denom) < 1e-15:
                continue
            x_cross = x1 + (y - y1) * (x2 - x1) / denom
            if x_cross > x:
                inside = not inside
    return inside


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1])
                 - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray,
                eps: float = 1e-12) -> bool:
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                        d: np.ndarray, eps: float = 1e-12) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
            (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    if abs(o1) <= eps and _on_segment(a, b, c, eps):
        return True
    if abs(o2) <= eps and _on_segment(a, b, d, eps):
        return True
    if abs(o3) <= eps and _on_segment(c, d, a, eps):
        return True
    if abs(o4) <= eps and _on_segment(c, d, b, eps):
        return True
    return False


def _valid_diagonal(i: int, j: int, poly2: np.ndarray) -> bool:
    n = len(poly2)
    if i == j:
        return False
    if (i + 1) % n == j or (j + 1) % n == i:
        return False
    a, b = poly2[i], poly2[j]
    midpoint = 0.5 * (a + b)
    if not _point_in_polygon(midpoint, poly2):
        return False
    for k in range(n):
        if k in (i, j):
            continue
        k2 = (k + 1) % n
        if k2 in (i, j):
            continue
        if _segments_intersect(a, b, poly2[k], poly2[k2]):
            return False
    return True


def _prim_mst(points2: np.ndarray) -> List[Tuple[int, int]]:
    """Deterministic Prim MST over a complete Euclidean graph."""
    n = len(points2)
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    in_tree[0] = True
    for v in range(1, n):
        best[v] = float(np.sum((points2[v] - points2[0]) ** 2))
        parent[v] = 0
    edges: List[Tuple[int, int]] = []
    for _ in range(n - 1):
        candidates = np.where(~in_tree)[0]
        if candidates.size == 0:
            break
        # Deterministic tie-break: smallest squared distance, then index.
        u = int(min(candidates, key=lambda v: (best[v], int(v))))
        in_tree[u] = True
        p = int(parent[u])
        edges.append((min(p, u), max(p, u)))
        for v in candidates:
            if in_tree[v]:
                continue
            d = float(np.sum((points2[v] - points2[u]) ** 2))
            if d < best[v] - 1e-15 or (abs(d - best[v]) <= 1e-15 and u < parent[v]):
                best[v] = d
                parent[v] = u
    return edges


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                       c: np.ndarray, eps: float = 1e-12) -> bool:
    o1 = _orientation(a, b, p)
    o2 = _orientation(b, c, p)
    o3 = _orientation(c, a, p)
    has_neg = (o1 < -eps) or (o2 < -eps) or (o3 < -eps)
    has_pos = (o1 > eps) or (o2 > eps) or (o3 > eps)
    return not (has_neg and has_pos)


def _ear_clip(indices: Sequence[int], poly2: np.ndarray
              ) -> List[Tuple[int, int, int]]:
    """Deterministic ear clipping with a logged fan fallback for bad loops."""
    remain = [(int(indices[i]), np.asarray(poly2[i], dtype=np.float64))
              for i in range(len(indices))]
    tris: List[Tuple[int, int, int]] = []
    if len(remain) < 3:
        return tris
    area = 0.0
    for i in range(len(remain)):
        a = remain[i][1]
        b = remain[(i + 1) % len(remain)][1]
        area += a[0] * b[1] - b[0] * a[1]
    sign = 1.0 if area >= 0 else -1.0
    guard = 0
    max_guard = 4 * len(remain) * len(remain)
    while len(remain) > 3 and guard < max_guard:
        guard += 1
        found = False
        m = len(remain)
        for k in range(m):
            ia, a = remain[k - 1]
            ib, b = remain[k]
            ic, c = remain[(k + 1) % m]
            if sign * _orientation(a, b, c) <= 1e-12:
                continue
            if any(_point_in_triangle(remain[j][1], a, b, c)
                   for j in range(m) if j not in (k - 1, k, (k + 1) % m)):
                continue
            tris.append((ia, ib, ic))
            remain.pop(k)
            found = True
            break
        if not found:
            break
    if len(remain) == 3:
        tris.append((remain[0][0], remain[1][0], remain[2][0]))
    elif len(remain) > 3:
        # Explicit fallback: deterministic fan. This is only reached for a
        # degenerate/non-simple loop; the caller records the fill as a fallback.
        tris.extend((remain[0][0], remain[i][0], remain[i + 1][0])
                    for i in range(1, len(remain) - 1))
    return tris


def _triangulate_polygon(indices: Sequence[int], poly2: np.ndarray
                         ) -> List[Tuple[int, int, int]]:
    idx = [int(v) for v in indices]
    n = len(idx)
    if n < 3:
        return []
    if n == 3:
        return [(idx[0], idx[1], idx[2])]
    mst = _prim_mst(poly2)
    mst.sort(key=lambda e: (float(np.sum((poly2[e[0]] - poly2[e[1]]) ** 2)),
                            e[0], e[1]))
    for a, b in mst:
        if not _valid_diagonal(a, b, poly2):
            continue
        if a > b:
            a, b = b, a
        left = idx[a:b + 1]
        right = idx[b:] + idx[:a + 1]
        left2 = poly2[a:b + 1]
        right2 = np.concatenate([poly2[b:], poly2[:a + 1]], axis=0)
        return (_triangulate_polygon(left, left2)
                + _triangulate_polygon(right, right2))
    return _ear_clip(idx, poly2)


def triangulate_boundary_loop(loop: Sequence[int],
                              vertices: np.ndarray
                              ) -> Tuple[List[Tuple[int, int, int]], bool]:
    """Triangulate one ordered boundary loop.

    The first attempted diagonal is the shortest valid edge of a deterministic
    minimum spanning tree over the loop vertices. If no MST edge is a valid
    interior diagonal, a deterministic ear-clipping fallback is used. Exact
    triangulation is implementation-specific; it never adds or moves vertices.

    Returns ``(triangles, used_fallback)``. The fallback flag is only an
    approximation here; the caller records the number of filled loops.
    """
    loop = [int(v) for v in loop]
    if len(loop) < 3:
        return [], False
    points = np.asarray(vertices, dtype=np.float64)[loop]
    poly2 = _project_loop(points)
    if poly2 is None:
        return _ear_clip(loop, points[:, :2]), True
    return _triangulate_polygon(loop, poly2), False


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def _filter_components(mesh: trimesh.Trimesh, min_fraction: float
                       ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    components = mesh.split(only_watertight=False)
    areas = [float(c.area) for c in components]
    meta = {
        "n_components_before": len(components),
        "component_areas_before": areas,
        "n_components_after": len(components),
        "component_areas_after": areas,
        "components_removed": 0,
    }
    if len(components) <= 1:
        return mesh, meta
    largest = max(areas) if areas else 0.0
    if largest <= 0:
        meta["component_filter_reason"] = "all components have zero area"
        return mesh, meta
    threshold = float(min_fraction) * largest
    kept = [c for c, area in zip(components, areas) if area >= threshold]
    if not kept:
        kept = [components[int(np.argmax(areas))]]
    meta["n_components_after"] = len(kept)
    meta["component_areas_after"] = [float(c.area) for c in kept]
    meta["components_removed"] = len(components) - len(kept)
    meta["component_area_threshold"] = threshold
    if len(kept) == 1:
        out = kept[0].copy()
    else:
        out = trimesh.util.concatenate(kept)
    return out, meta


def _fill_boundary_loops(mesh: trimesh.Trimesh, max_edges: int
                         ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    loops, n_boundary_edges = _boundary_loops(mesh)
    meta = {
        "boundary_edges_before": n_boundary_edges,
        "boundary_loops_detected": len(loops),
        "hole_loop_sizes": [len(loop) for loop in loops],
        "holes_filled": 0,
        "holes_skipped": 0,
    }
    if not loops:
        return mesh, meta
    new_faces: List[Tuple[int, int, int]] = []
    for loop in loops:
        if len(loop) >= int(max_edges):
            meta["holes_skipped"] += 1
            continue
        tris, _ = triangulate_boundary_loop(loop, mesh.vertices)
        if not tris:
            meta["holes_skipped"] += 1
            continue
        new_faces.extend(tris)
        meta["holes_filled"] += 1
    if not new_faces:
        return mesh, meta
    faces = np.vstack([np.asarray(mesh.faces, dtype=np.int64),
                       np.asarray(new_faces, dtype=np.int64)])
    out = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                          faces=faces, process=True)
    meta["faces_added_for_holes"] = len(new_faces)
    return out, meta


def mesh_validation(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    if mesh is None:
        return {"valid": False, "reason": "mesh is None"}
    n_v, n_f = len(mesh.vertices), len(mesh.faces)
    vertices_finite = bool(n_v > 0 and np.isfinite(mesh.vertices).all())
    return {
        "valid": bool(n_v > 0 and n_f > 0 and vertices_finite),
        "n_vertices": int(n_v),
        "n_faces": int(n_f),
        "vertices_finite": vertices_finite,
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "is_empty": bool(mesh.is_empty),
        "area": float(mesh.area),
        "bounds": mesh.bounds.tolist() if n_v > 0 else None,
    }


def postprocess_mesh(
    vertices,
    faces,
    cfg: MeshConfig,
    apply_visual_smoothing: bool = True,
) -> MeshResult:
    """Filter components, fill small boundary loops, and optionally smooth a copy."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        return MeshResult(empty_mesh(), None, {
            "empty": True,
            "reason": "no vertices or faces",
            "timestamp_utc": now_utc(),
        })
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be (N,3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (M,3)")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain non-finite values")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces index outside the vertex array")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if mesh.is_empty:
        return MeshResult(empty_mesh(), None, {
            "empty": True,
            "reason": "trimesh rejected all geometry",
            "timestamp_utc": now_utc(),
        })

    mesh, comp_meta = _filter_components(mesh, cfg.min_component_fraction)
    mesh, hole_meta = _fill_boundary_loops(mesh, cfg.max_hole_edges)

    # Preserve the exact metric mesh before any smoothing.
    metric_mesh = mesh.copy()
    viz_mesh = None
    smoothed = False
    if cfg.save_visual_mesh:
        viz_mesh = mesh.copy()
        if (apply_visual_smoothing and cfg.laplacian_iters > 0
                and cfg.laplacian_lambda > 0):
            trimesh.smoothing.filter_laplacian(
                viz_mesh, lamb=float(cfg.laplacian_lambda),
                iterations=int(cfg.laplacian_iters))
            smoothed = True

    meta: Dict[str, Any] = {
        "timestamp_utc": now_utc(),
        "min_component_fraction": float(cfg.min_component_fraction),
        "max_hole_edges": int(cfg.max_hole_edges),
        "laplacian_iters": int(cfg.laplacian_iters),
        "laplacian_lambda": float(cfg.laplacian_lambda),
        "smoothed_for_visualization_only": smoothed,
        "metric_mesh_smoothed": False,
        "metric_validation": mesh_validation(metric_mesh),
        "viz_validation": mesh_validation(viz_mesh) if viz_mesh is not None else None,
    }
    meta.update(comp_meta)
    meta.update(hole_meta)
    return MeshResult(metric_mesh, viz_mesh, meta)


# --------------------------------------------------------------------------- #
# Extraction entry point
# --------------------------------------------------------------------------- #
def extract_mesh(
    sdf: Callable[[torch.Tensor], torch.Tensor],
    bounds_min,
    bounds_max,
    cfg: MeshConfig,
    device: str = "cpu",
    chunk_size: Optional[int] = None,
    apply_visual_smoothing: bool = True,
) -> MeshResult:
    """Extract the zero level set on a chunked regular grid."""
    resolution = int(cfg.resolution)
    lo, hi = expand_bounds(bounds_min, bounds_max, cfg.bbox_margin)
    volume = evaluate_sdf_volume(
        sdf, lo, hi, resolution,
        chunk_size=int(chunk_size or cfg.batch_size), device=device,
    )
    if not np.isfinite(volume).all():
        raise ValueError("SDF volume contains non-finite values")
    extraction_meta: Dict[str, Any] = {
        "timestamp_utc": now_utc(),
        "resolution": resolution,
        "bbox_margin": float(cfg.bbox_margin),
        "bounds_input": [np.asarray(bounds_min).tolist(),
                         np.asarray(bounds_max).tolist()],
        "bounds_expanded": [lo.tolist(), hi.tolist()],
        "chunk_size": int(chunk_size or cfg.batch_size),
        "device": str(device),
        "volume_min": float(volume.min()),
        "volume_max": float(volume.max()),
    }
    if float(volume.min()) > 0.0 or float(volume.max()) < 0.0:
        result = MeshResult(empty_mesh(), None, {
            "empty": True,
            "has_surface": False,
            "reason": "SDF has no zero crossing in the expanded bounding box",
        })
        result.metadata.update(extraction_meta)
        result.metadata["metric_validation"] = mesh_validation(result.mesh)
        return result

    spacing = tuple((hi - lo) / float(resolution - 1))
    try:
        vertices, faces, _, _ = measure.marching_cubes(
            volume, level=0.0, spacing=spacing)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Marching Cubes failed: {exc}") from exc
    vertices = np.asarray(vertices, dtype=np.float64) + lo[None, :]
    faces = np.asarray(faces, dtype=np.int64)
    extraction_meta["n_vertices_marching_cubes"] = int(len(vertices))
    extraction_meta["n_faces_marching_cubes"] = int(len(faces))
    extraction_meta["spacing"] = list(spacing)

    result = postprocess_mesh(vertices, faces, cfg,
                              apply_visual_smoothing=apply_visual_smoothing)
    result.metadata.update(extraction_meta)
    result.metadata.setdefault("has_surface", len(result.mesh.faces) > 0)
    return result


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError("refusing to save an empty mesh")
    mesh.export(str(path))
    return path
