"""
mesh.py

Subbab 3.6 (Mesh extraction): Marching Cubes on the zero-crossing of the
optimized SDF, grid resolution configurable (paper uses 512^3; see config
for the smaller resolution actually used in the smoke test and why).
Subbab 3.7 (Mesh refinement):
  (1) connected-component filtering
  (2) hole filling for small holes (< max_hole_edges boundary edges)
  (3) OPTIONAL Laplacian smoothing (3 iters, lambda=0.5) -- qualitative
      visualization only, NEVER applied to the mesh used for geometric
      metrics (Chamfer/F-score), exactly as specified in the paper.

*** DEVIATION FROM PAPER (this revision, documented explicitly) ***
Default component filtering changed from relative-area threshold
(min_component_frac=0.01) to keep-only-largest, since the DTU ground-truth
object is a single connected surface and the relative threshold let
multiple fragments survive. Old behavior still available via
keep_only_largest=False.

*** FIX (this revision) ***
Hole filling now uses size-filtered boundary-loop detection instead of
trimesh's unconditional fill_holes(). Holes exceeding max_hole_edges are
logged and left open. Bare except:pass removed -- errors propagate.
"""
import numpy as np
import torch
import trimesh
import networkx as nx
from trimesh.grouping import group_rows
from skimage import measure


@torch.no_grad()
def extract_mesh(sdf_net, bounds_min, bounds_max, resolution, device, batch_size=65536):
    xs = np.linspace(bounds_min[0], bounds_max[0], resolution)
    ys = np.linspace(bounds_min[1], bounds_max[1], resolution)
    zs = np.linspace(bounds_min[2], bounds_max[2], resolution)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)

    vals = np.empty((grid.shape[0],), dtype=np.float32)
    for i in range(0, grid.shape[0], batch_size):
        chunk = torch.tensor(grid[i:i + batch_size], device=device)
        vals[i:i + batch_size] = sdf_net(chunk).detach().cpu().numpy()

    vol = vals.reshape(resolution, resolution, resolution)

    if vol.min() > 0 or vol.max() < 0:
        return None, None

    spacing = ((bounds_max[0] - bounds_min[0]) / (resolution - 1),
               (bounds_max[1] - bounds_min[1]) / (resolution - 1),
               (bounds_max[2] - bounds_min[2]) / (resolution - 1))
    verts, faces, normals, _ = measure.marching_cubes(vol, level=0.0, spacing=spacing)
    verts = verts + np.array(bounds_min, dtype=np.float32)
    return verts.astype(np.float32), faces.astype(np.int64)


def _print_component_diagnostic(components):
    sizes = np.array([c.area for c in components])
    order = np.argsort(-sizes)
    print(f"== Mesh component diagnostic: {len(components)} connected component(s) found ==")
    for rank, i in enumerate(order):
        c = components[i]
        frac = sizes[i] / sizes[order[0]]
        print(f"  component {rank}: area={sizes[i]:.6f}  frac_of_largest={frac:.4f}  "
              f"V={len(c.vertices)}  F={len(c.faces)}  centroid={np.round(c.centroid, 4)}")


def postprocess(verts, faces, min_component_frac=0.01, max_hole_edges=100,
                 keep_only_largest=True,
                 laplacian_smooth=False, laplacian_iters=3, laplacian_lambda=0.5):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        _print_component_diagnostic(components)
        if keep_only_largest:
            mesh = max(components, key=lambda c: c.area)
        else:
            sizes = np.array([c.area for c in components])
            keep = sizes >= (min_component_frac * sizes.max())
            kept = [c for c, k in zip(components, keep) if k]
            mesh = trimesh.util.concatenate(kept) if kept else mesh

    # --- Size-filtered hole filling ---
    # Detect boundary edges (edges with only one adjacent face)
    boundary_groups = group_rows(mesh.edges_sorted, require_count=1)
    if len(boundary_groups) > 0:
        boundary_edges = mesh.edges[boundary_groups]
        hole_loops = nx.cycle_basis(nx.from_edgelist(boundary_edges.tolist()))
        filled_count = 0
        skipped_count = 0
        for loop in hole_loops:
            if len(loop) <= max_hole_edges:
                # Small hole: triangulate via fan and append
                loop_verts = mesh.vertices[loop]
                # Fan-triangulate from vertex 0
                new_faces = np.array([[loop[0], loop[i], loop[i+1]]
                                      for i in range(1, len(loop)-1)], dtype=mesh.faces.dtype)
                mesh.faces = np.vstack([mesh.faces, new_faces])
                filled_count += 1
            else:
                skipped_count += 1
        if filled_count > 0:
            print(f"  [mesh] filled {filled_count} hole(s) (<= {max_hole_edges} edges each)")
        if skipped_count > 0:
            print(f"  [mesh] WARNING: {skipped_count} hole(s) exceed {max_hole_edges} edges, left open")

    mesh_for_metrics = mesh.copy()

    mesh_for_viz = mesh.copy()
    if laplacian_smooth:
        trimesh.smoothing.filter_laplacian(mesh_for_viz, lamb=laplacian_lambda,
                                            iterations=laplacian_iters)

    return mesh_for_metrics, mesh_for_viz


def validate_mesh(mesh: trimesh.Trimesh, name="mesh"):
    n_v, n_f = len(mesh.vertices), len(mesh.faces)
    has_nan = not np.isfinite(mesh.vertices).all()
    ok = (n_v > 0) and (n_f > 0) and (not has_nan)
    print(f"== Mesh validation: {name} ==")
    print(f"  vertices={n_v}  faces={n_f}  NaN/Inf={has_nan}  VALID={ok}")
    return ok


def save_mesh(mesh: trimesh.Trimesh, path: str):
    mesh.export(path)
    return path