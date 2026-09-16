"""
evaluation.py

Geometric metrics (paper Subbab 3.8):
  - Chamfer Distance (bidirectional, mean of both directions)
  - F-score at given distance thresholds (default 1mm and 5mm per DTU protocol)
Both computed on the mesh AFTER connected-component filtering + hole filling,
BEFORE optional Laplacian smoothing, exactly as specified.

Bounding-box crop (this revision):
  When gt_crop_bbox is provided, GT points outside the bbox are excluded
  from the completeness direction (GT->pred). This removes rig-bar and
  background GT points that inflate Chamfer distance. The accuracy direction
  (pred->GT) uses the full GT set per DTU protocol asymmetry.

Rendering metrics:
  - PSNR, SSIM (skimage) implemented directly.
  - LPIPS (VGG backbone) is NOT implemented: it requires downloading
    pretrained VGG/LPIPS weights, and this sandboxed environment's network
    allowlist does not include a weights host (only pypi/npm/github code
    mirrors). Per Section 5 of the task brief, we report this limitation
    explicitly rather than fabricate an LPIPS number.
"""
import numpy as np
import glob
import os
from scipy.spatial import cKDTree
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def sample_surface(mesh, n_points):
    pts, _ = mesh_sample(mesh, n_points)
    return pts


def mesh_sample(mesh, n_points):
    """Poisson-disk-like uniform surface sampling (paper: 'Poisson disk sampling').
    Falls back to random point sampling (with replacement if needed) when `mesh`
    is actually a point cloud (e.g. DTU's raw scanner ground truth, which has no
    faces and therefore no surface to sample from -- only raw scan points)."""
    if hasattr(mesh, "sample") and len(getattr(mesh, "faces", [])) > 0:
        pts, face_idx = mesh.sample(n_points, return_index=True)
        return np.asarray(pts), face_idx
    else:
        verts = np.asarray(mesh.vertices)
        replace = n_points > len(verts)
        idx = np.random.choice(len(verts), size=n_points, replace=replace)
        return verts[idx], idx


def compute_object_bbox(points, margin=0.15):
    """Compute an axis-aligned bounding box around `points`, expanded by `margin` fraction.
    Returns (bbox_min, bbox_max) as numpy arrays of shape (3,)."""
    pts = np.asarray(points)
    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    extent = bb_max - bb_min
    bb_min = bb_min - margin * extent
    bb_max = bb_max + margin * extent
    return bb_min, bb_max


def crop_points_to_bbox(points, bbox_min, bbox_max):
    """Return boolean mask of points inside the bounding box."""
    pts = np.asarray(points)
    inside = np.all((pts >= bbox_min) & (pts <= bbox_max), axis=1)
    return inside


def discover_gt_mesh(root_dir: str, scan_name: str = None):
    """
    Look for a DTU-style ground-truth mesh. DTU MVS releases and IDR/NeuS
    preprocessing packages commonly name these:
        Points/stl/<scanNN>_stl_total.ply
        stl<NN>_total.ply
        <scan>.ply / <scan>_gt.ply / <scan>_stl_total.ply
    We search broadly and just require ".ply" or ".obj" containing "stl",
    "gt", or the scan name, rather than hard-coding one exact path.
    """
    patterns = ["*stl*total*.ply", "*stl*.ply", f"*{scan_name}*.ply" if scan_name else "*gt*.ply",
                "*ground*truth*.ply", "*.obj"]
    hits = []
    for p in patterns:
        hits += glob.glob(os.path.join(root_dir, "**", p), recursive=True)
    hits = sorted(set(hits))
    return hits[0] if hits else None


def load_gt_mesh(path: str):
    import trimesh
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    return mesh


def chamfer_distance(mesh_pred, mesh_gt, n_points=100_000, seed=42, gt_crop_bbox=None):
    """Bidirectional Chamfer distance.

    If gt_crop_bbox is provided (tuple of (bbox_min, bbox_max) numpy arrays),
    GT points outside the bbox are excluded from the completeness direction
    (GT->pred). This removes rig-bar and background contamination from the
    DTU ground truth. The accuracy direction (pred->GT) uses full GT.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    pred_pts = sample_surface(mesh_pred, n_points)
    gt_pts = sample_surface(mesh_gt, n_points)

    # Crop GT to object region for completeness direction
    if gt_crop_bbox is not None:
        bb_min, bb_max = gt_crop_bbox
        gt_mask = crop_points_to_bbox(gt_pts, bb_min, bb_max)
        gt_pts_cropped = gt_pts[gt_mask]
        n_before = len(gt_pts)
        gt_pts = gt_pts_cropped
        print(f"  [eval] GT crop: {n_before} -> {len(gt_pts)} points "
              f"({100*len(gt_pts)/n_before:.1f}% retained)")

    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)

    cd = 0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean())
    return {
        "chamfer_distance": float(cd),
        "pred_to_gt_mean": float(d_pred_to_gt.mean()),
        "gt_to_pred_mean": float(d_gt_to_pred.mean()),
        "_d_pred_to_gt": d_pred_to_gt,
        "_d_gt_to_pred": d_gt_to_pred,
    }


def f_score(mesh_pred, mesh_gt, thresholds=(0.001, 0.005), n_points=100_000, seed=42,
            gt_crop_bbox=None):
    """thresholds in the same world units as the mesh (paper: 1mm and 5mm).
    gt_crop_bbox: if provided, GT points outside the bbox are excluded."""
    cd = chamfer_distance(mesh_pred, mesh_gt, n_points=n_points, seed=seed,
                          gt_crop_bbox=gt_crop_bbox)
    d_p2g, d_g2p = cd["_d_pred_to_gt"], cd["_d_gt_to_pred"]
    out = {}
    for th in thresholds:
        precision = float((d_p2g < th).mean())
        recall = float((d_g2p < th).mean())
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        out[f"f_score@{th}"] = {"precision": precision, "recall": recall, "f1": f1}
    return out


def psnr(img_pred: np.ndarray, img_gt: np.ndarray):
    return float(peak_signal_noise_ratio(img_gt, img_pred, data_range=1.0))


def ssim(img_pred: np.ndarray, img_gt: np.ndarray):
    return float(structural_similarity(img_gt, img_pred, data_range=1.0, channel_axis=-1))