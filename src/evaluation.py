"""
Official DTU Benchmark Evaluation & Multi-View 3D Surface Visualization (`evaluation.py`).

Implements:
1. Official DTU MVS Evaluation Protocol (Jensen et al., CVPR 2014 / Aanæs et al., IJCV 2016):
   - Aligns COLMAP/3DGS world coordinates to DTU millimeter coordinates using ONLY the
     camera-derived Sim(3) transform (`compute_colmap_to_dtu_sim3` from `cameras.npz`),
     eliminating the 4.3-4.8 mm GT-ICP alignment distortion identified in the thesis audit.
   - Applies voxel grid subsampling (`thresh = 0.2 mm`), `ObsMask{scan}_10.mat` volumetric
     visibility mask filtering, `Plane{scan}.mat` ground-plane filtering, and `max_dist = 20.0 mm`
     outlier cap.
   - Reports Accuracy (mm), Completeness (mm), Chamfer Distance (mean & median mm),
     Precision/Recall/F-score at 1.0 mm, 2.0 mm, and 5.0 mm, and mesh topology diagnostics.
2. Evaluates Pure Metric SDF Mesh vs. Post-Processed Viz Mesh vs. Stage 1 Mesh vs.
   MVC Point Cloud vs. Original 3DGS Centers vs. Historical Thesis Baselines.
3. High-Resolution Multi-View Lambertian 3D Surface Renderer (pure NumPy/Matplotlib software
   z-buffer with directional key + fill + rim lighting) for visual comparison across multiple
   viewpoints and zoomed close-up crops without requiring headless OpenGL on Paperspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from plyfile import PlyData
import scipy.io as sio
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import trimesh


# Reference baseline table from Laporan_SilvanusAlvan_Revised.docx (Table 4.2 & Table 4.10)
# Clearly labeled as Historical Thesis Reference for comparison when external baseline PLYs
# are not present on a remote Paperspace instance.
THESIS_HISTORICAL_BASELINES: Dict[int, Dict[str, Dict[str, float]]] = {
    105: {
        "Thesis_Original_SDF_GS (Stage 2)": {"chamfer_mean_mm": 2.80, "accuracy_mean_mm": 2.95, "completeness_mean_mm": 2.65},
        "Thesis_Original_SDF_GS (Stage 1)": {"chamfer_mean_mm": 2.27, "accuracy_mean_mm": 2.41, "completeness_mean_mm": 2.13},
        "Thesis_Reported_PGSR": {"chamfer_mean_mm": 0.69},
        "Thesis_Reported_2DGS": {"chamfer_mean_mm": 0.76},
        "Thesis_Reported_GOF": {"chamfer_mean_mm": 0.85},
        "Thesis_Reported_SuGaR": {"chamfer_mean_mm": 1.18},
    },
    24: {
        "Thesis_Original_SDF_GS (Stage 2)": {"chamfer_mean_mm": 3.07},
        "Thesis_Original_SDF_GS (Stage 1)": {"chamfer_mean_mm": 2.34},
        "Thesis_Reported_PGSR": {"chamfer_mean_mm": 0.56},
        "Thesis_Reported_2DGS": {"chamfer_mean_mm": 0.65},
        "Thesis_Reported_GOF": {"chamfer_mean_mm": 0.68},
        "Thesis_Reported_SuGaR": {"chamfer_mean_mm": 1.14},
    },
    37: {
        "Thesis_Original_SDF_GS (Stage 2)": {"chamfer_mean_mm": 5.97},
        "Thesis_Original_SDF_GS (Stage 1)": {"chamfer_mean_mm": 3.55},
        "Thesis_Reported_PGSR": {"chamfer_mean_mm": 0.72},
        "Thesis_Reported_2DGS": {"chamfer_mean_mm": 0.89},
        "Thesis_Reported_GOF": {"chamfer_mean_mm": 0.90},
        "Thesis_Reported_SuGaR": {"chamfer_mean_mm": 1.26},
    },
}


def reduce_pts_grid(pts: np.ndarray, thresh: float = 0.2) -> np.ndarray:
    """Official DTU 0.2 mm voxel grid point reduction."""
    if len(pts) == 0 or thresh <= 0:
        return pts
    coords = np.floor(pts / thresh).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return pts[np.sort(unique_idx)]


def load_dtu_gt_points(
    gt_ply_path: Optional[str | Path],
    gt_thin_npy_path: Optional[str | Path] = None,
    max_points: int = 2000000,
) -> np.ndarray:
    """Load official DTU ground-truth point cloud in millimeter space."""
    if gt_ply_path is not None and Path(gt_ply_path).exists():
        ply = PlyData.read(str(gt_ply_path))
        v = ply["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    elif gt_thin_npy_path is not None and Path(gt_thin_npy_path).exists():
        pts = np.load(str(gt_thin_npy_path)).astype(np.float32)
    else:
        raise FileNotFoundError(
            "Neither gt_ply_path nor gt_thin_npy_path exists. "
            "On Paperspace, run `kagglehub.dataset_download('silvanusalvan/dtu-groundtruth')`."
        )

    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts


def compute_mesh_topology_metrics(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Compute structural & topological diagnostics of an extracted 3D mesh."""
    n_verts = int(len(mesh.vertices))
    n_faces = int(len(mesh.faces))
    if n_faces == 0:
        return {
            "vertices": 0,
            "faces": 0,
            "components": 0,
            "is_watertight": False,
            "euler_number": 0,
        }
    comps = mesh.split(only_watertight=False)
    return {
        "vertices": n_verts,
        "faces": n_faces,
        "components": int(len(comps)),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
    }


def evaluate_dtu_point_cloud(
    data_pts_dtu: np.ndarray,
    gt_pts_dtu: np.ndarray,
    obs_mask_path: Optional[str | Path] = None,
    plane_mat_path: Optional[str | Path] = None,
    downsample_density: float = 0.2,
    max_dist: float = 20.0,
    patch_radius: float = 60.0,
) -> Dict[str, float]:
    """
    Official DTU accuracy, completeness, Chamfer distance, and multi-threshold F-scores.
    """
    pts = reduce_pts_grid(data_pts_dtu.astype(np.float32), thresh=downsample_density)
    if len(pts) == 0:
        return {
            "accuracy_mean_mm": 99.9,
            "completeness_mean_mm": 99.9,
            "chamfer_mean_mm": 99.9,
            "fscore_1mm": 0.0,
            "fscore_2mm": 0.0,
            "fscore_5mm": 0.0,
        }

    # Filter predicted points to patch_radius (60 mm) of GT
    gt_tree = cKDTree(gt_pts_dtu)
    d_patch, _ = gt_tree.query(pts, k=1, workers=-1)
    pts = pts[d_patch < patch_radius]
    if len(pts) == 0:
        pts = reduce_pts_grid(data_pts_dtu.astype(np.float32), thresh=downsample_density)

    # Apply official Plane.mat and ObsMask.mat filtering for Accuracy (Data -> GT)
    pts_acc = pts
    if plane_mat_path is not None and Path(plane_mat_path).exists():
        P = sio.loadmat(str(plane_mat_path))["P"].astype(np.float32).ravel()
        pts_hom = np.concatenate([pts_acc, np.ones((len(pts_acc), 1), dtype=np.float32)], axis=1)
        plane_mask = (pts_hom @ P) > 0
        pts_acc = pts_acc[plane_mask]

    if obs_mask_path is not None and Path(obs_mask_path).exists() and len(pts_acc) > 0:
        mat = sio.loadmat(str(obs_mask_path))
        ObsMask = mat["ObsMask"]
        BB = mat["BB"].astype(np.float32)
        Res = float(mat["Res"].item())

        grid_coords = (pts_acc - BB[0:1, :]) / Res
        in_bbox = np.all((grid_coords >= 0) & (grid_coords < np.array(ObsMask.shape) - 1), axis=1)
        valid_idx = np.where(in_bbox)[0]
        if len(valid_idx) > 0:
            gc = grid_coords[valid_idx].T
            interp = map_coordinates(ObsMask.astype(np.float32), gc, order=1, mode="constant", cval=0.0)
            keep_obs = valid_idx[interp > 0.5]
            if len(keep_obs) > 100:
                pts_acc = pts_acc[keep_obs]

    # 1. Accuracy: Data -> GT
    dist_d2s, _ = gt_tree.query(pts_acc, k=1, workers=-1)
    valid_d2s = dist_d2s[dist_d2s < max_dist]
    acc_mean = float(np.mean(valid_d2s)) if len(valid_d2s) else float(max_dist)
    acc_med = float(np.median(valid_d2s)) if len(valid_d2s) else float(max_dist)

    # 2. Completeness: GT -> Data
    pred_tree = cKDTree(pts)
    dist_s2d, _ = pred_tree.query(gt_pts_dtu, k=1, workers=-1)
    valid_s2d = dist_s2d[dist_s2d < max_dist]
    comp_mean = float(np.mean(valid_s2d)) if len(valid_s2d) else float(max_dist)
    comp_med = float(np.median(valid_s2d)) if len(valid_s2d) else float(max_dist)

    chamfer_mean = 0.5 * (acc_mean + comp_mean)
    chamfer_med = 0.5 * (acc_med + comp_med)

    # Multi-threshold F-scores (1.0 mm, 2.0 mm, 5.0 mm)
    fscores = {}
    for tau in [1.0, 2.0, 5.0]:
        prec = float(np.mean(valid_d2s < tau)) if len(valid_d2s) else 0.0
        rec = float(np.mean(valid_s2d < tau)) if len(valid_s2d) else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 1e-8 else 0.0
        key = f"{int(tau)}mm"
        fscores[f"precision_{key}"] = prec
        fscores[f"recall_{key}"] = rec
        fscores[f"fscore_{key}"] = f1

    return {
        "accuracy_mean_mm": acc_mean,
        "completeness_mean_mm": comp_mean,
        "chamfer_mean_mm": chamfer_mean,
        "accuracy_median_mm": acc_med,
        "completeness_median_mm": comp_med,
        "chamfer_median_mm": chamfer_med,
        "num_eval_points": int(len(pts_acc)),
        **fscores,
    }


def evaluate_mesh_on_dtu(
    mesh_or_path: trimesh.Trimesh | str | Path,
    sim3_scale: float,
    sim3_R: np.ndarray,
    sim3_t: np.ndarray,
    gt_pts_dtu: np.ndarray,
    obs_mask_path: Optional[str | Path] = None,
    plane_mat_path: Optional[str | Path] = None,
    num_surface_samples: int = 350000,
    already_in_dtu_space: bool = False,
) -> Dict[str, float]:
    """
    Sample surface points uniformly from a mesh, transform to DTU millimeter space
    via the camera-derived Sim(3), and run the official DTU evaluation + topology check.
    """
    if isinstance(mesh_or_path, (str, Path)):
        mesh = trimesh.load(str(mesh_or_path), process=False)
    else:
        mesh = mesh_or_path

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    topo = {}
    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        topo = compute_mesh_topology_metrics(mesh)
        pts_surf, _ = trimesh.sample.sample_surface(mesh, count=num_surface_samples, seed=42)
        pts_all = np.vstack([ np.asarray(mesh.vertices, dtype=np.float32), pts_surf.astype(np.float32) ])
    else:
        pts_all = np.asarray(mesh.vertices, dtype=np.float32)

    if already_in_dtu_space:
        pts_dtu = pts_all
    else:
        pts_dtu = (sim3_scale * (pts_all @ sim3_R.T) + sim3_t).astype(np.float32)

    metrics = evaluate_dtu_point_cloud(
        data_pts_dtu=pts_dtu,
        gt_pts_dtu=gt_pts_dtu,
        obs_mask_path=obs_mask_path,
        plane_mat_path=plane_mat_path,
    )
    metrics.update(topo)
    return metrics


def _rotation_matrix_from_angles(elev_deg: float, azim_deg: float) -> np.ndarray:
    el = np.deg2rad(elev_deg)
    az = np.deg2rad(azim_deg)
    Rz = np.array([
        [np.cos(az), -np.sin(az), 0.0],
        [np.sin(az),  np.cos(az), 0.0],
        [0.0,         0.0,        1.0],
    ], dtype=np.float32)
    Rx = np.array([
        [1.0, 0.0,          0.0],
        [0.0, np.cos(el), -np.sin(el)],
        [0.0, np.sin(el),  np.cos(el)],
    ], dtype=np.float32)
    return Rx @ Rz


def render_mesh_lambertian_ax(
    ax: plt.Axes,
    mesh: trimesh.Trimesh,
    elev_deg: float = 20.0,
    azim_deg: float = -35.0,
    base_color: Tuple[float, float, float] = (0.76, 0.83, 0.92),
    max_faces_plot: int = 85000,
    zoom_box: Optional[Tuple[float, float, float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Render a 3D triangle mesh onto a 2D Matplotlib Axes with painter's z-sort and
    three-point studio Lambertian + Blinn-Phong shading so fine geometric details,
    surface continuity, and holes/artifacts are immediately visible.
    """
    ax.set_facecolor("#10141c")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#2a3242")

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        if title:
            ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        return

    # Normalize vertices into [-1, 1]^3 for consistent camera framing
    center = 0.5 * ( np.percentile(verts, 1, axis=0) + np.percentile(verts, 99, axis=0) )
    span = float(np.max(np.percentile(verts, 99, axis=0) - np.percentile(verts, 1, axis=0))) + 1e-6
    v_norm = (verts - center[None, :]) / (0.55 * span)

    # Subsample faces if needed for fast figure generation
    if len(faces) > max_faces_plot:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(faces), size=max_faces_plot, replace=False)
        faces = faces[sel]

    R_view = _rotation_matrix_from_angles(elev_deg, azim_deg)
    v_rot = v_norm @ R_view.T  # (V, 3): x right, y depth, z up

    tris = v_rot[faces]  # (F, 3, 3)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    fn = np.cross(e1, e2)
    fn_len = np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-12
    fn /= fn_len

    # Orient normals toward viewer (+y is depth away, -y is toward camera)
    fn_orient = np.where(fn[:, 1:2] > 0, -fn, fn)

    # Three-point studio lighting in view space (x right, -y toward viewer, z up)
    L_key = np.array([-0.45, -0.75, 0.48], dtype=np.float32)
    L_key /= np.linalg.norm(L_key)
    L_fill = np.array([0.60, -0.55, 0.25], dtype=np.float32)
    L_fill /= np.linalg.norm(L_fill)
    L_rim = np.array([0.0, 0.65, 0.75], dtype=np.float32)
    L_rim /= np.linalg.norm(L_rim)

    diff_key = np.clip(fn_orient @ L_key, 0.0, 1.0)
    diff_fill = np.clip(fn_orient @ L_fill, 0.0, 1.0)
    diff_rim = np.clip(fn_orient @ L_rim, 0.0, 1.0)

    # Half-vector specular highlight for fine relief visibility
    H_vec = L_key + np.array([0.0, -1.0, 0.0], dtype=np.float32)
    H_vec /= np.linalg.norm(H_vec)
    spec = np.clip(fn_orient @ H_vec, 0.0, 1.0) ** 24

    intensity = 0.18 + 0.62 * diff_key + 0.22 * diff_fill + 0.14 * diff_rim
    base_c = np.array(base_color, dtype=np.float32)[None, :]
    rgb = np.clip(intensity[:, None] * base_c + 0.18 * spec[:, None], 0.0, 1.0)

    # Painter's algorithm sort from back (+y) to front (-y)
    z_depth = tris[:, :, 1].mean(axis=1)
    order = np.argsort(-z_depth)
    polys_2d = tris[order][:, :, [0, 2]]  # project onto (x, z) screen plane
    colors_sorted = rgb[order]

    coll = PolyCollection(
        polys_2d,
        facecolors=colors_sorted,
        edgecolors="none",
        antialiased=False,
    )
    ax.add_collection(coll)

    if zoom_box is not None:
        ax.set_xlim(zoom_box[0], zoom_box[1])
        ax.set_ylim(zoom_box[2], zoom_box[3])
    else:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")

    if title:
        ax.set_title(title, color="white", fontsize=9.5, fontweight="bold", pad=6)


def generate_reconstruction_comparison_figures(
    scan_name: str,
    our_mesh: trimesh.Trimesh,
    our_raw_mesh: trimesh.Trimesh,
    baseline_meshes: Dict[str, trimesh.Trimesh],
    metrics_dict: Dict[str, Dict[str, float]],
    out_dir: str | Path,
) -> Dict[str, Path]:
    """
    Generate publication-grade 3D reconstruction visual comparison figures:
    1. Multi-view angle comparison (Front, Left-Oblique, Right-Oblique, Top-Oblique)
       comparing Our Improved SDF-GS Refine vs. Raw Manifold vs. Baselines.
    2. Zoomed close-up geometry inspection highlighting fine surface details, edges,
       and hole elimination.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_plot: List[Tuple[str, trimesh.Trimesh, Tuple[float, float, float]]] = [
        ("SDF-GS Refine (Ours)", our_mesh, (0.72, 0.86, 0.98)),
        ("Ours (Untrimmed Manifold)", our_raw_mesh, (0.78, 0.80, 0.88)),
    ]
    palette = {
        "Stage1_Metric": (0.68, 0.88, 0.82),
        "PostProcessed_Viz": (0.76, 0.82, 0.96),
        "Prev_SDF_GS": (0.95, 0.72, 0.68),
        "PGSR": (0.75, 0.92, 0.78),
        "2DGS": (0.92, 0.85, 0.70),
        "GOF": (0.86, 0.76, 0.92),
        "SuGaR": (0.90, 0.75, 0.80),
    }
    for b_name, b_mesh in baseline_meshes.items():
        models_to_plot.append((b_name, b_mesh, palette.get(b_name, (0.82, 0.82, 0.82))))

    views = [
        ("Front View", 15.0, -25.0),
        ("Left Oblique", 20.0, -70.0),
        ("Right Oblique", 20.0, 25.0),
        ("Top-Oblique Relief", 48.0, -30.0),
    ]

    n_cols = len(models_to_plot)
    n_rows = len(views)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.6 * n_cols, 3.3 * n_rows),
        dpi=180,
        facecolor="#0b0e14",
    )
    if n_cols == 1:
        axes = axes[:, None]

    for c_idx, (m_label, m_obj, col) in enumerate(models_to_plot):
        cd_val = metrics_dict.get(m_label, {}).get("chamfer_mean_mm", None)
        f1_val = metrics_dict.get(m_label, {}).get("fscore_1mm", None)
        sub_txt = ""
        if cd_val is not None:
            sub_txt = f"\nCD: {cd_val:.2f} mm"
            if f1_val is not None:
                sub_txt += f" | F1@1mm: {f1_val*100:.1f}%"

        for r_idx, (v_name, el, az) in enumerate(views):
            ax = axes[r_idx, c_idx]
            ttl = f"{m_label}{sub_txt}" if r_idx == 0 else None
            render_mesh_lambertian_ax(
                ax=ax,
                mesh=m_obj,
                elev_deg=el,
                azim_deg=az,
                base_color=col,
                title=ttl,
            )
            if c_idx == 0:
                ax.set_ylabel(v_name, color="#d0d8e8", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"DTU {scan_name.upper()} — Multi-View 3D Surface Geometry Comparison (Shaded Mesh Relief)",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    multiview_path = out_dir / f"{scan_name}_multiview_3d_comparison.png"
    fig.savefig(multiview_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Figure 2: Close-up Detail Inspection (Full + 2 Zoom Regions)
    zoom_regions = [
        ("Full Object Geometry", None),
        ("Close-Up: Upper / Structural Detail", (-0.48, 0.48, 0.02, 0.85)),
        ("Close-Up: Central Surface & Crevices", (-0.45, 0.45, -0.45, 0.45)),
    ]
    fig2, axes2 = plt.subplots(
        len(zoom_regions), n_cols,
        figsize=(3.6 * n_cols, 3.4 * len(zoom_regions)),
        dpi=180,
        facecolor="#0b0e14",
    )
    if n_cols == 1:
        axes2 = axes2[:, None]

    for c_idx, (m_label, m_obj, col) in enumerate(models_to_plot):
        for r_idx, (z_name, z_box) in enumerate(zoom_regions):
            ax = axes2[r_idx, c_idx]
            ttl = m_label if r_idx == 0 else None
            render_mesh_lambertian_ax(
                ax=ax,
                mesh=m_obj,
                elev_deg=18.0,
                azim_deg=-30.0,
                base_color=col,
                zoom_box=z_box,
                title=ttl,
            )
            if c_idx == 0:
                ax.set_ylabel(z_name, color="#d0d8e8", fontsize=9.5, fontweight="bold")

    fig2.suptitle(
        f"DTU {scan_name.upper()} — Close-Up 3D Geometric Detail & Continuity Inspection",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.992,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.975])
    closeup_path = out_dir / f"{scan_name}_closeup_detail_comparison.png"
    fig2.savefig(closeup_path, facecolor=fig2.get_facecolor())
    plt.close(fig2)

    return {
        "multiview_comparison": multiview_path,
        "closeup_comparison": closeup_path,
    }
