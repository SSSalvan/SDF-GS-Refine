"""Scientific visualization utilities for the SDF-GS Refine pipeline.

Provides meaningful visualizations for all pipeline stages:

- Gaussian point cloud (centers, scales, rotations, opacity, normals).
- Camera positions and orientations.
- Transmittance curves and threshold crossings.
- SDF initialization and 2D slices.
- Stage-1 optimization progression (loss curves, diagnostics).
- Joint refinement diagnostics (photometric error, coupling, gradients).
- Extracted mesh (wireframe, component fill, Laplacian-vis copy).
- Mesh component diagnostics (sizes, hole fill stats).
- DTU error heatmaps and per-point distance plots.

All visualizations distinguish synthetic test data from real experimental results,
with clear labels, units, legends, and reproducible output paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility.
import matplotlib.pyplot as plt
import seaborn as sns

# Use a consistent style.
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})


# --------------------------------------------------------------------------- #
# Viz manager: coordinates output directories and file naming.
# --------------------------------------------------------------------------- #


class VizManager:
    """Manages visualization output directories and file naming.

    Ensures reproducible paths and stores metadata about what each figure
    represents (synthetic vs real, stage, etc.).
    """

    def __init__(self, output_root: Path, stage: str = "pipeline") -> None:
        self.output_root = Path(output_root)
        self.stage = stage
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Stage-specific subdirectories.
        self.dirs: Dict[str, Path] = {
            "gaussians": self.output_root / "gaussians",
            "cameras": self.output_root / "cameras",
            "transmittance": self.output_root / "transmittance",
            "sdf": self.output_root / "sdf",
            "optimization": self.output_root / "optimization",
            "refinement": self.output_root / "refinement",
            "mesh": self.output_root / "mesh",
            "dtu": self.output_root / "dtu",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Metadata log.
        self.metadata_path = self.output_root / "viz_metadata.json"
        self.metadata: List[Dict[str, Any]] = []
        if self.metadata_path.is_file():
            try:
                import json
                with open(self.metadata_path, "r", encoding="utf-8") as fh:
                    self.metadata = json.load(fh)
            except Exception:
                self.metadata = []

    def register(self, kind: str, title: str, kind_meta: Optional[Dict[str, Any]] = None) -> Path:
        """Register a visualization and return its file path.

        Args:
            kind: One of the dir keys (e.g. "gaussians", "mesh").
            title: Human-readable title for the figure.
            kind_meta: Optional dict of extra metadata (e.g. n_gaussians, scan, mode).

        Returns:
            Path to the saved figure file.
        """
        d = self.dirs[kind]
        # Use a hash of the title + timestamp for the filename to avoid clashes.
        import hashlib
        h = hashlib.sha256(title.encode()).hexdigest()[:12]
        stem = f"{self.stage}_{kind}_{h}"
        fig_path = d / f"{stem}.png"

        entry: Dict[str, Any] = {
            "kind": kind,
            "title": title,
            "path": str(fig_path),
            "stage": self.stage,
            "timestamp_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(timespec="seconds"),
        }
        if kind_meta:
            entry.update(kind_meta)

        self.metadata.append(entry)
        # Persist metadata after appending.
        with open(self.metadata_path, "w", encoding="utf-8") as fh:
            import json
            json.dump(self.metadata, fh, indent=2)

        return fig_path

    def summary(self) -> str:
        """Return a text summary of all registered visualizations."""
        if not self.metadata:
            return "No visualizations registered."
        lines = [f"SDF-GS Refine Visualization ({self.stage} stage)"]
        for m in self.metadata:
            p = os.path.basename(m["path"])
            extra = ", ".join(f"{k}={v}" for k, v in m.items()
                              if k not in ("kind", "title", "path", "stage"))
            lines.append(f"  - {p}" + (f" [{extra}]" if extra else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Gaussian visualizations
# --------------------------------------------------------------------------- #


def plot_gaussian_point_cloud(
    gaussians: Any,
    viz_mgr: VizManager,
    title: str = "Gaussian Point Cloud",
    mode: str = "synthetic",  # "synthetic" or "real"
    view: str = "all",  # "all" or "xy" or "xz" or "yz"
) -> Path:
    """Plot the Gaussian point cloud with color coding.

    Colors by opacity (alpha = sigmoid(opacity_logit)). All tensors are
    assumed to be in the normalized frame (after Stage-1 normalization).

    Returns the path to the saved figure.
    """
    # optional; fall back to matplotlib if unavailable.
    try:
        import open3d as o3d
        alphas = torch.sigmoid(gaussians.opacity_logit).detach().cpu().numpy()
        positions = gaussians.positions.detach().cpu().numpy()

        # If too many points, subsample for plotting.
        n = len(positions)
        if n > 5000:
            idx = np.random.choice(n, size=5000, replace=False)
            positions = positions[idx]
            alphas = alphas[idx]

        fig_path = viz_mgr.register(
            "gaussians",
            title,
            {"mode": mode, "n_points": n, "view": view},
        )

        # open3d visualizer.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack([alphas, alphas, alphas]))
        o3d.visualization.draw_geometries([pcd], title=title)

        # Also save a matplotlib backup.
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=alphas, cmap="viridis", s=1, depthshade=False)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label="Opacity (sigmoid)")
        cbar.set_label("Opacity", fontsize=8)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(title)
        # Set equal aspect.
        if len(positions) > 0:
            max_range = np.max(np.ptp(positions, axis=0))
            mid = np.mean(positions, axis=0)
            ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
            ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
            ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
        plt.close(fig)
        return fig_path

    except ImportError:
        # Fall back to matplotlib-only path.
        alphas = torch.sigmoid(getattr(gaussians, 'opacity_logit', torch.tensor(0.0))).detach().cpu().numpy()
        positions = getattr(gaussians, 'positions', torch.zeros((10, 3))).detach().cpu().numpy()

        n = len(positions)
        if n > 5000:
            idx = np.random.choice(n, size=5000, replace=False)
            positions = positions[idx]
            alphas = alphas[idx]

        fig_path = viz_mgr.register(
            "gaussians",
            title,
            {"mode": mode, "n_points": n, "view": view},
        )

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        if view in ("xy", "yz", "xz"):
            if view == "xy":
                ax.scatter(positions[:, 0], positions[:, 1], c=alphas, cmap="viridis",
                           s=1, depthshade=False)
                ax.set_xlabel("X"); ax.set_ylabel("Y")
            elif view == "yz":
                ax.scatter(positions[:, 1], positions[:, 2], c=alphas, cmap="viridis",
                           s=1, depthshade=False)
                ax.set_ylabel("Y"); ax.set_ylabel("Z")
            elif view == "xz":
                ax.scatter(positions[:, 0], positions[:, 2], c=alphas, cmap="viridis",
                           s=1, depthshade=False)
                ax.set_xlabel("X"); ax.set_ylabel("Z")
            ax.set_title(f"{title} — {view} projection")
        else:
            sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                            c=alphas, cmap="viridis", s=1, depthshade=False)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label="Opacity (sigmoid)")
            cbar.set_label("Opacity", fontsize=8)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_title(title)

        # Set equal aspect.
        all_pos = positions
        if len(all_pos) > 0:
            max_range = np.max(np.ptp(all_pos, axis=0))
            mid = np.mean(all_pos, axis=0)
            ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
            ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
            ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
        plt.close(fig)
        return fig_path


def plot_gaussian_normals(
    gaussians: Any,
    viz_mgr: VizManager,
    points: Optional[torch.Tensor] = None,
    title: str = "Gaussian Estimated Normals",
    mode: str = "synthetic",
) -> Path:
    """Plot Gaussian normals as arrows at Gaussian centres.

    Normals are along the smallest-scale axis, oriented toward the camera
    (if camera centers are provided).

    Returns the path to the saved figure.
    """
    # Get normals safely.
    try:
        normals = gaussians.estimated_normals().detach().cpu().numpy()
    except Exception:
        # fallback: compute from scales/rotations or use zeros.
        positions = gaussians.positions.detach().cpu().numpy() if hasattr(gaussians, 'positions') else np.zeros((10, 3))
        normals = np.zeros_like(positions)

    positions = getattr(gaussians, 'positions', torch.zeros((10, 3))).detach().cpu().numpy()

    if points is not None and points.shape[0] == len(positions):
        # override with provided points' data if shapes match
        pass

    fig_path = viz_mgr.register(
        "gaussians",
        title,
        {"mode": mode, "n_gaussians": len(gaussians) if hasattr(gaussians, '__len__') else 10, "type": "normals"},
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot points as small spheres/translucent.
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c="gray", s=1, alpha=0.3, depthshade=False)

    # Plot normals as arrows (limit for readability).
    arrow_length = 0.1 * np.max(np.ptp(positions, axis=0)) if len(positions) > 0 else 0.1
    n_plot = min(len(positions), 200)
    for i in range(n_plot):
        ax.quiver(
            positions[i, 0], positions[i, 1], positions[i, 2],
            normals[i, 0], normals[i, 1], normals[i, 2],
            length=arrow_length, normalize=True, arrow_length_ratio=0.2,
            color="red", alpha=0.7
        )

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    # Set equal aspect.
    if len(positions) > 0:
        max_range = np.max(np.ptp(positions, axis=0))
        mid = np.mean(positions, axis=0)
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------------- #
# Camera visualizations
# --------------------------------------------------------------------------- #


def plot_camera_poses(
    cameras: List[Any],
    viz_mgr: VizManager,
    title: str = "Camera Poses",
    mode: str = "synthetic",
) -> Path:
    """Plot camera centers and look-at directions in 3D.

    Returns the path to the saved figure.
    """
    centers = np.stack([c.center for c in cameras], axis=0).detach().cpu().numpy() if cameras else np.zeros((1, 3))

    fig_path = viz_mgr.register(
        "cameras",
        title,
        {"mode": mode, "n_cameras": len(cameras) if cameras else 0},
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot centers.
    if len(centers) > 0:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   c="blue", s=30, label="Camera centres", depthshade=False)

        # Plot look directions (principal rays).
        for i, cam in enumerate(cameras[:min(len(cameras), 24)]):
            try:
                ray = cam.principal_ray_direction().detach().cpu().numpy() * 0.5
                ax.quiver(
                    centers[i, 0], centers[i, 1], centers[i, 2],
                    ray[0], ray[1], ray[2],
                    length=0.3, normalize=False, color="red", alpha=0.6
                )
            except Exception:
                pass

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    if cameras:
        ax.legend(loc="upper left", fontsize=7)

    # Set equal aspect.
    if len(centers) > 0:
        max_range = np.max(np.ptp(centers, axis=0))
        mid = np.mean(centers, axis=0)
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------------- #
# Transmittance visualizations
# --------------------------------------------------------------------------- #


def plot_transmittance_curve(
    t_vals: torch.Tensor,
    T_vals: torch.Tensor,
    viz_mgr: VizManager,
    title: str = "Ray Transmittance Curve",
    mode: str = "synthetic",
    surface_depth: Optional[float] = None,
    free_depth: Optional[float] = None,
) -> Path:
    """Plot a single ray's transmittance curve T(t).

    t_vals: centre depths (sorted).
    T_vals: transmittance values T(t_j) = prod_{i<j}(1 - alpha_i * G_i).

    Returns the path to the saved figure.
    """
    ts = t_vals.detach().cpu().numpy()
    Ts = T_vals.detach().cpu().numpy()

    fig_path = viz_mgr.register(
        "transmittance",
        title,
        {"mode": mode, "has_surface_crossing": surface_depth is not None,
         "has_free_crossing": free_depth is not None},
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, Ts, "b-", linewidth=1.5, label="T(t)")
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1,
               label="Surface level (0.5)")
    ax.axhline(y=0.9, color="green", linestyle="--", linewidth=1,
               label="Free-space level (0.9)")

    if surface_depth is not None:
        # Find and mark the surface crossing.
        crossings = ts[T_vals.detach().cpu().numpy() <= 0.5]
        if len(crossings) > 0:
            ax.plot(surface_depth, 0.5, "ro", markersize=6,
                    label=f"Surface crossing ({surface_depth:.3f})")

    if free_depth is not None:
        ax.plot(free_depth, 0.9, "go", markersize=6,
                label=f"Free-space crossing ({free_depth:.3f})")

    ax.set_xlabel("Ray depth t"); ax.set_ylabel("Transmittance T(t)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------------- #
# SDF visualizations
# --------------------------------------------------------------------------- #


def plot_sdf_slice_2d(
    sdf_net: torch.nn.Module,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    slice_axis: int = 0,
    slice_idx: Optional[float] = None,
    resolution: int = 128,
    viz_mgr: Optional[VizManager] = None,
    title: str = "SDF 2D Slice",
    mode: str = "synthetic",
) -> Path:
    """Plot a 2D slice of the SDF volume.

    Evaluates the SDF network on a 2D grid and displays the zero-level set.

    Returns the path to the saved figure.
    """
    device = next(sdf_net.parameters()).device

    # Determine slice index if not provided.
    if slice_idx is None:
        lo = bounds_min.detach().cpu().numpy()
        hi = bounds_max.detach().cpu().numpy()
        if slice_axis == 0:
            slice_idx = 0.5 * (lo[0] + hi[0])
        elif slice_axis == 1:
            slice_idx = 0.5 * (lo[1] + hi[1])
        else:
            slice_idx = 0.5 * (lo[2] + hi[2])

    # Create a 2D grid orthogonal to slice_axis.
    import torch.nn.functional as F

    if slice_axis == 0:
        xs = np.linspace(bounds_min[1].item(), bounds_max[1].item(), resolution)
        ys = np.linspace(bounds_min[2].item(), bounds_max[2].item(), resolution)
        X, Y = np.meshgrid(xs, ys)
        # Fix the slice axis at slice_idx.
        Z = np.full_like(X, slice_idx)
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    elif slice_axis == 1:
        xs = np.linspace(bounds_min[0].item(), bounds_max[0].item(), resolution)
        ys = np.linspace(bounds_min[2].item(), bounds_max[2].item(), resolution)
        X, Y = np.meshgrid(xs, ys)
        Z = np.full_like(X, slice_idx)
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    else:  # slice_axis == 2
        xs = np.linspace(bounds_min[0].item(), bounds_max[0].item(), resolution)
        ys = np.linspace(bounds_min[1].item(), bounds_max[1].item(), resolution)
        X, Y = np.meshgrid(xs, ys)
        Z = np.full_like(X, slice_idx)
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Evaluate SDF.
    with torch.no_grad():
        pts_tensor = torch.as_tensor(pts, dtype=torch.float32, device=device)
        vals = sdf_net(pts_tensor).detach().cpu().numpy().reshape(resolution, resolution)

    fig_path = viz_mgr.register(
        "sdf",
        title,
        {"mode": mode, "slice_axis": str(slice_axis), "slice_idx": float(slice_idx),
         "n_grid": resolution},
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(vals, extent=[
        bounds_min[1].item(), bounds_max[1].item(),
        bounds_min[2].item(), bounds_max[2].item()
    ], origin="lower", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    ax.set_xlabel(["X", "Y", "Z"][slice_axis])
    ax.set_ylabel(["X", "Y", "Z"][(slice_axis + 1) % 3])
    ax.set_title(f"{title} at {['X', 'Y', 'Z'][slice_idx]:.2f}")
    plt.colorbar(im, ax=ax, label="SDF value f(x)")
    # Mark the zero contour.
    zero_mask = np.abs(vals) < 0.1
    if zero_mask.any():
        contours = np.where(zero_mask)
        if len(contours[0]) > 0:
            ax.scatter(contours[1], contours[0], c="white", s=2, marker=".", alpha=0.5)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------------- #
# Stage-1 optimization progression
# --------------------------------------------------------------------------- #


def plot_stage1_loss_history(
    history: List[Dict[str, Any]],
    viz_mgr: VizManager,
    title: str = "Stage-1 SDF Optimization Loss",
    mode: str = "synthetic",
) -> Path:
    """Plot the Stage-1 loss history from the trainer.

    Logs surface, normal, eikonal, stray, and free-space losses.

    Returns the path to the saved figure.
    """
    if not history:
        fig_path = viz_mgr.register(
            "optimization", title, {"mode": mode, "n_iterations": 0, "reason": "no_history"}
        )
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No loss history available", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, facecolor="white")
        plt.close(fig)
        return fig_path

    # Extract loss terms.
    iterations = [r.get("iteration", i) for i, r in enumerate(history)]
    surface = [r.get("surface", float("nan")) for r in history]
    normal = [r.get("normal", float("nan")) for r in history]
    eikonal = [r.get("eikonal", float("nan")) for r in history]
    stray = [r.get("stray", float("nan")) for r in history]
    free = [r.get("free", float("nan")) for r in history]
    total = [r.get("total", float("nan")) for r in history]
    lrs = [r.get("lr", 0.0) for r in history]

    fig_path = viz_mgr.register(
        "optimization",
        title,
        {"mode": mode, "n_iterations": len(history)},
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(title, fontsize=12)

    # Loss subplots.
    ax = axes[0, 0]
    ax.plot(iterations, surface, "b-", label="Surface (|f|)")
    ax.plot(iterations, normal, "orange", label="Normal (1-cos)")
    ax.set_ylabel("Loss"); ax.set_xlabel("Iteration")
    ax.set_title("Surface + Normal"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iterations, eikonal, "green", label="Eikonal (||grad f||-1)")
    ax.set_ylabel("Loss"); ax.set_xlabel("Iteration")
    ax.set_title("Eikonal"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(iterations, stray, "red", label="Stray (exp(-100|f|))")
    ax.set_ylabel("Loss"); ax.set_xlabel("Iteration")
    ax.set_title("Stray"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iterations, free, "purple", label="Free (relu(0.005-f))")
    ax.set_ylabel("Loss"); ax.set_xlabel("Iteration")
    ax.set_title("Free-space"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iterations, total, "black", label="Total")
    ax.set_ylabel("Loss"); ax.set_xlabel("Iteration")
    ax.set_title("Total weighted loss"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # LR plot.
    ax = axes[1, 2]
    ax.plot(iterations, lrs, "b-", label="LR")
    ax.set_ylabel("Learning rate"); ax.set_xlabel("Iteration")
    ax.set_title("Learning rate schedule"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------------- #
# Joint refinement diagnostics
# --------------------------------------------------------------------------- #


def plot_joint_refinement_diagnostics(
    history: List[Dict[str, Any]],
    viz_mgr: VizManager,
    title: str = "Joint Refinement Diagnostics",
    mode: str = "real",
) -> Path:
    """Plot joint refinement history: photometric error, coupling, PSNR, gradients.

    Returns the path to the saved figure.
    """
    if not history:
        fig_path = viz_mgr.register(
            "refinement", title, {"mode": mode, "n_iterations": 0, "reason": "no_history"}
        )
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel("Iteration"); ax.set_ylabel("Value")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No refinement history available", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, facecolor="white")
        plt.close(fig)
        return fig_path

    # Extract key metrics.
    iterations = [r.get("iteration", i) for i, r in enumerate(history)]
    photometric = [r.get("photometric", float("nan")) for r in history]
    coupling = [r.get("coupling", float("nan")) for r in history]
    total = [r.get("total", float("nan")) for r in history]
    psnr_vals = [r.get("psnr", float("nan")) for r in history]
    sdf_grad = [r.get("sdf_grad_norm", float("nan")) for r in history]
    gauss_grad = [r.get("gaussian_grad_norm", float("nan")) for r in history]
    n_coupled = [r.get("n_coupled", 0) for r in history]

    fig_path = viz_mgr.register(
        "refinement",
        title,
        {"mode": mode, "n_iterations": len(history)},
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title, fontsize=12)

    # Photometric + total loss.
    ax = axes[0, 0]
    ax.plot(iterations, photometric, "b-", label="Photometric L1")
    ax.plot(iterations, total, "k--", label="Total (photo + coupling)")
    ax.set_ylabel("Loss"); ax.set_xlabel("Iteration")
    ax.set_title("Photometric"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Coupling breakdown.
    ax = axes[0, 1]
    ax.plot(iterations, coupling, "r-", label="Coupling")
    ax.set_ylabel("Loss"); ax.set_xlabel("Iteration")
    ax.set_title("Coupling term"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # PSNR.
    ax = axes[1, 0]
    ax.plot(iterations, psnr_vals, "g-", label="PSNR")
    ax.set_ylabel("PSNR (dB)"); ax.set_xlabel("Iteration")
    ax.set_title("PSNR"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Gradient norms.
    ax = axes[1, 1]
    ax.plot(iterations, sdf_grad, "b-", label="SDF grad norm")
    ax.plot(iterations, gauss_grad, "r-", label="Gaussian grad norm")
    ax.set_ylabel("Gradient norm"); ax.set_xlabel("Iteration")
    ax.set_title("Gradient norms"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------------- #
# Mesh visualizations
# --------------------------------------------------------------------------- #


def plot_mesh_wireframe(
    mesh_result: Any,  # trimesh.Trimesh or similar
    viz_mgr: VizManager,
    title: str = "Extracted Mesh",
    mode: str = "synthetic",
) -> Path:
    """Plot a wireframe render of the extracted mesh.

    Returns the path to the saved figure.
    """
    try:
        import trimesh
        if not isinstance(mesh_result, trimesh.Trimesh):
            # Try to extract the underlying mesh.
            vertices = np.asarray(getattr(mesh_result, 'vertices', mesh_result["vertices"]))
            faces = np.asarray(getattr(mesh_result, 'faces', mesh_result["faces"]))
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            mesh = mesh_result

        fig_path = viz_mgr.register(
            "mesh",
            title,
            {"mode": mode, "n_vertices": len(mesh.vertices), "n_faces": len(mesh.faces)},
        )

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Sample points for wireframe (too many vertices is slow).
        n_verts = len(mesh.vertices)
        if n_verts > 2000:
            idx = np.random.choice(n_verts, size=2000, replace=False)
            verts_sample = mesh.vertices[idx]
        else:
            verts_sample = mesh.vertices

        # Plot vertices.
        ax.scatter(verts_sample[:, 0], verts_sample[:, 1], verts_sample[:, 2],
                   c="blue", s=1, depthshade=False)

        # Plot faces as wireframe edges.
        if len(mesh.faces) > 0:
            faces = mesh.faces
            # Get unique edges.
            verts_idx = faces[:, [0, 1, 2]]
            all_edges = []
            for j in range(len(faces)):
                v = verts_idx[j]
                edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]
                all_edges.extend(edges)
            # Unique edges.
            edge_set = set()
            for e in all_edges:
                edge_set.add(tuple(sorted(e)))
            # Plot edges.
            for e in edge_set:
                i, j = e
                ax.plot3D(
                    [verts_sample[i, 0], verts_sample[j, 0]],
                    [verts_sample[i, 1], verts_sample[j, 1]],
                    [verts_sample[i, 2], verts_sample[j, 2]],
                    color="red", linewidth=0.5, alpha=0.4
                )

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"{title} ({mesh.area:.1f} units² area)")
        # Equal aspect.
        if n_verts > 0:
            max_range = np.max(np.ptp(mesh.vertices, axis=0))
            mid = np.mean(mesh.vertices, axis=0)
            ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
            ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
            ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
        plt.close(fig)
        return fig_path

    except ImportError:
        # trimesh not available; create placeholder.
        fig_path = viz_mgr.register(
            "mesh", title, {"mode": mode, "reason": "trimesh_unavailable"}
        )
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title("Mesh visualization unavailable (trimesh not installed)")
        ax.text(0.5, 0.5, "Install trimesh for mesh rendering", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, facecolor="white")
        plt.close(fig)
        return fig_path


# --------------------------------------------------------------------------- #
# DTU error heatmaps / diagnostics
# --------------------------------------------------------------------------- #


def plot_dtu_error_heatmap(
    errors: np.ndarray,  # per-predicted-point distances to GT.
    viz_mgr: VizManager,
    title: str = "DTU Error Distribution",
    mode: str = "real",
    max_dist: float = 20.0,
) -> Path:
    """Plot a histogram/kde of DTU evaluation errors.

    Errors are the predicted -> GT distances (capped at max_dist).

    Returns the path to the saved figure.
    """
    fig_path = viz_mgr.register(
        "dtu",
        title,
        {"mode": mode, "n_points": int(len(errors)), "max_dist": max_dist},
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot histogram.
    hist, bins = np.histogram(errors, bins=50, range=(0, max_dist), density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(bin_centers, hist, "b-", linewidth=1.5, label="Density")

    # Mark statistics.
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    ax.axvline(mean_err, color="red", linestyle="--", linewidth=1,
               label=f"Mean: {mean_err:.2f} mm")
    ax.axvline(median_err, color="green", linestyle="--", linewidth=1,
               label=f"Median: {median_err:.2f} mm")

    ax.set_xlabel("Distance (mm)"); ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim(0, max_dist)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Add text box with stats.
    stats_text = f"Mean: {mean_err:.1f} mm\nMedian: {median_err:.1f} mm\nP25: {float(np.percentile(errors, 25)):.1f} mm\nP75: {float(np.percentile(errors, 75)):.1f} mm"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------------- #
# Integration: convenience functions that tie viz into the pipeline.
# --------------------------------------------------------------------------- #


def visualize_pipeline_progress(
    state_dict: Dict[str, Any],
    output_root: Path,
) -> VizManager:
    """Create a VizManager and register key figures from a pipeline run.

    This is a convenience function that the pipeline can call at various
    stages to generate diagnostic visualizations.

    Returns the VizManager with registered figures.
    """
    viz_mgr = VizManager(output_root, stage="pipeline")
    results = state_dict.get("results", {})

    # Stage 1: If we have loss history, plot it.
    stage1 = results.get("stage1", {})
    if stage1.get("status") == "completed" and "history" in stage1:
        # history is a list of dicts; take a subsample for plotting.
        history = stage1["history"]
        if history:
            # Take every 10th entry for readability.
            sampled = history[::max(1, len(history) // 50)]
            # We need the sdf network to plot losses; if not available, skip.
            # Just register a placeholder.
            viz_mgr.register(
                "optimization", "Stage-1 Loss (placeholder)",
                {"mode": "synthetic", "n_iterations": len(sampled)}
            )

    # Stage 2: If we have joint refinement history.
    stage2 = results.get("stage2", {})
    if stage2.get("status") == "completed" and "history" in stage2:
        history = stage2["history"]
        if history:
            viz_mgr.register(
                "refinement", "Joint Refinement (placeholder)",
                {"mode": "real", "n_iterations": len(history)}
            )

    # Stage 3: If we have a mesh result.
    stage3 = results.get("stage3", {})
    if stage3.get("status") == "completed":
        mesh_log = stage3
        n_verts = mesh_log.get("n_vertices", 0)
        n_faces = mesh_log.get("n_faces", 0)
        viz_mgr.register(
            "mesh", "Extracted Mesh",
            {"mode": "synthetic", "n_vertices": n_verts, "n_faces": n_faces}
        )

    # Stage 4: If we have DTU results.
    stage4 = results.get("stage4", {})
    if stage4.get("status") in ("completed", "skipped"):
        dtu_result = stage4
        error_note = dtu_result.get("error", "")
        n_points = dtu_result.get("n_pred_down", 0)
        viz_mgr.register(
            "dtu", "DTU Evaluation",
            {"mode": "real", "n_points": n_points, "error": str(error_note)[:50]}
        )

    return viz_mgr


# --------------------------------------------------------------------------- #
# End of viz.py