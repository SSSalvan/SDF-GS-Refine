"""
Camera calibration, COLMAP-to-DTU coordinate alignment, and ray generation.

Handles:
1. Loading 3DGS `cameras.json` (intrinsics fx, fy, width, height and extrinsics R, t).
2. Loading DTU `cameras.npz` (`world_mat_i` and `scale_mat_i`) to compute the exact
   camera-derived similarity transform (Sim(3)) between COLMAP/3DGS world space
   and official DTU millimeter evaluation space without GT point-cloud ICP leakage.
3. Bundled DTU calibration fallback (`configs/dtu_calibration/`) and Paperspace /
   KaggleHub multi-root path resolution so official `scale_mat_0`, `ObsMask`, and
   `Plane` metadata are always available on Paperspace GPU instances.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch


PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLED_CALIB_DIR = PACKAGE_ROOT / "configs" / "dtu_calibration"


@dataclass
class CameraPose:
    id: int
    img_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_c2w: np.ndarray        # (3, 3) camera-to-world rotation
    C_world: np.ndarray      # (3,) camera center in COLMAP/3DGS world coordinates
    R_w2c: np.ndarray        # (3, 3) world-to-camera rotation
    t_w2c: np.ndarray        # (3,) world-to-camera translation
    alpha_mask: Optional[np.ndarray] = None  # (H, W) float32 in [0, 1] if available


def load_cameras_json(
    cameras_json_path: str | Path,
    images_dir: Optional[str | Path] = None,
    load_alpha_masks: bool = True,
) -> List[CameraPose]:
    """
    Load 3DGS `cameras.json`.
    Note on 3DGS `cameras.json` convention:
      `position` is camera center C_world in world coordinates.
      `rotation` is camera-to-world rotation matrix R_c2w (columns are camera axes in world).
      World-to-camera transformation is: X_cam = R_c2w^T @ (X_world - C_world).
    """
    cameras_json_path = Path(cameras_json_path)
    if not cameras_json_path.exists():
        raise FileNotFoundError(
            f"Required camera calibration file not found: {cameras_json_path}\n"
            f"Please verify `cameras_json` in your config YAML or check configs/dtu_calibration/."
        )

    with open(cameras_json_path, "r", encoding="utf-8") as f:
        raw_cams = json.load(f)

    images_root = Path(images_dir) if images_dir is not None else None
    cameras: List[CameraPose] = []

    for entry in raw_cams:
        w = int(entry["width"])
        h = int(entry["height"])
        fx = float(entry["fx"])
        fy = float(entry["fy"])
        cx = w / 2.0
        cy = h / 2.0

        C_world = np.array(entry["position"], dtype=np.float32)
        R_c2w = np.array(entry["rotation"], dtype=np.float32)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ C_world

        img_name = str(entry["img_name"])
        alpha_mask = None
        if load_alpha_masks and images_root is not None and images_root.exists():
            for ext in [".png", ".PNG", ".jpg", ".JPG"]:
                cand = images_root / f"{img_name}{ext}"
                if not cand.exists() and img_name.endswith((".png", ".jpg")):
                    cand = images_root / img_name
                if cand.exists():
                    with Image.open(cand) as im:
                        if im.mode == "RGBA":
                            alpha_mask = np.asarray(im, dtype=np.float32)[..., 3] / 255.0
                    break

        cameras.append(
            CameraPose(
                id=int(entry["id"]),
                img_name=img_name,
                width=w,
                height=h,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                R_c2w=R_c2w,
                C_world=C_world,
                R_w2c=R_w2c,
                t_w2c=t_w2c,
                alpha_mask=alpha_mask,
            )
        )

    return cameras


def compute_colmap_to_dtu_sim3(
    cameras: List[CameraPose],
    cameras_npz_path: str | Path,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the exact Umeyama similarity transform (scale, R, t) mapping
    COLMAP/3DGS world coordinates -> Official DTU millimeter coordinates
    using camera centers from `cameras.json` and `cameras.npz` (`world_mat_i`, `scale_mat_i`).

    No Ground-Truth geometry is used (eliminating evaluation alignment leakage).

    Returns:
        scale (float), R (3, 3), t (3,) such that:
        X_dtu = scale * (X_colmap @ R.T) + t
    """
    cameras_npz_path = Path(cameras_npz_path)
    if not cameras_npz_path.exists():
        raise FileNotFoundError(
            f"DTU cameras.npz not found at {cameras_npz_path}. "
            f"Check configs/dtu_calibration/ for bundled calibration matrices."
        )

    npz = np.load(cameras_npz_path)

    src_centers = []
    dst_centers = []

    for i, cam in enumerate(cameras):
        w_key = f"world_mat_{i}"
        s_key = f"scale_mat_{i}"
        if w_key not in npz:
            continue
        P = npz[w_key] @ (npz[s_key] if s_key in npz else np.eye(4))
        P34 = P[:3, :4].astype(np.float64)
        # Camera center is null-space of P[:3, :4]: P34 @ [C; 1] = 0
        _, _, Vt = np.linalg.svd(P34)
        C_dtu = Vt[-1, :3] / Vt[-1, 3]

        src_centers.append(cam.C_world.astype(np.float64))
        dst_centers.append(C_dtu)

    src = np.stack(src_centers, axis=0)  # (N, 3)
    dst = np.stack(dst_centers, axis=0)  # (N, 3)

    # Umeyama Sim(3) alignment
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    scale = float(np.sum(D * np.diag(S)) / (var_src + 1e-12))
    t = mu_dst - scale * (R @ mu_src)

    return scale, R.astype(np.float32), t.astype(np.float32)


def transform_points_sim3(
    points: np.ndarray,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply X_out = scale * (X @ R.T) + t."""
    return (scale * (points @ R.T) + t).astype(np.float32)


def transform_points_dtu_to_colmap(
    points_dtu: np.ndarray,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply inverse Sim(3): X_colmap = ((X_dtu - t) / scale) @ R."""
    return (((points_dtu - t) / max(scale, 1e-12)) @ R).astype(np.float32)


def generate_camera_rays(
    cam: CameraPose,
    stride: int = 4,
    alpha_threshold: float = 0.5,
    use_alpha_mask: bool = True,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate world-space rays (origins, unit directions, pixel_uv) on a strided pixel grid.

    Returns:
        rays_o: (M, 3) float32 tensor
        rays_d: (M, 3) float32 tensor (unit vectors)
        uv_coords: (M, 2) float32 tensor (pixel coordinates u, v)
    """
    u_grid = np.arange(stride // 2, cam.width, stride, dtype=np.float32)
    v_grid = np.arange(stride // 2, cam.height, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u_grid, v_grid)

    if use_alpha_mask and cam.alpha_mask is not None:
        vi = np.clip(vv.astype(np.int32), 0, cam.height - 1)
        ui = np.clip(uu.astype(np.int32), 0, cam.width - 1)
        keep = cam.alpha_mask[vi, ui] >= alpha_threshold
        uu = uu[keep]
        vv = vv[keep]
    else:
        uu = uu.ravel()
        vv = vv.ravel()

    # Camera space directions
    x_cam = (uu - cam.cx) / cam.fx
    y_cam = (vv - cam.cy) / cam.fy
    z_cam = np.ones_like(x_cam)
    dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (M, 3)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-12

    # Rotate to world space: dirs_world = dirs_cam @ R_c2w.T
    dirs_world = dirs_cam @ cam.R_c2w.T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-12

    origins_world = np.broadcast_to(cam.C_world[None, :], dirs_world.shape).copy()
    uv = np.stack([uu, vv], axis=-1)

    return (
        torch.from_numpy(origins_world).to(device=device, dtype=torch.float32),
        torch.from_numpy(dirs_world).to(device=device, dtype=torch.float32),
        torch.from_numpy(uv).to(device=device, dtype=torch.float32),
    )


def _discover_kagglehub_roots() -> List[Path]:
    """Return candidate KaggleHub cache directories across Windows, Linux, and Paperspace."""
    roots: List[Path] = []
    for env_key in ["SDFGS_3DGS_ROOT", "SDFGS_DTU_ROOT", "KAGGLEHUB_CACHE"]:
        val = os.environ.get(env_key)
        if val and Path(val).exists():
            roots.append(Path(val))

    candidates = [
        Path.home() / ".cache" / "kagglehub" / "datasets",
        Path("/root/.cache/kagglehub/datasets"),
        Path("/notebooks/.cache/kagglehub/datasets"),
        Path("/notebooks/data"),
        Path(r"C:\Users\rapha\.cache\kagglehub\datasets"),
    ]
    for c in candidates:
        if c.exists() and c not in roots:
            roots.append(c)
    return roots


def maybe_download_kaggle_datasets(auto_download: bool = False) -> List[Path]:
    """Optionally trigger kagglehub download on Paperspace if datasets are not yet cached."""
    if not auto_download:
        return []
    downloaded = []
    try:
        import kagglehub  # type: ignore

        for slug in ["silvanusalvan/3dgs-trainingsdfgs", "silvanusalvan/dtu-groundtruth"]:
            try:
                p = Path(kagglehub.dataset_download(slug))
                if p.exists():
                    downloaded.append(p)
            except Exception as exc:
                print(f"[cameras] kagglehub download skipped for {slug}: {exc}")
    except ImportError:
        pass
    return downloaded


def auto_resolve_scan_paths(
    workspace_root: str | Path,
    scan_id: int,
    config_overrides: Optional[Dict[str, str]] = None,
    auto_download_kaggle: bool = False,
) -> Dict[str, Optional[Path]]:
    """
    Resolve all required paths for a DTU scan (`scan105`, `scan24`, `scan37`, etc.)
    across Paperspace (`/notebooks`), KaggleHub caches, local experiment folders,
    and bundled calibration assets (`configs/dtu_calibration/`).

    Supports explicit path overrides via `config_overrides` or environment variables:
      - `SDFGS_WORKSPACE_ROOT`
      - `SDFGS_3DGS_ROOT`
      - `SDFGS_DTU_ROOT`
    """
    env_ws = os.environ.get("SDFGS_WORKSPACE_ROOT")
    ws = Path(env_ws) if env_ws else Path(workspace_root)
    scan_name = f"scan{scan_id}"
    overrides = config_overrides or {}

    kag_roots = _discover_kagglehub_roots()
    if auto_download_kaggle:
        for p in maybe_download_kaggle_datasets(auto_download=True):
            if p not in kag_roots:
                kag_roots.append(p)

    # 1. Pretrained 3DGS PLY
    ply_candidates: List[Path] = []
    if overrides.get("ply_path"):
        ply_candidates.append(Path(overrides["ply_path"]))
    for kr in kag_roots:
        ply_candidates.extend([
            kr / "silvanusalvan" / "3dgs-trainingsdfgs" / "versions" / "3" / "3DGS" / f"{scan_name}_proper" / scan_name / "point_cloud" / "iteration_30000" / "point_cloud.ply",
            kr / "3DGS" / f"{scan_name}_proper" / scan_name / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        ])
        # Also search any version folder inside silvanusalvan/3dgs-trainingsdfgs
        base_3dgs = kr / "silvanusalvan" / "3dgs-trainingsdfgs" / "versions"
        if base_3dgs.exists():
            for ver_dir in sorted(base_3dgs.iterdir(), reverse=True):
                ply_candidates.append(
                    ver_dir / "3DGS" / f"{scan_name}_proper" / scan_name / "point_cloud" / "iteration_30000" / "point_cloud.ply"
                )
    ply_candidates.extend([
        ws / "data" / "3DGS" / f"{scan_name}_proper" / scan_name / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        ws / "data" / scan_name / "point_cloud.ply",
        ws / "checkpoints" / "recovered" / f"kaggle_{scan_name}_proper" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        ws / "HasilTraining" / f"{scan_name}_proper" / scan_name / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        ws / "archive" / "historical" / "all_notebooks_backup" / "outputs" / "3dgs" / scan_name / "point_cloud" / "iteration_30000" / "point_cloud.ply",
    ])
    ply_path = next((p for p in ply_candidates if p.exists()), None)
    if ply_path is None and not auto_download_kaggle:
        for p in maybe_download_kaggle_datasets(auto_download=True):
            if p not in kag_roots:
                kag_roots.append(p)
        for kr in kag_roots:
            ply_candidates.extend([
                kr / "silvanusalvan" / "3dgs-trainingsdfgs" / "versions" / "3" / "3DGS" / f"{scan_name}_proper" / scan_name / "point_cloud" / "iteration_30000" / "point_cloud.ply",
                kr / "3DGS" / f"{scan_name}_proper" / scan_name / "point_cloud" / "iteration_30000" / "point_cloud.ply",
            ])
            base_3dgs = kr / "silvanusalvan" / "3dgs-trainingsdfgs" / "versions"
            if base_3dgs.exists():
                for ver_dir in sorted(base_3dgs.iterdir(), reverse=True):
                    ply_candidates.append(
                        ver_dir / "3DGS" / f"{scan_name}_proper" / scan_name / "point_cloud" / "iteration_30000" / "point_cloud.ply"
                    )
        ply_path = next((p for p in ply_candidates if p.exists()), None)

    # 2. cameras.json (with automatic fallback to bundled configs/dtu_calibration/)
    cam_json_candidates: List[Path] = []
    if overrides.get("cameras_json"):
        cam_json_candidates.append(Path(overrides["cameras_json"]))
    for kr in kag_roots:
        cam_json_candidates.extend([
            kr / "silvanusalvan" / "3dgs-trainingsdfgs" / "versions" / "3" / "3DGS" / f"{scan_name}_proper" / scan_name / "cameras.json",
            kr / "3DGS" / f"{scan_name}_proper" / scan_name / "cameras.json",
        ])
        base_3dgs = kr / "silvanusalvan" / "3dgs-trainingsdfgs" / "versions"
        if base_3dgs.exists():
            for ver_dir in sorted(base_3dgs.iterdir(), reverse=True):
                cam_json_candidates.append(
                    ver_dir / "3DGS" / f"{scan_name}_proper" / scan_name / "cameras.json"
                )
    cam_json_candidates.extend([
        ws / "data" / "3DGS" / f"{scan_name}_proper" / scan_name / "cameras.json",
        ws / "data" / scan_name / "cameras.json",
        ws / "checkpoints" / "recovered" / f"kaggle_{scan_name}_proper" / "cameras.json",
        ws / "HasilTraining" / f"{scan_name}_proper" / scan_name / "cameras.json",
        ws / "archive" / "historical" / "all_notebooks_backup" / "outputs" / "3dgs" / scan_name / "cameras.json",
        BUNDLED_CALIB_DIR / f"cameras_{scan_name}.json",
    ])
    cameras_json = next((p for p in cam_json_candidates if p.exists()), None)

    # 3. cameras.npz (with automatic fallback to bundled configs/dtu_calibration/)
    npz_candidates: List[Path] = []
    if overrides.get("cameras_npz"):
        npz_candidates.append(Path(overrides["cameras_npz"]))
    npz_candidates.extend([
        BUNDLED_CALIB_DIR / f"cameras_{scan_name}.npz",
        ws / "THESIS_BENCHMARK_PACKAGE" / "eval_assets" / f"cameras_{scan_name}.npz",
        ws / "data" / "dtu" / scan_name / "cameras.npz",
        ws / "archive" / "historical" / "all_notebooks_backup" / "Baseline" / "data" / "dtu" / scan_name / "cameras.npz",
        ws / "archive" / "historical" / "all_notebooks_backup" / "Baseline" / "data" / "dtu_raw" / "DTU" / scan_name / "cameras.npz",
    ])
    cameras_npz = next((p for p in npz_candidates if p.exists()), None)

    # 4. Images directory (optional, for RGBA alpha masks & held-out PSNR checks)
    img_candidates: List[Path] = []
    if overrides.get("images_dir"):
        img_candidates.append(Path(overrides["images_dir"]))
    for kr in kag_roots:
        img_candidates.extend([
            kr / "silvanusalvan" / "3dgs-trainingsdfgs" / "versions" / "3" / "SampleSet" / "MVS Data" / "Rectified" / scan_name,
            kr / "SampleSet" / "MVS Data" / "Rectified" / scan_name,
            kr / "silvanusalvan" / "dtu-groundtruth" / "versions" / "1" / "DTU" / scan_name / "images",
        ])
    img_candidates.extend([
        ws / "data" / "dtu" / scan_name / "images",
        ws / "archive" / "historical" / "all_notebooks_backup" / "Baseline" / "data" / "dtu" / scan_name / "images",
        ws / "archive" / "historical" / "all_notebooks_backup" / "Baseline" / "data" / "dtu_raw" / "DTU" / scan_name / "images",
    ])
    images_dir = next((p for p in img_candidates if p.exists()), None)

    # 5. Official DTU Ground Truth & Evaluation Masks (with bundled ObsMask & Plane fallback)
    eval_dir = ws / "THESIS_BENCHMARK_PACKAGE" / "eval_assets"

    gt_ply_candidates: List[Path] = []
    if overrides.get("gt_ply"):
        gt_ply_candidates.append(Path(overrides["gt_ply"]))
    gt_ply_candidates.extend([
        eval_dir / f"stl{scan_id:03d}_total.ply",
        eval_dir / f"stl{scan_id}_total.ply",
    ])
    for kr in kag_roots:
        gt_ply_candidates.extend([
            kr / "silvanusalvan" / "dtu-groundtruth" / "versions" / "1" / "dtu ground truth" / "Points" / "stl" / f"stl{scan_id:03d}_total.ply",
            kr / "silvanusalvan" / "dtu-groundtruth" / "versions" / "1" / "dtu ground truth" / "Points" / "stl" / f"stl{scan_id}_total.ply",
            kr / "dtu ground truth" / "Points" / "stl" / f"stl{scan_id:03d}_total.ply",
            kr / "dtu ground truth" / "Points" / "stl" / f"stl{scan_id}_total.ply",
            kr / "silvanusalvan" / "dtu-groundtruth" / "versions" / "1" / "Points" / "stl" / f"stl{scan_id:03d}_total.ply",
            kr / "silvanusalvan" / "dtu-groundtruth" / "versions" / "1" / "Points" / "stl" / f"stl{scan_id}_total.ply",
            kr / "Points" / "stl" / f"stl{scan_id:03d}_total.ply",
            kr / "Points" / "stl" / f"stl{scan_id}_total.ply",
        ])
        base_gt = kr / "silvanusalvan" / "dtu-groundtruth" / "versions"
        if base_gt.exists():
            for ver_dir in sorted(base_gt.iterdir(), reverse=True):
                gt_ply_candidates.append(ver_dir / "dtu ground truth" / "Points" / "stl" / f"stl{scan_id:03d}_total.ply")
                gt_ply_candidates.append(ver_dir / "dtu ground truth" / "Points" / "stl" / f"stl{scan_id}_total.ply")
                gt_ply_candidates.append(ver_dir / "Points" / "stl" / f"stl{scan_id:03d}_total.ply")
                gt_ply_candidates.append(ver_dir / "Points" / "stl" / f"stl{scan_id}_total.ply")
                gt_ply_candidates.append(ver_dir / f"stl{scan_id:03d}_total.ply")
                gt_ply_candidates.append(ver_dir / f"stl{scan_id}_total.ply")
        # Dynamic search for any stl file matching this scan number
        if kr.exists():
            for cand in kr.glob(f"**/*stl*{scan_id}*.ply"):
                gt_ply_candidates.append(cand)
    gt_ply_candidates.extend([
        ws / "data" / "dtu_eval" / f"stl{scan_id:03d}_total.ply",
        ws / "archive" / "historical" / "all_notebooks_backup" / "Baseline" / "data" / "dtu_eval" / "Points" / "stl" / f"stl{scan_id:03d}_total.ply",
    ])
    gt_ply = next((p for p in gt_ply_candidates if p.exists()), None)
    if gt_ply is None:
        # Fallback recursive search across all kagglehub roots
        for root_cand in ["/root/.cache/kagglehub", str(Path.home() / ".cache" / "kagglehub"), "/notebooks", str(ws)]:
            rc = Path(root_cand)
            if rc.exists():
                for cand in rc.rglob(f"*stl*{scan_id}*.ply"):
                    if cand.is_file():
                        gt_ply = cand
                        break
            if gt_ply is not None:
                break

    gt_thin_candidates = [
        eval_dir / f"gt_stl_thin_{scan_name}.npy",
        ws / "data" / "dtu_eval" / f"gt_stl_thin_{scan_name}.npy",
    ]
    gt_thin_npy = next((p for p in gt_thin_candidates if p.exists()), None)

    obs_mask_candidates = [
        BUNDLED_CALIB_DIR / f"ObsMask{scan_id}_10.mat",
        eval_dir / f"ObsMask{scan_id}_10.mat",
    ]
    obs_mask = next((p for p in obs_mask_candidates if p.exists()), None)

    plane_candidates = [
        BUNDLED_CALIB_DIR / f"Plane{scan_id}.mat",
        eval_dir / f"Plane{scan_id}.mat",
    ]
    plane_mat = next((p for p in plane_candidates if p.exists()), None)

    # 6. Existing Baseline Meshes for Comparison
    baselines: Dict[str, Path] = {}
    base_root = ws / "archive" / "historical" / "all_notebooks_backup" / "Baseline" / "runs"
    baseline_patterns = {
        "PGSR": base_root / "PGSR" / scan_name / "mesh" / "tsdf_fusion_post.ply",
        "2DGS": base_root / "2DGS" / scan_name / "train" / "ours_30000" / "fuse_post.ply",
        "GOF": base_root / "GOF" / scan_name / "train" / "ours_30000" / "extracted_mesh_100_poisson.ply",
        "SuGaR": base_root / "SuGaR" / scan_name / "sugar_mesh.ply",
    }
    for name, p in baseline_patterns.items():
        if p.exists():
            baselines[name] = p

    # Previous SDF-GS Refine baseline mesh if available
    prev_candidates = [
        ws / "Improvement_3" / "results_full" / f"{scan_name}_improved_mesh.ply",
        ws / "archive" / "historical" / "all_notebooks_backup" / "experiment_output" / scan_name / "meshes" / "final_mesh.ply",
    ]
    for p in prev_candidates:
        if p.exists():
            baselines["Prev_SDF_GS"] = p
            break

    return {
        "ply_path": ply_path,
        "cameras_json": cameras_json,
        "cameras_npz": cameras_npz,
        "images_dir": images_dir,
        "gt_ply": gt_ply,
        "gt_thin_npy": gt_thin_npy,
        "obs_mask": obs_mask,
        "plane_mat": plane_mat,
        "baselines": baselines,
    }


def validate_scan_paths(
    paths: Dict[str, Optional[Path]],
    scan_id: int,
    require_ply: bool = True,
    require_gt: bool = False,
) -> None:
    """
    Validate resolved scan paths and raise an informative FileNotFoundError if any
    required input file is missing (never silently substituting synthetic data).
    """
    missing = []
    if require_ply and (paths.get("ply_path") is None or not Path(paths["ply_path"]).exists()):
        missing.append(
            f"  - Pretrained 3DGS PLY (`ply_path`) for scan{scan_id}. "
            f"On Paperspace, run `kagglehub.dataset_download('silvanusalvan/3dgs-trainingsdfgs')` "
            f"in 00_Environment_Setup.ipynb or set `paths.ply_path` in configs/scan{scan_id}.yaml."
        )
    if paths.get("cameras_json") is None or not Path(paths["cameras_json"]).exists():
        missing.append(f"  - Camera calibration `cameras.json` for scan{scan_id}.")
    if paths.get("cameras_npz") is None or not Path(paths["cameras_npz"]).exists():
        missing.append(f"  - Official DTU `cameras.npz` (`scale_mat_0`) for scan{scan_id}.")
    if require_gt and (
        (paths.get("gt_ply") is None or not Path(paths["gt_ply"]).exists())
        and (paths.get("gt_thin_npy") is None or not Path(paths["gt_thin_npy"]).exists())
    ):
        missing.append(
            f"  - Official DTU Ground Truth point cloud (`stl{scan_id:03d}_total.ply`) for scan{scan_id}. "
            f"On Paperspace, run `kagglehub.dataset_download('silvanusalvan/dtu-groundtruth')` "
            f"or set `paths.gt_ply` in configs/scan{scan_id}.yaml."
        )

    if missing:
        raise FileNotFoundError(
            f"Missing required files for DTU scan{scan_id}:\n" + "\n".join(missing)
        )


# --------------------------------------------------------------------------- #
# Pinhole Camera Model & Loaders
# --------------------------------------------------------------------------- #

class Camera:
    """A single calibrated pinhole camera: intrinsics K, world-to-cam [R|t]."""

    def __init__(self, K, R, t, H, W, img_name=None):
        self.K = np.asarray(K, dtype=np.float32)
        self.R = np.asarray(R, dtype=np.float32)
        self.t = np.asarray(t, dtype=np.float32)
        self.H, self.W = int(H), int(W)
        self.img_name = img_name
        self.center = (-self.R.T @ self.t).astype(np.float32)

    def rays(self, device, stride: int = 1):
        """Generate one ray per pixel (subsampled by stride). Returns (origins, dirs, (ys, xs))."""
        ys, xs = np.meshgrid(np.arange(0, self.H, stride), np.arange(0, self.W, stride), indexing="ij")
        pix = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs).ravel()], axis=1).astype(np.float32)
        Kinv = np.linalg.inv(self.K)
        cam_dirs = (Kinv @ pix.T).T
        world_dirs = (self.R.T @ cam_dirs.T).T
        world_dirs = world_dirs / np.linalg.norm(world_dirs, axis=1, keepdims=True)
        origins = np.tile(self.center[None, :], (world_dirs.shape[0], 1))
        return (torch.tensor(origins, device=device, dtype=torch.float32),
                torch.tensor(world_dirs, device=device, dtype=torch.float32),
                (ys.ravel(), xs.ravel()))


def rq3(A):
    """RQ decomposition of a 3x3 matrix (used to split K and R from P[:, :3])."""
    Q, R = np.linalg.qr(np.flipud(A).T)
    R = np.flipud(R.T)
    Q = Q.T
    R = np.fliplr(R)
    Q = np.flipud(Q)
    T = np.diag(np.sign(np.diag(R)))
    R = R @ T
    Q = T @ Q
    return R, Q


def decompose_projection(P):
    """Standard RQ-decomposition of a 3x4 camera matrix P = K[R|t]."""
    M = P[:, :3]
    K, R = rq3(M)
    t = np.linalg.inv(K) @ P[:, 3]
    K = K / K[2, 2]
    return K, R, t


def load_dtu_cameras(npz_path: str | Path, H: int = 512, W: int = 512) -> List[Camera]:
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


def load_cameras_auto(npz_path: str | Path, resolution: Tuple[int, int] = (512, 512)) -> Tuple[List[Camera], float]:
    """Automatically load cameras from npz and return (cameras, scale)."""
    H, W = resolution
    cams = load_dtu_cameras(npz_path, H, W)
    return cams, 1.0


def load_official_3dgs_cameras(json_path: str | Path) -> List[Camera]:
    """Parse 3DGS cameras.json into a list of Camera objects."""
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    cams = []
    for e in entries:
        W, H = int(e["width"]), int(e["height"])
        fx, fy = float(e["fx"]), float(e["fy"])
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        R_c2w = np.array(e["rotation"], dtype=np.float64)
        position = np.array(e["position"], dtype=np.float64)

        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ position

        cams.append(Camera(K, R_w2c, t_w2c, H, W, img_name=e.get("img_name")))
    return cams
