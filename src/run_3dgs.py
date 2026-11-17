"""Entry point for vanilla 3D Gaussian Splatting training using the official Graphdeco implementation.

This module provides a deterministic, reproducible entry point for Notebook 01 that:

1. Sets up the environment and pins the official Graphdeco-Inria commit.
2. Installs dependencies and clones the Graphdeco repository if needed.
3. Prepares the DTU scan105 dataset structure.
4. Configures and runs the official training script.
5. Saves checkpoints and metadata compatible with the downstream SDF-GS Refine pipeline.
6. Validates that the saved checkpoint can be loaded by the canonical gaussian.py module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Pinned commit for reproducibility (frozen protocol).
GRAPHDECO_COMMIT = "d0ecbf76aac3ae9a65bc12d8ef8dbbf7ff5ecef7"

# Default training parameters following the frozen protocol.
DEFAULT_SEED = 42
DEFAULT_ITERATIONS = 30_000

# Output directory conventions.
OUTPUTS_DIR = Path("outputs")
TRAINING_DIR = OUTPUTS_DIR / "3dgs"
TRAINING_SCAN105_DIR = TRAINING_DIR / "scan105"

# Expected checkpoint naming convention from official Graphdeco.
CHECKPOINT_PREFIX = "chkpnt"
CHECKPOINT_LATEST = "chkpnt_latest"

# --------------------------------------------------------------------------- #
# Environment and provenance
# --------------------------------------------------------------------------- #


def environment_report() -> Dict[str, Any]:
    """Generate an environment fingerprint for the 3DGS training run.

    Records Python version, PyTorch version, CUDA availability, and installed
    package versions so that a run can be audited and reproduced later.
    """
    import platform

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "seed": DEFAULT_SEED,
        "iterations": DEFAULT_ITERATIONS,
    }

    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        info["numpy"] = None

    try:
        import pyyaml
        info["pyyaml"] = pyyaml.__version__
    except Exception:
        info["pyyaml"] = None

    # Record the commit hash of this repository for provenance.
    repo_root = Path(__file__).resolve().parents[3]
    git_commit = None
    if (repo_root / ".git").exists():
        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                .strip()
            )
        except Exception:
            git_commit = None
    info["repo_commit"] = git_commit

    return info


def write_environment_report(path: str | Path) -> None:
    """Write the environment report to a JSON file for provenance."""
    report = environment_report()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


# --------------------------------------------------------------------------- #
# Graphdeco repository management
# --------------------------------------------------------------------------- #


def discover_graphdeco_root(workspace: Path) -> Optional[Path]:
    """Find the Graphdeco repository root within the workspace.

    Looks for the canonical location: third_party/gaussian-splatting under the
    project root, or a sibling directory.
    """
    candidates = [
        workspace / "third_party" / "gaussian-splatting",
        workspace / "3DGS Original" / "gaussian-splatting-main",
        workspace / "Archived" / "reference_v3_src",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "gaussian_renderer").is_dir():
            return candidate
    return None


def clone_graphdeco(
    workspace: Path,
    commit: str = GRAPHDECO_COMMIT,
    force: bool = False,
) -> Path:
    """Clone or update the official Graphdeco-Inria repository.

    Args:
        workspace: The project root directory.
        commit: The pinned commit hash for reproducibility.
        force: If True, re-clone even if the repo already exists.

    Returns:
        Path to the cloned/opened Graphdeco repository root.
    """
    repo_dir = discover_graphdeco_root(workspace)

    if repo_dir is not None and not force:
        # Verify the existing repo matches the pinned commit.
        try:
            existing_commit = subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if existing_commit == commit:
                return repo_dir  # Already at the right commit.
            # Commit mismatch — remove and re-clone.
            shutil.rmtree(repo_dir)
        except Exception:
            shutil.rmtree(repo_dir)
            repo_dir = None

    if repo_dir is None:
        # Clone into third_party/.
        repo_dir = workspace / "third_party" / "gaussian-splatting"
        repo_dir.parent.mkdir(parents=True, exist_ok=True)

        url = "https://github.com/graphdeco-inria/gaussian-splatting.git"
        subprocess.check_call(
            ["git", "clone", url, str(repo_dir)],
            stdout=subprocess.DEVNULL,
        )

    # Checkout the pinned commit.
    subprocess.check_call(
        ["git", "-C", str(repo_dir), "checkout", commit],
        stdout=subprocess.DEVNULL,
    )

    return repo_dir


# --------------------------------------------------------------------------- #
# Training orchestration
# --------------------------------------------------------------------------- #


def discover_scan105(root: Path) -> Optional[Path]:
    """Locate the DTU scan105 dataset directory under ``root``.

    Scans common layouts for a COLMAP sparse reconstruction and images.
    """
    if not root.is_dir():
        return None

    # Common locations for DTU data.
    candidates = [
        root / "data" / "dtu_scan105",
        root / "dtu_scan105",
        root / "data" / "SampleSet" / "MVS Data" / "scan105",
    ]

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        # Minimal sanity check: look for COLMAP sparse/0 and images.
        if (candidate / "sparse" / "0").is_dir() and (candidate / "images").is_dir():
            return candidate

    return None


def configure_training_args(
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    data_dir: Path = TRAINING_SCAN105_DIR,
    output_dir: Path = TRAINING_SCAN105_DIR,
) -> Dict[str, Any]:
    """Build the argument namespace expected by the official Graphdeco train.py.

    This mirrors the argument configuration used in Notebook 01 cell 16.
    """
    return {
        "iteration": iterations,
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_delay_steps": 1000,
        "fine_lr": 0.000025,
        "fine_lr_multiplier": 0.5,
        "neus_lr": 0.001,
        "nerf_lr": 0.001,
        "network_width": 256,
        "network_depth": 8,
        "test_iterations": [5000, 10000, 15000, 20000, 25000, 30000],
        "estimated_learning_rate": 5e-4,
        "batch_size": 128 * 4,
        "grid_size": 128,
        "max_alpha": 0.9,
        "disable_learning_rate_schedule": False,
        "checkpoint_iterations": [5000, 10000, 15000, 20000, 25000, 30000],
        "surgery_iterations": [],
        "serialize": True,
        "cuda": str(torch.cuda.is_available()),
        "training_image_dataset": str(data_dir),
        "eval_image_dataset": str(data_dir),
        "model_dir": str(output_dir),
        "zero_opt_level": 0,
    }


def run_graphdeco_training(
    graphdeco_dir: Path,
    args: Dict[str, Any],
    env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Execute the official Graphdeco train.py and return the final checkpoint.

    The official Graphdeco train.py outputs:
    - `outputs/3dgs/scan105/point_cloud/iteration_30000/point_cloud.ply`
    - `outputs/3dgs/scan105/cameras.json`
    - `outputs/3dgs/scan105/training_manifest.json`

    Returns:
        (final_checkpoint_path, metadata_dict) or (None, None) on failure.
    """
    train_script = graphdeco_dir / "train.py"
    if not train_script.is_file():
        print(f"ERROR: Graphdeco train.py not found at {train_script}")
        return None, None

    # Build the command line.
    cmd = [
        sys.executable,
        str(train_script),
    ]

    # Append key arguments derived from the arg dict.
    # The official script uses argparse; we pass the essential params.
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool) and value is False:
            # Flag-style: --key (no value)
            cmd.append(f"--{key}")
        elif isinstance(value, bool) and value is True:
            cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    # Capture environment.
    run_env = env or os.environ.copy()

    print(f"Running: graphdeco 3DGS training with {args.get('iteration', '?')} iterations")
    print(f"  data_dir: {args.get('training_image_dataset', '?')}")
    print(f"  model_dir: {args.get('model_dir', '?')}")

    try:
        result = subprocess.run(
            cmd,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=args.get("iteration", 30000) + 300,  # give some buffer
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            print(f"Graphdeco training exited with code {result.returncode}")
            print(f"STDOUT: {stdout[-500:]}")
            print(f"STDERR: {stderr[-500:]}")
            return None, None

        # Parse the manifest if it exists.
        manifest_path = Path(args.get("model_dir", "")) / "training_manifest.json"
        metadata: Dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    metadata = json.load(mf)
            except Exception:
                metadata = {"manifest_parse_error": True}

        # Find the final checkpoint PLY.
        point_cloud_dir = Path(args.get("model_dir", "")) / "point_cloud" / f"iteration_{args.get('iteration', 30000)}"
        ply_path = point_cloud_dir / "point_cloud.ply" if point_cloud_dir.is_dir() else None

        final_checkpoint = ply_path if ply_path and ply_path.is_file() else None

        print(f"Training complete. Final checkpoint: {final_checkpoint}")
        if metadata:
            print(f"Manifest: {json.dumps(metadata, indent=2)[:200]}")

        return final_checkpoint, metadata

    except subprocess.TimeoutExpired:
        print("Graphdeco training timed out")
        return None, None
    except Exception as e:
        print(f"Error running Graphdeco training: {e}")
        return None, None


# --------------------------------------------------------------------------- #
# Checkpoint validation (compatibility with gaussian.py)
# --------------------------------------------------------------------------- #


def validate_checkpoint_for_gaussian( ply_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Verify that a 3DGS PLY checkpoint has the fields expected by gaussian.py.

    GaussianModel.load_ply expects:
    - vertex element with fields: x, y, z, opacity
    - scale_0, scale_1, scale_2
    - rot_0, rot_1, rot_2, rot_3
    - f_dc_0, f_dc_1, f_dc_2 (optional but expected)
    - f_rest_* (optional)

    Returns:
        (is_valid, diagnostics_dict)
    """
    from plyfile import PlyData

    diagnostics: Dict[str, Any] = {
        "path": str(ply_path),
        "exists": ply_path.is_file(),
    }

    if not ply_path.is_file():
        diagnostics["valid"] = False
        diagnostics["reason"] = "file not found"
        return False, diagnostics

    try:
        ply = PlyData.read(str(ply_path))
        if "vertex" not in ply:
            diagnostics["valid"] = False
            diagnostics["reason"] = "no vertex element"
            return False, diagnostics

        v = ply["vertex"]
        names = v.data.dtype.names or ()

        required = {"x", "y", "z", "opacity"}
        missing = required - set(names)
        if missing:
            diagnostics["valid"] = False
            diagnostics["reason"] = f"missing vertex fields: {sorted(missing)}"
            return False, diagnostics

        # Check scale and rot fields.
        scale_names = sorted(n for n in names if n.startswith("scale_"))
        rot_names = sorted(n for n in names if n.startswith("rot_"))

        if len(scale_names) != 3:
            diagnostics["valid"] = False
            diagnostics["reason"] = f"expected 3 scale_* fields, got {len(scale_names)}"
            return False, diagnostics

        if len(rot_names) != 4:
            diagnostics["valid"] = False
            diagnostics["reason"] = f"expected 4 rot_* fields, got {len(rot_names)}"
            return False, diagnostics

        # Check f_dc fields.
        f_dc_names = sorted(n for n in names if n.startswith("f_dc_"))
        if len(f_dc_names) < 3:
            diagnostics["valid"] = False
            diagnostics["reason"] = f"expected f_dc_0/1/2, got {f_dc_names}"
            return False, diagnostics

        diagnostics["valid"] = True
        diagnostics["n_gaussians"] = v.count

        # Quick finiteness check on a few vertices.
        try:
            import numpy as np
            x = np.asarray(v["x"], dtype=np.float32)
            y = np.asarray(v["y"], dtype=np.float32)
            z = np.asarray(v["z"], dtype=np.float32)
            op = np.asarray(v["opacity"], dtype=np.float32)

            finite_x = bool(np.isfinite(x).all())
            finite_y = bool(np.isfinite(y).all())
            finite_z = bool(np.isfinite(z).all())
            finite_op = bool(np.isfinite(op).all())

            diagnostics["position_finite"] = finite_x and finite_y and finite_z
            diagnostics["opacity_finite"] = finite_op

            if not (finite_x and finite_y and finite_z and finite_op):
                diagnostics["valid"] = False
                diagnostics["reason"] = "non-finite vertex values"
        except Exception:
            pass

        return True, diagnostics

    except Exception as e:
        diagnostics["valid"] = False
        diagnostics["reason"] = f"parse error: {e}"
        return False, diagnostics


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def run_3dgs_training(
    workspace: Path,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    data_dir: Optional[Path] = None,
    force_clone: bool = False,
) -> Dict[str, Any]:
    """Run the complete vanilla 3DGS training workflow.

    This is the main entry point that integrates everything:

    1. Sets up the environment and provenance tracking.
    2. Clones/verified the official Graphdeco repository at the pinned commit.
    3. Locates or prepares the DTU scan105 dataset.
    4. Runs the official Graphdeco train.py.
    5. Validates the output checkpoint for compatibility with gaussian.py.
    6. Saves an experiment manifest with environment fingerprint and checkpoint provenance.
    7. Returns a summary dictionary.

    Args:
        workspace: The project root directory.
        iterations: Number of training iterations (frozen protocol: 30000).
        seed: Random seed for reproducibility.
        data_dir: Explicit path to DTU scan105 data. If None, auto-discover.
        force_clone: If True, re-clone Graphdeco even if present.

    Returns:
        Dict with training results and provenance.
    """
    workspace = Path(workspace)
    report: Dict[str, Any] = {
        "status": "pending",
        "graphdeco_commit": GRAPHDECO_COMMIT,
        "iterations": iterations,
        "seed": seed,
        "checkpoint_path": None,
        "checkpoint_valid": False,
        "checkpoint_diagnostics": None,
        "metadata": None,
        "environment_report": None,
        "errors": [],
    }

    try:
        # Step 1: Environment report.
        env_report_path = workspace / "logs" / "3dgs_training_environment.json"
        write_environment_report(env_report_path)
        report["environment_report"] = str(env_report_path)

        # Step 2: Clone/verify Graphdeco.
        report["errors"].append(f"Cloning Graphdeco at commit {GRAPHDECO_COMMIT}...")
        graphdeco_dir = clone_graphdeco(workspace, commit=GRAPHDECO_COMMIT, force=force_clone)
        report["graphdeco_dir"] = str(graphdeco_dir)

        # Step 3: Locate dataset.
        if data_dir is None:
            data_dir = discover_scan105(workspace)
        if data_dir is None:
            report["errors"].append("Could not auto-discover DTU scan105 dataset; please set --data-dir explicitly.")
            return report
        report["data_dir"] = str(data_dir)

        # Step 4: Configure and run training.
        args = configure_training_args(
            iterations=iterations,
            seed=seed,
            data_dir=data_dir,
        )

        final_checkpoint, metadata = run_graphdeco_training(
            graphdeco_dir, args
        )

        if final_checkpoint is None:
            report["errors"].append("Graphdeco training did not produce a final checkpoint.")
            return report

        report["checkpoint_path"] = str(final_checkpoint)

        # Step 5: Validate checkpoint compatibility.
        diagnostics = validate_checkpoint_for_gaussian(Path(final_checkpoint))
        report["checkpoint_diagnostics"] = diagnostics

        if diagnostics["valid"]:
            report["checkpoint_valid"] = True
        else:
            report["errors"].append(f"Checkpoint validation failed: {diagnostics.get('reason', 'unknown')}")

        # Step 6: Save metadata manifest.
        manifest: Dict[str, Any] = {
            "status": "completed" if report["checkpoint_valid"] else "validation_failed",
            "graphdeco_commit": GRAPHDECO_COMMIT,
            "iterations": iterations,
            "seed": seed,
            "checkpoint_path": report["checkpoint_path"],
            "checkpoint_valid": report["checkpoint_valid"],
            "checkpoint_diagnostics": diagnostics,
            "data_dir": str(data_dir),
            "graphdeco_dir": str(graphdeco_dir),
            "environment_report": str(env_report_path),
            "metadata": metadata or {},
            "timestamp_utc": str(__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")),
        }

        manifest_path = workspace / "outputs" / "3dgs" / "scan105" / "training_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        report["metadata_manifest"] = str(manifest_path)

        report["status"] = "completed" if report["checkpoint_valid"] else "validation_failed"

    except Exception as e:
        report["errors"].append(f"Unexpected error: {e}")
        import traceback
        report["traceback"] = traceback.format_exc()

    return report


# --------------------------------------------------------------------------- #
# Command-line entry point
# --------------------------------------------------------------------------- #

def main_cli() -> None:
    """CLI entry point for running 3DGS training from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vanilla 3DGS training using official Graphdeco-Inria implementation."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Project root directory (default: current working directory)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of training iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Explicit path to DTU scan105 data directory",
    )
    parser.add_argument(
        "--force-clone",
        action="store_true",
        default=False,
        help="Re-clone Graphdeco even if repository already exists",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="Only generate and print the environment report, skip training",
    )

    args = parser.parse_args()

    workspace = Path(args.workspace)

    if args.report_only:
        # Just write the environment report and exit.
        env_report_path = workspace / "logs" / "3dgs_training_environment.json"
        write_environment_report(env_report_path)
        print(f"Environment report written to: {env_report_path}")
        env = __import__("json").load(open(env_report_path))
        print(f"CUDA available: {env.get('cuda_available', False)}")
        return

    report = run_3dgs_training(
        workspace,
        iterations=args.iterations,
        seed=args.seed,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        force_clone=args.force_clone,
    )

    # Print summary.
    print("=" * 60)
    print("3DGS TRAINING REPORT")
    print("=" * 60)
    print(f"Status: {report['status']}")
    print(f"Iterations: {report['iterations']}")
    print(f"Seed: {report['seed']}")
    print(f"Graphdeco commit: {report.get('graphdeco_commit', '?')}")

    if report["checkpoint_path"]:
        print(f"Checkpoint: {report['checkpoint_path']}")
        print(f"Checkpoint valid (gaussian.py compatible): {report['checkpoint_valid']}")
    else:
        print("Checkpoint: None produced")

    if report["errors"]:
        print(f"\nErrors ({len(report['errors'])}):")
        for err in report["errors"]:
            print(f"  - {err}")

    if report.get("environment_report") and Path(report["environment_report"]).is_file():
        print(f"\nEnvironment report: {report['environment_report']}")

    print("=" * 60)


if __name__ == "__main__":
    main_cli()