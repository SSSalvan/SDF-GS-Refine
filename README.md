# SDF-GS Refine

Implementation of *Signed Distance Field untuk Mesh Extraction dari 3D Gaussian Splat* (Silvanus Alvan). Extracts high-fidelity meshes from 3D Gaussian Splatting scenes via neural SDF optimization guided by transmittance-based surface estimation.

## Project Structure

```
Generated/
├── src/                          # Core library (used by notebooks)
├── SDF-GS-Refine-implementation/ # Standalone implementation with configs
│   ├── src/                      # Implementation source
│   ├── configs/                  # Run configurations
│   ├── 3DGS_Pretrained_SDFGS/    # Pretrained Gaussian models (via Git LFS)
│   └── run_artifacts/            # Generated results (via Git LFS)
├── run_artifacts/                # Root-level run outputs (via Git LFS)
├── outputs/                      # Loss/gradient plots
└── *.ipynb                       # Jupyter notebooks (pipeline + analysis)
```

## Setup

```bash
# Clone with LFS (required for model weights and artifacts)
git lfs install
git clone <repo-url>
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Open3D (for mesh extraction)
- Jupyter

## Quick Start

Open any notebook and run all cells. Notebooks contain embedded outputs for preview without re-execution.

For the full implementation pipeline, see `SDF-GS-Refine-implementation/README.md`.

## Running on Real Data

1. Place trained 3DGS model at `3DGS_Pretrained_SDFGS/<scan>/point_cloud/iteration_30000/point_cloud.ply`
2. Add `cameras.json` and `images/` from official 3DGS training output
3. Add ground-truth mesh under `DTU_GT/`
4. Run the implementation notebook

## Note

- Large files (`.pt`, `.ply`, `.npy`) are tracked via **Git LFS**
- **Never commit** `credentials.json`, `kaggle.json`, or `token.json`
