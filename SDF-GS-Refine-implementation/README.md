# SDF-GS Refine — Implementation

Real implementation of the methodology in *"Signed Distance Field untuk Mesh
Extraction dari 3D Gaussian Splat"* (Silvanus Alvan). Not pseudo-code — every
equation in Bab III (Persamaan 3.1–3.3) has a corresponding, executed
implementation.

## Run it

Open `implementation.ipynb` and run all cells top to bottom. It already
contains one full executed run (outputs embedded) so you can inspect results
without re-running anything.

## Zero-pathing option: auto-download from Kaggle

If you have Kaggle credentials available in whatever environment you run
this notebook (your machine, a Kaggle Notebook, Colab, etc.), **Section 3
now tries to auto-download your two published datasets first**, before
looking at any local folder:

- `silvanusalvan/3dgs-trainingsdfgs` (your trained 3DGS output)
- `silvanusalvan/dtu-groundtruth` (GT meshes)

via `kagglehub.dataset_download(...)`. No manual path setup needed if this
succeeds — the discovered Kaggle cache directory is used directly for
point-cloud / camera / image / GT-mesh discovery.

**How to give it credentials** (any ONE of these, before running the notebook):
```python
# Option 1: in a cell before Section 3
import kagglehub
kagglehub.login()   # interactive prompt

# Option 2: environment variables (e.g. in your shell, or os.environ in a cell)
import os
os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"] = "your_api_key"   # from kaggle.com/settings/api -> Create New Token

# Option 3: place the token file yourself
# ~/.kaggle/kaggle.json  containing {"username": "...", "key": "..."}
```
If running this literally as a **Kaggle Notebook** with both datasets
attached as inputs, kagglehub/the Kaggle environment handles auth
automatically — no setup needed at all.

**I could not test this download myself** — this sandbox's network only
allows a fixed allowlist (pypi/github/npm/etc.), and `kaggle.com` is not on
it (confirmed: the request reaches my sandbox's egress proxy and gets
blocked with `x-deny-reason: host_not_allowed`, not a real Kaggle response).
The download code is written directly against `kagglehub`'s real installed
API (verified via `inspect.signature`), and the notebook's fallback chain
(Kaggle → local folders → synthetic) is exercised and confirmed working
end-to-end here using the local/synthetic branches — but the Kaggle branch
itself needs to be verified by you, in an environment with real Kaggle
access. If it errors, paste me the exact error and I'll fix it.

If Kaggle download fails or isn't configured, discovery automatically falls
back to the local-folder layout below.

## Using local folders (no Kaggle)

This now works for real data trained with the **official
`graphdeco-inria/gaussian-splatting` repo**. Drop your files in like this:

```
3DGS_Pretrained_SDFGS/
└── scan24/                              <- any name, auto-discovered
    ├── point_cloud/
    │   └── iteration_30000/
    │       └── point_cloud.ply          <- your trained 3DGS
    ├── cameras.json                     <- official 3DGS training output
    └── images/                          <- your original training images
        ├── img_0001.jpg
        └── ...
DTU_GT/
    └── scan24_stl_total.ply             <- ground-truth mesh (e.g. from your
                                             Kaggle DTU-groundtruth dataset)
```

Then just **Run All**. Section 6 will report
`Loaded N REAL cameras from official 3DGS export` instead of falling back to
synthetic, and immediately runs `sanity_check_cameras` — if fewer than 80% of
your cameras don't point at the scene, the notebook **stops with an
assertion** rather than silently training on broken calibration (this is
usually a sign the images/cameras.json don't match the point cloud, not a
bug in the parser — the R/T convention was verified line-by-line against the
official repo's actual source on GitHub, see `src/camera.py` docstrings).

Section 9 then automatically loads your real images (matched to each camera
via `img_name`) for the photometric loss, and Section 13 loads your real GT
mesh from `DTU_GT/` for Chamfer Distance / F-score — no more synthetic
fallback once real files are present.

**Note on scan24/37/40 or whichever scans you trained:** the notebook
processes ONE scan per run — whichever one it discovers first under
`3DGS_Pretrained_SDFGS/`. If you have all three, either run the notebook
three times (moving/renaming folders), or say the word and I'll add a
multi-scan loop that reproduces the paper's per-scan + averaged table
(Bab III, Tabel 3.2) directly.

**Compute note:** this sandbox has 1 CPU core, no GPU, ~4GB RAM. The paper's
full settings (10,000 refinement iterations, 512³ Marching Cubes grid,
1024 rays/batch) are configured in `configs/config.json` but the notebook
currently runs the *smoke-test* reduced settings
(`n_iterations_smoke_test`, `grid_resolution_smoke_test`, etc.) for
tractability here. If you run this notebook on your own machine with a GPU,
bump those to the `_paper` values in the config for a faithful full run.

## IMPORTANT — what the executed copy in this notebook actually ran on

**Only your thesis document was uploaded with this request.** No pretrained
`point_cloud.ply`, no DTU multi-view images, no camera calibration file, no
DTU ground-truth mesh. Per your instructions (Section 5, "DATASET REALITY")
I did not fabricate any of these. Instead:

- The notebook auto-discovers a real point cloud under
  `3DGS_Pretrained_SDFGS/<scan>/point_cloud/iteration_XXXXX/point_cloud.ply`
  and real DTU cameras under `DTU/<scan>/cameras_sphere.npz` — **if you drop
  your real files into those folders, the notebook will use them
  automatically, no code changes needed.**
- Because neither was present, it fell back to a small, clearly-labeled
  **synthetic** Gaussian-splat scene (`src/synthetic.py`): 4000 flat
  Gaussians arranged on a unit sphere, with a known analytic ground-truth
  mesh. This lets every stage of the *real* pipeline code be exercised and
  checked against a verifiable answer (does the transmittance-crossing
  estimator recover a sphere of the correct radius? does Chamfer distance
  actually improve during optimization? etc.) — but it is **not** a DTU
  scan24/37/40 result, and the numbers in `outputs/results.json` should not
  be read as reproducing the paper's target metrics.
- The photometric loss target is likewise a frozen render of the
  pre-refinement Gaussian scene (documented in the notebook, Section 9) —
  not real DTU photographs.

## What's implemented for real

| File | Implements |
|---|---|
| `src/gaussian.py` | Loading a real 3DGS `point_cloud.ply` (official schema), covariance `Sigma_i = R_i S_i S_i^T R_i^T`, full 3D Gaussian density, opacity filtering, validation |
| `src/camera.py` | DTU/IDR-style camera parsing (`cameras_sphere.npz`), ray generation, synthetic fallback camera ring |
| `src/transmittance.py` | **Core contribution**: discrete transmittance (Eq. 3.2) and the `T(t*)=0.5` crossing-point estimator (Eq. 3.3), with both paper-specified mitigations |
| `src/sdf.py` | Neural SDF, 8×256 MLP, Softplus(β=100), IGR geometric initialization, autograd Eikonal gradient |
| `src/losses.py` | Eikonal loss, normal-alignment + zero-boundary loss, naive differentiable alpha-compositing renderer for the photometric term |
| `src/mesh.py` | Marching Cubes at isovalue 0, connected-component filtering, hole filling, optional Laplacian smoothing (metrics computed pre-smoothing, exactly as specified) |
| `src/evaluation.py` | Chamfer Distance (bidirectional), F-score, PSNR, SSIM |
| `src/synthetic.py` | The clearly-labeled fallback scene described above |

## What's NOT implemented, and why

- **LPIPS**: needs offline pretrained VGG weights; this sandbox's network
  allowlist has no weights host (only pypi/npm/github code mirrors). Works
  fine on your own machine if you install `lpips` — not wired in here.
- **The official CUDA tile-based 3DGS rasterizer**: replaced by a
  mathematically-identical but unoptimized per-ray renderer, since no GPU is
  available in this sandbox. This is an implementation detail (same alpha-
  compositing equation), not a methodology change — see `src/losses.py`.
  On real DTU-resolution images (1600x1200) with hundreds of thousands of
  Gaussians, this naive renderer will be slow; consider swapping in the
  official `diff-gaussian-rasterization` CUDA extension if you have a GPU.
- **Multi-scan batch run** (scan24 + scan37 + scan40 in one go, matching
  paper Table 3.2's per-scan + average report): not built yet, single-scan
  per run for now — ask if you want this added.

## Where I disagreed with / need to flag the paper

See **Section 16 (Technical Review)** at the end of `implementation.ipynb`
for the full component-by-component comparison table and the "Recommended
Paper Revisions" list — in short:

1. Table 3.3 in your document is mostly blank / has values that look
   misplaced from Table 3.2.
2. Eq. 3.1 doesn't assign an explicit weight to the `f_theta(x_s)≈0`
   boundary term mentioned in prose (Subbab 3.6) — I folded it into
   `L_normal` and flagged this rather than inventing a new weight silently.
3. The paper asserts Gaussian surface normals are "available" (Subbab 2.7)
   but never states the convention; I used the standard shortest-scale-axis
   proxy (as in 2DGS) and flagged it.

## To run on your real data (recap)

1. Copy your trained model to
   `3DGS_Pretrained_SDFGS/<scan_name>/point_cloud/iteration_30000/point_cloud.ply`
2. Copy `cameras.json` (from the official 3DGS training output folder) next
   to it, and your training `images/` folder alongside.
3. Download your GT mesh from your Kaggle dataset and place it under
   `DTU_GT/<scan_name>_stl_total.ply` (or any filename containing "stl" or
   the scan name — the discovery glob is not strict about exact naming).
4. Run All. Watch the Section 6 camera sanity-check output first — if it's
   not ~100%, stop and tell me what it printed before letting training run
   further.
