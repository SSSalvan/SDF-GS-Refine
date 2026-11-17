# SDF-GS Refine: Neural Signed Distance Field for High-Fidelity Mesh Extraction from 3D Gaussian Splatting

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DTU Benchmark](https://img.shields.io/badge/Benchmark-DTU%20MVS-blue.svg)](http://roboim.dtu.dk/data/mvs/)

> **Undergraduate Thesis Research Project**  
> *Author:* Silvanus Alvan (UPH Informatics / AI & Data Science)  
> *Focus:* Continuous Implicit Neural SDF Regularization for Watertight 3D Mesh Extraction from Pretrained 3D Gaussian Splatting (3DGS).

---

## Overview

**SDF-GS Refine** is a post-hoc implicit surface reconstruction framework designed to extract clean, watertight triangle meshes from pretrained 3D Gaussian Splatting models.

3DGS excels at novel view synthesis but lacks continuous surface topology, leading to discrete particle artifacts, depth curtains, and hollow shell defects when extracted using naive density thresholding. SDF-GS Refine resolves this by coupling:

1. **Ray-wise Attenuation Crossing & Multi-View Consistency (MVC) Filtering ($	au = 2.5\text{ mm}$)**: Eliminates specular outliers and floating particles across camera frustums.
2. **Gradient-Balanced Joint Optimization ($W_{\text{normal}} = 0.05$)**: Rebalances surface loss gradient allocation to 31%, preventing surface drift and eliminating micro-porosity.
3. **High-Resolution Slab-wise Marching Cubes ($480^3$) with Sub-voxel Newton-Raphson Projection**: Snaps mesh vertices onto the exact zero-level set $\mathbf{v}_{\text{new}} = \mathbf{v} - \frac{f(\mathbf{v})}{\|\nabla f(\mathbf{v})\|^2} \nabla f(\mathbf{v})$, achieving sub-millimeter median precision.

```
+------------------+     Ray Attenuation     +--------------------+
| Pretrained 3DGS  | ----------------------> | Multi-View Filter  |
| (COLMAP Cameras) |                         | (tau = 2.5 mm)     |
+------------------+                         +--------------------+
                                                       |
                                                       v
+------------------+     Joint Refinement    +--------------------+
| High-Res Mesh    | <---------------------- | Implicit SDF MLP   |
| (Newton-Raphson) |   (W_normal = 0.05)     | (Softplus, beta=100)
+------------------+                         +--------------------+
```

---

## Quantitative Benchmark Results (DTU MVS)

Evaluated under the official DTU evaluation protocol using camera-derived $\text{Sim}(3)$ transformation with **Zero Ground-Truth Leakage**:

| Scenario | Metric | Baseline (Initial Run) | **SDF-GS Refine (Ours)** | Relative Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **DTU Scan 24** *(Brick House)* | **Chamfer Distance (Mean)** | 3.07 mm | **1.561 mm** | **-49.2% Error Reduction** |
| | Chamfer Distance (Median) | — | **0.763 mm** | **Sub-millimeter Precision** |
| | F-Score @ 1 mm | 21.5% | **58.5%** | **+172.1% Gain** |
| | Topology Genus | 276 (Porous) | **36 (Watertight Solid)** | **-87% Cavity Elimination** |
| **DTU Scan 105** *(Organic Dome)*| **Chamfer Distance (Mean)** | 2.80 mm | **1.687 mm** | **-39.7% Error Reduction** |
| | Chamfer Distance (Median) | — | **0.801 mm** | **Sub-millimeter Precision** |
| | F-Score @ 1 mm | 26.1% | **56.3%** | **+115.7% Gain** |
| **DTU Scan 37** *(Thin Scissor)* | **Chamfer Distance (Mean)** | 5.97 mm | **2.842 mm** | **-52.4% Error Reduction** |
| | F-Score @ 1 mm | 5.8% | **41.9%** | **>7× Gain (+622%)** |
| | Thin Blade Structure | Fractured / Fused | **Cleanly Separated** | **Resolved Narrow Air Gaps** |
| **Overall Mean** | **Chamfer Distance** | **3.95 mm** | **2.030 mm** | **-48.6% Total Error Reduction** |
| | **F-Score @ 1 mm** | **17.8%** | **52.2%** | **Almost 3× Improvement** |
| | **MVC Surface Cloud** | — | **0.848 mm** | **Global Sub-millimeter** |

*Detailed per-scan logs, JSON summaries, and loss curves are located in `results/`.*

---

## ⚙️ Installation & Setup

### Requirements
* Linux / Windows (WSL2 recommended)
* NVIDIA GPU with CUDA support (tested on RTX A6000 48GB and RTX 3090/4090)
* Python 3.9+ / PyTorch 2.0+

```bash
git clone https://github.com/SSSalvan/SDF-GS-Refine.git
cd SDF-GS-Refine
pip install -r requirements.txt
```

---

## Quickstart

### 1. Interactive Notebook Pipeline
Launch the self-contained, end-to-end Jupyter Notebook:
```bash
jupyter notebook SDF_GS_Refine_Pipeline.ipynb
```

### 2. Command Line Execution
Run the full 2-stage pipeline on a specific DTU scan:
```bash
python pipeline.py --scan 24 --stage1_iters 3500 --stage2_iters 1000 --grid_res 480
```

Arguments:
* `--scan`: Scan ID (`24`, `37`, or `105`).
* `--stage1_iters`: Number of iterations for implicit SDF fitting (default: 3500).
* `--stage2_iters`: Number of joint refinement iterations (default: 1000).
* `--grid_res`: Resolution of Marching Cubes grid (default: 480).
* `--w_normal`: Weight of normal consistency loss (default: 0.05).
* `--mvc_thresh`: Multi-view consistency distance threshold $\tau$ in mm (default: 2.5).

---

## Repository Structure

```
.
├── SDF_GS_Refine_Pipeline.ipynb  # End-to-end interactive execution notebook
├── pipeline.py                   # Canonical pipeline CLI runner
├── requirements.txt              # Environment dependencies
├── README.md                     # Project overview and benchmarks
├── src/                          # Core modular engine
│   ├── cameras.py                # Camera projection and ray generation
│   ├── gaussian.py               # 3D Gaussian Splatting parser and utilities
│   ├── sdf.py                    # Implicit MLP architecture with Softplus activation
│   ├── losses.py                 # Multi-objective losses (surface, eikonal, normal, NCC)
│   ├── mesh.py                   # Slab-wise Marching Cubes and Newton-Raphson snapping
│   ├── sampling.py               # Attenuation crossing and ray sampling
│   ├── transmittance.py          # Transmittance computation along camera rays
│   ├── dtu_eval.py               # Clean camera Sim(3) DTU evaluation protocol
│   └── evaluation.py             # Geometric accuracy, completeness, and F-score
└── results/                      # Latest raw evaluation logs and metric tables
    ├── summary_all_scans.json    # Complete multi-scan quantitative summary
    ├── dtu_evaluation_scan24.json# Raw evaluation outputs for Scan 24
    ├── dtu_evaluation_scan37.json# Raw evaluation outputs for Scan 37
    ├── dtu_evaluation_scan105.json# Raw evaluation outputs for Scan 105
    ├── benchmark_metrics_dtu.csv # Benchmark metrics in CSV format
    └── training_loss_logs/       # Detailed iteration-by-iteration loss logs
```

---

## Citation & Academic Use

If you find this work helpful for your research, please cite:
```bibtex
@thesis{alvan2026sdfgs,
  title={Signed Distance Field untuk Mesh Extraction dari 3D Gaussian Splat},
  author={Alvan, Silvanus},
  school={Universitas Pelita Harapan},
  year={2026}
}
```
