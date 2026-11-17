"""Configuration objects for the SDF-GS Refine pipeline.

All defaults follow the frozen protocol in ``docs/preregistration.md`` and the
thesis implementation details recovered in ``METHODOLOGY.md``. Any deviation must
be explicit in the run manifest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class PathConfig:
    """Paths. Relative paths are resolved against the project root."""
    data_root: str = "data"
    dtu_scan_root: str = "data/dtu_scan105"
    gaussian_ply: str = ""
    cameras_json: str = ""
    images_root: str = ""
    output_root: str = "outputs/sdf_gs_refine"
    dtu_assets_root: str = "data/dtu_eval_assets"
    graphdeco_root: str = "third_party/gaussian-splatting"
    scan_id: int = 105


@dataclass
class GaussianConfig:
    min_opacity: float = 0.05
    cull_sigma: float = 4.0
    max_candidates: int = 4096
    pixel_stride: int = 16
    n_cameras: int = 24
    chunk_rays: int = 256
    min_candidates: int = 1


@dataclass
class TransmittanceConfig:
    level_surface: float = 0.5
    level_free: float = 0.9
    free_margin_fraction: float = 0.05
    crossing_epsilon: float = 1e-8


@dataclass
class SamplingConfig:
    surface_batch: int = 8192
    uniform_batch: int = 4096
    near_batch: int = 4096
    free_batch: int = 8192
    jitter_sigma: float = 0.01
    sor_k: int = 16
    sor_std: float = 2.5
    component_radius_factor: float = 3.0
    component_min_fraction: float = 0.1
    sphere_radius: float = 0.8
    cache_samples: bool = True
    seed: int = 42


@dataclass
class SDFConfig:
    hidden_layers: int = 8
    hidden_width: int = 256
    softplus_beta: float = 100.0
    radius_init: float = 0.5
    iterations: int = 10_000
    lr: float = 5e-4
    warmup_fraction: float = 0.05
    cosine_min_ratio: float = 0.02
    grad_clip: float = 10.0
    surface_weight: float = 3.0
    normal_weight: float = 0.2
    eikonal_weight: float = 0.1
    stray_weight: float = 0.1
    free_weight: float = 1.0
    free_margin: float = 0.005
    surface_loss_norm: str = "l1"  # "l1" or "l2"
    log_every: int = 100
    checkpoint_every: int = 1000
    seed: int = 42


@dataclass
class JointConfig:
    iterations: int = 10_000
    sdf_lr: float = 1e-4
    position_lr: float = 1.6e-5      # multiplied by camera-derived scene extent
    opacity_lr: float = 5e-3
    sh_lr: float = 2.5e-4
    scale_lr: float = 5e-4
    rotation_lr: float = 1e-4
    warmup_fraction: float = 0.05
    cosine_min_ratio: float = 0.02
    grad_clip: float = 10.0
    patch_size: int = 32
    heldout_every: int = 8
    photometric_weight: float = 1.0
    coupling_epsilon: float = 0.02
    coupling_zero_weight: float = 0.1
    coupling_normal_weight: float = 0.05
    coupling_samples: int = 8192
    enabled: bool = False
    log_every: int = 100
    checkpoint_every: int = 1000
    seed: int = 42


@dataclass
class MeshConfig:
    resolution: int = 512
    bbox_margin: float = 0.10
    min_component_fraction: float = 0.01
    max_hole_edges: int = 100
    laplacian_iters: int = 3
    laplacian_lambda: float = 0.5
    batch_size: int = 262_144
    save_visual_mesh: bool = True
    seed: int = 42


@dataclass
class DTUConfig:
    enabled: bool = True
    # Official DTU names are ``ObsMask{scan}_10.mat`` and ``Plane{scan}.mat``.
    # dtu_eval.py also accepts zero-padded and older variants.
    mask_name_template: str = "ObsMask{scan}_10.mat"
    plane_name_template: str = "Plane{scan}.mat"
    downsample_mm: float = 0.2
    max_dist_mm: float = 20.0
    patch_mm: float = 60.0
    fscore_thresholds_mm: tuple = (1.0, 2.0, 5.0)
    density_points_per_mm2: float = 0.04
    min_points: int = 200_000
    max_points: int = 4_000_000
    seed: int = 42
    transform_mode: str = "auto"  # auto|scale_mat|calibration|none


@dataclass
class VizConfig:
    enabled: bool = True
    dpi: int = 130
    max_points: int = 60_000
    sdf_slice_points: int = 160
    seed: int = 42


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    transmittance: TransmittanceConfig = field(default_factory=TransmittanceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    sdf: SDFConfig = field(default_factory=SDFConfig)
    joint: JointConfig = field(default_factory=JointConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    dtu: DTUConfig = field(default_factory=DTUConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        if not isinstance(data, dict):
            raise TypeError("config must be a mapping")
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            value = data.get(f.name)
            if value is None:
                continue
            target = f.type
            if isinstance(target, str):
                target = _TYPE_MAP[target]
            if is_dataclass(target) and isinstance(value, dict):
                kwargs[f.name] = target(**value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def save(self, path: str | Path) -> None:
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    return PipelineConfig.load(path)


def save_config(cfg: PipelineConfig, path: str | Path) -> None:
    """Save a PipelineConfig to a YAML file."""
    cfg.save(path)


_TYPE_MAP = {
    "PathConfig": PathConfig,
    "GaussianConfig": GaussianConfig,
    "TransmittanceConfig": TransmittanceConfig,
    "SamplingConfig": SamplingConfig,
    "SDFConfig": SDFConfig,
    "JointConfig": JointConfig,
    "MeshConfig": MeshConfig,
    "DTUConfig": DTUConfig,
    "VizConfig": VizConfig,
}
