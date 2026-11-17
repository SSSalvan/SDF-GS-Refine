"""3D Gaussian Splatting primitive loading and geometry.

The PLY schema is the official graphdeco-inria one:

* ``x, y, z``              : mean ``mu``
* ``opacity``              : raw logit, ``alpha = sigmoid(opacity)``
* ``scale_0..2``           : raw log-scale, ``s = exp(scale)``
* ``rot_0..3``             : raw quaternion ``(w, x, y, z)``
* ``f_dc_0..2``            : SH DC coefficients, base colour
* ``f_rest_0..``           : higher-order SH (kept for completeness)

``Sigma = R S S^T R^T`` and the full 3D kernel is
``G(x) = exp(-0.5 (x-mu)^T Sigma^-1 (x-mu))``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

SH_C0 = 0.28209479177387814


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Normalized quaternion ``(w, x, y, z)`` -> rotation matrix.

    Accepts ``(4,)`` or ``(N, 4)`` and returns ``(3, 3)`` or ``(N, 3, 3)``.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must have last dim 4, got {tuple(q.shape)}")
    squeeze = q.dim() == 1
    if squeeze:
        q = q.unsqueeze(0)
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    R = torch.empty((n, 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R.squeeze(0) if squeeze else R


def _sorted_fields(names, prefix: str):
    return sorted([n for n in names if n.startswith(prefix)],
                  key=lambda s: int(s.split("_")[-1]))


class GaussianModel:
    """Container for Gaussian parameters.

    Raw parameters are stored (logit opacity, log scale, unnormalized quaternion)
    so that the same object can be used for fixed inference or for optimization.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        opacity_logit: torch.Tensor,
        log_scales: torch.Tensor,
        quats: torch.Tensor,
        sh_dc: torch.Tensor,
        sh_rest: Optional[torch.Tensor] = None,
        source_path: Optional[str] = None,
    ):
        n = positions.shape[0]
        if positions.shape != (n, 3):
            raise ValueError("positions must be (N,3)")
        if opacity_logit.shape != (n,):
            raise ValueError("opacity_logit must be (N,)")
        if log_scales.shape != (n, 3):
            raise ValueError("log_scales must be (N,3)")
        if quats.shape != (n, 4):
            raise ValueError("quats must be (N,4)")
        if sh_dc.shape != (n, 3):
            raise ValueError("sh_dc must be (N,3)")
        self.positions = positions
        self.opacity_logit = opacity_logit
        self.log_scales = log_scales
        self.quats = quats
        self.sh_dc = sh_dc
        self.sh_rest = sh_rest
        self.source_path = source_path
        self._cov_cache = None

    # ------------------------------------------------------------------ #
    # Loading / saving
    # ------------------------------------------------------------------ #
    @classmethod
    def load_ply(cls, path: str | Path, device: str | torch.device = "cpu") -> "GaussianModel":
        from plyfile import PlyData
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"point cloud not found: {path}")
        ply = PlyData.read(str(path))
        if "vertex" not in ply:
            raise ValueError(f"'{path}' has no 'vertex' element")
        v = ply["vertex"]
        names = v.data.dtype.names or ()
        required = {"x", "y", "z", "opacity"}
        missing = required - set(names)
        if missing:
            raise ValueError(f"'{path}' is not a 3DGS PLY; missing {sorted(missing)}")
        scale_names = _sorted_fields(names, "scale_")
        rot_names = _sorted_fields(names, "rot_")
        if len(scale_names) != 3 or len(rot_names) != 4:
            raise ValueError(
                f"'{path}': expected 3 scale_* and 4 rot_* fields, got "
                f"{len(scale_names)} and {len(rot_names)}"
            )
        xyz = np.stack([np.asarray(v[n], np.float32) for n in ("x", "y", "z")], 1)
        opacity = np.asarray(v["opacity"], np.float32)
        scales = np.stack([np.asarray(v[n], np.float32) for n in scale_names], 1)
        quats = np.stack([np.asarray(v[n], np.float32) for n in rot_names], 1)
        if all(f"f_dc_{i}" in names for i in range(3)):
            sh_dc = np.stack([np.asarray(v[f"f_dc_{i}"], np.float32) for i in range(3)], 1)
        else:
            sh_dc = np.zeros_like(xyz, dtype=np.float32)
        rest_names = _sorted_fields(names, "f_rest_")
        sh_rest = (np.stack([np.asarray(v[n], np.float32) for n in rest_names], 1)
                   if rest_names else None)
        return cls(
            positions=torch.as_tensor(xyz, device=device),
            opacity_logit=torch.as_tensor(opacity, device=device),
            log_scales=torch.as_tensor(scales, device=device),
            quats=torch.as_tensor(quats, device=device),
            sh_dc=torch.as_tensor(sh_dc, device=device),
            sh_rest=torch.as_tensor(sh_rest, device=device) if sh_rest is not None else None,
            source_path=str(path),
        )

    def save_ply(self, path: str | Path) -> Path:
        """Write the current (possibly refined) Gaussians in the 3DGS schema."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.concatenate([
            self.positions.detach().cpu().numpy(),
            np.zeros((len(self), 3), np.float32),
            self.sh_dc.detach().cpu().numpy(),
            self.opacity_logit.detach().cpu().numpy()[:, None],
            self.log_scales.detach().cpu().numpy(),
            self.quats.detach().cpu().numpy(),
        ], axis=1)
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]
        vertex = np.empty(len(self), dtype=dtype)
        for i, (name, _) in enumerate(dtype):
            vertex[name] = data[:, i]
        from plyfile import PlyData, PlyElement
        elements = [PlyElement.describe(vertex, "vertex")]
        if self.sh_rest is not None:
            rest = self.sh_rest.detach().cpu().numpy()
            rest_dtype = [(f"f_rest_{i}", "f4") for i in range(rest.shape[1])]
            rest_data = np.empty(len(self), dtype=rest_dtype)
            for i in range(rest.shape[1]):
                rest_data[f"f_rest_{i}"] = rest[:, i]
            elements.append(PlyElement.describe(rest_data, "f_rest"))
        PlyData(elements, text=False).write(str(path))
        return path

    # ------------------------------------------------------------------ #
    # Activations / derived tensors
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.positions.shape[0]

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @property
    def rotations(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.quats, dim=-1)

    @property
    def colors_raw(self) -> torch.Tensor:
        return 0.5 + SH_C0 * self.sh_dc

    @property
    def colors(self) -> torch.Tensor:
        return self.colors_raw.clamp(0.0, 1.0)

    def covariance(self, eps: float = 1e-8) -> torch.Tensor:
        """``Sigma = R diag(s^2) R^T + eps I`` as ``(N,3,3)``."""
        R = quat_to_rotmat(self.rotations)
        S2 = torch.diag_embed(self.scales * self.scales + eps)
        Sigma = R @ S2 @ R.transpose(1, 2)
        eye = torch.eye(3, dtype=Sigma.dtype, device=Sigma.device).unsqueeze(0)
        return Sigma + eps * eye

    def covariance_inv(self, eps: float = 1e-8) -> torch.Tensor:
        return torch.linalg.inv(self.covariance(eps))

    def density_at(self, x: torch.Tensor, idx: Optional[torch.Tensor] = None,
                   cov_inv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate the full anisotropic 3D kernel.

        ``idx`` pairs every query point with one Gaussian. Without ``idx`` the
        point count must equal the Gaussian count (paired evaluation), which is
        the only mode used by transmittance and coupling terms.
        """
        if cov_inv is None:
            cov_inv = self.covariance_inv()
        if idx is None:
            if x.shape[0] != len(self):
                raise ValueError(
                    "density_at without idx requires one query per Gaussian; "
                    f"got {x.shape[0]} queries and {len(self)} Gaussians"
                )
            mu = self.positions
            Sinv = cov_inv
        else:
            mu = self.positions[idx]
            Sinv = cov_inv[idx]
        d = x - mu
        quad = torch.einsum("ni,nij,nj->n", d, Sinv, d)
        return torch.exp(-0.5 * quad)

    def estimated_normals(self) -> torch.Tensor:
        """Unit normals along the smallest-scale axis (world frame)."""
        R = quat_to_rotmat(self.rotations)
        idx = torch.argmin(self.scales, dim=1).view(-1, 1, 1).expand(-1, 3, 1)
        normals = torch.gather(R, 2, idx).squeeze(-1)
        return torch.nn.functional.normalize(normals, dim=-1)

    def flatness(self) -> torch.Tensor:
        """``1 - s_min / s_mid`` in ``[0, 1]``."""
        s_sorted, _ = torch.sort(self.scales, dim=1)
        s_min, s_mid = s_sorted[:, 0], s_sorted[:, 1]
        return (1.0 - s_min / s_mid.clamp_min(1e-12)).clamp(0.0, 1.0)

    def oriented_normals(self, points: torch.Tensor,
                         camera_centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Orient normals toward their camera; returns ``(normals, flipped_mask)``."""
        normals = self.estimated_normals()
        if points.shape != normals.shape:
            raise ValueError("points must have shape (N,3) matching normals")
        view = camera_centers - points
        flip = (normals * view).sum(-1) < 0
        normals = torch.where(flip[:, None], -normals, normals)
        return torch.nn.functional.normalize(normals, dim=-1), flip

    # ------------------------------------------------------------------ #
    # Subsetting / transforms / trainability
    # ------------------------------------------------------------------ #
    def clone(self) -> "GaussianModel":
        return GaussianModel(
            self.positions.clone(),
            self.opacity_logit.clone(),
            self.log_scales.clone(),
            self.quats.clone(),
            self.sh_dc.clone(),
            self.sh_rest.clone() if self.sh_rest is not None else None,
            self.source_path,
        )

    def detached(self) -> "GaussianModel":
        return GaussianModel(
            self.positions.detach().clone(),
            self.opacity_logit.detach().clone(),
            self.log_scales.detach().clone(),
            self.quats.detach().clone(),
            self.sh_dc.detach().clone(),
            self.sh_rest.detach().clone() if self.sh_rest is not None else None,
            self.source_path,
        )

    def filter_by_opacity(self, min_opacity: float = 0.05) -> Tuple["GaussianModel", torch.Tensor]:
        mask = self.opacity >= min_opacity
        out = GaussianModel(
            self.positions[mask],
            self.opacity_logit[mask],
            self.log_scales[mask],
            self.quats[mask],
            self.sh_dc[mask],
            self.sh_rest[mask] if self.sh_rest is not None else None,
            self.source_path,
        )
        return out, mask

    def apply_similarity(self, scale: float, center: torch.Tensor) -> None:
        """Apply ``x -> (x - center) * scale``; scales are multiplied by ``scale``."""
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.positions = (self.positions - center) * scale
        self.log_scales = self.log_scales + float(np.log(scale))
        self._cov_cache = None

    def make_trainable(self) -> "GaussianModel":
        """Replace raw tensors with leaf ``nn.Parameter`` objects."""
        self.positions = torch.nn.Parameter(self.positions.detach().clone())
        self.opacity_logit = torch.nn.Parameter(self.opacity_logit.detach().clone())
        self.log_scales = torch.nn.Parameter(self.log_scales.detach().clone())
        self.quats = torch.nn.Parameter(self.quats.detach().clone())
        self.sh_dc = torch.nn.Parameter(self.sh_dc.detach().clone())
        return self

    def parameters(self):
        seen = set()
        for name in ("positions", "opacity_logit", "log_scales", "quats", "sh_dc"):
            p = getattr(self, name)
            if isinstance(p, torch.nn.Parameter) and p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                yield p

    def trainable_parameters(self) -> dict:
        """Named trainable parameters, in a fixed, documented order."""
        out = {}
        for name in ("positions", "opacity_logit", "log_scales", "quats", "sh_dc"):
            p = getattr(self, name)
            if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                out[name] = p
        return out

    def state_dict(self) -> dict:
        return {
            "positions": self.positions.detach().cpu(),
            "opacity_logit": self.opacity_logit.detach().cpu(),
            "log_scales": self.log_scales.detach().cpu(),
            "quats": self.quats.detach().cpu(),
            "sh_dc": self.sh_dc.detach().cpu(),
            "sh_rest": self.sh_rest.detach().cpu() if self.sh_rest is not None else None,
            "source_path": self.source_path,
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: str | torch.device = "cpu") -> "GaussianModel":
        def to_dev(x):
            return None if x is None else torch.as_tensor(x, device=device)
        return cls(
            positions=to_dev(state["positions"]),
            opacity_logit=to_dev(state["opacity_logit"]),
            log_scales=to_dev(state["log_scales"]),
            quats=to_dev(state["quats"]),
            sh_dc=to_dev(state["sh_dc"]),
            sh_rest=to_dev(state.get("sh_rest")),
            source_path=state.get("source_path"),
        )

    def validate(self) -> dict:
        with torch.no_grad():
            scale = self.scales
            stats = {
                "n_gaussians": len(self),
                "position_finite": bool(torch.isfinite(self.positions).all()),
                "opacity_finite": bool(torch.isfinite(self.opacity).all()),
                "scale_finite": bool(torch.isfinite(scale).all()),
                "quat_norm_mean": float(self.rotations.norm(dim=-1).mean()),
                "opacity_min": float(self.opacity.min()),
                "opacity_max": float(self.opacity.max()),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "n_below_min_opacity": int((self.opacity < 0.05).sum()),
            }
            stats["valid"] = (stats["n_gaussians"] > 0
                              and stats["position_finite"]
                              and stats["opacity_finite"]
                              and stats["scale_finite"])
        return stats
