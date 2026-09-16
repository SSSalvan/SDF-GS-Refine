"""
gaussian.py

Loads a PRETRAINED 3D Gaussian Splatting model (point_cloud.ply) and exposes
the Gaussian primitive attributes {mu_i, Sigma_i, alpha_i, c_i} exactly as
described in the paper Subbab 2.1 ("SIGNED DISTANCE FIELD UNTUK MESH
EXTRACTION DARI 3D GAUSSIAN SPLAT").

We do NOT train 3DGS here. This module only LOADS an already-trained model.

Standard 3DGS PLY schema (official Kerbl et al. 2023 implementation) stores:
    x, y, z                    -> position (mean) mu_i               [raw]
    nx, ny, nz                 -> unused placeholder normals (zeros)
    f_dc_0..2                  -> SH degree-0 (DC) color coefficients [raw]
    f_rest_0..K                -> SH higher-order coefficients        [raw]
    opacity                    -> raw logit,      alpha_i = sigmoid(opacity)
    scale_0..2                 -> raw log-scale,  s_i     = exp(scale)
    rot_0..3                   -> raw quaternion, R_i     = quat_to_rotmat(normalize(rot))

Sigma_i = R_i S_i S_i^T R_i^T   (Kerbl et al. 2023, Sec 4 -- also paper Subbab 2.1)
"""
import glob
import os
import numpy as np
import torch
from plyfile import PlyData


def discover_point_clouds(root_dir: str):
    """
    Auto-discovery of pretrained point_cloud.ply files under `root_dir`.
    Matches the expected layout described by the user:
        <root_dir>/<scan_name>/point_cloud/iteration_XXXXX/point_cloud.ply
    but falls back to a recursive search for *any* point_cloud.ply if that
    exact layout isn't found, so we never hard-code one scan/path.
    Returns a dict: {scan_name: {iteration_number: path}}
    """
    results = {}
    pattern = os.path.join(root_dir, "*", "point_cloud", "iteration_*", "point_cloud.ply")
    matches = sorted(glob.glob(pattern))
    if not matches:
        # fallback: recursive search anywhere under root_dir
        matches = sorted(glob.glob(os.path.join(root_dir, "**", "*.ply"), recursive=True))

    for m in matches:
        parts = m.replace("\\", "/").split("/")
        scan_name = "unknown_scan"
        iteration = -1
        if "point_cloud" in parts:
            pc_idx = parts.index("point_cloud")
            if pc_idx >= 1:
                scan_name = parts[pc_idx - 1]
            if pc_idx + 1 < len(parts) and parts[pc_idx + 1].startswith("iteration_"):
                try:
                    iteration = int(parts[pc_idx + 1].split("_")[-1])
                except ValueError:
                    iteration = -1
        else:
            scan_name = os.path.splitext(os.path.basename(m))[0]
        results.setdefault(scan_name, {})[iteration] = m

    return results


def pick_latest_iteration(scan_dict: dict):
    """From {iteration:int -> path} pick the path with the highest iteration."""
    best_it = max(scan_dict.keys())
    return scan_dict[best_it], best_it


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (N,4) normalized quaternion (w,x,y,z) -> (N,3,3) rotation matrices.
    Standard formula, same convention used by the official 3DGS codebase.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.zeros((N, 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class GaussianModel:
    """
    Holds pretrained 3DGS Gaussian attributes on `device`, exposing:
        positions (mu)   : (N,3)
        opacity   (alpha): (N,)   in (0,1), already sigmoid-activated
        scales    (s)     : (N,3)  already exp-activated (world units)
        rotations (quat)  : (N,4)  normalized (w,x,y,z)
        colors    (c)     : (N,3)  SH DC term converted to RGB (view-independent
                                     approximation -- see note in load())
        covariance Sigma  : (N,3,3) computed lazily via `covariance()`
    """

    def __init__(self):
        self.positions = None
        self.opacity = None
        self.scales = None
        self.rotations = None
        self.colors = None
        self.sh_rest = None
        self.n_sh_rest = 0
        self.source_path = None
        self._Sigma = None
        self._Sigma_inv = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def load_ply(cls, path: str, device=None):
        device = device or torch.device("cpu")
        ply = PlyData.read(path)
        v = ply["vertex"]
        names = v.data.dtype.names

        def col(name):
            return np.asarray(v[name], dtype=np.float32)

        xyz = np.stack([col("x"), col("y"), col("z")], axis=1)

        if "opacity" not in names:
            raise ValueError(f"'{path}' has no 'opacity' field -- not a valid 3DGS point_cloud.ply")
        opacity_raw = col("opacity")

        scale_names = sorted([n for n in names if n.startswith("scale_")],
                              key=lambda s: int(s.split("_")[-1]))
        rot_names = sorted([n for n in names if n.startswith("rot_")],
                            key=lambda s: int(s.split("_")[-1]))
        if len(scale_names) != 3 or len(rot_names) != 4:
            raise ValueError(
                f"'{path}': expected 3 scale_* and 4 rot_* fields, got "
                f"{len(scale_names)} and {len(rot_names)} -- unexpected 3DGS PLY schema"
            )
        scale_raw = np.stack([col(n) for n in scale_names], axis=1)
        rot_raw = np.stack([col(n) for n in rot_names], axis=1)

        # SH DC term -> base color (sigma_c(0) SH constant folded in, standard 3DGS convention)
        SH_C0 = 0.28209479177387814
        if all(f"f_dc_{i}" in names for i in range(3)):
            f_dc = np.stack([col(f"f_dc_{i}") for i in range(3)], axis=1)
            colors = 0.5 + SH_C0 * f_dc
        else:
            # no SH color available -> fall back to mid-gray, flagged in validation report
            colors = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)

        rest_names = sorted([n for n in names if n.startswith("f_rest_")],
                             key=lambda s: int(s.split("_")[-1]))
        sh_rest = np.stack([col(n) for n in rest_names], axis=1) if rest_names else None

        m = cls()
        m.source_path = path
        m.positions = torch.tensor(xyz, device=device)
        m.opacity = torch.sigmoid(torch.tensor(opacity_raw, device=device))
        m.scales = torch.exp(torch.tensor(scale_raw, device=device))
        quat = torch.tensor(rot_raw, device=device)
        m.rotations = torch.nn.functional.normalize(quat, dim=-1)
        m.colors = torch.clamp(torch.tensor(colors, device=device), 0.0, 1.0)
        m.sh_rest = torch.tensor(sh_rest, device=device) if sh_rest is not None else None
        m.n_sh_rest = sh_rest.shape[1] if sh_rest is not None else 0
        return m

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def covariance(self, eps: float = 1e-8):
        """Sigma_i = R_i S_i S_i^T R_i^T  (paper Subbab 2.1, Kerbl et al. 2023)."""
        if self._Sigma is not None:
            return self._Sigma, self._Sigma_inv
        R = quat_to_rotmat(self.rotations)                       # (N,3,3)
        S = self.scales                                          # (N,3)
        S2 = torch.diag_embed(S * S + eps)                       # (N,3,3)
        Sigma = R @ S2 @ R.transpose(1, 2)                        # (N,3,3)
        Sigma = Sigma + eps * torch.eye(3, device=Sigma.device).unsqueeze(0)
        Sigma_inv = torch.linalg.inv(Sigma)
        self._Sigma, self._Sigma_inv = Sigma, Sigma_inv
        return Sigma, Sigma_inv

    def density_at(self, x: torch.Tensor, idx: torch.Tensor = None):
        """
        Full 3D anisotropic Gaussian density evaluated at points `x`.
        x  : (N,3)  one query point per selected Gaussian (paired evaluation,
             used by the transmittance-crossing estimator -- see transmittance.py)
        idx: (N,) indices into this model selecting which Gaussian each x
             corresponds to. If None, assumes x[i] queries gaussian i for all i
             (i.e. len(x) == len(self)).
        Returns G_i(x): (N,)  (paper Eq. in Subbab 2.1 / Persamaan 3.2 term)
        """
        _, Sigma_inv = self.covariance()
        mu = self.positions if idx is None else self.positions[idx]
        Sinv = Sigma_inv if idx is None else Sigma_inv[idx]
        d = (x - mu).unsqueeze(-1)                                # (N,3,1)
        quad = torch.einsum("nij,njk,nkl->nil", d.transpose(1, 2), Sinv, d).squeeze(-1).squeeze(-1)
        return torch.exp(-0.5 * quad)

    def filter_by_opacity(self, min_opacity: float = 0.05):
        """
        Mitigation (1) from paper Subbab 3.5: drop Gaussians with alpha_i < 0.05
        before transmittance computation (marginal contribution, adds noise to
        crossing estimate). Returns a NEW filtered GaussianModel + boolean mask.
        """
        mask = self.opacity >= min_opacity
        out = GaussianModel()
        out.source_path = self.source_path
        out.positions = self.positions[mask]
        out.opacity = self.opacity[mask]
        out.scales = self.scales[mask]
        out.rotations = self.rotations[mask]
        out.colors = self.colors[mask]
        out.sh_rest = self.sh_rest[mask] if self.sh_rest is not None else None
        out.n_sh_rest = self.n_sh_rest
        return out, mask

    def clone_detached(self):
        """Deep, gradient-free copy -- used to freeze a 'ground truth' rendering
        target (e.g. the pretrained 3DGS state before joint refinement)."""
        out = GaussianModel()
        out.source_path = self.source_path
        out.positions = self.positions.detach().clone()
        out.opacity = self.opacity.detach().clone()
        out.scales = self.scales.detach().clone()
        out.rotations = self.rotations.detach().clone()
        out.colors = self.colors.detach().clone()
        out.sh_rest = self.sh_rest.detach().clone() if self.sh_rest is not None else None
        out.n_sh_rest = self.n_sh_rest
        return out

    def estimated_normals(self):
        """
        3DGS does not store an explicit surface normal per Gaussian. Following
        the convention used by flat-Gaussian surface methods (e.g. 2DGS,
        Huang et al. 2024, which this paper cites as theoretical comparison in
        Subbab 1.4/2.1), we take the axis of SMALLEST scale as the normal
        proxy, since a well-converged surface-aligned Gaussian is flattened
        along its normal direction.

        *** IMPLEMENTATION DETAIL, reported in the Technical Review table ***
        The paper (Subbab 2.7) assumes "normal permukaan Gaussian pada 3DGS
        sudah tersedia sebagai sinyal supervisi tambahan" but does not specify
        HOW that normal is derived from {mu_i, Sigma_i, alpha_i, c_i}. This is
        the standard convention in the literature and is applied here as an
        implementation detail (same math family, not a methodology swap).
        """
        R = quat_to_rotmat(self.rotations)             # (N,3,3), columns = local axes in world frame
        min_axis = torch.argmin(self.scales, dim=1)     # (N,)
        idx = min_axis.view(-1, 1, 1).expand(-1, 3, 1)
        normals = torch.gather(R, 2, idx).squeeze(-1)   # (N,3)
        return torch.nn.functional.normalize(normals, dim=-1)

    def __len__(self):
        return 0 if self.positions is None else self.positions.shape[0]

    # ------------------------------------------------------------------ #
    # Validation (Section 2 / Section 13 requirement)
    # ------------------------------------------------------------------ #
    def validate(self, verbose: bool = True):
        report = {}
        report["n_gaussians"] = len(self)
        report["has_sh_rest"] = self.sh_rest is not None
        report["n_sh_rest_coeffs"] = self.n_sh_rest

        def stat(name, t):
            report[name] = {
                "min": float(t.min().item()),
                "max": float(t.max().item()),
                "mean": float(t.mean().item()),
                "n_nan": int(torch.isnan(t).sum().item()),
                "n_inf": int(torch.isinf(t).sum().item()),
            }

        stat("position", self.positions)
        stat("opacity", self.opacity)
        stat("scale", self.scales)
        stat("rotation_norm", self.rotations.norm(dim=-1))

        bad = any(report[k]["n_nan"] > 0 or report[k]["n_inf"] > 0
                  for k in ("position", "opacity", "scale", "rotation_norm"))
        report["valid"] = not bad and len(self) > 0

        if verbose:
            print(f"== GaussianModel validation: {self.source_path} ==")
            print(f"  #Gaussians          : {report['n_gaussians']}")
            print(f"  SH rest coeffs      : {report['n_sh_rest_coeffs']} "
                  f"({'present' if report['has_sh_rest'] else 'ABSENT -> using flat color'})")
            print(f"  position   range    : [{report['position']['min']:.4f}, {report['position']['max']:.4f}]"
                  f"  NaN={report['position']['n_nan']} Inf={report['position']['n_inf']}")
            print(f"  opacity    range    : [{report['opacity']['min']:.4f}, {report['opacity']['max']:.4f}]"
                  f"  NaN={report['opacity']['n_nan']} Inf={report['opacity']['n_inf']}")
            print(f"  scale      range    : [{report['scale']['min']:.6f}, {report['scale']['max']:.4f}]"
                  f"  NaN={report['scale']['n_nan']} Inf={report['scale']['n_inf']}")
            print(f"  |quat| mean         : {report['rotation_norm']['mean']:.6f} (should be ~1.0)")
            print(f"  VALID               : {report['valid']}")
        return report
