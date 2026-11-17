"""Differentiable 3D Gaussian splatting for small photometric patches.

This renderer is deliberately **not** an SDF renderer. It rasterizes the 3D
Gaussian primitives' projected 2D kernels and is used only for the optional
photometric joint-refinement term. The Gaussian-transmittance estimator in
:mod:`sdfgs.transmittance` remains a separate geometric formulation.

The implementation is a direct, small-scale 3DGS projection:

* world covariance ``Sigma = R S S^T R^T``;
* 2D covariance ``J W Sigma W^T J^T`` with the standard pinhole Jacobian;
* alpha compositing in depth order, ``alpha_i * prod_{j<i}(1-alpha_j)``;
* DC spherical-harmonic colour (higher-order SH is not used).

It is intended for 32x32 patches, not full-resolution 3DGS rasterization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .cameras import Camera
from .gaussian import GaussianModel


@dataclass
class RenderDiagnostics:
    n_gaussians: int
    n_selected: int
    patch_xy: Tuple[int, int]
    patch_size: int
    depth_min: float
    depth_max: float


def sample_patch_xy(camera: Camera, patch_size: int,
                    generator: Optional[torch.Generator] = None,
                    training: bool = True) -> Tuple[int, int]:
    """Choose a valid top-left patch coordinate."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if patch_size > camera.H or patch_size > camera.W:
        raise ValueError(
            f"patch_size {patch_size} exceeds image size "
            f"({camera.H}x{camera.W})"
        )
    if training:
        if generator is None:
            raise ValueError("a generator is required for random patch sampling")
        x = int(torch.randint(0, camera.W - patch_size + 1, (1,),
                              generator=generator).item())
        y = int(torch.randint(0, camera.H - patch_size + 1, (1,),
                              generator=generator).item())
        return x, y
    return (camera.W - patch_size) // 2, (camera.H - patch_size) // 2


class GaussianPatchRenderer:
    """Differentiable patch renderer for Gaussian primitives."""

    def __init__(self, background: float = 0.0, max_candidates: int = 8192,
                 radius_sigma: float = 3.0, min_depth: float = 1e-4,
                 cov_epsilon_px: float = 0.3) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if radius_sigma <= 0:
            raise ValueError("radius_sigma must be positive")
        if min_depth <= 0:
            raise ValueError("min_depth must be positive")
        self.background = float(background)
        self.max_candidates = int(max_candidates)
        self.radius_sigma = float(radius_sigma)
        self.min_depth = float(min_depth)
        self.cov_epsilon_px = float(cov_epsilon_px)

    def _project(self, gaussians: GaussianModel, camera: Camera):
        device = gaussians.positions.device
        dtype = gaussians.positions.dtype
        R = torch.as_tensor(camera.R, dtype=dtype, device=device)
        t = torch.as_tensor(camera.t, dtype=dtype, device=device)
        K = camera.K
        xyz = torch.einsum("ij,nj->ni", R, gaussians.positions) + t
        z = xyz[:, 2]
        valid = z > self.min_depth
        z_safe = torch.where(valid, z, torch.ones_like(z))
        u = float(K[0, 0]) * xyz[:, 0] / z_safe + float(K[0, 2])
        v = float(K[1, 1]) * xyz[:, 1] / z_safe + float(K[1, 2])

        # Camera-frame covariance: W Sigma W^T with W = R.
        cov = gaussians.covariance()
        cov_cam = torch.einsum("ij,njk,lk->nil", R, cov, R)
        # Pinhole Jacobian in the (x_cam, y_cam, z_cam) convention.
        fx, fy = float(K[0, 0]), float(K[1, 1])
        a = fx / z_safe
        c = -fx * xyz[:, 0] / (z_safe * z_safe)
        e = fy / z_safe
        f = -fy * xyz[:, 1] / (z_safe * z_safe)
        c00 = a * a * cov_cam[:, 0, 0] + 2 * a * c * cov_cam[:, 0, 2] \
            + c * c * cov_cam[:, 2, 2]
        c01 = (a * e * cov_cam[:, 0, 1] + a * f * cov_cam[:, 0, 2]
               + c * e * cov_cam[:, 1, 2] + c * f * cov_cam[:, 2, 2])
        c11 = e * e * cov_cam[:, 1, 1] + 2 * e * f * cov_cam[:, 1, 2] \
            + f * f * cov_cam[:, 2, 2]
        c00 = c00 + self.cov_epsilon_px
        c11 = c11 + self.cov_epsilon_px
        radius = self.radius_sigma * torch.sqrt(torch.clamp(
            torch.maximum(c00, c11), min=1e-12))
        return u, v, z, valid, c00, c01, c11, radius

    @torch.no_grad()
    def _select(self, u, v, z, valid, radius, opacity, x0, y0, patch_size):
        sel = (valid
               & (u + radius >= x0) & (u - radius < x0 + patch_size)
               & (v + radius >= y0) & (v - radius < y0 + patch_size)
               & (opacity > 1e-4))
        idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
        if idx.numel() > self.max_candidates:
            top = torch.topk(opacity[idx], self.max_candidates,
                             largest=True).indices
            idx = idx[top]
        return idx

    def render(self, gaussians: GaussianModel, camera: Camera,
               patch_size: int, patch_xy: Optional[Tuple[int, int]] = None,
               return_diagnostics: bool = False):
        """Render a ``(patch_size, patch_size, 3)`` image.

        Gradients flow to positions, opacity, scales, rotations and DC colour.
        """
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if patch_size > camera.H or patch_size > camera.W:
            raise ValueError("patch_size exceeds image size")
        if patch_xy is None:
            patch_xy = ((camera.W - patch_size) // 2,
                        (camera.H - patch_size) // 2)
        x0, y0 = int(patch_xy[0]), int(patch_xy[1])
        if not (0 <= x0 <= camera.W - patch_size):
            raise ValueError(f"patch x0={x0} outside image")
        if not (0 <= y0 <= camera.H - patch_size):
            raise ValueError(f"patch y0={y0} outside image")
        if len(gaussians) == 0:
            img = torch.full((patch_size, patch_size, 3), self.background,
                             dtype=torch.float32,
                             device=gaussians.positions.device)
            if return_diagnostics:
                return img, RenderDiagnostics(0, 0, (x0, y0), patch_size, 0.0, 0.0)
            return img

        u, v, z, valid, c00, c01, c11, radius = self._project(
            gaussians, camera)
        opacity_all = gaussians.opacity
        idx = self._select(u, v, z, valid, radius, opacity_all, x0, y0,
                           patch_size)
        if idx.numel() == 0:
            img = torch.full((patch_size, patch_size, 3), self.background,
                             dtype=gaussians.positions.dtype,
                             device=gaussians.positions.device)
            if return_diagnostics:
                return img, RenderDiagnostics(len(gaussians), 0, (x0, y0),
                                              patch_size, 0.0, 0.0)
            return img

        u_s, v_s, z_s = u[idx], v[idx], z[idx]
        a_s, b_s, c_s = c00[idx], c01[idx], c11[idx]
        det = (a_s * c_s - b_s * b_s).clamp_min(1e-12)
        inv00 = c_s / det
        inv01 = -b_s / det
        inv11 = a_s / det

        ys = torch.arange(y0, y0 + patch_size, dtype=torch.float32,
                          device=u.device) + 0.5
        xs = torch.arange(x0, x0 + patch_size, dtype=torch.float32,
                          device=u.device) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        dx = gx.reshape(-1)[None, :] - u_s[:, None]
        dy = gy.reshape(-1)[None, :] - v_s[:, None]
        power = -0.5 * (inv00[:, None] * dx * dx
                        + 2.0 * inv01[:, None] * dx * dy
                        + inv11[:, None] * dy * dy)
        alpha = opacity_all[idx][:, None] * torch.exp(power)
        alpha = alpha.clamp(0.0, 0.999)

        order = torch.argsort(z_s.detach(), stable=True)
        alpha = alpha[order]
        colors = gaussians.colors[idx][order]
        ones = torch.ones((1, alpha.shape[1]), dtype=alpha.dtype,
                          device=alpha.device)
        T_after = torch.cumprod(1.0 - alpha, dim=0)
        T_before = torch.cat([ones, T_after[:-1]], dim=0)
        weights = alpha * T_before

        img = torch.einsum("mp,mc->pc", weights, colors)
        final_T = T_after[-1]
        img = img + self.background * final_T[:, None]
        img = img.reshape(patch_size, patch_size, 3)

        if return_diagnostics:
            diag = RenderDiagnostics(
                n_gaussians=len(gaussians), n_selected=int(idx.numel()),
                patch_xy=(x0, y0), patch_size=patch_size,
                depth_min=float(z_s.min().detach()),
                depth_max=float(z_s.max().detach()),
            )
            return img, diag
        return img


def crop_observed(image: torch.Tensor, patch_size: int,
                  patch_xy: Tuple[int, int]) -> torch.Tensor:
    """Crop an observed image to the exact patch used by the renderer."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("observed image must be (H,W,3)")
    x0, y0 = int(patch_xy[0]), int(patch_xy[1])
    if not (0 <= x0 <= image.shape[1] - patch_size):
        raise ValueError("patch x coordinate outside observed image")
    if not (0 <= y0 <= image.shape[0] - patch_size):
        raise ValueError("patch y coordinate outside observed image")
    return image[y0:y0 + patch_size, x0:x0 + patch_size, :]
