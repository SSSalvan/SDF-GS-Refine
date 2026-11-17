"""Ray-wise 3D Gaussian transmittance and transmittance crossings.

This is the thesis estimator, not the official 3DGS tile rasterizer:

    T(t_j) = prod_{i<j} (1 - alpha_i * G_i(r(t_i)))
    G_i(x) = exp(-0.5 (x - mu_i)^T Sigma_i^-1 (x - mu_i))

Candidates are ordered by the depth of their centre. The first transmittance
bracket that crosses a level is linearly interpolated *between Gaussian
centres*, exactly as written in the thesis. A ray that never reaches the level
is discarded rather than extrapolated.

Discrete-index convention:
    T_before[j] = product over candidates 0..j-1
    T_after[j]  = T_before[j] * (1 - a_j)
The surface is the first j with ``T_before[j] <= level``; the crossing is
interpolated between centres ``j-1`` and ``j``. If the last candidate is the
only one that pushes the product below the level, no next centre exists; the
depth falls back to that centre and the point is flagged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .gaussian import GaussianModel


# --------------------------------------------------------------------------- #
# Culling
# --------------------------------------------------------------------------- #
def cull_candidates(
    model: GaussianModel,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    cull_sigma: float = 4.0,
    max_candidates: int = 4096,
    chunk_rays: int = 32,
    require_positive_depth: bool = True,
) -> List[torch.Tensor]:
    """Return per-ray candidate Gaussian indices, ordered by centre depth.

    A Gaussian is a candidate when its centre is in front of the camera and its
    perpendicular distance to the ray is at most ``cull_sigma * max_scale``.
    When a ray has more than ``max_candidates`` hits, the closest centres are
    kept. This is an approximation stated in the thesis, not an exact
    ray-ellipsoid intersection.
    """
    if origins.shape != dirs.shape or origins.shape[-1] != 3:
        raise ValueError("origins and dirs must both be (R,3)")
    if cull_sigma <= 0 or max_candidates < 1:
        raise ValueError("cull_sigma and max_candidates must be positive")
    n_rays = origins.shape[0]
    if n_rays == 0:
        return []
    pos = model.positions
    scale_max = model.scales.max(dim=1).values
    thresh2 = (float(cull_sigma) * scale_max) ** 2
    k = min(int(max_candidates), len(model))
    out: List[torch.Tensor] = []
    for start in range(0, n_rays, chunk_rays):
        end = min(start + chunk_rays, n_rays)
        o = origins[start:end]
        d = dirs[start:end]
        v = pos.unsqueeze(0) - o.unsqueeze(1)              # (B,N,3)
        t = torch.einsum("bnj,bj->bn", v, d)               # (B,N)
        v2 = (v * v).sum(-1)
        perp2 = (v2 - t * t).clamp_min(0.0)
        valid = perp2 <= thresh2.unsqueeze(0)
        if require_positive_depth:
            valid = valid & (t > 0)
        score = torch.where(valid, perp2, torch.full_like(perp2, math.inf))
        top_score, top_idx = torch.topk(score, k=k, dim=1, largest=False)
        for b in range(end - start):
            keep = torch.isfinite(top_score[b])
            idx = top_idx[b][keep]
            if idx.numel() == 0:
                out.append(idx)
                continue
            order = torch.argsort(t[b, idx])
            out.append(idx[order])
    return out


# --------------------------------------------------------------------------- #
# Padded transmittance computation
# --------------------------------------------------------------------------- #
@dataclass
class TransmittanceBatch:
    t: torch.Tensor             # (B,J) centre depths, inf for padding
    factor: torch.Tensor        # (B,J) 1-alpha*G, zero for padding
    T_before: torch.Tensor      # (B,J)
    T_after: torch.Tensor       # (B,J)
    alpha: torch.Tensor         # (B,J) effective alpha*G
    idx: torch.Tensor           # (B,J) candidate Gaussian indices, -1 padding
    lengths: torch.Tensor       # (B,)
    mask: torch.Tensor          # (B,J) valid entries


def _build_padded(
    model: GaussianModel,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    candidates: Sequence[torch.Tensor],
    cov_inv: Optional[torch.Tensor] = None,
) -> TransmittanceBatch:
    B = len(candidates)
    if B != origins.shape[0]:
        raise ValueError("candidate count must match ray count")
    if cov_inv is None:
        cov_inv = model.covariance_inv()
    lengths = torch.as_tensor([len(c) for c in candidates], dtype=torch.long,
                              device=origins.device)
    J = int(lengths.max().item()) if B else 0
    t_pad = torch.full((B, J), math.inf, dtype=origins.dtype, device=origins.device)
    factor = torch.zeros((B, J), dtype=origins.dtype, device=origins.device)
    alpha_pad = torch.zeros_like(factor)
    idx_pad = torch.full((B, J), -1, dtype=torch.long, device=origins.device)
    if J == 0:
        empty = torch.zeros((B, 0), dtype=origins.dtype, device=origins.device)
        return TransmittanceBatch(t_pad, factor, empty, empty, alpha_pad, idx_pad,
                                  lengths, torch.zeros_like(factor, dtype=torch.bool))
    ray_ids = torch.repeat_interleave(
        torch.arange(B, device=origins.device), lengths
    )
    flat_idx = torch.cat([c for c in candidates if len(c) > 0]) if any(len(c) for c in candidates) \
        else torch.empty(0, dtype=torch.long, device=origins.device)
    if flat_idx.numel() > 0:
        # position of each entry inside its own ray
        offsets = torch.cumsum(lengths, 0) - lengths
        local = torch.arange(flat_idx.numel(), device=origins.device) - offsets[ray_ids]
        o = origins[ray_ids]
        d = dirs[ray_ids]
        mu = model.positions[flat_idx]
        v = mu - o
        t_flat = (v * d).sum(-1)
        perp = v - t_flat.unsqueeze(-1) * d
        Sinv = cov_inv[flat_idx]
        quad = torch.einsum("ni,nij,nj->n", perp, Sinv, perp)
        g = torch.exp(-0.5 * quad.clamp_min(0.0))
        a = (model.opacity[flat_idx] * g).clamp(0.0, 1.0 - 1e-6)
        t_pad[ray_ids, local] = t_flat
        factor[ray_ids, local] = 1.0 - a
        alpha_pad[ray_ids, local] = a
        idx_pad[ray_ids, local] = flat_idx
    mask = torch.arange(J, device=origins.device).unsqueeze(0) < lengths.unsqueeze(1)
    # cumprod of factors, shifted by one: T_before[0] = 1
    cum = torch.cumprod(factor, dim=1)
    T_before = torch.cat(
        [torch.ones((B, 1), dtype=origins.dtype, device=origins.device), cum[:, :-1]],
        dim=1,
    )
    T_after = T_before * factor
    return TransmittanceBatch(t_pad, factor, T_before, T_after, alpha_pad,
                              idx_pad, lengths, mask)


# --------------------------------------------------------------------------- #
# Crossing search
# --------------------------------------------------------------------------- #
@dataclass
class Crossing:
    depth: torch.Tensor        # (B,)
    valid: torch.Tensor        # (B,) bool
    fraction: torch.Tensor     # (B,) interpolation fraction
    left_index: torch.Tensor   # (B,) candidate index of left centre (-1 if none)
    right_index: torch.Tensor  # (B,) candidate index of right centre
    T_left: torch.Tensor       # (B,)
    T_right: torch.Tensor      # (B,)
    fallback_last: torch.Tensor  # (B,) bool: no next centre existed


def _first_true(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    any_ = mask.any(dim=1)
    idx = torch.where(any_, mask.float().argmax(dim=1),
                      torch.full((mask.shape[0],), -1, dtype=torch.long,
                                 device=mask.device))
    return idx, any_


def find_crossing(batch: TransmittanceBatch, level: float) -> Crossing:
    """Find the first transmittance crossing of ``level``.

    Returns depths and validity per ray. Crossings are interpolated between
    centres; a crossing that only occurs at the final candidate is flagged.
    """
    B, J = batch.T_before.shape
    dev = batch.T_before.device
    if J == 0:
        z = torch.zeros(B, dtype=torch.float32, device=dev)
        f = torch.ones(B, dtype=torch.bool, device=dev)
        return Crossing(z, ~f, z, torch.full((B,), -1, dtype=torch.long, device=dev),
                        torch.full((B,), -1, dtype=torch.long, device=dev),
                        torch.ones_like(z), torch.ones_like(z), f)

    # Search only valid entries, and only j >= 1 (T_before[0] == 1).
    valid_mask = batch.mask.clone()
    if J > 0:
        valid_mask[:, 0] = False
    cross_before = valid_mask & (batch.T_before <= level)
    j, found = _first_true(cross_before)

    # Fallback: the final candidate(s) push T_after below the level, but no
    # later centre exists to bracket the crossing.
    cross_after = batch.mask & (batch.T_after <= level)
    j2, found2 = _first_true(cross_after)

    depth = torch.zeros(B, dtype=batch.t.dtype, device=dev)
    fraction = torch.zeros(B, dtype=batch.t.dtype, device=dev)
    T_left = torch.ones(B, dtype=batch.t.dtype, device=dev)
    T_right = torch.ones_like(T_left)
    left_idx = torch.full((B,), -1, dtype=torch.long, device=dev)
    right_idx = torch.full((B,), -1, dtype=torch.long, device=dev)
    fallback = torch.zeros(B, dtype=torch.bool, device=dev)
    valid = found | found2

    def gather_j(ji: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return batch.t.gather(1, pos.clamp_min(0).unsqueeze(1)).squeeze(1)

    # Bracket crossings: j >= 1.
    if found.any():
        jj = j.clamp_min(1)
        tl = gather_j(jj, jj - 1)
        tr = gather_j(jj, jj)
        Tl = batch.T_before.gather(1, (jj - 1).unsqueeze(1)).squeeze(1)
        Tr = batch.T_before.gather(1, jj.unsqueeze(1)).squeeze(1)
        denom = Tr - Tl
        frac = torch.where(denom.abs() > 1e-12, (level - Tl) / denom,
                           torch.full_like(denom, 0.5)).clamp(0.0, 1.0)
        d = tl + frac * (tr - tl)
        m = found
        depth = torch.where(m, d, depth)
        fraction = torch.where(m, frac, fraction)
        T_left = torch.where(m, Tl, T_left)
        T_right = torch.where(m, Tr, T_right)
        lidx = batch.idx.gather(1, (jj - 1).unsqueeze(1)).squeeze(1)
        ridx = batch.idx.gather(1, jj.unsqueeze(1)).squeeze(1)
        left_idx = torch.where(m, lidx, left_idx)
        right_idx = torch.where(m, ridx, right_idx)

    # Fallback for rays without a bracket: use the candidate that crossed.
    if found2.any():
        m = found2 & ~found
        jj = j2.clamp_min(0)
        tc = gather_j(jj, jj)
        Tc = batch.T_before.gather(1, jj.unsqueeze(1)).squeeze(1)
        Ta = batch.T_after.gather(1, jj.unsqueeze(1)).squeeze(1)
        denom = Ta - Tc
        frac = torch.where(denom.abs() > 1e-12, (level - Tc) / denom,
                           torch.full_like(denom, 1.0)).clamp(0.0, 1.0)
        depth = torch.where(m, tc, depth)
        fraction = torch.where(m, frac, fraction)
        T_left = torch.where(m, Tc, T_left)
        T_right = torch.where(m, Ta, T_right)
        cidx = batch.idx.gather(1, jj.unsqueeze(1)).squeeze(1)
        left_idx = torch.where(m, cidx, left_idx)
        right_idx = torch.where(m, cidx, right_idx)
        fallback = fallback | m

    depth = torch.where(valid, depth, torch.zeros_like(depth))
    return Crossing(depth, valid, fraction, left_idx, right_idx,
                    T_left, T_right, fallback)


# --------------------------------------------------------------------------- #
# Public ray API
# --------------------------------------------------------------------------- #
@dataclass
class RayResult:
    surface_depth: torch.Tensor
    surface_valid: torch.Tensor
    surface_index: torch.Tensor
    surface_fallback_last: torch.Tensor
    surface_T_left: torch.Tensor
    surface_T_right: torch.Tensor
    free_depth: torch.Tensor
    free_valid: torch.Tensor
    free_index: torch.Tensor
    n_candidates: torch.Tensor
    batch: TransmittanceBatch


def transmittance_on_rays(
    model: GaussianModel,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    candidates: Sequence[torch.Tensor],
    surface_level: float = 0.5,
    free_level: float = 0.9,
    cov_inv: Optional[torch.Tensor] = None,
) -> RayResult:
    """Compute surface and free-space crossings for a batch of rays."""
    batch = _build_padded(model, origins, dirs, candidates, cov_inv=cov_inv)
    surf = find_crossing(batch, surface_level)
    free = find_crossing(batch, free_level)
    return RayResult(
        surface_depth=surf.depth,
        surface_valid=surf.valid,
        surface_index=surf.left_index,
        surface_fallback_last=surf.fallback_last,
        surface_T_left=surf.T_left,
        surface_T_right=surf.T_right,
        free_depth=free.depth,
        free_valid=free.valid,
        free_index=free.left_index,
        n_candidates=batch.lengths,
        batch=batch,
    )


def ray_transmittance_full(
    model: GaussianModel,
    origin: torch.Tensor,
    direction: torch.Tensor,
    candidates: Optional[torch.Tensor] = None,
    cull_sigma: float = 4.0,
    max_candidates: int = 4096,
    surface_level: float = 0.5,
    free_level: float = 0.9,
) -> dict:
    """Diagnostic path for a single ray: full ordered transmittance curves."""
    o = origin.reshape(1, 3)
    d = direction.reshape(1, 3)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if candidates is None:
        candidates = cull_candidates(model, o, d, cull_sigma, max_candidates,
                                     chunk_rays=1)[0]
    batch = _build_padded(model, o, d, [candidates])
    surf = find_crossing(batch, surface_level)
    free = find_crossing(batch, free_level)
    n = int(batch.lengths[0].item())
    return {
        "t": batch.t[0, :n].detach().cpu(),
        "T_before": batch.T_before[0, :n].detach().cpu(),
        "T_after": batch.T_after[0, :n].detach().cpu(),
        "alpha": batch.alpha[0, :n].detach().cpu(),
        "indices": batch.idx[0, :n].detach().cpu(),
        "surface_depth": float(surf.depth[0]),
        "surface_valid": bool(surf.valid[0]),
        "surface_fallback_last": bool(surf.fallback_last[0]),
        "free_depth": float(free.depth[0]),
        "free_valid": bool(free.valid[0]),
    }
