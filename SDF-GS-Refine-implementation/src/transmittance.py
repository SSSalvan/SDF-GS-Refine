"""
transmittance.py

Implements the paper's CORE CONTRIBUTION #1 (Subbab 2.5 / Bab III Subbab 3.5):
extending NeuS's (Wang et al. 2023) transmittance-crossing criterion T(t*)=0.5
to the discrete kernel-sum density of 3DGS.

Persamaan 3.2:  T(t_j) = prod_{i<j} (1 - alpha_i * G_i(r(t_i)))
Persamaan 3.3:  t* ~= t_j + (0.5 - T(t_j)) / (T(t_j+1) - T(t_j)) * (t_j+1 - t_j)

where index i runs over Gaussians sorted by their own depth t_i = <mu_i - o, d>
along the ray r(t) = o + t*d, and G_i(r(t_i)) is the FULL 3D anisotropic
Gaussian density (gaussian.py: GaussianModel.density_at), matching the
paper's stated equation (Bab III Subbab 3.5) as opposed to the official
3DGS tile-based 2D-projected screen-space splat weight -- the paper's
formulation deliberately extends NeuS's ray-parametrized T(t), so we
implement exactly that, not the rasterizer's 2D approximation.

Mitigations (paper Subbab 3.5, both implemented and REQUIRED for a valid
dense surface-sample set):
  (1) Gaussians with alpha_i < 0.05 are filtered out BEFORE this stage
      (see gaussian.GaussianModel.filter_by_opacity, applied by the caller).
  (2) Rays whose transmittance never reaches T(t) <= 0.5 along their sampled
      trajectory (T_final > 0.5) are DISCARDED, not extrapolated.

Engineering-only addition (not a methodological change, explicitly allowed
under "Implementation detail" per the task brief): a perpendicular-distance
culling step so we don't evaluate every Gaussian against every ray -- any
Gaussian whose distance from the ray line exceeds `cull_sigma` times its own
max scale contributes G_i(r(t_i)) ~= 0 and would multiply the transmittance
product by ~1 anyway, so excluding it does not change Eq. 3.2/3.3 in any way
that matters numerically; it only makes the computation tractable on a
single CPU core.
"""
import torch


@torch.no_grad()
def _cull_candidates(gauss, origins, dirs, cull_sigma: float = 3.0, max_candidates: int = 4096):
    """
    For each ray, return indices of Gaussians whose center lies within
    `cull_sigma` * max_scale of the ray line (perpendicular distance).
    Returns a list (len = n_rays) of 1D LongTensors of Gaussian indices.
    """
    mu = gauss.positions                       # (N,3)
    max_scale = gauss.scales.max(dim=1).values  # (N,)
    n_rays = origins.shape[0]
    out = []
    for r in range(n_rays):
        o, d = origins[r], dirs[r]
        v = mu - o                              # (N,3)
        t = (v * d).sum(-1)                     # (N,)
        proj = o + t.unsqueeze(-1) * d           # (N,3)
        perp = (mu - proj).norm(dim=-1)          # (N,)
        thresh = cull_sigma * max_scale
        mask = (perp < thresh) & (t > 0)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() > max_candidates:
            # keep the closest (smallest perpendicular distance) candidates
            keep = torch.topk(-perp[idx], k=max_candidates).indices
            idx = idx[keep]
        out.append(idx)
    return out


def transmittance_crossing_points(gauss, origins, dirs, cull_sigma: float = 3.0,
                                   max_candidates: int = 4096, min_gaussians: int = 1):
    """
    Compute the T(t*)=0.5 surface sample for each ray (paper Eq 3.2-3.3).

    Returns:
        surface_points : (K,3) tensor, K <= n_rays (invalid rays discarded)
        surface_normals: (K,3) tensor, estimated normal at each surface point
                          (gradient direction of local density, finite-diff
                          free: uses the SDF-agnostic Gaussian normal proxy
                          of the nearest contributing Gaussian -- see note)
        valid_ray_idx  : (K,) indices into the original ray batch
    """
    device = origins.device
    candidates = _cull_candidates(gauss, origins, dirs, cull_sigma, max_candidates)

    normals_all = gauss.estimated_normals()

    surf_pts, surf_nrm, valid_idx = [], [], []
    for r in range(origins.shape[0]):
        idx = candidates[r]
        if idx.numel() < min_gaussians:
            continue
        o, d = origins[r], dirs[r]
        v = gauss.positions[idx] - o
        t = (v * d).sum(-1)                                  # depth of each candidate gaussian
        order = torch.argsort(t)
        idx_sorted = idx[order]
        t_sorted = t[order]

        x_i = o.unsqueeze(0) + t_sorted.unsqueeze(-1) * d.unsqueeze(0)   # (M,3) ray points at each t_i
        Gi = gauss.density_at(x_i, idx_sorted)                            # (M,)
        alpha_i = gauss.opacity[idx_sorted]                               # (M,)
        term = torch.clamp(1.0 - alpha_i * Gi, min=1e-6)

        # cumulative transmittance T(t_j) = prod_{i<j} term_i  (Eq. 3.2)
        cum = torch.cumprod(term, dim=0)
        T = torch.cat([torch.ones(1, device=device), cum[:-1]])   # T(t_0)=1

        below = torch.nonzero(T <= 0.5, as_tuple=False).squeeze(-1)
        if below.numel() == 0:
            continue  # mitigation (2): ray never crosses 0.5 -> discard
        j = below[0].item()
        if j == 0:
            t_star = t_sorted[0]
            nearest_idx = idx_sorted[0]
        else:
            Tj, Tj1 = T[j - 1], T[j]
            tj, tj1 = t_sorted[j - 1], t_sorted[j]
            denom = (Tj1 - Tj)
            frac = (0.5 - Tj) / denom if torch.abs(denom) > 1e-8 else torch.tensor(0.5, device=device)
            frac = torch.clamp(frac, 0.0, 1.0)
            t_star = tj + frac * (tj1 - tj)                       # Eq. 3.3
            nearest_idx = idx_sorted[j - 1 if frac < 0.5 else j]

        p_star = o + t_star * d
        surf_pts.append(p_star)
        surf_nrm.append(normals_all[nearest_idx])
        valid_idx.append(r)

    if len(surf_pts) == 0:
        return (torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device),
                torch.zeros((0,), dtype=torch.long, device=device))

    return (torch.stack(surf_pts), torch.stack(surf_nrm),
            torch.tensor(valid_idx, dtype=torch.long, device=device))
