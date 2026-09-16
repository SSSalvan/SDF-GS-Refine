"""
losses.py

Implements Persamaan 3.1:
    L_total = L_photo + lambda_e * L_eikonal + lambda_n * L_normal

L_eikonal (Subbab 2.7 / Gropp et al. 2020):
    E_{x~D} [ (||grad f(x)||_2 - 1)^2 ]

L_normal (Subbab 2.8, "Normal Alignment Loss"):
    Encourages grad f(x_s)/||grad f(x_s)|| to align with the Gaussian-derived
    normal at each transmittance-crossing surface sample x_s, AND (per Subbab
    3.6, "fθ(xs) ~ 0 untuk titik-titik permukaan xs") the zero-boundary
    condition at those same points.
    *** IMPLEMENTATION NOTE (reported in Technical Review) ***
    The paper's Eq. 3.1 lists three terms (photo, eikonal, normal) but the
    text of Subbab 3.6 separately states the zero-boundary condition without
    giving it its own explicit weight in Table 3.3 (which is left blank for
    this term). We fold the zero-boundary constraint into L_normal (same
    lambda_n) since both are supervised at the identical surface-sample
    locations and the paper does not specify a distinct weight for it -- this
    is flagged explicitly rather than silently invented.

L_photo (Subbab 2.4/2.5, alpha-compositing volume rendering, discretized
exactly as Persamaan 2.7/3.2 but accumulating COLOR instead of transmittance):
    C_hat(r) = sum_i T_i * alpha_i * G_i(x_i) * c_i
    L_photo = mean |C_hat(r) - C_gt(r)|   (L1, standard choice, matches the
    "photometric reconstruction loss" language in the paper, exact norm not
    specified further than "meminimalkan photometric loss" in Bab III).

*** KNOWN ISSUE / FIX (this revision) ***
When no real DTU images are available, the pipeline falls back to a
"frozen render of the pretrained Gaussian scene" as the L_photo target
(see driving notebook, Section 11, DATASET REALITY). That frozen target is
built by cloning the SAME pretrained state that gauss_f is reset to at the
start of training. Under L1, d/d(rendered) |rendered - target| = sign(rendered
- target). At iteration 0, rendered == target exactly (elementwise), so
sign(0) == 0: L_photo genuinely evaluates to 0 AND its gradient into the
Gaussian parameters is exactly zero, every iteration, forever -- Adam never
receives a nonzero gradient for {positions, opacity, scales, rotations,
colors}, so opt_gauss.step() is a permanent no-op even though it runs every
iteration. This is a true fixed point, not noise: it does not "wear off"
with more iterations or different camera batches, because render(gauss_f,
cam) == render(gauss_frozen_target, cam) for any cam as long as the two
parameter sets are identical, and nothing ever pushes them apart.

Fix: build_frozen_photometric_target() below perturbs the cloned pretrained
state by a small amount before it is frozen, so the target is numerically
distinct from gauss_f at reset time. This does not change what L_photo is
supposed to measure -- the target is still anchored to the real pretrained
appearance (per the paper's justification that the 3DGS model is already
fully trained), it is simply no longer bit-identical to the live parameters,
which is what gives L1 a nonzero gradient direction to work with. Magnitude
is small relative to parameter scale, so the term still primarily penalizes
genuine drift-from-pretrained, rather than being dominated by injected noise.
"""
import torch
from src.transmittance import _cull_candidates


# Default perturbation scale for the frozen photometric target, expressed as
# a fraction of each tensor's own per-tensor std. Small enough that the
# target still represents "the pretrained appearance," large enough that
# rendered != target_rgb (in float32) at iteration 0.
DEFAULT_PERTURB_FRACTION = 0.02


def eikonal_loss(grad: torch.Tensor) -> torch.Tensor:
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def normal_and_boundary_loss(f_at_surface: torch.Tensor, grad_at_surface: torch.Tensor,
                              target_normals: torch.Tensor) -> torch.Tensor:
    zero_term = (f_at_surface ** 2).mean()
    grad_dir = torch.nn.functional.normalize(grad_at_surface, dim=-1)
    cos = (grad_dir * target_normals).sum(-1)
    align_term = (1.0 - cos.abs()).mean()   # abs() -> normal orientation sign is ambiguous from scale-axis proxy
    return zero_term + align_term


@torch.no_grad()
def _order_candidates_for_render(gauss, origins, dirs, cull_sigma=3.0, max_candidates=4096):
    return _cull_candidates(gauss, origins, dirs, cull_sigma, max_candidates)


def render_rays(gauss, origins, dirs, cull_sigma=3.0, max_candidates=4096):
    """
    Naive (non-tile-based) differentiable point renderer implementing the
    SAME alpha-compositing math as the official 3DGS tile rasterizer
    (Kerbl et al. 2023, Subbab 2.1) -- an "implementation detail" / engineering
    substitution (different code path, identical math), used here because no
    CUDA tile-rasterizer build is available in this CPU-only sandbox. This is
    NOT a methodological change: it computes the exact same discretized
    volume-rendering equation, just per-ray instead of per-tile, and is
    therefore unsuitable for full-resolution real-time rendering but
    perfectly valid for verifying pipeline correctness at small batch sizes.
    """
    device = origins.device
    candidates = _order_candidates_for_render(gauss, origins, dirs, cull_sigma, max_candidates)
    colors = []
    for r in range(origins.shape[0]):
        idx = candidates[r]
        if idx.numel() == 0:
            colors.append(torch.zeros(3, device=device))
            continue
        o, d = origins[r], dirs[r]
        v = gauss.positions[idx] - o
        t = (v * d).sum(-1)
        order = torch.argsort(t)
        idx_sorted = idx[order]
        t_sorted = t[order]
        x_i = o.unsqueeze(0) + t_sorted.unsqueeze(-1) * d.unsqueeze(0)
        Gi = gauss.density_at(x_i, idx_sorted)
        alpha_i = gauss.opacity[idx_sorted]
        a = torch.clamp(alpha_i * Gi, 0.0, 1.0 - 1e-6)
        term = 1.0 - a
        cum = torch.cumprod(term, dim=0)
        T = torch.cat([torch.ones(1, device=device), cum[:-1]])
        w = (T * a).unsqueeze(-1)                     # (M,1)
        c = gauss.colors[idx_sorted]                   # (M,3)
        colors.append((w * c).sum(0))
    return torch.stack(colors)


def photometric_loss(gauss, origins, dirs, target_rgb, cull_sigma=3.0, max_candidates=4096):
    rendered = render_rays(gauss, origins, dirs, cull_sigma, max_candidates)
    return torch.nn.functional.l1_loss(rendered, target_rgb), rendered


def build_frozen_photometric_target(gauss_initial_state, perturb_fraction=DEFAULT_PERTURB_FRACTION,
                                     generator=None):
    """
    Build the "frozen render of the pretrained Gaussian scene" target used as
    a synthetic L_photo target when no real DTU images are available (see
    module docstring, KNOWN ISSUE / FIX).

    gauss_initial_state must already be an independent clone of the
    pretrained state (e.g. gauss_pretrained_state.clone_detached()) -- this
    function perturbs that clone in place and returns it; it does not clone
    again, so callers must pass a clone they are willing to have mutated,
    never the live gauss_f or the shared immutable pretrained baseline.

    Only `positions` and `colors` are perturbed. These are the two tensors
    that most directly determine rendered RGB via `render_rays` (positions
    move where along each ray G_i(x_i) is evaluated; colors are compositied
    directly), so perturbing them is sufficient to break the rendered ==
    target identity and is more conservative than also perturbing
    opacity/scales/rotations, which shape the compositing weights themselves
    and would change what "drift from pretrained" means for those params.

    perturb_fraction is relative to each tensor's own per-tensor std, so the
    perturbation scales sensibly regardless of the underlying scene's units.
    """
    if not (0.0 < perturb_fraction < 1.0):
        raise ValueError(f"perturb_fraction should be a small positive fraction, got {perturb_fraction}")

    with torch.no_grad():
        pos_std = gauss_initial_state.positions.std()
        color_std = gauss_initial_state.colors.std()
        pos_noise = torch.randn(gauss_initial_state.positions.shape,
                                 device=gauss_initial_state.positions.device,
                                 generator=generator) * (pos_std * perturb_fraction)
        color_noise = torch.randn(gauss_initial_state.colors.shape,
                                   device=gauss_initial_state.colors.device,
                                   generator=generator) * (color_std * perturb_fraction)
        gauss_initial_state.positions += pos_noise
        gauss_initial_state.colors += color_noise

    return gauss_initial_state