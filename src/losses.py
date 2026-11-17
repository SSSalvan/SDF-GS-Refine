"""Loss terms for Stage 1 and joint refinement.

Stage-1 loss (thesis, recovered operational form):

    L = 3.0 * mean |f(x_s)|
      + 0.2 * weighted_mean [ w * (1 - cos(grad f, n)) ]
      + 0.1 * mean (||grad f|| - 1)^2
      + 0.1 * mean exp(-100 |f(x_near)|)
      + 1.0 * mean relu(0.005 - f(x_free))

The |f| vs f^2 choice is explicit (``surface_loss_norm``). No term silently
detaches gradients. The near-surface term penalizes stray zero crossings; it
is not a data term.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def surface_loss(f: torch.Tensor, norm: str = "l1") -> torch.Tensor:
    if f.numel() == 0:
        raise ValueError("surface_loss received an empty batch")
    if norm == "l1":
        return f.abs().mean()
    if norm == "l2":
        return (f * f).mean()
    raise ValueError(f"surface_loss_norm must be 'l1' or 'l2', got {norm!r}")


def normal_loss(grad: torch.Tensor, normals: torch.Tensor,
                weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if grad.shape != normals.shape:
        raise ValueError("grad and normals must have the same shape")
    if weights.shape != grad.shape[:1]:
        raise ValueError("weights must be (N,)")
    g = grad / grad.norm(dim=-1, keepdim=True).clamp_min(eps)
    n = normals / normals.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos = (g * n).sum(-1).clamp(-1.0, 1.0)
    per = (1.0 - cos) * weights
    denom = weights.sum().clamp_min(eps)
    return per.sum() / denom


def eikonal_loss(grad: torch.Tensor) -> torch.Tensor:
    if grad.numel() == 0:
        raise ValueError("eikonal_loss received an empty batch")
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def stray_surface_loss(f: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    if f.numel() == 0:
        raise ValueError("stray_surface_loss received an empty batch")
    if beta <= 0:
        raise ValueError("beta must be positive")
    return torch.exp(-beta * f.abs()).mean()


def free_space_loss(f: torch.Tensor, margin: float = 0.005) -> torch.Tensor:
    if f.numel() == 0:
        raise ValueError("free_space_loss received an empty batch")
    return F.relu(margin - f).mean()


def stage1_losses(
    f_surface: torch.Tensor,
    grad_surface: torch.Tensor,
    normals: torch.Tensor,
    normal_weights: torch.Tensor,
    f_uniform: torch.Tensor,
    grad_uniform: torch.Tensor,
    f_near: torch.Tensor,
    f_free: torch.Tensor,
    cfg,
) -> Dict[str, torch.Tensor | float]:
    """Compute all Stage-1 terms and the weighted total."""
    terms = {
        "surface": surface_loss(f_surface, cfg.surface_loss_norm),
        "normal": normal_loss(grad_surface, normals, normal_weights),
        "eikonal": eikonal_loss(grad_uniform),
        "stray": stray_surface_loss(f_near),
        "free": free_space_loss(f_free, cfg.free_margin),
    }
    total = (cfg.surface_weight * terms["surface"]
             + cfg.normal_weight * terms["normal"]
             + cfg.eikonal_weight * terms["eikonal"]
             + cfg.stray_weight * terms["stray"]
             + cfg.free_weight * terms["free"])
    terms["total"] = total
    return terms


def photometric_l1(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute photometric error, both ``(H,W,3)`` or ``(N,3)``."""
    if rendered.shape != target.shape:
        raise ValueError(
            f"rendered/target shape mismatch: {tuple(rendered.shape)} vs "
            f"{tuple(target.shape)}"
        )
    if rendered.numel() == 0:
        raise ValueError("photometric_l1 received empty tensors")
    return (rendered - target).abs().mean()


def coupling_loss(
    f_mu: torch.Tensor,
    grad_mu: torch.Tensor,
    normals: torch.Tensor,
    epsilon: float = 0.02,
    zero_weight: float = 0.1,
    normal_weight: float = 0.05,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Couple the SDF to nearby Gaussian centres.

    Applies only to centres with ``|f(mu)| <= epsilon`` (the thesis's
    ``sigma = 0.02`` surface band). Returns ``(loss, metrics)``. When no centre
    is in the band, the loss is exactly zero but remains connected to the graph.
    """
    if f_mu.shape != grad_mu.shape[:1]:
        raise ValueError("f_mu must be (N,) and grad_mu (N,3)")
    if normals.shape != grad_mu.shape:
        raise ValueError("normals must be (N,3)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mask = f_mu.abs() <= epsilon
    n = int(mask.sum().item())
    zero = f_mu.sum() * 0.0
    if n == 0:
        return zero, {"n_coupled": 0, "zero": 0.0, "normal": 0.0}
    f_sel = f_mu[mask]
    g = grad_mu[mask]
    nrm = normals[mask]
    zero_term = f_sel.abs().mean()
    g_unit = g / g.norm(dim=-1, keepdim=True).clamp_min(eps)
    n_unit = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos = (g_unit * n_unit).sum(-1).abs().clamp(0.0, 1.0)
    normal_term = (1.0 - cos).mean()
    loss = zero_weight * zero_term + normal_weight * normal_term
    return loss, {
        "n_coupled": n,
        "zero": float(zero_term.detach()),
        "normal": float(normal_term.detach()),
    }


def lr_scale(step: int, total_steps: int, warmup_fraction: float = 0.05,
             min_ratio: float = 0.02) -> float:
    """Linear warm-up then cosine decay, as ``base_lr * scale``."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not (0.0 <= warmup_fraction < 1.0):
        raise ValueError("warmup_fraction must be in [0,1)")
    if not (0.0 <= min_ratio <= 1.0):
        raise ValueError("min_ratio must be in [0,1]")
    step = max(0, int(step))
    warmup = max(1, int(round(warmup_fraction * total_steps)))
    if step < warmup:
        return float(step + 1) / float(warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    t = min(max(t, 0.0), 1.0)
    return float(min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))


def psnr(rendered: torch.Tensor, target: torch.Tensor,
         max_value: float = 1.0) -> float:
    mse = float(((rendered - target) ** 2).mean())
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)
