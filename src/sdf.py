"""Neural signed-distance field used by SDF-GS Refine.

Architecture from the thesis: 8 hidden layers of width 256, Softplus with
beta = 100, and a single linear scalar output. No skip connections are used:
the previous implementation's skip path is not part of the described network.

Initialization is the IGR geometric initialization with radius ``r = 0.5``
(Gropp et al. 2020): hidden weights He-normal, final weights very small with
mean ``sqrt(pi) / sqrt(fan_in)``, final bias ``-r``. With a near-ReLU
activation this gives ``f(x) ~= |x| - r`` at initialization, i.e. a sphere.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFNetwork(nn.Module):
    def __init__(self, hidden_layers: int = 8, hidden_width: int = 256,
                 softplus_beta: float = 100.0, radius_init: float = 0.5,
                 geometric_init: bool = True, out_dim: int = 1) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        if hidden_width < 1:
            raise ValueError("hidden_width must be >= 1")
        if softplus_beta <= 0:
            raise ValueError("softplus_beta must be positive")
        self.hidden_layers = int(hidden_layers)
        self.hidden_width = int(hidden_width)
        self.softplus_beta = float(softplus_beta)
        self.radius_init = float(radius_init)
        self.out_dim = int(out_dim)
        self.geometric_init = bool(geometric_init)

        dims = [3] + [self.hidden_width] * self.hidden_layers + [self.out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self._init_weights()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            fan_in, fan_out = layer.weight.shape[1], layer.weight.shape[0]
            if i == n - 1:
                if self.geometric_init:
                    # Official IGR final layer: tiny spread around a positive mean
                    # so the initial field is approximately |x| - radius.
                    nn.init.normal_(layer.weight,
                                    mean=math.sqrt(math.pi) / math.sqrt(fan_in),
                                    std=1e-5)
                    nn.init.constant_(layer.bias, -self.radius_init)
                else:
                    nn.init.normal_(layer.weight, 0.0, 1.0 / math.sqrt(fan_in))
                    nn.init.zeros_(layer.bias)
            else:
                if self.geometric_init:
                    # Official IGR uses fan_out, not fan_in, for hidden layers.
                    # Using fan_in makes the first layer expand a 3-vector to a
                    # unit-norm 256-vector and destroys the sphere property.
                    nn.init.normal_(layer.weight, 0.0,
                                    math.sqrt(2.0) / math.sqrt(fan_out))
                else:
                    nn.init.normal_(layer.weight, 0.0, 1.0 / math.sqrt(fan_in))
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"expected (N,3) input, got {tuple(x.shape)}")
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.softplus(h, beta=self.softplus_beta)
        if self.out_dim == 1:
            return h[:, 0]
        return h

    def gradient(self, x: torch.Tensor,
                 create_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(f(x), df/dx)`` with autograd, keeping the graph by default."""
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        f = self.forward(x)
        grad = torch.autograd.grad(
            outputs=f,
            inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return f, grad

    def eikonal_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias used in the loss code."""
        return self.gradient(x, create_graph=True)

    # ------------------------------------------------------------------ #
    # Saving / loading
    # ------------------------------------------------------------------ #
    def config_dict(self) -> Dict[str, float | int | bool]:
        return {
            "hidden_layers": self.hidden_layers,
            "hidden_width": self.hidden_width,
            "softplus_beta": self.softplus_beta,
            "radius_init": self.radius_init,
            "out_dim": self.out_dim,
            "geometric_init": self.geometric_init,
        }

    def save(self, path: str | Path, extra: Optional[dict] = None) -> Path:
        from .io_utils import atomic_torch_save
        path = Path(path)
        payload = {
            "state_dict": self.state_dict(),
            "config": self.config_dict(),
        }
        if extra:
            payload.update(extra)
        atomic_torch_save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu",
             strict: bool = True) -> "SDFNetwork":
        payload = torch.load(str(path), map_location=device, weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(f"'{path}' is not an SDF checkpoint")
        cfg = payload.get("config", {})
        net = cls(**{k: cfg[k] for k in cfg if k in {
            "hidden_layers", "hidden_width", "softplus_beta", "radius_init",
            "out_dim", "geometric_init"}})
        missing, unexpected = net.load_state_dict(payload["state_dict"], strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, "
                               f"unexpected={unexpected}")
        net.to(device)
        net.eval()
        return net

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_sdf(cfg, device: str = "cpu") -> SDFNetwork:
    """Build an SDF network from an ``SDFConfig``."""
    net = SDFNetwork(
        hidden_layers=cfg.hidden_layers,
        hidden_width=cfg.hidden_width,
        softplus_beta=cfg.softplus_beta,
        radius_init=cfg.radius_init,
    )
    return net.to(device)
