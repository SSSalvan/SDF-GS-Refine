"""
sdf.py

Neural SDF f_theta: R^3 -> R  (paper Bab III, "Penerapan pipeline SDF-GS
Refine"): 8 hidden layers, width 256, Softplus(beta=100) activation, single
linear output layer, geometric initialization from Gropp et al. (2020, IGR)
so the network starts close to the signed distance function of the unit
sphere. Eikonal term |grad f(x)| computed via autograd (paper Subbab 2.7 /
Eq. Eikonal), exactly as required ("Jika menggunakan Eikonal, hitung nabla
f(x) dengan autograd").
"""
import numpy as np
import torch
import torch.nn as nn


class SDFNetwork(nn.Module):
    def __init__(self, d_in=3, d_hidden=256, n_layers=8, radius_init=1.0, beta=100.0):
        super().__init__()
        dims = [d_in] + [d_hidden] * n_layers + [1]
        self.n_layers = len(dims)
        self.radius_init = radius_init

        for l in range(self.n_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            self._geometric_init(lin, dims[l], dims[l + 1], is_last=(l == self.n_layers - 2),
                                  d_in=d_in, radius_init=radius_init)
            setattr(self, f"lin{l}", lin)

        self.activation = nn.Softplus(beta=beta)

    @staticmethod
    def _geometric_init(lin, dim_in, dim_out, is_last, d_in, radius_init):
        """
        Geometric initialization (Gropp et al. 2020, IGR, Sec 3.3 / App A):
        drives the network's initial zero-level-set close to a sphere of
        radius `radius_init`, which stabilizes early Eikonal optimization
        instead of starting from an arbitrary random SDF (paper Bab III:
        "Bobot jaringan diinisialisasi menggunakan geometric initialization
        dari Gropp et al.").
        """
        if is_last:
            mean = np.sqrt(np.pi) / np.sqrt(dim_in)
            nn.init.normal_(lin.weight, mean=mean, std=1e-5)
            nn.init.constant_(lin.bias, -radius_init)
        else:
            nn.init.constant_(lin.bias, 0.0)
            nn.init.normal_(lin.weight, mean=0.0, std=np.sqrt(2.0) / np.sqrt(dim_out))

    def forward(self, x):
        h = x
        for l in range(self.n_layers - 1):
            lin = getattr(self, f"lin{l}")
            h = lin(h)
            if l < self.n_layers - 2:
                h = self.activation(h)
        return h.squeeze(-1)  # (N,)

    def gradient(self, x):
        """
        grad_x f(x) via autograd -- required for the Eikonal constraint.
        Wrapped in `torch.enable_grad()` so this also works when called from
        inside an ambient `torch.no_grad()` block (e.g. quick sanity checks
        in the notebook) -- the returned y/grad are still detached from any
        OUTER graph in that case, only the local f(x)->grad computation runs.
        """
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            y = self.forward(x)
            grad = torch.autograd.grad(
                outputs=y, inputs=x,
                grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True, only_inputs=True,
            )[0]
        return y, grad
