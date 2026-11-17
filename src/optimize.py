"""Stage-1 SDF optimization.

This is a real optimization loop, not a placeholder. It:

* samples surface, uniform, near-surface and free-space points every step;
* computes the exact Stage-1 loss from :mod:`sdfgs.losses`;
* steps Adam with linear warm-up and cosine decay;
* clips gradients, detects non-finite losses/gradients, and fails loudly;
* writes atomic checkpoints that restore network, optimizer, iteration and
  random-generator state;
* records loss, learning-rate and gradient diagnostics.

Joint photometric refinement is implemented in :mod:`sdfgs.render` and
:class:`JointRefiner` (added after the renderer is validated).
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from .cameras import Camera
from .config import JointConfig, SDFConfig, SamplingConfig, TransmittanceConfig
from .gaussian import GaussianModel
from .io_utils import atomic_torch_save, ensure_dir, now_utc, write_json
from .losses import (coupling_loss, lr_scale, photometric_l1, psnr,
                     stage1_losses)
from .render import GaussianPatchRenderer, crop_observed, sample_patch_xy
from .sampling import (FreeSpaceRays, SurfaceSamples, sample_free_space,
                       sample_near_surface, sample_surface_batch,
                       sample_uniform_points)
from .sdf import SDFNetwork


class Stage1Trainer:
    """Train the SDF on Gaussian-derived surface and free-space samples."""

    def __init__(
        self,
        sdf: SDFNetwork,
        surface: SurfaceSamples,
        free: FreeSpaceRays,
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
        cfg: SDFConfig,
        sampling_cfg: SamplingConfig,
        out_dir: str | Path,
        device: str = "cpu",
        transmittance_cfg: Optional[TransmittanceConfig] = None,
        viz_callback: Optional[Callable[[SDFNetwork, int, Dict, List[Dict]],
                                        None]] = None,
        viz_every: int = 1000,
    ) -> None:
        if cfg.iterations <= 0:
            raise ValueError("cfg.iterations must be positive")
        if len(surface) == 0:
            raise ValueError("Stage-1 training requires surface samples")
        if len(free) == 0:
            raise ValueError(
                "Stage-1 training requires free-space rays with a T<=0.9 "
                "crossing; the free-space loss cannot be evaluated otherwise"
            )
        if bounds_min.shape != (3,) or bounds_max.shape != (3,):
            raise ValueError("bounds must be (3,) tensors")
        if not (torch.isfinite(bounds_min).all() and torch.isfinite(bounds_max).all()):
            raise ValueError("bounds must be finite")
        if bool((bounds_max <= bounds_min).any()):
            raise ValueError("bounds_max must exceed bounds_min on every axis")
        if cfg.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")

        self.sdf = sdf
        self.surface = surface
        self.free = free
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.cfg = cfg
        self.sampling_cfg = sampling_cfg
        self.tcfg = transmittance_cfg or TransmittanceConfig()
        self.device = torch.device(device)
        self.out_dir = ensure_dir(out_dir)
        self.viz_callback = viz_callback
        self.viz_every = int(viz_every)

        if next(sdf.parameters()).device != self.device:
            raise ValueError(
                f"SDF is on {next(sdf.parameters()).device}, trainer expects "
                f"{self.device}"
            )
        if surface.points.device != self.device:
            raise ValueError(
                f"surface samples are on {surface.points.device}, trainer "
                f"expects {self.device}"
            )
        if free.origins.device != self.device:
            raise ValueError(
                f"free rays are on {free.origins.device}, trainer expects "
                f"{self.device}"
            )

        self.batch_surface = int(sampling_cfg.surface_batch)
        self.batch_uniform = int(sampling_cfg.uniform_batch)
        self.batch_near = int(sampling_cfg.near_batch)
        self.batch_free = int(sampling_cfg.free_batch)
        for name, value in (
            ("surface_batch", self.batch_surface),
            ("uniform_batch", self.batch_uniform),
            ("near_batch", self.batch_near),
            ("free_batch", self.batch_free),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        gen_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.generator = torch.Generator(device=gen_device)
        self.generator.manual_seed(int(cfg.seed))

        self.optimizer = torch.optim.Adam(sdf.parameters(), lr=float(cfg.lr))
        self.iterations_completed = 0
        self.history: List[Dict] = []
        self.last_terms: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _set_lr(self, iteration: int) -> float:
        scale = lr_scale(
            iteration,
            self.cfg.iterations,
            warmup_fraction=self.cfg.warmup_fraction,
            min_ratio=self.cfg.cosine_min_ratio,
        )
        lr = float(self.cfg.lr) * scale
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _sample_batches(self):
        return (
            sample_surface_batch(self.surface, self.batch_surface, self.generator),
            sample_uniform_points(self.batch_uniform, self.bounds_min,
                                  self.bounds_max, self.generator),
            sample_near_surface(self.surface.points, self.batch_near,
                                self.sampling_cfg.jitter_sigma, self.generator),
            sample_free_space(self.free, self.batch_free, self.generator,
                              margin_fraction=self.tcfg.free_margin_fraction),
        )

    def _loss(self, batches):
        (pts_s, nrm_s, w_s), pts_u, pts_n, pts_f = batches
        f_s, g_s = self.sdf.gradient(pts_s)
        f_u, g_u = self.sdf.gradient(pts_u)
        f_near = self.sdf(pts_n)
        f_free = self.sdf(pts_f)
        terms = stage1_losses(
            f_s, g_s, nrm_s, w_s,
            f_u, g_u, f_near, f_free, self.cfg,
        )
        diagnostics = {
            "f_surface_abs_mean": float(f_s.detach().abs().mean()),
            "f_uniform_mean": float(f_u.detach().mean()),
            "f_free_mean": float(f_free.detach().mean()),
            "normal_weight_mean": float(w_s.detach().mean()),
        }
        return terms, diagnostics

    def _check_finite(self, terms: Dict[str, torch.Tensor], iteration: int) -> None:
        for name, value in terms.items():
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(
                    f"non-finite Stage-1 loss term {name!r} at iteration "
                    f"{iteration}: {value}"
                )

    def _record(self, iteration: int, terms: Dict[str, torch.Tensor],
                diagnostics: Dict[str, float], lr: float, grad_norm: float,
                param_norm: float) -> Dict:
        record = {"iteration": int(iteration), "lr": float(lr),
                  "grad_norm": float(grad_norm), "param_norm": float(param_norm),
                  "timestamp_utc": now_utc()}
        for name, value in terms.items():
            record[name] = float(value.detach())
        record.update(diagnostics)
        return record

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def train(self, iterations: Optional[int] = None) -> List[Dict]:
        """Run until ``iterations`` total steps have been completed.

        ``iterations`` defaults to ``cfg.iterations``. On a resumed run this
        only executes the remaining steps. Exactly one optimizer step is taken
        per uncompleted iteration; iterations are never silently skipped.
        """
        target = int(self.cfg.iterations if iterations is None else iterations)
        if target <= 0:
            raise ValueError("target iterations must be positive")
        if target < self.iterations_completed:
            raise ValueError(
                f"target ({target}) is below already completed steps "
                f"({self.iterations_completed})"
            )
        while self.iterations_completed < target:
            it = self.iterations_completed
            lr = self._set_lr(it)
            batches = self._sample_batches()
            terms, diagnostics = self._loss(batches)
            self._check_finite(terms, it)

            self.optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            for p in self.sdf.parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError(
                        f"non-finite gradient at iteration {it} for parameter "
                        f"shape {tuple(p.shape)}"
                    )
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.sdf.parameters(), float(self.cfg.grad_clip)))
            param_norm = float(torch.sqrt(sum(
                (p.detach() ** 2).sum() for p in self.sdf.parameters())))
            self.optimizer.step()

            self.last_terms = {k: float(v.detach()) for k, v in terms.items()}
            self.last_terms.update(diagnostics)
            self.iterations_completed += 1
            should_log = (
                it == 0
                or (it + 1) % max(1, int(self.cfg.log_every)) == 0
                or self.iterations_completed == target
            )
            if should_log:
                self.history.append(self._record(
                    it, terms, diagnostics, lr, grad_norm, param_norm))
            if (self.viz_callback is not None and self.viz_every > 0
                    and (it + 1) % self.viz_every == 0):
                self.viz_callback(self.sdf, self.iterations_completed,
                                  self.last_terms, self.history)
        return self.history

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def _payload(self) -> Dict:
        return {
            "stage": "sdf_stage1",
            "iteration": int(self.iterations_completed),
            "state_dict": self.sdf.state_dict(),
            "sdf_config": self.sdf.config_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator_state": self.generator.get_state(),
            "train_config": asdict(self.cfg),
            "sampling_config": asdict(self.sampling_cfg),
            "transmittance_config": asdict(self.tcfg),
            "history": self.history,
            "created_utc": now_utc(),
            "data_stats": {
                "n_surface": len(self.surface),
                "n_free_rays": len(self.free),
                "surface_batch": self.batch_surface,
                "uniform_batch": self.batch_uniform,
                "near_batch": self.batch_near,
                "free_batch": self.batch_free,
            },
        }

    def save_checkpoint(self, path: Optional[str | Path] = None) -> Path:
        if path is None:
            path = self.out_dir / "checkpoints" / (
                f"sdf_stage1_{self.iterations_completed:06d}.pt")
        path = Path(path)
        atomic_torch_save(self._payload(), path)
        latest = path.parent / "sdf_stage1_latest.pt"
        if latest != path:
            atomic_torch_save(self._payload(), latest)
        write_json(path.parent / "stage1_history.json", {
            "iterations_completed": self.iterations_completed,
            "history": self.history,
        })
        return path

    def load_checkpoint(self, path: str | Path) -> Dict:
        payload = torch.load(str(path), map_location=self.device,
                             weights_only=False)
        if payload.get("stage") != "sdf_stage1":
            raise ValueError(f"'{path}' is not a Stage-1 checkpoint")
        self.sdf.load_state_dict(payload["state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iterations_completed = int(payload["iteration"])
        self.history = list(payload.get("history", []))
        state = payload.get("generator_state")
        if state is not None:
            self.generator.set_state(state.cpu())
        return payload

    # ------------------------------------------------------------------ #
    @staticmethod
    def seed_all(seed: int) -> None:
        from .io_utils import set_seed
        set_seed(seed)


class JointRefiner:
    """Photometric joint refinement of the SDF and Gaussian primitives.

    The implementation follows the frozen protocol in ``METHODOLOGY.md``:

    * two Adam optimizers (SDF and Gaussians);
    * SDF LR ``1e-4``, Gaussian LRs at 0.1x the official 3DGS values;
    * a 32x32 photometric L1 patch;
    * the coupling term ``0.1 |f(mu)| + 0.05 (1 - |cos(grad f, n)|)`` on
      Gaussian centres inside ``|f(mu)| <= 0.02``;
    * every eighth camera held out from the photometric term.

    Every Gaussian parameter is optimized in raw space (logit opacity, log
    scale, unnormalized quaternion). Higher-order SH coefficients are not
    optimized because the patch renderer uses DC colour only; this limitation
    is explicit, not silently ignored.

    The photometric target is a **real observed image**. ``observed_images``
    must contain an image for every training camera; the constructor raises if
    one is missing. No image is ever fabricated. ``photometric_target_mode=
    "frozen_render"`` exists only to reproduce the zero-gradient defect and
    must be selected explicitly.
    """

    PARAM_NAMES = ("positions", "opacity_logit", "log_scales", "quats", "sh_dc")

    def __init__(
        self,
        sdf: SDFNetwork,
        gaussians: GaussianModel,
        cameras: List[Camera],
        observed_images: Dict[int, torch.Tensor],
        cfg: JointConfig,
        out_dir: str | Path,
        device: str = "cpu",
        scene_extent: float = 1.0,
        renderer: Optional[GaussianPatchRenderer] = None,
        heldout_cameras: Optional[List[int]] = None,
        viz_callback: Optional[Callable] = None,
        viz_every: int = 1000,
        photometric_target_mode: str = "observed",
    ) -> None:
        if not cfg.enabled:
            raise ValueError(
                "joint refinement is disabled in the configuration; set "
                "cfg.enabled=True only after real observed images are available"
            )
        if cfg.iterations <= 0:
            raise ValueError("joint iterations must be positive")
        if cfg.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if cfg.heldout_every <= 0:
            raise ValueError("heldout_every must be positive")
        if cfg.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if cfg.coupling_samples < 0:
            raise ValueError("coupling_samples must be non-negative")
        if scene_extent <= 0 or not math.isfinite(scene_extent):
            raise ValueError("scene_extent must be finite and positive")
        if photometric_target_mode not in {"observed", "frozen_render"}:
            raise ValueError(
                "photometric_target_mode must be 'observed' (default) or "
                "'frozen_render' (defect demonstration only)"
            )
        if not cameras:
            raise ValueError("joint refinement requires at least one camera")
        if len(gaussians) == 0:
            raise ValueError("joint refinement requires a non-empty Gaussian model")

        self.sdf = sdf
        self.gaussians = gaussians
        self.cameras = list(cameras)
        self.cfg = cfg
        self.device = torch.device(device)
        self.out_dir = ensure_dir(out_dir)
        self.scene_extent = float(scene_extent)
        self.renderer = renderer or GaussianPatchRenderer()
        self.viz_callback = viz_callback
        self.viz_every = int(viz_every)
        self.photometric_target_mode = photometric_target_mode

        if next(sdf.parameters()).device != self.device:
            raise ValueError(
                f"SDF is on {next(sdf.parameters()).device}, expected {self.device}"
            )
        if gaussians.positions.device != self.device:
            raise ValueError(
                f"Gaussians are on {gaussians.positions.device}, expected "
                f"{self.device}"
            )
        self._validate_gaussians()

        # Observed images: validate shape, dtype, range and finiteness. We do
        # not silently convert 0..255 images or grayscale images.
        self.observed_images: Dict[int, torch.Tensor] = {}
        for key, value in observed_images.items():
            idx = int(key)
            if idx < 0 or idx >= len(self.cameras):
                raise ValueError(f"observed image key {idx} has no camera")
            image = torch.as_tensor(value)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"observed image {idx} must be (H,W,3), got "
                    f"{tuple(image.shape)}"
                )
            if not image.is_floating_point():
                raise TypeError(
                    f"observed image {idx} must be floating point in [0,1], "
                    f"got {image.dtype}"
                )
            if not bool(torch.isfinite(image).all()):
                raise ValueError(f"observed image {idx} contains non-finite values")
            if float(image.min()) < -1e-3 or float(image.max()) > 1.0 + 1e-3:
                raise ValueError(
                    f"observed image {idx} must be in [0,1]; values "
                    f"[{float(image.min()):.3f}, {float(image.max()):.3f}] "
                    "would silently mismatch the renderer"
                )
            self.observed_images[idx] = image.detach().clone()

        # Every camera used photometrically must have a real image.
        for idx, cam in enumerate(self.cameras):
            image = self.observed_images.get(idx)
            if image is not None and tuple(image.shape[:2]) != (cam.H, cam.W):
                raise ValueError(
                    f"observed image {idx} is {tuple(image.shape[:2])}, camera "
                    f"is {(cam.H, cam.W)}"
                )
            if cfg.patch_size > cam.H or cfg.patch_size > cam.W:
                raise ValueError(
                    f"patch_size {cfg.patch_size} exceeds camera {idx} size "
                    f"{(cam.H, cam.W)}"
                )

        if heldout_cameras is None:
            heldout = list(range(0, len(self.cameras), int(cfg.heldout_every)))
        else:
            heldout = [int(i) for i in heldout_cameras]
        if len(set(heldout)) != len(heldout):
            raise ValueError("heldout_cameras contains duplicates")
        for idx in heldout:
            if idx < 0 or idx >= len(self.cameras):
                raise ValueError(f"held-out camera {idx} is out of range")
        self.heldout_cameras = sorted(heldout)
        self.train_cameras = [i for i in range(len(self.cameras))
                              if i not in set(self.heldout_cameras)]
        if not self.train_cameras:
            raise ValueError("every camera is held out; nothing to optimize")
        missing = [i for i in self.train_cameras if i not in self.observed_images]
        if missing:
            raise FileNotFoundError(
                "joint refinement requires real observed images for every "
                f"training camera; missing {missing}. No placeholder or "
                "synthetic target is substituted."
            )

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(cfg.seed))

        self.sdf_optimizer = torch.optim.Adam(
            sdf.parameters(), lr=float(cfg.sdf_lr))
        self._gaussian_base_lrs: List[float] = []
        groups = []
        lrs = {
            "positions": cfg.position_lr * self.scene_extent,
            "opacity_logit": cfg.opacity_lr,
            "sh_dc": cfg.sh_lr,
            "log_scales": cfg.scale_lr,
            "quats": cfg.rotation_lr,
        }
        for name, param in gaussians.trainable_parameters().items():
            base = float(lrs[name])
            self._gaussian_base_lrs.append(base)
            groups.append({"params": [param], "lr": base, "base_lr": base,
                           "name": name})
        self.gaussian_optimizer = torch.optim.Adam(groups, lr=1.0)

        self.iterations_completed = 0
        self.history: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _validate_gaussians(self) -> None:
        params = self.gaussians.trainable_parameters()
        missing = [name for name in self.PARAM_NAMES if name not in params]
        if missing:
            raise ValueError(
                "Gaussian model is not trainable; missing parameters "
                f"{missing}. Call make_trainable() before joint refinement."
            )
        with torch.no_grad():
            for name, p in params.items():
                if not bool(torch.isfinite(p).all()):
                    raise ValueError(f"Gaussian parameter {name!r} is non-finite")
            if not bool(torch.isfinite(self.gaussians.positions).all()):
                raise ValueError("Gaussian positions contain non-finite values")

    def _set_lrs(self, iteration: int) -> float:
        scale = lr_scale(
            iteration,
            self.cfg.iterations,
            warmup_fraction=self.cfg.warmup_fraction,
            min_ratio=self.cfg.cosine_min_ratio,
        )
        for group in self.sdf_optimizer.param_groups:
            group["lr"] = float(self.cfg.sdf_lr) * scale
        for group, base in zip(self.gaussian_optimizer.param_groups,
                               self._gaussian_base_lrs):
            group["lr"] = base * scale
        return scale

    # ------------------------------------------------------------------ #
    # Data and loss
    # ------------------------------------------------------------------ #
    def _photometric_target(self, camera_index: int,
                            rendered: torch.Tensor,
                            patch_xy: tuple) -> torch.Tensor:
        if self.photometric_target_mode == "frozen_render":
            # Defect-demonstration baseline only. The target is the current
            # render with its graph detached, so the L1 gradient is zero.
            return rendered.detach().clone()
        image = self.observed_images.get(camera_index)
        if image is None:
            raise FileNotFoundError(
                f"no real observed image for camera {camera_index}; refusing "
                "to fabricate a photometric target"
            )
        target = crop_observed(image, rendered.shape[0], patch_xy)
        target = target.to(device=rendered.device, dtype=rendered.dtype)
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError(
                f"observed target patch for camera {camera_index} contains "
                "non-finite values"
            )
        if target.shape != rendered.shape:
            raise ValueError(
                f"observed target shape {tuple(target.shape)} does not match "
                f"rendered patch {tuple(rendered.shape)}"
            )
        return target

    def _coupling_term(self):
        if int(self.cfg.coupling_samples) == 0:
            # Explicit ablation: no centres sampled, but keep the zero tensor
            # connected to the SDF graph so backward() remains well-defined.
            zero = sum(p.sum() for p in self.sdf.parameters()) * 0.0
            return zero, {"n_coupled": 0, "coupling_zero": 0.0,
                          "coupling_normal": 0.0}
        n = min(int(self.cfg.coupling_samples), len(self.gaussians))
        if n == len(self.gaussians):
            idx = torch.arange(len(self.gaussians), dtype=torch.long)
        else:
            idx = torch.randint(0, len(self.gaussians), (n,),
                                generator=self.generator)
        idx = idx.to(self.device)
        centres = self.gaussians.positions[idx]
        f_mu, grad_mu = self.sdf.gradient(centres)
        normals = self.gaussians.estimated_normals()[idx]
        loss, metrics = coupling_loss(
            f_mu, grad_mu, normals,
            epsilon=float(self.cfg.coupling_epsilon),
            zero_weight=float(self.cfg.coupling_zero_weight),
            normal_weight=float(self.cfg.coupling_normal_weight),
        )
        return loss, metrics

    @torch.no_grad()
    def _heldout_photometric(self) -> Optional[float]:
        """Monitoring-only held-out photometric error (never model selection)."""
        values = []
        for idx in self.heldout_cameras:
            image = self.observed_images.get(idx)
            if image is None:
                continue
            cam = self.cameras[idx]
            ps = int(self.cfg.patch_size)
            x, y = sample_patch_xy(cam, ps, training=False)
            rendered = self.renderer.render(self.gaussians, cam, ps, (x, y))
            target = crop_observed(image, ps, (x, y)).to(
                device=rendered.device, dtype=rendered.dtype)
            values.append(float(photometric_l1(rendered, target)))
        if not values:
            return None
        return float(sum(values) / len(values))

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def train(self, iterations: Optional[int] = None) -> List[Dict]:
        target_total = int(self.cfg.iterations if iterations is None
                           else iterations)
        if target_total <= 0:
            raise ValueError("target iterations must be positive")
        if target_total < self.iterations_completed:
            raise ValueError(
                f"target ({target_total}) is below completed steps "
                f"({self.iterations_completed})"
            )
        while self.iterations_completed < target_total:
            it = self.iterations_completed
            scale = self._set_lrs(it)
            cam_idx = self.train_cameras[
                int(torch.randint(0, len(self.train_cameras), (1,),
                                  generator=self.generator).item())]
            cam = self.cameras[cam_idx]
            patch_x, patch_y = sample_patch_xy(
                cam, int(self.cfg.patch_size), self.generator, training=True)
            rendered, render_diag = self.renderer.render(
                self.gaussians, cam, int(self.cfg.patch_size),
                (patch_x, patch_y), return_diagnostics=True)
            if rendered.shape != (self.cfg.patch_size, self.cfg.patch_size, 3):
                raise ValueError(
                    f"renderer returned {tuple(rendered.shape)}, expected "
                    f"{(self.cfg.patch_size, self.cfg.patch_size, 3)}"
                )
            if not bool(torch.isfinite(rendered).all()):
                raise FloatingPointError(
                    f"non-finite rendered patch at iteration {it}")
            target = self._photometric_target(cam_idx, rendered,
                                              (patch_x, patch_y))
            if not bool(torch.isfinite(target).all()):
                raise FloatingPointError(
                    f"non-finite photometric target at iteration {it}")
            photo = photometric_l1(rendered, target)
            coupling, coupling_metrics = self._coupling_term()
            total = float(self.cfg.photometric_weight) * photo + coupling
            for name, value in (("photometric", photo), ("coupling", coupling),
                                ("total", total)):
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(
                        f"non-finite joint loss term {name!r} at iteration {it}: "
                        f"{value}")

            self.sdf_optimizer.zero_grad(set_to_none=True)
            self.gaussian_optimizer.zero_grad(set_to_none=True)
            total.backward()
            all_params = list(self.sdf.parameters()) + list(
                self.gaussians.parameters())
            for p in all_params:
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError(
                        f"non-finite joint gradient at iteration {it} for "
                        f"parameter shape {tuple(p.shape)}"
                    )
            sdf_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.sdf.parameters(), float(self.cfg.grad_clip)))
            gaussian_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.gaussians.parameters(), float(self.cfg.grad_clip)))
            self.sdf_optimizer.step()
            self.gaussian_optimizer.step()
            self.iterations_completed += 1

            should_log = (
                it == 0
                or (it + 1) % max(1, int(self.cfg.log_every)) == 0
                or self.iterations_completed == target_total
            )
            if should_log:
                psnr_value = psnr(rendered.detach(), target.detach())
                record = {
                    "iteration": int(it),
                    "camera_index": int(cam_idx),
                    "patch_x": int(patch_x),
                    "patch_y": int(patch_y),
                    "lr_scale": float(scale),
                    "photometric": float(photo.detach()),
                    "coupling": float(coupling.detach()),
                    "total": float(total.detach()),
                    "psnr": float(psnr_value) if math.isfinite(psnr_value) else None,
                    "sdf_grad_norm": sdf_grad_norm,
                    "gaussian_grad_norm": gaussian_grad_norm,
                    "n_selected": int(render_diag.n_selected),
                    "photometric_target_mode": self.photometric_target_mode,
                    "timestamp_utc": now_utc(),
                }
                record.update(coupling_metrics)
                heldout = self._heldout_photometric()
                if heldout is not None:
                    record["heldout_photometric"] = heldout
                self.history.append(record)
                if (self.viz_callback is not None and self.viz_every > 0
                        and (it + 1) % self.viz_every == 0):
                    self.viz_callback(self, self.iterations_completed, record,
                                      self.history)
        return self.history

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def _payload(self) -> Dict:
        return {
            "stage": "sdfgs_joint",
            "iteration": int(self.iterations_completed),
            "sdf_state_dict": self.sdf.state_dict(),
            "sdf_config": self.sdf.config_dict(),
            "gaussian_state": self.gaussians.state_dict(),
            "sdf_optimizer": self.sdf_optimizer.state_dict(),
            "gaussian_optimizer": self.gaussian_optimizer.state_dict(),
            "generator_state": self.generator.get_state(),
            "history": self.history,
            "joint_config": asdict(self.cfg),
            "train_cameras": list(self.train_cameras),
            "heldout_cameras": list(self.heldout_cameras),
            "photometric_target_mode": self.photometric_target_mode,
            "scene_extent": self.scene_extent,
            "created_utc": now_utc(),
        }

    def save_checkpoint(self, path: Optional[str | Path] = None) -> Path:
        if path is None:
            path = self.out_dir / "checkpoints" / (
                f"joint_{self.iterations_completed:06d}.pt")
        path = Path(path)
        atomic_torch_save(self._payload(), path)
        latest = path.parent / "joint_latest.pt"
        if latest != path:
            atomic_torch_save(self._payload(), latest)
        write_json(path.parent / "joint_history.json", {
            "iterations_completed": self.iterations_completed,
            "history": self.history,
        })
        return path

    def load_checkpoint(self, path: str | Path) -> Dict:
        payload = torch.load(str(path), map_location=self.device,
                             weights_only=False)
        if payload.get("stage") != "sdfgs_joint":
            raise ValueError(f"'{path}' is not a joint-refinement checkpoint")
        if payload.get("photometric_target_mode") != self.photometric_target_mode:
            raise ValueError(
                "checkpoint photometric_target_mode does not match this refiner"
            )
        if list(payload.get("train_cameras", [])) != list(self.train_cameras):
            raise ValueError("checkpoint training-camera split does not match")
        if list(payload.get("heldout_cameras", [])) != list(self.heldout_cameras):
            raise ValueError("checkpoint held-out split does not match")

        self.sdf.load_state_dict(payload["sdf_state_dict"], strict=True)
        g_state = payload["gaussian_state"]
        for name in self.PARAM_NAMES:
            param = getattr(self.gaussians, name)
            if not isinstance(param, torch.nn.Parameter):
                raise ValueError(
                    f"Gaussian parameter {name!r} is not trainable during resume")
            param.data.copy_(g_state[name].to(device=param.device,
                                              dtype=param.dtype))
        if (self.gaussians.sh_rest is not None
                and g_state.get("sh_rest") is not None):
            self.gaussians.sh_rest = g_state["sh_rest"].to(
                device=self.gaussians.positions.device)
        self.sdf_optimizer.load_state_dict(payload["sdf_optimizer"])
        self.gaussian_optimizer.load_state_dict(payload["gaussian_optimizer"])
        self.iterations_completed = int(payload["iteration"])
        self.history = list(payload.get("history", []))
        state = payload.get("generator_state")
        if state is not None:
            self.generator.set_state(state.cpu())
        return payload

    def save_refined_gaussians(self, path: str | Path) -> Path:
        return self.gaussians.detached().save_ply(path)

    # ------------------------------------------------------------------ #
    # Frozen-target defect diagnostic through the real renderer
    # ------------------------------------------------------------------ #
    def diagnose_frozen_render_zero_gradient(
        self,
        camera_index: int = 0,
        patch_size: Optional[int] = None,
        patch_xy: Optional[tuple] = None,
    ) -> Dict:
        """Reproduce the zero photometric gradient with the real renderer.

        A detached clone of the Gaussians is rendered, the render is frozen as
        the target, and the L1 loss is differentiated. Because the render and
        target are identical, ``d|x-y|/dx = sign(0) = 0`` for every element,
        so every Gaussian gradient is exactly zero. This is the defect of the
        frozen-self-target formulation, preserved here as a baseline.
        """
        if camera_index < 0 or camera_index >= len(self.cameras):
            raise ValueError("camera_index out of range")
        cam = self.cameras[camera_index]
        ps = int(self.cfg.patch_size if patch_size is None else patch_size)
        if ps <= 0 or ps > cam.H or ps > cam.W:
            raise ValueError("invalid diagnostic patch size")
        if patch_xy is None:
            patch_xy = sample_patch_xy(cam, ps, training=False)
        clone = self.gaussians.detached().make_trainable()
        rendered = self.renderer.render(clone, cam, ps, patch_xy)
        frozen_target = rendered.detach().clone()
        loss = photometric_l1(rendered, frozen_target)
        loss.backward()
        per_param = {}
        for name, param in clone.trainable_parameters().items():
            per_param[name] = (float(param.grad.norm())
                               if param.grad is not None else 0.0)
        total = math.sqrt(sum(v * v for v in per_param.values()))
        return {
            "loss": float(loss.detach()),
            "per_parameter_grad_norm": per_param,
            "total_grad_norm": total,
            "patch_size": ps,
            "patch_xy": (int(patch_xy[0]), int(patch_xy[1])),
            "camera_index": int(camera_index),
        }


def diagnose_frozen_target_gradient(
    render_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
) -> Dict[str, float]:
    """Reproduce the frozen-target zero-gradient defect.

    ``render_fn`` must be a differentiable function of ``parameters``. The
    target is a detached copy of the current render, i.e. the frozen-self-target
    formulation. Returns the L1 loss and gradient norm. Because
    ``d|x-y|/dx = sign(x-y)`` and ``x == y`` at the frozen point, the gradient
    is exactly zero almost everywhere.
    """
    parameters = parameters.detach().clone().requires_grad_(True)
    rendered = render_fn(parameters)
    frozen_target = rendered.detach().clone()
    loss = photometric_l1(rendered, frozen_target)
    loss.backward()
    grad = parameters.grad
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(grad.norm()) if grad is not None else 0.0,
        "n_parameters": int(parameters.numel()),
    }
