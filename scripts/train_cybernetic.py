"""Experiment 3: Cybernetic Bridge Training.

The lattice IS the steersman.

Previous experiments:
  Exp 2:   Open loop — train, checkpoint, analyze post-hoc
  Exp 2.5: Geometric data — still open loop, fixed schedule
  Exp 2.6: Contrastive loss — directional pressure but static scheduling

This experiment closes the loop. Every N steps:
  1. SENSOR: Extract bridge matrices from all adapters
  2. OSCILLOSCOPE: Compute spectral diagnostics (Fiedler, co/cross ratio,
     spectral gap, deviation, eigenvalue distribution)
  3. STEERSMAN: Decide parameter adjustments based on measurement history
     and derivative trends (PID-like control)
  4. ACTUATOR: Adjust learning rate, contrastive weight, spectral
     regularization weight, bridge learning rate
  5. SYSTEM: Training continues with adjusted parameters -> back to SENSOR

Two new loss components:
  - Spectral regularization: differentiable Fiedler proxy drives algebraic
    connectivity directly as a training objective (not post-hoc measurement)
  - Adaptive contrastive: same as Exp 2.6 but weight determined by the
    Steersman based on co/cross ratio trajectory

The Steersman (kybernetes) implements three control laws:
  1. CONNECTIVITY: If Fiedler declining → increase spectral reg
  2. DIRECTIONALITY: If co/cross stagnant → increase contrastive weight
  3. STABILITY: If deviation growing too fast → dampen bridge LR

Deployment:
  # Quick local test (6 minutes)
  python scripts/train_cybernetic.py --max-steps 500 --feedback-interval 50

  # Full run on RunPod A100
  python scripts/train_cybernetic.py --max-steps 20000 --output results/exp3

  # Extended convergence study
  python scripts/train_cybernetic.py --max-steps 50000 --feedback-interval 200

Usage:
  TASUMER MAF = 1093 = kybernetes (exact). The lattice steers itself.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rhombic.nn.rhombi_lora import RhombiLoRALinear, EmanationBridge
from rhombic.nn.topology import direction_pair_coupling
from rhombic.spectral import (
    weighted_laplacian,
    fiedler_value,
    spectral_gap,
    spectrum,
    eigenvalue_multiplicity_pattern,
)

# Reuse from Exp 2
from train_exp2_scale import (
    ExperimentConfig,
    AlpacaDataset,
    inject_lora,
    collect_metrics,
    gradient_effective_rank,
    coplanar_crossplanar_ratio,
    evaluate,
)

# Co-planar pairs from Exp 2.6
from train_contrastive_bridge import (
    _compute_pair_indices,
    _compute_wrong_label_pairs,
    _compute_resonance_pairs,
    contrastive_bridge_loss,
    circular_resonance_loss,
)


# ── Differentiable spectral diagnostics ──────────────────────────────


def differentiable_fiedler(bridge: torch.Tensor) -> torch.Tensor:
    """Compute Fiedler value of bridge matrix as a differentiable operation.

    Constructs the weighted Laplacian from |B[i,j]| off-diagonal elements,
    then returns the second-smallest eigenvalue via torch.linalg.eigvalsh
    (which IS differentiable through autograd).

    Parameters
    ----------
    bridge : (C, C) tensor — the bridge matrix

    Returns
    -------
    Scalar tensor — the algebraic connectivity (Fiedler value)
    """
    C = bridge.shape[0]
    # Absolute off-diagonal weights
    W = bridge.abs()
    W = W - torch.diag(torch.diag(W))  # zero diagonal

    # Weighted Laplacian: L[i,i] = sum of row, L[i,j] = -W[i,j]
    L = -W
    L = L + torch.diag(W.sum(dim=1))

    # Eigenvalues (ascending) — differentiable
    eigenvalues = torch.linalg.eigvalsh(L)
    return eigenvalues[1]  # Second smallest = Fiedler


def spectral_reg_loss(
    injected: dict[str, RhombiLoRALinear],
    target_fiedler: float = 0.5,
) -> torch.Tensor:
    """Spectral regularization: drive Fiedler value toward target.

    L_spectral = mean((fiedler_i - target)^2) for all adapters.

    The loss is minimized when every bridge's algebraic connectivity
    equals the target. The Steersman adjusts the target dynamically.
    """
    device = next(iter(injected.values())).effective_bridge.device
    losses = []

    for lora in injected.values():
        B = lora.effective_bridge
        if B.shape[0] < 2:
            continue
        fv = differentiable_fiedler(B)
        losses.append((fv - target_fiedler) ** 2)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# ── The Steersman (kybernetes) ───────────────────────────────────────

# ── C6b stability-detector version + "settled" (genuine-convergence) constants ─
#
# C6b DEFECT (see docs/C6B_FIX_NOTE.md, docs/PAPER4_EXPOSURE_CLASSIFICATION.md §2):
# Control Law 1 (CONNECTIVITY) and Control Law 3 (STABILITY) declared "STABLE"
# on a short-window trend *deadband alone*. Because the deadband bounds the
# per-sample slope, a slow *monotonic* drift whose slope stays just inside the
# band is declared STABLE indefinitely while the governed metric moves the full
# width of its attractor. Empirically (WL-001, results/channel-ablation/WL-001):
# deviation climbed 0.00 -> 2.11 over a 10K-step run with deviation_trend never
# exceeding 0.04841 (< the 0.05 FAST-GROWTH threshold), so bridge_lr_scale stayed
# pinned at 1.0 and STABLE was declared 101/101 samples — the metric was still
# materially moving at every "STABLE" call.
#
# ROOT CAUSE: STABLE was keyed to "trend small" without ALSO requiring "the level
# has actually settled". A small instantaneous slope is *necessary but not
# sufficient* for convergence: sustained below-threshold slope integrates to
# large net displacement. The fix (detector v2) additionally requires the metric
# to be settled over a longer window before STABLE is declared.
#
# Bump this tag whenever the STABLE-declaration logic changes so every run's
# config.json self-documents which detector governed it. PAST runs used v1
# (defective); any FUTURE run records v2.
STEERSMAN_DETECTOR_VERSION = "v2-2026-07-05"

# The longer window (in feedback samples) over which "settled" is judged. Must be
# >= window_size (the short trend window). At a 100-step feedback interval this is
# ~1500 steps — long enough that a below-deadband slope integrates to a
# detectable net displacement.
SETTLE_WINDOW = 15
# Max |net drift| / level over SETTLE_WINDOW for the metric to count as settled.
# Net drift = OLS slope over the settle window x (window-1) = the net linear
# displacement. This is the term that catches slow monotonic creep.
SETTLE_REL_DRIFT = 0.10
# Max std / level over SETTLE_WINDOW for the metric to count as settled. Catches
# noisy, non-monotonic non-stationarity that the drift term alone would miss.
SETTLE_REL_SPREAD = 0.10
# Genuine-convergence HARD CAP: a peak-to-peak band this tight is settled
# regardless of the relative ratios. Guards near-zero metrics (e.g. a collapsed
# Fiedler ~1e-5) whose negligible absolute wiggle would otherwise blow up the
# relative spread and prevent STABLE from ever being declared (Task 4 guard).
SETTLE_ABS_BAND = 1e-3
# Scale floor to avoid division by zero when the level is ~0.
SETTLE_ABS_EPS = 1e-6


@dataclass
class SteersmanState:
    """Snapshot of what the Steersman observes and decides."""
    step: int
    # Measurements (sensor → oscilloscope)
    fiedler_mean: float
    fiedler_std: float
    co_cross_ratio: Optional[float]
    spectral_gap_mean: float
    deviation_mean: float
    deviation_std: float
    grad_eff_rank: float
    eigenvalue_pattern: list  # multiplicity pattern from first adapter
    # Decisions (steersman → actuator)
    lr_scale: float          # multiplier on base LR
    contrastive_weight: float
    spectral_weight: float
    spectral_target: float   # target Fiedler value
    bridge_lr_scale: float   # multiplier on bridge param LR
    # Derivative estimates
    fiedler_trend: float     # slope of Fiedler over last K measurements
    co_cross_trend: float    # slope of co/cross over last K measurements
    deviation_trend: float   # slope of deviation over last K measurements
    # Diagnostic
    control_signals: dict = field(default_factory=dict)


class Steersman:
    """Cybernetic controller for RhombiLoRA bridge training.

    Implements three control laws based on spectral measurement history:
    1. CONNECTIVITY: Fiedler value declining → increase spectral reg
    2. DIRECTIONALITY: co/cross ratio stagnant → increase contrastive weight
    3. STABILITY: deviation growing too fast → dampen bridge LR

    The Steersman maintains a sliding window of measurements and computes
    linear trends (derivatives) over that window. Control outputs are
    bounded to prevent oscillation.
    """

    def __init__(
        self,
        # Control law parameters
        base_contrastive_weight: float = 0.1,
        base_spectral_weight: float = 0.05,
        initial_spectral_target: float = 0.1,
        # Bounds
        max_contrastive_weight: float = 0.5,
        max_spectral_weight: float = 0.2,
        max_bridge_lr_scale: float = 3.0,
        min_bridge_lr_scale: float = 0.1,
        # Sensitivity
        fiedler_decline_threshold: float = -0.001,
        co_cross_stagnation_band: float = 0.02,
        deviation_rate_threshold: float = 0.05,
        # Spectral target adaptation
        spectral_target_tracking_rate: float = 0.1,
        # Window size for trend estimation (short window — slope deadband)
        window_size: int = 5,
        # C6b detector v2: genuine-convergence ("settled") parameters. STABLE is
        # declared only when the short-window trend is inside its deadband AND the
        # metric is settled over these longer-window criteria.
        settle_window: int = SETTLE_WINDOW,
        settle_rel_drift: float = SETTLE_REL_DRIFT,
        settle_rel_spread: float = SETTLE_REL_SPREAD,
        settle_abs_band: float = SETTLE_ABS_BAND,
        settle_abs_eps: float = SETTLE_ABS_EPS,
        # Fixed contrastive weight (bypasses Control Law 2)
        fixed_contrastive_weight: Optional[float] = None,
    ):
        self.base_contrastive = base_contrastive_weight
        self.base_spectral = base_spectral_weight
        self.spectral_target = initial_spectral_target

        self.max_contrastive = max_contrastive_weight
        self.max_spectral = max_spectral_weight
        self.max_bridge_lr = max_bridge_lr_scale
        self.min_bridge_lr = min_bridge_lr_scale

        self.fiedler_decline_thresh = fiedler_decline_threshold
        self.co_cross_stagnation = co_cross_stagnation_band
        self.deviation_rate_thresh = deviation_rate_threshold
        self.target_tracking_rate = spectral_target_tracking_rate

        self.window_size = window_size
        # C6b detector v2 settling parameters (see module-level constants).
        self.settle_window = max(int(settle_window), window_size)
        self.settle_rel_drift = settle_rel_drift
        self.settle_rel_spread = settle_rel_spread
        self.settle_abs_band = settle_abs_band
        self.settle_abs_eps = settle_abs_eps
        self.detector_version = STEERSMAN_DETECTOR_VERSION
        self.fixed_contrastive = fixed_contrastive_weight
        self.history: list[SteersmanState] = []

        # Current actuator outputs
        self._lr_scale = 1.0
        self._contrastive_weight = (
            fixed_contrastive_weight if fixed_contrastive_weight is not None
            else base_contrastive_weight
        )
        self._spectral_weight = base_spectral_weight
        self._bridge_lr_scale = 1.0

    def _trend(self, values: list[float]) -> float:
        """Linear regression slope over the last window_size values."""
        # Filter out None, NaN, and inf
        clean = [v for v in values if v is not None and np.isfinite(v)]
        if len(clean) < 2:
            return 0.0
        v = clean[-self.window_size:]
        n = len(v)
        x = np.arange(n, dtype=np.float64)
        y = np.array(v, dtype=np.float64)
        # OLS slope
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom < 1e-12:
            return 0.0
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        return float(slope) if np.isfinite(slope) else 0.0

    def _is_settled(self, series: list[float]) -> tuple[bool, float, bool]:
        """Genuine-convergence ("settled") test — the C6b detector-v2 gate.

        Trend-flatness over the short window is necessary but NOT sufficient for
        convergence. A slow monotonic creep keeps the per-sample slope inside the
        STABLE deadband while the *level* moves materially across the run (the
        C6b defect: WL-001 deviation climbed 0.00 -> 2.11 with every sample's
        slope inside the deadband, declared STABLE 101/101 times). A metric is
        "settled" only when, over the longer ``settle_window``, it neither DRIFTS
        (small net displacement relative to its level) nor SPREADS (low relative
        std). The drift term is what catches the slow monotonic creep the short
        deadband misses.

        Returns
        -------
        (settled, net_drift, confident)
            settled   : True only when genuinely converged over the settle window.
            net_drift : signed net linear displacement over the window
                        (> 0 rising, < 0 falling). Direction lets the caller
                        respond to a residual drift instead of sitting idle.
            confident : False when there is not yet enough history to judge — the
                        caller must then neither declare STABLE nor actuate on a
                        drift it cannot yet confirm.
        """
        clean = [v for v in series if v is not None and np.isfinite(v)]
        if len(clean) < self.settle_window:
            # Not enough samples to distinguish "settled" from "creeping slowly".
            return False, 0.0, False

        w = np.asarray(clean[-self.settle_window:], dtype=np.float64)
        n = len(w)
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum()
        slope = (
            ((x - x_mean) * (w - w.mean())).sum() / denom
            if denom > 1e-12 else 0.0
        )
        net_drift = float(slope * (n - 1))          # net linear displacement
        level = float(w.mean())
        band = float(w.max() - w.min())             # peak-to-peak movement
        std = float(w.std())
        scale = max(abs(level), self.settle_abs_eps)

        # Genuine-convergence HARD CAP (Task 4 regression guard): negligible
        # absolute movement is settled regardless of the relative ratios. Without
        # this a near-zero metric (collapsed Fiedler ~1e-5) whose tiny wiggle is
        # large *relative* to its level would never be declared STABLE.
        if band <= self.settle_abs_band:
            return True, net_drift, True

        rel_drift = abs(net_drift) / scale
        rel_spread = std / scale
        settled = (
            rel_drift <= self.settle_rel_drift
            and rel_spread <= self.settle_rel_spread
        )
        return bool(settled), net_drift, True

    def _connectivity_law(self, fiedler_series: list[float]) -> tuple[str, float]:
        """Control Law 1 (CONNECTIVITY) decision. Mutates ``self._spectral_weight``.

        Governs the spectral-regularization weight from the Fiedler trend. STABLE
        (detector v2) requires BOTH a small short-window trend AND a genuinely
        settled level; a below-deadband monotonic drift is no longer mislabelled
        STABLE — it is answered with the SAME proportional gains as the DECLINING
        / IMPROVING branches, so the controller stops sitting idle while
        connectivity creeps across its attractor.

        Returns ``(signal_string, fiedler_trend)``.
        """
        fiedler_trend = self._trend(fiedler_series)

        # If Fiedler is declining, boost spectral regularization.
        if fiedler_trend < self.fiedler_decline_thresh:
            boost = min(
                abs(fiedler_trend) * 10.0,  # proportional gain
                self.max_spectral - self._spectral_weight,
            )
            self._spectral_weight = min(
                self._spectral_weight + boost,
                self.max_spectral,
            )
            return (
                f"DECLINING (trend={fiedler_trend:.5f}), spec_reg +{boost:.4f}",
                fiedler_trend,
            )
        elif fiedler_trend > 0.001:
            # Fiedler growing — ease off spectral pressure to let it find its way.
            decay = min(0.01, self._spectral_weight * 0.1)
            self._spectral_weight = max(
                self._spectral_weight - decay,
                self.base_spectral * 0.5,
            )
            return (
                f"IMPROVING (trend={fiedler_trend:.5f}), spec_reg -{decay:.4f}",
                fiedler_trend,
            )

        # Short-window trend is inside the deadband. C6b v2: only STABLE if the
        # Fiedler level is GENUINELY settled over the longer window.
        settled, net_drift, confident = self._is_settled(fiedler_series)
        if settled:
            return f"STABLE (trend={fiedler_trend:.5f}, settled)", fiedler_trend
        if not confident:
            # Not enough history to confirm convergence; hold and keep collecting
            # (same actuator behaviour as the pre-fix idle branch, honest label).
            return (
                f"SETTLING (trend={fiedler_trend:.5f}, collecting)",
                fiedler_trend,
            )
        # Confirmed slow drift the deadband missed — respond in the drift
        # direction with the SAME gain as the branches above (long-window slope).
        drift_slope = net_drift / (self.settle_window - 1)
        if net_drift < 0.0:
            boost = min(
                abs(drift_slope) * 10.0,
                self.max_spectral - self._spectral_weight,
            )
            self._spectral_weight = min(
                self._spectral_weight + boost,
                self.max_spectral,
            )
            return (
                f"DRIFTING-DOWN (trend={fiedler_trend:.5f}, "
                f"net={net_drift:.5f}), spec_reg +{boost:.4f}",
                fiedler_trend,
            )
        decay = min(0.01, self._spectral_weight * 0.1)
        self._spectral_weight = max(
            self._spectral_weight - decay,
            self.base_spectral * 0.5,
        )
        return (
            f"DRIFTING-UP (trend={fiedler_trend:.5f}, "
            f"net={net_drift:.5f}), spec_reg -{decay:.4f}",
            fiedler_trend,
        )

    def _stability_law(self, deviation_series: list[float]) -> tuple[str, float]:
        """Control Law 3 (STABILITY) decision. Mutates ``self._bridge_lr_scale``.

        Governs the bridge-LR scale from the deviation trend. STABLE (detector v2)
        requires BOTH a small short-window trend AND a genuinely settled level. A
        below-deadband monotonic climb — the exact WL-001 defect (deviation ->
        2.11 with bridge_lr_scale pinned at 1.0) — is no longer mislabelled
        STABLE; it is dampened with the SAME gain as the FAST-GROWTH branch.

        Returns ``(signal_string, deviation_trend)``.
        """
        deviation_trend = self._trend(deviation_series)

        # If deviation growing too fast, dampen bridge learning.
        if deviation_trend > self.deviation_rate_thresh:
            dampen = max(0.8, 1.0 - deviation_trend)
            self._bridge_lr_scale = max(
                self._bridge_lr_scale * dampen,
                self.min_bridge_lr,
            )
            return (
                f"FAST GROWTH (trend={deviation_trend:.5f}), "
                f"bridge_lr x{dampen:.3f} -> {self._bridge_lr_scale:.3f}",
                deviation_trend,
            )
        elif deviation_trend < -0.01:
            # Bridge converging back — can restore LR.
            recover = min(1.1, 1.0 + abs(deviation_trend))
            self._bridge_lr_scale = min(
                self._bridge_lr_scale * recover,
                self.max_bridge_lr,
            )
            return (
                f"CONVERGING (trend={deviation_trend:.5f}), "
                f"bridge_lr x{recover:.3f} -> {self._bridge_lr_scale:.3f}",
                deviation_trend,
            )

        # Short-window trend is inside the deadband. C6b v2: only STABLE if
        # deviation is GENUINELY settled over the longer window.
        settled, net_drift, confident = self._is_settled(deviation_series)
        if settled:
            return f"STABLE (trend={deviation_trend:.5f}, settled)", deviation_trend
        if not confident:
            # Not enough history to confirm convergence; hold and keep collecting.
            return (
                f"SETTLING (trend={deviation_trend:.5f}, collecting)",
                deviation_trend,
            )
        # Confirmed slow creep the deadband missed — answer with the SAME gain as
        # the FAST-GROWTH / CONVERGING branches (long-window slope), in the
        # drift's direction, instead of pinning the bridge LR while it moves.
        drift_slope = net_drift / (self.settle_window - 1)
        if net_drift > 0.0:
            dampen = max(0.8, 1.0 - drift_slope)
            self._bridge_lr_scale = max(
                self._bridge_lr_scale * dampen,
                self.min_bridge_lr,
            )
            return (
                f"DRIFTING-UP (trend={deviation_trend:.5f}, "
                f"net={net_drift:.5f}), bridge_lr x{dampen:.3f} "
                f"-> {self._bridge_lr_scale:.3f}",
                deviation_trend,
            )
        recover = min(1.1, 1.0 + abs(drift_slope))
        self._bridge_lr_scale = min(
            self._bridge_lr_scale * recover,
            self.max_bridge_lr,
        )
        return (
            f"DRIFTING-DOWN (trend={deviation_trend:.5f}, "
            f"net={net_drift:.5f}), bridge_lr x{recover:.3f} "
            f"-> {self._bridge_lr_scale:.3f}",
            deviation_trend,
        )

    def observe_and_decide(
        self,
        step: int,
        injected: dict[str, RhombiLoRALinear],
    ) -> SteersmanState:
        """The full cybernetic loop: sense → analyze → decide.

        Parameters
        ----------
        step : current training step
        injected : dict of adapter name → RhombiLoRALinear module

        Returns
        -------
        SteersmanState with measurements and decisions
        """
        # ── SENSOR: extract bridge matrices ──
        bridges = {}
        for name, lora in injected.items():
            bridges[name] = lora.effective_bridge.detach().cpu().numpy()

        # ── OSCILLOSCOPE: spectral diagnostics ──
        fiedlers = []
        gaps = []
        deviations = []
        patterns = []
        for name, B in bridges.items():
            n = B.shape[0]
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            weights = [float(abs(B[i, j])) for i, j in edges]
            fiedlers.append(fiedler_value(n, edges, weights))
            gaps.append(spectral_gap(n, edges, weights))
            # Deviation from identity
            dev = float(np.linalg.norm(B - np.eye(n)))
            deviations.append(dev)
            # Eigenvalue pattern (first adapter only for logging)
            if not patterns:
                eigs = spectrum(n, edges, weights)
                patterns.append(eigenvalue_multiplicity_pattern(eigs))

        f_mean = float(np.mean(fiedlers))
        f_std = float(np.std(fiedlers))
        gap_mean = float(np.mean(gaps))
        d_mean = float(np.mean(deviations))
        d_std = float(np.std(deviations))

        # Co/cross ratio — aggregate across ALL bridges
        co_cross_values = []
        for B in bridges.values():
            r = coplanar_crossplanar_ratio(B)
            if r is not None:
                ratio = r["ratio"]
                if np.isfinite(ratio):
                    co_cross_values.append(ratio)
        co_cross = float(np.mean(co_cross_values)) if co_cross_values else None

        # Gradient effective rank
        eff_rank = gradient_effective_rank(injected)

        # ── STEERSMAN: compute trends and decide ──
        fiedler_history = [s.fiedler_mean for s in self.history]
        co_cross_history = [
            s.co_cross_ratio for s in self.history
            if s.co_cross_ratio is not None
        ]
        deviation_history = [s.deviation_mean for s in self.history]

        fiedler_series = fiedler_history + [f_mean]
        deviation_series = deviation_history + [d_mean]
        co_cross_trend = self._trend(
            co_cross_history + ([co_cross] if co_cross is not None else [])
        )

        signals = {}

        # Control Law 1: CONNECTIVITY (see Steersman._connectivity_law).
        # Mutates self._spectral_weight; STABLE now requires a genuinely settled
        # Fiedler level, not merely a small short-window trend (C6b v2).
        signals["connectivity"], fiedler_trend = self._connectivity_law(
            fiedler_series
        )

        # Adapt spectral target: track upward but slowly
        if f_mean > self.spectral_target:
            self.spectral_target += self.target_tracking_rate * (
                f_mean - self.spectral_target
            )
            signals["target_adaptation"] = f"target -> {self.spectral_target:.4f}"

        # Control Law 2: DIRECTIONALITY
        # If co/cross ratio is stagnant near 1.0, increase contrastive pressure
        # SKIP when fixed_contrastive is set — weight stays constant
        if self.fixed_contrastive is not None:
            signals["directionality"] = (
                f"FIXED (c_w={self.fixed_contrastive:.4f}, "
                f"ratio={co_cross:.3f})" if co_cross is not None
                else f"FIXED (c_w={self.fixed_contrastive:.4f})"
            )
        elif co_cross is not None and len(co_cross_history) >= 2:
            if abs(co_cross_trend) < self.co_cross_stagnation and co_cross < 1.1:
                boost = min(0.02, self.max_contrastive - self._contrastive_weight)
                self._contrastive_weight = min(
                    self._contrastive_weight + boost,
                    self.max_contrastive,
                )
                signals["directionality"] = (
                    f"STAGNANT (ratio={co_cross:.3f}, trend={co_cross_trend:.5f}), "
                    f"contrastive +{boost:.4f}"
                )
            elif co_cross > 1.2:
                # Strong directional preference achieved — ease off
                decay = min(0.01, self._contrastive_weight * 0.1)
                self._contrastive_weight = max(
                    self._contrastive_weight - decay,
                    self.base_contrastive * 0.25,
                )
                signals["directionality"] = (
                    f"STRONG (ratio={co_cross:.3f}), contrastive -{decay:.4f}"
                )
            else:
                signals["directionality"] = (
                    f"MOVING (ratio={co_cross:.3f}, trend={co_cross_trend:.5f})"
                )

        # Control Law 3: STABILITY (see Steersman._stability_law).
        # Mutates self._bridge_lr_scale; STABLE now requires a genuinely settled
        # deviation level, not merely a small short-window trend (C6b v2). This
        # is the law whose defect is documented in WL-001.
        signals["stability"], deviation_trend = self._stability_law(
            deviation_series
        )

        # Build state snapshot
        state = SteersmanState(
            step=step,
            fiedler_mean=f_mean,
            fiedler_std=f_std,
            co_cross_ratio=co_cross,
            spectral_gap_mean=gap_mean,
            deviation_mean=d_mean,
            deviation_std=d_std,
            grad_eff_rank=eff_rank,
            eigenvalue_pattern=patterns[0] if patterns else [],
            lr_scale=self._lr_scale,
            contrastive_weight=self._contrastive_weight,
            spectral_weight=self._spectral_weight,
            spectral_target=self.spectral_target,
            bridge_lr_scale=self._bridge_lr_scale,
            fiedler_trend=fiedler_trend,
            co_cross_trend=co_cross_trend,
            deviation_trend=deviation_trend,
            control_signals=signals,
        )

        self.history.append(state)
        return state


# ── Resume from checkpoint ──────────────────────────────────────────


def _load_resume_checkpoint(
    resume_dir: Path,
    injected: dict[str, RhombiLoRALinear],
    device: torch.device,
    emanation_bridge=None,
) -> int:
    """Load adapter/bridge state from a checkpoint directory and return the resume step.

    Loads lora_A, lora_B, and bridge weights from adapter_state.pt.
    Reads the last checkpoint step from results.json.

    NOTE: Optimizer state is NOT saved in current checkpoints, so the optimizer
    will restart from scratch. The model continues from saved adapter/bridge
    weights, but momentum and adaptive learning rate state are lost. This means
    the first few hundred steps after resume may have slightly different dynamics
    than an uninterrupted run. For long runs this is acceptable; for precise
    reproducibility, full optimizer state checkpointing would need to be added.

    Parameters
    ----------
    resume_dir : Path to the checkpoint directory
    injected : dict of adapter name -> RhombiLoRALinear (already on device)
    device : torch device
    emanation_bridge : optional EmanationBridge instance

    Returns
    -------
    int : the global step to resume from
    """
    adapter_path = resume_dir / "adapter_state.pt"
    results_path = resume_dir / "results.json"

    if not adapter_path.exists():
        raise FileNotFoundError(f"No adapter_state.pt found in {resume_dir}")

    # Load adapter state
    adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    loaded_count = 0
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        key_a = f"{safe}.lora_A"
        key_b = f"{safe}.lora_B"
        key_bridge = f"{safe}.bridge"
        if key_a in adapter_state and key_b in adapter_state:
            lora.lora_A.data.copy_(adapter_state[key_a].to(device))
            lora.lora_B.data.copy_(adapter_state[key_b].to(device))
            loaded_count += 1
        else:
            print(f"  WARNING: adapter state not found for {name}, using fresh init")
        if key_bridge in adapter_state:
            lora.bridge_param.data.copy_(adapter_state[key_bridge].to(device))

    print(f"  Loaded adapter state for {loaded_count}/{len(injected)} layers")

    # Load emanation master bridge if applicable
    if emanation_bridge is not None:
        master_path = resume_dir / "bridge_emanation_master_final.npy"
        if master_path.exists():
            master_np = np.load(master_path)
            emanation_bridge.master.data.copy_(
                torch.from_numpy(master_np).to(device)
            )
            print(f"  Loaded emanation master bridge")

    # Determine resume step from results.json
    resume_step = 0
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        checkpoints = results.get("checkpoints", [])
        if checkpoints:
            resume_step = checkpoints[-1].get("step", 0)
    else:
        print(f"  WARNING: No results.json in {resume_dir}, resuming from step 0")

    print(f"  Resuming from step {resume_step}")
    return resume_step


# ── BM-004 paired-transit corpus loading ────────────────────────────
#
# Wiring for the BM-004 GPU phase (docs/BM004_PREREGISTRATION_v2_2026-07-07.md).
# The paired-transit corpus is built by scripts/bm004_transit_data.py; the
# runner (scripts/bm004_runner.py) hands this trainer a corpus directory and an
# arm name. Corpus -> training text is a PURE function (load_transit_texts) with
# no tokenizer dependency, so it is unit-testable without a model; tokenization
# happens in TransitCorpusDataset, mirroring AlpacaDataset's conventions exactly.

# transit-arm -> (geometry arm written by the builder, include articulation?).
# The builder writes files per GEOMETRY arm ("matched"/"shuffled"); the trainer
# arm names are training compositions (prereg §6 run table): "matched" and
# "matched+articulation" both draw the matched geometry, "shuffled" draws the
# shuffled-adjacency twin.
_TRANSIT_ARMS = {
    "matched": ("matched", False),
    "matched+articulation": ("matched", True),
    "shuffled": ("shuffled", False),
}

# Record kinds pulled in deterministic order (walks, then pairs, then — only for
# the articulation arm — articulation). One JSONL "text" field = one sequence.
_TRANSIT_KINDS = ("walks", "pairs", "articulation")


def load_transit_texts(corpus_dir, transit_arm: str) -> list[str]:
    """Assemble the training text sequences for one transit arm (no tokenizer).

    Reads the builder's ``{geometry}_{kind}.jsonl`` files and returns the list
    of ``text`` fields in a deterministic order — walks, then pairs, then
    articulation (articulation only for the ``matched+articulation`` arm). This
    is the pure corpus->text stage; ``TransitCorpusDataset`` tokenizes the
    result. Kept tokenizer-free so it is testable without a model download.
    """
    if transit_arm not in _TRANSIT_ARMS:
        raise ValueError(
            f"unknown transit arm {transit_arm!r}; "
            f"choose from {sorted(_TRANSIT_ARMS)}")
    geometry, include_articulation = _TRANSIT_ARMS[transit_arm]
    corpus_dir = Path(corpus_dir)

    kinds = ["walks", "pairs"]
    if include_articulation:
        kinds.append("articulation")

    texts: list[str] = []
    for kind in kinds:
        path = corpus_dir / f"{geometry}_{kind}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"transit corpus is missing {path.name} "
                f"(expected under {corpus_dir} for arm {transit_arm!r})")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                texts.append(json.loads(line)["text"])
    if not texts:
        raise ValueError(
            f"transit corpus for arm {transit_arm!r} in {corpus_dir} is empty")
    return texts


class TransitCorpusDataset(Dataset):
    """Tokenized BM-004 paired-transit dataset for one arm.

    Corpus->text is delegated to ``load_transit_texts`` (pure, testable). Each
    text is tokenized with the SAME conventions the trainer's other datasets use
    (AlpacaDataset/CodeAlpacaDataset): fixed ``max_length`` padding, truncation,
    labels == input_ids. Sequence ORDER is the deterministic load order; epoch
    shuffling is left to the DataLoader (shuffle=True), matching the existing
    datasets' seed discipline.
    """

    def __init__(self, corpus_dir, transit_arm: str, tokenizer,
                 max_len: int = 512):
        texts = load_transit_texts(corpus_dir, transit_arm)
        self.examples = []
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0).clone(),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def resolve_dataset_source(transit_corpus, dataset):
    """Resolve the mutually-exclusive dataset selectors to one source.

    Returns ``("transit", corpus_dir)`` when a transit corpus is given, else
    ``("dataset", dataset_name)`` with the generic dataset (defaulting to
    "alpaca" when unset — the pre-wiring default, byte-identical). Raises
    ValueError if BOTH are explicitly given: ``--transit-corpus`` selects the
    BM-004 paired-transit corpus and ``--dataset`` selects a generic HF dataset;
    they cannot both drive one run.
    """
    if transit_corpus is not None and dataset is not None:
        raise ValueError(
            "--transit-corpus and --dataset are mutually exclusive: the first "
            "selects the BM-004 paired-transit corpus, the second a generic "
            "HuggingFace dataset. Pass exactly one (omit --dataset for transit "
            "arms; omit --transit-corpus for alpaca/code/math).")
    if transit_corpus is not None:
        return ("transit", transit_corpus)
    return ("dataset", dataset if dataset is not None else "alpaca")


# ── E-5 contrastive-spec corruption ──────────────────────────────────
#
# E-5 dated edit 2026-07-10 (docs/E5_BIFURCATION_PREREG_2026-07-10.md §2).
# Interpolates the coherence of the contrastive pair specification by a control
# parameter f (pair-correctness) so a training-time bifurcation over spec
# coherence can be swept. Pure, tokenizer/torch-free, and byte-identical at the
# default f=1.0 — the trainer's default path is provably unchanged.


def apply_pair_correctness(co_pairs, cross_pairs, f, seed):
    """Corrupt the contrastive pair spec by pair-correctness ``f`` (E-5 §2).

    Implements docs/E5_BIFURCATION_PREREG_2026-07-10.md §2 exactly. The
    ``co_pairs`` (attract, label "co") and ``cross_pairs`` (repel, label
    "cross") lists form an ordered labeled universe (all co pairs, then all
    cross pairs). ``k = round((1-f) * len(universe))`` pairs are selected
    uniformly at random with a DEDICATED, LOCAL RNG seeded ``seed*100003 + 51``;
    the labels among the selected pairs are shuffled (count-preserving — the
    global co/cross budget is exact at every f) and the pairs re-sorted into the
    two output lists in universe order. The draw never touches the global
    random / numpy / torch streams, so training stochasticity is untouched.

    ``f >= 1.0`` is a byte-identical no-op: the input lists are returned
    unchanged and NO RNG is constructed (returns before any RNG object exists).

    Returns ``(co_pairs_out, cross_pairs_out, record)``. ``record`` is the
    realized-corruption audit written into the run's ``config.json``:
    at ``f >= 1.0`` exactly ``{"pair_correctness", "n_corrupted",
    "corrupted_pairs", "realized_agreement"}``; at ``f < 1`` additionally ``k``,
    ``pre_labels`` and ``post_labels`` for the selected pairs.
    """
    if f >= 1.0:
        # Byte-identical no-op — inputs returned unchanged, no RNG constructed.
        return co_pairs, cross_pairs, {
            "pair_correctness": 1.0,
            "n_corrupted": 0,
            "corrupted_pairs": [],
            "realized_agreement": 1.0,
        }

    # Ordered labeled universe: every co pair, then every cross pair.
    universe = ([(tuple(p), "co") for p in co_pairs]
                + [(tuple(p), "cross") for p in cross_pairs])
    n_total = len(universe)
    true_labels = [lab for _, lab in universe]
    k = int(round((1.0 - f) * n_total))

    rng = random.Random(seed * 100003 + 51)
    selected = rng.sample(range(n_total), k)
    pre = [true_labels[i] for i in selected]
    post = list(pre)
    rng.shuffle(post)

    labels = list(true_labels)
    for idx, lab in zip(selected, post):
        labels[idx] = lab

    co_pairs_out = [universe[i][0] for i in range(n_total)
                    if labels[i] == "co"]
    cross_pairs_out = [universe[i][0] for i in range(n_total)
                       if labels[i] == "cross"]

    realized_agreement = (
        sum(1 for i in range(n_total) if labels[i] == true_labels[i]) / n_total
    )
    record = {
        "pair_correctness": float(f),
        "n_corrupted": int(k),
        "k": int(k),
        "corrupted_pairs": [[int(universe[i][0][0]), int(universe[i][0][1])]
                            for i in selected],
        "pre_labels": pre,
        "post_labels": post,
        "realized_agreement": float(realized_agreement),
    }
    return co_pairs_out, cross_pairs_out, record


# ── Cybernetic Training Loop ─────────────────────────────────────────


def train_cybernetic(
    config: ExperimentConfig,
    output_dir: Path,
    # Feedback parameters
    feedback_interval: int = 100,
    # Contrastive topology override
    contrastive_topology: str = "auto",
    # Steersman initial values
    initial_contrastive: float = 0.1,
    initial_spectral: float = 0.05,
    initial_spectral_target: float = 0.1,
    # Optional Steersman overrides (corpus preset passes these)
    fiedler_decline_threshold: Optional[float] = None,
    co_cross_stagnation_band: Optional[float] = None,
    deviation_rate_threshold: Optional[float] = None,
    # Emanation architecture
    emanation: bool = False,
    # Model export
    save_merged: bool = False,
    # Fixed contrastive weight (bypasses adaptive Control Law 2)
    fixed_contrastive: Optional[float] = None,
    # Resume from checkpoint
    resume: Optional[str] = None,
    # Seed bridges from prior run (bridge-only transfer)
    seed_bridges: Optional[str] = None,
    # Dataset selection
    dataset_name: str = "alpaca",
    # BM-004 paired-transit corpus (mutually exclusive with dataset_name)
    transit_corpus: Optional[str] = None,
    transit_arm: str = "matched",
    # E-5 contrastive-spec coherence (docs/E5_BIFURCATION_PREREG_2026-07-10.md);
    # 1.0 = byte-identical no-op (guard-tested).
    pair_correctness: float = 1.0,
):
    """Experiment 3: Cybernetic bridge training with closed-loop feedback.

    The lattice diagnoses itself and adjusts training parameters in real time.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = asdict(config)
    config_dict["experiment"] = "3.0"
    config_dict["feedback_interval"] = feedback_interval
    config_dict["initial_contrastive"] = initial_contrastive
    config_dict["initial_spectral"] = initial_spectral
    config_dict["initial_spectral_target"] = initial_spectral_target
    config_dict["emanation"] = emanation
    config_dict["fixed_contrastive"] = fixed_contrastive
    config_dict["resumed_from"] = resume
    config_dict["dataset_name"] = dataset_name
    config_dict["transit_corpus"] = str(transit_corpus) if transit_corpus else None
    config_dict["transit_arm"] = transit_arm if transit_corpus else None
    config_dict["seed_bridges"] = seed_bridges
    # C6b: self-document which Steersman STABLE-detector governed this run so
    # future audits can tell v2-governed runs (corrected) from v1 runs (the
    # defective detector that could declare STABLE while the metric still moved).
    # PAST runs' stored config.json lack this field == detector v1 by definition.
    config_dict["steersman_detector_version"] = STEERSMAN_DETECTOR_VERSION

    if contrastive_topology == "wrong-labels":
        co_pairs, cross_pairs = _compute_wrong_label_pairs()
        geom_label = "WRONG-LABELS (random partition of 6 channels)"
    elif contrastive_topology == "resonance":
        co_pairs, cross_pairs = _compute_resonance_pairs(config.n_channels)
        geom_label = "RESONANCE (prime threading topology)"
    else:
        co_pairs, cross_pairs = _compute_pair_indices(config.n_channels)
        geom_label = {
            4: "Octahedron (2-axis partition)",
            6: "RD (rhombic dodecahedron)",
            8: "Tesseract (4D hypercube)",
            24: "24-cell (D4 root polytope, 12 antipodal axes)",
        }.get(config.n_channels, f"n={config.n_channels} (no geometric prior)")

    # E-5 dated edit 2026-07-10 (docs/E5_BIFURCATION_PREREG_2026-07-10.md);
    # default 1.0 preserves pre-edit behavior — guard-tested. Corrupt the
    # contrastive spec by pair-correctness f (§2) BEFORE it drives training and
    # record the realized corruption in config.json. Applied only when both pair
    # lists are non-empty (an empty spec has nothing to shuffle). The config.json
    # dump is moved below this branch so the record lands in the file on disk.
    pair_correctness_record = None
    if co_pairs and cross_pairs:
        co_pairs, cross_pairs, pair_correctness_record = apply_pair_correctness(
            co_pairs, cross_pairs, pair_correctness, config.seed
        )
        config_dict.update(pair_correctness_record)
        config_dict["co_pairs_trained"] = [[int(i), int(j)]
                                           for i, j in co_pairs]
        config_dict["cross_pairs_trained"] = [[int(i), int(j)]
                                              for i, j in cross_pairs]

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    contrastive_active = bool(co_pairs and cross_pairs)

    print(f"\n{'='*70}")
    print(f"Experiment 3.0: CYBERNETIC Bridge Training")
    print(f"  The lattice IS the steersman.")
    print(f"{'='*70}")
    print(f"Model:              {config.model_name}")
    print(f"Rank:               {config.rank}, Channels: {config.n_channels}")
    print(f"Geometry:           {geom_label}")
    print(f"Contrastive:        {'ACTIVE' if contrastive_active else 'DISABLED (no pairs for this n)'}")
    if pair_correctness < 1.0 and pair_correctness_record is not None:
        _n_univ = len(co_pairs) + len(cross_pairs)
        print(
            f"Pair-correctness:   "
            f"f={pair_correctness_record['pair_correctness']:.2f} "
            f"({pair_correctness_record['n_corrupted']}/{_n_univ} labels "
            f"shuffled, realized agreement "
            f"{pair_correctness_record['realized_agreement']:.2f})"
        )
    print(f"Steps:              {config.max_steps}")
    print(f"Feedback interval:  every {feedback_interval} steps")
    print(f"Initial contrastive: {initial_contrastive}")
    print(f"Initial spectral:   {initial_spectral}")
    print(f"Spectral target:    {initial_spectral_target}")
    if contrastive_active:
        print(f"Co-planar pairs:    {co_pairs}")
        print(f"Cross-planar pairs: {len(cross_pairs)} pairs")
    print(f"Emanation:          {'ACTIVE' if emanation else 'DISABLED'}")
    print(f"Bridge trainable:   {config.bridge_trainable}")
    if not config.bridge_trainable:
        print(f"  *** STANDARD LORA MODE — bridge frozen at identity ***")
    if fixed_contrastive is not None:
        print(f"Fixed contrastive:  {fixed_contrastive} (Control Law 2 DISABLED)")
    if resume is not None:
        print(f"Resume from:        {resume}")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    # Emanation architecture: single master bridge emanates into per-layer bridges
    emanation_bridge = None
    if emanation:
        n_layers = len(injected)
        emanation_bridge = EmanationBridge(
            n_channels=config.n_channels,
            n_layers=n_layers,
        ).to(device)
        # Wire each adapter to its emanated bridge
        for layer_idx, (name, lora) in enumerate(injected.items()):
            # Freeze the adapter's own bridge — emanation replaces it
            lora.freeze_bridge()
            # Set external bridge function: this layer's effective_bridge
            # will call emanation.get_bridge(layer_idx) instead of using
            # the local bridge parameter
            idx = layer_idx  # capture in closure
            lora._external_bridge_fn = lambda _idx=idx: emanation_bridge.get_bridge(_idx)
        print(f"Emanation: master bridge ({config.n_channels}x{config.n_channels}) "
              f"-> {n_layers} layer offsets")
        print(f"Emanation coherence: {emanation_bridge.coherence:.4f}")

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    # Add emanation params if present (they're not part of the model)
    if emanation_bridge is not None:
        trainable_params += sum(
            p.numel() for p in emanation_bridge.parameters()
        )
    total_params = sum(p.numel() for p in model.parameters())
    if emanation_bridge is not None:
        bridge_params = sum(
            p.numel() for p in emanation_bridge.parameters()
        )
    else:
        bridge_params = sum(
            lora.bridge_param.numel() for lora in injected.values()
            if lora.bridge_param.requires_grad
        )
    print(f"Trainable: {trainable_params:,} / {total_params:,}")
    print(f"Bridge params: {bridge_params:,}")

    # Resume from checkpoint: load adapter/bridge weights before optimizer setup
    resume_step = 0
    if resume is not None:
        resume_dir = Path(resume)
        print(f"\n  Loading checkpoint from {resume_dir}...")
        resume_step = _load_resume_checkpoint(
            resume_dir, injected, device,
            emanation_bridge=emanation_bridge,
        )
        print(f"  Training will continue from step {resume_step} to {config.max_steps}")
        if resume_step >= config.max_steps:
            print(f"  WARNING: resume step {resume_step} >= max_steps {config.max_steps}, nothing to do")
            return []
        print()

    # Seed bridges from a prior run (bridge-only transfer, fresh lora_A/B)
    if seed_bridges is not None:
        seed_dir = Path(seed_bridges)
        seeded = 0
        missing = 0
        for name, lora in injected.items():
            safe = name.replace(".", "_")
            npy_path = seed_dir / f"bridge_final_{safe}.npy"
            if npy_path.exists():
                bridge_np = np.load(npy_path)
                lora.bridge_param.data.copy_(torch.from_numpy(bridge_np).float().to(device))
                seeded += 1
            else:
                missing += 1
        print(f"  Bridge seeding: loaded {seeded}/{seeded+missing} bridges from {seed_dir}")
        if missing > 0:
            print(f"  WARNING: {missing} bridges not found, left at identity")
        print()

    # Separate parameter groups: LoRA A/B vs Bridge
    lora_params = []
    bridge_param_list = []
    for lora in injected.values():
        lora_params.extend([lora.lora_A, lora.lora_B])
        if not emanation and lora.bridge_param.requires_grad:
            bridge_param_list.append(lora.bridge_param)

    # Emanation: master + offsets replace per-adapter bridge params
    if emanation_bridge is not None:
        bridge_param_list.append(emanation_bridge.master)
        bridge_param_list.extend(list(emanation_bridge.offsets))

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": config.lr, "name": "lora"},
            {"params": bridge_param_list, "lr": config.lr, "name": "bridge"},
        ],
        weight_decay=0.01,
    )

    total_steps = config.max_steps

    def lr_schedule(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_schedule, lr_schedule]  # one per param group
    )

    # Dataset
    if transit_corpus is not None:
        # BM-004 paired-transit corpus (prereg §6). No held-out split is drawn
        # here — E1 eval uses fresh builder walks per the pre-registration; the
        # val loader mirrors the train arm for in-loop loss reporting only.
        print(f"Loading BM-004 transit corpus ({transit_arm}) "
              f"from {transit_corpus}...")
        dataset = TransitCorpusDataset(
            transit_corpus, transit_arm, tokenizer,
            max_len=config.max_seq_len)
        val_dataset = TransitCorpusDataset(
            transit_corpus, transit_arm, tokenizer,
            max_len=config.max_seq_len)
    elif dataset_name == "code":
        from train_task_fingerprint import CodeAlpacaDataset
        print("Loading CodeAlpaca-20k dataset...")
        dataset = CodeAlpacaDataset(tokenizer, max_len=config.max_seq_len)
        val_dataset = CodeAlpacaDataset(tokenizer, max_len=config.max_seq_len)
    elif dataset_name == "math":
        from train_task_fingerprint import GSM8KDataset
        print("Loading GSM8K dataset...")
        dataset = GSM8KDataset(tokenizer, max_len=config.max_seq_len)
        val_dataset = GSM8KDataset(tokenizer, max_len=config.max_seq_len)
    else:
        print("Loading Alpaca-cleaned dataset...")
        dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len)
        val_dataset = AlpacaDataset(
            tokenizer, max_len=config.max_seq_len, max_samples=1000
        )
    print(f"Training examples: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Initialize the Steersman (with optional preset overrides)
    steersman_kwargs = dict(
        base_contrastive_weight=initial_contrastive,
        base_spectral_weight=initial_spectral,
        initial_spectral_target=initial_spectral_target,
    )
    if fiedler_decline_threshold is not None:
        steersman_kwargs["fiedler_decline_threshold"] = fiedler_decline_threshold
    if co_cross_stagnation_band is not None:
        steersman_kwargs["co_cross_stagnation_band"] = co_cross_stagnation_band
    if deviation_rate_threshold is not None:
        steersman_kwargs["deviation_rate_threshold"] = deviation_rate_threshold
    if fixed_contrastive is not None:
        steersman_kwargs["fixed_contrastive_weight"] = fixed_contrastive
    steersman = Steersman(**steersman_kwargs)

    # Training state
    checkpoints: list[dict] = []
    feedback_log: list[dict] = []
    model.train()
    global_step = resume_step
    accumulation_loss = 0.0
    accumulation_contrastive = 0.0
    accumulation_spectral = 0.0
    steps_since_log = 0
    start_time = time.time()

    # Advance LR scheduler to resume position so cosine schedule is correct
    if resume_step > 0:
        for _ in range(resume_step):
            scheduler.step()
        print(f"  Advanced LR scheduler to step {resume_step}")

    # Save step-0 bridges (skip when resuming — the checkpoint has them)
    if resume_step == 0:
        for name, lora in injected.items():
            safe = name.replace(".", "_")
            np.save(
                output_dir / f"bridge_step0_{safe}.npy",
                lora.effective_bridge.detach().cpu().numpy(),
            )
    if emanation_bridge is not None and resume_step == 0:
        np.save(
            output_dir / "bridge_emanation_master_step0.npy",
            emanation_bridge.master.detach().cpu().numpy(),
        )

    # Initial Steersman read (before training begins, or at resume point)
    initial_state = steersman.observe_and_decide(resume_step, injected)
    feedback_log.append(asdict(initial_state))
    print(f"\n  *** STEERSMAN @ step {resume_step} ***")
    print(f"      Fiedler:       {initial_state.fiedler_mean:.5f}")
    print(f"      Spectral gap:  {initial_state.spectral_gap_mean:.5f}")
    print(f"      Deviation:     {initial_state.deviation_mean:.5f}")
    if initial_state.co_cross_ratio is not None:
        print(f"      Co/Cross:      {initial_state.co_cross_ratio:.3f}")
    if emanation_bridge is not None:
        print(f"      Emanation coh: {emanation_bridge.coherence:.4f}")
    print()

    if resume_step > 0:
        print(f"Resuming training from step {resume_step} to {total_steps}...")
    else:
        print(f"Training for {total_steps} steps...")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation}")
    print()

    done = False
    for epoch in range(config.num_epochs):
        if done:
            break
        for batch_idx, batch in enumerate(dataloader):
            if done:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass — language modeling loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss / config.gradient_accumulation

            # Contrastive bridge loss (Steersman-controlled weight)
            c_weight = steersman._contrastive_weight
            if c_weight > 0 and contrastive_active:
                c_loss = contrastive_bridge_loss(
                    injected, co_pairs, cross_pairs
                )
                scaled_c = c_weight * c_loss / config.gradient_accumulation
                accumulation_contrastive += scaled_c.item()
            else:
                scaled_c = 0.0

            # Spectral regularization (Steersman-controlled weight + target)
            s_weight = steersman._spectral_weight
            if s_weight > 0:
                s_loss = spectral_reg_loss(
                    injected,
                    target_fiedler=steersman.spectral_target,
                )
                scaled_s = s_weight * s_loss / config.gradient_accumulation
                accumulation_spectral += scaled_s.item()
            else:
                scaled_s = 0.0

            total_loss = lm_loss
            if isinstance(scaled_c, torch.Tensor):
                total_loss = total_loss + scaled_c
            if isinstance(scaled_s, torch.Tensor):
                total_loss = total_loss + scaled_s

            total_loss.backward()
            accumulation_loss += lm_loss.item()

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )

                # Apply Steersman's bridge LR scaling
                bridge_scale = steersman._bridge_lr_scale
                for pg in optimizer.param_groups:
                    if pg.get("name") == "bridge":
                        pg["lr"] = config.lr * lr_schedule(global_step) * bridge_scale

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                steps_since_log += 1

                # Progress display every 100 steps
                if global_step % 100 == 0:
                    avg_lm = accumulation_loss / steps_since_log
                    avg_c = accumulation_contrastive / max(steps_since_log, 1)
                    avg_s = accumulation_spectral / max(steps_since_log, 1)
                    elapsed = time.time() - start_time
                    speed = global_step / elapsed
                    eta = (total_steps - global_step) / speed

                    print(
                        f"  Step {global_step:>5}/{total_steps} | "
                        f"LM: {avg_lm:.4f} | "
                        f"C: {avg_c:.5f}(w={c_weight:.3f}) | "
                        f"S: {avg_s:.5f}(w={s_weight:.3f}) | "
                        f"BLR: {bridge_scale:.2f} | "
                        f"{speed:.1f} step/s | "
                        f"ETA: {eta/60:.0f}m"
                    )

                # ── FEEDBACK LOOP ──
                if (
                    global_step % feedback_interval == 0
                    or global_step >= total_steps
                ):
                    # The Steersman reads the lattice and decides
                    state = steersman.observe_and_decide(global_step, injected)
                    feedback_log.append(asdict(state))

                    # Dynamic bridge temperature annealing (L9)
                    if config.dynamic_bridge:
                        progress = min(global_step / total_steps, 1.0)
                        T = (
                            config.gate_temperature_start
                            + progress * (
                                config.gate_temperature_end
                                - config.gate_temperature_start
                            )
                        )
                        for lora in injected.values():
                            lora.set_gate_temperature(T)

                    # Validation loss
                    val_loss = evaluate(model, val_dataloader, device)

                    avg_lm = accumulation_loss / steps_since_log
                    avg_c = accumulation_contrastive / max(steps_since_log, 1)
                    avg_s = accumulation_spectral / max(steps_since_log, 1)

                    # Checkpoint data
                    cp = {
                        "step": global_step,
                        "train_loss": avg_lm,
                        "val_loss": val_loss,
                        "contrastive_loss": avg_c,
                        "spectral_loss": avg_s,
                        "fiedler_mean": state.fiedler_mean,
                        "fiedler_std": state.fiedler_std,
                        "co_cross_ratio": state.co_cross_ratio,
                        "spectral_gap": state.spectral_gap_mean,
                        "deviation_mean": state.deviation_mean,
                        "deviation_std": state.deviation_std,
                        "grad_eff_rank": state.grad_eff_rank,
                        # Steersman decisions
                        "contrastive_weight": state.contrastive_weight,
                        "spectral_weight": state.spectral_weight,
                        "spectral_target": state.spectral_target,
                        "bridge_lr_scale": state.bridge_lr_scale,
                        "fiedler_trend": state.fiedler_trend,
                        "co_cross_trend": state.co_cross_trend,
                        "deviation_trend": state.deviation_trend,
                        "control_signals": state.control_signals,
                        "wall_time": time.time() - start_time,
                    }
                    # Dynamic bridge metrics (L9)
                    if config.dynamic_bridge:
                        sample_lora = next(iter(injected.values()))
                        cp["gate_temperature"] = sample_lora.gate_temperature()
                        cp["gate_sparsity"] = sample_lora.gate_sparsity()
                    # Emanation metrics
                    if emanation_bridge is not None:
                        cp["emanation_coherence"] = emanation_bridge.coherence
                    checkpoints.append(cp)

                    # Save bridge matrices
                    for name, lora in injected.items():
                        safe = name.replace(".", "_")
                        np.save(
                            output_dir / f"bridge_step{global_step}_{safe}.npy",
                            lora.effective_bridge.detach().cpu().numpy(),
                        )
                    # Save emanation master bridge
                    if emanation_bridge is not None:
                        np.save(
                            output_dir / f"bridge_emanation_master_step{global_step}.npy",
                            emanation_bridge.master.detach().cpu().numpy(),
                        )

                    # Emanation coherence monitoring: if coherence drops
                    # below 0.5, the system is fragmenting — increase
                    # spectral regularization to pull layers back toward master
                    if emanation_bridge is not None:
                        eman_coh = emanation_bridge.coherence
                        if eman_coh < 0.5:
                            boost = min(
                                0.02,
                                steersman.max_spectral - steersman._spectral_weight,
                            )
                            steersman._spectral_weight = min(
                                steersman._spectral_weight + boost,
                                steersman.max_spectral,
                            )
                            state.control_signals["emanation"] = (
                                f"FRAGMENTING (coherence={eman_coh:.4f}), "
                                f"spec_reg +{boost:.4f}"
                            )
                        else:
                            state.control_signals["emanation"] = (
                                f"COHERENT ({eman_coh:.4f})"
                            )

                    # Print Steersman report
                    co_str = (
                        f"{state.co_cross_ratio:.3f}" if state.co_cross_ratio
                        else "N/A"
                    )
                    eman_str = ""
                    if emanation_bridge is not None:
                        eman_str = (
                            f"    Emanation:     {emanation_bridge.coherence:.4f}\n"
                        )
                    print(
                        f"\n  {'='*60}\n"
                        f"  STEERSMAN @ step {global_step}\n"
                        f"  {'='*60}\n"
                        f"  Lattice Health:\n"
                        f"    Fiedler:       {state.fiedler_mean:.5f} (trend: {state.fiedler_trend:+.5f})\n"
                        f"    Spectral gap:  {state.spectral_gap_mean:.5f}\n"
                        f"    Co/Cross:      {co_str} (trend: {state.co_cross_trend:+.5f})\n"
                        f"    Deviation:     {state.deviation_mean:.5f} (trend: {state.deviation_trend:+.5f})\n"
                        f"    Eff rank:      {state.grad_eff_rank:.2f}\n"
                        f"{eman_str}"
                        f"  Loss Components:\n"
                        f"    LM:            {avg_lm:.4f}\n"
                        f"    Val:           {val_loss:.4f}\n"
                        f"    Contrastive:   {avg_c:.5f}\n"
                        f"    Spectral:      {avg_s:.5f}\n"
                        f"  Control Decisions:\n"
                        f"    Contrastive w: {state.contrastive_weight:.4f}\n"
                        f"    Spectral w:    {state.spectral_weight:.4f}\n"
                        f"    Spectral tgt:  {state.spectral_target:.4f}\n"
                        f"    Bridge LR:     {state.bridge_lr_scale:.3f}x\n"
                        f"  Signals:"
                    )
                    for law, msg in state.control_signals.items():
                        print(f"    {law}: {msg}")
                    print(f"  {'='*60}\n")

                    # Reset accumulators
                    accumulation_loss = 0.0
                    accumulation_contrastive = 0.0
                    accumulation_spectral = 0.0
                    steps_since_log = 0

                    # Save intermediate results
                    _save_results(
                        output_dir, config, checkpoints, feedback_log,
                        trainable_params, total_params, bridge_params,
                        injected,
                        emanation_bridge=emanation_bridge,
                    )

                if global_step >= total_steps:
                    done = True

    # Final save
    _save_results(
        output_dir, config, checkpoints, feedback_log,
        trainable_params, total_params, bridge_params,
        injected,
        emanation_bridge=emanation_bridge,
    )

    # Optionally merge LoRA+bridge into base and save as HF model
    if save_merged:
        print("\nMerging LoRA+bridge into base model for lm-eval...")
        merged_dir = merge_and_save(model, injected, output_dir)
        # Also save the tokenizer so lm-eval can load it
        tokenizer.save_pretrained(merged_dir)
        print(f"  Merged model ready at {merged_dir}")

    elapsed = time.time() - start_time
    print(f"\nExperiment 3.0 COMPLETE")
    print(f"Total time: {elapsed/3600:.2f}h")
    print(f"Feedback cycles: {len(feedback_log)}")
    print(f"Results: {output_dir}")

    # Summary: control trajectory
    if len(feedback_log) > 1:
        print(f"\n  Control Trajectory:")
        print(f"  {'Step':>6} | {'Fiedler':>8} | {'Co/Cross':>8} | "
              f"{'Dev':>8} | {'C_wt':>6} | {'S_wt':>6} | {'B_lr':>5}")
        print(f"  {'-'*60}")
        for entry in feedback_log:
            co = f"{entry['co_cross_ratio']:.3f}" if entry.get('co_cross_ratio') else "  N/A "
            print(
                f"  {entry['step']:>6} | "
                f"{entry['fiedler_mean']:>8.5f} | "
                f"{co:>8} | "
                f"{entry['deviation_mean']:>8.5f} | "
                f"{entry['contrastive_weight']:>6.3f} | "
                f"{entry['spectral_weight']:>6.3f} | "
                f"{entry['bridge_lr_scale']:>5.2f}"
            )

    return checkpoints


def _save_results(
    output_dir, config, checkpoints, feedback_log,
    trainable_params, total_params, bridge_params,
    injected,
    emanation_bridge=None,
):
    """Save all results with cybernetic metadata."""
    results = {
        "config": asdict(config),
        "experiment": "3.0",
        "description": "Cybernetic bridge training — the lattice IS the steersman",
        "checkpoints": checkpoints,
        "feedback_log": feedback_log,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "bridge_params": bridge_params,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save final bridges
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        np.save(
            output_dir / f"bridge_final_{safe}.npy",
            lora.effective_bridge.detach().cpu().numpy(),
        )
    # Save final emanation master bridge
    if emanation_bridge is not None:
        np.save(
            output_dir / "bridge_emanation_master_final.npy",
            emanation_bridge.master.detach().cpu().numpy(),
        )

    # Save adapter state dict (lora_A, bridge, lora_B per layer — compact)
    adapter_state = {}
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        adapter_state[f"{safe}.lora_A"] = lora.lora_A.detach().cpu()
        adapter_state[f"{safe}.lora_B"] = lora.lora_B.detach().cpu()
        adapter_state[f"{safe}.bridge"] = lora.effective_bridge.detach().cpu()
        adapter_state[f"{safe}.scaling"] = torch.tensor(lora.scaling)
        adapter_state[f"{safe}.n_channels"] = torch.tensor(lora.n_channels)
        adapter_state[f"{safe}.rank"] = torch.tensor(lora.rank)
    torch.save(adapter_state, output_dir / "adapter_state.pt")
    print(f"  Saved adapter state ({len(injected)} layers)")


def merge_and_save(model, injected, output_dir):
    """Merge LoRA+bridge deltas into base weights and save as HF model.

    This produces a standard HuggingFace model that lm-eval can load directly.
    The delta for each adapted layer is: scaling * lora_B @ bridge_expanded @ lora_A
    where bridge_expanded is the block-diagonal expansion of the n×n bridge to rank×rank.
    """
    import copy
    merged = copy.deepcopy(model)

    for name, lora in injected.items():
        # Navigate to the LoRAWrappedLinear and get the base layer
        parts = name.split(".")
        parent = merged
        for p in parts[:-1]:
            parent = getattr(parent, p)
        wrapped = getattr(parent, parts[-1])
        base_linear = wrapped.base if hasattr(wrapped, 'base') else wrapped

        # Compute expanded bridge: (rank, rank) block-diagonal
        bridge = lora.effective_bridge.detach().float()  # (C, C)
        C, R = lora.n_channels, lora.channel_size
        bridge_expanded = torch.zeros(lora.rank, lora.rank, device=bridge.device)
        for i in range(C):
            for j in range(C):
                bridge_expanded[i*R:(i+1)*R, j*R:(j+1)*R] = bridge[i, j] * torch.eye(R, device=bridge.device)

        # delta_W = scaling * lora_B @ bridge_expanded @ lora_A
        lora_A = lora.lora_A.detach().float()   # (rank, in_features)
        lora_B = lora.lora_B.detach().float()   # (out_features, rank)
        delta_W = lora.scaling * (lora_B @ bridge_expanded @ lora_A)

        # Merge into base weight
        base_linear.weight.data += delta_W.to(base_linear.weight.dtype).to(base_linear.weight.device)

        # Replace the wrapped module with the plain base linear
        setattr(parent, parts[-1], base_linear)

    # Save merged model
    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    print(f"  Saved merged model to {merged_dir}")
    return merged_dir


# ── CLI ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 3.0: Cybernetic Bridge Training"
    )
    parser.add_argument(
        "--output", type=str, default="results/exp3",
        help="Output directory",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--max-steps", type=int, default=20000,
        help="Total training steps",
    )
    parser.add_argument(
        "--feedback-interval", type=int, default=100,
        help="Steps between Steersman feedback cycles",
    )
    parser.add_argument(
        "--initial-contrastive", type=float, default=0.1,
        help="Initial contrastive loss weight",
    )
    parser.add_argument(
        "--initial-spectral", type=float, default=0.05,
        help="Initial spectral regularization weight",
    )
    parser.add_argument(
        "--spectral-target", type=float, default=0.1,
        help="Initial Fiedler target for spectral reg",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=8,
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=200,
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, default=100,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--bridge-mode", type=str, default="identity",
        choices=["identity", "geometric", "corpus", "corpus_coupled",
                 "rd_graph", "shuffled_rd"],
        help="Bridge initialization mode. 'rd_graph' uses RD graph convolution "
             "with fixed topology mask and learnable edge weights. "
             "'shuffled_rd' is the BM-004 wrong-symmetry twin (hard fix F3): "
             "the same mechanism over a seeded shuffle of the RD mask, seeded "
             "from --seed.",
    )
    parser.add_argument(
        "--n-channels", type=int, default=6,
        help="Number of bridge channels (default 6 = FCC direction pairs)",
    )
    parser.add_argument(
        "--steersman-preset", type=str, default="default",
        choices=["default", "corpus"],
        help="Steersman parameter preset. 'corpus' uses tighter thresholds "
             "calibrated to corpus-coupled bridge dynamics.",
    )
    parser.add_argument(
        "--dynamic-bridge", action="store_true",
        help="Enable input-dependent dynamic bridge gating (L9 prototype)",
    )
    parser.add_argument(
        "--gate-temperature-start", type=float, default=5.0,
        help="Initial gate temperature for dynamic bridge (default 5.0, soft)",
    )
    parser.add_argument(
        "--gate-temperature-end", type=float, default=0.1,
        help="Final gate temperature for dynamic bridge (default 0.1, hard)",
    )
    parser.add_argument(
        "--contrastive-topology", type=str, default="auto",
        choices=["auto", "wrong-labels", "resonance"],
        help="Contrastive pair topology. 'auto' selects based on n-channels "
             "(n=6=RD, n=8=tesseract). 'wrong-labels' uses random partition "
             "of 6 channels (tests whether RD geometry is special). "
             "'resonance' uses prime-threading topology from the corpus "
             "(Sophie Germain, consecutive primes, mod-6 residue).",
    )
    parser.add_argument(
        "--fixed-contrastive", type=float, default=None,
        help="Fixed contrastive weight (skips adaptive Control Law 2). "
             "None = adaptive.",
    )
    parser.add_argument(
        "--emanation", action="store_true",
        help="Enable emanation architecture: a single master bridge emanates "
             "into per-layer bridges via learned additive offsets. "
             "The One produces without diminishing.",
    )
    parser.add_argument(
        "--save-merged", action="store_true",
        help="After training, merge LoRA+bridge into base model and save as "
             "standard HuggingFace model for lm-eval benchmarking.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume training from. "
             "Loads adapter weights, bridge matrices, and training step. "
             "Optimizer state is NOT restored (restarts from scratch). "
             "Results are written to --output (must differ from --resume).",
    )
    parser.add_argument(
        "--no-bridge-training", action="store_true",
        help="Freeze bridge matrices at identity (exact standard LoRA). "
             "Disables Steersman feedback and contrastive/spectral losses.",
    )
    parser.add_argument(
        "--seed-bridges", type=str, default=None,
        help="Path to directory containing bridge_final_*.npy files from a "
             "prior training run.  Initialises each adapter's bridge matrix "
             "from the matching file instead of identity.  lora_A/B start "
             "fresh; training begins at step 0.  Use with Steersman active "
             "to test whether pre-discovered topology accelerates learning.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["alpaca", "code", "math"],
        help="Training dataset: alpaca (default), code (CodeAlpaca-20k), "
             "math (GSM8K). Mutually exclusive with --transit-corpus.",
    )
    parser.add_argument(
        "--transit-corpus", type=str, default=None,
        help="Directory of a BM-004 paired-transit corpus "
             "(scripts/bm004_transit_data.py output). Selects the "
             "paired-transit dataset instead of a generic HF dataset; "
             "mutually exclusive with --dataset (prereg §6).",
    )
    parser.add_argument(
        "--transit-arm", type=str, default="matched",
        choices=["matched", "matched+articulation", "shuffled"],
        help="Which transit training composition to load from the corpus "
             "(prereg §6 run table). Ignored unless --transit-corpus is set.",
    )
    parser.add_argument(
        "--no-steersman", action="store_true",
        help="Disable Steersman feedback loop and all auxiliary losses "
             "(contrastive, spectral). Bridge remains trainable with LM loss "
             "only. Use with --bridge-mode rd_graph for structural RD topology.",
    )
    parser.add_argument(
        "--pair-correctness", type=float, default=1.0,
        help="E-5 contrastive-spec coherence f in [0,1] "
             "(docs/E5_BIFURCATION_PREREG_2026-07-10.md §2). 1.0 (default) is a "
             "byte-identical no-op; f<1 shuffles the labels of "
             "round((1-f)*len(pairs)) contrastive pairs via a dedicated local "
             "RNG (seed*100003+51), count-preserving, recorded in config.json.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # E-5: f outside [0,1] is meaningless — f>1 would silently record 1.0 and
    # f<0 would raise an opaque ValueError deep in rng.sample. Reject loudly.
    if not 0.0 <= args.pair_correctness <= 1.0:
        parser.error(f"--pair-correctness must be in [0.0, 1.0], "
                     f"got {args.pair_correctness}")

    # --transit-corpus and --dataset are mutually exclusive (prereg §6). Resolve
    # to one source; error if both were explicitly given. When neither is set,
    # dataset defaults to "alpaca" — byte-identical to the pre-wiring behavior.
    try:
        source_kind, source_value = resolve_dataset_source(
            args.transit_corpus, args.dataset)
    except ValueError as e:
        parser.error(str(e))
    transit_corpus = source_value if source_kind == "transit" else None
    dataset_name = source_value if source_kind == "dataset" else "alpaca"

    # --no-bridge-training → exact standard LoRA (frozen identity bridge)
    no_bridge = getattr(args, 'no_bridge_training', False)
    # --no-steersman → bridge trainable, but LM loss only (no auxiliary losses)
    no_steersman = getattr(args, 'no_steersman', False)

    config = ExperimentConfig(
        name=f"exp3_cybernetic_{args.bridge_mode}",
        rank=24,
        n_channels=args.n_channels,
        bridge_mode=args.bridge_mode,
        bridge_trainable=not no_bridge,
        model_name=args.model,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        checkpoint_steps=args.checkpoint_steps,
        seed=args.seed,
        dynamic_bridge=args.dynamic_bridge,
        gate_temperature_start=args.gate_temperature_start,
        gate_temperature_end=args.gate_temperature_end,
    )

    # Steersman preset: corpus uses tighter thresholds derived from
    # Fiedler ~0.10 convergence point and corpus-coupled init dynamics
    steersman_kwargs = {}
    if args.steersman_preset == "corpus":
        steersman_kwargs = dict(
            initial_spectral_target=0.1,
            fiedler_decline_threshold=-0.000894,
            co_cross_stagnation_band=0.00957,
            deviation_rate_threshold=0.04677,
        )

    # Validate --resume and --output are different directories
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        output_path = Path(args.output).resolve()
        if resume_path == output_path:
            parser.error(
                f"--resume and --output must be different directories. "
                f"Both resolve to: {resume_path}"
            )

    # When bridge is frozen OR steersman is disabled, zero all auxiliary losses
    disable_aux = no_bridge or no_steersman
    eff_contrastive = 0.0 if disable_aux else args.initial_contrastive
    eff_spectral = 0.0 if disable_aux else args.initial_spectral
    eff_fixed_contrastive = 0.0 if disable_aux else args.fixed_contrastive

    train_cybernetic(
        config,
        Path(args.output),
        feedback_interval=args.feedback_interval,
        contrastive_topology=args.contrastive_topology,
        initial_contrastive=eff_contrastive,
        initial_spectral=eff_spectral,
        initial_spectral_target=(
            steersman_kwargs.get("initial_spectral_target", args.spectral_target)
        ),
        emanation=args.emanation,
        save_merged=args.save_merged,
        fixed_contrastive=eff_fixed_contrastive,
        resume=args.resume,
        seed_bridges=args.seed_bridges,
        dataset_name=dataset_name,
        transit_corpus=transit_corpus,
        transit_arm=args.transit_arm,
        pair_correctness=args.pair_correctness,
        **{k: v for k, v in steersman_kwargs.items()
           if k != "initial_spectral_target"},
    )


if __name__ == "__main__":
    main()
