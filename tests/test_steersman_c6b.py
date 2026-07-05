"""C6b regression tests — Steersman STABLE-declaration detector (v2).

These tests pin the fix for the C6b defect (see docs/C6B_FIX_NOTE.md and
docs/PAPER4_EXPOSURE_CLASSIFICATION.md §2): Control Law 1 (CONNECTIVITY) and
Control Law 3 (STABILITY) declared "STABLE" on a short-window trend deadband
ALONE, so a slow monotonic drift whose per-sample slope stayed inside the
deadband was declared STABLE indefinitely while the governed metric kept moving.

The v2 detector requires BOTH a small short-window trend AND a genuinely settled
level. The tests demonstrate, on synthetic signals and the real WL-001 trace,
that:

  * a "trend-flat but far from target" signal that the OLD rule declared STABLE
    is NO LONGER declared STABLE by the new detector (and the idle actuator now
    responds); and
  * a "genuinely converged" signal is STILL declared STABLE by BOTH the old rule
    and the new detector (convergence detection is preserved — no
    never-declare-STABLE regression).

The changed logic lives in the pure methods Steersman._is_settled,
._connectivity_law and ._stability_law, which are exercised directly here.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

# The Steersman lives in scripts/train_cybernetic.py; put scripts/ on the path.
_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import train_cybernetic as tc  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WL001 = os.path.join(_REPO, "results", "channel-ablation", "WL-001", "results.json")


# ── Reference implementations of the OLD (v1, defective) STABLE rule ──────────
# These reproduce the pre-fix else-branch semantics exactly. In v1 both laws
# declared STABLE whenever the short-window trend sat inside its deadband,
# with no settled check. Kept here so each test can assert "the old rule WOULD
# have fired here" against the identical trend the new detector sees.

def _old_cl1_declares_stable(trend: float) -> bool:
    # v1 CL1: STABLE when -fiedler_decline_thresh <= trend <= 0.001
    return not (trend < -0.001) and not (trend > 0.001)


def _old_cl3_declares_stable(trend: float) -> bool:
    # v1 CL3: STABLE when -0.01 <= trend <= deviation_rate_thresh (0.05)
    return not (trend > 0.05) and not (trend < -0.01)


# ── Version tag ───────────────────────────────────────────────────────────────

def test_detector_version_tag_present():
    """The detector self-documents its version for run-level provenance."""
    assert isinstance(tc.STEERSMAN_DETECTOR_VERSION, str)
    assert tc.STEERSMAN_DETECTOR_VERSION.startswith("v2-")
    assert tc.Steersman().detector_version == tc.STEERSMAN_DETECTOR_VERSION


# ── _is_settled unit behaviour ────────────────────────────────────────────────

def test_settled_accepts_genuine_convergence():
    """A flat, low-noise level over the settle window is settled."""
    s = tc.Steersman()
    series = [0.095 + 0.0003 * np.sin(i) for i in range(30)]
    settled, net_drift, confident = s._is_settled(series)
    assert confident is True
    assert settled is True
    assert abs(net_drift) < 0.01


def test_settled_rejects_slow_monotonic_drift():
    """A slow monotonic climb (the C6b failure mode) is NOT settled."""
    s = tc.Steersman()
    # Per-sample slope 0.02: below the CL3 fast-growth deadband (0.05) yet the
    # level moves the full width of an attractor over the run.
    series = [0.02 * i for i in range(30)]
    settled, net_drift, confident = s._is_settled(series)
    assert confident is True
    assert settled is False
    assert net_drift > 0.0  # direction recovered so the caller can respond


def test_settled_insufficient_history_is_not_confident():
    """With < settle_window samples the detector refuses to judge."""
    s = tc.Steersman()
    settled, _, confident = s._is_settled([0.1, 0.1, 0.1])
    assert confident is False
    assert settled is False


def test_settled_hard_cap_admits_pinned_near_zero_metric():
    """Genuine-convergence hard cap: negligible absolute movement is settled
    even when it is large relative to a near-zero level (collapsed Fiedler)."""
    s = tc.Steersman()
    rng = np.random.default_rng(0)
    series = list(1.26e-5 + 1e-7 * rng.standard_normal(30))  # pinned ~1e-5
    settled, _, confident = s._is_settled(series)
    assert confident is True
    assert settled is True  # would fail the relative test; hard cap saves it


# ── CL3 (STABILITY): old fires, new does not, on a slow climb ─────────────────

def _slow_climb_deviation(n=40, slope=0.03):
    """Deviation series whose per-sample slope stays inside the CL3 deadband
    (-0.01, 0.05) yet accumulates to a large level — a synthetic WL-001."""
    return [slope * i for i in range(n)]


def test_cl3_old_fires_new_does_not_on_slow_climb():
    series = _slow_climb_deviation()
    s = tc.Steersman()

    trend = s._trend(series)
    # Precondition: the OLD rule WOULD declare STABLE here (trend in deadband).
    assert -0.01 <= trend <= 0.05
    assert _old_cl3_declares_stable(trend) is True

    blr_before = s._bridge_lr_scale
    signal, _ = s._stability_law(series)

    # NEW detector must NOT declare STABLE — the level is still climbing.
    assert not signal.startswith("STABLE"), signal
    assert "DRIFTING-UP" in signal
    # And the previously-idle actuator now responds (was pinned at 1.0).
    assert s._bridge_lr_scale < blr_before


def test_cl3_genuine_convergence_still_stable_and_idle():
    """A settled deviation plateau is STABLE under BOTH rules; actuator untouched."""
    s = tc.Steersman()
    series = [1.20 + 0.0004 * np.sin(i) for i in range(30)]

    trend = s._trend(series)
    assert _old_cl3_declares_stable(trend) is True  # old rule: STABLE

    blr_before = s._bridge_lr_scale
    signal, _ = s._stability_law(series)
    assert signal.startswith("STABLE"), signal          # new rule: STABLE too
    assert s._bridge_lr_scale == blr_before             # no spurious actuation


# ── CL1 (CONNECTIVITY): old fires, new does not, on a slow decline ────────────

def test_cl1_old_fires_new_does_not_on_slow_decline():
    """Fiedler declining at a per-sample slope inside the CL1 deadband
    (|slope| < 0.001) but drifting materially over the settle window."""
    s = tc.Steersman()
    # slope -0.0008: |slope| < fiedler_decline_thresh magnitude (0.001) -> old
    # else-branch (STABLE); net drift over 15 samples ~-0.011 on a ~0.09 level.
    series = [0.10 - 0.0008 * i for i in range(30)]

    trend = s._trend(series)
    assert -0.001 <= trend <= 0.001
    assert _old_cl1_declares_stable(trend) is True

    sw_before = s._spectral_weight
    signal, _ = s._connectivity_law(series)
    assert not signal.startswith("STABLE"), signal
    assert "DRIFTING-DOWN" in signal
    # Idle CL1 now boosts spectral regularization to arrest the decline.
    assert s._spectral_weight > sw_before


def test_cl1_genuine_convergence_still_stable_and_idle():
    s = tc.Steersman()
    series = [0.095 + 0.0003 * np.sin(i) for i in range(30)]

    trend = s._trend(series)
    assert _old_cl1_declares_stable(trend) is True

    sw_before = s._spectral_weight
    signal, _ = s._connectivity_law(series)
    assert signal.startswith("STABLE"), signal
    assert s._spectral_weight == sw_before


# ── Never-declare-STABLE regression guard (Task 4) ────────────────────────────

def test_settles_after_drift_then_plateau():
    """After a drift resolves into a plateau, the detector DOES declare STABLE —
    the fix must not permanently withhold convergence detection."""
    s = tc.Steersman()
    drift = [0.02 * i for i in range(20)]        # climb 0 -> 0.38
    plateau = [0.38 + 0.0003 * np.sin(i) for i in range(20)]  # settle at 0.38
    series = drift + plateau

    # During the drift the detector withholds STABLE...
    assert not s._stability_law(drift).__getitem__(0).startswith("STABLE")
    # ...but once genuinely plateaued it declares STABLE.
    signal, _ = s._stability_law(series)
    assert signal.startswith("STABLE"), signal


# ── Real WL-001 trace regression ──────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(_WL001), reason="WL-001 trace not on disk")
def test_wl001_real_trace_old_fires_new_does_not():
    """Replay the real WL-001 deviation trace. The stored run declared STABLE at
    every sample (v1) while deviation climbed 0.00 -> 2.11 with bridge_lr_scale
    pinned at 1.0. The v2 detector must declare STABLE at strictly fewer samples
    and NEVER during the steep climb, and its (counterfactual) actuator must
    respond instead of staying pinned."""
    data = json.load(open(_WL001))
    fl = data["feedback_log"]
    devs = [e["deviation_mean"] for e in fl]
    steps = [e["step"] for e in fl]

    # Ground truth: the stored run labelled every sample STABLE (v1 behaviour).
    stored_stable = sum(
        1 for e in fl if e["control_signals"].get("stability", "").startswith("STABLE")
    )
    assert stored_stable == len(fl)  # 101/101

    s = tc.Steersman()
    old_stable = new_stable = 0
    new_stable_during_steep_climb = 0
    for i in range(len(devs)):
        series = devs[: i + 1]
        trend = s._trend(series)
        if _old_cl3_declares_stable(trend):
            old_stable += 1
        signal, _ = s._stability_law(series)  # mutates bridge_lr (counterfactual)
        if signal.startswith("STABLE"):
            new_stable += 1
            # "Steep climb" = deviation still well below its 2.11 asymptote.
            if devs[i] < 1.9:
                new_stable_during_steep_climb += 1

    assert old_stable == len(fl)          # old rule fires everywhere
    assert new_stable < old_stable        # new detector fires strictly less
    assert new_stable_during_steep_climb == 0  # never STABLE while clearly moving
    # The actuator that was pinned at 1.0 for the entire real run is now driven
    # down by the corrected detector.
    assert s._bridge_lr_scale < 1.0


# ── Integration: observe_and_decide still wires the refactored laws ───────────

class _FakeLoRA:
    """Minimal stand-in for RhombiLoRALinear exposing what observe_and_decide
    reads: an effective_bridge tensor and a gradient-free lora_A."""

    def __init__(self, bridge):
        import torch

        self.effective_bridge = torch.as_tensor(bridge, dtype=torch.float32)

        class _P:
            grad = None

        self.lora_A = _P()


def test_observe_and_decide_integration_flags_drift():
    """Drive the full loop with slowly diverging bridges and confirm the
    refactored laws are wired in: stability is not falsely STABLE during the
    climb, and bridge_lr_scale is driven below 1.0."""
    pytest.importorskip("torch")
    import torch

    n = 4
    base = torch.eye(n)
    off = torch.zeros(n, n)
    off[0, 1] = off[1, 0] = 1.0  # push deviation from identity, growing slowly

    s = tc.Steersman()
    last_state = None
    for k in range(40):
        bridge = base + (0.03 * k) * off
        injected = {"a": _FakeLoRA(bridge)}
        last_state = s.observe_and_decide(step=k * 100, injected=injected)

    assert last_state is not None
    assert "stability" in last_state.control_signals
    assert "connectivity" in last_state.control_signals
    # Deviation climbed monotonically; CL3 must not be sitting on a bare STABLE.
    assert not last_state.control_signals["stability"].startswith(
        "STABLE (trend"
    ), last_state.control_signals["stability"]
    assert s._bridge_lr_scale < 1.0
