# C6b Fix Note — Steersman STABLE-Detector (v1 → v2)

**Date:** 2026-07-05
**Scope:** `scripts/train_cybernetic.py` — `Steersman` control laws CL1
(CONNECTIVITY) and CL3 (STABILITY).
**Detector version tag:** `steersman_detector_version = "v2-2026-07-05"`
**Status:** code fixed, CPU unit tests green (417/417). No completed run's
stored data was altered. This is the agreed prerequisite for Paper 4
re-measurement.

---

## 1. Root cause

The Steersman declared **STABLE** for a governed metric using a *trend deadband
alone*. Both defective laws computed a short-window (5-sample) OLS slope and, if
that slope sat inside a deadband, fell through to an `else` branch that (a)
declared `STABLE` and (b) left the actuator untouched.

Because the deadband bounds only the **per-sample slope**, a slow *monotonic*
drift whose slope stays just inside the band is declared STABLE indefinitely
while the **level** moves the full width of its attractor. A small instantaneous
slope is **necessary but not sufficient** for convergence: a sustained
below-threshold slope integrates to a large net displacement.

`docs/PAPER4_EXPOSURE_CLASSIFICATION.md` §2 diagnoses this; the deadband widths
(CL1 ±0.001, CL3 −0.01..+0.05) over a 100-sample run admit cumulative drift on
the order of the full attractor magnitude.

### Empirical proof — WL-001 (`results/channel-ablation/WL-001/results.json`)

101 feedback samples over a 10 000-step run:

| Quantity | Value |
|---|---|
| deviation_mean, first → last | **0.0000 → 2.1125** (monotonic climb) |
| max `deviation_trend` over the run | **0.04841** (never reached the 0.05 FAST-GROWTH threshold) |
| CL3 `bridge_lr_scale` | **1.000 for all 101 samples** (never dampened) |
| CL3 `stability` = `STABLE (...)` | **101 / 101 samples** |

Representative mid-climb samples the stored run labelled STABLE while the metric
was plainly still moving:

```
step=2000  deviation=0.8434  trend=0.04592  ->  "STABLE"   (deviation@+300 steps = 0.9737)
step=2400  deviation=1.0157  trend=0.04307  ->  "STABLE"   (deviation@+300 steps = 1.1356)
step=3000  deviation=1.2407  trend=0.03585  ->  "STABLE"   (deviation@+300 steps = 1.3349)
```

This is the documented live instance: *"declares STABLE while the controlled
metric still moves"* (`PAPER4_EXPOSURE_CLASSIFICATION.md` §2.1, row 30).

---

## 2. The fix (detector v2)

STABLE now requires **BOTH** (a) a small short-window trend (unchanged) **AND**
(b) the metric **genuinely settled** over a longer window. The settled test is
`Steersman._is_settled()`; a metric counts as settled only when, over
`SETTLE_WINDOW` samples, it neither **drifts** (small net displacement relative
to its level) nor **spreads** (low relative std).

When the short-window trend is inside the deadband but the metric is **not**
settled, the previously-idle branch now (i) does **not** emit STABLE and (ii)
responds in the drift's direction **using the same proportional gains as the
adjacent DECLINING / IMPROVING / FAST-GROWTH / CONVERGING branches** — it stops
sitting idle while the metric creeps. Genuinely-converged behaviour is
byte-identical to v1 (STABLE, actuator untouched).

The change is localised to the STABLE-declaration logic; **no loss magnitude,
gain, cap, deadband edge, or pair topology was retuned.** The CL1 and CL3
if/elif branches were extracted verbatim into `_connectivity_law` /
`_stability_law` so the logic is unit-testable without building torch adapters.

### Named constants (`scripts/train_cybernetic.py`, also `Steersman.__init__` params)

```python
STEERSMAN_DETECTOR_VERSION = "v2-2026-07-05"
SETTLE_WINDOW    = 15     # feedback samples judged for settling (>= window_size)
SETTLE_REL_DRIFT = 0.10   # max |net drift| / level over the settle window
SETTLE_REL_SPREAD= 0.10   # max std / level over the settle window
SETTLE_ABS_BAND  = 1e-3   # genuine-convergence HARD CAP for near-zero metrics
SETTLE_ABS_EPS   = 1e-6   # scale floor (avoids divide-by-zero)
```

### Before / after — STABLE-declaration conditions

Let `T` = short-window trend (`_trend`, window 5); `settled`, `net_drift`,
`confident` = `_is_settled(series)` over `SETTLE_WINDOW`.

**CL1 (CONNECTIVITY), Fiedler:**

| | v1 (defective) | v2 (fixed) |
|---|---|---|
| STABLE when | `-0.001 <= T <= 0.001` | `-0.001 <= T <= 0.001` **AND** `settled` |
| `-0.001<=T<=0.001` but drifting **down** | STABLE, spectral weight idle | `DRIFTING-DOWN`, boost spectral reg (same gain) |
| `-0.001<=T<=0.001` but drifting **up** | STABLE, idle | `DRIFTING-UP`, ease spectral reg (same gain) |
| insufficient history | STABLE | `SETTLING (collecting)`, actuator held (no false STABLE) |

**CL3 (STABILITY), deviation-from-identity:**

| | v1 (defective) | v2 (fixed) |
|---|---|---|
| STABLE when | `-0.01 <= T <= 0.05` | `-0.01 <= T <= 0.05` **AND** `settled` |
| `-0.01<=T<=0.05` but creeping **up** | STABLE, `bridge_lr` pinned | `DRIFTING-UP`, dampen `bridge_lr` (same FAST-GROWTH gain) |
| `-0.01<=T<=0.05` but creeping **down** | STABLE, idle | `DRIFTING-DOWN`, recover `bridge_lr` (same CONVERGING gain) |
| insufficient history | STABLE | `SETTLING (collecting)`, actuator held |

The `FAST GROWTH`, `CONVERGING`, `DECLINING`, `IMPROVING` branches and all their
thresholds/gains are unchanged.

### Regression guard against never-declaring-STABLE (Task 4)

- `_is_settled` returns `confident=False` only until `SETTLE_WINDOW` samples
  exist; once a drift resolves into a plateau it returns `settled=True` and
  STABLE is declared (test `test_settles_after_drift_then_plateau`).
- The `SETTLE_ABS_BAND` hard cap declares STABLE for any metric whose
  peak-to-peak band is ≤ `1e-3`, so a collapsed near-zero Fiedler (~1e-5) — whose
  tiny wiggle is large *relative* to its level — is still correctly called
  settled (test `test_settled_hard_cap_admits_pinned_near_zero_metric`).

---

## 3. Test evidence (`tests/test_steersman_c6b.py`, 12 tests)

- **Old-fires / new-does-not (synthetic).** `test_cl3_old_fires_new_does_not_on_slow_climb`
  and `test_cl1_old_fires_new_does_not_on_slow_decline` build "trend-flat but far
  from target" signals, assert the **old rule would have declared STABLE**
  (reference predicates `_old_cl3_declares_stable` / `_old_cl1_declares_stable`
  on the identical trend), and assert the **v2 detector does not** (emits
  `DRIFTING-*`) and that the previously-idle actuator now moves.
- **Genuine convergence — both STABLE.**
  `test_cl3_genuine_convergence_still_stable_and_idle` /
  `test_cl1_genuine_convergence_still_stable_and_idle`: a settled plateau is
  STABLE under **both** old rule and v2, with the actuator untouched.
- **Real WL-001 regression.** `test_wl001_real_trace_old_fires_new_does_not`
  replays the on-disk WL-001 deviation trace: the stored run labelled STABLE
  **101/101**; v2 labels STABLE **35/101**, **0** of them while deviation is
  still below its 2.11 asymptote, and drives the counterfactual `bridge_lr_scale`
  below 1.0 (the real run left it pinned at 1.0).
- **Integration.** `test_observe_and_decide_integration_flags_drift` runs the
  full `observe_and_decide` loop with minimal fake adapters and confirms the
  refactored laws are wired in.
- **Unit + version.** `_is_settled` acceptance/rejection/insufficiency/hard-cap;
  `STEERSMAN_DETECTOR_VERSION` tag present.

**Results:** `tests/test_steersman_c6b.py` → 12 passed. Full suite under the
`falco` env (CPU-only, `CUDA_VISIBLE_DEVICES=""`) → **417 passed** (405 prior +
12 new). No GPU touched.

---

## 4. Version discipline

`train_cybernetic.py` now writes `steersman_detector_version` into each run's
`config.json`. Every **future** run self-documents that it used the corrected
detector. **Past** runs' `config.json` lack this field — by definition detector
**v1** (defective). No stored run data was altered or reinterpreted by this fix.

---

## 5. Re-measurement requirement

> **Adaptive-governed Paper 4 numbers were produced under the defective v1
> detector and MUST be re-measured under detector v2 before publication.**

Scope is defined by `docs/PAPER4_EXPOSURE_CLASSIFICATION.md`:

- **P1 (headline, re-measure first):** O-001 473,622:1 / Fiedler 1.1e-5;
  H-ch6 70,404:1 / Fiedler 9e-5; T-001r2 41,564:1 / Fiedler 1.91e-4;
  WL-001 & R-001 collapse-regime numbers (8.7e-6:1 / 1.26e-5 and 8.9e-6:1 /
  1.2e-5) including the deviation-2.12 / CL3-non-intervention instance; CW-001
  (if promoted). (§6)
- **P2/P3:** supporting and insensitive numbers, re-derived in the same campaign.
- **FIXED-WEIGHT runs** (`--fixed-contrastive`, spectral-only) remain **partially
  exposed** — CL2 is out of the loop but CL1 (spectral weight) and CL3 (bridge
  LR) carried the same defective STABLE logic; re-measure per their rows. (§2.1)
- **CONTROLLER-FREE** numbers (all P0 rows; FI-003 decay dynamics) are **immune** —
  no re-measurement. (§7)

Procedure per the classification's closing note: land this fix, then run the P1
list under v2, then P2 within the same campaign.
