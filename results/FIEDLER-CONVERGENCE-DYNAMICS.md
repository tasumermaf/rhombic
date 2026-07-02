# Steersman Convergence Dynamics: The Three-Phase Fiedler Trajectory

**Date:** 2026-03-19 (CW11)
**Data sources:** Seed-43, Seed-44, C-003, T-001r2 (all 10K-step runs, 100-step resolution)
**Cross-validated with:** P-000 (FI-002, 2000 steps at time of analysis)

## Finding

The Steersman's contrastive training produces a **universal three-phase Fiedler
trajectory**: rapid descent → controlled rebound → final convergence. The rebound
is not noise — it is a deterministic consequence of the adaptive contrastive weight
controller, and its magnitude ratio (~4×) is invariant across init strategies,
channel counts, and corpus conditions.

## The Three Phases

### Phase I: Contrastive Overshoot (steps 0–900)

The contrastive weight starts at its maximum (0.10) and drives aggressive
block-diagonal formation. Fiedler drops 10–100× in the first 500 steps.
Identity-init reaches Fiedler < 0.001 by step 200; corpus-coupled by step 400.

During this phase, the contrastive loss becomes strongly negative (BD forming),
the spectral loss drops from its initial value, and the spectral gap collapses
from ~0.16–0.28 to ~0.001.

**Key mechanism:** The adaptive controller REDUCES contrastive weight as the
Fiedler target is achieved. Weight decays: 0.100 → 0.073 → 0.043 → **0.025
(floor)** by step ~1000.

### Phase II: Controller-Induced Rebound (steps 900–4500)

With contrastive weight at floor (0.025), the drive force for BD formation
is reduced by 4×. The spectral regularizer (constant weight) now has
proportionally more influence. Fiedler RISES by ~4× from its minimum.

**Critically: co/cross ratio CONTINUES RISING through the rebound.** The
block-diagonal structure keeps strengthening even as the Fiedler eigenvalue
increases. This means BD formation and spectral gap regulation are partially
independent objectives — the contrastive loss magnitude grows monotonically
(more negative) throughout, but the reduced weight means the gradient signal
is smaller.

During Phase II, Fiedler and co/cross are POSITIVELY correlated (r = 0.3–0.7).
This is the opposite of their equilibrium relationship.

### Phase III: Reconvergence (steps 4500–10000)

The accumulated contrastive gradient, despite the reduced weight, eventually
dominates. Fiedler reconverges toward its equilibrium value. During this phase,
Fiedler and co/cross are strongly ANTI-correlated (r = -0.94 across all
experiments), which is the expected BD-regime relationship: as Fiedler
decreases, co/cross increases.

Final equilibria:
- n=3 (6×6 bridges): Fiedler ~0.000085, Co/Cross ~70,000
- n=4 (8×8 bridges): Fiedler ~0.000191, Co/Cross ~41,500

## Quantitative Summary

| Experiment | Init | Min Fiedler (step) | Max Fiedler (step) | Rebound Ratio | Eq. Fiedler |
|------------|------|-------------------|-------------------|---------------|-------------|
| Seed-43 | Identity | 0.000106 (900) | 0.000455 (3500) | **4.3×** | 0.000085 |
| Seed-44 | Identity | 0.000115 (600) | 0.000455 (4000) | **4.0×** | 0.000092 |
| C-003 | Corpus-coupled | 0.000115 (900) | 0.000472 (4200) | **4.1×** | 0.000085 |
| T-001r2 | Tesseract | 0.000212 (500) | 0.000997 (3600) | **4.7×** | 0.000191 |

**Rebound ratio: 4.3 ± 0.3× (mean ± std across 4 experiments).**

## The Mechanism: Control-Theoretic Overshoot

The rebound is a classic **control-theoretic overshoot** caused by the adaptive
gain schedule:

1. The controller reduces contrastive weight as Fiedler drops (target met)
2. The weight reaches its floor (0.025) right as Fiedler hits minimum
3. At floor weight, the drive force is 4× weaker than at start
4. The spectral regularizer (constant weight) temporarily dominates
5. Fiedler rises until accumulated contrastive gradient overwhelms spectral force
6. The system reconverges to final equilibrium

The ~4× rebound ratio is determined by the ratio of initial weight to floor
weight (0.10 / 0.025 = 4.0). This is not coincidence — it is the direct
consequence of the controller's gain range.

## Contrastive Weight Controller Trajectory

| Step | Contr. Weight | Fiedler (Seed-43) | Phase |
|------|--------------|-------------------|-------|
| 100 | 0.1000 | 0.003246 | I — rapid descent |
| 500 | 0.0729 | 0.000121 | I — near minimum |
| 1000 | 0.0430 | 0.000108 | I→II transition |
| 1500 | **0.0250** | 0.000190 | II — rebound begins |
| 2000 | 0.0250 | 0.000263 | II — rebound continues |
| 3500 | 0.0250 | 0.000455 | II — rebound peak |
| 5000 | 0.0250 | 0.000376 | III — reconvergence |
| 8000 | 0.0250 | 0.000091 | III — equilibrium |

The weight hits floor at step ~1500, exactly when the rebound begins.

## Fiedler–CoCross Phase Coupling

| Phase | Fiedler–CoCross Correlation | Interpretation |
|-------|----------------------------|----------------|
| II (1000–4000) | **r = +0.3 to +0.7** | Both rising together — anomalous |
| III (5000–10000) | **r = −0.94** | Anti-correlated — normal BD regime |

During Phase II, the system is in a transient state where both metrics increase
simultaneously. This is only possible because Fiedler (spectral gap property)
and co/cross (magnitude ratio property) measure different aspects of BD
formation. The contrastive loss drives magnitude suppression (raising co/cross)
while the spectral structure temporarily loosens (raising Fiedler).

## Convergence Speed and Init Strategy

| Init Strategy | Steps to Fiedler < 0.001 | Implication |
|---------------|--------------------------|-------------|
| Identity | **200** | Near-zero cross-planar from start → fast |
| Tesseract (identity variant) | **300** | Slightly slower (8×8 vs 6×6) |
| Corpus-coupled | **400** | Non-zero cross-planar init → 2× slower |

Identity-init starts with near-zero cross-planar values (it's an identity
matrix), so the contrastive loss has less work to do. Corpus-coupled init
carries corpus-derived off-diagonal structure that must be suppressed.

## Implications

### For Paper 4
The three-phase trajectory is a universal feature of Steersman training.
Any paper claiming "monotonic convergence" would be incorrect. The rebound
is real, predictable, and mechanistically explained.

### For FI-002
P-000 (3K-step run) does NOT show the rebound despite identical init strategy
(corpus-coupled) and contrastive weight trajectory as C-003 (10K-step run).
Both hit weight floor (0.025) at step ~1500. But P-000's Fiedler continues
declining to 0.000031 at step 2700, while C-003 rebounds to 0.000472.

**The difference is learning rate schedule.** P-000's 3K schedule has the LR
nearly fully decayed by step 2700 (90% through training), while C-003's 10K
schedule still has high LR at step 2700 (27% through). The rebound requires
sufficient remaining LR for the spectral regularizer gradient to overcome the
contrastive gradient at reduced weight. With decayed LR, gradients are too
small to cause the rebound.

**Refinement:** The rebound is caused by the interaction of (1) contrastive
weight decay and (2) LR schedule. Short runs with compressed LR schedules
may bypass Phase II entirely, converging directly from Phase I to equilibrium.
P-000 reaches Fiedler 0.000031 at step 2700 — DEEPER than any 10K experiment
reaches at its Phase I minimum (~0.000106-0.000212).

### For the Steersman
The rebound could be eliminated by holding contrastive weight constant or
by using a slower decay schedule. Whether this is desirable depends on
whether the Phase II rebound serves any useful purpose (e.g., exploration
of the loss landscape). The current schedule reaches equilibrium by step
~8000 regardless — the rebound delays but does not prevent convergence.

### For Deployment
The non-monotonic trajectory means that intermediate checkpoints (steps
1000–4000) may have WORSE Fiedler values than both earlier and later
checkpoints. Any checkpoint selection strategy should use final-third
checkpoints (steps 7000+), not the "best so far" approach that works
for monotonic training.

## Contrast: Spectral-Only Experiments Show No Rebound

The three-phase pattern is specific to **BD-forming experiments** (Regime 1).
Spectral-only experiments (Regime 2: H-ch3, H-ch12) show monotonically
INCREASING Fiedler — from ~0.01 to ~0.095/0.102 over 10K steps. No overshoot,
no rebound. The spectral regularizer alone produces smooth convergence to its
target without the control-theoretic dynamics.

This confirms the mechanism: the rebound requires the contrastive weight
controller, which only activates when contrastive loss is driving BD formation.
Spectral-only experiments never trigger contrastive weight decay because the
contrastive loss never reaches the threshold that causes the controller to
reduce the weight.

| Experiment | Regime | Min Fiedler | Max Fiedler | Rebound? |
|------------|--------|-------------|-------------|----------|
| Seed-43 | BD | 0.000106 | 0.000455 | YES (4.3×) |
| exp3 (O-001) | BD | 0.000095 | 0.000470 | YES (4.9×) |
| H-ch3 | Spectral | 0.012 (init) | 0.095 (final) | NO (monotonic) |
| H-ch12 | Spectral | 0.033 (init) | 0.102 (final) | NO (monotonic) |

## Val Loss Is Orthogonal to Fiedler Dynamics

Val loss decreases monotonically through all three phases (0.476 → 0.400),
completely unaffected by the Fiedler rebound. Task performance (language
modeling) is independent of bridge topological dynamics. The model learns
language while the Steersman shapes the bridge — two orthogonal optimization
surfaces sharing the same parameters but not interfering.

## Methodological Notes

- 100-step checkpoint resolution for 10K-step experiments
- Fiedler values are means across 88 adapters
- Co/cross ratios are means across 88 adapters
- Correlations computed within phase windows using Pearson r
- Rebound ratio = max(Fiedler in Phase II) / min(Fiedler in Phase I)
- Phase boundaries determined by Fiedler velocity sign changes
