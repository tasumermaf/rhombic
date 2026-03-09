# Phase 2A: Bridge-Level Adapter Merging

> **Experiment:** Interpolate final bridge matrices between task-specific
> adapters and measure structural preservation across the merge trajectory.
> No GPU required — pure linear algebra on 36-parameter bridges.

## Summary

| Pair | Linear (R^2) | Phase Transition | Best Structure Retention |
|------|-------------|------------------|-------------------------|
| alpaca_vs_code | NO (R^2=0.948) | Max residual: 0.00351 | cos(A)@0.1 = 1.0000 |
| alpaca_vs_math | NO (R^2=0.735) | Max residual: 0.00437 | cos(A)@0.1 = 1.0000 |
| code_vs_math | NO (R^2=0.782) | Max residual: 0.00347 | cos(A)@0.1 = 1.0000 |

## alpaca_vs_code

**Adapters:** 112 common positions
**Source A (alpaca):** Fiedler = 0.04005 +/- 0.01634, Deviation = 0.1988
**Source B (code):** Fiedler = 0.01800 +/- 0.00724, Deviation = 0.0338

### Interpolation Curve

| alpha | Fiedler | Deviation | dist(A) | dist(B) | cos(eig,A) | cos(eig,B) |
|-------|---------|-----------|---------|---------|------------|------------|
| 0.0 | 0.04005 | 0.1988 | 0.00000 | 0.18905 | 1.0000 | 0.9997 |
| 0.1 | 0.03666 | 0.1803 | 0.01891 | 0.17015 | 1.0000 | 0.9998 |
| 0.2 | 0.03346 | 0.1619 | 0.03781 | 0.15124 | 1.0000 | 0.9998 |
| 0.3 | 0.03039 | 0.1436 | 0.05672 | 0.13234 | 1.0000 | 0.9999 |
| 0.4 | 0.02735 | 0.1255 | 0.07562 | 0.11343 | 0.9999 | 0.9999 |
| 0.5 | 0.02448 | 0.1076 | 0.09453 | 0.09453 | 0.9999 | 0.9999 |
| 0.6 | 0.02187 | 0.0901 | 0.11343 | 0.07562 | 0.9998 | 1.0000 |
| 0.7 | 0.01965 | 0.0732 | 0.13234 | 0.05672 | 0.9998 | 1.0000 |
| 0.8 | 0.01817 | 0.0572 | 0.15124 | 0.03781 | 0.9998 | 1.0000 |
| 0.9 | 0.01808 | 0.0431 | 0.17015 | 0.01891 | 0.9997 | 1.0000 |
| 1.0 | 0.01800 | 0.0338 | 0.18905 | 0.00000 | 0.9997 | 1.0000 |

### Linearity Analysis

**R^2 (Fiedler vs alpha):** 0.94835
**Max residual:** 0.003507
**Verdict:** Fiedler trajectory is NON-LINEAR (R^2 = 0.948).
Phase transition or curvature detected in the merge trajectory. Some alpha values cause disproportionate structural change.

### Structure Preservation

**Eigenspectrum retention at alpha=0.1 (90% task A):** cos = 1.0000
Task A's spectral structure is almost perfectly preserved with 10% contamination from task B.
**Eigenspectrum retention at alpha=0.9 (90% task B):** cos = 1.0000

### Random Baseline Comparison (alpha=0.5)

| Metric | Task Merge | Random Merge | Better |
|--------|-----------|--------------|--------|
| Fiedler | 0.02448 | 0.88540 | Random |
| Deviation | 0.1076 | 2.5345 | Task |
| Eig cos(A) | 0.9999 | 0.5177 | Task |

**PASS** — Bridge merging preserves more task structure than merging with random noise. The merged bridge retains meaningful spectral properties from its sources.

### Per-Module-Type Sensitivity

Sensitivity = Fiedler change from alpha=0.0 to alpha=0.5, normalized by source Fiedler.

| Module | Fiedler@0.0 | Fiedler@0.5 | Delta | Sensitivity |
|--------|-------------|-------------|-------|-------------|
| q_proj | 0.05034 | 0.02873 | 0.02161 | 0.429 |
| k_proj | 0.03934 | 0.02529 | 0.01405 | 0.357 |
| v_proj | 0.03589 | 0.02256 | 0.01333 | 0.371 |
| o_proj | 0.03463 | 0.02133 | 0.01329 | 0.384 |

**Most affected by merging:** q_proj (sensitivity = 0.429)
**Least affected by merging:** k_proj (sensitivity = 0.357)

## alpaca_vs_math

**Adapters:** 112 common positions
**Source A (alpaca):** Fiedler = 0.04005 +/- 0.01634, Deviation = 0.1988
**Source B (math):** Fiedler = 0.02760 +/- 0.01582, Deviation = 0.0689

### Interpolation Curve

| alpha | Fiedler | Deviation | dist(A) | dist(B) | cos(eig,A) | cos(eig,B) |
|-------|---------|-----------|---------|---------|------------|------------|
| 0.0 | 0.04005 | 0.1988 | 0.00000 | 0.17126 | 1.0000 | 0.9998 |
| 0.1 | 0.03701 | 0.1831 | 0.01713 | 0.15413 | 1.0000 | 0.9999 |
| 0.2 | 0.03407 | 0.1677 | 0.03425 | 0.13701 | 1.0000 | 0.9999 |
| 0.3 | 0.03115 | 0.1526 | 0.05138 | 0.11988 | 1.0000 | 0.9999 |
| 0.4 | 0.02853 | 0.1379 | 0.06850 | 0.10276 | 0.9999 | 0.9999 |
| 0.5 | 0.02667 | 0.1237 | 0.08563 | 0.08563 | 0.9999 | 1.0000 |
| 0.6 | 0.02560 | 0.1101 | 0.10276 | 0.06850 | 0.9999 | 1.0000 |
| 0.7 | 0.02547 | 0.0973 | 0.11988 | 0.05138 | 0.9998 | 1.0000 |
| 0.8 | 0.02560 | 0.0857 | 0.13701 | 0.03425 | 0.9998 | 1.0000 |
| 0.9 | 0.02626 | 0.0759 | 0.15413 | 0.01713 | 0.9998 | 1.0000 |
| 1.0 | 0.02760 | 0.0689 | 0.17126 | 0.00000 | 0.9998 | 1.0000 |

### Linearity Analysis

**R^2 (Fiedler vs alpha):** 0.73464
**Max residual:** 0.004371
**Verdict:** Fiedler trajectory is NON-LINEAR (R^2 = 0.735).
Phase transition or curvature detected in the merge trajectory. Some alpha values cause disproportionate structural change.

### Structure Preservation

**Eigenspectrum retention at alpha=0.1 (90% task A):** cos = 1.0000
Task A's spectral structure is almost perfectly preserved with 10% contamination from task B.
**Eigenspectrum retention at alpha=0.9 (90% task B):** cos = 1.0000

### Random Baseline Comparison (alpha=0.5)

| Metric | Task Merge | Random Merge | Better |
|--------|-----------|--------------|--------|
| Fiedler | 0.02667 | 0.88272 | Random |
| Deviation | 0.1237 | 2.5417 | Task |
| Eig cos(A) | 0.9999 | 0.5204 | Task |

**PASS** — Bridge merging preserves more task structure than merging with random noise. The merged bridge retains meaningful spectral properties from its sources.

### Per-Module-Type Sensitivity

Sensitivity = Fiedler change from alpha=0.0 to alpha=0.5, normalized by source Fiedler.

| Module | Fiedler@0.0 | Fiedler@0.5 | Delta | Sensitivity |
|--------|-------------|-------------|-------|-------------|
| q_proj | 0.05034 | 0.03101 | 0.01932 | 0.384 |
| k_proj | 0.03934 | 0.03089 | 0.00845 | 0.215 |
| v_proj | 0.03589 | 0.02385 | 0.01205 | 0.336 |
| o_proj | 0.03463 | 0.02094 | 0.01369 | 0.395 |

**Most affected by merging:** o_proj (sensitivity = 0.395)
**Least affected by merging:** k_proj (sensitivity = 0.215)

## code_vs_math

**Adapters:** 112 common positions
**Source A (code):** Fiedler = 0.01800 +/- 0.00724, Deviation = 0.0338
**Source B (math):** Fiedler = 0.02760 +/- 0.01582, Deviation = 0.0689

### Interpolation Curve

| alpha | Fiedler | Deviation | dist(A) | dist(B) | cos(eig,A) | cos(eig,B) |
|-------|---------|-----------|---------|---------|------------|------------|
| 0.0 | 0.01800 | 0.0338 | 0.00000 | 0.06624 | 1.0000 | 0.9999 |
| 0.1 | 0.01682 | 0.0337 | 0.00662 | 0.05961 | 1.0000 | 0.9999 |
| 0.2 | 0.01599 | 0.0349 | 0.01325 | 0.05299 | 1.0000 | 0.9999 |
| 0.3 | 0.01619 | 0.0371 | 0.01987 | 0.04636 | 1.0000 | 0.9999 |
| 0.4 | 0.01676 | 0.0402 | 0.02649 | 0.03974 | 1.0000 | 0.9999 |
| 0.5 | 0.01781 | 0.0439 | 0.03312 | 0.03312 | 1.0000 | 1.0000 |
| 0.6 | 0.01941 | 0.0482 | 0.03974 | 0.02649 | 1.0000 | 1.0000 |
| 0.7 | 0.02139 | 0.0529 | 0.04636 | 0.01987 | 1.0000 | 1.0000 |
| 0.8 | 0.02341 | 0.0580 | 0.05299 | 0.01325 | 0.9999 | 1.0000 |
| 0.9 | 0.02551 | 0.0633 | 0.05961 | 0.00662 | 0.9999 | 1.0000 |
| 1.0 | 0.02760 | 0.0689 | 0.06624 | 0.00000 | 0.9999 | 1.0000 |

### Linearity Analysis

**R^2 (Fiedler vs alpha):** 0.78233
**Max residual:** 0.003473
**Verdict:** Fiedler trajectory is NON-LINEAR (R^2 = 0.782).
Phase transition or curvature detected in the merge trajectory. Some alpha values cause disproportionate structural change.

### Structure Preservation

**Eigenspectrum retention at alpha=0.1 (90% task A):** cos = 1.0000
Task A's spectral structure is almost perfectly preserved with 10% contamination from task B.
**Eigenspectrum retention at alpha=0.9 (90% task B):** cos = 1.0000

### Random Baseline Comparison (alpha=0.5)

| Metric | Task Merge | Random Merge | Better |
|--------|-----------|--------------|--------|
| Fiedler | 0.01781 | 0.89145 | Random |
| Deviation | 0.0439 | 2.5636 | Task |
| Eig cos(A) | 1.0000 | 0.4853 | Task |

**PASS** — Bridge merging preserves more task structure than merging with random noise. The merged bridge retains meaningful spectral properties from its sources.

### Per-Module-Type Sensitivity

Sensitivity = Fiedler change from alpha=0.0 to alpha=0.5, normalized by source Fiedler.

| Module | Fiedler@0.0 | Fiedler@0.5 | Delta | Sensitivity |
|--------|-------------|-------------|-------|-------------|
| q_proj | 0.01657 | 0.01922 | 0.00265 | 0.160 |
| k_proj | 0.02241 | 0.02073 | 0.00168 | 0.075 |
| v_proj | 0.01682 | 0.01611 | 0.00071 | 0.042 |
| o_proj | 0.01622 | 0.01517 | 0.00105 | 0.065 |

**Most affected by merging:** q_proj (sensitivity = 0.160)
**Least affected by merging:** v_proj (sensitivity = 0.042)

---

## Conclusions

**Phase transitions detected in 3/3 pairs:** alpaca_vs_code, alpaca_vs_math, code_vs_math. Bridge merging is NOT uniformly smooth — certain task combinations exhibit nonlinear structural reorganization at intermediate alpha values.

**Task structure is preserved at low alpha values.** At alpha=0.1 (10% contamination), the source task's eigenspectrum is retained with cos > 0.95 in all pairs. Small amounts of cross-task information can be injected without destroying the dominant task's structure.

---

*Generated by `scripts/experiment_bridge_merge.py`*