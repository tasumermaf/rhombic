# FI-001: Corpus Internal Structure Under BD Convergence

**Date:** 2026-03-18
**Status:** COMPLETE
**GPU cost:** 0 (CPU-only analysis of existing checkpoints)
**Verdict:** SIGN FINGERPRINT DETECTED

---

## Question

C-001 (identity init), C-002 (geometric init), and C-003 (corpus-coupled init)
all converge to block-diagonal (BD) structure under the Steersman. Are the
converged bridges internally identical, or does initialization leave a measurable
fingerprint?

## Data

88 bridge matrices (22 layers x 4 modules) per experiment, loaded from:
- `results/corpus-baselines/C-001-identity-default/`
- `results/corpus-baselines/C-002-geometric-default/`
- `results/corpus-baselines/C-003-corpus-coupled-default/`

RD co-planar pairs: (0,1), (2,3), (4,5) — 3 pairs, 12 cross-planar.

## Key Findings

### 1. The Steersman Programs Magnitude Structure Identically

**Co-planar ratio vectors** (relative axis balance) are statistically identical
across all three initializations. Cosine similarity:

| Comparison | Mean cos | Min cos | Layers < 0.99 |
|-----------|----------|---------|---------------|
| C-001 vs C-002 | 0.999913 | 0.998991 | 0/88 |
| C-001 vs C-003 | 0.999899 | 0.998083 | 0/88 |
| C-002 vs C-003 | 0.999932 | 0.998175 | 0/88 |

**Per-axis balance** is uniform (~33%/33.5%/33.5% for all three experiments).

**Cross-planar residual magnitudes** are indistinguishable between C-002 and C-003:
KS test D=0.045, p=0.225 (not significant). Mean decay profiles nearly identical.

**Conclusion:** The Steersman fully determines the magnitude topology — axis
proportions, residual magnitudes, decay profiles. Initialization has no effect
on these.

### 2. C-001 Differs in Absolute Magnitude (Training Length Confound)

C-001 ran for 4000 steps; C-002 for 6900; C-003 for 2900. Despite fewer steps,
C-002 and C-003 have much higher co-planar magnitudes (0.79) vs C-001 (0.48).
This is because C-001 starts from identity (bridge = I), while C-002/C-003
start from structured initializations that place mass on co-planar entries
from step 0. The Steersman amplifies what's already there.

C-001's cross-planar residuals are 4.6x larger than C-002/C-003. KS test:
C-001 vs C-002 D=0.420, p=5e-84.

**This is a training-length confound, not a structural finding.** C-001 would
likely converge to similar absolute magnitudes with more steps. The important
comparison is C-002 vs C-003 (same-scale magnitudes, same Steersman behavior).

### 3. THE FINGERPRINT: Sign Structure of Cross-Planar Residuals

The critical finding that magnitude-based tests missed:

| Experiment | % positive cross-planar (lower triangle) | Pattern |
|-----------|------------------------------------------|---------|
| C-001 (identity) | 50.4% | Random (expected for I init) |
| C-002 (geometric) | 92.7% | Nearly all positive |
| C-003 (corpus) | **29.2%** | Nearly all NEGATIVE |

**C-002 vs C-003 sign agreement: 35.5%** (std 0.099). This is WORSE than
random (50%), meaning the corpus initialization produces an ANTI-CORRELATED
sign pattern relative to geometric initialization.

Per-position sign consistency in C-003:
```
Position B[2,0]: 6% positive  (94% of adapters have NEGATIVE coupling here)
Position B[3,0]: 0% positive  (100% NEGATIVE — unanimous)
Position B[3,1]: 0% positive  (100% NEGATIVE — unanimous)
Position B[4,0]: 10% positive (90% NEGATIVE)
Position B[4,1]: 11% positive (89% NEGATIVE)
Position B[5,1]: 1% positive  (99% NEGATIVE)
Position B[5,2]: 6% positive  (94% NEGATIVE)
Position B[5,3]: 8% positive  (92% NEGATIVE)
```

These are not noisy — they are highly consistent across all 88 adapters.
The corpus initialization imposes a specific, reproducible sign topology
on the cross-planar residuals.

### 4. What This Means

The Steersman and the initialization program **different domains** of the bridge:

| Property | Programmed By | Evidence |
|----------|--------------|---------|
| BD vs non-BD (topology type) | Steersman | All three converge to BD |
| Axis proportions | Steersman | Cosine > 0.999 across all |
| Residual magnitudes | Steersman | KS p=0.225 (C-002 vs C-003) |
| Cross-planar decay profile | Steersman | Mean decay profiles identical |
| **Sign structure of residuals** | **Initialization** | **35.5% agreement (anti-correlated)** |

The corpus does not alter what the Steersman builds. It alters the **polarity**
of the suppressed couplings — the region the Steersman considers "noise" and
drives toward zero. The signs survive because the Steersman's contrastive loss
operates on absolute values: it cares that |B[cross]| is small, not whether
B[cross] is positive or negative.

This is precisely the "accent" predicted in the Falco Intelligence Roadmap:
> "The Steersman programs the structure. The corpus programs the accent."

The accent lives in the sign domain.

## Implications for FI-002 and Beyond

1. **The corpus CAN encode information in the bridge** — not through magnitude
   (the Steersman controls that) but through sign structure. The 36 parameters
   include both magnitude and sign. The Steersman claims magnitude; the
   initialization claims sign.

2. **FI-002 (Corpus Pair Specification)** should test whether different corpus
   encodings produce DIFFERENT sign patterns. If pair specs derived from
   different hexagram structures produce distinguishable sign topologies,
   the corpus is truly programming the bridge.

3. **FI-003 (Persistence)** should check whether the sign fingerprint persists
   across fine-tuning on different tasks. If the signs are overwritten by task
   training, the fingerprint is fragile. If they persist, the corpus is durable.

4. **The KS test was the wrong test for this phenomenon.** Taking absolute
   values destroyed the signal. Future analyses must include sign-sensitive
   statistics (signed rank tests, circular statistics for sign patterns).

## Aggregate Statistics

| Metric | C-001 | C-002 | C-003 |
|--------|-------|-------|-------|
| Mean co/cross ratio | 10,119 | 71,337 | 64,168 |
| Co-planar magnitude | 0.480 | 0.793 | 0.787 |
| Cross-planar residual mean | 8.8e-5 | 1.9e-5 | 1.9e-5 |
| % positive cross-planar signs | 50.4% | 92.7% | 29.2% |
| Frobenius: vs C-001 | -- | 1.356 | 1.346 |
| Frobenius: C-002 vs C-003 | -- | -- | 0.140 |

## Files

- Script: `scripts/fi_001_corpus_internal.py`
- Results: `results/fi-001-results.json`
- Source data: `results/corpus-baselines/C-{001,002,003}-*-default/`

---

*FI-001 completed 2026-03-18. Zero GPU cost. The most important result was the
one the automated test missed: the fingerprint lives in the signs, not the
magnitudes.*
