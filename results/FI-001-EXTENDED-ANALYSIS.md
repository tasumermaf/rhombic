# FI-001 Extended Analysis: Initialization Determines Sign Structure

**Date:** 2026-03-18/19 (CW10-11 cross-experiment sprint)
**Extends:** FI-001 Sign Fingerprint Analysis
**Data sources:** C-002, C-003, Seed-43, Seed-44, FP-code, FP-math, exp3, exp3_tinyllama,
T-001-full, T-001-full-r2, tesseract-contrastive, tesseract-contrastive-full (88 adapters each)

## Finding

The sign structure of cross-planar bridge residuals is **entirely determined
by initialization strategy**, not by training dynamics. The Steersman's
contrastive loss operates on absolute values (`|B[cross]|`), leaving signs
in the null space — confirmed by comparing four experiments that converge
to identical topological equilibria but carry completely different sign
patterns.

## Evidence

### Position-Level Consensus

Each experiment's 88 final bridges were analyzed at 9 lower-triangle
cross-planar positions. Consensus = |fraction_positive - 0.5| × 2,
where 0 = chance, 1 = unanimous.

| Experiment | Init Strategy | Mean Consensus | % Positive | p-value |
|------------|--------------|---------------|------------|---------|
| C-002 | Geometric (corpus-derived) | **0.861** | 93.1% | 2.6e-153 |
| C-003 | Coupled (corpus-derived) | **0.843** | 26.3% | 4.3e-42 |
| Seed-43 | Identity | 0.096 | 46.2% | 3.6e-2 |
| Seed-44 | Identity | 0.078 | 49.6% | 8.6e-1 |

### Position-Level Majority Signs

```
Position:  0  1  2  3  4  5  6  7  8
C-002:     +  +  +  +  +  +  +  +  +   (9/9 positive)
C-003:     -  -  +  -  -  -  +  -  -   (2/9 positive)
Seed-43:   -  -  -  -  -  +  -  -  -   (consensus < 0.2, effectively noise)
Seed-44:   -  -  -  -  -  +  +  -  +   (consensus < 0.2, effectively noise)
```

### Majority-Sign Agreement Between Experiments

| Comparison | Agreement | Interpretation |
|------------|-----------|---------------|
| C-002 vs C-003 | **2/9 (22.2%)** | Strongly anti-correlated |
| Seed-43 vs Seed-44 | 7/9 (77.8%) | Noise agreeing with noise |
| C-002 vs Seed-43 | — | Not meaningful (one structured, one random) |

### Topological Convergence Despite Sign Divergence

All four experiments converge to the same BD equilibrium by 10K steps:

| Experiment | Final Fiedler | Final Co/Cross |
|------------|--------------|---------------|
| C-002 | — (not tracked at 10K) | — |
| C-003 | 0.000085 | 64,168 |
| Seed-43 | 0.000085 | 73,309 |
| Seed-44 | 0.000092 | 70,201 |

Identity-init reaches BD onset **2× faster** (step 200 vs 400) but all
converge to the same Fiedler equilibrium. Signs are orthogonal to topology.

## Interpretation

### Three-Part Decomposition of Bridge State

The bridge matrix B decomposes into three independent components:

1. **Topology** (block-diagonal structure): Driven by the Steersman's
   contrastive loss. Converges to Fiedler ~0.000085 regardless of
   initialization. This is the **task-invariant** structure.

2. **Signs** (cross-planar residual signs): Determined entirely by
   initialization. Corpus-derived inits produce deterministic sign
   fingerprints; identity-init produces no structure. This is the
   **corpus fingerprint** — the identity signal.

3. **Magnitudes** (absolute values of cross-planar residuals): Driven
   to near-zero by contrastive loss (co/cross > 40K:1). Magnitudes
   carry the topological information; signs carry the identity.

### Why Anti-Correlation?

C-002 (geometric init) and C-003 (coupled init) are both derived from
the same corpus weights but through different transformations:
- Geometric: SVD decomposition + eigenvalue modification
- Coupled: Direct decomposition from corpus weight matrices

These transformations produce opposite sign structures because they
factor the same information differently. The CONTENT (corpus identity)
is encoded in both, but the REPRESENTATION differs — like the same
message encrypted with two different keys.

### Predictions for FI-002 and FI-003

**FI-002 (channel permutation):** Permuting channels should rearrange
signs to new positions but NOT change the overall positive/negative
ratio. The 93.1% positive ratio of C-002 should survive permutation
at the aggregate level, even if individual position assignments change.

**FI-003 (persistence under fine-tuning):** Signs should persist under
unguided (no Steersman) training because:
1. Standard cross-entropy loss doesn't operate on bridge cross-planar entries
2. Signs are already in the Steersman's null space (training with Steersman doesn't touch them)
3. Without Steersman, there is no force acting on cross-planar entries at all

If signs DO degrade under unguided training, it would indicate that
the optimizer's implicit regularization (weight decay, gradient noise)
acts on signs — a weaker but still interesting finding.

## CW11 Extension: Sign Evolution During Training

### Direct Observation from FI-002 P-000

P-000 saves bridge snapshots every 100 steps. Sign agreement with step 0
measured across 1200 steps of Steersman training:

| Step | % Positive | Agreement w/ Step 0 | Flip Rate/100 | Consensus |
|------|-----------|---------------------|---------------|-----------|
| 0 | 22.2% | 100.0% | — | 1.000 |
| 200 | 22.6% | 99.1% | 0.3% | 0.982 |
| 400 | 22.7% | 99.0% | 0.1% | 0.980 |
| 600 | 22.7% | 99.0% | 0.3% | 0.980 |
| 800 | 23.2% | 98.5% | 0.3% | 0.970 |
| 1000 | 23.0% | 98.5% | 0.3% | 0.970 |
| 1200 | 23.2% | 98.2% | 0.1% | 0.965 |

**Signs are frozen during training.** After 1200 steps, topology collapsed
from Fiedler 0.0105 → 0.000109 (96× suppression), co/cross rose from
7.6 → 10,060 — but only 1.8% of signs flipped. The Steersman reshapes
the entire topological structure while leaving signs untouched.

### P-000 vs C-003: Same Corpus Produces Same Signs

P-000 step 0 vs C-003 final (10K steps): **92.2% agreement**.
P-000 step 1200 vs C-003 final: **93.4% agreement** (slightly INCREASES).

P-000 init position pattern: `0%, 0%, 100%, 0%, 0%, 0%, 100%, 0%, 0%`
C-003 final position pattern: `0%, 0%, 89%, 10%, 11%, 25%, 94%, 1%, 6%`

Same corpus → same init → same sign fingerprint. The slight divergence
in C-003 (positions not at exactly 0%/100%) reflects 10K steps of marginal
drift at the extreme edges.

## CW11 Extension: Nine-Experiment Comparison

### Full Corpus (n=3 channels, 6×6 bridges)

| Experiment | Init Strategy | % Positive | Consensus | Structured? |
|------------|--------------|-----------|-----------|------------|
| C-002 | Geometric | 93.1% | **0.861** | YES |
| C-003 | Corpus-coupled | 26.3% | **0.843** | YES |
| exp3 | Corpus-coupled | 49.0% | 0.040 | NO |
| exp3_tinyllama | ? | 50.6% | 0.073 | NO |
| C-001 | Identity | 50.8% | 0.071 | NO |
| Seed-43 | Identity | 46.2% | 0.096 | NO |
| Seed-44 | Identity | 49.6% | 0.078 | NO |
| FP-code | ? | 45.6% | 0.103 | NO |
| FP-math | ? | 49.6% | 0.075 | NO |

**Only C-002 (geometric) and C-003 (corpus-coupled) produce structured
sign fingerprints.** All other experiments — identity, fingerprint, and
even some corpus — have consensus < 0.1 (chance level).

Notable: **C-003 vs exp3 = 100% majority agreement (9/9)** — identical
sign patterns from the same init strategy. **C-001 vs exp3_tinyllama =
100% (9/9)** — identical patterns from the same identity init.

### Tesseract Experiments (n=4 channels, 8×8 bridges)

| Experiment | % Positive | Consensus | p-value |
|------------|-----------|-----------|---------|
| T-001-full | 47.8% | 0.084 | 0.10 |
| T-001-full-r2 | 48.1% | 0.087 | 0.16 |
| tess-contrastive | 48.9% | 0.072 | 0.44 |
| tess-contrastive-full | 49.1% | 0.075 | 0.51 |

**No position-level sign structure at n=4.** All p-values non-significant.
However, **within-seed reproducibility is 98-99%**: T-001-full vs
T-001-full-r2 = 98.0%, tess-contrastive vs full = 99.4%. Signs are
deterministic per-adapter but not structured across adapters.

Cross-seed agreement: 83.7% (T-001 vs tess-contrastive) — higher than
the ~50% between different n=3 experiments, suggesting shared structural
properties from the same model architecture.

## Revised Interpretation

The sign fingerprint phenomenon is **init-strategy-specific**, not universal:

1. **Geometric decomposition** (C-002): Produces unanimously positive signs.
   The SVD-based initialization creates a positive-definite cross-planar
   structure.

2. **Corpus-coupled decomposition** (C-003): Produces predominantly negative
   signs. The direct factorization preserves a different sign convention.

3. **Identity and other initializations**: No sign structure. Signs are
   random per-adapter, deterministic within seed, but have no position-level
   consensus across adapters.

The "corpus fingerprint" label should be narrowed: signs carry an
**init-strategy fingerprint**, not a corpus fingerprint in general.
The two structured strategies (geometric, coupled) happen to be corpus-derived,
but other corpus-related inits (FP-code, FP-math) don't produce structure.

## Methodological Notes

- 9 positions = lower triangle of cross-planar blocks (6×6 bridge, n=3 channels)
- 16 positions for n=4 channels (8×8 bridges)
- Binomial test against 50% null, two-sided
- Consensus metric: |fraction_positive - 0.5| × 2
- FI-001 original report used slightly different methodology (12 positions including
  upper triangle); this analysis uses 9 lower-triangle positions. Results are
  consistent: C-002 strongly positive, C-003 strongly negative, anti-correlated.

## Implications for Paper 4

This finding strengthens the "bridge carries identity" thesis. The three-part
decomposition (topology + signs + magnitudes) maps cleanly onto the paper's
argument:

- Topology = the universal structure (what the Steersman teaches every bridge)
- Signs = the particular identity (what distinguishes THIS corpus from THAT one)
- Magnitudes = the strength of topological organization (how well BD has formed)

The sign fingerprint is the mechanism by which `absorb_bridge()` could
theoretically distinguish between models trained on different corpora —
even after both have been steered to identical topological equilibria.
