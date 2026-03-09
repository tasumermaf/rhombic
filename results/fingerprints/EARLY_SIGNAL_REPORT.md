# Phase 1A — Early Signal Report

> **Date:** 2026-03-08
> **Status:** PRELIMINARY — training still in progress
> **Comparison:** Alpaca (Exp 2, step 1000) vs Code (Phase 1A, step 500)
> **Caveat:** Mismatched training steps (1000 vs 500). True comparison at matched steps pending.

## Result: EARLY SIGNAL DETECTED

Bridges trained on different tasks (general instruction vs code) are
already measurably different at early training steps.

### Paired Comparison (112 adapters, same position, different task)

| Metric | Alpaca (step 1K) | Code (step 500) | Difference | Paired t | p-value |
|--------|-----------------|-----------------|------------|----------|---------|
| Fiedler | 0.01778 ± 0.00749 | 0.01329 ± 0.00489 | +0.00450 | 6.761 | 6.66e-10 |
| Deviation | 0.03197 | 0.02200 | +0.00997 | 8.400 | 1.65e-13 |

### Per-Module Breakdown (Fiedler)

| Module | Alpaca | Code | t | p |
|--------|--------|------|---|---|
| q_proj | 0.01674 | 0.01204 | 4.583 | 0.0001 |
| k_proj | 0.01711 | 0.01392 | 1.757 | 0.0902 |
| v_proj | 0.02012 | 0.01377 | 4.678 | 0.0001 |
| o_proj | 0.01716 | 0.01340 | 3.998 | 0.0004 |

### Direction Consistency

80.4% of adapters (90/112) show code Fiedler < alpaca Fiedler.

### Eigenspectrum Similarity

Cosine similarity: 0.99998 ± 0.00004 (range: 0.99971 to 1.00000).

**Interpretation:** Tasks change coupling MAGNITUDE, not coupling STRUCTURE.
The bridge develops the same pattern at different intensities depending on
the training data. Code (structured, formal) requires less cross-channel
mixing than general instruction following (diverse, multi-domain).

### What This Means for Phase 1A

If this early signal holds at matched training steps (both at 2K), the
task fingerprinting hypothesis is strongly supported. The bridge doesn't
just learn generic coupling — it learns task-appropriate coupling levels.

### Next: True Comparison

When code training reaches step 2000 and math training completes, run
`scripts/analyze_task_bridges.py` for the formal three-way comparison
with matched training durations, LOO SVM classification, and Mann-Whitney
between-task distance tests.
