# D3 Pair Design — Dated Pre-Declaration

**Date:** 2026-07-20 (~16:50Z)
**Declared by:** Timothy Paul Bielec (PI), via Meridian
**Status at declaration:** ZERO D3 labels exist. D2 Stage B qwen evals in
progress; no merge has been emitted, no merged adapter evaluated. This
declaration precedes Step 5 (`--make-pairs --emit-merges`) entirely.

## The declared design (per pipeline §4 D3a — mechanism previously approved)

```
d3.n_pairs        = 120 per family (vertex-disjoint maximum over 240 runs/family)
d3.stratification = all 21 unordered task-pair cells (15 cross-task + 6 same-task);
                    same-task cells = the cross-seed reference; ~5-6 pairs/cell
d3.alpha          = 0.5 (midpoint merges)
d3.max_run_uses   = 1 (vertex-disjoint — the dyadic-dependence-safe design)
d3.label_rule     = DECIDED previously (Director): fixed 5% relative-degradation
                    per endpoint, degenerate fallback <10% to median split
d3.headline       = per-family group-aware CV (StratifiedGroupKFold + cluster
                    bootstrap), per pipeline §4 D3c; pooled OOF AUC descriptive
```

## Provenance

The sampler mechanism (uniform without replacement, seeded, vertex-disjoint)
was approved in `docs/DIRECTOR_DECISIONS_2026-07-06.md` / pipeline §4; the
card left N and stratification open with the requirement that they be
declared **before labels exist**. The discipline is temporal — this document
satisfies it. The full record, including this declaration, goes to the
Director in the single campaign deliverable for review; any objection there
is a reportable dated amendment, not a silent revision (L-006 / R10).

## DATED AMENDMENT — 2026-07-20 (~19:4xZ), before Step 5 ran

**Zero pairs, merges, or labels existed at this amendment.** Pre-execution
inspection of the frozen tool (`asset1_d3_merge.py --help`) shows the
approved sampler is **uniform** over same-family run pairs — it has no
per-cell stratification mode. Building one now would mean modifying frozen,
adversarially-reviewed analysis code after the bank exists; the declaration
is amended instead (the lesser deviation):

```
d3.stratification = NONE (uniform without replacement, seed 0, vertex-disjoint)
                    — supersedes the "stratified ~5-6/cell" line above.
d3.cell_coverage  = the realized (task_i, task_j) cell distribution is
                    REPORTED as a descriptive property of the sample.
                    Expected mix under uniformity: ~16% same-task
                    (~20/120 pairs, the cross-seed reference), ~84% cross-task.
```

All other declared values (N=120/family, alpha 0.5, max_run_uses 1, label
rule, headline) unchanged.
