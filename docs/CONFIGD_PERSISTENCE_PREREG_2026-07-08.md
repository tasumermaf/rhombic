# Config-D Persistence — Pre-Registration Note (BM-000c)

> **Date:** 2026-07-08
> **Status:** Pre-registered null calibration; no trained result reported.
> **INTERNAL-ONLY.** BM-003 Config D is an internal initialization-variant arm;
> its results are **not published** (`docs/BM_BATTERY_PLAN.md:81` — "Labels D
> and E are reserved: D is an internal initialization-variant arm (results not
> published)"). This note pre-registers the reading of a single number that
> does not exist yet.
> **Artifacts:** `scripts/bm_configd_persistence_null.py`,
> `results/BM-000c-configd-persistence/nulls.json` (+ `RESULTS.md`),
> `tests/test_bm_configd_persistence.py`.

## The prediction (verbatim from the memo)

From `docs/GLOBAL_WORKSPACE_MAPPING_2026-07-07.md` §2.2 (line 58):

> "BM-003 Config D (corpus-modulated edge-weight init, internal-only) gets a
> **pre-registered prediction now** — the init pattern drifts into BM-000's
> rewire-null percentile band by end of training with no performance delta,
> jointly predicted by FI-002 and the structure/content split. If it nulls,
> the workspace paper supplies the citable general principle for the internal
> tombstone; null calibration for the persistence metric is a CPU-only
> `bm000_null_model.py` extension."

FI-002 (topology is pair-specification-determined, not init-determined) and the
workspace paper's structure/content split (installed content does not survive
unless the objective references it) jointly predict that corpus-derived initial
edge weights carry **zero task reward under generic data** and are therefore
dismantled — the trained bridge ending up no more correlated with its corpus
init than a random bridge is.

## The persistence metric (pinned 2026-07-08)

**Pearson correlation `r`** between a trained Config-D bridge's **30
off-diagonal directed edge-weight entries** (row-major order over `i != j`) and
the **Config-D corpus-modulated init template's** 30 off-diagonal entries.

- **Template:** `bridge_init(mode='corpus_coupled')` =
  `rhombic.corpus.corpus_coupled_matrix(rhombic.corpus.edge_values())` — the
  L-026-corrected corpus init that places corpus-derived weights on the
  off-diagonal *edge weights* (identity diagonal). Loaded at runtime.
  **Template identity (IP-safe):** SHA-256
  `cf763f54bcd3212f119e327f23e114037722a903aaf3ef97cc2ceb1f4e5f7aaa`. The raw
  template values are proprietary Stream-B IP and appear nowhere in the code,
  tests, results JSON, or this note.
- **Diagonal excluded:** the template diagonal is a constant identity (no
  corpus signal); including it would inject a degenerate constant block.
- `r` is **scale- and shift-invariant**, so the calibrated bands do not depend
  on the trained-bridge moment values — the calibration is valid *before* any
  real Config-D trained bridge exists.

## Calibrated null bands

`seed 20260708`, `N = 10,000` per null, moments from
`results/asset1-smoke` (BM-000 D5 policy; the live `asset1-bank` is never read).
Two BM-000 null families, both scored against the fixed template:

| Null | mean | std | p2.5 | p50 | p97.5 | **p99** | p99.9 | max |
|---|---|---|---|---|---|---|---|---|
| **rewire** (`gen_rdmask`, BM-000 N-B) — **PRIMARY** | −0.0011 | 0.1838 | **−0.3543** | −0.0008 | **+0.3597** | **+0.4189** | +0.5313 | +0.6368 |
| matched-moments (`gen_gauss`, BM-000 N-A/N-D) | −0.0016 | 0.1878 | −0.3664 | −0.0020 | +0.3678 | +0.4287 | +0.5519 | +0.6601 |

The two families agree to ~0.01 in the band edges (expected: Pearson `r` is
scale-invariant, so both reduce to a ~0-centred null-correlation distribution).
The **primary is the rewire null** because the prediction names "BM-000's
rewire-null percentile band," and BM-000's rewire null is `gen_rdmask`.

## Pre-registered three-outcome reading (PRIMARY = rewire null)

Primary band **[−0.3543, +0.3597]** two-sided; falsification threshold
**p99 = +0.4189**. Read the end-of-training persistence `r` (scored by the
metric above) once, against these fixed edges:

| End-of-training `r` | Reading |
|---|---|
| **INSIDE [−0.3543, +0.3597]** | **Prediction CONFIRMED** — the init does not persist. The workspace paper's structure/content split supplies the citable general principle for the internal tombstone. |
| **ABOVE p99 = +0.4189** | **Init PERSISTED, prediction falsified** — reported as such. |
| **intermediate** (between +0.3597 and +0.4189, or below −0.3543) | **Ambiguous** — reported with the exact percentile. **No threshold re-rolling.** |

The "no performance delta" half of the prediction is read from the existing
BM-003 benchmark endpoint on the Config-D arm, unchanged; it is not part of
this null.

## Approval status

Null-model calibration, **same class as BM-000 and BM-000b**: CPU-only, no
bank contact, no GPU. BM-000/BM-000b were run pre-approval and presented to the
Director afterward; this note follows the same path. It adds a metric and a
calibrated band only — it modifies no existing config, band, or published
claim. Config-D results remain internal and unpublished regardless of outcome.

*Metric pinned and bands calibrated 2026-07-08. Scale-invariant by
construction; re-running the script reproduces the bands byte-for-byte at the
same seed.*
