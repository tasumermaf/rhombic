# BM-000 — Null-Model Calibration: Pre-Registered Protocol

> **Date:** July 4, 2026 (pre-registered BEFORE any null run)
> **Plan reference:** `docs/BM_BATTERY_PLAN.md` § "BM-000 — Null-model calibration"
> **Seed:** 20260704 (single `numpy.random.default_rng` stream for all ensembles)
> **Compute:** CPU only (GPU reserved for the active training campaign; no model
> is loaded; torch is imported only as a side effect of reusing
> `scripts/train_exp2_scale.py` definitions, with `CUDA_VISIBLE_DEVICES=""`)
> **Script:** `scripts/bm000_null_model.py` — generates `RESULTS.md` and
> `nulls.json` in this directory. Every number in RESULTS.md is emitted by the
> script; nothing is hand-transcribed.

## Purpose

No topology metric in this program has a chance baseline (BM_BATTERY_PLAN
§BM-000). This protocol fixes, in advance: the metric definitions (cited to the
exact code that produced the program's reported values), the null ensembles,
the sample sizes, the seed, and the headline values that will be placed against
the nulls. After this file is committed, the generator runs once and the
outputs are published as-is.

---

## 1. Metric definitions (REUSED from existing code, not reimplemented)

Where a metric has more than one definition in the codebase, **all variants are
computed and labeled**. The "canonical" label goes to the code path that
produced the program's reported headline numbers.

### M1 — Co/cross ratio (canonical: RD face-pair, upper triangle)

- **Code:** `scripts/train_exp2_scale.py:155-181` `coplanar_crossplanar_ratio(B)`,
  imported directly. Pair sets from `_coplanar_crossplanar_indices(n)`
  (`scripts/train_exp2_scale.py:99-133`).
- **Definition:** ratio = mean(|B[i,j]|, (i,j) ∈ co-pairs) / mean(|B[i,j]|,
  (i,j) ∈ cross-pairs), **upper triangle only** (i < j).
- **Pairs (verified identical across the codebase before this run):**
  - n=6 (RD face-pair definition, coupling ≥ 4 in
    `rhombic/nn/topology.py:20-68` `direction_pair_coupling`):
    co = {(0,1),(2,3),(4,5)} — 3 co, 12 cross. Identical to the
    "consecutive 2×2" pairs of `scripts/ar001_asymmetry_analysis_v2.py:44`
    and to `CO_PLANAR_PAIRS` in `scripts/analyze_bridge_structure.py:37`.
  - n=4 (octahedron): co = {(0,1),(2,3)} — `scripts/train_contrastive_bridge.py:107-115`,
    identical in `train_exp2_scale.py:118-123`.
  - n=8 (tesseract co-axial): co = {(0,1),(2,3),(4,5),(6,7)} —
    `scripts/train_contrastive_bridge.py:116-124`, identical in
    `train_exp2_scale.py:124-133`.
- **This is the definition behind the training telemetry** (`train_cybernetic.py:306-314`
  aggregates exactly this function) and therefore behind H-ch6's 70,404:1.
- **Variant M1b — full-matrix co/cross** (`scripts/analyze_bridge_structure.py:155-187`):
  membership test over **both** triangles (i ≠ j) against `CO_PLANAR_SET`
  (6 directed co entries, 24 directed cross entries). The original code pools
  across a bridge set; we compute it per-bridge to get a distribution.

### M2 — Bridge Fiedler value (canonical: telemetry definition)

- **Code:** `rhombic/spectral.py:63-71` `fiedler_value(n, edges, weights)` with
  `edges = [(i,j) for i<j]`, `weights = |B[i,j]|` — exactly as invoked in
  `rhombic/nn/rhombi_lora.py:284-295` (`bridge_fiedler`) and
  `scripts/train_cybernetic.py:286-290` (Steersman sensor). λ₂ of the weighted
  graph Laplacian built from **upper-triangle** |B| as adjacency.
- **Variant M2b — analyze_task_bridges definition**
  (`scripts/analyze_task_bridges.py:397-405`): W = |B| with zero diagonal,
  L = diag(rowsum(W)) − W, λ₂ via `np.linalg.eigvalsh(L)`. For asymmetric B
  this L is **not symmetric**; `eigvalsh` reads the lower triangle
  (UPLO='L'), so the off-diagonal information comes from the lower triangle of
  |B| while the degree diagonal uses full row sums. This quirk is reproduced
  verbatim (copied, cited), not corrected — the null must calibrate the code
  as it ran.
- Note (`results/channel-ablation/FIEDLER_METRIC_NOTE.md`): the "correlation
  Fiedler" (~0.10 cross-layer consistency metric) is a different metric and is
  NOT calibrated here; only the bridge Fiedler is.

### M3 — Block-diagonal score

- **Canonical M3a:** `scripts/detect_blocks.py:16-132` `detect_blocks(B,
  threshold_ratio=10.0)`, imported directly. Outputs used: `is_block_diagonal`
  (bool), `ratio` (within/between coupling; 1.0 when not BD), `max_gap_ratio`,
  `n_blocks`. Null deliverables: chance BD-detection rate, and distributions of
  `max_gap_ratio` and of `ratio` conditional on detection.
- **Variant M3b — top-3 criterion** (`scripts/analyze_bridge_structure.py:168-172`):
  bridge counts as BD iff the top-3 off-diagonal magnitudes (of 30 directed
  entries) all lie in `CO_PLANAR_SET`. Analytic chance rate for exchangeable
  entries: C(6,3)/C(30,3) = 20/4060 ≈ 0.49%; the empirical null must agree.

### M4 — Asymmetry ratio

- **Code:** `scripts/ar001_asymmetry_analysis_v2.py:133-141` (and
  `results/AR-001-ASYMMETRY-ANALYSIS.md` §Method): A = (B − Bᵀ)/2,
  asym = ‖A‖_F / max(‖B‖_F, 1e-10). That file is a top-level script (runs its
  analysis on import), so the two-line formula is copied verbatim with this
  citation rather than imported.

---

## 2. Null ensembles (all from `numpy.random.default_rng(20260704)`, in this order)

Moment matching: from ALL trained smoke bridges
(`results/asset1-smoke/{qwen2.5-1.5b,llama3.2-1b}/{alpaca,squad}/run_*/bridge_final_*.npy`,
expected 352 matrices, 6×6), measure four numbers: mean/std of the pooled
diagonal entries (μ_d, σ_d) and mean/std of the pooled off-diagonal entries
(μ_o, σ_o). These are measured quantities, recorded in RESULTS.md; the
procedure is fixed here.

| ID | Ensemble | N | Construction |
|----|----------|---|--------------|
| N-A | `gauss6` | 10,000 | 6×6; diagonal iid N(μ_d, σ_d²), off-diagonal iid N(μ_o, σ_o²) |
| N-B | `rdmask6` | 10,000 | Random {0.5,1.0} mask matched to `rd_adjacency_mask(6)` (`rhombic/nn/topology.py:71-106`): symmetric, diagonal 1.0, a uniformly random 3 of the 15 unordered off-diagonal pairs set to 1.0 (both directions), the other 12 to 0.5 — preserving the RD mask's value multiset (this is the bridge-scale analogue of the plan's degree-preserving-rewire family). Bridge = mask ⊙ G, G drawn as in N-A. |
| N-C | `identity+eps` | 10,000 per ε | B = I₆ + N(0, ε²) iid on all 36 entries, ε ∈ {0.01, 0.05, 0.1}. Near-init / frozen-reference family (plan family (c): an exactly frozen identity bridge is degenerate — co/cross 0/0, Fiedler 0 — so the small-noise envelope brackets it). |
| N-D4 | `gauss4` | 10,000 | As N-A at n=4; co/cross uses octahedron pairs {(0,1),(2,3)} |
| N-D8 | `gauss8` | 10,000 | As N-A at n=8; co/cross uses tesseract pairs {(0,1),(2,3),(4,5),(6,7)} |

All four metrics (M1, M1b, M2, M2b, M3a outputs, M3b, M4) are computed for
every bridge in every ensemble (M1/M1b at n≠6 use that n's pair definition;
M3b is n=6-only since `CO_PLANAR_SET` is defined for n=6).

## 3. Empirical comparison sets (NOT nulls — placed as points against the nulls)

- Pilot task bridges: `results/fingerprints/code/bridge_final_*.npy` (112),
  `results/fingerprints/math/bridge_final_*.npy` (112),
  `results/exp2/rhombi_fcc_r24/bridge_final_*.npy` (112, the pilot "alpaca" set).
- Smoke-run bridges: the same 352 files used for moment matching (reported per
  family×task run).

Per set: per-bridge metric distributions (mean/std/min/max) and the percentile
of the set mean against the matching null (N-A for n=6).

## 4. Headline values to be placed against the nulls

| Value | Source (read programmatically where the file exists locally) |
|-------|--------|
| H-ch6 co/cross = 70,404.01 (final, step 10,000) | `results/channel-ablation/H-ch6/metrics_hermes.csv` last row, col 4 (verified in `paper/audit/round-2/full-audit-report.md` F-009) |
| H-ch6 bridge Fiedler = 8.93e-05 (final) | same CSV, col 3 |
| Spectral-attractor Fiedler ≈ 0.09 (band 0.084–0.102: E-001 0.084, H-ch12 0.102) | `docs/EXPERIMENT_TRACKER.md` (H-ch8/H-ch12/E-001 rows) |
| AR-001 asymmetry means: Seed-43 0.3196, Seed-44 0.3203, exp3-Qwen 0.4963, T-001-full 0.3026, T-001r2 0.3193; non-BD H-ch3 0.0194, H-ch4 0.0142 | `results/AR-001-raw-results.json` (read programmatically) |
| AR-001 co/cross means: Seed-43 73,308, Seed-44 70,201, exp3-Qwen 33,458, T-001-full 20,944, T-001r2 41,564 | `results/AR-001-raw-results.json` |
| Pilot task bridge metrics (code/math/alpaca) | computed directly from the .npy files by this script |

Percentile rule: pct(v) = 100 · #{null < v} / N. Values above the null maximum
are reported as ">max" with the null max and 99.9th percentile alongside.

## 5. Deliverables

1. `results/BM-000/RESULTS.md` — per metric × ensemble: mean, std, percentiles
   {50, 90, 99, 99.9}, max; the headline-vs-null percentile table; and, per the
   plan's key question, **for each headline metric the null 99.9th percentile
   and a verdict: is the trained/programmed value clearly outside the null
   band?** Any metric where it is NOT is flagged explicitly.
2. `results/BM-000/nulls.json` — raw percentile grids (0.1, 1, 5, 10, 25, 50,
   75, 90, 95, 99, 99.5, 99.9, 100) per metric × ensemble, plus the matched
   moments and seed, for programmatic reuse.
3. `tests/test_bm000.py` — determinism: same seed ⇒ identical percentile
   tables from two independent generator invocations (small N, fixed moments);
   plus pair-definition consistency assertions.

## 6. Pre-registered expectations (falsifiable)

- M3b null BD rate ≈ 0.49% (analytic). Large deviation ⇒ generator bug.
- The extreme trained co/cross values (10⁴–10⁵:1) should sit far beyond any
  null's 99.9th percentile; if a null family reaches within 10× of them, the
  metric is NOT evidence of programmed structure.
- The spectral-attractor Fiedler (~0.09) is expected to be scale-confounded:
  Fiedler under these nulls scales with the off-diagonal magnitude, so its
  percentile depends on the matched σ_o. If 0.09 falls inside the central null
  band of any matched ensemble, that flag is published (this is precisely the
  calibration the plan asks for).

*Amendments to this protocol after the run are dated edits, not silent
revisions.*
