# BM-003: RD Graph Convolution vs Standard LoRA Benchmark Protocol

> **Purpose:** Test whether structurally-imposed RD topology improves or
> maintains benchmark performance compared to standard LoRA, without any
> auxiliary losses (Steersman removed).
> **Date:** April 7, 2026
> **Status:** READY — awaiting deployment
> **Amended:** July 7, 2026 — hub-motif mask arms (Configs G/H) +
> dissociation eval endpoint. Dated edit per L-006, Director-approved
> 2026-07-07 (`docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md`, A4).
> See the Amendment section at the end of this document. The original
> April text above is unmodified.

## The Architectural Pivot

BM-001 and BM-002 tested a **dense bridge + Steersman** (auxiliary contrastive
and spectral losses pushing the bridge toward RD topology). The Steersman was
an optimization target, not a structural feature. BM-003 tests the corrected
architecture: **RD graph convolution bridge** where topology is structural by
construction, trained with LM loss only.

### What changed

| Component | Old (BM-001/002) | New (BM-003) |
|-----------|-----------------|--------------|
| Bridge | Dense 6×6 (36 params) | Fixed RD mask × learnable edge weights (36 params) |
| Topology | Optimization target | Structural by construction |
| Auxiliary losses | Contrastive + Spectral | None |
| Controller | Steersman (3 control laws) | None |
| Edge weights | All equal at init | Topology-weighted (co-planar=strong, cross-planar=weak) |

### Key files modified

- `rhombic/nn/topology.py` — Added `rd_adjacency_mask()`, `bridge_init('rd_graph')`
- `rhombic/nn/rhombi_lora.py` — Added `rd_graph` bridge mode with `rd_mask` buffer + `edge_weights` parameter
- `scripts/train_cybernetic.py` — Added `--no-steersman`, `--bridge-mode rd_graph`

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Dataset | yahma/alpaca-cleaned |
| Steps | 10,000 |
| Rank | 24 |
| Channels | 6 |
| Seed | 42 |
| Target modules | q_proj, k_proj, v_proj, o_proj (4 × 28 layers = 112) |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Learning rate | 2e-4 |

## Config A: Standard LoRA (Control)

```bash
python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --bridge-mode identity \
  --no-bridge-training \
  --max-steps 10000 \
  --output results/BM-003-standard-lora \
  --save-merged \
  --seed 42
```

- `--no-bridge-training` → bridge frozen at identity = exact standard LoRA
- This is the same control config as BM-001 Config A

## Config B: RD Graph Convolution (LM Loss Only)

```bash
python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --bridge-mode rd_graph \
  --no-steersman \
  --max-steps 10000 \
  --output results/BM-003-rd-graph \
  --save-merged \
  --feedback-interval 100 \
  --seed 42
```

- `--bridge-mode rd_graph` → fixed RD topology mask × learnable edge weights
- `--no-steersman` → LM loss only, no contrastive or spectral losses
- `--feedback-interval 100` → spectral diagnostics still logged (read-only)
- Edge weights train via backpropagated LM loss gradient through the bridge

## Config C: RD Graph Convolution with Corpus-Seeded Weights

```bash
python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --bridge-mode rd_graph \
  --no-steersman \
  --dataset code \
  --max-steps 10000 \
  --output results/BM-003-rd-graph-seeded \
  --save-merged \
  --seed-bridges results/BM-003-rd-graph \
  --feedback-interval 100 \
  --seed 42
```

- Same as Config B but trained on CodeAlpaca-20k with Alpaca-trained edge weights
- Tests whether pre-trained topology transfers across tasks

## What makes this fair

1. Same codebase (`train_cybernetic.py`)
2. Same model, rank, dataset (A/B), steps, seed
3. Same target modules
4. Only difference: bridge parameterization (identity vs RD graph conv)
5. Both produce merged models via identical absorption path
6. Config B has same parameter count as dense bridge (36 edge weights)

## Evaluation

Three models benchmarked via `scripts/eval_language_benchmarks.py`:

1. **Base:** Qwen/Qwen2.5-7B-Instruct (reuse BM-001 results)
2. **Standard LoRA:** `results/BM-003-standard-lora/merged_model/`
3. **RD Graph Conv:** `results/BM-003-rd-graph/merged_model/`

Benchmarks: MMLU (5-shot), ARC-Challenge, HellaSwag, WinoGrande.

## Decision Criteria

| Result | Action |
|--------|--------|
| RD Graph ≥ Std LoRA (mean) | **Vindication.** Structural topology improves or matches performance |
| RD Graph within -1% | Proceed — minor trade-off for structural guarantees |
| RD Graph -1% to -3% | Investigate initialization, edge weight scaling |
| RD Graph > -5% worse | Fundamental problem — topology may constrain too much |

## Hypothesis

The dense bridge + Steersman achieved co/cross ratios of 30-80k but no
benchmark improvement because auxiliary losses compete with LM loss for
gradient bandwidth. The RD graph convolution spends 100% of gradient
bandwidth on the task while getting the topology for free. If the topology
helps the task, the edge weights will amplify it. If it doesn't, the edge
weights will attenuate it. Either way, the task decides — not an auxiliary
loss function.

---

## Amendment 2026-07-07 — Hub-Motif Mask Arms + Dissociation Eval Endpoint

> **Dated edit per L-006** (not a silent revision; the April text above is
> unmodified). Submitted 2026-07-07; **Director-approved 2026-07-07**
> (`docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md`, A4 — the Director
> re-derived the BM-000b null-calibration numbers from `nulls.json` before
> ruling). Approval governs pre-registered status only; no GPU work before
> bank completion (~Jul 20). Also referenced from `docs/BM_BATTERY_PLAN.md`.

### Motivation

The workspace paper shows unconstrained computation organizes around hubs
(~100× connectivity); the anticipated referee attack on the BD/RD results is
"any hub-ish topology would do." **BM-000b closes this at the null level**
(`results/BM-000b-hub-motifs/nulls.json`, seed 20260707, N=10,000 × 15
ensembles): no untrained star / expander / degree-matched-regular /
shuffled-RD mask ensemble reproduces any trained structure-bearing headline
(H-ch6 co/cross 7.04e4 vs motif null maxima ≤ 9.48; H-ch6 Fiedler 8.93e-5
below every motif minimum; AR-001 n=8 headlines vs maxima ≤ 6.61). Honest
negative, reported: single untrained mask draws are not topology-classifiable
at all (0/48 under the pre-stated rule; the polytope partition shifts the
co/cross median only 2.04–2.20×; ref_rd6 biases the top-3 co-planar criterion
to 23.24% vs star 0.21% / expander 1.01% — real but nowhere near the trained
100% BD). The trained-mask question therefore CANNOT be answered by nulls
alone; the arms below answer it, read against BM-000b's calibrated bands.
All invented defaults are recorded in
`results/BM-000b-hub-motifs/nulls.json["invented_defaults"]`.

### Config G — Shuffled-Adjacency (trained arm)

All parameters identical to Config B except the fixed mask. The
SVFT-Random-style arm the July-3 literature watch mandated, now
theory-motivated. Strong set pinned to **{(0,3), (1,2), (4,5)}**, drawn
deterministically by `random_perfect_matching(default_rng(20260707), 6)`.

**Disclosure (required in the paper, not just this protocol — Director
condition, 2026-07-07):** this draw shares one strong pair, (4,5), with the
RD co-planar set. The seed is the date, not selected; the overlap is
disclosed rather than re-rolled (re-rolling would be post-hoc selection).

### Config H — Expander (trained arm)

All parameters identical to Config B except the fixed mask. Strong set
pinned to **K3,3 with parts {0,1,2} / {3,4,5}** — the unique
maximal-Fiedler 3-regular graph on 6 vertices (the deterministic limit of
BM-000b's expander generator).

Both G and H use the BM-000 mask convention (diag 1.0, strong 1.0, weak
0.5) and Config B's edge-weight init (I + 0.1·(mask − I)). Est. ~2–3
GPU-days per arm, post-bank.

### Pre-registered reading — topology-specificity band (±0.5%)

Benchmark mean (the existing primary endpoint) compared pairwise B-vs-G and
B-vs-H:

| Outcome | Reading |
|---------|---------|
| \|B−G\| and \|B−H\| < 0.5% | Any benchmark effect is **not topology-specific**; the structural-prior claim narrows to the structure-metric level |
| B exceeds both G and H by ≥ 0.5% | **Topology-specificity supported** |
| G or H exceeds B by ≥ 0.5% | **The hub attack is CONFIRMED at the trained level** and will be reported as such |

All three outcomes are pre-stated and publishable; no threshold re-rolling.

### Dissociation eval endpoint (eval-only; hours, post-bank)

`eval_language_benchmarks.py` gains GSM8K (5-shot, direct, exact match) and
SST-2 (zero-shot), reported alongside the unchanged 4-benchmark primary
endpoint.

**TASK-CLASS ASSIGNMENT — FROZEN. Freeze timestamp: 2026-07-07** (this
dated edit; the same-day git commit of this file provides the
tamper-evident record — Director condition A4, 2026-07-07: the freeze
precedes any dissociation data on the record):

- **Workspace-dependent class = {GSM8K-direct}**
- **Automatic class = {SST-2, MMLU, ARC-Challenge, HellaSwag, WinoGrande}**

No post-hoc task reclassification under any outcome.

**Bridge-ablation sub-endpoint:** each trained arm (A, B, F, G, H) evaluated
against a bridge-ablated variant (trained bridge → identity at merge; A/B
factors untouched). Pre-registered reading: IF the rd_graph bridge carries
workspace-like routing, ablation selectively degrades the
workspace-dependent class — pinned criterion: **GSM8K-direct delta more
negative than the automatic-class mean delta by > 2pp**, automatic class
within its own eval noise. **Honest null (pre-stated):** uniform, absent,
or automatic-concentrated degradation disconfirms H-dissociation for
rd_graph and is reported as such — no post-hoc task reclassification, no
threshold re-rolling.

*This amendment adds arms and endpoints only; it modifies no existing
config, band, or claim.*
