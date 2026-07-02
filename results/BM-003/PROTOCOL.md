# BM-003: RD Graph Convolution vs Standard LoRA Benchmark Protocol

> **Purpose:** Test whether structurally-imposed RD topology improves or
> maintains benchmark performance compared to standard LoRA, without any
> auxiliary losses (Steersman removed).
> **Date:** April 7, 2026
> **Status:** READY — awaiting deployment

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
