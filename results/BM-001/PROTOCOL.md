# BM-001: TeLoRA vs Standard LoRA Benchmark Protocol

> **Purpose:** First head-to-head comparison on standard language benchmarks.
> **Date:** April 5, 2026
> **Status:** COMPLETE — Verdict: PROCEED

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
| Batch size | 4 (default) |
| Gradient accumulation | 4 (default) |
| Learning rate | 2e-4 (default) |

## Config A: Standard LoRA

```bash
python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --bridge-mode identity \
  --no-bridge-training \
  --max-steps 10000 \
  --output results/BM-001-standard-lora \
  --save-merged \
  --seed 42
```

- `--no-bridge-training` → `bridge_trainable=False`, contrastive=0, spectral=0
- Bridge remains identity matrix throughout training → exact standard LoRA

## Config B: TeLoRA with Steersman

```bash
python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --max-steps 10000 \
  --output results/BM-001-telora \
  --save-merged \
  --feedback-interval 100 \
  --seed 42
```

- Default: `bridge_trainable=True`, Steersman active
- Contrastive topology: RD (rhombic dodecahedron, n=6)
- Feedback interval: 100 steps

## What makes this fair

1. Same codebase (`train_cybernetic.py`)
2. Same model, rank, dataset, steps, seed
3. Same target modules
4. Only difference: bridge trainability and Steersman feedback
5. Both produce merged models via identical absorption path

## Evaluation

Three models benchmarked via `scripts/eval_language_benchmarks.py` (EleutherAI lm-eval v0.4.0+):

1. **Base:** Qwen/Qwen2.5-7B-Instruct (unmodified)
2. **Standard LoRA:** `results/BM-001-standard-lora/merged_model/`
3. **TeLoRA:** `results/BM-001-telora/merged_model/`

Benchmarks: MMLU (5-shot), ARC-Challenge, HellaSwag, WinoGrande.

## Decision Criteria

| Result | Action |
|--------|--------|
| TeLoRA ≥ Std LoRA (mean) | Proceed — structure adds value |
| TeLoRA within -1% | Proceed with caveats |
| TeLoRA -1% to -3% | Investigate Steersman tuning |
| TeLoRA > -5% worse | Fundamental problem — stop |
