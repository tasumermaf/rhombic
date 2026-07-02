# lm-eval Results — TinyLlama 1.1B

## Baseline: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Model: TinyLlama 1.1B Chat v1.0 (fp16)
Eval: lm-eval v0.4.11, 0-shot, batch_size=8
Device: RTX 6000 Ada (CUDA 12.4)
Time: 601s (~10 min)

| Task | Acc | Acc_norm | N |
|------|-----|----------|---|
| hellaswag | 0.4653 | 0.6045 | 10,042 |
| arc_easy | 0.6183 | 0.5476 | 2,376 |
| piqa | 0.7432 | 0.7459 | 1,838 |
| winogrande | 0.6030 | — | 1,267 |
| openbookqa | 0.2520 | 0.3540 | 500 |
| boolq | 0.6168 | — | 3,270 |

Raw JSON: `tinyllama-baseline/TinyLlama__TinyLlama-1.1B-Chat-v1.0/results_2026-03-16T14-43-44.305832.json`

## Adapted: T-001r1 (n=8 tesseract, 2700 steps, co/cross 5,395:1)

Model: TinyLlama 1.1B with TeLoRA bridge (n=8, rank=24) merged into base weights.
Adapter: T-001r1 (partial 2700-step run, 4+4 block-diagonal achieved).

| Task | Acc | Acc_norm | N |
|------|-----|----------|---|
| hellaswag | 0.4590 | 0.6009 | 10,042 |
| arc_easy | 0.6048 | 0.5156 | 2,376 |
| piqa | 0.7394 | 0.7345 | 1,838 |
| winogrande | 0.6259 | — | 1,267 |
| openbookqa | 0.2440 | 0.3680 | 500 |
| boolq | 0.5902 | — | 3,270 |

Raw JSON: `t001-adapted/results__T-001-full__merged_model/results_2026-03-16T16-53-24.533556.json`

## Comparison (using acc_norm where available, else acc)

| Task | Baseline | Adapted | Delta | Delta% |
|------|----------|---------|-------|--------|
| hellaswag | 0.6045 | 0.6009 | -0.0036 | -0.6% |
| arc_easy | 0.5476 | 0.5156 | -0.0320 | -5.8% |
| piqa | 0.7459 | 0.7345 | -0.0114 | -1.5% |
| winogrande | 0.6030 | 0.6259 | +0.0229 | +3.8% |
| openbookqa | 0.3540 | 0.3680 | +0.0140 | +4.0% |
| boolq | 0.6168 | 0.5902 | -0.0266 | -4.3% |
| **Mean** | | | | **-0.75%** |

## Interpretation

The adapted model shows **mixed-direction, modest deltas** across benchmarks.
Mean delta is -0.75% — topology programming does not catastrophically degrade
task performance. The direction is mixed (3 tasks slightly down, 2 slightly up,
1 negligible), consistent with the bridge matrix's geometric structure being
orthogonal to task-relevant weight updates.

## Caveats

- T-001r1 is a **partial run** (2,700/10,000 steps). The full T-001r2 (41,564:1)
  has bridge data but no adapter_state.pt (merge_and_save device bug crashed).
- T-001r1 uses n=8 (tesseract), not n=6 (RD). The primary Paper 3 experiments
  are n=6, but no n=6 adapter_state.pt exists.
- Val_loss comparison during training (0.4016 adapted vs ~0.40 baseline, 0.17%
  delta) remains the primary evidence for task performance neutrality.
- lm-eval provides supplementary benchmark-level confirmation.
- A proper comparison would use the same training steps and data split.
  The baseline is unmodified; the adapted model has been fine-tuned on
  alpaca-cleaned AND topology-programmed. Some delta is expected from
  fine-tuning alone, independent of the topology programming.
