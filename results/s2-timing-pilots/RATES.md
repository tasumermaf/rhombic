# S2 measured rates — running aggregate

Rebuilt by `python scripts/s2_timing_pilot.py --rates` from the per-run TIMING.md files. Measured only; no estimates. The Section 3 cost-table restatement required before the card locks is written separately, from these numbers.

| family | run | task | status | wall_clock_min | peak_vram_gb | mean_step_s | tokens_trained | cache_warm |
|---|---|---|---|---|---|---|---|---|
| gemma2-2b | 1 | alpaca | COMPLETE | 72.08 | 14.07 | 1.9358 | 16384000 | true |
| gemma2-2b | 2 | code | COMPLETE | 77.75 | 14.07 | 2.0883 | 16384000 | true |
| gemma2-2b | 3 | math | COMPLETE | 76.70 | 14.07 | 2.0658 | 16384000 | true |
| qwen2.5-3b | 1 | alpaca | COMPLETE | 82.21 | 11.05 | 2.2080 | 16384000 | true |
| qwen2.5-3b | 2 | code | COMPLETE | 82.34 | 11.05 | 2.2311 | 16384000 | true |
| qwen2.5-3b | 3 | math | COMPLETE | 84.03 | 11.05 | 2.2848 | 16384000 | true |
| qwen2.5-7b | 1 | alpaca | COMPLETE | 153.20 | 20.30 | 4.1214 | 16384000 | true |
| qwen2.5-7b | 2 | code | COMPLETE | 152.25 | 20.30 | 4.1148 | 16384000 | true |
| qwen2.5-7b | 3 | math | COMPLETE | 153.58 | 20.30 | 4.1566 | 16384000 | true |

## Per-family mean (completed runs only)

gemma2-2b_mean_min_per_run   = 75.51
gemma2-2b_n_runs_measured    = 3
qwen2.5-3b_mean_min_per_run  = 82.86
qwen2.5-3b_n_runs_measured   = 3
qwen2.5-7b_mean_min_per_run  = 153.01
qwen2.5-7b_n_runs_measured   = 3
llama3.1-8b_mean_min_per_run = —
