# S2 measured rates — running aggregate

Rebuilt by `python scripts/s2_timing_pilot.py --rates` from the per-run TIMING.md files. Measured only; no estimates. The Section 3 cost-table restatement required before the card locks is written separately, from these numbers.

| family | run | task | status | wall_clock_min | peak_vram_gb | mean_step_s | tokens_trained | cache_warm |
|---|---|---|---|---|---|---|---|---|
| gemma2-2b | 1 | alpaca | COMPLETE | 72.08 | 14.07 | 1.9358 | 16384000 | true |

## Per-family mean (completed runs only)

gemma2-2b_mean_min_per_run   = 72.08
gemma2-2b_n_runs_measured    = 1
qwen2.5-3b_mean_min_per_run  = —
qwen2.5-7b_mean_min_per_run  = —
llama3.1-8b_mean_min_per_run = —
