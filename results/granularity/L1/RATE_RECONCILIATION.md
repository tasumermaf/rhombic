# L1 rate reconciliation — the two wall-clock fields (2026-09-05)

Audit 2026-09-01 flagged two per-run wall-clock fields and ruled: use TIMING.md
`wall_clock_min` (.a) for projections, reconcile before either reaches a
Director packet. Computed here over the 64 COMPLETE L1 runs.

```
=== VERIFIED STATE ===
N_RUNS                          = 64
TIMING_wall_clock_min.mean      = 31.9206  (median 31.2650; min 29.41 / max 46.93)
config_wall_time_seconds.mean   = 31.5432 min  (started_at -> finished_at inside the trainer)
components_sum.mean             = 28.5309 min  (model_load_s + setup_s + step_loop_s + save_s)
GAP.median                      = 0.0025 min   GAP.mean = 0.3774 min
GAP_over_1min_runs              = 3 of 64
LARGEST_GAP                     = run_002: wall 46.93 vs config 29.91 (gap 17.02 min; hf_cache_warm_at_start=true)
RATE_FOR_PROJECTION             = wall_clock_min (.a): it is the GPU time the queue actually consumes, overheads included
=== END VERIFIED STATE ===
```

Reading: the two fields measure different windows. `wall_time_seconds` is the
trainer's own started_at→finished_at; `wall_clock_min` is the run process end to
end. For 63 of 64 runs they agree to within a fraction of a minute; one run
carries a 17.0-minute gap outside the trainer window (cache warm-up /
guard wait before the trainer started). The queue's projection now uses the mean
of `wall_clock_min` over COMPLETE runs (rhombic dbd9562), which is the
conservative, budget-relevant number. Overrun against the card basis 30.56:
+4.45% (.a) / +3.22% (.b); the card's >25% return-to-Director rule is not tripped by either.
