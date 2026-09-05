# L0 exploratory dry run — 2026-09-05 (NOT the registered run; no gate recorded)

```
=== VERIFIED STATE ===
command             = scripts/granularity_analysis.py --level L0 --allow-partial --representation all --out-dir <this dir>
environment         = CUDA_VISIBLE_DEVICES="" (CPU only), HF_HUB_OFFLINE=1 (offline cache), falco conda env
git_commit          = d46c082
started_utc         = 2026-09-05T21:23:34Z
finished_utc        = 2026-09-05T22:33:02Z
wall_clock          = 69 min 28 s  (raw + canonical + vocab_signature x2, 1,000 permutations each, + D6 at 1,000/class)
first_attempt       = 23 min 48 s to the end of the raw + canonical nulls, then the D6 tokenizer fault fixed in d46c082
exploratory_only    = true  (--allow-partial => record_gate skipped; TIER_GATES.json absent after the run)
=== END VERIFIED STATE ===
```

Results: `L0_REPORT.md`, `L0_results.json` (same directory). The run log is untracked (`*.log`).
