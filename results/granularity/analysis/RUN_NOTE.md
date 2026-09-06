# L0 gate-recording run — 2026-09-06 (REGISTERED; the L0 gate is recorded)

```
=== VERIFIED STATE ===
command               = scripts/granularity_analysis.py --level L0 --representation all   (no --allow-partial)
environment           = CUDA_VISIBLE_DEVICES="" (CPU only), HF_HUB_OFFLINE=1 (offline cache), falco conda env
git_commit            = 174ae95 (scripts/granularity_analysis.py byte-identical to d46c082 — the Director's condition, Item 1, 2026-09-06)
authorization         = docs/DIRECTOR_REVIEW_GRANULARITY_2026-09-06.md, Item 1 (AUTHORIZED on the exploratory dry run of 2026-09-05)
launched_utc          = 2026-09-06T03:36:55Z
gate_fired_utc        = 2026-09-06T04:48:57Z
wall_clock            = 72 min 02 s  (raw + canonical + vocab_signature x2, 1,000 permutations each, + D6 at 1,000/class; ran concurrently with the A7b CPU run)
exploratory_only      = false
ledger_entry          = results/granularity/TIER_GATES.json: tier L0 · level L0 · tiers_already_unblinded [] · k 6 · n_runs 240 · git_commit 174ae959f2ac42d9ea2230e3ddc0f15e57ce6632
identity_with_dry_run = every value in L0_REPORT.md equals the 2026-09-05 exploratory dry run (results/granularity/analysis-exploratory/L0-dryrun-2026-09-05/) — same seed streams, same code path; the differences are exploratory_only and the ledger entry
=== END VERIFIED STATE ===
```

Results: `L0_REPORT.md`, `L0_results.json` (this directory). The run log is untracked
(`*.log`): `results/granularity/logs/L0_gate_run_20260906T033655Z.log`.
