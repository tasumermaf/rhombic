# L1 exploratory dry run — 2026-09-24 (NOT the registered run; no gate recorded)

```
=== VERIFIED STATE ===
command                = scripts/granularity_analysis.py --level L1 --allow-partial --representation all --out-dir <this dir>
environment            = CUDA_VISIBLE_DEVICES="" (CPU only), HF_HUB_OFFLINE=1 (offline cache; the log shows 12 dataset loads from the local cache, one per D6 class), falco conda env (Python 3.10.19)
git_commit             = 2aba358 (L1_results.json git_commit 2aba35825491a4c821982031777e7e9fd7369844)
script_last_changed    = 8f43a42 (scripts/granularity_analysis.py; no diff 8f43a42..2aba358; working tree clean for scripts/)
launched_utc           = 2026-09-24T16:00:10Z
finished_utc           = 2026-09-24T17:29:01Z  (L1_REPORT.md mtime; the report's generated_at_utc 2026-09-24T16:00:15Z is stamped at analysis start, scripts/granularity_analysis.py:526)
wall_clock             = 88 min 51 s  (raw + canonical + vocab_signature x2, 1,000 permutations each, + D6 at 1,000/class)
exploratory_only       = true  (--allow-partial => record_gate skipped, scripts/granularity_analysis.py:617; all 240 of 240 planned runs carried COMPLETE, so the flag's only effects were exploratory_only and the skipped gate)
ledger                 = results/granularity/TIER_GATES.json untouched: sha256 919f811cfe4cc56f1cb34d7b223c75a1514db0e4d4da261429ef8c7d1dc5b5a6, 1 entry (tier L0), no L1 entry, git blob 0a1cdf83 clean against HEAD
identity_with_gate_run = not yet run
=== END VERIFIED STATE ===
```

Results: `L1_REPORT.md`, `L1_results.json` (same directory). The run log is untracked
(`results/granularity/logs/` is gitignored, `.gitignore:204`):
`results/granularity/logs/L1_dryrun_20260924T160010Z.log`.
