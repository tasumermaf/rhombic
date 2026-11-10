# L1 gate-recording run — 2026-11-10 (REGISTERED; the L1 gate is recorded)

```
=== VERIFIED STATE ===
command               = scripts/granularity_analysis.py --level L1 --representation all   (no --allow-partial)
environment           = CUDA_VISIBLE_DEVICES="" (CPU only), HF_HUB_OFFLINE=1 (offline cache; the log shows 12 dataset loads from the local cache, one per D6 class), falco conda env (Python 3.10.19)
git_commit            = 4279ab0 (ledger entry and L1_results.json: 4279ab00f30edc3b65590c238addb8c7ec1162e8; branch main, clean for scripts/ and the ledger at launch)
authorization         = the Director's rulings on handover 2, dated 2026-09-29, ruling (s): the word GIVEN on the exploratory dry run of 2026-09-24, on two mechanical conditions (the rulings file is in the private record, not in this repository)
condition_1_script    = scripts/granularity_analysis.py is git blob 166b4480d5cd4638f0c8073cc2000c5843f03a91 at the gate run's HEAD 4279ab0, at c4a93ae (the commit that records the exploratory pass) and at 2aba358 (the tree the exploratory pass ran from); git diff c4a93ae HEAD on the script: 0 lines
script_sha256_lf      = cfbecead8fcadd08da06e14de085ac8943ed19fab7a066d0f61cf90680bda99c (the committed LF blob content; equal at gate-run time and at c4a93ae)
script_sha256_disk    = 61df751b2a29d66ea3339dc7e6fd1827f164188ad1adaa8ac1bd8c4989b5bf78 (the working-tree file as the interpreter read it: CRLF under core.autocrlf=true; bytes on disk / LF = 48815 / 47830 (CR count 985); re-hashed equal after the run)
condition_1_placement = the ruling asks for the hash "printed into the gate entry". record_gate writes no script hash, the script was not edited to add one (that edit would itself have broken condition 1), and the ledger is not hand-edited (Director's ruling of 2026-09-11, option (b) declined). The hash is therefore carried in this note, in the preflight record and in the commit message: the L0 pattern (RUN_NOTE.md, git_commit line). A stated departure from the letter of condition 1, returned to the Director.
condition_2_epoch     = history_epoch = "2026-09-11-rewrite" in the ledger entry
launched_utc          = 2026-11-10T13:42:25Z
gate_fired_utc        = 2026-11-10T15:10:44Z
wall_clock            = 88 min 19 s  (raw + canonical + vocab_signature x2, 1,000 permutations each, + D6 at 1,000/class)
exploratory_only      = false
ledger_entry          = results/granularity/TIER_GATES.json: tier L1 · level L1 · tiers_already_unblinded ["L0"] · k 12 · n_runs 240 · git_commit 4279ab00f30edc3b65590c238addb8c7ec1162e8 · history_epoch 2026-09-11-rewrite
ledger_sha256         = before 919f811cfe4cc56f1cb34d7b223c75a1514db0e4d4da261429ef8c7d1dc5b5a6 (the pre-run working file: the CRLF checkout of git blob 0a1cdf83, 378 B, 1 entry, L0; that blob's own LF content is 365 B, sha256 fed343168288e1a2836b10bad75dd7b1db6b7c056b6b3ed93fa103a6aac3c333) · after 274bf78dd05635cfd30ff1083adae33e69045c14fba1e43b9dc4354270171f9a (the file as record_gate wrote it: LF, 639 B, 2 entries; the committed blob has this content); the diff is 12 added lines and 0 removed, so the L0 entry is unchanged
identity_with_dry_run = every value in L1_REPORT.md equals the exploratory dry run of 2026-09-24 (results/granularity/analysis-exploratory/L1-dryrun-2026-09-24/): the reports differ in 3 lines of 136 (:10, :131, :132; exploratory_only, git_commit, generated_at_utc), and L1_results.json in 4 of 1,378 leaves (allow_partial, exploratory_only, generated_at, git_commit)
outputs               = L1_REPORT.md 5,043 B sha256 131b3a25c1bbcb94ade106bdd31b0b75706d691c1bf1b307a6e6d2c4ac4e8f21 · L1_results.json 31,101 B sha256 4bb5803bdfe4ac0ddb5a18fc53e4c6410f0823d56ff64a014997cb004c3b07c4
campaign              = 240 of 240 runs COMPLETE, 0 failed, 0 pending (results/granularity/QUEUE_STATE.md, updated_at_utc 2026-09-18T20:21:33.934622+00:00)
other_preflights      = two more PASS preflight records sit in results/granularity/logs/, each a preflight-only pass of the launcher with no run log beside it: L1_gate_preflight_20260930T201801Z.txt (HEAD 670be69) and L1_gate_preflight_20261110T134157Z.txt (HEAD 4279ab0); no run was launched from either and no gate fired; the ledger's last commit before this run is 2bddf53
scope_limit           = llama3.2-1b only; it belongs in every claim taken from this run (L1_REPORT.md states it in its second line)
verified              = a fresh read-only verifier graded the ledger, the report, the results file, this note and the two tracker rows before the commit: PASS WITH CORRECTIONS; 182 values re-derived (104 report values from the JSON, 78 primitives from the confusion matrices), 0 mismatches. Its six corrections (the line count, the two forms of the ledger digest, these three lines, the tracker row's dropped cell and scope limit, the hashes in the commit message) were applied by the hub and are not re-graded
=== END VERIFIED STATE ===
```

Results: `L1_REPORT.md`, `L1_results.json` (this directory). The run log and the
preflight record are untracked (`results/granularity/logs/` is gitignored, `.gitignore:204`):
`results/granularity/logs/L1_gate_20261110T134225Z.log`, `results/granularity/logs/L1_gate_preflight_20261110T134225Z.txt`.
The L0 note is `RUN_NOTE.md` in this directory; it is not edited.
