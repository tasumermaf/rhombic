# S2 timing pilots — run queue

Registered card: `docs/REGISTRATION_H2_SCALE_2026-07-30.md` (S2, promoted to a
precondition). Three timing-only runs per family, excluded from every bank,
computing no H2 statistic. **One run at a time — sequential GPU discipline.**
Never launch a run while another is training; the single RTX 6000 Ada is the
whole campaign's instrument, and concurrent runs would corrupt the very
quantity being measured.

## State

**All nine accessible timing runs are measured, and the §3 cost table has been
restated against them** (`docs/S2_COST_RESTATEMENT_2026-08-04.md`), so both
lock conditions the registration names are satisfied for the three measured
families — satisfying them is not itself the lock, which is a separate act,
closed for H2-D on 2026-09-11 by the lock declaration filed in this same series
(`docs/LOCK_DECLARATION_H2D_2026-09-11.md`).
**The queue was automated as of `96c52bf` — this section's original
claim that it "does not self-advance" is superseded.** The hub hand-launched
`gemma2-2b` runs 1 and 2 with the `Start-Process` pattern below, then started
`scripts/s2_queue_runner.py`, which waited out run 2 and then chained the
remaining seven runs one at a time. The manual launch commands further down
remain valid as the fallback if the runner is stopped.

One family is still outstanding: `llama3.1-8b`. Its licence gate was accepted
on 2026-08-04 and it now probes OK, but its three runs have never been
launched.

=== VERIFIED STATE ===
updated_at               = 2026-09-11 (every value below re-derived from artifacts this date; none carried forward)
gemma2-2b_run_1_status   = COMPLETE
gemma2-2b_run_1_measured = 72.08 min/run
gemma2-2b_run_1_peak_vram = 14.07 GB
launched_at_utc          = 2026-07-30T01:30:50Z DERIVED
# CHANGED from 01:33Z, which reproduces from no artifact: no artifact logs a launch time, so this is DERIVED as finished_at_utc - wall_clock_incl_process_min (72.20) [gemma2-2b/run_1/TIMING.md], which by construction is the pilot process's own start (symbol `PROCESS_T0`, set at module import in scripts/s2_timing_pilot.py, and carried to the record as `wall_proc_s`) and so runs a second or two behind the Start-Process call itself; on a strict reading of "re-derivable from an artifact" the alternative line is UNMEASURED.
finished_at_utc          = 2026-07-30T02:43Z (gemma2-2b run_1; exact 02:43:02.307819Z)
# Value unchanged, scope annotation added: it is gemma2-2b/run_1/TIMING.md `finished_at_utc`, i.e. run 1 only, not the queue.
launch_mode              = runs 1-2 PowerShell Start-Process, hidden, detached; runs 3-9 launched by s2_queue_runner.py via subprocess.run (synchronous), the runner itself Start-Process detached
# CHANGED from run_1-only: seven of the nine runs were launched by the runner, not by hand (`[queue] launching ...` for gemma2-2b run 3 and all six qwen runs in logs/queue_runner.log; scripts/s2_queue_runner.py:62-66; detached-launch pattern in its docstring, lines 10-14).
runs_measured            = 9 of 12 registered (gemma2-2b 3/3, qwen2.5-3b 3/3, qwen2.5-7b 3/3, llama3.1-8b 0/3); all nine status = COMPLETE, zero FAILED; last finish 2026-07-30T17:10:45Z
# CHANGED from "9 of 9 accessible ... 10:10": the denominator is 12 (FAMILIES x RUNS, scripts/s2_queue_runner.py:26-27) now that llama3.1-8b probes OK; each of the nine TIMING.md files reads `status = COMPLETE`; 17:10:45Z is qwen2.5-7b/run_3 `finished_at_utc`, of which the old "10:10" was the same instant in local time sitting inside a block of `_utc` keys.
queue_mode               = automated (scripts/s2_queue_runner.py, commit 96c52bf; gpu_guard retrofit 88b90e0e, 2026-08-11)
# CHANGED from 1cccb90, a dead pre-rewrite SHA (`git cat-file -t` returns "Not a valid object name"): docs/history-rewrite/cited-commit-map.txt:9 maps it to 96c52bf, which resolves and is an ancestor of HEAD; the retrofit commit is from `git log -- scripts/s2_queue_runner.py`.
queue_runner_log         = results/s2-timing-pilots/logs/queue_runner.log [GITIGNORED - .gitignore:154; workstation-local, not auditable from the repo]
# CHANGED only by the marker: the path is right, but `git ls-files results/s2-timing-pilots/logs/` returns nothing, so this witness cannot be checked by anyone cloning the repo.
llama3.1-8b_access       = OK (probed 2026-08-04, account timotheospaul, asset1_bank.probe_family_access)
# CHANGED from BLOCKED / "re-probed 2026-07-30 ~03:0xZ": GATING.md carries the later probe and is the source of record, and the flip is commit f62767f6 (2026-08-04); the old timestamp was independently impossible, since the commit that wrote it (e79d33de) is dated 2026-07-30T02:49:48Z. Not re-probed today - a live probe would be a new measurement, not a re-derivation.
=== END VERIFIED STATE ===

Measured against the 10-step dry-run projections, per family — the mean of that
family's three runs against its own `dryrun/DRYRUN.md`
`PROJECTED_min_per_full_run`:

| family | projected | measured mean | delta |
|---|---|---|---|
| gemma2-2b | 73.03 | 75.51 | +3.40% |
| qwen2.5-3b | 70.64 | 82.86 | +17.30% |
| qwen2.5-7b | 156.07 | 153.01 | -1.96% |

This section originally read that run 1 landed 1.3% under its projection "so
the projections for the remaining families are worth trusting to roughly that
tolerance." Measurement falsified that: qwen2.5-3b missed by more than an order
of magnitude more. The dry runs were directionally useful; they were not a
tolerance. For run 1 specifically, step time was slightly *worse* than the dry
run predicted (1.9358 s vs 1.8860 s) and the setup/eval terms slightly better.

## Queue (launch in this order, one at a time)

Order = the draft's Section 3 table order, which is also the S9 tier order
frozen at registration (cheapest first). `proj_min` is a PROJECTION, not a
measurement — the point of these runs is to replace it. For rows 1–9 it is the
family's dry run; for the llama row it is the affine fit (see below the table).

| # | family | run | task | status | proj_min |
|---|---|---|---|---|---|
| 1 | gemma2-2b | run_1 | alpaca | **COMPLETE — 72.08 measured** | 73.0 |
| 2 | gemma2-2b | run_2 | code | **COMPLETE — 77.75 measured** | 73.0 |
| 3 | gemma2-2b | run_3 | math | **COMPLETE — 76.70 measured** | 73.0 |
| 4 | qwen2.5-3b | run_1 | alpaca | **COMPLETE — 82.21 measured** | 70.6 |
| 5 | qwen2.5-3b | run_2 | code | **COMPLETE — 82.34 measured** | 70.6 |
| 6 | qwen2.5-3b | run_3 | math | **COMPLETE — 84.03 measured** | 70.6 |
| 7 | qwen2.5-7b | run_1 | alpaca | **COMPLETE — 153.20 measured** | 156.1 |
| 8 | qwen2.5-7b | run_2 | code | **COMPLETE — 152.25 measured** | 156.1 |
| 9 | qwen2.5-7b | run_3 | math | **COMPLETE — 153.58 measured** | 156.1 |
| 10–12 | llama3.1-8b | run_1–3 | — | gate ACCEPTED 2026-08-04, not yet run | 159.2 PROJECTED |

Every status above comes from that run's `TIMING.md`; every measured minute is
its `wall_clock_min`. The llama `proj_min` is not a dry run — it is the affine
fit in `docs/S2_COST_RESTATEMENT_2026-08-04.md` §3 evaluated at 8.0B, and the
Director accepted it as shape, not evidence.

Projected total for the nine accessible runs was 899.22 min = **14.99 h GPU**
(sum of the nine dry-run projections). The measurement supersedes it:
**934.14 min = 15.57 GPU-hours**, the sum of the nine `wall_clock_min` fields.
The Director re-derived the same total independently
(`docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md`, Part 1).

`llama3.1-8b` was gated on this account when the queue first ran (see the
refusal, exit 78, at the tail of `logs/queue_runner.log`). Timothy accepted the
gate on 2026-08-04; `GATING.md` now records `llama3.1-8b_access = OK` and is
the probe of record. Its three runs still join the queue at the end, unchanged
— the runner skips any run whose `TIMING.md` already exists
(`scripts/s2_queue_runner.py:54-57`), so it can be relaunched as-is. The
Director ruled against both alternatives: "Do not carry Llama-3.1-8B as
unmeasured, and do not substitute Mistral-7B… Accept the gate, run the three
pilots, then the fourth row is measured rather than projected and S2 is
satisfied outright." Remaining cost, at the projected rate: 3 x 159.20 min =
477.6 min = 7.96 GPU-hours (the Director's and the board's "roughly 8").

## Before each launch

```bash
# 1. previous run finished and wrote its measurement
python scripts/s2_timing_pilot.py --list
# 2. GPU actually free (no stragglers)
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

A run is done when `results/s2-timing-pilots/<family>/run_<k>/TIMING.md` exists
with `status = COMPLETE`. The script refuses to overwrite a completed timing
run: re-measuring means deleting it deliberately.

## Launch command (substitute FAMILY and K)

PowerShell, from any directory. The env vars are **required** — the default HF
cache is a junction onto a full drive, and without them the dataset build fails
on xsum/squad.

```powershell
$env:HF_HUB_CACHE="C:\falco\hf-cache\hub"
$env:HF_DATASETS_CACHE="C:\falco\hf-cache\datasets"
$env:PYTHONUNBUFFERED="1"
Start-Process -FilePath "C:\miniconda3\envs\falco\python.exe" `
  -ArgumentList "scripts\s2_timing_pilot.py","--family","FAMILY","--run","K" `
  -WorkingDirectory "C:\falco\rhombic" -WindowStyle Hidden `
  -RedirectStandardOutput "C:\falco\rhombic\results\s2-timing-pilots\logs\run_FAMILY_K.log" `
  -RedirectStandardError  "C:\falco\rhombic\results\s2-timing-pilots\logs\run_FAMILY_K.err" `
  -PassThru
```

The nine accessible runs are all complete. The commands that remain are the
three outstanding `llama3.1-8b` runs, in queue order:

```
--family llama3.1-8b --run 1
--family llama3.1-8b --run 2
--family llama3.1-8b --run 3
```

## On failure

A `status = FAILED` TIMING.md **is the deliverable** for that run, not a reason
to retry differently. It carries the error excerpt (OOM detail included). Do
not change the batch geometry, the step count, or any other recipe constant to
make a run fit: per-family geometry is S3 territory and belongs in a proposal
to the Director, never in a silent code edit. Record, report, stop.

## When the nine runs were done — spent

1. ~~`python scripts/s2_timing_pilot.py --rates`~~ — done; `RATES.md` carries
   the nine measured rows and the three family means.
2. ~~Restate the draft's Section 3 cost table against the measured rates.~~ —
   done and published as `docs/S2_COST_RESTATEMENT_2026-08-04.md`.
3. ~~The >~25% overrun rule (drop option: Llama-3.1-8B).~~ — did not fire:
   every measured family came in *under* its draft estimate; the narrowest
   margin was gemma2-2b at -4.42% (75.51 measured vs ~79 estimated).

## When the three llama runs are done

1. `python scripts/s2_timing_pilot.py --rates` — rebuilds `RATES.md`; the
   `llama3.1-8b_mean_min_per_run = —` line becomes a measurement.
2. Restate the §3 table's fourth row, replacing **159.20 min/run PROJECTED /
   6.633 GPU-days PROJECTED** with the measurement. S2 is then satisfied
   outright rather than for three families of four.
3. S1's GPU-day commitment remains the Director's call. Note that the H2 card
   is held on a different axis than cost: the Director's Ask 1 disposition is
   to "lock H2-D on the three measured families; hold H2-S" pending the
   rank-fraction control — a design confound, not a budget
   (`docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md`, Ask 1;
   `docs/AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md`). Completing the llama
   row does not by itself discharge that hold.
