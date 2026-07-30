# S2 timing pilots — run queue

Registered card: `docs/REGISTRATION_H2_SCALE_2026-07-30.md` (S2, promoted to a
precondition). Three timing-only runs per family, excluded from every bank,
computing no H2 statistic. **One run at a time — sequential GPU discipline.**
Never launch a run while another is training; the single RTX 6000 Ada is the
whole campaign's instrument, and concurrent runs would corrupt the very
quantity being measured.

## In flight

=== VERIFIED STATE ===
launched          = gemma2-2b run_1
pid               = 94948
launched_at_utc   = 2026-07-30T01:33Z
launch_mode       = PowerShell Start-Process, hidden, detached (survives session end)
log               = results/s2-timing-pilots/logs/run_gemma2-2b_1.log
=== END VERIFIED STATE ===

## Queue (launch in this order, one at a time)

Order = the draft's Section 3 table order, which is also the S9 tier order
frozen at registration (cheapest first). `proj_min` is the dry-run PROJECTION
for that family, not a measurement — the point of these runs is to replace it.

| # | family | run | task | status | proj_min |
|---|---|---|---|---|---|
| 1 | gemma2-2b | run_1 | alpaca | LAUNCHED (pid 94948) | 73.0 |
| 2 | gemma2-2b | run_2 | code | queued | 73.0 |
| 3 | gemma2-2b | run_3 | math | queued | 73.0 |
| 4 | qwen2.5-3b | run_1 | alpaca | queued | 70.6 |
| 5 | qwen2.5-3b | run_2 | code | queued | 70.6 |
| 6 | qwen2.5-3b | run_3 | math | queued | 70.6 |
| 7 | qwen2.5-7b | run_1 | alpaca | queued | 156.1 |
| 8 | qwen2.5-7b | run_2 | code | queued | 156.1 |
| 9 | qwen2.5-7b | run_3 | math | queued | 156.1 |
| — | llama3.1-8b | run_1–3 | — | BLOCKED (license gate) | — |

Projected total for the nine accessible runs: **~15 h GPU** (sum of the
projections). Whatever the runs measure supersedes that number.

`llama3.1-8b` is gated on this account (see `GATING.md`). Timothy accepts
model gates manually; after acceptance, re-run `--gate-check` and its three
runs join the queue at the end, unchanged. Until then the family has no
measured rate, so the Section 3 cost table cannot be restated for it and S1
stays open on that row.

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

The nine commands, in queue order:

```
--family gemma2-2b  --run 2
--family gemma2-2b  --run 3
--family qwen2.5-3b --run 1
--family qwen2.5-3b --run 2
--family qwen2.5-3b --run 3
--family qwen2.5-7b --run 1
--family qwen2.5-7b --run 2
--family qwen2.5-7b --run 3
```

## On failure

A `status = FAILED` TIMING.md **is the deliverable** for that run, not a reason
to retry differently. It carries the error excerpt (OOM detail included). Do
not change the batch geometry, the step count, or any other recipe constant to
make a run fit: per-family geometry is S3 territory and belongs in a proposal
to the Director, never in a silent code edit. Record, report, stop.

## When the nine runs are done

1. `python scripts/s2_timing_pilot.py --rates` — rebuilds `RATES.md` from the
   per-run TIMING.md files (measured only).
2. Restate the draft's Section 3 cost table against the measured rates. That
   restatement, published, is what the registered card requires **before the
   card locks**; until then, no bank runs.
3. S1's GPU-day commitment is then the Director's call, with the >~25% overrun
   rule (drop option: Llama-3.1-8B) available if measurement demands it.
