# CARD — E-T4 Tinker Mini-Bank (exploratory; RULED 2026-08-04)

> **Director's ruling (2026-08-04, Ask 5): N = 9 (54 runs), and not as a preference —
> option (a)'s 60 runs would total ~$29.63 billed (measured $0.4460/run + the audited
> $2.7754 spent + the documented +0.35% meter delta), breaching the card's own $28 hard
> abort around run 56 and stranding paid-for adapters mid-collection. Option (b) totals
> ~$26.94 billed, clearing the abort by ~$1.06. Exploratory classification blessed.**
> The N-choice section below is retained as drafted for the record; the ruling governs.
> Arithmetic corrected accordingly: the draft's "(a) … inside, ~$0.70 margin" was wrong
> against the abort guard — the same class of defect as the QUEUE.md stale state,
> self-noted per the Director's instruction.

**Draft for the Director, 2026-07-30.** Class: exploratory/descriptive — no locks, no
confirmatory claims, no multiplicity slots. Purpose: external-validity demonstration +
public artifact for the Thinking Machines "Featured Project" channel and grant application.

## Basis (measured, from the completed signal pilot)

Pilot record: `results/tinker-pilot/PILOT_REPORT.md` (commit 0f39f77). Typed core:
SIGNAL = MIXED as pre-specified best case — raw 1-NN task accuracy 0.000 (0/6),
canonical 1.000 (6/6) — on Tinker's standard bridgeless PEFT LoRA (Qwen3-8B, rank 32,
all-linear + unembed, their optimizer defaults). Cost measured $0.4416–0.4460/run at
~1M train tokens; wall ~3.2 min/run; spend to date $2.687 of the $30 project cap;
billing API cross-check +0.35% vs local ledger. (The pilot report's own header says
"project cap $5" — that was the *pilot phase* sub-cap, not the project cap; the
project cap is $30. Corrected here 2026-08-04 so the two documents do not
disagree.) The pilot report carries two design
notes (loss-weight normalization; batch-size/LR interaction) that bind the mini-bank
config — the card adopts them by reference, not restatement.

## Design

- 6 tasks (asset1 set) × N seeds on Qwen/Qwen3-8B, pilot config held fixed.
- **N is the one open choice, budget-driven:**
  - (a) N=10 → 60 runs ≈ $26.6; total ≈ $29.3 vs $30 cap — inside, ~$0.70 margin.
  - (b) N=9 → 54 runs ≈ $23.9; total ≈ $26.6 — ~$3.4 margin for re-runs/storage drip.
  - Drafter's proposal: **(b)**, margin over maximum n, per the S12 lesson.
- Deliverables: LOO task-identity readout raw vs canonical (descriptive); merge_lint
  demo on vertex-disjoint pairs; all adapters exported, remote checkpoints deleted;
  public write-up + open-source scripts (Featured Project submission).
- Guards: $28 hard abort (unchanged); spend ledger vs billing API reconciliation in
  the report; no claim of confirmatory status anywhere in the write-up.
- Scope sentence for every public artifact: pilot-scale, single family, single recipe,
  descriptive; the registered local cards (H2-at-scale, granularity) remain the
  program's confirmatory instruments.

## Asks

1. Director: rule on N (a/b) and bless the exploratory classification.
2. Timothy: confirm the remaining-budget read or adjust the cap.

## Budget, audited 2026-08-04 (supersedes the ~$27.3 estimate above)

Queried from Tinker's `get_billing_usage` across every day since the key was issued —
the billing API, not our local ledger:

```
2026-07-30  Qwen/Qwen3-8B          training 6,107,108 tok  + 8 checkpoints   = $2.6871
2026-08-03  Qwen/Qwen3.5-9B        training    59,460 tok                    = $0.0870
2026-08-03  4 big models (survey)  training       304 tok                    = $0.0013
TOTAL_METERED_SPEND = $2.7754   REMAINING_OF_30 = $27.22   [derived from metered
tokens x price table; dollar authority is the billing page]
```

The Jul-30 line reconciles to the pilot's own ledger to +180 tokens (+0.003%), which is
exactly the smoke-run delta that report already documented. The Aug-3 lines are a
separate session's work on the same account (a non-WS-4 project) — small, but real, and
recorded here because two of our own records disagreed about whether it existed. **Both
N options remain inside the cap.**
