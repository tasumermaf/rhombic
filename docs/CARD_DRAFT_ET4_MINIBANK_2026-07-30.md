# CARD DRAFT — E-T4 Tinker Mini-Bank (exploratory, light card)

**Draft for the Director, 2026-07-30.** Class: exploratory/descriptive — no locks, no
confirmatory claims, no multiplicity slots. Purpose: external-validity demonstration +
public artifact for the Thinking Machines "Featured Project" channel and grant application.

## Basis (measured, from the completed signal pilot)

Pilot record: `results/tinker-pilot/PILOT_REPORT.md` (commit 0f39f77). Typed core:
SIGNAL = MIXED as pre-specified best case — raw 1-NN task accuracy 0.000 (0/6),
canonical 1.000 (6/6) — on Tinker's standard bridgeless PEFT LoRA (Qwen3-8B, rank 32,
all-linear + unembed, their optimizer defaults). Cost measured $0.4416–0.4460/run at
~1M train tokens; wall ~3.2 min/run; spend to date $2.687 of the $30 project cap;
billing API cross-check +0.35% vs local ledger. The pilot report carries two design
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
2. Timothy: confirm the remaining-budget read (~$27.3 available) or adjust the cap.
