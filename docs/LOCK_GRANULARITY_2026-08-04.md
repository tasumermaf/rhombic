# LOCKED: Granularity Bridge Ladder — Lock Declaration

**Locked 2026-08-04 by Meridian** (decider authority per the PI's 2026-08-04 delegation),
on the registered card (`REGISTRATION_GRANULARITY_2026-07-30.md` = design + Director
Part-5 rulings) as further conditioned by the Director's 2026-08-04 rulings. From this
moment the card changes only by dated amendment (L-006). No level is analyzed before its
tier gate fires.

## Lock-condition checklist, each against its artifact

| Condition | Status | Evidence |
|---|---|---|
| Cost basis measured + auditable | ✅ | `results/asset1-bank/RATE_EXTRACT.md` @ fa1c4f0 — llama3.2-1b 30.56 min/run (n=240); ladder 18.546 GPU-days incl. Arm B + D7 (`docs/S2_COST_RESTATEMENT_2026-08-04.md` §5) |
| D6 subsample size pinned | ✅ | Director 2026-08-04 Ask 3: **1,000/class**, with (i) CV within-subsample, (ii) ragged-L3 classes use whole pool + realized per-class n reported, (iii) same nominal n at every level |
| D10 lock form | ✅ | acc > 1.5×chance(K) AND perm p < 0.01 AND **κ ≥ 0.40** at every level, κ reported at all levels incl. L0. Objection window closes at this lock; **no objection filed** |
| D7 split-pool control | ✅ funded | mandatory per ruling; 10 runs |
| D8 Arm B | ✅ funded | squad-only deep ladder 2→16 title groups, ~144 runs; promoted over D9 |
| D9 qwen spot check | deferred | revisit only on a llama departure |
| D3/T3 restriction | in force | clean-core (T1+T2) curve reported alongside all-classes at every level; clean-core is the claim on divergence |
| Rate-extract audit gap | ✅ closed | Director's "subject only to the rate extract landing" — landed, pushed, re-derivable |

## Frozen at lock

- Family: llama3.2-1b only (gate ACCEPTED on the account since Jul 3; weights cached).
- Ladder: L1 12×20 · L2 24×10 · L3 ~48×5 (~240 adapters/level); D4 pool floor 1,000/class.
- Fallback 16/8/4: OFF (D9 not funded).
- Recipe: Asset-1 trainer unchanged; cohort tags mandatory; runs excluded from no
  interlock — the completeness gate pattern applies per level.
- **Analysis tier order, frozen (cheapest-first, per the S9 doctrine applied here):**
  L0 re-baseline → L1 → Arm B → L2 → L3 → D7 → D6-per-level runs alongside each level's
  analysis. Each tier's gate records which tiers were already unblinded when it fired.
- **GPU discipline:** every run launches through `scripts/gpu_guard.py` (preflight +
  PAUSE sentinel + claim) — adopted 2026-08-04, non-negotiable given shared-card use.
- Both-outcomes wording for every endpoint as registered; departure-point claims carry
  the D3/T3 clean-core rule; scope limit ("llama3.2-1b only") in every claim.

*The Director retains the grader role in full: per-level results go to him with per-item
artifacts, and the final write-up takes a ladder pass on the-adversary v2.0.2 before any
submission. Lock declared under MERIDIAN_PROGRAM_PLAN_2026-08-04 §3 Track A.*
