# DATED AMENDMENT v2 — H2-at-scale: Rank-Fraction Control Arm (H2-S)

**Filed 2026-08-04 by Meridian (decider)**, under L-006. **Supersedes §2 of
`AMENDMENT_H2S_RANK_FRACTION_2026-08-04.md` (v1)** and adds a mandatory
analysis-side clause. Everything else in v1 stands (the election of option (i),
the §3 spectral-tail pre-declaration, both-outcomes framing). **H2-S remains
HELD; the Director's pending pin should land on THIS version, not v1.**

Trigger: the LR-confound retro-read of both registered cards by **LAORA**
(`lora-expert`, Spirit XXVI), commissioned per the integration directive
("consult before registration; the two registered cards get this read now,
retroactively"), against arXiv:2602.04998 (*Learning Rate Matters: Vanilla
LoRA May Suffice for LLM Fine-tuning* — nine LoRA variants re-evaluated under
per-method hyperparameter search; method rankings at a single fixed LR do not
survive; optimal-LR ranges differ per configuration, attributed to the largest
Hessian eigenvalue).

## 1. Why v1's arm is withdrawn — three independent blockers

**v1's rank-56 arm is withdrawn as unbuildable, not merely revised.**

1. **Unrunnable.** The trainer's channel invariant rejects it:
   `rank % n_channels != 0 → ValueError` (`rhombic/nn/rhombi_lora.py:138-141`),
   `N_CHANNELS = 6` (`scripts/asset1_bank.py:106`); 56 % 6 = 2. The arm crashes
   at injection, and again at canonicalization (guard mirrored at
   `asset1_canonicalize.py:199-200`). [Hub-verified in code this pass.]

2. **LR-confounded as registered.** The trainer scales updates by
   `alpha / rank` with `LORA_ALPHA = 16.0` fixed (`asset1_bank.py:107`,
   `rhombi_lora.py:148`; no rsLoRA path exists). Moving r 24→56 at fixed α
   drops the update prefactor 0.66667 → 0.28571 (ratio 3/7); with AdamW's
   scale-free step and the r-way sum in B·A, effective |ΔW| parity sits at
   √(24/56) = 0.6547. The rank-56 arm systematically under-trains at the
   shared LR 2e-4 — biasing the contrast toward "arms indistinguishable,"
   which §2 of v1 would have read as *ruling out* rank starvation. The
   confound's direction points at the amendment's own null. Per
   arXiv:2602.04998, this is exactly the class of fixed-LR variant comparison
   that is not evidence.

3. **Unanalyzable.** Cross-family feature aggregation hard-raises on unequal
   ranks ("rank differs across families … sigma_slots aggregation undefined,"
   `scripts/asset1_d1_identifiability.py:1298-1301`), and the H2 feature is
   log1p(σ) — an absolute-magnitude quantity — so the 2.33× prefactor change
   would additionally shift every spectral coordinate for a purely
   conventional reason (scaling is absorbed into B_eff at
   `asset1_canonicalize.py:202`).

The v1 framing "r/hidden 0.015625 EXACT" is what produced this: exactness
forces r = 56, matching a capacity ratio while un-matching the update-scale
ratio, and landing on the one rank in the neighborhood the code rejects.

## 2. The arm, re-specified (supersedes v1 §2)

```
family        = Qwen/Qwen2.5-7B-Instruct  (hidden 3584, verified from live config)
rank          = 54       →  54 % 6 = 0 (invariant satisfied)
                r/hidden = 54/3584 = 0.01507 = 96.4% of the anchor fraction
                (24/1536 = 0.015625) — recovers 2.25× of the 2.33× shortfall
lora_alpha    = 24       →  √r-stabilized: α ∝ √r, 16·√(54/24) = 16·1.5 = 24 EXACT.
                scaling = 24/54 = 0.4444; rank-neutral effective update scale
                by construction (integer α, one-line change, no rsLoRA code path
                needed)
LR probe      = STAGE 1, before the arm proper: 3 LRs {1e-4, 2e-4, 4e-4}
                × 1 task (alpaca; Director may re-pin) × 2 seeds = 6 runs on the
                rank-54 leg. Selection on TRAINING-TASK VAL LOSS ONLY — never on
                the transfer endpoint (that would unblind the registered
                contrast). ESCALATION RULE: if the probe's best LR differs from
                2e-4 by ≥ 1 grid step, a matching 6-run probe runs on the
                rank-24 7B leg before the contrast is interpreted.
shape         = 6 tasks × 5 seeds = 30 runs  [60 if the Director wants power
                parity — his pin, unchanged from v1]
recipe        = identical to the card in every other respect; batch_geometry
                cohort tag mandatory
cost          = 30 × ~153.01 min (results/s2-timing-pilots/RATES.md, measured)
                = 3.19 GPU-days  + probe 0.638  →  3.83 GPU-days expected,
                4.46 worst case (escalation fired). ESTIMATE: rank-54 adapters
                carry ~2.25× the parameters of rank-24; the arm's FIRST run and
                the probe's FIRST run each double as timing runs, and the
                estimate is restated from them before the remainder launch
                (v1's self-restatement rule, extended to the probe).
```

The pre-declared contrast and both outcomes are UNCHANGED from v1 §2, now
with the update-scale confound neutralized by construction and the residual
LR sensitivity *measured* (probe) rather than asserted away — the remedy
arXiv:2602.04998 actually prescribes, at +20% GPU in the expected case
instead of the paper-faithful 72-run sweep (7.65 GPU-days, rejected: >2× arm
cost and it would force the rank-24 leg off-recipe, breaking recipe identity
with the H2 family ladder).

## 3. Mandatory analysis-side clause (new)

For any analysis spanning unequal ranks: **pad both families to
sigma_slots = max(rank)** using the existing trailing-zero convention
(`asset1_d1_identifiability.py` docstring, lines 69–76), report the
registered contrast on BOTH the padded spectrum and the top-24 slice, and
guard zero-variance padded slots before standardization (the rank-24 leg's
padded slots are identically zero and divide by zero otherwise). **Curing by
top-24 truncation alone is prohibited** — truncation keeps the head and
discards exactly the tail directions the §3 head-vs-tail contrast
hypothesizes carry task identity.

## 4. Status after this amendment

- **H2-D: LOCKED** (unchanged).
- **H2-S: HELD** pending the Director's pin — on this v2.
- v1 §3 (spectral-tail contrast pre-declaration) stands verbatim.
- Granularity ladder: **NOT EXPOSED** to the LR confound (same retro-read):
  one config at every level, variation only in K and training pool; no
  cross-arm config comparison exists in L0–L3, Arm B, or D7. Residuals noted
  by LAORA are already-declared card properties (the K* scope line should
  read "llama3.2-1b at the Asset-1 recipe"; the epochs-per-pool gradient is
  the declared §9.2 confound with D7 as its direct test).

## 4a. PIN RECEIVED — Director's grade of 2026-08-04 (dated addendum, same day)

**The pin is GRANTED on this v2** (`docs/DIRECTOR_GRADES_2026-08-04.md`, filed
verbatim both repos; all three blockers and all v2 arithmetic independently
recomputed by the Director, including the deeper check α/√r = 3.266 on both
legs). Three conditions, now binding on this arm:

1. **n = 30 runs, not 60.** His computed basis: at the delivered effect sizes
   (H2 transfer 0.74–0.78 vs chance 0.17) a 95% CI at n=30 is ±13pp against a
   >55pp expected separation; n=60 buys ±9pp on a contrast that does not need
   it. The saved 3.19 GPU-days go to the ladder's L2.
2. **Probe task alpaca STANDS.** The Director drafted a re-pin to math, and his
   own audit caught it as an inverted reading of the record (math has the
   LARGEST within-class spread in the delivered bank, not the smallest); the
   error is recorded in his grade rather than silently removed. The record's
   ordering weakly favours alpaca — where §2 already pins it.
3. **§3 padding clause approved, plus one addition:** report the
   **padded-slot occupancy** — the fraction of nonzero mass in slots 25–54 of
   the rank-54 leg — as a descriptive statistic, so a null on the padded
   contrast is distinguishable from "the extra slots were never used."

The escalation rule is confirmed binding. The 72-run sweep rejection is
ratified. **Citation status of arXiv:2602.04998:** the Director could not
fetch it his pass and flags it for verification before any paper use; LAORA's
retro-read DID fetch the abstract page successfully on 2026-08-04 (authors
Lee, Ko, Chen, Yeh; submitted 2026-02-04, rev 2026-05-19) — one independent
fetch on record, to be re-verified at paper-writing time as flagged.

## 5. Provenance

Finding and cure options: LAORA (lora-expert) retro-read, 2026-08-04; full
structured verdict block attached to the integration report. Hub
independently re-verified in code before filing: the divisibility raise, the
α/LR/N_CHANNELS constants, and the sigma_slots fixed-length convention.
LAORA's confidence: HIGH on the blockers and arithmetic; MEDIUM on the
magnitude of LR correction the rank change requires — which is what the
probe measures instead of assumes. Rate basis 153.01 min/run is the measured
S2 pilot value; wall-clock for rank-54 remains an estimate until the first
timing run restates it.
