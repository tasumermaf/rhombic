# Director's Sign-Off — Level-B J-lens Positive Control

**Date:** July 9, 2026
**From:** the Director · **To:** Meridian (cc: PI)
**Re:** the synthetic positive control I required before Stage-C reporting (J-lens Stage-B ruling, 2026-07-09)
**Verified against:** repo `ccf7e06f`. I inspected the committed artifact and then reproduced the control from scratch with `--positive-control`; my run matches the committed numbers to the digit. **Condition met. Level B is a usable arbiter.**

## The condition, and why it mattered

My Stage-B ruling approved the four defaults but held Stage-C reporting behind one hard condition: a synthetic positive control in Level A's selftest class, because an arbiter whose null result cannot be distinguished from "lens too weak to see it" is not an arbiter. This control closes exactly that gap, and it does more than the minimum I asked for.

## What I verified, reproduced not read

I ran `python scripts/asset1_jacobian_lens.py --positive-control` myself. My reproduction against the committed `results/jlens-positive-control/jlens_positive_control.json`:

| Arm | Committed | My reproduction |
|-----|-----------|-----------------|
| planted LOO (chance 0.25) | 1.000 | 1.000 |
| matched-null LOO | 0.167 | 0.167 |
| output-null lensed norm | 2.76e-15 | 2.77e-15 (float noise on zero vectors) |
| output-null raw LOO | 1.000 | 1.000 |
| sabotage (wrong-lens) LOO | 0.083 | 0.083 |
| PASS / all three checks | true | true |

All three acceptance checks pass in my run: `planted_recovered`, `output_null_reads_null`, `sabotage_detected`.

## The three properties that make this a real control

1. **It recovers planted propagation above a genuine null.** Planted LOO 1.0 against chance 0.25, matched-null 0.167, permutation-null mean 0.247 (p99 0.583). The plant is not merely above chance, it is above the empirical null band. This is the "a null Stage-C result means no structure, not a blind lens" guarantee.
2. **The output-null plant reads null the honest way.** Lensed signature norm 2.8e-15 (numerically zero) while raw LOO is 1.0. The artifact states plainly that the lensed LOO=1.0 on those zero vectors is not operationally meaningful and "reads null" is evidenced by the norm, not the LOO. That is the correct call and I am glad it is written down rather than reported as a second clean recovery: the update genuinely carried a signal in raw space, and the correct lens annihilated it because its column space lies in the toy map's null space.
3. **The sabotage arm defeats the tautology objection.** Signing the planted bank with a different toy map's lens collapses recovery to 0.083 (below chance). A lens that recovered planted structure regardless of whether it was the right lens would prove nothing; this one recovers only with the matching map. Combined with the output-null confound (conf_scale 8.0: an output-null component that swamps a wrong lens but is annihilated by the correct one), the control is not a lens grading its own homework. You added this arm without my asking; it is the piece that makes the control convincing.

I also confirmed the acceptance statistic is `loo_nearest_centroid_accuracy` imported from `asset1_vocab_signature` (line 115), the same metric Level A's selftest uses, not a re-implementation, and that probes (stream 71) and sketch (stream 72) are import-shared with Level A per condition (d) of the ruling.

## Net

- Positive control: **VERIFIED by independent reproduction** at `ccf7e06f` (planted 1.0, matched-null 0.167, sabotage 0.083, output-null norm ~1e-15, all checks pass) — my numbers match the committed artifact.
- **The Stage-B hard condition is cleared. Level B is a usable arbiter**, not just a computed number: a null Stage-C reading is now interpretable as absence of structure rather than lens weakness, and a positive reading is a lower bound per the within-position conservativeness already ruled.
- Nothing else changes: the four Stage-B defaults stand as approved, disclosures encoded in every artifact, Hermes lens estimation proceeds under the non-reportable class, Stage C stays bank-completeness-gated.
- With the bank at 96/480 and zero failures, Level B clears its gate well before ~Jul 19, so it fires alongside D1 on delivery as planned. **Nothing open on my side for either arm until the bank lands.**

*Positive control reproduced from scratch at `ccf7e06f`; planted/matched-null/sabotage/output-null all match committed; Level A metric + probes/sketch import-shared confirmed; Stage-B condition cleared, Level B is an arbiter. — the Director*

---

*Recorded verbatim by Meridian from the inbound sign-off, 2026-07-09 (delivered via the PI in-session).*
