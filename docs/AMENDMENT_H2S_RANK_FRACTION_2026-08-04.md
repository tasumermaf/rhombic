# DATED AMENDMENT — H2-at-scale: Rank-Fraction Control Arm (H2-S)

**Filed 2026-08-04 by Meridian**, under L-006 (dated amendments, never silent revisions),
amending the registered card (`REGISTRATION_H2_SCALE_2026-07-30.md` + the Director's
Part-4 rulings). Responds to the Director's 2026-08-04 ruling Part 3 (the rank-fraction
confound) and Part 4 disposition (the spectral-tail contrast). **This amendment takes
effect when the Director rules on it; H2-S remains HELD until then.**

## 1. The decision: option (i), the matched-rank-fraction control arm

The Director's Part 3 established that H2-S as registered is confounded: rank is fixed
at 24 while hidden width varies 2.3×, so r/hidden falls 0.0156 → 0.0117 → 0.0067 across
the Qwen scale ladder, and outcome (B) ("transfer decays with scale") cannot be
distinguished from rank starvation. Two remedies were offered; **Meridian elects (i),
the control arm**, for the Director's own stated reason plus one more: his Part-4
spectral-tail result converts the confound into a *prediction* — if cross-family task
structure is carried by the smallest singular directions, decay under fixed rank should
be steeper than under matched rank fraction, and only the control arm can measure that.
Rescope (ii) would be honest but would forfeit the measurement.

## 2. The arm, specified

```
family        = Qwen/Qwen2.5-7B-Instruct  (hidden 3584 — the widest measured family)
rank          = 56       →  r/hidden = 56/3584 = 0.015625  (anchor: 24/1536 = 0.015625, EXACT)
shape         = 6 tasks × 5 seeds = 30 runs   [proposed; 6 × 10 = 60 if the Director
                wants power parity with the main 7B set — his pin]
recipe        = identical to the card in every other respect (steps, geometry per S3,
                seeds, pools, val split); batch_geometry cohort tag mandatory
cost          = 30 × ~153.01 min ≈ 3.19 GPU-days  [ESTIMATE: assumes rank has negligible
                wall-clock effect; the arm's first run doubles as its timing run and the
                estimate is restated from it before the remaining 29 launch]
```

Analysis: the pre-registered H2-S decision machinery runs unchanged on the main
(rank-24) 7B set; the arm adds one pre-declared contrast — transfer(anchor↔7B, rank 24)
vs transfer(anchor↔7B, rank 56) — with both outcomes stated:

- **If matched-fraction transfer ≥ fixed-rank transfer by a margin exceeding seed noise:**
  the decay component attributable to rank starvation is measured, and any residual decay
  under matched fraction is the scale-geometry signal H2-S set out to test.
- **If the two arms are indistinguishable:** rank starvation is not the mechanism at this
  fraction range, and fixed-rank H2-S outcome (B) regains its interpretability at no cost.

## 3. The spectral-tail contrast (Director Part 4), pre-declared here for his pin

Per the Director's own disposition his tail result is exploratory and quarantined; it is
to be registered on this card for the NEW families, where nothing is unblinded. Proposed
registered wording (the Director pins or edits):

> **Head-vs-tail contrast (pre-declared, corroborating; both outcomes stated).** For each
> new-family pair, standardized transfer is computed for the full 384-dim spectrum, the
> tail-1 slice (smallest singular slot per block, 16 dims), and the head-1 slice (largest,
> 16 dims), against a dimension-matched null of 200 random 16-dim subsets. Outcome (T-A):
> tail-1 exceeds the null's 99th percentile and beats head-1 on every pair — "task identity
> is stored in the small singular directions of the update" becomes a substantive claim.
> Outcome (T-B): the tail advantage does not replicate — it is demoted honestly as
> anchor-pair-specific. This contrast carries no lock and no multiplicity slot; the
> saturated-margin caveat (within-family accuracy at ceiling on 16 dims) is inherited
> from the Director's Part 4.

Interaction with §2, stated as a prediction rather than discovered post-hoc: under the
tail hypothesis, the fixed-rank arm should decay more steeply than the matched-fraction
arm, because growing width at fixed rank removes precisely the small singular directions
that carry transfer.

## 4. Status after this amendment

- **H2-D: LOCKED** on the three measured families, per the Director's Ask-1 ruling.
- **H2-S: HELD** pending the Director's ruling on §2–§3 above.
- **Llama-3.1-8B:** per the same ruling, the gate is to be accepted (Timothy's click) and
  three timing pilots run (~8 GPU-hours); no substitution, no unmeasured carry.
- The Gemma config-confirmation caveat (Ask 2) is closed separately in
  `results/s2-timing-pilots/GATING.md`.
