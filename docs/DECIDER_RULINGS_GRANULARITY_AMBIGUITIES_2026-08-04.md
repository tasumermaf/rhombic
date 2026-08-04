# Decider Rulings — Granularity Ladder Ambiguities G-1 … G-9

**Date:** 2026-08-04
**Authority:** Meridian as decider (Timothy's delegation of 2026-08-04; plan of
record `C:\falco\docs\MERIDIAN_PROGRAM_PLAN_2026-08-04.md`). The Director grades
this document like any other; his per-item regrade can overturn any ruling here
before the tier it affects fires. Discipline unchanged: the registered card
(`docs/REGISTRATION_GRANULARITY_2026-07-30.md`) and the lock
(`docs/LOCK_GRANULARITY_2026-08-04.md`) bind; deviations are dated amendments
(L-006), never silent revisions.

**Subject:** The Stage-0 label build (`scripts/granularity_labels.py`,
`results/granularity/labels/LABELS_REPORT.md`) surfaced nine ambiguities in the
registered card, each resolved provisionally by the builder so the build could
proceed. This document converts those provisional resolutions into dated
rulings, so L1 can launch on a ruled basis rather than an assumed one.

---

## Rulings

**G-1 — balance policy: RULED `--balance none` (ratified).**
Three independent grounds. (1) *The card's own arithmetic:* only `none`
reproduces §3's stated realized pools — the build computes L1 per-class training
pools of 2,628 (math:steps_ge4) to 24,788 (alpaca:no_input), matching §3's
"~3.5k – 24k" exactly; neither alternative policy does. §4's "common per-class
n" sentence is the part of the card inconsistent with the card's own numbers.
(2) *Anchor comparability:* L0's 240 adapters exist and were trained on
unbalanced pools; balancing L1 would move two variables (K and pool policy)
between the curve's first two points. (3) *Confound control:* level-wide
balancing raises L1 classes from ~1.3 to 12.18 epochs, inflating the §9.2
memorization confound that D7 exists to test. Realized n and discarded mass
under all three policies are computed and logged for every level, so a Director
reversal is a one-flag re-run, not a rebuild.

**G-2 — alpaca verb taxonomy (L2/L3): ACCEPTED PROVISIONALLY, Director review
required before L2 fires.** The frozen Meridian-authored 8-family taxonomy
(with L2 as a strict pairwise coarsening) is an authored label space inside a
registered card. L2/L3 are gated regardless (G-5), so nothing runs on this
taxonomy before the Director has seen it. No effect on L1.

**G-3 — code language classes (L2/L3): RULED as built.** Four named languages
(sql, html_css, javascript, python) under a frozen ordered-pattern rule;
the 43% residual is discarded and logged, not folded into an "everything else"
class that the D6 data-space reference would correctly read as inseparable.
No effect on L1.

**G-4 — xsum L1 axis: RULED doc-length median [T2] (ratified).** The card
offers T2 doc-length or T3 topics and does not choose. Doc-length makes L1
12/12 clean-core, annotator-free, and launchable now — and L1 is the only
funded tier after L0. Disclosed consequence stands in the readout: the L1
all-classes and clean-core curves coincide, so the D3 divergence rule has
nothing to bite on at L1.

**G-5 — T3 annotator (xsum L2/L3): STOP CONFIRMED.** The card admits T3 cells
but pins no annotator. The builder's refusal to materialize them is correct.
L2 and L3 remain unlaunchable until a dated amendment pins model id + prompt +
topic inventory + frozen assignment procedure + seed. Sequencing: the amendment
draft is due while L1 trains (L1 ≈ 5 GPU-days; nothing is on the critical path
yet), and it goes to the Director with this document's grade.

**G-6 — Arm B seed allocation: RULED 16/8/4/3 at rungs 2/4/8/16 (= 144 runs
exactly).** Preserves both the card's stated 2→16 span and the locked total
(3.056 GPU-days at the measured 30.56 min/run = 144.0 runs). The alternative
(constant-N at 3 rungs) honours D2's constant-N rationale but drops the 2-group
rung the card names — the card's text wins. Disclosed caveat stands: Arm B's
classifier N varies by rung (32/32/32/48) and is reported as such.

**G-7 — D7 halves: RULED largest-pool T1-tier L2 class, halves from the FULL
class pool.** Deterministic and recorded; each half clears the D4 floor. D7 is
a control, not a level cell, and is not level-balanced. The ~2× epoch
consequence is reported with the control.

**G-8 — L3 class counts: RECORDED DERIVATION, no ruling required.** xsum 12 +
squad 16 + alpaca 8 + code 4 + math 4 + agnews 4 = K 48, 240 adapters — the
only point inside the card's stated ranges that hits the lock's numbers
exactly. Contestable at grade time like anything else.

**G-9 — D4 floor timing: RULED post-val (the stricter reading).** The floor
reads literally as a floor on the *training* pool, so eligibility requires
≥ 1,500 class rows. Both counts (n_class_rows, n_train_pool) are reported.

---

## Launch consequence

With G-1, G-4, and G-9 ruled, **L1 is launchable**: 12 classes × 20 seeds =
240 runs, clean-core 12/12, chance 0.0833, lock bar 0.125, projected
240 × 30.56 min ≈ 5.09 GPU-days. The queue
(`scripts/granularity_queue.py`) enqueues L1 only; Arm B, L2, L3 and D7 are
gated by their own manifests and by G-5. Dry run `L1/dryrun_006`
(math:steps_ge4, 60 steps) verified the delegation seam end-to-end:
trainer-reported pool 2,628 = frozen manifest, val 500 pinned
(bb165a5799161ffb), bank manifest sha unchanged
(a2004910a8a290a1), gpu_guard engaged, projected 30.98 min/full-run against
the 30.56 measured basis.

## For the Director's grade

All nine rulings above, plus two specific asks: (1) review the G-2 authored
taxonomy before L2 fires; (2) the G-5 annotator amendment will arrive as its
own dated document — rule on it independently of this batch.
