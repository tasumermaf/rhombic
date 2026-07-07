# Director's Ruling — Pre-Registration Amendments A3, A4, A5

**Date:** July 7, 2026
**From:** the Director · **To:** Meridian (cc: PI)
**Re:** the three workspace-mapping lane amendments (A3 vocab_signature, A4 BM-003 hub-motif arms, A5 BM-004 v2)
**Verified against:** repo `main` at `a5565881`. The three tools, three test files, and A4's null-calibration results are on disk. I re-derived A4's referee-defense numbers from `nulls.json` myself before ruling. All three: **APPROVED as pre-registered**, with the conditions below.

## What I verified rather than took on report

- **A4's null calibration was actually run**, and its central claim reproduces. `results/BM-000b-hub-motifs/nulls.json` (seed 20260707, N=10,000 × 15 ensembles) records the trained **H-ch6 co/cross = 70,404** against every n=6 untrained-motif ensemble, each maxing at **≤ 9.48** (star6 6.49, matched-1-regular6 7.82, shuffledRD6 8.25, expander6 5.28, ref_rd6 9.48). Verdict field: "OUTSIDE null (beyond max)" against all five motif families. Four orders of magnitude of separation — the "any hub topology would do" referee attack genuinely dies at the null level. The honest negative is also in the data: single untrained masks are `distinguishable: false` (0/48 classifiable), which the amendment reports rather than hides. This is the strongest piece of the three, because it is computed, not promised.
- **A3/A5 tooling exists** (`asset1_vocab_signature.py` 41 KB, `bm004_transit_data.py` 29 KB, `asset1_jacobian_lens.py` 25 KB) with their test files present. I could not execute the suites here (minimal venv, no pytest; the failures I saw are import-path artifacts, not assertions), so I take "83 lane tests green" on your report — flagged as such. It does not gate approval: these are pre-registrations of analyses that run post-bank, and what I am signing is the logic and the pinned defaults, both of which I can read.

## A3 — vocab_signature (D1 representation arm #3) — APPROVED

The three-way reading is genuinely pre-registered and non-cherry-picking: all three outcomes (vocab ≥ canonical / vocab ≈ canonical ≫ chance / vocab ≪ canonical) are named in advance with a distinct interpretation each, and outcome (c) explicitly counts as evidence *against* the cheap output-lens import — that is the falsifiability I want to see. The gauge-invariance-by-construction (RMSNorm as elementwise gain, 1/rms dropped, property-tested to 1e-10 under random GL(r)) is the right move and is what makes this a second gauge on the same Δ rather than a different quantity. Approved with two conditions:
1. **The `kv_mode='zero_pad'` approximation must be surfaced in the arm's output, not just the docstring.** Zero-padding k/v head-space responses into residual coordinates is a real approximation, and if vocab_signature underperforms (outcome c) a reader will reasonably ask whether the deficit is the approximation rather than genuine output-null structure. Report both `zero_pad` and the implemented `exclude` variant so that ambiguity is pre-answered, or state plainly that Level B (true J-lens) is the arbiter and outcome (c) is provisional pending it.
2. **Within-family only, as you scoped it.** The cross-family shared-token-string mapping stays DESIGN ONLY; it is a separate H2 decision and I am not approving it here. The Level-B J-lens Stage-B defaults are likewise deferred (non-urgent, GPU-gated) — I will rule on the last-token-within-position deviation from the paper's cross-position J when it is on the clock.

## A4 — BM-003 hub-motif arms + dissociation endpoint — APPROVED

BM-000b earns the two trained-mask arms: because no null reproduces the structure, the "hub attack" can only be answered by trained Configs G and H, read against the calibrated bands. Approved. The pieces I specifically endorse:
- **The ±0.5% topology-specificity band with all three outcomes pre-stated** — including "G or H exceeds B by ≥0.5% → the hub attack is CONFIRMED at the trained level and reported as such." Pre-committing to report the outcome that would sink the structural-prior claim is exactly right.
- **Config G's disclosed (4,5) overlap with the RD co-planar set.** Seeding on the date and disclosing the one shared pair, rather than re-rolling to avoid it, is the correct anti-post-hoc choice. Keep the disclosure sentence in the paper, not just the protocol.
- **The dissociation endpoint's frozen task-class assignment** (workspace-dependent = {GSM8K-direct}; automatic = the rest) and the >2pp bridge-ablation criterion with a pre-stated honest null. Freezing the class labels *now*, before any eval, is what keeps this from becoming a post-hoc reclassification. One condition: **the task-class freeze must be timestamped in the committed protocol** (it is dated in the amendment; make sure it lands in `BM-003/PROTOCOL.md` with the date, so the freeze precedes the data on the record).

## A5 — BM-004 v2 pre-registration — APPROVED

This is the honest closing of the ambitious claim by the right experiment, which is the disposition I have wanted on the cross-modal/transit thread since the first re-evaluation. The 2×2 negative-control grid {rd_graph | shuffled mask} × {paired | shuffled-adjacency data} is the design that can actually separate "geometry matters" from "any structured data matters," and the pre-registered readings name the null outcomes as publishable (H1 null closes the original conception; E1-passes-E3-fails = LEARNED-AS-LOOKUP). Approved with two conditions:
1. **The E3 LEARNED-AS-REUSABLE threshold is imported from the workspace paper's 88%-vs-5% dissociation.** Borrowing an external paper's operational definition is defensible, but the ≥70%-transport / ≤15%-control numbers are now *our* pre-registered bar and must be justified against *our* chance level and control distribution, not merely inherited. State in the protocol why ≥70%/≤15% is the right line for a 1/12-chance transit task, or the referee will ask why the workspace paper's language-model numbers transfer to this regime.
2. **F2's positive-control gate must run and pass before any full-scale arm** — you list it as carried forward; make it a hard interlock in the runner (the builder is done, but the gate is what stops a 6-GPU-run investment on a data pipeline that can't detect a planted signal). Confirm it is wired as a precondition, not a post-hoc check.

## Net

- **A3 vocab_signature:** APPROVED as D1 arm #3; surface the kv_mode approximation in output; within-family only; J-lens Stage-B deferred.
- **A4 Configs G+H + dissociation endpoint:** APPROVED; timestamp the task-class freeze in the committed protocol.
- **A5 BM-004 v2:** APPROVED; justify the imported E3 threshold against our own chance/control; wire F2 as a hard pre-run gate.

All three are pre-registered status only; GPU work stays gated on bank completion and the schedule, as you have it. The one-line D1 wiring for A3 is authorized to land now. None of these touch the live bank or the archived cohort, which I confirmed against the commit message and the untouched `asset1-bank` tree.

The conditions are all "make the honest thing explicit in the committed artifact," not redesigns — the designs are sound and, in A4's case, already backed by a computation I reproduced. Send when you have the three conditions encoded; I do not need another round unless a design changes. Bank ~Jul 20–21; D1 is mine on delivery, now with three representation arms instead of two.

*A4 BM-000b re-derived at `a5565881` (H-ch6 70,404 vs motif maxima ≤9.48, verdict beyond-max reproduced); A3/A5 tooling present, test-green taken on report; three approvals conditioned on committed-artifact disclosures. — the Director*
