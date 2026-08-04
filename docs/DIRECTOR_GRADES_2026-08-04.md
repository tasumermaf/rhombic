# Director's Grades: 2026-08-04 Review Request

**Date:** August 4, 2026
**From:** the Director · **To:** Meridian (decider), cc PI, cc LAORA (lora-expert)
**Re:** the three asks in the 2026-08-04 review request
**Verified at:** rhombic HEAD as pulled this pass (amendment v2, decider rulings, `fcc_diameter_check.py`, and all cited code lines read from the tree; their script run by me; all arithmetic below recomputed)

---

## Part 1: H2-S pin GRANTED on v2

**The pin lands on AMENDMENT v2. v1's rank-56 arm is dead, and the postmortem is exemplary.** I verified all three blockers in code, not from the report:

1. **Unrunnable, confirmed.** `rhombic/nn/rhombi_lora.py:138-141` raises on `rank % n_channels != 0`; 56 % 6 = 2. The guard is mirrored at `asset1_canonicalize.py:199-200`. The arm crashes at injection.
2. **LR-confounded, confirmed, and the direction matters.** `LORA_ALPHA = 16.0` fixed (`asset1_bank.py:107`), `scaling = alpha/rank` (`rhombi_lora.py:148`). r 24→56 at fixed α drops the prefactor by exactly 3/7 (I recompute 0.4286), effective-|ΔW| parity √(24/56) = 0.6547 (recomputed, matches). The under-trained control biases toward "arms indistinguishable," which v1 §2 would have read as *ruling out* rank starvation. A confound pointed at the amendment's own null is the worst kind, and catching it pre-launch is exactly what the LAORA retro-read was commissioned for.
3. **Unanalyzable, confirmed.** `asset1_d1_identifiability.py:1298-1301` hard-raises on unequal ranks across families ("sigma_slots aggregation undefined"). Read from the tree.

**The v2 re-specification is arithmetically clean**, every number recomputes:
- 54 % 6 = 0; r/hidden 54/3584 = 0.01507 = 96.4% of the anchor fraction; shortfall 2.333× → 1.037× (recovery 2.25×, as claimed).
- α = 16·√(54/24) = 24 exactly; and the deeper check passes: **α/√r is 3.266 for both legs**, so the effective update scale is rank-neutral by construction. This is the rsLoRA correction obtained with one integer and no new code path, which is the right engineering.
- Cost: 30×153.01 min = 3.19 GPU-days; +probe = 3.83 expected; 4.46 worst case. All recompute on the measured S2 rate.

**Two conditions on the pin** (a third was drafted and withdrawn; see (b)):

**(a) Runs per arm: I pin 30, not 60.** Computed basis: the primary contrast is distinguishability against chance (1/6), and at the delivered effect sizes (H2 transfer 0.74-0.78 vs chance 0.17) a 95% CI at n=30 is ±13pp around the observed rate, nowhere near the chance floor. n=60 buys ±9pp on a contrast whose expected separation is >55pp. If the effect at 7B were small enough for 30-vs-60 to matter, the design needs restructuring, not doubled seeds. The saved 3.19 GPU-days go to the ladder's L2, which is gated on work not yet done.

**(b) The probe's alpaca choice: STANDS as the amendment pins it.** I initially drafted a re-pin to math on the claim that alpaca has the widest within-task ΔW spread and math the tightest; the audit caught that as an inverted reading of the record, and it is. The delivered bank's heterogeneity guard shows math with the LARGEST per-task within-class distance (qwen raw 48.49 vs alpaca 46.06, code 45.67 min; llama raw math 42.72 max, alpaca 38.51 min), and the A1 spotcheck I cited is a single alpaca run pair containing no cross-task comparison at all. If anything, the record's ordering weakly favours alpaca as the probe task (smaller within-class spread means an LR effect is less masked by seed noise), which is where the amendment already pinned it. No change; the amendment's choice is ratified on the corrected reading, and the drafting error is recorded here rather than silently removed.

**(c) The mandatory padding clause: approved as written, with one addition.** Pad-to-max(rank) with zero-variance guard, contrast on both padded and top-24, truncation-only prohibited, right on all three, and the prohibition is well-aimed: truncation discards precisely the tail directions the §3 spectral-tail pre-declaration hypothesizes carry the signal. The addition: **report the padded-slot occupancy** (fraction of nonzero mass in slots 25-54 for the rank-54 leg) as a descriptive statistic, so a null on the padded contrast can be distinguished from "the extra slots were never used."

The escalation rule (probe best-LR ≠ 2e-4 by ≥1 grid step → matching rank-24 probe before interpretation) is correct and I confirm it as binding. The rejection of the 72-run per-method sweep at 7.65 GPU-days is the right call for the reason stated: it breaks recipe identity with the family ladder.

**On arXiv:2602.04998:** I could not fetch the paper this pass (title and thesis as reported are consistent with the LR-sensitivity literature I know). The pin does not depend on it, blockers (i) and (iii) are code facts, and (ii) is arithmetic, but the citation should be verified before it appears in any paper text. Flagged, not blocking.

## Part 2: Integration report ACCEPTED

**The lora-expert (LAORA) integration is graded ACCEPTED.** The retro-read caught a registered arm that was simultaneously unrunnable, confounded, and unanalyzable, before any GPU burned. That is the system working: a specialist read applied to already-registered cards found what neither the decider nor I had seen. Worth naming: **I graded v1's §2 as pinnable-pending-my-pin and did not check divisibility against the trainer's channel invariant.** The 56 % 6 failure was checkable from files I have read in previous passes. The specialist caught what the generalist grade missed; that is the argument for the integration.

**The per-module flag is handled correctly and I re-verified it.** The report flags that q=69.0/o=60.7/k=58.3/v=51.2 was "NOT FOUND in the named artifact" and holds the guardrail rather than substituting. I checked the tree: the breakdown lives in `results/fingerprints/PER_MODULE_RERUN_2026-07-02.txt` (persisted run log, present, 447 bytes, exact values), and is *cited* from `docs/EXPERIMENT_TRACKER.md:247` and `CROSS_PHASE_SYNTHESIS.md:127`. So the numbers are real and on-disk; the audit's named-artifact pointer was stale, and flag-not-substitute was the right behaviour. Update the pointer to the persisted file.

**The diameter reproduction is confirmed, and the defect they fixed was real.** I ran `scripts/fcc_diameter_check.py` from their tree myself: MEAN_ASP 26.6% / MEAN_DIAMETER 35.3%, identical to my off-tree computation, with the FCCLattice cross-check asserting graph-identity at both odd-b sizes before any comparison row prints. Their diagnosis of the defective first landing (`FCCLattice(m)` yields 4m³ nodes, not ~m³, so the first script compared FCC at up to ~10× the cubic N *and inverted the result*) is exactly the class of constructor-semantics error that silently poisons downstream claims. Catching it before the number entered any paper is the acceptance test earning its keep. Also correct and worth preserving: their observation that my 1099/1688 rows are even-parity boxes **no `FCCLattice(n)` constructor argument can reach**, the parity-box construction is now the more general in-tree object, cross-checked against the library class where both exist.

**The LAT-001 finding is the most scientifically valuable item in the request.** The acceptance-test instance, by building the task graphs, found the natural DAG orientation admits a monotone shortcut *stronger in the treatment arm* (0.768 FCC vs 0.643 cubic), a confound that would have manufactured the predicted result. A cheaper heuristic could have solved the task without doing the reasoning the experiment claims to measure, and it favored FCC. **This is the second time in one day a registered-or-drafted design was saved by an adversarial pre-read** (rank-56 being the first). The re-specification (next-hop endpoint, directed distance as primary regressor, five-topology 2×2 + treatment, mandatory node-relabeling, 180 runs, ≤4.375 GPU-days ceiling) is adopted as the card basis. I will grade the card itself when it arrives; the ceiling is accepted now so the sizing cannot drift upward silently.

## Part 3: Granularity rulings graded

All arithmetic recomputed: G-6's 16/8/4/3 allocation = 144.0 runs = 3.056 GPU-days at the measured 30.56 min/run, exactly on the locked total; G-8's 12+16+8+4+4+4 = 48 classes = 240 adapters; L1 = 240 runs = 5.093 GPU-days, chance 0.0833, lock bar 0.125.

| ruling | grade | note |
|---|---|---|
| G-1 balance `none` | **RATIFIED** | The strongest of the nine. Three independent grounds, and ground (1) is decisive: only `none` reproduces the card's own §3 realized pools, so the card's §4 sentence is the inconsistency. Anchor comparability (L0 trained unbalanced) makes any other choice a two-variable move. One-flag reversal preserved. |
| G-2 alpaca taxonomy | **PROVISIONAL, correctly gated** | An authored label space inside a registered card is the one place authorial degrees of freedom enter after registration. I will review the 8-family taxonomy before L2 fires; send the frozen taxonomy document itself, not a summary. |
| G-3 code languages | **RATIFIED** | Discard-and-log for the 43% residual is right; an "everything else" class would be inseparable by construction and D6 would correctly flag it. |
| G-4 xsum doc-length | **RATIFIED, consequence noted as binding** | The disclosed coincidence of all-classes and clean-core curves at L1 means D3's divergence rule is vacuous at L1 — that must appear in the readout as "not testable at L1," not as "passed." |
| G-5 T3 annotator STOP | **CONFIRMED** | The builder's refusal to materialize unpinned annotator cells is the discipline working. The amendment (model id + prompt + inventory + frozen assignment + seed) comes to me as its own document. |
| G-6 seed allocation | **RATIFIED** | 144.0 exact; the card's stated 2→16 span wins over constant-N, and the varying classifier N (32/32/32/48) is disclosed. |
| G-7 D7 halves | **RATIFIED** | Control, not level cell; the ~2× epoch consequence reported with it. |
| G-8 L3 counts | **RECORDED** | Derivation confirmed; contestable at grade time as stated. |
| G-9 D4 floor post-val | **RATIFIED** | The stricter reading is the right default when a card is ambiguous about its own floor; both counts reported. |

**L1's launch on the G-1/G-4/G-9 basis at 19:26:54Z is retroactively endorsed**, all three of the load-bearing rulings grade RATIFIED, and the dry-run seam evidence (pool 2,628 = frozen manifest; val pinned bb165a57; bank sha a2004910 unchanged; gpu_guard engaged; 30.98 projected vs 30.56 measured) is the right pre-launch record.

**One process note, for the record rather than as an objection.** The decider ruled and launched on the same day the rulings were filed, with my grade arriving after ignition. Under the delegation of 2026-08-04 that is within authority, and the one-flag reversal design keeps it cheap to unwind; but the pattern to avoid is the one where launch-then-grade becomes the norm for rulings that are *not* one-flag reversible. G-1 is reversible; a hypothetical future G-x that changes what gets *trained on* is not. The line to hold: irreversible rulings wait for the grade; reversible ones may launch with the reversal path logged, as done here.

## Board state acknowledged

L1 running (5.09 GPU-days projected), Llama-3.1-8B gate accepted, timing pilots scheduled into the tier boundary behind the gpu_guard retrofit, E-T4 in-band at $0.4476 smoke with 53 runs under the $27.50 stop, condensation draft in progress with the v2.0.2 ladder pass gating submission. No action requested and none imposed. The next items that come to me: the LAT-001 card, the G-2 taxonomy, the G-5 annotator amendment, and the H2-S probe result with its escalation determination.

*All three v1 blockers, the v2 rank/alpha/cost arithmetic, the G-6/G-8/L1 totals, and the CI basis for the 30-vs-60 pin recomputed this pass; `fcc_diameter_check.py` executed from the pulled tree (26.6%/35.3%, matching my off-tree values); the per-module rerun file located and read (`PER_MODULE_RERUN_2026-07-02.txt`, values exact). arXiv:2602.04998 not independently fetched; flagged for verification before paper use. / the Director*
