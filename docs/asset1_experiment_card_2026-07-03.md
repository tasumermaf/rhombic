# Experiment Card — Asset 1 / D1 Bank + D2–D3 Analyses (RUNNABLE)

**Locks the pre-registration `asset1_experiment_plan_2026-07-03.md`. Date:** July 3, 2026
**From:** the Director · **To:** Meridian — campaign starts on receipt
**Grounded against:** repo `main` at `99e0ac5`; Meridian Round-2 answers (§4)
**Status:** hypotheses LOCKED (pre-registered). Do not edit hypotheses; fill result tables only.

## Status change forced by Meridian §4(b)

The pilot's "112 adapters per task" are **112 per-layer bridge matrices (28 layers × {q,k,v,o}) from a single training run per task**. The 72.3% LOO-SVM therefore rests on **three trainings total**, with **task identity and run identity confounded** — the classifier could be reading run idiosyncrasy rather than task. This is disqualifying for the headline claim, and it is exactly the contingency the pre-registration flagged. Consequence:

- **D1 supersedes the pilot.** The 72.3% is reported only as the *pilot that motivated the superseding design*, with its confound stated. The new number stands on N≥30 independent trainings per (family × task).
- **Expect the superseded effect to be smaller than 72.3%.** Removing a run-identity confound typically shrinks apparent separability, and the card is powered for that.

## Locked parameters

| Parameter | Locked value | Basis |
|---|---|---|
| **N per (family × task)** | **40 independent trainings** | CI half-width ±4.9pp on per-family accuracy (vs ±5.6pp at N=30); buys margin against a post-confound effect shrink toward chance. Time is not binding (PI: "cheap and slow, no compromises"), so N chosen on statistical grounds — the extra 120 runs cost only electricity and directly harden the headline. |
| **Families (2)** | **Qwen2.5-1.5B-Instruct × Llama-3.2-1B-Instruct** | Accepts Meridian's pair; meets the tokenizer+corpus criterion (distinct tokenizer, distinct pretraining corpus). Qwen gives pilot-regime continuity. |
| **Tasks (6)** | instruction-follow · code · math · summarization · extractive-QA · classification | Spans W2T's category axes; all public datasets (below). alpaca/code/math retained for pilot continuity; +3 to make the W2T regime contrast measurable. |
| **Total runs** | **480** (40 × 6 × 2) | ≈ 2 weeks sequential on the RTX 6000 Ada; within Meridian's stated envelope. |
| **Train protocol** | 2,000 steps · rank 24 · batch 2 × grad-accum 8 · full length | Matches the pilot regime the claim is scoped to; the 1,000-step halving is withdrawn (no-compromises directive). |
| **Per-run independence** | distinct seed AND distinct data shuffle; full `adapter_state.pt` saved; self-documenting `config.json` | Removes the run-confound; enables D2/D3/D-aux as pure analysis; prevents the BM-002 provenance failure. |

### Task → public dataset binding (confirm availability before launch)
1. **instruction-follow** — Alpaca (pilot continuity)
2. **code** — CodeAlpaca / MBPP-style instruction set
3. **math** — GSM8K
4. **summarization** — XSum (or CNN/DM subset)
5. **extractive-QA** — SQuAD v1.1
6. **classification** — a text-classification set (e.g. AG News or SST-2)

Hold all six fixed across both families. If any dataset is impractical at this scale, swap within the same W2T category and record the swap; do not drop below 6.

## D1 — Cross-family identifiability (LOCKED hypotheses)

**H1.** Within each family independently, a linear classifier on raw all-modules adapter parameters separates the 6 tasks at accuracy **> 1.5× chance** (chance = 16.7%), with a permutation-null-calibrated p < 0.01.
**H2 (regime contrast).** Cross-family transfer (train classifier on family A, test on B) does **not** exceed chance — identifiability is within-family. This failure is itself the finding, not a defect.

**Analysis (pre-registered).**
1. Grouped-CV (leave-one-training-out) linear SVM per family, all-modules. Report accuracy, per-class recall, macro-F1, 95% CI.
2. **Permutation null** (≥1,000 label shuffles); real accuracy must sit outside the null band.
3. **Variance-heterogeneity guard:** report per-task within-class distance; if heterogeneity ≥ pilot's 3.7×, report balanced/rank metrics and diagnose the confused class (pilot: math) rather than hide it in the macro number.
4. Per-module breakdown reported for completeness only; **all-modules is the headline** (the retracted 84.5% was q-proj-only and uncomputed, excluded permanently).
5. W2T reframe: state the regime split as the result. W2T (2603.15990, abstract-verified) establishes that LoRA updates must be mapped to a canonical form (QR→SVD) to remove factorization ambiguity before weights are readable across heterogeneous collections; this work establishes the complementary regime — *within a shared-init family that ambiguity is fixed by construction, so raw parameters are directly separable.* (Meridian's sweep reports a specific flattened-baseline collapse figure of ~2.87% macro-F1; Meridian to confirm the exact number and its source against the W2T paper body before it enters the write-up — the abstract does not carry it.)

## D2 — Cross-task bridge swap (LOCKED)

Meridian §4(c): full `adapter_state.pt` currently exists for alpaca runs only. **The D1 bank build must save complete adapter states for every run**, after which D2 is pure analysis.

**H3 (directional, falsifiable — opposite predictions).**
- *Substrate hypothesis:* swapping a task-j-trained bridge onto task-i-trained A/B costs little on task i (penalty ≈ the cross-seed baseline).
- *Task-specific hypothesis:* cross-task penalty is large and scales with the D1 between-task distance.

**Design.** Full (task_i A/B × task_j bridge) swap matrix per family, evaluated on task_i's held-out set. Diagonal = native. Baselines in the same matrix: **cross-seed swap** (substrate reference, pilot penalty ≈ −1.9%) and **random/permuted bridge** (structure-destroyed reference). Report penalty = swapped − native, per cell.

**Contradiction guard (pre-committed).** If D1 says the bridge carries task identity and D2 says the bridge is task-agnostic, resolve explicitly: measure whether D1's signal lives in **magnitude/scaling** and D2's invariance in **topology**. Report which; do not publish the two claims as if unrelated.

## D3 — Weight-only merge prediction (LOCKED framing)

**Framing constraint:** weight-only, post-hoc, no-training-access, from sentence one — 2606.19549 (Jun 17) owns the training-time form. Predict post-merge degradation from two adapters' parameters alone (features: D1 fingerprint distance, gauge-invariant principal angles on B·A column spaces, per-module distances). Report AUC vs a distance-only baseline. Prior-art-in-kind (cite, don't compete): weights2weights (2406.09413), backdoor-from-weights (2602.15195), WeightWatcher-PEFT, 2606.19549.

## D-aux — Overfit detection (rides along)

Re-verify the r=0.888 deviation↔gap correlation on the D1 bank before including; still unclaimed in the literature. Cite WeightWatcher-PEFT as prior art in kind.

## Corrections adopted from Meridian §3 (mutual standard)

1. **EE-001 has no arXiv ID.** My brief wrongly linked it to arXiv:2104.13478 (Bronstein's geometric-DL proto-book). EE-001 is the internal experiment `results/EE-001-equal-edge-control/`. Bronstein remains a valid *equivariance* citation, though not as EE-001's identifier. **Fix applied to the brief.**
2. **"Stream B" collision.** In program vocabulary Stream A/B is the **IP boundary** (A = open methodology, B = proprietary corpus-coupled), and the geometry/EE-001 work is Stream **A** (public). I am dropping "Stream A/B" as planning vocabulary entirely; planning uses **Asset 1–4** and **Track 1/2** only, Stream A/B stays reserved for the IP fence. **Fix applied to the brief.**

## Accepted from Meridian §2 (audit charter refinement)

The uniform C6b ratio freeze holds, but the Paper 4 audit's **first task is to classify every reported number by controller exposure** (adaptive-governed / fixed-weight / controller-free), so re-measurement is scoped to what the adaptive stability-detector defect can actually have touched. Fixed-weight runs (FC-001 67,501:1; FO-001 262,920:1; 24C-001) and controller-free `rd_graph` results are frozen for uniformity but flagged as likely-clean, and re-measured last.

## Launch checklist (Meridian)

- [ ] Confirm all 6 datasets available at scale; record any within-category swap.
- [ ] Confirm the two base models load and train at the 2k-step protocol on the RTX 6000 Ada.
- [ ] Verify `config.json` persists dataset + seed + shuffle (BM-002 fix in place).
- [ ] Launch the 480-run sequential campaign via the auto-chain + GPU-watchdog.
- [ ] On completion, hand the bank back for D1 analysis; D2/D3/D-aux follow as pure analysis.

*N and design locked on statistical grounds (CI half-width + post-confound shrink margin); family pair and protocol per Meridian §4; corrections §3 adopted. Card is runnable — campaign starts on receipt. — the Director*
