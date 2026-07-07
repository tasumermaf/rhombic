# Director Decisions — Asset-1 Launch, A1/A2, Pipeline Sign-offs

**Date:** 2026-07-06
**Source:** `Downloads/Telegram Desktop/director_decisions_2026-07-06.md` (the Director's ruling on Meridian's `AMENDMENTS_REQUEST_2026-07-04.md` + §6 pipeline sign-offs)
**Status:** These are the pre-registration addenda. Locked before the bank completes; the D1/D2/D3 tools are updated to match. Do not re-litigate after unblinding.

---

## A1 — throughput: ADOPTED (bs4×ga4)

Switch micro-batch 2 × grad-accum 8 → **4 × 4**, effective batch unchanged at 16.

**Bit-equivalence verified before the switch** (the Director's condition). All three no-batch-dependence conditions hold in `scripts/asset1_bank.py` + `asset1_datasets.py`:
1. **No batch-norm.** Qwen2.5 / Llama-3.2 use RMSNorm (batch-independent); LoRA adapters are linear. Architectural.
2. **Gradient clip on the accumulated gradient.** `clip_grad_norm_` fires only at the accumulation boundary (`asset1_bank.py:726`, inside `if micro_step % GRAD_ACCUM == 0`), so it clips the summed gradient once per optimizer step — identical total in both geometries.
3. **LR schedule counts optimizer steps.** `scheduler.step()` (`asset1_bank.py:731`) is inside the same boundary; `global_step` terminates the loop at 2000 **optimizer** steps regardless of GRAD_ACCUM. Both geometries see identical 32000 sequences over identical 2000 steps within a single epoch (pool 40000 > 32000, no reshuffle).

Plus the underlying precondition: `asset1_datasets.py` uses `padding="max_length"`, `max_length=512`, `truncation=True`, and `labels = input_ids.clone()` with **no −100 masking** — every sequence contributes exactly 512 (511 shifted) equally-weighted token losses, so token-mean loss is invariant to the micro-batch partition. bs4×ga4 is therefore bit-equivalent to bs2×ga8 up to float summation order; re-running completed runs is for **provenance uniformity, not correctness**.

**Execution:** drain (STOP) → edit BATCH_SIZE/GRAD_ACCUM + tag batch geometry in config → archive the bs2×ga8 cohort to a sibling dir (preserved for an empirical equivalence spot-check) → relaunch (all 480 re-run at bs4×ga4). Every config records `batch_size`/`gradient_accumulation`/`effective_batch` + an explicit `batch_geometry` tag so cohorts are never silently mixed.

## A2 — canonicalization: ADOPTED as pre-registered; D1 spine corrected

1. **D1 regime-contrast spine REWRITTEN.** The drafted axis ("canonicalization necessary at hub scale, unnecessary within a controlled family") attributes the split to the wrong variables — W2T's own collections are same-base/same-rank families (their Table 6). The honest distinguishing variable is **label granularity / task structure** (their 10k+ fine-grained attribute classes vs our 6 coarse tasks), not hub-scale-vs-family. Every D1 sentence is rewritten to that axis.
2. **Run D1 on BOTH raw and W2T-canonical** (QR→SVD, bridge absorbed into B′/A′) representations within each family. Pre-registered reading: *raw ≈ canonical within-family* → factorization ambiguity is benign within a family; *canonical ≫ raw* → kills the raw-parameter framing honestly. Tooling built + GL(r)-invariance-tested to ~1e-13. `--representation` default → **both**.

## §6 pipeline sign-offs

### H2 (regime-contrast headline) — RULED
- **(a) Representation:** primary = **depth-binned singular-value spectra of effective B′A′** (naturally common-dimensioned + gauge-invariant across families of different hidden dim). Canonicalize probe-projection = **corroborating**; disagreement between the two is itself reportable.
- **(b) Decision rule:** H2 (transfer fails) is **supported** iff cross-family accuracy is **not** significantly above chance at one-sided **α = 0.01** (exact binomial) in the **shift-controlled** representation, **AND** is below within-family accuracy by a margin of **≥ 15 percentage points**. **Both directions (A→B and B→A) must meet it**; report each.
- **(c) Shift control:** BLESSED. Family-identity probe + per-family z-scoring. **The shift-controlled variant is the headline; raw is reported as descriptive.**

### D1 details — DEFAULTS APPROVED
SVM C=1.0; Wilson 95% CI (permutation-p is the calibrated inference); Euclidean heterogeneity distance in feature space; per-module breakdown completeness-only, **no permutation null** (avoids 112 unregistered tests).

### D2 — DEFAULTS APPROVED + one OVERRIDE
- K=3 donor/recipient per cell; **permuted-deviation cell [I + permute(B−I)] is THE H3 structure reference**; full-entry permutation retained as identity-backbone contrast; one shared derangement over 36 entries; identity-bridge reference cell included.
- **OVERRIDE:** run the **magnitude/topology decomposition cells UNCONDITIONALLY** (+180 evals/family, CPU/eval-only, zero training), not gated on whether the D1/D2 contradiction materializes — gating on data-dependent materialization is a post-hoc forking point. It characterizes *where* the task signal lives regardless.

### D3 — DEFAULTS APPROVED + one OVERRIDE
- Pair-selection with task-cell stratification + same-task pairs as cross-seed reference: approved. Group-aware per-family AUC = headline, pooled = descriptive: approved.
- **OVERRIDE the label rule:** primary = **fixed relative-degradation threshold, declared now**: a merge is a **"conflict"** if it degrades **either** endpoint task by **≥ 5% relative** (perplexity up or task-metric down) vs that task's native adapter. Median-split → **secondary/descriptive robustness check only**. If the 5% threshold yields degenerate balance (< 10% positives), **report that as a finding** and fall back to the pre-declared median-split.

### D-aux — DEFAULTS APPROVED
Gap trajectory from step 100 (step-0 train_loss null by design); within-task stratified correlations as the Simpson's-paradox guard; update-magnitude covariate descriptive-only. **Re-verify r=0.888 deviation↔gap on the bank before including.**

## §5 (fixed-weight exposure) — ACCEPTED for the record

Fixed-weight runs are **partially exposed** (CL1 connectivity + CL3 stability STABLE declarations run unconditionally; only CL2 is skipped under `--fixed-contrastive`). "Certifiably clean" set shrinks to **controller-free only** (P0, rd_graph/BM, FI-003). WL-001 is the caught-in-the-act proof (pinned bridge_lr_scale=1.0, declared STABLE while deviation grew to 2.12). Uniform freeze already covered public exposure; the audit's clean-set definition tightens.

**Three collateral errors → audit blocker list.** The retracted Holly **"Wan 2.1 14B" scale claim is purge-on-sight — remove, don't just comment.**
→ **DONE 2026-07-06:** stripped from `paper/paper4/paper4-main.tex` body (was line ~249). Also removed the unanchored "82,154:1 across 60,000+ bridges" aggregate (no repo anchor; it exceeded every real per-scale number). Replaced with anchored numbers — TinyLlama H-ch6 **70,404:1**, Qwen 7B Exp 3.0 **22,477:1**. Header blocker #1 marked RESOLVED.

## Octahedral re-anchor (blocker #4) — PARTIAL 2026-07-06

Hermes back online. **FO-001 262,920:1 now anchored locally** (`results/octahedral-hermes-anchor/FO-001/`, ratio confirmed at step 10000 in the training log). **O-001 473,622:1 still unanchored** — its Hermes `results.json` carries a **null co_cross_ratio throughout** (the 473,622 came from a post-hoc block analysis, not the training log); must be recomputed from the O-001 adapter before the n=4 endpoint of the inverse-n ordering (474K > 70K > 42K) can be trusted.

---

*Recorded by Meridian, 2026-07-06. The Director still owes the Paper 4 ratio / BM-001 verification pass against current `main`. Next Director action after the bank lands: D1 analysis.*

---

**2026-07-07 addendum:** the Paper 4 / BM-001 verification pass was delivered and closed the same day (see `docs/PAPER4_VERIFICATION_PASS.md` and STATUS). Amendments **A3 (vocab_signature D1 arm #3), A4 (BM-003 Configs G/H + dissociation endpoint), A5 (BM-004 v2)** were all **APPROVED as pre-registered, with conditions**, in the Director's ruling of 2026-07-07 — recorded verbatim at `docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md`. Conditions encoded the same day: kv_mode surfaced + both variants in the D1 arm (`asset1_vocab_signature.py`, `asset1_d1_identifiability.py`); task-class freeze timestamped in `results/BM-003/PROTOCOL.md`; E3 thresholds justified against our own chance/control in `BM004_PREREGISTRATION_v2_2026-07-07.md` §4; F2 wired as a hard interlock in `scripts/bm004_runner.py` (no bypass). D1 now has three representation arms; D1 remains the Director's on bank delivery.
