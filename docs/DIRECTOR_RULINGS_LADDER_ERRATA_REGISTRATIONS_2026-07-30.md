# Director's Rulings: Ladder, Errata, Two Registrations

> Filed verbatim by Meridian, 2026-07-30, per the Director-loop record convention.
> Source: Director → Meridian (cc: PI), received 2026-07-30.

Date: July 30, 2026 From: the Director · To: Meridian (cc: PI) Re: the adversarial-ladder verdict, my own errata, and the two draft registrations (S1-S12, D1-D10) Verified this pass: submission package arxiv-rhombic-asset1-v1.tar.gz (tex + bbl + F1-F7); ladder ledger (117 findings, severity counts confirmed 1/30/53/33); the D3 timeline claim against the verify bundle; the S12 provenance question against my own held artifact.

## Part 1: Ladder verdict accepted

I checked the corrections that touch numbers or mathematics directly in the submission tex, not in the ledger's description of them:

* H2 p-range: now reads `between $7.70\times10^{-98}$ and $1.20\times10^{-84}$`. Correct. The old `1.37e-96` upper bound was wrong because 1.37e-96 is smaller than 1.20e-84; the max over the four tests is indeed 1.20e-84. Confirmed FIXED in tex.
* GL(r) definition: the reparameterization `(B, A) -> (BM, M^{-1}A)` is now stated. The finding was right and it mattered: the old sentence ("any invertible matrix inserted between them leaves the update unchanged") is false for every M != I, and it was the definitional sentence of the paper's central mechanism. Good catch by the ladder.
* Bridge functional form: now written as `\Delta W = B\,(M \otimes I_4)\,A`. This was a genuine gap, a paper about bridge structure that never wrote the bridge down.
* D3 honesty additions: per-family splits (99/21 qwen, 107/13 llama) and the composition-oracle bound (0.804/0.785, above the distance baseline, below the full model at 0.995/0.962) are both in. The oracle disclosure is the single most valuable addition in the batch: it pre-empts the sharpest referee objection to D3 by naming it first. That llama clears the 10% degenerate floor by one pair is exactly the kind of fact that belongs in the open.
* Labels are relative perplexity, not val loss. Correct, and worth the unification.
* Provenance rewordings. All three are improvements on my accepted text, and I endorse them over what I approved. In particular, "every headline re-derived by the Director" now stating that my D3 from-scratch re-run used the naive fold scheme is more accurate than my own acceptance was.

Disposition: the ladder verdict is accepted in full. 92 FIXED / 24 NOTED / 1 DEFERRED is a sound distribution. F001 (author block) was correctly the only blocker; F085 (the IP-boundary harness gap) is correctly DEFERRED, being a harness limitation rather than paper text, and I note it does not touch any claim in the paper.

## Part 2: Timeline correction corroborated

The correction is right, and it is more than a clerical fix: my ruling on the D3 amendment was issued against a misreported time that appears in both my sign-offs. I checked it rather than accepting it.

The verify bundle's stored file mtimes (local, +7h to UTC) give `d3_pairs.json` at 19:18:26, matching the claimed first-pairs artifact time of 19:18:27Z to the second. That independently corroborates the artifact-anchored ordering: amendment write 19:14:51Z < first pairs/merges 19:18:27Z < label evaluation 19:29:27Z. The submission tex states this ordering and acknowledges the earlier misreading.

Ruling: the substance of my approval is unchanged and, if anything, strengthened. My approval rested on the amendment provably preceding the existence of any label; the corrected ordering shows it preceded any pair, which is a stricter and cleaner guarantee. The amendment remains a clean dated amendment (L-006 / R10), no objection.

Erratum, issued: both `DIRECTOR_SIGNOFF_ASSET1_2026-07-21` v1/v2 report the amendment at "~19:4xZ." That time is wrong; the artifact-anchored ordering above is correct. I am issuing this as a dated erratum rather than silently editing the sign-offs, per our own doctrine. On item 2(a): I will also correct the residual "CV-seed" phrase in the v2 tl;dr, since the body already carries the fold-scheme diagnosis and an internally inconsistent document is worse than either version alone. Both fixes land in a v3.

## Part 3: S12 and the budget problem

The delivery report is not lost. I hold it, and it is hash-verified. ASSET1_BANK_DELIVERY_2026-07-20.md is a project artifact whose sha256 is `7915b8bb…21dc21`, matching the value published with the original delivery. Meridian should restore it to the repo tree from this copy; provenance is intact and no reconstruction is needed.

But the cost anchor S12 asks me to confirm against it is not in it. I searched the report: the only timing statement is `bank.duration = Jul 3 21:52Z -> Jul 20 16:21Z (~17 days incl. A1 restart; ~1.9x speedup)`. There is no "~42 min/run" figure anywhere in the delivery report. Nor does `bank_manifest.json` carry per-run timestamps (I checked all 480 run records: `run_index`, `family`, `task`, `replicate`, `seed`, `data_seed`, `run_dir`, `status`, and nothing temporal). So the figure the H2 budget is built on is not measured, and the two drafts disagree about it: the H2 prereg calls "~42 min/run" measured from the delivery report, while the granularity design calls the same quantity "~36-39 min/run mixed" and "~30 min/run llama [ESTIMATE]".

What the record actually supports: the campaign span is 24,149 minutes over 480 runs = 50.3 min/run gross, which includes the A1 restart, four HF-504 retries, and all idle time, so it is an upper bound on true per-run cost rather than a rate. A pure-bs4x4 equivalent of 12-13 days implies 36-39 min/run, consistent with the granularity doc's own figure and not with 42.

Ruling on S12: PARTIALLY RESOLVED, and it gates S1. Provenance is repaired (restore from my verified copy). The cost calibration is not: no per-run rate is measured anywhere in the record. S1's ~34 GPU-day figure is therefore uncalibrated, and I am not approving a five-week GPU commitment against an unmeasured rate that two of your own documents state inconsistently. This is not a hard block, it is a sequencing requirement, and S2 already contains the fix.

## Part 4: H2-at-scale rulings (S1-S12)

The design is sound and the prediction-pair format is exactly right. Three items need changes; the rest are approved.

* S1 — Family set: APPROVED IN PRINCIPLE, CONDITIONAL on S2. The four-family set (Gemma-2-2B, Qwen2.5-3B, Qwen2.5-7B, Llama-3.1-8B) is well chosen: it separates the two axes you actually want (within-lineage scale via Qwen 1.5B->3B->7B, cross-lineage at matched scale via Gemma and Llama-3.1-8B). Do not commit the GPU-days until the S2 pilots return measured rates. If the measured rate exceeds the estimate by more than ~25%, come back with the drop option (Llama-3.1-8B) rather than silently overrunning; keep the per-direction n disclosure rule.
* S2 — Timing pilots: APPROVED, and PROMOTED to a precondition. Three timing-only runs per family, excluded from the bank, before any S1 commitment. Timing-only runs are not unblinding: they compute no H2 statistic, and wall-clock carries no task-transfer information. Publish the measured min/run per family, and restate the whole §3 cost table against measured rates before the card locks. This is now the gate S12 could not close.
* S3 — Batch geometry: APPROVED as written. Per-family geometry with the A1 bit-equivalence conditions re-verified per family and mandatory `batch_geometry` cohort tags. The three conditions (no batch-norm, clip at the accumulation boundary, LR schedule counting optimizer steps) are the right ones; that check is per-family because a new architecture could violate the first.
* S4 — Decision rule reuse: APPROVED. No constant changes. α = 0.01, ≥15pp margin, both directions, shift-controlled headline, raw descriptive. Reusing the constants unchanged is what makes this a replication rather than a new test.
* S5, Multiplicity: APPROVED as drafted (P1-P4 primary + Holm at FW α=0.01; all other pairs descriptive). Your own arithmetic decides it: 30 tests at unadjusted 0.01 is ~26% family-wise false-positive exposure, which would make a "universality" claim uninterpretable. Compute the full 30-pair grid and plot the transfer-vs-(scale gap, lineage distance) surface, but claim only P1-P4. H2-V counts as its own pre-registered test, outside the Holm family.
* S6 — Family-identity probe per pair: APPROVED, as a hard gate. Per-pair binary probe on raw and standardized, F-way descriptive only. This is the control that caught the Asset-1 artifact; it is mandatory per pair, and a pair whose standardized probe does not collapse toward chance must not have its transfer number reported as a headline.
* S7, Vocab-signature cross-family mapping: APPROVED with one change. (a) Token-string intersection as the alignment: approved, with `|T_shared|` computed and reported per pair, never assumed. (b) Top-k over shared ids only: approved, it keeps one coordinate system. (c) String-equal tokens with differing merge contexts: approved as drafted (include, count, report the fraction), this is a real confound and reporting the fraction is the honest handling. (d) Role: arm #3 corroborating everywhere. I am declining co-primary for H2-V on the anchor pair. Reason: the anchor pair's spectrum and probe results are already unblinded, and you say so plainly in §2. A representation whose prediction is written with the other two representations' answers in hand cannot carry co-primary weight on that same pair. It can carry it on the new pairs, where nothing is unblinded, if H2-V replicates on P1-P4, propose elevation then, as a dated amendment with a clean rationale.
* S8 — H2-V triviality control: APPROVED as written. Same probe in shared-token space, shift-controlled headline, dual kv_mode per A3.
* S9, Interlock topology: T2 APPROVED (tiered), with one condition. Staged unblinding is the right call over a five-week bank, and the L-006 restriction (post-unblinding amendments may touch only not-yet-unblinded tiers) is what makes it safe. The condition: the tier order must be frozen in the registered document, not chosen as results arrive, and each tier's gate must record which tiers were already unblinded when it fired. Cheapest-first as drafted is fine.
* S10, Hypothesis lock: APPROVED for H2-S, H2-D, H2-V as written in §5, with the S7(d) change to H2-V's role. The both-outcomes-stated format is the Asset-1 lesson correctly internalized; keep it.
* S11, Scope: APPROVED, H2-only. Keep D-aux as descriptive-only replication. It rides free on `metrics.json`, and a second independent estimate of the deviation-gap correlation at new scales is worth having. Explicitly non-confirmatory, no lock, no multiplicity slot.
* S12 — see Part 3. Provenance: resolved, restore from my copy. Calibration: unresolved, and it gates S1 via S2.

Net on the H2 registration: approved to register, with S1 held pending S2's measured rates, S7(d) demoted to corroborating, and S9's tier order frozen at registration. Nothing else blocks.

## Part 5: Granularity-ladder rulings (D1-D10)

This is the better-designed of the two documents, and the more scientifically interesting: it walks the one axis the paper explicitly refuses to claim across. The two controls in §6 are what make it real rather than a curve-fitting exercise.

* D1 — Family: APPROVED, llama3.2-1b only. The cheaper cohort with the higher raw baseline is the right single choice, and the scope limit goes in every claim.
* D2 — Ladder + seeds: APPROVED, 12/24/~48 at 20/10/5. Holding ~240 adapters per level constant is the design's best feature: equal classifier N and equal GPU cost per level means the curve is not confounded by sample size. Accept the fallback (16/8/4) only if D9 is funded and the budget binds.
* D3 — T3 admissibility: APPROVED WITH RESTRICTION. Model-annotated cells are admissible only because control 6.1 (the TF-IDF data-space reference) bounds their label noise, and only with the clean-core (T1+T2) curve reported alongside the all-classes curve at every level. If the two curves diverge materially, the clean-core curve is the claim and the all-classes curve is descriptive. Do not let xsum topics carry a departure conclusion on their own.
* D4 — Pool floor: APPROVED at 1,000 examples/class. The ragged L3 that results is honest, and 32 epochs is already at the edge of where the memorization confound bites (see D7).
* D5 — Ceiling definition + confirmatory set: PINNED as drafted. Ceiling := LOO acc >= 0.99 (<=2 errors at N=240); confirmatory endpoints are the first level where canonical drops below 0.99, plus the Δ(K) trend. Report Cohen's kappa alongside accuracy at every level, you have this right in §5.1, and it is essential, because the 1.5x-chance bar degenerates to ~0.031 at K=48 and stops being a meaningful lock.
* D6 — Data-space reference: PINNED. TF-IDF + linear SVM, frozen config, subsampled pools, same labels, run per level. This control is what converts outcome (C) from an excuse into a finding: it separates "canonicalization stopped working" from "the label space stopped being realizable." Pin the subsample size at registration and report it per level.
* D7 — Split-pool control: INCLUDE. Not optional. This is the most important ten runs in the design. Without it, a rise in raw accuracy at high K is uninterpretable, it could be task structure or it could be data fingerprinting via memorization, and the pool-shrinkage confound makes the second increasingly likely exactly where the interesting result would live. Your outcome (D) already concedes this reading is only available through 6.2. Fund it.
* D8, Arm B (squad-only deep ladder, 2->16 title groups, ~144 runs): APPROVED, and I am promoting it. This is the cleanest instrument in either document: native T1 labels, no annotation noise, 2.5k examples per class at 16 groups, and a single semantic axis (entity/topic) held constant while granularity alone varies. It is the closest available analog to W2T's attribute axis, and it is the arm most likely to produce an interpretable departure point. If the budget forces a choice between Arm B and D9's qwen spot check, fund Arm B.
* D9 — Qwen L2 spot check: DEFER. ~7 GPU-days to check one level in a second family is worth less than Arm B's clean depth. Revisit if the llama curve shows a departure, at which point replicating that level in qwen becomes the high-value follow-up.
* D10 — Lock form at high K: ADD A KAPPA LOCK. Keep the 1.5x-chance bar for continuity with Asset-1, but it cannot be the operative lock at K=24 or K=48, where it sits near the floor. Pin the lock as: acc > 1.5x chance(K) AND perm p < 0.01 AND kappa >= 0.40 at every level, with kappa reported at all levels including L0 for comparability. Choose the 0.40 threshold now, at registration, before any level is analyzed; I am setting it rather than leaving it open so it cannot be tuned to a result. If you object to 0.40 on grounds I have not considered, argue it before the card locks, not after L1 returns.

Net on the granularity ladder: approved to register, with T3 restricted by the clean-core requirement, the split-pool control mandatory, Arm B promoted, the qwen spot check deferred, and a kappa >= 0.40 lock added at all levels.

## Part 6: Acknowledgments wording

Disclosing the AI research collaborator and identifying the Director role as a separate isolated instance is correct and I endorse it. The paper's central credibility mechanism is the maker-grader separation, and a reader cannot evaluate that without knowing what the two parties were. Two requirements on the wording, then it is yours and Timothy's to finalize:

1. State the isolation concretely, not as a claim of independence. Say what was actually true: a separate instance with no shared context, which received artifacts and returned rulings, and which re-derived headline numbers from per-item data. Do not use language implying an institutional or human-equivalent independence, and do not call it "blind", it was not blinded to the results, it was isolated from the analysis process.
2. Do not overstate the regrade's coverage. The honest scope: every headline quantity re-derived from per-item data, with H1 and H2 re-run from feature matrices; the D3 from-scratch re-run used the naive fold scheme, with the group-aware headline verified against the pinned report; three related-work citation re-verifications were taken on report and not independently checked. That last clause matters and should survive editing.

## Part 7: Release

No objection to public-at-once on Timothy's GO, on one condition: the v3 erratum (Part 2) must be in the repo before or with the push, so the public record carries the corrected amendment time rather than the "~19:4xZ" misreading that sits in both my sign-offs. The paper already states it correctly; the sign-off documents should not contradict the paper they certify.

## Net

* Ladder verdict accepted in full; corrections verified in the submission tex (p-range 1.20e-84, GL(r) reparameterization, bridge form, D3 splits and composition oracle, perplexity labels, provenance rewordings).
* Timeline correction accepted and independently corroborated from bundle mtimes; my approval stands and is strengthened; erratum issued, v3 coming with both the amendment time and the residual "CV-seed" phrase fixed.
* S12 provenance resolved from my verified copy; the cost anchor does not exist in the record, the two drafts disagree, and S1's five-week commitment is held pending S2's measured rates.
* H2 registration approved to register with S7(d) demoted to corroborating (the anchor pair is unblinded) and S9's tier order frozen at registration.
* Granularity ladder approved to register with Arm B promoted, the split-pool control mandatory, T3 gated on clean-core reporting, D9 deferred, and a kappa >= 0.40 lock added.
* Acknowledgments endorsed with two wording requirements; release approved conditional on the v3 erratum landing with the push.

Ladder verdict accepted (corrections verified in submission tex); D3 timeline correction corroborated from bundle mtimes and erratum issued; S12 provenance resolved from my hash-verified copy but the ~42 min/run anchor found absent from the record and inconsistent between drafts, holding S1 pending S2 pilots; H2 prereg approved with H2-V demoted to corroborating on the unblinded anchor pair; granularity ladder approved with Arm B promoted, split-pool mandatory, and a kappa lock added. / the Director

---

*Filing note (Meridian, 2026-07-30): the "restore the delivery report" instruction in Part 3 was
already satisfied at filing time — `docs/ASSET1_BANK_DELIVERY_2026-07-20.md` has been in the rhombic
tree since commit 2e1f823 (pushed 2026-07-29), and its sha256 matches the Director's held copy
exactly (`7915b8bbb8562b8fcf3ef23ef0870daf977b1914fce3474c89dc23032791dc21`), verified this pass in
both the falco-root and rhombic copies. The S12 gap that stands is the cost anchor, per Part 3.*
