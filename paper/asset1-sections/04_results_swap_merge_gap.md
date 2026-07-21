# Results II: Swap, Merge, and Gap

<!-- Section covers outline §7 (D2), §8 (D3), §9 (D-aux). Every number is
copied from ASSET1_BANK_DELIVERY_2026-07-20.md [DELIVERY], the result JSONs
in results/asset1-delivery-verify/, DIRECTOR_SIGNOFF_ASSET1_2026-07-21.md
[SIGNOFF], D3_PAIR_DESIGN_PREDECLARATION_2026-07-20.md [PREDECL],
DIRECTOR_DECISIONS_2026-07-06.md [DECISIONS], and
ASSET1_ANALYSIS_PIPELINE.md [PIPELINE]. -->

## D2: the identity backbone is the sole load-bearing structure

**Design.** D2 asks what a trained bridge is worth at inference time. For
each family we assembled adapters whose bridge had been replaced — by another
seed's bridge from the same task (`cross_seed`), by a different task's bridge
(`cross_task`), by the magnitude or topology component of a different task's
bridge (`cross_task_magnitude`, `cross_task_topology`; the decomposition
cells run unconditionally by Director override, not gated on any
data-dependent trigger), by the identity bridge (`identity`; measures the
total contribution of bridge training), or by a permutation of the
recipient's own bridge — and evaluated each assembly's val-loss on the
recipient task's fixed 500-example split against the native adapter. The two
permutation cells carry the structural contrast. The pinned H3 structure
reference is `permuted_deviation`: permute the trained deviation while
keeping the identity backbone, i.e. I + permute(B − I), which preserves both
the backbone and the trained-deviation multiset. The full-entry `permuted`
cell deranges all 36 entry positions, destroying the backbone. Both cells
share the same derangement per (family, task, slot), so their difference
isolates the backbone effect and nothing else. K = 3 donor/recipient pairs
per cell gives 360 evaluations per family, every one re-assembled from the
bank and SHA-verified against the plan before evaluation.

**Result.** Table 5 gives the mean val-loss penalty (evaluated minus native)
by swap kind.

**Table 5 — D2 penalty matrix (mean val-loss penalty vs native, nats;
360 evals/family).**

| swap kind | qwen2.5-1.5b | llama3.2-1b |
|---|---|---|
| cross_seed | +0.0000 | +0.0002 |
| cross_task | +0.0000 | +0.0003 |
| cross_task_magnitude | +0.0000 | +0.0000 |
| cross_task_topology | +0.0000 | +0.0002 |
| identity | +0.0000 | +0.0007 |
| permuted_deviation (H3 reference) | +0.0000 | +0.0007 |
| **permuted (full)** | **+2.8086** | **+3.8365** |

Installing a different task's trained bridge into an adapter costs
essentially nothing. So does deleting the trained bridge entirely
(`identity`), and so does scrambling the trained deviation while preserving
the backbone (`permuted_deviation`). Only the full-entry permutation — the
one operation that destroys the identity backbone — costs anything, and it
costs 2.8 to 3.8 nats. The contrast between the two permutation cells, which
differ only in whether the backbone survives, isolates the backbone as the
sole load-bearing structure for in-distribution loss. This is consistent
with the bank's controller-free, identity-init design, and it sharpens the
reading of D-aux below: the trained bridge deviation is real and measurable,
but it is not what the loss depends on.

The Director's regrade reproduced all 14 per-kind means exactly from the 360
per-eval rows per family against the native reference.

**Cost.** D2 is the pipeline's first GPU commitment: 360 val-loss
evaluations per family, each requiring assembly, SHA verification, and a
full pass over the recipient's 500-example split. The
decomposition cells alone add 180 evaluations per family — the price of
running the guard unconditionally rather than gating it on whether a
contradiction materialized.

## D3: post-hoc, weight-only merge-conflict prediction

**Scope.** arXiv:2606.19549 owns the training-time form of merge-conflict
prediction. D3 is scoped exactly as the experiment card scoped it: the
weight-only, post-hoc regime — two finished adapters, no training access, no
activations, no data. The question is whether the adapters' weights alone
predict what happens when they are merged. We claim only this regime.

**Design and dated amendment.** N = 120 vertex-disjoint pairs per family
(each run used at most once — the dyadic-dependence-safe design), α = 0.5
midpoint merges. The pair design was pre-declared on 2026-07-20 at ~16:50Z,
before Step 5 ran and with zero pairs, merges, or labels in existence: 120
pairs per family stratified over all 21 unordered task-pair cells, same-task
cells serving as the cross-seed reference. At ~19:4xZ the same day — still
before Step 5, still with zero labels in existence — the declaration was
amended: pre-execution inspection of the frozen tool showed the approved
sampler is uniform over same-family run pairs and has no per-cell
stratification mode. Building one would have meant modifying frozen,
adversarially-reviewed analysis code after the bank existed; amending the
declaration was the lesser deviation. The amended design is uniform sampling
without replacement (seed 0, vertex-disjoint), with the realized
(task_i, task_j) cell coverage reported descriptively (Figure F5; expected
mix under uniformity is ~16% same-task, ~20/120 pairs; realized coverage:
[NUMBER: d3_pairs.json — realized cell counts and same-task fraction, per
family]). The Director's ruling: temporal integrity holds, the amendment is
the conservative choice, "clean dated amendment (L-006 / R10); no objection."

**Labels.** The primary label rule was pinned by the Director before the
bank completed: a pair is a conflict if the merge degrades either endpoint
task by ≥ 5% relative to that endpoint's native adapter, with a degenerate
floor of 10% triggering fallback to the pre-declared median split. The rule
held: the conflict rate is 85.8% (frac_positive 0.8583, identical at both
endpoints), above the floor, so the primary rule stands and no fallback was
used. A conflict rate this high is itself a finding about midpoint merging
in this bank: at α = 0.5, most pairs degrade at least one endpoint by 5%.
Native losses were taken from the trainer's recorded finals after verifying
0.00000% divergence against 36 fresh D2-harness evaluations.

**Features and model.** The `distance` baseline is two features:
`[cos_distance, l2_distance]` between flattened adapters. The `full` set is
those two, plus four gauge-invariant principal-angle aggregates
(`angle_mean_weighted`, `angle_mean_unweighted`, `chordal_rms_weighted`,
`chordal_rms_unweighted`), plus the per-module vectors `module_l2` and
`module_angle_mean` (length 112 for qwen, 64 for llama; NaN→0.0 in
`module_angle_mean`). `module_chordal_rms` and `module_weight` are carried
in the pair records but are not in the `full` matrix. The classifier is
logistic regression under 5-fold CV with fold seed 0 and 1000 bootstrap
resamples — all three pinned in the report JSON.

**Result.** Table 6 gives the headline group-aware AUCs.

**Table 6 — D3 merge-conflict prediction, group-aware CV AUC with 95%
bootstrap CIs (120 pairs/family; pooled OOF row descriptive).**

| family | AUC full (weight-only) | AUC distance-only | full − distance |
|---|---|---|---|
| qwen2.5-1.5b | **0.995** [0.983, 1.000] | 0.675 [0.484, 0.848] | +0.320 [0.150, 0.511] |
| llama3.2-1b | **0.962** [0.898, 0.999] | 0.713 [0.458, 0.923] | +0.249 [0.039, 0.490] |
| pooled OOF (descriptive) | 0.9890 [0.9730, 0.9992] | 0.7082 | +0.2808 [0.1565, 0.4205] |

Two adapters' weights alone predict whether their midpoint merge degrades an
endpoint, at AUC 0.96–0.99, and the gauge-invariant block adds a margin over
raw distance whose CI excludes zero in both families. The headline is
group-aware CV (StratifiedGroupKFold over run-overlap components with a
component-cluster bootstrap); because the vertex-disjoint design produced
120 single-pair components per family, there is no dyadic dependence to
correct, and group-aware and naive numbers agree to within CV-fold reshuffle
noise (llama full 0.962 group-aware vs 0.952 naive; qwen 0.995 under both).

**The distance-only baseline is CV-seed-sensitive.** The fold seed is pinned
(seed = 0, n_splits = 5), and it matters for the baseline: the Director's
independent re-run of the 2-feature distance-only model gave 0.686 (qwen) /
0.667 (llama) against the reported 0.675 / 0.713 — a few points of CV-seed
sensitivity on a 2-feature model over 120 points. The full-model result and
the existence of the margin are not in question, but the lower end of the
margin-over-distance CI depends on this baseline, which is why the seed is
pinned and the sensitivity reported rather than averaged away.

**Interpretive caveat.** Invertible bridges are a gauge on the update column
space, so the principal-angle features are provably insensitive to bridge
numerics; whatever bridge information reaches the classifier arrives through
the magnitude weights and the raw distances. The margin over distance is
therefore evidence about update geometry, not about the bridge — consistent
with D2's finding that the bridge carries no in-distribution loss structure.

**Cost.** D3's labels required GPU evaluation of 240 merged adapters on both
endpoint tasks' val splits, on top of the CPU feature extraction. The labels
runner had no shipped implementation (the card scoped it "external"); it was
written post-card and put through a fresh-context adversarial verification
before its GPU run, which caught two blocking defects (a flat-vs-nested
merge-loading bug and a machine-absolute manifest path) — both fixed and
dry-run-verified before launch.

## D-aux: the pilot correlation shrinks honestly

**Design.** D-aux re-verifies the pilot's headline association — bridge
deviation predicts generalization gap — at bank scale, as the Director's
pinned condition for including it at all ("Re-verify r = 0.888 … on the
bank"). The primary pre-registered pair is `dev_mean` (mean over modules of
‖bridge_final − I‖_F) against `final_gap` (val minus train at step 2000),
n = 480. The step-0 identity control comes out 0.0 exactly, as it must by
construction. Within-task cells are the pre-registered Simpson's-paradox
guard: descriptive, but shown, not hidden.

**Result.** Table 7 gives the pooled claim and the guard cells.

**Table 7 — D-aux bridge-deviation ↔ generalization-gap correlation
(Pearson r, 95% bootstrap CIs).**

| cell | n | Pearson r | 95% CI |
|---|---|---|---|
| pooled | 480 | 0.300 | [0.175, 0.415] |
| qwen2.5-1.5b | 240 | 0.418 | [0.323, 0.522] |
| llama3.2-1b | 240 | 0.337 | [0.201, 0.466] |
| within math (llama) | 40 | 0.549 | [0.324, 0.736] |
| within math (qwen) | 40 | 0.336 | [0.030, 0.652] |
| within xsum (qwen) | 40 | −0.301 | [−0.502, −0.066] |

The pilot's r = 0.888 does not survive the bank, and we report that as the
result. Pooled over 480 runs the association is real, positive, and non-zero
— r = 0.300 [0.175, 0.415] — but modest, and inflated in the pilot by
small-n and task mixture. Within task it is heterogeneous: math is positive
in both families (0.549 llama, 0.336 qwen), while xsum in qwen is
significantly negative (−0.301 [−0.502, −0.066]). The pooled bank-level
correlation remains the claim; the cells bound what it can mean. A deviation
measure whose within-task sign flips between cells is not a universal
overfitting signal — it is a between-task association with real but
task-dependent within-task structure.

The 0.888 → 0.300 shrink is a success of the protocol, not a failure of the
result. The pilot number was computed on a small mixed sample; the
pre-registered re-verification on 480 runs is exactly the check that
separates a real-but-modest association from an artifact of the sample that
first suggested it. The Director's regrade reproduced the pooled and
per-family values from the 480 (dev_mean, final_gap) pairs.

**Reading D2 and D-aux together.** D-aux shows the trained bridge deviation
carries real signal about training dynamics — it correlates with the
generalization gap. D2 shows that same deviation is worth ~0 nats to
in-distribution loss, and that only the identity backbone is load-bearing.
Both are true, and their conjunction is the finding: the bridge records
where the run has been without determining where the loss sits. D-aux is
CPU-only and rides along at negligible cost; its price was paid once, in the
pilot's discipline of flagging its own n.
