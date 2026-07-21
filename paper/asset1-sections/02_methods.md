# 2 Experimental design and methods

## 2.1 The adapter bank

The artifact under study is a bank of 480 LoRA adapters: 2 model families
× 6 tasks × 40 seeds. The families are Qwen2.5-1.5B-Instruct and
Llama-3.2-1B-Instruct; the tasks are alpaca, code, math, xsum, squad, and
agnews. Every run trains a rank-24 adapter with an identity-initialized
bridge, using plain language-modeling loss on all tokens: sequences are
padded to a fixed length of 512 with `labels = input_ids.clone()` and no
−100 masking, so every sequence contributes 511 equally weighted shifted
token losses. Each run takes 2000 optimizer steps at effective batch 16
(micro-batch 4 × gradient accumulation 4), consuming 32,000 sequences from
a 40,000-example pool in a single epoch with no reshuffle. Training seeds
derive from `seed_base = 10000` and `data_seed_base = 20000` by run index.
Validation is a fixed 500-example split per task, cut with the locked
`val_seed = 777` — the same split for every analysis that touches val loss.
Adapter module counts are discovered from each adapter rather than
hardcoded: 112 modules per run for Qwen, 64 for Llama.

The batch geometry deserves one paragraph, because it changed mid-campaign
under a dated amendment (A1, Director-adopted 2026-07-06). The original
geometry was micro-batch 2 × accumulation 8; the amendment moved to 4 × 4
with the effective batch unchanged, after verifying the three conditions
for bit-equivalence up to float summation order: the models use RMSNorm
(no batch normalization) and the adapters are linear; the gradient clip
fires once per optimizer step on the accumulated gradient; and the LR
schedule counts optimizer steps, so both geometries see identical
sequences over identical steps. All 480 runs were re-executed at 4 × 4 for
provenance uniformity, not correctness, and every config records its batch
geometry as a cohort tag so cohorts are never silently mixed.

The campaign ran on a single local GPU from July 3 21:52Z to July 20
16:21Z — roughly 17 days including the A1 restart, at about 1.9× the
original throughput. The manifest closed at 480/480 COMPLETE on
2026-07-20T16:21:19Z. The outage ledger contains exactly four failures,
all HF Hub 504 errors on July 16 (run indices 329–332), all retried to
COMPLETE. One run-log item from the analysis phase belongs in the open:
the first D1 launch died on a default-HF-cache environment trap and was
relaunched clean — an environment-level failure that unblinded nothing.
Chance accuracy for the 6-task classification problems throughout is
0.1667.

## 2.2 Pre-registration machinery

The analyses were pre-registered in three layers, each with a paper trail.

**The locked experiment card.** The hypotheses (H1, H2, H3, the D3
prediction target, the D-aux re-verification) were locked in the
experiment card of 2026-07-03, before the bank existed. The card's
standing instruction is that hypotheses are locked and the deliverable
fills result tables only.

**Director-pinned decisions.** Every analytical choice the card left open
was pinned by an independent Director on 2026-07-06 — before the bank
completed — and baked into the analysis tools, which record each pinned
choice in their output JSON so runs self-document. The pins that carry the
results in this paper: the H2 decision rule (one-sided exact-binomial
α = 0.01 and a ≥ 15 percentage-point within-minus-cross margin, required
in **both** transfer directions, evaluated on the shift-controlled
representation with raw reported as descriptive); the D2 structure
reference (`permuted_deviation` is the primary H3 reference, with the
full-entry permutation retained as the identity-backbone contrast) and the
ruling that the magnitude/topology decomposition cells run
unconditionally, because gating them on whether a contradiction
materializes would be a post-hoc forking point; and the D3 label rule
(a fixed 5% relative-degradation threshold per endpoint, with a degenerate
floor at 10% class balance below which the analysis falls back to a
pre-declared median split and reports the fallback as a finding). A later
round (2026-07-07/08) added the vocab-signature D1 arm as amendment A3 and
closed with the Director independently verifying the condition encodings.

**The completeness interlock.** The entire analysis layer was built and
validated against synthetic fixtures only, before the bank finished, so
that analysis could fire on completion day with zero unblinded development
in between. Every CLI that can touch the real bank calls a single gate,
`require_complete_bank()`, before reading a single adapter; the gate
passes only when the manifest lists exactly 480 runs, every one COMPLETE.
The manifest, not the filesystem, is the source of truth, so half-written
run directories are invisible to analysis until the campaign runner marks
them done. A `--allow-partial-bank` escape exists for smoke-testing the
tooling against the real tree; it prints a multi-line pre-registration
warning and marks output `exploratory_only` — nothing produced under it
may be reported. The tooling was re-verified on completion day: 171/171
tests plus the D1/D3/D-aux selftests passed on 2026-07-20, and the
interlock HELD — no real-bank statistic was computed before 480/480.

**Dated amendments, never silent revisions.** Deviations from a
declaration are handled by amending the declaration with a date and a
reason, not by silently revising code or claims. Section 2.3.4 reports the
one amendment this campaign required.

## 2.3 Analysis pipeline

The five locked analyses ran against the real bank exactly once, in a
fixed fire sequence: the CPU-only analyses first (D1, D-aux, D2 assembly),
then the first GPU commitment (D2 evaluation), then D3, whose labels
cannot exist until its own merges are GPU-evaluated.

### 2.3.1 D1 — representations and task identifiability (H1, H2)

H1 asks whether an adapter's weights reveal the task it was trained on,
within a family. The classifier is deliberately weak — a linear SVM
(C = 1.0) under leave-one-out cross-validation with a 1,000-shuffle
permutation null — so that the representation, not model capacity, carries
the result. The pass lock is accuracy above 1.5× chance (0.2500) **and**
permutation p < 0.01; the lock is designed to reject weak-but-nonzero
signal that clears the null without clearing the bar. Wilson intervals are
reported with the standing caveat that LOO folds are not independent; the
permutation p is the calibrated inference. A variance-heterogeneity guard
(Euclidean distances in the exact feature space the classifier saw,
trigger at the pilot's 3.7× ratio) checks that a ceiling result is not a
variance artifact.

Four representations are computed per family. **Raw** flattens the adapter
weights in a deterministic module order (6,541,248 dimensions for Qwen,
5,114,112 for Llama). **Canonical** applies GL(r) gauge canonicalization
in the QR→SVD lineage, absorbing the bridge into B′/A′; the implementation
was verified GL(r)-invariant to ~1e-13, and the feature vector uses the
`'full'` variant with `proj_dim = 16`, `proj_seed = 0` (88,704 dimensions
for Qwen, 50,688 for Llama). **Vocab-signature** (amendment A3, arm #3) is
an output-referenced representation computed through the base model's
unembedding, reported in both kv modes: including k/v projections (15,232
/ 8,704 dimensions) and excluding them (7,616 / 4,352).

H2 asks whether task structure transfers across the two families. Because
the families have different hidden dimensions, H2 uses two
dimension-agnostic representations: depth-binned singular-value spectra of
the effective update (**spectrum**, the pinned primary; 384 dimensions;
4 depth bins, 24 singular-value slots per bucket, buckets keyed by
projection type q/k/v/o and depth bin, mean aggregation, empty buckets
zero, depth fraction (L + 0.5)/n_layers) and a canonicalized
probe-projection (**probe**, corroborating; 12,672 dimensions).
Disagreement between the two is itself reportable. The pre-registered
prediction was that transfer FAILS, under the pinned decision rule of
§2.2.

The triviality control is the part of H2 that mattered most, and it was
added in round-1 review, before any real data existed. A cross-family
"transfer at chance" result is trivially produced by covariate shift if
the representations encode which family they came from; the control
therefore (i) trains a family-identity probe on each representation, and
(ii) computes a shift-controlled variant via `familywise_standardize` —
per-family z-scoring, an unsupervised operation that never touches task
labels and so has no capacity to manufacture task structure. The pinned
protocol makes the shift-controlled variant the headline and raw
descriptive, and the decision rule runs on the shift-controlled numbers.

### 2.3.2 D2 — bridge-swap assembly and evaluation (H3)

D2 measures what the trained bridge is worth by transplanting bridges
between adapters and measuring the val-loss consequence. Stage A (CPU)
plans and assembles the swapped states; Stage B (GPU) evaluates each
assembled adapter on the recipient task's fixed 500-example split. The
plan uses K = 3 donor/recipient pairs per cell, with the same K recipients
serving every cell of a task row (amortized natives, within-row
comparability), for 360 evaluations per family with the decomposition
cells included.

Seven kinds are evaluated: cross-seed and cross-task bridge swaps; the
magnitude/topology decomposition of the cross-task swap (two-sided
row/column-norm factorization D = diag(r)^½ P diag(c)^½, an exact
round-trip, separating how much of a donor bridge's effect is scale versus
wiring); an identity-bridge cell (measuring the total contribution of
bridge training, justified by the identity init); `permuted_deviation`
(I + permute(B − I)) — the pinned H3 structure reference, which scrambles
the trained deviation while preserving the identity backbone and the
deviation multiset; and the full-entry permutation (one shared derangement
over all 36 entry positions, diagonal included, applied to all modules),
which destroys the backbone and serves as the identity-backbone contrast.
The two permutation kinds share the same derangement per (family, task,
slot), so their contrast isolates the backbone effect.

Integrity is mechanical: the Stage-A plan records an `assembled_sha256`
for every evaluation, and Stage B re-assembles each state from the bank
and verifies the recorded digest before installing it — a hard error if
the bank changed between planning and evaluation. All 360 evaluations per
family passed the SHA check. Native reference losses come from the same
harness on the same fixed splits.

### 2.3.3 D3 — weight-only merge-conflict prediction

D3 asks whether two adapters' weights alone — post-hoc, with no access to
training — predict whether their merge degrades. For each family, 120
pairs of runs are drawn (the design is §2.3.4), each pair is merged at
α = 0.5 (midpoint), and each merged adapter is GPU-evaluated on both
endpoint tasks' fixed val splits. The primary label rule, pinned in §2.2:
a pair is a "conflict" if the merge degrades **either** endpoint by ≥ 5%
relative to that endpoint's native adapter, with the < 10% degenerate-
balance floor triggering a reported fallback to median split; the rule
actually used is recorded in the report's binarization block.

The feature sets are fixed, and because the margin between them is a
headline, we state them exactly. `distance` = `[cos_distance, l2_distance]` —
two scalars. `full` = those 2, plus 4 gauge-invariant principal-angle
aggregates `[angle_mean_weighted, angle_mean_unweighted,
chordal_rms_weighted, chordal_rms_unweighted]`, plus the per-module vector
`module_l2` (length 112 for Qwen, 64 for Llama), plus the per-module
vector `module_angle_mean` (same length, NaN→0.0). `module_chordal_rms`
and `module_weight` are carried in the pair records but are **not** in the
`full` matrix. The principal-angle features are provably insensitive to
bridge numerics (an invertible bridge is a gauge on the update column
space), so bridge information reaches the classifier only through the
magnitude weights and raw distances — a caveat that binds any
interpretation of which block drives the gain.

The classifier is logistic regression under 5-fold cross-validation with
the fold seed pinned at `seed = 0` and `n_boot = 1000` bootstrap
replicates for the confidence intervals. The pinned seed is a Director
write-up requirement, because the distance-only baseline is CV-seed-
sensitive: the Director's independent re-run of the 2-feature baseline
gave AUC 0.686/0.667 (Qwen/Llama) against the reported 0.675/0.713 — a
few points of CV-seed sensitivity on a 2-feature model over 120 points.
The full-model result and the existence of the margin are not in
question, but the lower end of the margin-over-distance CI depends on the
baseline, so the seed is pinned and the sensitivity reported. The
headline numbers are group-aware: StratifiedGroupKFold over run-overlap
connected components with a component-cluster bootstrap, guarding against
dyadic dependence between pairs that share an endpoint run; naive
pair-level CV is reported as an explicitly anti-conservative secondary
block. Under the vertex-disjoint design each run appears in exactly one
pair, so the pair graph has 120 single-pair components and group-aware and
naive numbers agree to within fold-reshuffle noise.

This analysis is scoped precisely against arXiv:2606.19549, which owns the
training-time form of merge-conflict prediction. D3 is the weight-only,
post-hoc regime — no training access — and claims only that regime.

### 2.3.4 The D3 pair design and its dated amendment

The card left the pair count and stratification open, with the requirement
that they be declared before labels exist. The pre-declaration was filed
2026-07-20 at ~16:50Z, while D2 Stage-B evaluations were in progress and
zero D3 pairs, merges, or labels existed: N = 120 vertex-disjoint pairs
per family (the maximum over 240 runs/family; `max_run_uses = 1`, the
dyadic-dependence-safe design), α = 0.5, and stratification over all 21
unordered task-pair cells (15 cross-task plus 6 same-task cells as the
cross-seed reference, ~5–6 pairs per cell).

At ~19:4xZ the same day — still before Step 5 ran, still with zero pairs,
merges, or labels in existence — the declaration was amended.
Pre-execution inspection of the frozen tool showed the approved sampler is
uniform without replacement over same-family run pairs; it has no
per-cell stratification mode. Building one would have meant modifying
frozen, adversarially reviewed analysis code after the bank existed, so
the declaration was amended instead, as the lesser deviation: uniform
sampling, seed 0, vertex-disjoint, with the realized (task_i, task_j)
cell coverage reported descriptively (expected mix under uniformity:
~16% same-task, ~20 of 120 pairs). All other declared values were
unchanged. The Director's ruling on review: temporal integrity holds, the
amendment is the conservative choice, "clean dated amendment (L-006 /
R10); no objection."

### 2.3.5 D-aux — bridge deviation versus generalization gap

D-aux re-verifies, at bank scale, a pilot correlation of r = 0.888 between
bridge deviation and generalization gap — with the re-verification itself
pinned in the Director decisions before the bank completed ("Re-verify
r = 0.888 … on the bank"). The primary pre-registered pair is `dev_mean`
(mean over modules of ‖bridge_final − I‖_F) against `final_gap` (val
minus train loss at the last metrics record, step 2000). A step-0
identity control must read exactly 0.0 by construction (the bridge is
identity-initialized); it did. Gap-trajectory AUC (trapezoidal,
effectively from step 100 since the step-0 gap is NaN by design) and an
update-magnitude covariate (mean ‖ΔW‖_F versus gap) are descriptive only.
Within-task stratified correlations are computed as a Simpson's-paradox
guard: the pooled bank-level correlation is the claim, and the per-family
and within-task cells are shown so that heterogeneity — including any
sign reversal — is visible rather than averaged away.

## 2.4 Verification protocol: maker–grader separation

Verification is part of the method, not an afterthought, and it operates
at three levels.

**Fresh-context adversarial verifiers on every custom component.** Any
analysis code written for this campaign was audited by a verifier agent
with no shared context with the author before it touched data. The case
that proves the value: Step 6 (D3 label generation) had no shipped runner
— the card scoped it "external" — so `asset1_d3_labels.py` was written
post-card. Because the entire D3 AUC rests on its labels, the runner went
through a fresh-context adversarial verification **before** its GPU run,
which caught two blocking defects: a flat-versus-nested merge-state
loading bug (reproduced to an actual crash) and a machine-absolute
manifest path. Both were fixed and dry-run-verified before launch. The
runner's native-loss shortcut — taking native reference losses from the
trainer's `metrics.json` finals rather than re-evaluating — was accepted
only after verifying 0.00000% divergence against 36 fresh D2-harness
evaluations.

**The interlock as verifier of the analysts.** The completeness gate
(§2.2) is itself a maker–grader device: the people and tools that built
the analyses could not, even accidentally, run them early, and the
manifest — written by the campaign runner, read by the analysis layer —
is the independent arbiter of "complete."

**Independent Director regrade from per-item data.** After delivery, the
Director regraded the entire packet from a per-item verification bundle
pinned at rhombic commit `638f4a8` (archive sha256 `c1891d50…`; all 16
files — 14 Tier-1 result files plus both Tier-2 feature matrices —
matching the SHA256SUMS manifest byte-for-byte). The regrade recomputed
every headline from per-item rows rather than summary tables, and for the
two highest-risk results went further: the H1 canonical LOO-SVM was re-run
from scratch on the 88,704/50,688-dimension feature matrices
(precomputed-Gram LOO, C = 1.0), and the full H2 pipeline was re-run
end-to-end from the spectrum and probe matrices, with the Director
applying the standardization independently and confirming in source that
`familywise_standardize` never touches task labels. D2's 14 per-kind means
were reproduced exactly from the 360 per-eval rows per family; the D3
conflict rate and full-model AUC and the D-aux correlations were
reproduced from per-pair and per-run records. Every headline reproduced;
the one note — the D3 distance-baseline seed sensitivity — is reported in
§2.3.3 and wherever the baseline appears. The delivery report itself
states that all numbers trace to result trees on disk and none is restated
from memory; the same rule governs this paper.
