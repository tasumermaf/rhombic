# Discussion

**The gauge, not the information, was the obstacle.** The five results share
one mechanism. In H1, raw adapter weights fail the task-identity lock (LOO
accuracy 0.0792 qwen / 0.1375 llama against the 1.5×-chance bar of 0.2500,
chance 0.1667), while the GL(r)-canonical representation identifies all six
tasks at 1.0000 in both families — at a fraction of the dimensionality
(88,704 / 50,688 features against 6,541,248 / 5,114,112 raw). The task signal
was in the weights the whole time; raw's tiny permutation p (0.000999) with
sub-threshold accuracy shows a weak recoverable trace (code recall 0.40–0.45),
but the factorization gauge buries it below the lock. In H2, the same story
repeats one level up: raw cross-family transfer sat at chance, and the
triviality control revealed why — a linear probe reads family identity from
raw features at 1.0000, so raw "transfer failure" was family-scale covariate
shift, not absent structure. Unsupervised per-family standardization collapses
the family probe to 0.1521 (spectrum) / 0.0000 (probe projection) and transfer
rises to 0.7375–0.7833 (binomial p ≤ 1.20e-84). One reading covers both:
nuisance variation — the GL(r) factorization gauge within a family, the
scale gauge between families — dominates raw weight coordinates and obscures
task structure that is intact underneath. Task identity is gauge-obscured,
not absent. The scope of this claim runs on the label-granularity axis: our
six coarse tasks are the opposite regime from W2T's 10k+ fine-grained
attribute classes, and we make no claim about what a linear probe recovers,
raw or canonical, at that granularity.

**The backbone is load-bearing; the trained bridge deviation is real but
cheap.** D2 and D-aux look contradictory until they are read together. D-aux
shows the trained bridge deviation is a real, measurable quantity: it
correlates with the generalization gap at r = 0.300 [0.175, 0.415] pooled
over all 480 runs, with the step-0 identity control at 0.0 exactly. D2 shows
the same deviation is nearly worthless to in-distribution loss: installing a
different task's trained bridge costs +0.0000 (qwen) / +0.0003 (llama) nats
of val loss, and even permuting the trained deviation while keeping the
identity backbone (`permuted_deviation`, the pinned H3 reference) costs
+0.0000 / +0.0007. Only the full-entry permutation, which destroys the
identity backbone itself, costs anything: +2.8086 / +3.8365 nats. The
resolution is that the deviation carries information *about* the run — enough
for a weak correlate of its generalization gap — without carrying the
in-distribution task solution, which lives in the identity backbone the
controller-free, identity-init design builds in. The bridge is a
diagnostic-bearing structure, not a load-bearing one.

**Angles predict merge conflict, post-hoc, from weights alone.** D3 closes
the loop from description to prediction: two adapters' weights, with no
training access and no evaluation of the merge, predict whether their
α = 0.5 midpoint merge degrades either endpoint by ≥5% relative, at
group-aware AUC 0.995 [0.983, 1.000] (qwen) / 0.962 [0.898, 0.999] (llama).
The margin over a distance-only baseline (+0.320 [0.150, 0.511] /
+0.249 [0.039, 0.490], both CIs excluding 0) is what makes this a geometry
result rather than a magnitude result: the gauge-invariant principal-angle
block is what raw cosine and L2 distance lack. One interpretive caveat is
pinned in the pipeline: invertible bridges are a gauge on the update column
space, so principal-angle features are provably insensitive to bridge
numerics — bridge information reaches the classifier only through the
magnitude weights and raw distances. The scope boundary is equally pinned:
arXiv:2606.19549 owns the training-time form of merge-conflict prediction;
this result is weight-only and post-hoc, and we claim only that regime.

**What pre-registration bought.** Two of the five results went against us,
and those two are the credibility of the other three. The H2 prediction —
that task structure would not transfer across families — was refuted, and
refuted specifically by the triviality control added in round-1 review:
without the family-identity probe, raw transfer-at-chance would have read as
a confirmation, and it would have been false. A pre-registration that flips
via its own control is the protocol working as designed, not failing. The
D-aux pilot correlation shrank from r = 0.888 to r = 0.300 [0.175, 0.415] at
bank scale — small-n and task-mixture inflation, exposed by the pre-declared
Simpson's-guard cells, which also show the association is heterogeneous
within task (llama math 0.549 [0.324, 0.736]; qwen xsum −0.301
[−0.502, −0.066]). A pipeline that can only confirm is not measuring; this
one refuted its boldest prediction and deflated its own pilot, under an
interlock that made post-hoc rescue impossible.

**Run-log honesty.** Three operational facts belong in the record rather
than a drawer. D1's first launch died on a default-HF-cache environment trap;
it was environment-level, nothing was unblinded, and the relaunch was clean.
The D3 label generator had no shipped runner (the experiment card scoped it
"external"); the runner written for it was put through a fresh-context
adversarial verification *before* its GPU run, which caught two blocking
defects (a flat-vs-nested merge-state loader bug and a machine-absolute
manifest path), both fixed and dry-run-verified before launch. And the
native-loss shortcut — taking natives from the trainer's metrics rather than
re-evaluating — was verified at 0.00000% divergence against 36 fresh D2
harness evaluations before use. Maker–grader separation was applied exactly
where the pipeline was newest.

## Limitations

Cost first: the bank consumed roughly 17 days of a single local GPU
(2026-07-03 to 2026-07-20, including the A1 geometry restart), with four
HF-Hub-504 failures retried to completion. The analyses themselves are cheap
— D1, D-aux, and D3 feature extraction are CPU-only — but the artifact they
require is not.

**Two families at ~1–1.5B scale.** The bank spans Qwen2.5-1.5B-Instruct and
Llama-3.2-1B-Instruct: two lineages, but both small instruct-tuned decoders.
The H2 transfer result crosses these two families and no others; whether
canonicalization ceilings, backbone dominance, or angle-based merge
prediction hold at 7B+ scale or across architecture classes is unmeasured.

**Six coarse tasks.** The label-granularity axis cuts both ways. Our regime
— 6 coarse task labels, 40 seeds each — is where a linear probe on canonical
features reaches ceiling; we make no claim about fine-grained regimes such as
W2T's 10k+ attribute classes, in either direction. Ceiling accuracy on six
classes also means H1 cannot rank representations above the lock: canonical
and vocab-signature all sit at 1.0000, and separating them would need a
harder label space.

**In-distribution val loss only.** Every D2 penalty and every D3 label rests
on loss over fixed 500-example validation splits (val_seed 777) of the
training task's own distribution. The D2 conclusion is explicitly
in-distribution: the bridge deviation is negligible to *this* loss. Behavioral,
out-of-distribution, or instruction-following consequences of bridge swaps
and merges are unmeasured, and D-aux's r = 0.300 against the generalization
gap is a hint that the deviation matters for something val loss on the
training distribution does not fully capture.

**Single merge operator.** D3 predicts conflict for α = 0.5 midpoint merges
only. Other α values, TIES/DARE-style merge operators, and merges of more
than two adapters are outside the measured claim.

**Uniform pair sampling, by dated amendment.** The D3 pair design was
pre-declared (2026-07-20 ~16:50Z) with per-cell stratification over task
pairs, then amended (~19:4xZ, before Step 5 ran, with zero pairs, merges, or
labels in existence) to the uniform vertex-disjoint sampler the frozen tool
actually implements — the lesser deviation, against modifying frozen,
adversarially-reviewed analysis code after the bank existed. The Director
approved it as a clean dated amendment (L-006 / R10). The consequence is
uneven realized coverage of the 21 task-pair cells (expected ~16% same-task
under uniformity), reported descriptively; per-cell conflict-rate estimates
are correspondingly noisy, and the headline AUC is a claim about the uniform
pair population, not any single cell.

**D3 baseline seed-sensitivity.** The cross-validation configuration is
pinned: fold seed 0, 5 splits, logistic model, 1,000 bootstrap resamples.
The Director's independent re-run of the 2-feature distance-only baseline
gave 0.686 / 0.667 against the reported 0.675 / 0.713 — a few points of
CV-seed sensitivity on a 2-feature model over 120 points. The full-model
result and the existence of the margin are not in question, but the lower
end of the margin-over-distance CI depends on the baseline, which is why the
seed is pinned and the sensitivity is reported rather than averaged away.

**D-aux heterogeneity.** The pooled r = 0.300 conceals sign-varying
within-task structure: math is positive in both families (0.549 llama, 0.336
qwen) while xsum in qwen is significantly negative (−0.301 [−0.502, −0.066]).
The pooled bank-level claim is the pre-registered one; any within-task use of
the deviation–gap association would need task-specific calibration this bank
is only pilot-scale for (n = 40 per cell).

## Reproducibility

Every headline number in this paper is re-derivable from a per-item
verification bundle, and every one has already been re-derived by someone
who was not us. The bundle (`results/asset1-delivery-verify/`) carries two
tiers. Tier 1 is per-item result data: the per-family 6×6 confusion matrices
and permutation summaries behind every H1 accuracy; the full H2 transfer and
family-identity-probe tables; 360 per-eval D2 rows per family, each with its
assembly SHA-256 and val loss; 240 per-pair D3 feature dicts and labels with
per-endpoint merged and native perplexities; and 480 per-run D-aux
(dev_mean, final_gap) rows. Tier 2 is the reduced feature matrices
(`features_<family>.npz`, 90.6 MB qwen / 56.7 MB llama, float32), exported
by the frozen D1 feature functions so a from-scratch classifier re-run
bit-matches. Feature extraction and all Tier-1 outputs are deterministic —
fixed seeds, no wall-clock dependence — with recompute recipes per analysis
in the bundle README.

The bundle is anchored: rhombic commit `638f4a8`, archive SHA-256
`c1891d50…`, all 16 files matching the SHA256SUMS manifest byte-for-byte.
The independent Director regrade worked from exactly those bytes and
re-derived every headline: H1 accuracies reproduced from the confusion
matrices and the canonical LOO-SVM re-run from scratch on the feature
matrices (1.0000 both families); the entire H2 pipeline re-run end-to-end
from the spectrum and probe matrices, with the standardization applied
independently and `familywise_standardize` confirmed unsupervised in source;
all 14 D2 per-kind means reproduced exactly from per-eval rows; the D3
conflict rate (0.858) and full-model AUC reproduced from raw labels and
pairs; and the D-aux correlations reproduced from the 480 per-run pairs.
The one deviation found — the D3 distance-baseline CV-seed sensitivity —
is reported above as a limitation, per the Director's write-up requirement.

Upstream of the bundle, the analysis layer itself is the reproducibility
mechanism. A completeness interlock (`require_complete_bank`) refused every
real-bank statistic until the manifest showed 480/480 COMPLETE, so all five
analyses ran against the finished bank exactly once, with every open
analytical choice pinned by the Director before completion and recorded in
each tool's output JSON; deviations are dated amendments in the record, never
silent revisions. The tooling passed 171/171 tests plus the D1/D3/D-aux
synthetic selftests, re-verified on delivery day before any real-bank
command ran.

Release scope: the adapter bank and the reduced feature matrices are
currently local artifacts (the feature matrices are larger than the repository
carries and were transferred out-of-band for the regrade). The public release
surface for the bank, features, and verification bundle is a pending decision
and is not claimed here.
