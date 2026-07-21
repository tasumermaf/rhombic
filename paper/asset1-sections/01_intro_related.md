# 1. Introduction

A trained LoRA adapter is a small stack of low-rank matrices. Adapters now
accumulate by the thousands — shared on hubs, merged into products, composed
by people who never saw the training run. This paper asks what the weights
alone reveal, post hoc: given only the adapter tensors — no
training data, no logs, and no access to the training process — what can be
read out, and what does reading it out require?

The obstacle is not information but parameterization. A LoRA update
factorizes as a product of low-rank factors, and any invertible $r \times r$
matrix inserted between them leaves the effective update unchanged — the
GL(r) gauge. Two adapters trained on the same task can therefore sit far
apart in raw parameter space while encoding the same update, and two adapters
from different model families can sit apart for reasons that have nothing to
do with what they learned. Raw flattened weights are the unexamined default
of adapter analysis, and this paper measures what the default costs.

Our answer is an artifact and a discipline. The artifact is a bank of 480
LoRA adapters — 2 model families × 6 tasks × 40 seeds (Qwen2.5-1.5B-Instruct
and Llama-3.2-1B-Instruct; tasks alpaca, code, math, xsum, squad, agnews) —
trained over ~17 days on a single local GPU. The discipline is
pre-registration with teeth: five weight-only analyses (task identifiability
within and across families, bridge-swap ablation, merge-conflict prediction,
and a pilot-correlation re-verification) were locked before the bank
completed, every open analytical choice was pinned by an independent Director
in dated rulings, and a completeness interlock refused every real-bank
statistic until all 480 runs were complete. Each analysis then ran against
the real bank exactly once. Every headline number was subsequently
re-derived by the Director from per-item data — for the two identifiability
results, re-run from the feature matrices rather than from saved predictions.

The results, in delivery order. First, raw adapter weights do not reveal
their training task: leave-one-out accuracy 0.0792 (qwen) and 0.1375 (llama)
against chance 0.1667, both failing the pre-registered 1.5×-chance lock —
while the GL(r)-gauge-canonical and vocabulary-signature representations
identify all six tasks at 1.0000, permutation p = 0.000999. The information
is in the weights; the gauge is what makes it illegible. Second, our own
pre-registered prediction that task structure would *not* transfer across
model families was refuted — and refuted specifically by the triviality
control added in round-1 review. A family-identity probe reads 1.0000 on raw
representations (the families are perfectly separable by scale), so raw
transfer-at-chance was a covariate-shift artifact; once per-family
standardization removes it, task structure transfers across a 1.5B and a 1B
model of different lineages at 0.7375–0.7833 (binomial p ≤ 1.20e-84). The
control converted a would-be false confirmation into a refutation. That is
pre-registration working as designed, and we report it as the refutation it
is. Third, the trained bridge is nearly free to swap between tasks:
installing a different task's bridge costs ~0.0000 nats of validation loss,
and only full-entry permutation that destroys the identity backbone costs
anything (+2.8086 qwen / +3.8365 llama). Fourth, two adapters' weights alone
predict whether their midpoint merge degrades, at group-aware AUC 0.995
[0.983, 1.000] (qwen) and 0.962 [0.898, 0.999] (llama) — a +0.320 [0.150,
0.511] and +0.249 [0.039, 0.490] margin over a distance-only baseline, both
margin CIs excluding 0. This is the weight-only, post-hoc regime: no
training access, distinct from the training-time form of merge-conflict
prediction (arXiv:2606.19549; Section 2). Fifth, a pilot correlation between
bridge deviation and generalization gap (r = 0.888) shrinks at bank scale to
r = 0.300 [0.175, 0.415] pooled over 480 runs — real, positive, modest, and
heterogeneous within task. We report the shrink as a headline property of
the result, not a footnote.

Two of these five outcomes went against us — the H2 refutation and the D-aux
shrink — and they are the credibility of the other three. A pipeline that
can only confirm is not a measurement instrument. The interlock, the pinned
decisions, the dated amendments, and the independent regrade exist so that
the ceiling results (H1 at 1.0000, D3 at 0.96–0.99) arrive with the same
evidentiary standing as the refutation that sits beside them.

The costs are reported alongside the benefits, here and in the discussion:
~17 days of a single local GPU for the bank; two model families of 1–1.5B
parameters only; 6 coarse tasks, so the label-granularity axis (Section 2)
cuts both ways and we claim nothing about fine-grained regimes; midpoint
merges only; validation-loss-based labels; and a distance-only baseline with
a few points of CV-seed sensitivity, pinned and reported where it appears.

**Contributions.** The bank artifact, and five results measured against it
exactly once:

1. **The bank.** A 480-adapter, 2-family × 6-task × 40-seed LoRA bank,
   cohort-tagged and outage-audited, with a completeness interlock that held
   — no real-bank statistic was computed before 480/480 — and a per-item
   verification bundle from which every headline was independently
   re-derived.
2. **The canonicalization ceiling (H1).** Raw weights fail the
   task-identifiability lock (0.0792 / 0.1375 vs chance 0.1667); the
   gauge-canonical and vocab-signature representations reach 1.0000 in both
   families. Within a controlled family, a linear probe suffices once the
   GL(r) gauge is removed — canonicalization, not classifier capacity, is
   the binding constraint.
3. **The refutation-by-control (H2).** The pre-registered no-transfer
   prediction is NOT supported: raw transfer-at-chance was a family-scale
   artifact exposed by the family-identity probe (1.0000 raw →
   0.1521/0.0000 standardized), and shift-controlled transfer runs
   0.7375–0.7833 in both directions and both representations.
4. **The backbone result (D2).** Across 360 evaluations per family, every
   swap that preserves the identity backbone costs ~0.0000 nats; only
   destroying the backbone costs (+2.8086 / +3.8365). The backbone is the
   sole load-bearing structure for in-distribution loss.
5. **The post-hoc merge predictor (D3).** Weight-only features predict
   midpoint-merge conflict at AUC 0.995 / 0.962 with a CI-separated margin
   over raw distance, on 120 vertex-disjoint pairs per family — the first
   result we know of in the weight-only, post-hoc, no-training-access
   regime.
6. **The honest re-verification (D-aux).** The pilot deviation↔gap
   correlation r = 0.888 shrinks to 0.300 [0.175, 0.415] at bank scale,
   with within-task cells shown, including a significantly negative one.
   Small-n, task-mixture pilot correlations do not survive scale, and a
   pre-registered bank is how you find out.

# 2. Related work

**Weight-space readout of LoRA adapters.** W2T ("LoRA Weights Already Know
What They Can Do," arXiv:2603.15990, March 2026) established that LoRA
checkpoints are legible objects: attribute classification and retrieval over
collections of 10k+ adapters, using a QR→SVD canonicalization and
symmetry-aware encoders that substantially outperform raw-flattened
baselines. Our regime contrast with W2T runs on the label-granularity and
task-structure axis: W2T classifies 10k+ fine-grained attribute classes,
where even its best models leave large headroom; we classify 6 coarse tasks,
where a linear probe on canonical features reaches ceiling. The contrast is
not attributable to collection scale — W2T's collections are themselves
same-base, same-rank families (their Table 6) — so the honest distinguishing
variable is what the labels ask of the weights, not how many adapters sit in
the pile. Our within-family raw-vs-canonical comparison is the complement of
their result, not a contradiction of it: both find that the raw
parameterization is the obstacle, at opposite ends of the granularity axis.

**Canonicalization and weight-space symmetry.** The GL(r) gauge is the
shared enemy of this line. W2T's QR→SVD canonicalization is the lineage our
canonical representation follows; Spectral Geometry (arXiv:2604.08844) reads
training objective from adapter spectra with linear classifiers (AUC ≈ 1.00
within-method, failing cross-method); and the symmetry-aware-encoder line
W2T reports is the learned alternative to explicit canonicalization. Our H1
result places a point on this map: within a controlled family at coarse
label granularity, no learned encoder is needed — explicit gauge removal
plus a linear SVM is already at ceiling, and the same features left in the
raw gauge fail a 1.5×-chance lock. Canonicalization is the whole story at
this operating point.

**Merge-conflict prediction.** arXiv:2606.19549 (June 2026) owns the
training-time form of this problem: predicting merge conflict with access to
the training process. Our D3 result is scoped exactly as our experiment card
scoped it — weight-only and post-hoc, from the two adapters' tensors alone,
with no training access — and we claim only that regime. MERGE-PEFT
provides benchmark territory adjacent to both forms. The two regimes answer
different operational questions: theirs, whether a conflict can be
anticipated while training is still running; ours, whether a repository of
finished adapters can be triaged for mergeability with nothing but the
files.

**Weight-only diagnostics in kind.** Reading properties of a model from its
weights alone has precedent outside merge prediction: backdoor forensics
from adapter weights (arXiv:2602.15195, with a 500+-adapter benchmark) and
WeightWatcher-PEFT's overfit-adjacent weight diagnostics, the neighborhood
our D-aux re-verification lives in. Our contribution to this kind is less
any single detector than the evidentiary standard: a pre-registered bank
large enough to shrink a pilot correlation honestly.

**Method provenance.** The operational method that produced these numbers —
typed state blocks over prose restatement, maker–grader separation,
pre-registration with pinned decisions and dated amendments — is the house
discipline measured in rhombic-xr001 and enforced here by the experiment
card and Director-decision protocol. We cite it for method provenance, not
for numbers; no result from that work enters this paper.

<!-- Writer's note (not for publication): per the outline's verification
note, the arXiv IDs and paper claims in this section (W2T 2603.15990,
2606.19549, 2604.08844, 2602.15195, MERGE-PEFT, WeightWatcher-PEFT) are
copied from docs/LITERATURE_WATCH_2026-07-03.md and must be re-checked
against the papers themselves before submission. W2T Table 6 (same-base/
same-rank collections) and Table 1 claims likewise. -->
