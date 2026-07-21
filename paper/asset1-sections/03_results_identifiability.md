# Results I: Task Identifiability (H1) and Cross-Family Transfer (H2)

Both analyses in this section ran against the real bank exactly once, on
completion day, under the completeness interlock, with every analytical
choice pinned before the bank existed (Section [PREREG]). Every number below
is copied from the delivery report and its per-item verification bundle
(`d1_results.json`); the Director independently re-derived each headline
from the feature matrices, not from our saved predictions, and we note the
re-derivations where they occurred.

## H1: task identity is legible only after canonicalization

H1 asks whether an adapter's weights identify the task it was trained on,
within a family. The classifier is a leave-one-out linear SVM (C = 1.0) over
240 adapters per family (6 tasks × 40 seeds), with a 1,000-shuffle
permutation null. The pre-registered lock is deliberately two-sided: a
representation passes only if LOO accuracy exceeds 1.5× chance (0.2500, at
chance 0.1667) *and* the permutation p is below 0.01. Four representations
were computed per family: the raw flattened adapter weights; the
GL(r)-gauge-canonical representation (QR→SVD, bridge absorbed); and the two
pre-registered vocab-signature variants (amendment A3), with and without
k/v modules.

| family | representation | dim | LOO acc | perm p | H1 lock |
|---|---|---|---|---|---|
| qwen2.5-1.5b | raw | 6,541,248 | **0.0792** | 0.000999 | **FAIL** |
| qwen2.5-1.5b | canonical | 88,704 | **1.0000** | 0.000999 | **PASS** |
| qwen2.5-1.5b | vocab_signature | 15,232 | 1.0000 | 0.000999 | PASS |
| qwen2.5-1.5b | vocab_sig_kv_exclude | 7,616 | 1.0000 | 0.000999 | PASS |
| llama3.2-1b | raw | 5,114,112 | **0.1375** | 0.000999 | **FAIL** |
| llama3.2-1b | canonical | 50,688 | **1.0000** | 0.000999 | **PASS** |
| llama3.2-1b | vocab_signature | 8,704 | 1.0000 | 0.000999 | PASS |
| llama3.2-1b | vocab_sig_kv_exclude | 4,352 | 1.0000 | 0.000999 | PASS |

*Table T2: within-family task identifiability by representation. LOO
accuracy, linear SVM (C = 1.0), 1,000-shuffle permutation null. Lock:
acc > 1.5× chance (0.2500) AND perm p < 0.01. Chance = 0.1667.*

The pattern is stark and consistent across both families. Raw adapter
weights — the highest-dimensional representation by three orders of
magnitude — fail the lock at 0.0792 (qwen) and 0.1375 (llama), at or below
chance. The canonical representation identifies all six tasks perfectly, as
do both vocab-signature variants, at a fraction of the dimensionality. The
permutation p is 0.000999 in every row.

**Raw carries weak signal, and the lock rejects it by design.** Raw's tiny
permutation p alongside sub-threshold accuracy is not a contradiction: raw
weights carry a *weak* recoverable signal (per-class recall on the code task
of 0.40–0.45 drives the null rejection) but come nowhere near the 1.5× bar.
This is exactly the case the two-sided lock was built to reject — a
representation that is statistically distinguishable from noise but not
usefully legible. We report the failure as a failure.

**Heterogeneity guard.** A perfect accuracy invites the suspicion that the
classifier is reading per-task variance scale rather than task structure.
The pre-registered variance-heterogeneity guard (Euclidean, in the exact
feature space the classifier saw) reads ratios of 1.00–1.46 across cells
against a trigger of 3.7 — not triggered. The canonical ceiling is not a
variance artifact.

**Independent re-derivation.** The Director re-derived all eight LOO
accuracies from the per-family confusion matrices, then re-ran the canonical
LOO-SVM from scratch on the 88,704- and 50,688-dimensional feature matrices
(precomputed-Gram LOO, C = 1.0), obtaining 1.0000 for both families. The
ceiling result is real, not a summary artifact.

The reading: task identity lives in the adapter weights, but only becomes
*legible* once the GL(r) gauge is removed. The vocab-signature arm matches
canonical at ceiling in both kv modes, so the result is not specific to one
canonicalization route. Within a controlled family at coarse label
granularity — 6 tasks here, against the 10k+ fine-grained attribute classes
of the W2T setting (Section [RELATED]) — a linear probe suffices once the
gauge is gone; canonicalization, not classifier capacity, is the binding
constraint.

## H2: the pre-registered prediction, refuted by its own control

H2 was our boldest pre-registered prediction, and it was wrong. We predicted
that task structure would *not* transfer across model families. The verdict
is **NOT supported** — and the refutation was produced by our own triviality
control, added in round-1 review before any data existed. We regard this as
the pre-registration working as designed, and we tell it in that order.

**The pinned decision rule.** Two dimension-agnostic representations carry
the analysis: depth-binned singular-value spectra of the effective update
(`spectrum`, primary) and the canonicalized probe-projection (`probe`,
corroborating), with any disagreement between them reportable. The rule,
pinned by the Director on 2026-07-06 before the bank completed: H2 (transfer
fails) is supported iff, for BOTH directions in the shift-controlled
representation, (i) cross-family accuracy is not significantly above chance
at one-sided exact-binomial α = 0.01, and (ii) within-family accuracy
exceeds cross-family accuracy by ≥ 15 percentage points. The
shift-controlled variant is the headline; raw transfer is descriptive. The
constants (`H2_ALPHA = 0.01`, `H2_MARGIN_PP = 15.0`) are recorded in the
output JSON.

**The triviality control.** The round-1 review fix required a
family-identity probe: before reading any transfer number, ask whether a
classifier can tell *which family* an adapter came from.

| representation | variant | family-identity probe acc | chance |
|---|---|---|---|
| spectrum | raw | 1.0000 | 0.5000 |
| spectrum | family_standardized | 0.1521 | 0.5000 |
| probe | raw | 1.0000 | 0.5000 |
| probe | family_standardized | 0.0000 | 0.5000 |

*Table T3: the triviality control. Raw representations perfectly encode
family identity; per-family standardization removes it.*

Raw representations encode the family perfectly (1.0000 against chance
0.5000). A classifier trained on one family and tested on the other is
therefore tested under total covariate shift — raw "transfer-at-chance"
would confirm our prediction while measuring nothing but family scale.
Per-family z-standardization (`familywise_standardize`) collapses the probe
to 0.1521 (spectrum) and 0.0000 (probe). The Director confirmed in source
that the standardization is unsupervised — per-family z-scoring in which
task labels are never touched — so it removes family scale without any
capacity to manufacture task structure.

**Transfer.** With the artifact removed, the picture inverts:

| representation | direction | raw | standardized | binom p (std) | margin |
|---|---|---|---|---|---|
| spectrum | qwen→llama | 0.1667 | **0.7833** | 7.70e-98 | 21.67pp |
| spectrum | llama→qwen | 0.1667 | **0.7375** | 1.20e-84 | 26.25pp |
| probe | qwen→llama | 0.1750 | **0.7792** | 1.37e-96 | 22.08pp |
| probe | llama→qwen | 0.2167 | **0.7792** | 1.37e-96 | 22.08pp |

*Table T4: cross-family transfer accuracy, raw vs family-standardized.
Chance = 0.1667. The headline is the standardized column, per the pinned
rule; raw is descriptive.*

Raw transfer sits at or near chance in every direction — the number that
would have "confirmed" the prediction. Standardized transfer runs
0.7375–0.7833 across a 1.5B and a 1B model of different lineages, with
binomial p between 7.70e-98 and 1.37e-96. Condition (i) of the pinned rule
fails decisively in both directions, in both representations; spectrum and
probe agree.

**Verdict: H2 NOT supported.** The pre-registered "transfer fails"
prediction is refuted. In the shift-controlled representation, task
structure transfers cross-family at ~74–78%. The finding is the opposite of
what we predicted, and the control added in review is exactly what prevented
a false confirmation on a covariate-shift artifact.

**Independent re-derivation.** Because a pre-registration that flips via a
control added in review carries the highest motivated-reasoning risk in the
set, the Director re-ran the entire H2 pipeline end-to-end from the spectrum
and probe feature matrices, applying the standardization independently. The
triviality control reproduces (1.0000 raw → 0.1521 / 0.0000 standardized);
all four standardized transfer accuracies, all four raw baselines, and all
four binomial p-values reproduce to the digit.

Had we run H2 without the triviality control, we would now be reporting a
confirmed prediction that is actually a measurement of family scale. The
cost of the control was one additional probe and a standardization pass; the
benefit was the difference between a false confirmation and a true
refutation. That asymmetry, not the transfer number itself, is what this
subsection is evidence for.
