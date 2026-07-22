# Asset-1 Weight-Only Diagnostics — Complete Campaign Report

**To:** The Director
**From:** Meridian (hub), for Timothy Paul Bielec — TASUMER MAF
**Date:** 2026-07-20
**Re:** The 480-adapter D1 bank, trained and fully analyzed. Every
pre-registered analysis (D1/H1, D1/H2, D2, D3, D-aux) has run against the
real bank exactly once, with the locked hypotheses and Director-pinned
decisions. This is the single delivery packet; result tables only,
hypotheses unchanged.

---

## 0. One-paragraph result

Canonicalization is decisive and the pre-registration held its ground twice.
Raw adapter weights do **not** reveal the task they were trained on (H1 fails
on raw); the GL(r)-gauge-canonical and output-referenced representations
identify all six tasks **perfectly** (100%, p < 0.001). The pre-registered
prediction that task structure would **not** transfer across model families
was **refuted** — and refuted specifically by the triviality control that was
added in round-1 review: raw "transfer-at-chance" turned out to be a
family-scale artifact, and once controlled, task structure transfers
cross-family at ~74–78%. The bridge (topology) is nearly free to swap between
tasks (~0 val-loss penalty); only destroying the identity backbone costs
anything. Post-hoc, weight-only features predict merge conflict at AUC
0.96–0.99, well above a distance-only baseline. D-aux's pilot correlation
(r = 0.888) shrinks honestly to r = 0.300 at bank scale and is
between-task, not universal.

## 1. Bank (typed core)

```
bank.complete        = 480 / 480 COMPLETE [bank_manifest.json 2026-07-20T16:21:19Z]
bank.geometry        = bs4xga4 (A1 amendment, Director-adopted 2026-07-06); cohort-tagged
bank.outage_ledger   = 4 FAILED (HF Hub 504, Jul 16): idx 329/330/331/332 all retried COMPLETE
bank.duration        = Jul 3 21:52Z -> Jul 20 16:21Z (~17 days incl. A1 restart; ~1.9x speedup)
interlock.status     = HELD — no real-bank statistic computed before 480/480
tooling.verification = 171/171 tests + D1/D3/D-aux selftests PASSED (re-run 2026-07-20)
chance               = 0.1667 (6 tasks)
```

## 2. D1 / H1 — within-family task identifiability

LOO accuracy, linear SVM (C=1.0), 1,000-shuffle permutation null. Lock:
acc > 1.5x chance (0.2500) AND permutation p < 0.01.

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

Variance-heterogeneity guard: all ratios 1.00–1.46 (trigger 3.7) — not
triggered; the perfect canonical result is not a variance artifact. Raw's
tiny p (0.000999) with sub-threshold accuracy means raw carries *weak*
recoverable signal (code recall 0.40–0.45 drives it) but nowhere near the
1.5x bar — the lock is designed to reject exactly this. **Reading:** task
identity lives in adapter weights, but only *legibly* once the GL(r) gauge is
removed. A3 arm #3 (vocab_signature) matches canonical at ceiling in both
kv_modes, so there is no outcome-(c) deficit and nothing is left provisional
on Level B.

## 3. D1 / H2 — cross-family transfer (regime contrast)

Pinned: spectrum PRIMARY / probe corroborating; shift-controlled variant is
the headline; supported iff BOTH directions are NOT above chance at
alpha=0.01 AND within-minus-cross >= 15pp.

**Triviality control (why the raw reading is untrustworthy):**

| representation | variant | family-identity probe acc | chance |
|---|---|---|---|
| spectrum | raw | 1.0000 | 0.5000 |
| spectrum | family_standardized | 0.1521 | 0.5000 |
| probe | raw | 1.0000 | 0.5000 |
| probe | family_standardized | 0.0000 | 0.5000 |

Raw representations perfectly encode *which family* — so raw transfer-at-
chance is covariate shift, not a genuine result. Standardization removes it.

**Transfer accuracy (headline = family_standardized):**

| representation | direction | raw | standardized | binom p (std) | margin |
|---|---|---|---|---|---|
| spectrum | qwen->llama | 0.1667 | **0.7833** | 7.70e-98 | 21.67pp |
| spectrum | llama->qwen | 0.1667 | **0.7375** | 1.20e-84 | 26.25pp |
| probe | qwen->llama | 0.1750 | **0.7792** | 1.37e-96 | 22.08pp |
| probe | llama->qwen | 0.2167 | **0.7792** | 1.37e-96 | 22.08pp |

**H2 verdict: NOT supported** (spectrum and probe agree). The pre-registered
"transfer fails" prediction is refuted: in the shift-controlled representation
task structure transfers across a 1.5B and a 1B model of different lineages at
~74–78%. The finding is the opposite of the prediction, and the control added
in review is exactly what prevented a false confirmation on an artifact.

## 4. D2 — bridge-swap penalty matrix (360 evals/family, all SHA-verified)

Mean val-loss penalty vs the native adapter, by swap kind:

| kind | qwen | llama |
|---|---|---|
| cross_seed | +0.0000 | +0.0002 |
| cross_task | +0.0000 | +0.0003 |
| cross_task_magnitude | +0.0000 | +0.0000 |
| cross_task_topology | +0.0000 | +0.0002 |
| identity | +0.0000 | +0.0007 |
| permuted_deviation (H3 ref) | +0.0000 | +0.0007 |
| **permuted (full)** | **+2.8086** | **+3.8365** |

**Reading:** installing a *different task's* trained bridge into an adapter
costs essentially nothing on val-loss. The pinned H3 reference
`permuted_deviation` (permute the trained deviation, keep the identity
backbone) also costs ~0; only the full-entry `permuted` (which destroys the
backbone) costs ~3–4 nats. Their contrast isolates the backbone as the sole
load-bearing structure: the trained bridge deviation is real and measurable
(D-aux sees it) but negligible to in-distribution loss. Consistent with the
controller-free, identity-init design.

## 5. D3 — post-hoc weight-only merge-conflict prediction

N=120 vertex-disjoint pairs/family (pre-declared
`docs/D3_PAIR_DESIGN_PREDECLARATION_2026-07-20.md`; amended to uniform, no
per-cell stratification, before any label existed). Primary label = fixed 5%
relative-degradation, EITHER endpoint. Conflict rate 85.8% (above the 10%
degenerate floor — primary rule holds, no median fallback). Headline =
group-aware CV; the vertex-disjoint design gave 120 single-pair components,
so there is no dyadic dependence to correct and group-aware and naive numbers
agree to within CV-fold reshuffle noise (e.g. llama full 0.962 vs naive 0.952).

| family | AUC full (weight-only) | AUC distance-only | full − distance |
|---|---|---|---|
| qwen2.5-1.5b | **0.995** [0.983, 1.000] | 0.675 [0.484, 0.848] | +0.320 [0.150, 0.511] |
| llama3.2-1b | **0.962** [0.898, 0.999] | 0.713 [0.458, 0.923] | +0.249 [0.039, 0.490] |

**Reading:** the two adapters' weights alone predict whether their midpoint
merge degrades, at AUC 0.96–0.99, and the gauge-invariant principal-angle
block adds a large, CI-separated margin over raw distance (both diff CIs
exclude 0). Weight-only, post-hoc, no training access — the regime the card
scoped against arXiv:2606.19549's training-time form.

## 6. D-aux — bridge deviation ↔ generalization gap

Primary pair: dev_mean vs final_gap (step 2000). Step-0 identity control =
0.0 exactly.

| cell | n | pearson r | 95% CI |
|---|---|---|---|
| pooled | 480 | 0.300 | [0.175, 0.415] |
| qwen2.5-1.5b | 240 | 0.418 | [0.323, 0.522] |
| llama3.2-1b | 240 | 0.337 | [0.201, 0.466] |
| within math (llama) | 40 | 0.549 | [0.324, 0.736] |
| within math (qwen) | 40 | 0.336 | [0.030, 0.652] |
| within xsum (qwen) | 40 | −0.301 | [−0.502, −0.066] |

**Reading:** the pilot's r = 0.888 was inflated by small-n and task mixture.
At bank scale the association is real, positive, and non-zero pooled, but
modest and heterogeneous within task (math positive in both families; one
cell significantly negative). Reported as a pooled bank-level claim with the
Simpson's-guard cells shown, not hidden.

## 7. Open items for the Director

None gate this report. Two are for the record:

- **D3 pre-declaration amendment** (§5): the approved sampler is uniform, not
  per-cell stratified; the declaration was amended before any label existed
  (temporal integrity intact), realized cell coverage reported descriptively.
  Flagged for your review as a dated amendment (L-006 / R10).
- **D2 K = 3** stood as approved; no override was requested.

## 8. Provenance and run log

- All numbers trace to `results/asset1-d1/`, `-d2/`, `-d3/`, `-daux/` on disk;
  none restated from memory.
- **Run log (honest):** (a) D1's first launch died on the default-HF-cache
  trap (ops-manual env vars omitted) — environment-level, nothing unblinded;
  relaunched clean. (b) Step 6 (D3 label generation) had no shipped runner
  (the card scoped it "external"); `scripts/asset1_d3_labels.py` was written
  for it, put through a fresh-context adversarial verifier BEFORE the GPU run,
  which caught two blocking defects (flat-vs-nested merge load; a
  machine-absolute manifest path) — both fixed and dry-run-verified before
  launch. Natives were taken from the trainer's `metrics.json` finals after
  verifying 0.00000% divergence against 36 fresh D2 harness evals.
- Verification discipline: maker–grader separation on every custom step;
  the interlock refused all real-bank access until 480/480.

*Prepared by Meridian. Hypotheses locked per the experiment card; this packet
fills result tables only.*
