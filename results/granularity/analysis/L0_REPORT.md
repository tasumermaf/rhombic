# GRANULARITY L0 — LEVEL REPORT

Produced by `scripts/granularity_analysis.py` on the `asset1_d1_identifiability` H1 machinery (imported, not forked). Scope limit: **llama3.2-1b only** — it belongs in every claim taken from this file.

=== VERIFIED STATE ===
level                                 = L0
k                                     = 6
n_runs                                = 240
chance                                = 0.166667
exploratory_only                      = false
raw_acc                               = 0.1375
raw_kappa                             = -0.0350
raw_p                                 = 0.001998
raw_d10_pass                          = false
raw_at_ceiling                        = false
canonical_acc                         = 1.0000
canonical_kappa                       = 1.0000
canonical_p                           = 0.000999
canonical_d10_pass                    = true
canonical_at_ceiling                  = true
vocab_signature_acc                   = 1.0000
vocab_signature_kappa                 = 1.0000
vocab_signature_p                     = 0.000999
vocab_signature_d10_pass              = true
vocab_signature_at_ceiling            = true
vocab_signature_kv_exclude_acc        = 1.0000
vocab_signature_kv_exclude_kappa      = 1.0000
vocab_signature_kv_exclude_p          = 0.000999
vocab_signature_kv_exclude_d10_pass   = true
vocab_signature_kv_exclude_at_ceiling = true
delta_K_canonical_minus_raw           = +0.8625
d6_cv_accuracy                        = 0.9745
d6_kappa                              = 0.9694
d6_realized_n_min                     = 1000
=== END VERIFIED STATE ===

## All Classes

### raw

loo_accuracy              = 0.1375
wilson_ci_95              = [0.0996, 0.1868]
cohens_kappa              = -0.0350
permutation_p             = 0.001998
null_mean                 = 0.0575
macro_f1                  = 0.0714
acc_bar_1p5x_chance       = 0.2500
D10_accuracy              = false
D10_p                     = true
D10_kappa                 = false
D10_PASS                  = FALSE
at_ceiling_0p99           = false
errors                    = 207
parent_collapsed_accuracy = 0.1375
feature_dim               = 5114112

### canonical

loo_accuracy              = 1.0000
wilson_ci_95              = [0.9842, 1.0000]
cohens_kappa              = 1.0000
permutation_p             = 0.000999
null_mean                 = 0.1211
macro_f1                  = 1.0000
acc_bar_1p5x_chance       = 0.2500
D10_accuracy              = true
D10_p                     = true
D10_kappa                 = true
D10_PASS                  = TRUE
at_ceiling_0p99           = true
errors                    = 0
parent_collapsed_accuracy = 1.0000
feature_dim               = 50688

### vocab_signature

loo_accuracy              = 1.0000
wilson_ci_95              = [0.9842, 1.0000]
cohens_kappa              = 1.0000
permutation_p             = 0.000999
null_mean                 = 0.1582
macro_f1                  = 1.0000
acc_bar_1p5x_chance       = 0.2500
D10_accuracy              = true
D10_p                     = true
D10_kappa                 = true
D10_PASS                  = TRUE
at_ceiling_0p99           = true
errors                    = 0
parent_collapsed_accuracy = 1.0000
feature_dim               = 8704

### vocab_signature_kv_exclude

loo_accuracy              = 1.0000
wilson_ci_95              = [0.9842, 1.0000]
cohens_kappa              = 1.0000
permutation_p             = 0.000999
null_mean                 = 0.1578
macro_f1                  = 1.0000
acc_bar_1p5x_chance       = 0.2500
D10_accuracy              = true
D10_p                     = true
D10_kappa                 = true
D10_PASS                  = TRUE
at_ceiling_0p99           = true
errors                    = 0
parent_collapsed_accuracy = 1.0000
feature_dim               = 4352

## Clean core (D3)

every class at this level is T1+T2 — NO T3 CELLS, so the D3 clean-core requirement is NOT TESTABLE at this level. This is not a statement that it passed: the clean-core and all-classes label spaces are identical by construction, so the divergence rule has nothing to bite on (G-4 principle, applied per Director condition (c) of 2026-08-05).

## D6 data-space reference

k                     = 6
n_documents           = 6000
n_features            = 18497
cv_accuracy           = 0.9745
cohens_kappa          = 0.9694
realized_n_min        = 1000
realized_n_max        = 1000
at_nominal_everywhere = true
cv                    = StratifiedKFold(n_splits=5, shuffle=True, random_state=6) WITHIN the subsample (Ask 3 condition i)

Bounds attainable weight-space accuracy: a class whose own text is inseparable cannot be read from weights, so a failure there is label noise, not a canonicalization failure (design §6.1; converts outcome (C) from an excuse into a finding).

## Provenance

git_commit       = 174ae959f2ac
generated_at_utc = 2026-09-06T03:37:17.994993+00:00
n_permutations   = 1000
seed             = 0
svm_c            = 1.0
tier_order       = L0 -> L1 -> ARMB -> L2 -> L3 -> D7
