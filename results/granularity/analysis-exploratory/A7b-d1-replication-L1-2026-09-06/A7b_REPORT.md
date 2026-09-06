# D1 replication on L1-taxonomy adapters, exploratory, n = 60

Produced by `scripts/a7b_d1_replication_l1.py` on the `asset1_d1_identifiability` H1 machinery (imported, not forked), per the Director's Item 6 ruling of 2026-09-06. **This is not a level.** No D10 lock, no D5 ceiling, no registered granularity metric, no clean-core split, no D6 reference, and no tier gate is recorded. Scope limit: **llama3.2-1b only**.

=== VERIFIED STATE ===
label                                     = D1 replication on L1-taxonomy adapters, exploratory, n = 60
exploratory_only                          = true
not_a_level                               = true
locks_applied                             = none
n_adapters                                = 60
k_parent_classes                          = 6
per_parent_n                              = 10
chance                                    = 0.166667
raw_acc                                   = 0.0167
raw_kappa                                 = -0.1800
raw_p                                     = 0.029970
raw_jeffreys_ci_95                        = [0.0018, 0.0753]
canonical_acc                             = 0.9500
canonical_kappa                           = 0.9400
canonical_p                               = 0.000999
canonical_jeffreys_ci_95                  = [0.8726, 0.9857]
vocab_signature_acc                       = 1.0000
vocab_signature_kappa                     = 1.0000
vocab_signature_p                         = 0.000999
vocab_signature_jeffreys_ci_95            = [0.9592, 1.0000]
vocab_signature_kv_exclude_acc            = 1.0000
vocab_signature_kv_exclude_kappa          = 1.0000
vocab_signature_kv_exclude_p              = 0.000999
vocab_signature_kv_exclude_jeffreys_ci_95 = [0.9592, 1.0000]
delta_canonical_minus_raw                 = +0.9333
=== END VERIFIED STATE ===

## Per representation

### raw

representation = raw
kv_mode        = —
loo_accuracy   = 0.0167
n_correct      = 1
n_errors       = 59
jeffreys_ci_95 = [0.0018, 0.0753]
wilson_ci_95   = [0.0029, 0.0886]
cohens_kappa   = -0.1800
permutation_p  = 0.029970
null_mean      = 0.0010
null_max       = 0.1167
macro_f1       = 0.0055
feature_dim    = 5114112
locks_applied  = none

### canonical

representation = canonical
kv_mode        = —
loo_accuracy   = 0.9500
n_correct      = 57
n_errors       = 3
jeffreys_ci_95 = [0.8726, 0.9857]
wilson_ci_95   = [0.8630, 0.9829]
cohens_kappa   = 0.9400
permutation_p  = 0.000999
null_mean      = 0.0082
null_max       = 0.0833
macro_f1       = 0.9496
feature_dim    = 50688
locks_applied  = none

### vocab_signature

representation = vocab_signature
kv_mode        = zero_pad
loo_accuracy   = 1.0000
n_correct      = 60
n_errors       = 0
jeffreys_ci_95 = [0.9592, 1.0000]
wilson_ci_95   = [0.9398, 1.0000]
cohens_kappa   = 1.0000
permutation_p  = 0.000999
null_mean      = 0.1391
null_max       = 0.3500
macro_f1       = 1.0000
feature_dim    = 8704
locks_applied  = none

### vocab_signature_kv_exclude

representation = vocab_signature
kv_mode        = exclude
loo_accuracy   = 1.0000
n_correct      = 60
n_errors       = 0
jeffreys_ci_95 = [0.9592, 1.0000]
wilson_ci_95   = [0.9398, 1.0000]
cohens_kappa   = 1.0000
permutation_p  = 0.000999
null_mean      = 0.1374
null_max       = 0.3833
macro_f1       = 1.0000
feature_dim    = 4352
locks_applied  = none

## What n = 60 can and cannot establish

At n = 60, K = 6, balanced 10 per class, a perfect LOO gives a Jeffreys 95% interval of [0.959, 1.000], so the 0.99 ceiling is inside the interval and A7b cannot distinguish "at ceiling" from "one error below it" (one error at n = 60 is 0.9833). What A7b can establish is the direction and rough size of the raw-vs-canonical gap on adapters trained under the L1 taxonomy rather than the Asset-1 taxonomy. The numbers are above; the reading belongs to the Director.

## Cohort — the 60 adapters by parent class

agnews = run_000 run_001 run_012 run_013 run_024 run_025 run_036 run_037 run_048 run_049
alpaca = run_002 run_003 run_014 run_015 run_026 run_027 run_038 run_039 run_050 run_051
code   = run_004 run_005 run_016 run_017 run_028 run_029 run_040 run_041 run_052 run_053
math   = run_006 run_007 run_018 run_019 run_030 run_031 run_042 run_043 run_054 run_055
squad  = run_008 run_009 run_020 run_021 run_032 run_033 run_044 run_045 run_056 run_057
xsum   = run_010 run_011 run_022 run_023 run_034 run_035 run_046 run_047 run_058 run_059

dropped_replicate_5 = run_060 run_061 run_062 run_063
cohort_rule         = the COMPLETE L1 runs whose replicate index is 0-4 in every one of the 12 L1 classes; the replicate-5 runs are dropped so the design is balanced

## Provenance

git_commit              = 174ae959f2ac
generated_at_utc        = 2026-09-06T03:58:36.281325+00:00
n_permutations          = 1000
seed                    = 0
null_stream_key         = default_rng([seed=0, level_index=101, rep_index]) — level_index 101 is outside every registered index space (granularity levels 0-8, D1 families 0-1) so this exploratory null never shares a stream with a registered run
level_index             = 101
svm_c                   = 1.0
note_written_before_run = docs/A7B_NOTE_2026-09-06.md
tier_gates_touched      = none
