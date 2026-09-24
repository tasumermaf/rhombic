# GRANULARITY L1 — run 121 · agnews:world_sports

Training run under the LOCKED granularity card (`docs/LOCK_GRANULARITY_2026-08-04.md`). Excluded from every bank. No ladder statistic is computed here. Every value below is measured, or derived from measured values by the formulas in `scripts/s2_timing_pilot.py` (reused unchanged, so these rates are comparable to the S2 rate basis).

=== VERIFIED STATE ===
level          = L1
run_k          = 121
class_id       = agnews:world_sports
status         = COMPLETE
steps          = 2000
wall_clock_min = 35.70
peak_vram_gb   = 6.40
tokens_trained = 16384000
date           = 2026-09-15
=== END VERIFIED STATE ===

## Context

model_id                = meta-llama/Llama-3.2-1B-Instruct
task                    = agnews
tier                    = T1
clean_core              = true
replicate               = 10
seed                    = 301121
data_seed               = 311121
batch_geometry          = bs4xga4
effective_batch         = 16
max_len                 = 512
n_train_pool_expected   = 19469
n_pool_trainer_reported = 19469
n_val_trainer_reported  = 500
val_ids_sha256_trainer  = 0d145f63a7efea46
k_level                 = 12
chance                  = 0.08333333333333333
hf_cache_warm_at_start  = true
peak_vram_reserved_gb   = 7.52
n_params_base_measured  = 1235814400
attn_implementation     = sdpa
model_dtype             = torch.bfloat16

## Measured breakdown

model_load_s    = 2.78
setup_s         = 19.70
step_loop_s     = 1909.75
steps_recorded  = 2000
mean_step_s     = 0.9549
val_evals       = 21
mean_val_eval_s = 10.09
save_s          = 0.20

## Provenance

EXCLUDED_FROM_BANK          = true
computes_ladder_statistics  = false
campaign_tag                = granularity-L1
trainer                     = asset1_bank.run_single (imported unmodified)
dataset_seam                = asset1_bank.build_dataset rebound to serve ds.select(frozen row_ids) via the task class's raw= argument; restored after the run
labels_manifest_git         = e2386854a43b
row_ids_sha256              = d07010635b5831ed
gpu_guard                   = guarded(needed_gb=14, expected_min=32)
bank_manifest_sha256_before = a2004910a8a290a1
bank_manifest_sha256_after  = a2004910a8a290a1
bank_manifest_unchanged     = true
git_commit                  = 1a149ddd746baef405318504c0d1f128628d3107
gpu                         = NVIDIA RTX 6000 Ada Generation
python                      = 3.10.19
torch                       = 2.6.0+cu124
transformers                = 4.57.3
finished_at_utc             = 2026-09-16T01:39:37.885614+00:00
