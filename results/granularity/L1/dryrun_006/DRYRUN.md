# GRANULARITY L1 — DRY RUN · math:steps_ge4

Training run under the LOCKED granularity card (`docs/LOCK_GRANULARITY_2026-08-04.md`). Excluded from every bank. No ladder statistic is computed here. Every value below is measured, or derived from measured values by the formulas in `scripts/s2_timing_pilot.py` (reused unchanged, so these rates are comparable to the S2 rate basis).

=== VERIFIED STATE ===
level          = L1
run_k          = 6
class_id       = math:steps_ge4
status         = COMPLETE
steps          = 60
wall_clock_min = 1.25
peak_vram_gb   = 6.40
tokens_trained = 491520
date           = 2026-08-04
=== END VERIFIED STATE ===

## Context

model_id                = meta-llama/Llama-3.2-1B-Instruct
task                    = math
tier                    = T2
clean_core              = true
replicate               = 0
seed                    = 301006
data_seed               = 311006
batch_geometry          = bs4xga4
effective_batch         = 16
max_len                 = 512
n_train_pool_expected   = 2628
n_pool_trainer_reported = 2628
n_val_trainer_reported  = 500
val_ids_sha256_trainer  = bb165a5799161ffb
k_level                 = 12
chance                  = 0.08333333333333333
hf_cache_warm_at_start  = true
peak_vram_reserved_gb   = 7.52
n_params_base_measured  = 1235814400
attn_implementation     = sdpa
model_dtype             = torch.bfloat16

## Measured breakdown

model_load_s    = 2.69
setup_s         = 6.72
step_loop_s     = 49.81
steps_recorded  = 60
mean_step_s     = 0.8302
val_evals       = 2
mean_val_eval_s = 9.13
save_s          = 0.10

## Projection (NOT a measurement)

projection_basis_steps           = 60
n_evals_in_full_run              = 21
PROJECTED_min_per_full_run       = 30.98
PROJECTED_min_excl_model_load    = 30.94
measured_llama_basis_min_per_run = 30.56
basis_source                     = results/asset1-bank/RATE_EXTRACT.md (n=240)
projection_bias                  = upper (first-step cost amortized over few steps)

## Provenance

EXCLUDED_FROM_BANK          = true
computes_ladder_statistics  = false
campaign_tag                = granularity-L1
trainer                     = asset1_bank.run_single (imported unmodified)
dataset_seam                = asset1_bank.build_dataset rebound to serve ds.select(frozen row_ids) via the task class's raw= argument; restored after the run
labels_manifest_git         = e2386854a43b
row_ids_sha256              = 7cbdd6aaf4c808a2
gpu_guard                   = guarded(needed_gb=14, expected_min=32)
bank_manifest_sha256_before = a2004910a8a290a1
bank_manifest_sha256_after  = a2004910a8a290a1
bank_manifest_unchanged     = true
git_commit                  = e2386854a43b85ff42f3e97c1b9f7689f136a80d
gpu                         = NVIDIA RTX 6000 Ada Generation
python                      = 3.10.19
torch                       = 2.6.0+cu124
transformers                = 4.57.3
finished_at_utc             = 2026-08-04T19:17:01.307248+00:00
