# S2 TIMING PILOT — qwen2.5-7b DRY RUN

Timing-only run under the registered H2-at-scale card (`docs/REGISTRATION_H2_SCALE_2026-07-30.md`, S2). Excluded from every bank. No H2 statistic is computed. Every value below is measured, or derived from measured values by the formula stated in `scripts/s2_timing_pilot.py`.

=== VERIFIED STATE ===
model_id       = Qwen/Qwen2.5-7B-Instruct
steps          = 10
batch_geometry = bs4xga4
wall_clock_min = 2.93
peak_vram_gb   = 20.30
tokens_trained = 81920
date           = 2026-07-29
=== END VERIFIED STATE ===

## Context

family                      = qwen2.5-7b
draft_label                 = Qwen2.5-7B
run_kind                    = dry-run (projection only)
status                      = COMPLETE
task                        = alpaca
task_choice_authority       = Meridian (card specifies count, not tasks)
seed                        = 90020
data_seed                   = 91020
effective_batch             = 16
max_len                     = 512
wall_clock_incl_process_min = 3.04
peak_vram_reserved_gb       = 21.58
tokens_per_second           = 466.4
hf_cache_warm_at_start      = true
n_params_base_measured      = 7615616512
n_injected_modules          = 112
attn_implementation         = sdpa
model_dtype                 = torch.bfloat16

## Measured breakdown

model_load_s    = 9.20
setup_s         = 49.58
step_loop_s     = 42.18
steps_recorded  = 10
mean_step_s     = 4.2182
val_evals       = 2
mean_val_eval_s = 41.82
save_s          = 0.10

## Projection (NOT a measurement)

projection_basis_steps         = 10
n_evals_in_full_run            = 21
PROJECTED_min_per_full_run     = 156.07
PROJECTED_min_excl_model_load  = 155.92
PROJECTED_gpu_days_per_60_runs = 6.50
draft_estimate_min_per_run     = 230
projection_bias                = upper (first-step cost amortized over the dry run's few steps)

## Provenance

EXCLUDED_FROM_BANK          = true
computes_h2_statistics      = false
campaign_tag                = s2-timing-pilot
config_asset_field          = asset1-bank
trainer                     = asset1_bank.run_single (imported unmodified)
recipe_deviations           = steps=10 (dry run); LR schedule truncated — timing only
bank_manifest_sha256_before = a2004910a8a290a1
bank_manifest_sha256_after  = a2004910a8a290a1
bank_manifest_unchanged     = true
git_commit                  = d8cf4785be2aaeb23b062d864e4d8e7685518ef3
gpu                         = NVIDIA RTX 6000 Ada Generation
python                      = 3.10.19
torch                       = 2.6.0+cu124
transformers                = 4.57.3
finished_at_utc             = 2026-07-30T01:30:31.160108+00:00
