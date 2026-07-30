# S2 TIMING PILOT — qwen2.5-3b DRY RUN

Timing-only run under the registered H2-at-scale card (`docs/REGISTRATION_H2_SCALE_2026-07-30.md`, S2). Excluded from every bank. No H2 statistic is computed. Every value below is measured, or derived from measured values by the formula stated in `scripts/s2_timing_pilot.py`.

=== VERIFIED STATE ===
model_id       = Qwen/Qwen2.5-3B-Instruct
steps          = 10
batch_geometry = bs4xga4
wall_clock_min = 1.83
peak_vram_gb   = 11.05
tokens_trained = 81920
date           = 2026-07-29
=== END VERIFIED STATE ===

## Context

family                      = qwen2.5-3b
draft_label                 = Qwen2.5-3B
run_kind                    = dry-run (projection only)
status                      = COMPLETE
task                        = alpaca
task_choice_authority       = Meridian (card specifies count, not tasks)
seed                        = 90010
data_seed                   = 91010
effective_batch             = 16
max_len                     = 512
wall_clock_incl_process_min = 1.95
peak_vram_reserved_gb       = 12.21
tokens_per_second           = 746.3
hf_cache_warm_at_start      = true
n_params_base_measured      = 3085938688
n_injected_modules          = 144
attn_implementation         = sdpa
model_dtype                 = torch.bfloat16

## Measured breakdown

model_load_s    = 5.33
setup_s         = 53.57
step_loop_s     = 18.99
steps_recorded  = 10
mean_step_s     = 1.8990
val_evals       = 2
mean_val_eval_s = 18.42
save_s          = 0.20

## Projection (NOT a measurement)

projection_basis_steps         = 10
n_evals_in_full_run            = 21
PROJECTED_min_per_full_run     = 70.64
PROJECTED_min_excl_model_load  = 70.55
PROJECTED_gpu_days_per_60_runs = 2.94
draft_estimate_min_per_run     = 93
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
finished_at_utc             = 2026-07-30T01:26:41.583950+00:00
