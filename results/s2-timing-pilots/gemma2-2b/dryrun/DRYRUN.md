# S2 TIMING PILOT — gemma2-2b DRY RUN

Timing-only run under the registered H2-at-scale card (`docs/REGISTRATION_H2_SCALE_2026-07-30.md`, S2). Excluded from every bank. No H2 statistic is computed. Every value below is measured, or derived from measured values by the formula stated in `scripts/s2_timing_pilot.py`.

=== VERIFIED STATE ===
model_id       = google/gemma-2-2b-it
steps          = 10
batch_geometry = bs4xga4
wall_clock_min = 4.30
peak_vram_gb   = 14.07
tokens_trained = 81920
date           = 2026-07-29
=== END VERIFIED STATE ===

## Context

family                      = gemma2-2b
draft_label                 = Gemma-2-2B
run_kind                    = dry-run (projection only)
status                      = COMPLETE
task                        = alpaca
task_choice_authority       = Meridian (card specifies count, not tasks)
seed                        = 90000
data_seed                   = 91000
effective_batch             = 16
max_len                     = 512
wall_clock_incl_process_min = 4.66
peak_vram_reserved_gb       = 16.19
tokens_per_second           = 317.3
hf_cache_warm_at_start      = false
n_params_base_measured      = 2614341888
n_injected_modules          = 104
attn_implementation         = sdpa
model_dtype                 = torch.bfloat16

## Measured breakdown

model_load_s    = 152.08
setup_s         = 199.94
step_loop_s     = 18.86
steps_recorded  = 10
mean_step_s     = 1.8860
val_evals       = 2
mean_val_eval_s = 19.50
save_s          = 0.20

## Projection (NOT a measurement)

projection_basis_steps         = 10
n_evals_in_full_run            = 21
PROJECTED_min_per_full_run     = 73.03
PROJECTED_gpu_days_per_60_runs = 3.04
draft_estimate_min_per_run     = 79
projection_bias                = upper (first-step cost amortized over the dry run's few steps)
