# S2 TIMING PILOT — qwen2.5-7b run_3

Timing-only run under the registered H2-at-scale card (`docs/REGISTRATION_H2_SCALE_2026-07-30.md`, S2). Excluded from every bank. No H2 statistic is computed. Every value below is measured, or derived from measured values by the formula stated in `scripts/s2_timing_pilot.py`.

=== VERIFIED STATE ===
model_id       = Qwen/Qwen2.5-7B-Instruct
steps          = 2000
batch_geometry = bs4xga4
wall_clock_min = 153.58
peak_vram_gb   = 20.30
tokens_trained = 16384000
date           = 2026-07-30
=== END VERIFIED STATE ===

## Context

family                      = qwen2.5-7b
draft_label                 = Qwen2.5-7B
run_kind                    = timing run 3 of 3
status                      = COMPLETE
task                        = math
task_choice_authority       = Meridian (card specifies count, not tasks)
seed                        = 90023
data_seed                   = 91023
effective_batch             = 16
max_len                     = 512
wall_clock_incl_process_min = 153.72
peak_vram_reserved_gb       = 21.58
tokens_per_second           = 1778.0
hf_cache_warm_at_start      = true
n_params_base_measured      = 7615616512
n_injected_modules          = 112
attn_implementation         = sdpa
model_dtype                 = torch.bfloat16

## Measured breakdown

model_load_s    = 11.14
setup_s         = 24.99
step_loop_s     = 8313.12
steps_recorded  = 2000
mean_step_s     = 4.1566
val_evals       = 21
mean_val_eval_s = 41.72
save_s          = 0.20

## Provenance

EXCLUDED_FROM_BANK          = true
computes_h2_statistics      = false
campaign_tag                = s2-timing-pilot
config_asset_field          = asset1-bank
trainer                     = asset1_bank.run_single (imported unmodified)
recipe_deviations           = none
bank_manifest_sha256_before = a2004910a8a290a1
bank_manifest_sha256_after  = a2004910a8a290a1
bank_manifest_unchanged     = true
git_commit                  = 898f3a6dc80997e75697adef635421cada3b7056
gpu                         = NVIDIA RTX 6000 Ada Generation
python                      = 3.10.19
torch                       = 2.6.0+cu124
transformers                = 4.57.3
finished_at_utc             = 2026-07-30T17:10:45.970643+00:00
