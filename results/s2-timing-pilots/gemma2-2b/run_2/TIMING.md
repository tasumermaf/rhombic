# S2 TIMING PILOT — gemma2-2b run_2

Timing-only run under the registered H2-at-scale card (`docs/REGISTRATION_H2_SCALE_2026-07-30.md`, S2). Excluded from every bank. No H2 statistic is computed. Every value below is measured, or derived from measured values by the formula stated in `scripts/s2_timing_pilot.py`.

=== VERIFIED STATE ===
model_id       = google/gemma-2-2b-it
steps          = 2000
batch_geometry = bs4xga4
wall_clock_min = 77.75
peak_vram_gb   = 14.07
tokens_trained = 16384000
date           = 2026-07-29
=== END VERIFIED STATE ===

## Context

family                      = gemma2-2b
draft_label                 = Gemma-2-2B
run_kind                    = timing run 2 of 3
status                      = COMPLETE
task                        = code
task_choice_authority       = Meridian (card specifies count, not tasks)
seed                        = 90002
data_seed                   = 91002
effective_batch             = 16
max_len                     = 512
wall_clock_incl_process_min = 77.88
peak_vram_reserved_gb       = 16.19
tokens_per_second           = 3512.0
hf_cache_warm_at_start      = true
n_params_base_measured      = 2614341888
n_injected_modules          = 104
attn_implementation         = sdpa
model_dtype                 = torch.bfloat16

## Measured breakdown

model_load_s    = 5.24
setup_s         = 31.76
step_loop_s     = 4176.60
steps_recorded  = 2000
mean_step_s     = 2.0883
val_evals       = 21
mean_val_eval_s = 21.74
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
git_commit                  = dfa0f911d23c287052d8335b76da858580c6e97a
gpu                         = NVIDIA RTX 6000 Ada Generation
python                      = 3.10.19
torch                       = 2.6.0+cu124
transformers                = 4.57.3
finished_at_utc             = 2026-07-30T04:04:05.320021+00:00
