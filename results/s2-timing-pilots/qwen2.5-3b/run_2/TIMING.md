# S2 TIMING PILOT — qwen2.5-3b run_2

Timing-only run under the registered H2-at-scale card (`docs/REGISTRATION_H2_SCALE_2026-07-30.md`, S2). Excluded from every bank. No H2 statistic is computed. Every value below is measured, or derived from measured values by the formula stated in `scripts/s2_timing_pilot.py`.

=== VERIFIED STATE ===
model_id       = Qwen/Qwen2.5-3B-Instruct
steps          = 2000
batch_geometry = bs4xga4
wall_clock_min = 82.34
peak_vram_gb   = 11.05
tokens_trained = 16384000
date           = 2026-07-30
=== END VERIFIED STATE ===

## Context

family                      = qwen2.5-3b
draft_label                 = Qwen2.5-3B
run_kind                    = timing run 2 of 3
status                      = COMPLETE
task                        = code
task_choice_authority       = Meridian (card specifies count, not tasks)
seed                        = 90012
data_seed                   = 91012
effective_batch             = 16
max_len                     = 512
wall_clock_incl_process_min = 82.48
peak_vram_reserved_gb       = 12.21
tokens_per_second           = 3316.2
hf_cache_warm_at_start      = true
n_params_base_measured      = 3085938688
n_injected_modules          = 144
attn_implementation         = sdpa
model_dtype                 = torch.bfloat16

## Measured breakdown

model_load_s    = 5.72
setup_s         = 30.46
step_loop_s     = 4462.25
steps_recorded  = 2000
mean_step_s     = 2.2311
val_evals       = 21
mean_val_eval_s = 21.31
save_s          = 0.30

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
finished_at_utc             = 2026-07-30T08:07:03.358798+00:00
