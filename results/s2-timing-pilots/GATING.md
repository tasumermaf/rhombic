# S2 timing pilots — family gating status

Probed 2026-08-04 on Timothy's workstation (single RTX 6000 Ada 48GB), account `timotheospaul`, via `asset1_bank.probe_family_access` (config-only download; the same classifier the Asset-1 campaign used).

## Model-id resolution (needs Director confirmation)

The registered card and the draft's Section 3 name families and require instruct checkpoints; neither prints HF repo ids. The ids below are Meridian's resolution of those labels, each verified to exist on the Hub. `Qwen/Qwen2.5-7B-Instruct` is independently attested in this repo (the pilot bank and BM-003 both used it).

=== VERIFIED STATE ===
gemma2-2b_model_id   = google/gemma-2-2b-it
qwen2.5-3b_model_id  = Qwen/Qwen2.5-3B-Instruct
qwen2.5-7b_model_id  = Qwen/Qwen2.5-7B-Instruct
llama3.1-8b_model_id = meta-llama/Llama-3.1-8B-Instruct
gemma2-2b_access     = OK
qwen2.5-3b_access    = OK
qwen2.5-7b_access    = OK
llama3.1-8b_access   = OK
=== END VERIFIED STATE ===

## Consequence

Accessible, pilots run: gemma2-2b, qwen2.5-3b, qwen2.5-7b, llama3.1-8b.

## Environment required on every model-load step

HF_HUB_CACHE      = C:\falco\hf-cache\hub
HF_DATASETS_CACHE = C:\falco\hf-cache\datasets
reason            = the default HF cache is a junction onto a full drive; launching without these fails on xsum/squad
