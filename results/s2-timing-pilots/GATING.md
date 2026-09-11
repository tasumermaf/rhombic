# S2 timing pilots — family gating status

Probed 2026-08-04 on Timothy's workstation (single RTX 6000 Ada 48GB), account `timotheospaul`, via `asset1_bank.probe_family_access` (config-only download; the same classifier the Asset-1 campaign used).

## Model-id resolution (Qwen ids confirmed by the Director 2026-08-04; gemma-2-2b-it resolved from the workstation cache 2026-09-11; llama-3.1-8b existence only, via its 401 gate error)

The registered card and the draft's Section 3 name families and require instruct checkpoints; neither prints HF repo ids. The ids below are Meridian's resolution of those labels, each verified to exist on the Hub. `Qwen/Qwen2.5-7B-Instruct` is independently attested in this repo (the pilot bank and BM-003 both used it). The Director's Ask 2 verdict was "three confirmed independently, one on report" (`docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:24`): he fetched the two Qwen configs himself; he confirmed `meta-llama/Llama-3.1-8B-Instruct` "only via the gate error, not its config"; and of `google/gemma-2-2b-it` he wrote that it "also returned 401 and I could not verify it at all", so that id was resolved instead from the workstation cache on 2026-09-11. **Resolved configs, revisions and config.json hashes for every family whose pilots ran: `RESOLVED_CONFIGS.md` (Director Ask 2 closure, 2026-09-11).**

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

Access probe OK: gemma2-2b, qwen2.5-3b, qwen2.5-7b, llama3.1-8b.

## Environment required on every model-load step

HF_HUB_CACHE      = C:\falco\hf-cache\hub
HF_DATASETS_CACHE = C:\falco\hf-cache\datasets
reason            = the default HF cache is a junction onto a full drive; launching without these fails on xsum/squad
