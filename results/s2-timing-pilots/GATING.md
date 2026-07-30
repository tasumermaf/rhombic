# S2 timing pilots — family gating status

Probed 2026-07-29 on Timothy's workstation (single RTX 6000 Ada 48GB), account `timotheospaul`, via `asset1_bank.probe_family_access` (config-only download; the same classifier the Asset-1 campaign used).

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
llama3.1-8b_access   = BLOCKED
=== END VERIFIED STATE ===

## Consequence

Accessible, pilots run: gemma2-2b, qwen2.5-3b, qwen2.5-7b.

SKIPPED (license gate not accepted on this account): llama3.1-8b. Not a blocker — Timothy accepts model gates manually; re-run `--gate-check` after acceptance and the skipped family's pilots proceed unchanged. Until then its measured rate does not exist, so the Section 3 cost table cannot be restated for it.

`llama3.1-8b` probe error, verbatim first line:

```
GatedRepoError: 403 Client Error. (Request ID: Root=1-6a6aa666-453189c34dd70ea1133b2233;166c5b04-a4fc-405c-b17e-7653c3c58a31)
```

Note for the Director: `meta-llama/Llama-3.2-1B-Instruct` (an Asset-1 anchor family) IS accessible on this account — Llama-3.1-8B is a separately gated repo, not a lapsed account. The draft's ALTERNATE for a license-blocked Llama-3.1-8B is Mistral-7B (Section 3), and its drop option is Llama-3.1-8B itself; both remain open. No substitution is made here — that is an S1 decision.

## Environment required on every model-load step

HF_HUB_CACHE      = C:\falco\hf-cache\hub
HF_DATASETS_CACHE = C:\falco\hf-cache\datasets
reason            = the default HF cache is a junction onto a full drive; launching without these fails on xsum/squad
