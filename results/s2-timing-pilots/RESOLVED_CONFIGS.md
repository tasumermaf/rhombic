# S2 timing pilots — resolved model configs

**Closure of the Director's Ask 2 condition of 2026-08-04.** He confirmed the two
Qwen ids by fetching their configs himself, could not reach `google/gemma-2-2b-it`
(401 anonymously), and directed: "Confirm it from the workstation cache (the
pilots ran, so the config is on disk) and record the resolved config alongside the
id, not just the string."
[`docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:24`, restated at `:96`]

Resolved **2026-09-11** by Meridian from the workstation Hugging Face cache at
`HF_HUB_CACHE = C:\falco\hf-cache\hub` — the cache every pilot loaded from
(`GATING.md`, "Environment required on every model-load step"). **Nothing was
downloaded and `--gate-check` was not re-run**; every value below is read from a
`config.json` already on disk at the revision the pilots used.

Each cached `config.json` in this cache is stored as a blob named by its git blob
SHA-1, and each verifies against that name, so the bytes these numbers were read
from are the Hub's bytes at the recorded revision.

`GATING.md` remains the file of record for ids and access states. It is
regenerated wholesale by `scripts/s2_timing_pilot.py --gate-check`
(`scripts/s2_timing_pilot.py:711-712`), so a closure hand-added there would be
erased by the next probe; the configs therefore live here, and `GATING.md` points
at this file.

## gemma2-2b — the family that was on report

=== VERIFIED STATE ===
gemma2-2b_model_id             = google/gemma-2-2b-it
gemma2-2b_snapshot_commit      = 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
gemma2-2b_config_sha256        = eacec6c5ca317a87ed2c46789d9705b9274db5027e7ba59da739bfae23addb55
gemma2-2b_config_git_blob_sha1 = 05131f6b339647ddff99327b199ac3d34a50bf2e
gemma2-2b_config_bytes         = 838
gemma2-2b_blob_name_verifies   = YES (blob filename == git blob SHA-1 of the bytes)
gemma2-2b_model_type           = gemma2
gemma2-2b_hidden_size          = 2304
gemma2-2b_num_hidden_layers    = 26
gemma2-2b_intermediate_size    = 9216
gemma2-2b_num_attention_heads  = 8
gemma2-2b_num_key_value_heads  = 4
gemma2-2b_head_dim             = 256
gemma2-2b_vocab_size           = 256000
gemma2-2b_resolved_from        = C:\falco\hf-cache\hub\models--google--gemma-2-2b-it\snapshots\299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8\config.json
gemma2-2b_resolved_at          = 2026-09-11
=== END VERIFIED STATE ===

`head_dim` is explicit in this config and is **not** `hidden_size //
num_attention_heads`: 8 x 256 = 2048, while `hidden_size` is 2304. That is correct
for Gemma-2 and is not a transcription error.

## Corroboration for gemma2-2b

1. **A second, physically distinct store holds identical bytes.** The default HF
   cache on this workstation is a junction: `~\.cache\huggingface` ->
   `F:\AI-Models\huggingface-cache`. Its copy resolves `refs/main` to the same
   commit `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` and its `config.json` to the
   same sha256 `eacec6c5...` at the same 838 bytes. Two stores on two volumes
   hold a byte-identical `config.json` at the same commit. How each store was
   populated was not measured, so this is corroboration by a second location,
   not by an independent download.
2. **A committed measurement artifact attests the layer count independently of
   any cache.** `results/s2-timing-pilots/gemma2-2b/run_1/config.json` records
   `model = google/gemma-2-2b-it`, `target_modules = [q_proj, k_proj, v_proj,
   o_proj]`, and `n_injected_modules = 104`. 104 = 26 x 4.
3. **The same procedure reproduces the Director's over-the-wire Qwen figures
   exactly** (below), so the instrument used for gemma is the one that already
   agrees with his independent fetch.

## qwen2.5-3b, qwen2.5-7b — recorded for symmetry; agree with the Director

Read from the same cache in the same pass. Both reproduce
`docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:24` ("hidden 2048, 36 layers" and
"hidden 3584, 28 layers") exactly.

=== VERIFIED STATE ===
qwen2.5-3b_model_id             = Qwen/Qwen2.5-3B-Instruct
qwen2.5-3b_snapshot_commit      = aa8e72537993ba99e69dfaafa59ed015b17504d1
qwen2.5-3b_config_sha256        = eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b
qwen2.5-3b_config_git_blob_sha1 = 51b83fbd4e2d363b60235b9c8d494046a83c1cbf
qwen2.5-3b_config_bytes         = 661
qwen2.5-3b_blob_name_verifies   = YES
qwen2.5-3b_model_type           = qwen2
qwen2.5-3b_hidden_size          = 2048
qwen2.5-3b_num_hidden_layers    = 36
qwen2.5-3b_intermediate_size    = 11008
qwen2.5-3b_num_attention_heads  = 16
qwen2.5-3b_num_key_value_heads  = 2
qwen2.5-3b_head_dim             = ABSENT from config.json (transformers derives hidden_size // num_attention_heads = 128)
qwen2.5-3b_vocab_size           = 151936
qwen2.5-3b_resolved_from        = C:\falco\hf-cache\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1\config.json
qwen2.5-7b_model_id             = Qwen/Qwen2.5-7B-Instruct
qwen2.5-7b_snapshot_commit      = a09a35458c702b33eeacc393d103063234e8bc28
qwen2.5-7b_config_sha256        = 7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c
qwen2.5-7b_config_git_blob_sha1 = 0178295f88afc3c7f279ed284f961f8c1be00654
qwen2.5-7b_config_bytes         = 663
qwen2.5-7b_blob_name_verifies   = YES
qwen2.5-7b_model_type           = qwen2
qwen2.5-7b_hidden_size          = 3584
qwen2.5-7b_num_hidden_layers    = 28
qwen2.5-7b_intermediate_size    = 18944
qwen2.5-7b_num_attention_heads  = 28
qwen2.5-7b_num_key_value_heads  = 4
qwen2.5-7b_head_dim             = ABSENT from config.json (transformers derives hidden_size // num_attention_heads = 128)
qwen2.5-7b_vocab_size           = 152064
qwen2.5-7b_resolved_from        = C:\falco\hf-cache\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\config.json
qwen2.5_resolved_at             = 2026-09-11
=== END VERIFIED STATE ===

Layer counts are attested a second time by the committed pilot configs:
`qwen2.5-3b/run_1/config.json` `n_injected_modules = 144` = 36 x 4;
`qwen2.5-7b/run_1/config.json` `n_injected_modules = 112` = 28 x 4.

## llama3.1-8b — cached on this workstation, NOT loaded by any S2 pilot

The Director confirmed this id's existence via its 401 gate error only, "not its
config" [`docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:24`]. Its config is on
disk, but it is a **workstation fact, not a program artifact**, and it is recorded
at that strength for three reasons: it sits in the default cache
(`F:\AI-Models\huggingface-cache\hub`), not the pilot cache; its pilots have never
run (`WS4_STATE_OF_PLAY_2026-09-01.md:236`, `S2_LLAMA31_8B_RATE = UNMEASURED`), so
nothing in this program ever loaded it; and on that volume the snapshot
`config.json` is a plain file rather than a blob named by its hash, so the
blob-name self-check available for the three families above does not exist here.
The hashes below are of the bytes as found.

=== VERIFIED STATE ===
llama3.1-8b_model_id             = meta-llama/Llama-3.1-8B-Instruct
llama3.1-8b_snapshot_commit      = 0e9e39f249a16976918f6564b8830bc894c89659
llama3.1-8b_config_sha256        = 29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e
llama3.1-8b_config_git_blob_sha1 = 0bb6fd75b3ad2fe988565929f329945262c2814e
llama3.1-8b_config_bytes         = 855
llama3.1-8b_blob_name_verifies   = NOT AVAILABLE (plain file on the F: volume, not a hash-named blob)
llama3.1-8b_model_type           = llama
llama3.1-8b_hidden_size          = 4096
llama3.1-8b_num_hidden_layers    = 32
llama3.1-8b_intermediate_size    = 14336
llama3.1-8b_num_attention_heads  = 32
llama3.1-8b_num_key_value_heads  = 8
llama3.1-8b_head_dim             = ABSENT from config.json (transformers derives hidden_size // num_attention_heads = 128)
llama3.1-8b_vocab_size           = 128256
llama3.1-8b_resolved_from        = F:\AI-Models\huggingface-cache\hub\models--meta-llama--Llama-3.1-8B-Instruct\snapshots\0e9e39f249a16976918f6564b8830bc894c89659\config.json
llama3.1-8b_resolved_at          = 2026-09-11
=== END VERIFIED STATE ===

## What this closes, and what it does not

**Closes:** the Ask 2 condition. Every id in `GATING.md` for a family whose pilots
ran now carries its resolved config, the revision it was read at, and a hash of
the bytes those numbers came from.

**Recorded but not claimed as closure:** `llama3.1-8b`, for the reasons in its own
section. Its access state remains `GATING.md`'s record, and `QUEUE.md`'s typed
block no longer conflicts with it: the same 2026-09-11 series rewrote that
block's `llama3.1-8b_access` to `OK (probed 2026-08-04, account timotheospaul,
asset1_bank.probe_family_access)` and names `GATING.md` as the probe of record.

**Downstream, not asserted here:** these widths are the inputs to any restatement
of r/hidden across the scale ladder
(`docs/AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md`). At the program's fixed rank
24 the cached widths give 24/2304, 24/2048 and 24/3584. No such restatement is
made in this file.
