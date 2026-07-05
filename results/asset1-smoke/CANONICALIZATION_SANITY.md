# Asset 1 — Canonicalization Sanity (smoke adapters)

Generated 2026-07-05T06:25:48.677950+00:00 by `scripts/asset1_canonicalize.py` (W2T QR-SVD canonicalization, arXiv:2603.15990 Section 3; bridge + alpha/rank scaling absorbed via the exact `into_B` strategy of `rhombic/nn/absorb.py` before decomposition, so spectra describe the TRUE effective DW).

Smoke context: 50-step runs with zero-initialized `lora_B`, so singular values are expected to be small but strictly positive — the sanity signal is a clean, rapidly decaying spectrum per module.

## `llama3.2-1b/alpaca/run_002/adapter_state.pt`

- modules canonicalized: **64**
- feature dims: sigma-only **1536**, full **50688** (proj_dim=16, proj_seed=0)
- top singular value across modules: max **0.179764**, median 0.076846, min 0.043900

| module (sample) | top-3 singular values |
|---|---|
| `model_layers_0_self_attn_q_proj` | 0.106320, 0.048061, 0.032510 |
| `model_layers_0_self_attn_k_proj` | 0.067214, 0.014872, 0.012946 |

## `llama3.2-1b/squad/run_003/adapter_state.pt`

- modules canonicalized: **64**
- feature dims: sigma-only **1536**, full **50688** (proj_dim=16, proj_seed=0)
- top singular value across modules: max **0.144440**, median 0.065897, min 0.031352

| module (sample) | top-3 singular values |
|---|---|
| `model_layers_0_self_attn_q_proj` | 0.092842, 0.045448, 0.036334 |
| `model_layers_0_self_attn_k_proj` | 0.054424, 0.021688, 0.019527 |

## `qwen2.5-1.5b/alpaca/run_000/adapter_state.pt`

- modules canonicalized: **112**
- feature dims: sigma-only **2688**, full **88704** (proj_dim=16, proj_seed=0)
- top singular value across modules: max **0.181379**, median 0.068504, min 0.025615

| module (sample) | top-3 singular values |
|---|---|
| `model_layers_0_self_attn_q_proj` | 0.094030, 0.045351, 0.032501 |
| `model_layers_0_self_attn_k_proj` | 0.037396, 0.019182, 0.012572 |

## `qwen2.5-1.5b/squad/run_001/adapter_state.pt`

- modules canonicalized: **112**
- feature dims: sigma-only **2688**, full **88704** (proj_dim=16, proj_seed=0)
- top singular value across modules: max **0.171826**, median 0.064011, min 0.022233

| module (sample) | top-3 singular values |
|---|---|
| `model_layers_0_self_attn_q_proj` | 0.087620, 0.049807, 0.030554 |
| `model_layers_0_self_attn_k_proj` | 0.050434, 0.015734, 0.012293 |
