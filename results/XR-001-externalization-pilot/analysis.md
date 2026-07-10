# XR-001 - Externalization-Robustness Pilot: Analysis

Protocol: `results/XR-001-externalization-pilot/PROTOCOL.md`. Confirmatory test = pre-registered paired McNemar (R1 vs R2); everything else is descriptive.

## Numeric-corruption rate (per model x regime, 95% Wilson CI)

| Model | Regime | n | corrupt | rate | 95% CI |
|---|---|---|---|---|---|
| gemma3:4b | R0 | 120 | 8 | 6.7% | [3.4%, 12.6%] |
| gemma3:4b | R1 | 120 | 44 | 36.7% | [28.6%, 45.6%] |
| gemma3:4b | R2 | 120 | 17 | 14.2% | [9.0%, 21.5%] |
| qwen3-coder:30b | R0 | 120 | 0 | 0.0% | [0.0%, 3.1%] |
| qwen3-coder:30b | R1 | 120 | 35 | 29.2% | [21.8%, 37.8%] |
| qwen3-coder:30b | R2 | 120 | 3 | 2.5% | [0.9%, 7.1%] |
| qwen3:14b | R0 | 120 | 8 | 6.7% | [3.4%, 12.6%] |
| qwen3:14b | R1 | 120 | 52 | 43.3% | [34.8%, 52.3%] |
| qwen3:14b | R2 | 120 | 14 | 11.7% | [7.1%, 18.6%] |

## Regime x class breakdown (pooled across models)

signature = off_by_one + conflation (P2 union class, protocol amendment 1)

| Regime | correct | off_by_one | conflation | other_wrong | omission | signature |
|---|---|---|---|---|---|---|
| R0 | 344 | 1 | 3 | 12 | 0 | 4 |
| R1 | 229 | 1 | 16 | 84 | 30 | 17 |
| R2 | 326 | 1 | 3 | 25 | 5 | 4 |

## Paired exact McNemar - R1 vs R2 (confirmatory)

Pooled across models: b (R1 correct, R2 wrong) = 11, c (R2 correct, R1 wrong) = 108, two-sided exact p = 3.524e-21, direction = R2_better.

| Model | b (R1>R2) | c (R2>R1) | p | direction |
|---|---|---|---|---|
| gemma3:4b | 5 | 32 | 7.428e-06 | R2_better |
| qwen3-coder:30b | 3 | 35 | 6.678e-08 | R2_better |
| qwen3:14b | 3 | 41 | 1.618e-09 | R2_better |

## Mean realized compaction tokens (matched-budget check)

| Regime | n calls | mean eval_count |
|---|---|---|
| R1 | 180 | 219.2 |
| R2 | 180 | 221.4 |

## Multi-hop completion (questions 7-8)

| Model | Regime | n | correct | rate |
|---|---|---|---|---|
| gemma3:4b | R0 | 30 | 22 | 73.3% |
| gemma3:4b | R1 | 30 | 9 | 30.0% |
| gemma3:4b | R2 | 30 | 15 | 50.0% |
| qwen3-coder:30b | R0 | 30 | 30 | 100.0% |
| qwen3-coder:30b | R1 | 30 | 14 | 46.7% |
| qwen3-coder:30b | R2 | 30 | 30 | 100.0% |
| qwen3:14b | R0 | 30 | 22 | 73.3% |
| qwen3:14b | R1 | 30 | 6 | 20.0% |
| qwen3:14b | R2 | 30 | 16 | 53.3% |
