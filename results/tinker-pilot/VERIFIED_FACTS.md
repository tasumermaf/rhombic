# Tinker pilot — verified facts (typed state)

Written at measurement time, per `context-engineering.md` §2a / XR-001: every
number here was read off an artifact or a live API response the moment it was
produced, and the PILOT_REPORT is assembled from these lines rather than from
conversational memory.

## Platform / account

```
TINKER_SDK_VERSION      = 0.24.0                         [pip, tinker env py3.12.13]
ACCOUNT_USER_URN        = tml:organization_user:be8fd2b8-df45-4db9-8f13-d711fe2242dc  [rest.whoami]
ACCOUNT_EMAIL           = timothy@promptcrafted.com       [rest.whoami]
N_SUPPORTED_MODELS      = 26                              [get_server_capabilities, 2026-07-30]
QWEN3_8B_SERVABLE       = True                            [get_server_capabilities]
PRICE_TRAIN_USD_PER_MTOK = 0.44                           [live pricing page, Qwen/Qwen3-8B, 2026-07-30]
PRICE_STORAGE_USD_PER_GB_MONTH = 0.10                     [live pricing page]
BILLING_ROWS_BEFORE_PILOT = 0                             [get_billing_usage, 13-day window]
BILLING_402_AT_FIRST_CALL = True (stale; training later succeeded unblocked)
```

## Loss normalization (MEASURED, not assumed)

The backend cross-entropy is an unnormalized SUM: `loss = sum(-logprob * w)`.
Measured on one 60-loss-token batch, same batch both ways:

```
LOSS_SUM_WEIGHTS_1.0    = 210.6964225769043               [forward_backward metrics loss:sum]
LOSS_SUM_WEIGHTS_1_OVER_N = 3.51160728931427              [forward_backward metrics loss:sum]
N_LOSS_TOKENS           = 60
RATIO_CHECK             = 210.6964225769043 / 60 = 3.5116070429...  (matches to 6 dp)
CONCLUSION              = weights carry the normalization; weights=1/N yields
                          MEAN per-token CE, keeping gradient scale independent
                          of batch size, which is what lr=1e-4 assumes.
```

## Loss convention (what the numbers ARE — checked, not assumed)

```
LOSS_DEFINITION         = mean per-token cross-entropy over the FULL, UNPADDED
                          sequence, prompt NOT masked
PADDING_USED            = none. Trainer calls tok.encode(text)[:512]; the
                          `padding` argument is never passed.
PAD_TOKENS_IN_SAMPLE    = 0   (200 seqs/task; pad_token_id 151643)
SAMPLE_MEAN_TOKEN_LEN   = alpaca 158.8 · math 204.2 · agnews 79.9
                          (matches the dry-run 161.9 / 196.1 / 78.8)
PADDED_LABEL_ARTIFACT   = RULED OUT
NOT_COMPARABLE_TO       = the Asset-1 bank finals (0.3703 Qwen2.5-1.5B,
                          0.3995 Llama-3.2-1B). That recipe sets
                          labels = input_ids over a FULLY PADDED sequence and
                          takes loss on every position including padding, so
                          its low finals are a padding effect, not the same
                          quantity. Cross-convention comparison is invalid.
MATH_NEAR_0.4_IS_REAL   = math finished 0.4088 / 0.4065, numerically close to
                          those padded finals but produced with zero padding.
                          Task spread (agnews 1.57 / alpaca 1.03 / math 0.41)
                          is task-characteristic, not a uniform collapse to
                          ~0.4 as an artifact would give.
```

## Locked run design

```
BASE_MODEL              = Qwen/Qwen3-8B
LORA_RANK               = 32                              (Tinker default)
TARGET_MODULES          = train_attn + train_mlp + train_unembed  (Tinker defaults)
BRIDGE                  = none (standard bridgeless PEFT LoRA)
LOSS_FN                 = cross_entropy, next-token shift
OPTIMIZER               = AdamW, lr 1e-4 constant, beta1 0.9, beta2 0.95,
                          eps 1e-12, weight_decay 0.0, grad_clip_norm 0.0
LR_JUSTIFICATION        = LoRA primer: optimal LoRA LR is ~10x full-FT and is
                          INDEPENDENT of rank, so 1e-4 stands at rank 32
MAX_SEQ_LEN             = 512, truncated, NOT padded (padding would be billed)
STEPS_PER_RUN           = 100
TOKENS_PER_STEP_TARGET  = 10000  (token-matched, so cost and gradient scale
                          match across tasks while sequences/step vary)
HARD_ABORT_USD          = 4.50
```

## Data provenance (Asset-1 locked machinery reused verbatim)

```
VAL_SEED = 777 · VAL_SIZE = 500 · POOL_CAP = 40000        [asset1_datasets]
alpaca  source = yahma/alpaca-cleaned   n_raw = 51760  pool = 40000  emitted = 15000
math    source = openai/gsm8k (main)    n_raw =  7473  pool =  6973  emitted =  6973
agnews  source = fancyzhx/ag_news       n_raw = 120000 pool = 40000  emitted = 15000
```
Per-stream sha256 recorded in `data/data_manifest.json`.

## Dry-run token/cost plan (local tokenizer, zero spend)

| run | sequences | train tokens | usd | mean tok/seq |
|---|---|---|---|---|
| alpaca_0 | 6,261 | 1,013,712 | 0.4460 | 161.9 |
| alpaca_1 | 6,204 | 1,012,688 | 0.4456 | 163.2 |
| math_0 | 5,152 | 1,010,503 | 0.4446 | 196.1 |
| math_1 | 5,164 | 1,011,173 | 0.4449 | 195.8 |
| agnews_0 | 12,730 | 1,003,742 | 0.4416 | 78.8 |
| agnews_1 | 12,726 | 1,004,046 | 0.4418 | 78.9 |

```
PLANNED_TOTAL_TRAIN_TOKENS = 6055864
PLANNED_TOTAL_USD          = 2.6646
POOL_DEPTH                 = sufficient for every run (no stream exhausted;
                             math is tightest at 5,152 of 6,973 available)
```

## Smoke run (alpaca seed 0)

```
SMOKE_STEPS             = 10 x ~5,000 tokens
SMOKE_TRAIN_TOKENS      = 51064                           [spend_ledger.json]
SMOKE_USD               = 0.02246816                      [spend_ledger.json]
SMOKE_TRAINING          = COMPLETED (no 402; compute unblocked)
SMOKE_CHECKPOINT_BYTES  = 369364358  (369.4 MB)           [list_user_checkpoints]
SMOKE_CHECKPOINT_PATH   = tinker://3d189c16-cb00-57ff-9c6f-5988902736f8:train:0/sampler_weights/alpaca_0-final
ADAPTER_FP32_PARAM_EST  = ~92.3M params (369.4 MB / 4 bytes) — consistent with
                          rank-32 LoRA over attn+mlp+unembed on Qwen3-8B
```

## Export-path validation

```
save_weights_for_sampler      = WORKS (checkpoint listed at 369.4 MB)
get_checkpoint_archive_url    = SLOW server-side build; first attempt raised
                                APITimeoutError at the SDK default 60s timeout,
                                AFTER the checkpoint was written (orphaning
                                billed storage). Client timeout raised to 1800s.
archive format                = plain tar containing a PEFT adapter +
                                'checkpoint_complete' marker   [tinker CLI source]
_extract validated            = tar, tar.gz, zip all extract
                                [adapter_config.json, adapter_model.safetensors,
                                checkpoint_complete]
_extract security             = '..' traversal REJECTED; symlink/hardlink
                                REJECTED; absolute path neutralised by
                                basename-flattening (stays inside dest)
```

## The six pilot runs (read from each run_record.json)

| run | task | seed | steps | train tokens | usd | sequences | train s | loss first | loss last | loss min |
|---|---|---|---|---|---|---|---|---|---|---|
| alpaca_0 | alpaca | 0 | 100 | 1,013,712 | 0.4460 | 6,261 | 187.8 | 1.4228 | 1.0284 | 0.9322 |
| alpaca_1 | alpaca | 1 | 100 | 1,012,688 | 0.4456 | 6,204 | 189.7 | 1.3807 | 1.0478 | 0.9268 |
| math_0 | math | 0 | 100 | 1,010,503 | 0.4446 | 5,152 | 198.9 | 1.2666 | 0.4088 | 0.3633 |
| math_1 | math | 1 | 100 | 1,011,173 | 0.4449 | 5,164 | 191.0 | 1.3923 | 0.4065 | 0.3686 |
| agnews_0 | agnews | 0 | 100 | 1,003,742 | 0.4416 | 12,730 | 190.0 | 2.8590 | 1.5731 | 1.5429 |
| agnews_1 | agnews | 1 | 100 | 1,004,046 | 0.4418 | 12,726 | 189.9 | 2.8530 | 1.6246 | 1.5440 |

```
SIX_RUN_TRAIN_TOKENS    = 6055864
SIX_RUN_USD             = 2.664579
DRYRUN_PREDICTION       = 6055864 tokens / $2.6646 — EXACT match (tokenization
                          is deterministic, so the plan priced the runs exactly)
TOTAL_WITH_SMOKE_TOKENS = 6106928                         [spend_ledger.json]
TOTAL_WITH_SMOKE_USD    = 2.687048                        [spend_ledger.json]
HARD_CAP_USD            = 4.50   (never approached; no abort triggered)
TRAIN_SECONDS_PER_RUN   = 187.8 - 198.9  (~3.2 min; ~1.9 s/step)
LOSS_IS_MEAN_PER_TOKEN_CE = yes, by the weights=1/N construction above
```

Loss separates by task both in level and in drop, with the two seeds of each
task landing on essentially the same value — the training-side hint that a
task-identity signal should exist in the weights:

```
alpaca  last loss 1.0284 / 1.0478      (spread 0.0194)
math    last loss 0.4088 / 0.4065      (spread 0.0023)
agnews  last loss 1.5731 / 1.6246      (spread 0.0515)
```

## Export latency (the operational finding)

```
ARCHIVE_BUILD_SMOKE     = ~54 min wall for one 369.4 MB rank-32 adapter
                          (first archive-URL request blocks while the server
                          builds; SDK default 60s timeout is far too short and
                          orphans the checkpoint)
SDK_INTERNAL_RETRIES    = the request keeps retrying internally past a 1800s
                          client timeout without surfacing a log line, so a
                          stalled export can block for hours
TINY_CHECKPOINT_PROBE   = a 15.46 MB rank-4 attn-only checkpoint ALSO entered
                          the same slow "Creating checkpoint archive" path,
                          so latency is not simply proportional to size
MITIGATION_ADOPTED      = --defer-export (train all runs, save checkpoints,
                          collect archives afterwards) + concurrent recovery
```

## Readout validation (synthetic, zero cost)

The GL(r) gauge invariance that the whole canonicalization claim rests on,
checked directly by replacing (B, A) with (B G, G^-1 A):

```
DW_PRESERVED_BY_GAUGE_RELDIFF = 1.841e-06   (float32 round-trip floor)
RAW_FEATURE_RELCHANGE         = 3.5611e+00   (raw features move: gauge-DEPENDENT)
CANONICAL_FEATURE_RELCHANGE   = 5.6893e-06   (canonical features do NOT move)
PLANTED_TASK_SIGNAL           = detected in both spaces, 1-NN accuracy 1.000
REPO_TESTS                    = tests/test_tinker_pilot_signal.py, 5 passed
REPO_TEST_COLLECTION          = 832 collected (was 827; +5 additive)
```

## Signal result on the real bank  [signal_results.json, signal_checks.json]

```
N_ADAPTERS              = 6   (3 tasks x 2 seeds)
N_MODULES_PER_ADAPTER   = 253  (q/k/v/o, gate/up/down x 36 layers, + unembed)
RAW_FEATURE_DIM         = 92286976
CANONICAL_FEATURE_DIM   = 267168   (full)   /   8096  (sigma)
SCALING                 = alpha/r = 1.0 for every run (lora_alpha 32, r 32)

RAW_MAX_WITHIN          = 1.028543
RAW_MIN_CROSS           = 0.057473
RAW_MARGIN              = -0.971071
RAW_SEPARATED           = False
RAW_NN_TASK_ACCURACY    = 0.000   (0/6; every NN is the same-SEED partner)

CANONICAL_MAX_WITHIN    = 0.991147
CANONICAL_MIN_CROSS     = 0.993282
CANONICAL_MARGIN        = +0.002135
CANONICAL_SEPARATED     = True
CANONICAL_NN_TASK_ACCURACY = 1.000  (6/6)

SIGNAL                  = MIXED  (raw False, canonical True — the predicted
                          pattern: "canonical separates; raw may not")
```

Mechanism of the raw failure (initialization dominance, not weak signal):

```
COS_A_SAME_SEED         = +0.966343 .. +0.971668   (n=6)
COS_A_DIFF_SEED         = -0.000633 .. -0.000182   (n=9)
NORM_B_OVER_NORM_A      =  0.28777 .. 0.37180
```

Robustness of the canonical separation (all separated, all 1-NN = 1.000):

| features | dim | max within | min cross | margin | relabeling p |
|---|---|---|---|---|---|
| full, proj_seed 0 | 267,168 | 0.991147 | 0.993282 | +0.002135 | 0.0667 |
| full, proj_seed 1 | 267,168 | 0.994067 | 0.994762 | +0.000695 | 0.0667 |
| full, proj_seed 7 | 267,168 | 0.993796 | 0.995620 | +0.001824 | 0.0667 |
| sigma | 8,096 | 0.005632 | 0.011849 | +0.006217 | 0.0667 |

```
RELABELING_NULL         = 15 perfect matchings of 6 runs into 3 pairs;
                          1 at least as separating -> p = 1/15 = 0.0667.
                          This is the FLOOR of a 2-seed design, not a weak
                          effect: no 6-adapter experiment can reach p<0.05
                          under this null. The exchangeable-pair-label
                          alternative (1/C(15,3)=0.0022) is NOT used: the 15
                          distances come from only 6 points and are not
                          exchangeable.
EXPORT_CLEANUP          = 0 remote checkpoints remaining (verified) -> no
                          storage billing
```

## Billing reconciliation (partial; usage API lags a few hours)

```
BILLED_TRAINING_TOKENS_SMOKE = 51244    [get_billing_usage]
LEDGER_TRAINING_TOKENS_SMOKE = 51064    [spend_ledger.json]
DELTA                        = +180 (+0.35%, meter higher than the estimate)
BILLED_CHECKPOINT_TOKENS     = 0
SIX_RUN_ROWS                 = not yet posted at time of writing
IMPLICATION                  = the in-code guard under-estimates by ~0.35%
                               (ledger counts len(seq)-1 per sequence); add a
                               safety factor in the mini-bank guard rather
                               than treating the estimate as exact.
```
