=== VERIFIED STATE: E-T4 Tinker mini-bank ===

```
N_RUNS_TRAINED        = 54
N_ADAPTERS_DOWNLOADED = 54
TASKS                 = ['agnews', 'alpaca', 'code', 'math', 'squad', 'xsum']
DATA_SEEDS            = [0, 1, 2]
INIT_SEEDS            = [0, 1, 2]
BASE_MODEL            = Qwen/Qwen3-8B
LORA_RANK             = 32
LEARNING_RATE         = 0.0001
MAX_SEQ_LEN           = 512

KEPT_RUN_TOKENS       = 54,606,009   [the 54 exported runs]
METERED_TOKENS_TOTAL  = 55,034,280   [all 5 ledgers; 54 completed entries]
WASTED_ON_RESTARTS    = 428,271 tokens = $0.1891   [shard-2 watchdog restarts; billed, no adapter]
USD_METER_ONLY        = 24.2151   [metered x $0.44/M]
USD_BILLED_EST        = 24.2998   [x1.0035 meter factor]
USD_KEPT_RUNS_ONLY    = 24.1107   [what 54 clean runs would have cost]
PRIOR_ACCOUNT_SPEND   = 2.7754   [audited via get_billing_usage]
ACCOUNT_TOTAL_EST     = 27.0752
STOP_REPORT_USD       = 27.5
HARD_ABORT_USD        = 28.0
HEADROOM_TO_STOP      = 0.4248

WALL_SECONDS_PER_RUN  = 183.6 - 1795.2 (mean 743.0)
TRAIN_SECONDS_PER_RUN = 182.3 - 1792.3 (mean 731.5)

BILLED_TRAIN_TOKENS   = 42,644,407   [get_billing_usage, all models]
BILLED_TRAINING_USD   = 18.7372   [priced models]
BILLED_STORAGE_GBH    = 1.6926 = $0.0002
BILLED_CHECKPOINT_EV  = 53
BANK_BILLED_TOKENS    = 36,477,535   [all-time billed minus the 6,166,872 pre-bank baseline]
BANK_METERED_TOKENS   = 55,034,280   [our ledgers]
BILLING_POSTED_FRAC   = 0.663   [<1.0 means the usage API has not finished posting; it lags real time by hours, so full reconciliation is OUTSTANDING]

DATA_STREAMS          = 18
DATA_TOKENIZER        = Qwen/Qwen3-8B

MERGE_LINT_PAIRS      = 27 (vertex-disjoint, cross-task)
MERGE_LINT_IN_FAMILY  = 0
MERGE_LINT_EXITS      = {'0': 27}
BRIDGELESS_REFUSAL    = exit 2
```

## Readout (plain LOO — CONFOUNDED, see below)

```
N_ADAPTERS = 54
TASKS = ['agnews', 'alpaca', 'code', 'math', 'squad', 'xsum']
RAW_DIM = 92,286,976
N_PERM = 2000

| space | label | classes | LOO-1NN | chance | LOO-3NN | LOO-5NN | perm p (1NN) | separated |
|---|---|---|---|---|---|---|---|---|
| raw | task | 6 | 1.000 | 0.017 | 1.000 | 0.222 | 0.0004998 | False |
| raw | init_seed | 3 | 1.000 | 0.018 | 1.000 | 1.000 | 0.0004998 | True |
| raw | data_seed | 3 | 0.000 | 0.018 | 0.000 | 0.000 | 1 | False |
| canonical_full | task | 6 | 1.000 | 0.017 | 1.000 | 1.000 | 0.0004998 | False |
| canonical_full | init_seed | 3 | 0.981 | 0.018 | 0.981 | 1.000 | 0.0004998 | False |
| canonical_full | data_seed | 3 | 0.019 | 0.018 | 0.093 | 0.111 | 1 | False |
| canonical_sigma | task | 6 | 1.000 | 0.017 | 1.000 | 1.000 | 0.0004998 | True |
| canonical_sigma | init_seed | 3 | 1.000 | 0.018 | 1.000 | 0.500 | 0.0004998 | False |
| canonical_sigma | data_seed | 3 | 0.000 | 0.018 | 0.000 | 0.000 | 1 | False |
```

## Controlled contrast (the decisive figures)

```
N_PERM = 2000

| space | contrast | accuracy | chance | perm p |
|---|---|---|---|---|
| raw | cross_init_task | 0.167 | 0.167 | 0.4453 |
| raw | within_init_task | 1.000 | 0.118 | 0.0004998 |
| raw | cross_task_init | 1.000 | 0.333 | 0.0004998 |
| canonical_full | cross_init_task | 1.000 | 0.167 | 0.0004998 |
| canonical_full | within_init_task | 1.000 | 0.118 | 0.0004998 |
| canonical_full | cross_task_init | 0.704 | 0.333 | 0.0004998 |
| canonical_sigma | cross_init_task | 1.000 | 0.167 | 0.0004998 |
| canonical_sigma | within_init_task | 1.000 | 0.118 | 0.0004998 |
| canonical_sigma | cross_task_init | 0.852 | 0.333 | 0.0004998 |
```

## Per-run

| run | task | data_seed | init_seed | steps | train tokens | usd (billed est) | train s | wall s | loss first | loss last | adapter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews_d0_i0 | agnews | 0 | 0 | 100 | 1,003,742 | 0.4432 | 601.4 | 602.7 | 2.8586 | 1.5746 | yes |
| agnews_d0_i1 | agnews | 0 | 1 | 100 | 1,003,742 | 0.4432 | 634.2 | 640.7 | 2.8581 | 1.5737 | yes |
| agnews_d0_i2 | agnews | 0 | 2 | 100 | 1,003,742 | 0.4432 | 184.7 | 185.9 | 2.8590 | 1.5742 | yes |
| agnews_d1_i0 | agnews | 1 | 0 | 100 | 1,004,046 | 0.4433 | 1383.1 | 1395.3 | 2.8538 | 1.6252 | yes |
| agnews_d1_i1 | agnews | 1 | 1 | 100 | 1,004,046 | 0.4433 | 1169.8 | 1180.9 | 2.8538 | 1.6248 | yes |
| agnews_d1_i2 | agnews | 1 | 2 | 100 | 1,004,046 | 0.4433 | 203.3 | 204.6 | 2.8530 | 1.6251 | yes |
| agnews_d2_i0 | agnews | 2 | 0 | 100 | 1,004,561 | 0.4436 | 732.3 | 736.1 | 2.8961 | 1.5728 | yes |
| agnews_d2_i1 | agnews | 2 | 1 | 100 | 1,004,561 | 0.4436 | 245.6 | 247.1 | 2.8969 | 1.5724 | yes |
| agnews_d2_i2 | agnews | 2 | 2 | 100 | 1,004,561 | 0.4436 | 193.3 | 540.1 | 2.8961 | 1.5727 | yes |
| alpaca_d0_i0 | alpaca | 0 | 0 | 100 | 1,013,712 | 0.4476 | 299.3 | 320.7 | 1.4228 | 1.0292 | yes |
| alpaca_d0_i1 | alpaca | 0 | 1 | 100 | 1,013,712 | 0.4476 | 798.9 | 800.4 | 1.4230 | 1.0282 | yes |
| alpaca_d0_i2 | alpaca | 0 | 2 | 100 | 1,013,712 | 0.4476 | 220.6 | 224.2 | 1.4228 | 1.0285 | yes |
| alpaca_d1_i0 | alpaca | 1 | 0 | 100 | 1,012,688 | 0.4471 | 238.7 | 240.0 | 1.3807 | 1.0479 | yes |
| alpaca_d1_i1 | alpaca | 1 | 1 | 100 | 1,012,688 | 0.4471 | 248.4 | 249.5 | 1.3807 | 1.0477 | yes |
| alpaca_d1_i2 | alpaca | 1 | 2 | 100 | 1,012,688 | 0.4471 | 283.6 | 285.7 | 1.3807 | 1.0476 | yes |
| alpaca_d2_i0 | alpaca | 2 | 0 | 100 | 1,013,636 | 0.4476 | 197.9 | 201.2 | 1.4352 | 1.0280 | yes |
| alpaca_d2_i1 | alpaca | 2 | 1 | 100 | 1,013,636 | 0.4476 | 1705.0 | 1717.4 | 1.4353 | 1.0281 | yes |
| alpaca_d2_i2 | alpaca | 2 | 2 | 100 | 1,013,636 | 0.4476 | 753.7 | 758.5 | 1.4350 | 1.0279 | yes |
| code_d0_i0 | code | 0 | 0 | 100 | 1,006,204 | 0.4443 | 1536.7 | 1549.0 | 1.2182 | 0.7327 | yes |
| code_d0_i1 | code | 0 | 1 | 100 | 1,006,204 | 0.4443 | 638.0 | 640.8 | 1.2183 | 0.7329 | yes |
| code_d0_i2 | code | 0 | 2 | 100 | 1,006,204 | 0.4443 | 815.6 | 825.9 | 1.2182 | 0.7330 | yes |
| code_d1_i0 | code | 1 | 0 | 100 | 1,005,326 | 0.4439 | 551.8 | 553.5 | 1.2302 | 0.7193 | yes |
| code_d1_i1 | code | 1 | 1 | 100 | 1,005,326 | 0.4439 | 923.7 | 925.7 | 1.2302 | 0.7194 | yes |
| code_d1_i2 | code | 1 | 2 | 100 | 1,005,326 | 0.4439 | 476.6 | 477.7 | 1.2302 | 0.7190 | yes |
| code_d2_i0 | code | 2 | 0 | 100 | 1,006,614 | 0.4445 | 294.7 | 303.2 | 1.2603 | 0.7487 | yes |
| code_d2_i1 | code | 2 | 1 | 100 | 1,006,614 | 0.4445 | 1438.9 | 1453.7 | 1.2603 | 0.7485 | yes |
| code_d2_i2 | code | 2 | 2 | 100 | 1,006,614 | 0.4445 | 593.3 | 594.9 | 1.2603 | 0.7485 | yes |
| math_d0_i0 | math | 0 | 0 | 100 | 1,010,503 | 0.4462 | 1605.5 | 1615.7 | 1.2662 | 0.4087 | yes |
| math_d0_i1 | math | 0 | 1 | 100 | 1,010,503 | 0.4462 | 324.2 | 326.5 | 1.2667 | 0.4082 | yes |
| math_d0_i2 | math | 0 | 2 | 100 | 1,010,503 | 0.4462 | 1644.2 | 1652.7 | 1.2671 | 0.4087 | yes |
| math_d1_i0 | math | 1 | 0 | 100 | 1,011,173 | 0.4465 | 403.4 | 404.5 | 1.3933 | 0.4082 | yes |
| math_d1_i1 | math | 1 | 1 | 100 | 1,011,173 | 0.4465 | 375.9 | 377.6 | 1.3933 | 0.4064 | yes |
| math_d1_i2 | math | 1 | 2 | 100 | 1,011,173 | 0.4465 | 598.6 | 603.0 | 1.3933 | 0.4070 | yes |
| math_d2_i0 | math | 2 | 0 | 100 | 1,011,185 | 0.4465 | 284.4 | 285.6 | 1.2878 | 0.3630 | yes |
| math_d2_i1 | math | 2 | 1 | 100 | 1,011,185 | 0.4465 | 632.8 | 635.0 | 1.2880 | 0.3631 | yes |
| math_d2_i2 | math | 2 | 2 | 100 | 1,011,185 | 0.4465 | 557.0 | 566.4 | 1.2878 | 0.3623 | yes |
| squad_d0_i0 | squad | 0 | 0 | 100 | 1,010,513 | 0.4462 | 1792.3 | 1795.2 | 2.3910 | 1.6187 | yes |
| squad_d0_i1 | squad | 0 | 1 | 100 | 1,010,513 | 0.4462 | 686.8 | 689.1 | 2.3910 | 1.6182 | yes |
| squad_d0_i2 | squad | 0 | 2 | 100 | 1,010,513 | 0.4462 | 184.2 | 186.4 | 2.3899 | 1.6163 | yes |
| squad_d1_i0 | squad | 1 | 0 | 100 | 1,011,395 | 0.4466 | 1661.8 | 1672.0 | 2.3399 | 1.7371 | yes |
| squad_d1_i1 | squad | 1 | 1 | 100 | 1,011,395 | 0.4466 | 191.0 | 192.7 | 2.3403 | 1.7383 | yes |
| squad_d1_i2 | squad | 1 | 2 | 100 | 1,011,395 | 0.4466 | 1449.2 | 1457.7 | 2.3399 | 1.7372 | yes |
| squad_d2_i0 | squad | 2 | 0 | 100 | 1,011,800 | 0.4467 | 661.3 | 666.7 | 2.3899 | 1.7233 | yes |
| squad_d2_i1 | squad | 2 | 1 | 100 | 1,011,800 | 0.4467 | 182.3 | 183.6 | 2.3907 | 1.7229 | yes |
| squad_d2_i2 | squad | 2 | 2 | 100 | 1,011,800 | 0.4467 | 676.9 | 685.5 | 2.3907 | 1.7239 | yes |
| xsum_d0_i0 | xsum | 0 | 0 | 100 | 1,022,814 | 0.4516 | 493.5 | 495.2 | 2.6241 | 2.0726 | yes |
| xsum_d0_i1 | xsum | 0 | 1 | 100 | 1,022,814 | 0.4516 | 484.6 | 486.0 | 2.6234 | 2.0735 | yes |
| xsum_d0_i2 | xsum | 0 | 2 | 100 | 1,022,814 | 0.4516 | 569.7 | 572.0 | 2.6234 | 2.0731 | yes |
| xsum_d1_i0 | xsum | 1 | 0 | 100 | 1,022,377 | 0.4514 | 261.4 | 262.8 | 2.5844 | 2.1470 | yes |
| xsum_d1_i1 | xsum | 1 | 1 | 100 | 1,022,377 | 0.4514 | 1343.9 | 1354.1 | 2.5841 | 2.1462 | yes |
| xsum_d1_i2 | xsum | 1 | 2 | 100 | 1,022,377 | 0.4514 | 1343.0 | 1354.2 | 2.5841 | 2.1466 | yes |
| xsum_d2_i0 | xsum | 2 | 0 | 100 | 1,019,714 | 0.4502 | 555.2 | 556.3 | 2.6290 | 2.1713 | yes |
| xsum_d2_i1 | xsum | 2 | 1 | 100 | 1,019,714 | 0.4502 | 1788.3 | 1794.8 | 2.6293 | 2.1706 | yes |
| xsum_d2_i2 | xsum | 2 | 2 | 100 | 1,019,714 | 0.4502 | 1383.9 | 1395.0 | 2.6293 | 2.1708 | yes |
