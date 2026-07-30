# Tinker Signal Pilot — E-T4 pre-step

**WS-4 (TASUMER MAF) · 2026-07-30 · exploratory, no locks, no confirmatory claims**

Purpose: before paying for a 60-run mini-bank, establish whether **task identity
is legible from the weights of adapters trained on Tinker at ~1M train tokens
per run** — and measure what a Tinker run actually costs in dollars and wall
clock. Every number below is quoted from an artifact in this directory
(`spend_ledger.json`, `*/run_record.json`, `signal_results.json`,
`signal_checks.json`, `VERIFIED_FACTS.md`); none is restated from narrative.

---

## 1. Headline

```
SIGNAL                 = MIXED  (pre-specified: raw fails, canonical separates)
RAW_SEPARATED          = False   1-NN task accuracy 0.000  (0/6)
CANONICAL_SEPARATED    = True    1-NN task accuracy 1.000  (6/6)
TOTAL_SPEND_USD        = 2.687048   (cap was $4.50; project cap $5)
TOTAL_TRAIN_TOKENS     = 6106928
RECOMMENDATION         = PROCEED to the 60-run mini-bank, with two design
                         changes and one operational fix (§6)
```

**MIXED is the outcome the readout program predicts.** Raw LoRA parameters do
not carry task identity here; the gauge-canonical representation does, at 1M
tokens, on a third-party stack we do not control. That is the external-validity
result E-T4 was designed to look for.

---

## 2. What was run

Six rank-32 LoRA adapters on **Qwen/Qwen3-8B**, 3 tasks x 2 seeds, one run per
(task, seed), plus a smoke run.

```
BASE_MODEL             = Qwen/Qwen3-8B          [26 models servable; verified]
LORA_RANK              = 32                     (Tinker default)
TARGET_MODULES         = all-linear + unembed   (Tinker defaults; 253 modules)
                         q/k/v/o, gate/up/down, unembed_tokens
BRIDGE                 = none — standard bridgeless PEFT LoRA
lora_alpha             = 32  ->  scaling = alpha/r = 1.0, use_rslora = false
LOSS                   = cross_entropy, next-token shift
OPTIMIZER              = AdamW lr 1e-4 constant (beta1 0.9, beta2 0.95,
                         eps 1e-12, wd 0.0)
MAX_SEQ_LEN            = 512, truncated, NOT padded
DESIGN                 = 100 optim steps x ~10,000 train tokens = ~1M tokens
```

Data reuses the **locked Asset-1 machinery verbatim** — `split_ids` with
`VAL_SEED=777`, `VAL_SIZE=500`, `POOL_CAP=40000`, and the same
`### Instruction: / ### Response:` templates (`scripts/tinker_pilot_data.py`).
Per-stream sha256 in `data/data_manifest.json`.

**Token-matched, not sequence-matched.** Each step is filled with whole
sequences until its token budget is met, so every run consumes the same train
tokens — the same cost and the same gradient scale — while sequences per step
vary by task (agnews sequences are short, math long). This is why the sequence
counts differ so much below at near-identical token counts.

### Two things that had to be measured, not assumed

**Loss normalization.** The backend cross-entropy is an **unnormalized sum**,
`loss = sum(-logprob * w)`. Measured on one 60-loss-token batch, both ways on
the same batch:

```
weights = 1.0    -> loss:sum = 210.6964225769043
weights = 1/N    -> loss:sum = 3.51160728931427
210.6964225769043 / 60 = 3.5116070...   (matches)
```

So the weights carry the normalization. Every datum uses
`w = 1 / (loss tokens in the batch)`, making the reported loss the mean
per-token CE and keeping the gradient scale independent of batch size — which
is what `lr=1e-4` assumes. Passing `weights=1.0` with a ~10,000-token batch
would have applied a gradient ~10^4 too large and silently ruined all six runs.

**Learning rate.** The LoRA primer states the optimal LoRA LR is ~10x full-FT
and is **independent of rank**, so 1e-4 stands unchanged at rank 32.

**What the reported loss is, exactly — and what it must not be compared to.**
The loss here is the **mean per-token cross-entropy over the full, unpadded
sequence, with the prompt NOT masked**. Three consequences, because each one is
a live way to misread the table in §3:

* It is **not** the Asset-1 bank convention. That recipe sets
  `labels = input_ids` over a *fully padded* sequence and takes loss on every
  position including padding, which is why its finals sit near 0.37-0.40 (most
  positions are padding, and padding is trivially predictable). Those numbers
  are **not comparable** to the values below.
* It is **not** completion-only loss either, since the prompt is unmasked.
* The padded-label artifact is **ruled out by construction and by check**: the
  trainer calls `tok.encode(text)[:512]` and never passes a `padding` argument,
  so no pad positions exist. Verified on a 200-sequence sample per task —
  **0 pad tokens found** (mean lengths 158.8 alpaca / 204.2 math / 79.9 agnews,
  matching the dry-run estimates).

This matters specifically because `math` finished at 0.4088 / 0.4065, which
lands in the same numeric range as the padded-bank finals and could be mistaken
for the artifact. It is not: with zero padding possible, the low value is real
learning on a rigidly templated task (GSM8K's step-by-step format plus a fixed
instruction line), and the spread across tasks — agnews 1.57, alpaca 1.03,
math 0.41 — is task-characteristic rather than the uniform collapse toward ~0.4
that a convention artifact would produce.

---

## 3. Per-run results

| run | task | seed | steps | train tokens | usd | sequences | train s | loss first | loss last | loss min |
|---|---|---|---|---|---|---|---|---|---|---|
| alpaca_0 | alpaca | 0 | 100 | 1,013,712 | 0.4460 | 6,261 | 187.8 | 1.4228 | 1.0284 | 0.9322 |
| alpaca_1 | alpaca | 1 | 100 | 1,012,688 | 0.4456 | 6,204 | 189.7 | 1.3807 | 1.0478 | 0.9268 |
| math_0 | math | 0 | 100 | 1,010,503 | 0.4446 | 5,152 | 198.9 | 1.2666 | 0.4088 | 0.3633 |
| math_1 | math | 1 | 100 | 1,011,173 | 0.4449 | 5,164 | 191.0 | 1.3923 | 0.4065 | 0.3686 |
| agnews_0 | agnews | 0 | 100 | 1,003,742 | 0.4416 | 12,730 | 190.0 | 2.8590 | 1.5731 | 1.5429 |
| agnews_1 | agnews | 1 | 100 | 1,004,046 | 0.4418 | 12,726 | 189.9 | 2.8530 | 1.6246 | 1.5440 |

```
SIX_RUN_TRAIN_TOKENS   = 6055864
SIX_RUN_USD            = 2.664579
SMOKE_TRAIN_TOKENS     = 51064          SMOKE_USD = 0.02246816
TOTAL_TRAIN_TOKENS     = 6106928        TOTAL_USD = 2.687048
PRICE_TRAIN_USD_PER_MTOK = 0.44         [live pricing page, Qwen3-8B]
COST_PER_RUN           = $0.4416 - $0.4460   (measured; matches the ESTIMATE)
WALL_PER_RUN_TRAINING  = 187.8 - 198.9 s  (~3.2 min, ~1.9 s/step)
BUDGET_GUARD           = $4.50, never approached, no abort triggered
```

The dry-run priced the six runs at **6,055,864 tokens / $2.6646** and they
consumed **exactly** that — tokenization is deterministic, so the plan is the
bill. The §4 ESTIMATE of ~$0.44/run in the assessment is now **measured**.

**Billing reconciliation (partial — billing lags real time by a few hours).**
At time of writing the usage API had posted only the smoke run:

```
BILLED_TRAINING_TOKENS (smoke)  = 51244        [get_billing_usage]
LEDGER_TRAINING_TOKENS (smoke)  = 51064        [spend_ledger.json]
DELTA                           = +180  (+0.35%, billing higher)
BILLED_CHECKPOINT_TOKENS        = 0
```

The ledger counts `len(seq) - 1` per sequence (the forward length after the
next-token shift); the meter counts slightly more. So **the in-code guard
under-estimates by ~0.35%** — immaterial here ($2.687 estimated implies ~$2.696
billed against a $4.50 cap), but the mini-bank guard should carry a small safety
factor rather than assume the estimate is exact. The six runs' rows had not yet
posted, so full reconciliation is outstanding.

Loss separates by task in both level and drop, with the two seeds of a task
landing on nearly the same value — the training-side hint that a weight-space
task signal should exist:

```
alpaca  last loss 1.0284 / 1.0478   (spread 0.0194)
math    last loss 0.4088 / 0.4065   (spread 0.0023)
agnews  last loss 1.5731 / 1.6246   (spread 0.0515)
```

---

## 4. The signal check

Two feature spaces over the same six exported adapters
(`scripts/tinker_pilot_signal.py`, 253 modules each, raw dim 92,286,976,
canonical dim 267,168):

* **raw** — flattened `lora_A` then `lora_B` per module, sorted module order.
  Gauge-DEPENDENT: `(B G, G^-1 A)` leaves the effective update unchanged for any
  `G` in GL(r) but moves this vector arbitrarily.
* **canonical** — the exact r-slot SVD of the effective update
  `DW = scaling * B @ A`, computed without materializing `DW`, via the
  **identical QR->SVD utilities and sign convention imported from
  `scripts/asset1_canonicalize.py`**. Bridgeless adaptation: the RhombiLoRA
  bridge-absorption step becomes the identity (`E = I`), leaving only the
  scalar `alpha/r`. Nothing about the canonicalization was reimplemented.

`SIGNAL = YES` was pre-defined as *every same-task pair closer than every
cross-task pair, in both spaces*; MIXED if exactly one space separates.

### Raw: fails, and fails informatively

```
max_within_task = 1.028543     min_cross_task = 0.057473
margin          = -0.971071    separated = False
1-NN task accuracy = 0.000  (0/6)
```

Every adapter's nearest neighbour in raw space is its **same-seed,
different-task** partner:

| run | raw nearest neighbour | distance | same task |
|---|---|---|---|
| agnews_0 | alpaca_0 | 0.065218 | no |
| agnews_1 | alpaca_1 | 0.062577 | no |
| alpaca_0 | math_0 | 0.057473 | no |
| alpaca_1 | math_1 | 0.057993 | no |
| math_0 | alpaca_0 | 0.057473 | no |
| math_1 | alpaca_1 | 0.057993 | no |

**Mechanism, quantified** (`signal_checks.json`). Tinker's `seed` sets the LoRA
*initialization*. `lora_A` is randomly initialized and stays large while
`lora_B` starts at zero, so after only 1M tokens the learned part is a minority
of the raw vector:

```
cos(A, A) same-seed pairs  = +0.966343 .. +0.971668   (n=6)
cos(A, A) diff-seed pairs  = -0.000633 .. -0.000182   (n=9)
||B|| / ||A||              =  0.28777 .. 0.37180
```

Raw space therefore encodes *which random init this adapter started from*, not
what it learned. Task identity is not weakly present in raw space — it is
**dominated**, 0/6. This is the same failure class as the GL(r) gauge (a nuisance
transform swamping the signal), arriving through initialization rather than
through a basis change.

### Canonical: separates, on every variant tried

```
max_within_task = 0.991147     min_cross_task = 0.993282
margin          = +0.002135    separated = True
1-NN task accuracy = 1.000  (6/6)
```

Every adapter's canonical nearest neighbour is its same-task partner. Because
the `full` margin is thin, the separation was re-checked under three probe seeds
and under the `sigma` variant (log1p singular values only — fully
rotation-invariant, **no random probes at all**):

| features | dim | max within | min cross | margin | separated | 1-NN | relabeling p |
|---|---|---|---|---|---|---|---|
| full, proj_seed 0 | 267,168 | 0.991147 | 0.993282 | +0.002135 | True | 1.000 | 0.0667 |
| full, proj_seed 1 | 267,168 | 0.994067 | 0.994762 | +0.000695 | True | 1.000 | 0.0667 |
| full, proj_seed 7 | 267,168 | 0.993796 | 0.995620 | +0.001824 | True | 1.000 | 0.0667 |
| **sigma** | 8,096 | 0.005632 | 0.011849 | **+0.006217** | True | 1.000 | 0.0667 |

The `sigma` variant is the cleanest reading: within-task distance 0.005632 vs
cross-task 0.011849, a **2.1x ratio** with no random projection involved. The
separation is not a probe artifact.

### Inference — and the honest ceiling of a 6-adapter design

The observed task grouping is the **most separating of all 15 possible ways to
partition 6 adapters into 3 pairs**, for every feature variant. Enumerating that
relabeling null exactly:

```
n_matchings = 15    n_at_least_as_separating = 1    p = 1/15 = 0.0667
```

`p = 0.0667` is the **floor this design can reach**, not a weak effect: with 2
seeds per task there are only 15 pairings, so no 6-adapter experiment can
produce p < 0.05 under this null however clean the geometry. (A naive
exchangeable-pair-label null would give 1/C(15,3) = 0.0022, but the 15 distances
are determined by only 6 points and are not exchangeable, so the relabeling
null is the correct and stricter one.) Perfect ordering + 1-NN 1.000 + stability
across four feature variants is the strongest statement available at this scale,
and the resolution argument is precisely what the 60-run mini-bank buys.

---

## 5. Operational findings (new, and load-bearing for E-T4 scheduling)

**Training is fast and cheap; export is the bottleneck.**

```
TRAINING_PER_RUN       = ~3.2 min, $0.44
ADAPTER_SIZE           = 369.4 MB (fp32, 92,286,976 params over 253 modules)
ARCHIVE_BUILD_SERIAL   = ~54 min for one adapter (contended)
ARCHIVE_BUILD_PARALLEL = 1728 - 1752 s (~29 min) each, 6 concurrent
TOTAL_EXPORT_WALL      = ~29 min for all 6 (6 workers) vs ~3 h serial
```

Three concrete traps, all now fixed in `scripts/tinker_pilot_train.py`:

1. **The SDK's default 60s HTTP timeout is far too short.** The first archive
   request raised `APITimeoutError` *after* the checkpoint was written, orphaning
   369 MB of billed storage. Client timeout raised to 1800s.
2. **A stalled archive request retries internally without surfacing anything.**
   It blocked past the 1800s timeout with no log line, and can hang for hours.
   Do not treat silence as progress.
3. **Archive builds persist server-side and are cached.** A checkpoint whose
   build had been abandoned earlier returned in **13 s** on a later request.
   So a timeout is recoverable — retry, do not retrain.

Mitigations adopted: `--defer-export` (train everything, save checkpoints, then
collect archives), `--recover` with concurrent workers, training records
flushed to disk *before* export so a slow archive can never lose paid-for
measurements, and non-fatal export errors so one slow archive cannot kill a
batch. All remote checkpoints were downloaded and **deleted** — verified 0
remaining, so storage billing is zero.

Latency is not simply proportional to size: a 15.46 MB rank-4 attention-only
checkpoint entered the same slow build path.

---

## 6. Recommendation: PROCEED, with two design changes

The pilot's stop condition ("if task signal is not cleanly present at 1M
tokens, stop and rescope") is **not** triggered: signal is present, perfectly
ordered, and stable across feature variants. Proceed to the 60-run mini-bank
(6 tasks x 10 seeds ~= $27 at the measured $0.44/run — the measured rate
confirms the budget). Three changes:

1. **Vary the LoRA init seed independently of the data seed, and record both.**
   The raw-space result is entirely an initialization effect. In the mini-bank,
   crossing data seed with init seed turns this confound into a *measurable
   factor* and lets the readout be tested against init identity as a named
   nuisance label — a stronger and more interesting claim than task identity
   alone. As run here, seed conflates both.
2. **Carry the `sigma` variant as a headline representation, not just `full`.**
   It separated with the largest relative margin (2.1x) using no random
   projection, making it the most defensible figure for a Featured Project
   write-up.
3. **Budget wall clock for export, not just dollars.** 60 adapters x 369 MB is
   ~22 GB and, at ~29 min per parallel build, hours of transfer. Consider
   `train_mlp=False`/`train_unembed=False` to cut adapter size ~3x if the
   readout does not need mlp/unembed modules — a downward-cost deviation that
   does not change token spend. Note this pilot's separation used all 253
   modules; dropping modules is a design change to validate, not assume.

**Caveats to carry forward.** Exploratory, n=6, one family, one token budget,
one rank; `p = 0.0667` is the design floor; the canonical `full` margins are
thin (+0.0007 to +0.0021) even though the ordering is perfect; and MIXED here
means raw failed — a Featured Project write-up must present that as the finding
it is, not bury it.

---

## 7. Artifacts

```
scripts/tinker_pilot_data.py           training-text emitter (falco env)
scripts/tinker_pilot_train.py          trainer + budget guard + export/recover
scripts/tinker_pilot_signal.py         the readout (falco env)
scripts/tinker_pilot_signal_checks.py  mechanism + robustness + permutation null
tests/test_tinker_pilot_signal.py      5 tests: loader, planted signal,
                                       DW preservation, GL(r) invariance
results/tinker-pilot/VERIFIED_FACTS.md typed state, written at measure time
results/tinker-pilot/spend_ledger.json cumulative token/dollar ledger
results/tinker-pilot/<run>/run_record.json  per-run measurements
results/tinker-pilot/signal_results.json    15 pairwise distances, both spaces
results/tinker-pilot/signal_checks.json     mechanism + robustness + null
results/tinker-pilot/data/data_manifest.json  stream provenance + sha256
```

Adapter payloads (369 MB each) and the 45 MB of training text are gitignored
per the repo convention for the Asset-1 bank and S2 pilots: regenerable or bulk
payloads stay on disk, the measurements are the deliverable and are tracked.

**Readout validation, before any real data was touched** — the property the
whole canonicalization claim rests on, checked by replacing `(B, A)` with
`(B G, G^-1 A)` on synthetic adapters:

```
DW preserved by the gauge      rel 1.841e-06   (float32 round-trip floor)
RAW features move              rel 3.5611e+00
CANONICAL features do NOT move rel 5.6893e-06
repo tests                     5 passed; collection 832 (was 827)
```
