# E-T4 Tinker Mini-Bank — 54 adapters on a third-party stack

**WS-4 (TASUMER MAF) · 2026-08-04 · exploratory, no locks, no confirmatory claims**

**Scope, verbatim from the ruled card and binding on every artifact derived
from this bank:** pilot-scale, single family, single recipe, descriptive; the
registered local cards (H2-at-scale, granularity) remain the program's
confirmatory instruments.

Executed under `docs/CARD_DRAFT_ET4_MINIBANK_2026-07-30.md` as ruled by the
Director on 2026-08-04: **N = 9, 54 runs**, exploratory classification
blessed. Every number below is quoted from an artifact in this directory
(`TYPED_CORE.md`, `spend_ledger*.json`, `*/run_record.json`,
`signal_results.json`, `controlled_contrast.json`, `merge_lint_results.json`,
`billing_usage.json`); none is restated from narrative. The typed core is
generated from those artifacts by `scripts/tinker_minibank_report.py` and is
reproduced whole in `TYPED_CORE.md`.

---

## 1. Headline

```
BANK                  = 54/54 adapters, 6 tasks x 3 data seeds x 3 init seeds
BASE_MODEL            = Qwen/Qwen3-8B, rank 32, 253 modules, bridgeless PEFT
ACCOUNT_TOTAL_EST     = $27.0752   (stop $27.50, hard abort $28.00)
HEADROOM_TO_STOP      = $0.4248
CROSS_INIT_TASK  raw            = 0.167  (chance 0.167, p = 0.4453)   AT CHANCE
CROSS_INIT_TASK  canonical_full = 1.000  (chance 0.167, p = 0.0004998)
CROSS_INIT_TASK  canonical_sigma= 1.000  (chance 0.167, p = 0.0004998)
```

**The pilot's MIXED verdict survives at n = 54, but only under a controlled
contrast — and the naive metric would have reversed it.** Plain leave-one-out
1-NN on this bank reports raw task accuracy **1.000**, the opposite of the
pilot's 0/6. That number is an artifact of the design, not a finding, and §4
explains why. Once the nearest neighbour is forced to come from a *different
initialization*, raw task identification sits **exactly at chance** while both
canonical representations hold at **1.000**.

---

## 2. What was run

54 rank-32 LoRA adapters on **Qwen/Qwen3-8B** — the full Asset-1 task set
(alpaca, code, math, xsum, squad, agnews) crossed 3 data seeds x 3 init seeds.
Pilot config held fixed; `adapter_config.json` from an exported adapter
confirms `r=32`, `lora_alpha=32`, `use_rslora=false`, `target_modules=all-linear`,
`lora_dropout=0`, peft 0.18.1.

The three PILOT_REPORT §6 bindings were adopted, not paraphrased:

1. **§6.1 — data seed and init seed vary independently, both recorded.** The
   pilot's single `seed` set both, and its raw-space null was entirely an
   initialization effect. Here `data_seed` selects the emitted text stream and
   `init_seed` is passed to `create_lora_training_client(seed=...)`. Init seeds
   are shared across tasks, so init identity is a 3-class nuisance label with
   18 adapters each, scored exactly like task identity. **This binding is what
   makes §4 possible**; without it the bank could not have distinguished the
   two hypotheses at all.
2. **§6.2 — `sigma` carried as a headline representation**, alongside `full`,
   not as a robustness footnote.
3. **§6.3 — export wall clock budgeted.** Deferred export during training,
   parallel archive recovery after. Module set held at all 253: the pilot's
   separation used all of them, so dropping mlp/unembed would be a design
   change to validate, not assume.

Loss-weight normalization (`w = 1 / loss-tokens in batch`) and `lr = 1e-4`
come from `scripts/tinker_pilot_train.py` **by import, not by copy**, so the
measured facts behind them cannot drift.

### Data provenance

`scripts/tinker_minibank_data.py` extends the pilot's three tokenizer-free
tasks to all six by calling the locked `asset1_datasets.py` registry classes
directly — `asset1_datasets.py` itself is untouched. The dataset objects are
built with `cls.__new__` to skip the bank's eager padded tokenization (Tinker
needs unpadded text; padded positions would be billed), and are given only the
two attributes their formatter needs, so `format_example` / `_fit_document` /
`_post_load` run verbatim off the registry class.

Two independent checks that the delegation introduced no drift:

* **Byte-identity.** The six streams shared with the pilot
  (alpaca/math/agnews x seeds 0,1) are **byte-identical** to the pilot
  emitter's output — 0 mismatches (`--verify-against-pilot`).
* **Token-identity.** The free dry-run reproduces the pilot's per-run token
  counts exactly: alpaca_0 1,013,712; math_0 1,010,503; math_1 1,011,173;
  agnews_0 1,003,742; agnews_1 1,004,046.

18 streams, sha256 per stream in `data/data_manifest.json`. A property test
(`tests/test_tinker_minibank.py`) additionally asserts the `__new__` shortcut
reproduces the real `Asset1TaskDataset.__init__` text on xsum and squad, where
`_fit_document` budgeting is most likely to depend on skipped state.

---

## 3. Cost and budget

```
KEPT_RUN_TOKENS       = 54,606,009      [the 54 exported runs]
METERED_TOKENS_TOTAL  = 55,034,280      [all 5 ledgers, 54 completed entries]
WASTED_ON_RESTARTS    =    428,271 tok = $0.1891
USD_METER_ONLY        = $24.2151        [metered x $0.44/M]
USD_BILLED_EST        = $24.2998        [x1.0035 measured meter factor]
PRIOR_ACCOUNT_SPEND   = $2.7754         [audited via get_billing_usage]
ACCOUNT_TOTAL_EST     = $27.0752
STOP_REPORT_USD       = $27.50          HARD_ABORT_USD = $28.00
HEADROOM_TO_STOP      = $0.4248
```

Price re-verified on the live models & pricing page on 2026-08-04 before the
first run: **Qwen3-8B training $0.44/M, storage $0.10/GB-month** — unchanged
from the pilot. The prior-spend figure was re-derived from `get_billing_usage`
across the key's whole life and reconciles to the card's audit exactly
(6,107,108 Qwen3-8B tokens Jul 30 + 59,460 Qwen3.5-9B + 304 survey Aug 3).

The free dry-run priced 54 runs at **54,606,009 tokens / $26.8861 account
total**; the 54 kept runs consumed exactly that. Tokenization is deterministic,
so the plan is the bill — now confirmed a third time (pilot, this dry-run,
this bank). The $0.1891 excess over plan is the restart waste in §6.

### Billing reconciliation — PARTIAL, and outstanding

```
BANK_BILLED_TOKENS    = 36,477,535   [all-time billed minus the 6,166,872 pre-bank baseline]
BANK_METERED_TOKENS   = 55,034,280   [our ledgers]
BILLING_POSTED_FRAC   = 0.663
BILLED_CHECKPOINT_EV  = 53
BILLED_STORAGE_GBH    = 1.6926 = $0.0002
```

The usage API lags real time by hours and had posted only 66.3% of the bank's
tokens at write time, so **full reconciliation is outstanding** — the same
condition the pilot reported. Two further caveats stated rather than hidden:
the API returns metered quantities only, no dollar amounts, so every dollar
here is quantity x the published price table and the billing page remains the
authority; and the storage line has not yet posted the ~19.2 GB that sat in
remote storage during the deferred-export window (bounded above by roughly
$0.011 at $0.10/GB-month, immaterial against the $0.4248 headroom but not zero).

---

## 4. The readout — and why the plain metric is not the result

Three feature spaces over the same 54 adapters, built by
`scripts/tinker_minibank_signal.py`:

* **raw** — flattened `lora_A` then `lora_B` per module, sorted module order;
  gauge-DEPENDENT, dimension **92,286,976**.
* **canonical_full** / **canonical_sigma** — the exact r-slot SVD of
  `DW = (alpha/r) B @ A`, computed without materializing `DW`, via the
  **identical QR->SVD utilities and sign convention imported from
  `scripts/asset1_canonicalize.py`**. Bridge absorption is the identity
  (E = I). Dimensions 267,168 and 8,096.

Raw features are never materialized: the Gram matrix is accumulated
module-major in float64 over column chunks, which is exact and bounded
(54 x 92M floats would be ~20 GB). A property test asserts this equals the
naive Gram of the fully concatenated vectors.

**Readout validation before any mini-bank number was believed.** The same
readout, run unchanged on the *pilot* bank, reproduces the pilot's published
figures to six decimals in all three spaces — raw max_within 1.028543 /
min_cross 0.057473 / 1-NN 0.000; canonical_full margin +0.002135 / 1.000;
sigma margin +0.006217 / 1.000; raw_dim 92,286,976.

### The confound

Plain LOO 1-NN over this bank reports:

| space | task | init_seed | data_seed |
|---|---|---|---|
| raw | 1.000 | 1.000 | 0.000 |
| canonical_full | 1.000 | 0.981 | 0.019 |
| canonical_sigma | 1.000 | 1.000 | 0.000 |

Read naively this says raw carries task identity at n = 54, reversing the
pilot. It does not. The design places 3 adapters (the three data seeds) in
every `(task, init)` cell, and those three share **both** labels. An adapter's
nearest neighbour is therefore almost always a cell-mate, and a cell-mate
match is simultaneously same-task and same-init. The metric cannot attribute
the success to either label — which is exactly why raw scores 1.000 on task
*and* 1.000 on init at once.

### The controlled contrast (`scripts/tinker_minibank_controlled.py`)

Restricting the neighbourhood so the labels come apart, geometry and mask held
fixed under a 2,000-draw label-permutation null:

| space | contrast | accuracy | chance | perm p |
|---|---|---|---|---|
| raw | **cross_init_task** | **0.167** | 0.167 | 0.4453 |
| raw | within_init_task | 1.000 | 0.118 | 0.0004998 |
| raw | cross_task_init | 1.000 | 0.333 | 0.0004998 |
| canonical_full | **cross_init_task** | **1.000** | 0.167 | 0.0004998 |
| canonical_full | within_init_task | 1.000 | 0.118 | 0.0004998 |
| canonical_full | cross_task_init | 0.704 | 0.333 | 0.0004998 |
| canonical_sigma | **cross_init_task** | **1.000** | 0.167 | 0.0004998 |
| canonical_sigma | within_init_task | 1.000 | 0.118 | 0.0004998 |
| canonical_sigma | cross_task_init | 0.852 | 0.333 | 0.0004998 |

Read across the rows:

* **Raw carries no task identity across initializations.** 0.167 against a
  chance of 0.167, p = 0.4453 over 54 scored adapters. Raw space encodes
  *which random init this adapter started from* — it identifies init identity
  across tasks at 1.000 — and resolves task only with init held fixed. This is
  the pilot's mechanism (`cos(A,A)` +0.966..+0.972 same-seed vs ~0 diff-seed;
  `||B||/||A||` 0.288..0.372) reproduced at nine times the scale and now
  measured as a factor rather than inferred from a confound.
* **Both canonical representations carry task identity independently of
  initialization**, at 1.000 with p at the 1/2001 permutation floor.
* **Canonicalization does not fully erase init identity.** cross_task_init is
  0.704 (full) and 0.852 (sigma) against chance 0.333 — reduced from raw's
  1.000, but well above chance. The GL(r) gauge is removed; the initialization
  imprint is only attenuated. That is an honest limitation of this
  representation at 1M tokens, and it is a finding the pilot could not have
  produced.

The pilot's strict global criterion (max within-task < min cross-task) is also
reported in `signal_results.json` for continuity: at n = 54 it holds only for
`canonical_sigma` on task (margin +0.001074; raw -1.001144, canonical_full
-0.009142). With 9 adapters per class the strict criterion is a far harsher
test than at n = 2 per class, and LOO accuracy is the informative statistic at
this scale; both are reported rather than the flattering one alone.

---

## 5. merge_lint — and an interface finding

`scripts/tinker_minibank_merge_lint.py` pairs adapters **vertex-disjointly**
(each adapter in at most one pair, so the pairs are independent draws rather
than an overlapping web) across **different tasks**, and invokes
`scripts/merge_lint.py` unmodified.

```
MERGE_LINT_PAIRS      = 27 (vertex-disjoint, cross-task)
MERGE_LINT_IN_FAMILY  = 0
MERGE_LINT_EXITS      = {'0': 27}
BRIDGELESS_REFUSAL    = exit 2
```

**Finding: merge_lint as shipped refuses a bridgeless adapter.** Omitting the
bridge key — which `asset1_canonicalize.effective_factors` documents as
degrading gracefully to E = I — exits 2 with `REFUSED: module
'base_model.model.model.layers.0.mlp.down_proj' has no 'bridge' tensor
(include requested 'bridge')`, because `asset1_analysis_io.flatten_features`
defaults to `include=("A","B","bridge")` and is stricter than the absorption
path it feeds. This is a real gap between two components of our own tooling,
surfaced by running the linter on genuinely foreign input, and it is recorded
here rather than silently worked around.

To obtain verdicts the bridge is written as the **exact** bridgeless identity:
`_expand_bridge(I_C, rank // C) == I_rank` for any C dividing rank, so `C = 1`
with `bridge = [[1.0]]` reproduces E = I exactly at minimal footprint.
Measured consequences, on a real pair rather than assumed:

* `l2_distance` and every per-module L2 are **exactly unchanged** — identical
  bridge blocks cancel in the difference (measured delta 3.6e-15).
* `cos_distance` **is** perturbed, because `flatten_features` concatenates raw
  stored parameters and a constant identity block adds a common component:
  true bridgeless 0.065207769 -> 0.060392835 (**-4.8e-03**). The equally valid
  C = 32 encoding would shift it -4.7e-02, ten times further, which is why
  C = 1 is used. Each pair's true bridgeless cos/l2 is recorded beside
  merge_lint's own values in `merge_lint_results.json`.

All 27 pairs land **out-of-family** (`module set matches NO bank family`) and
fall back to the pooled distance-only model with the linter's own
EXTRAPOLATION banner. The conflict probabilities (min 0.0, max 1.0, mean
0.7407 over 27 pairs) are therefore **uncalibrated for this input** and are
reported as a demonstration that the refusal/fallback discipline works on
foreign adapters — **not** as merge predictions. Treating them as predictions
would be exactly the error the linter's own banner warns against.

---

## 6. Operational record

**Throughput is the headline operational surprise, and it inverts §6.3.** The
pilot's export problem did not reproduce: archive builds ran **19.7 s**, not
the pilot's contended 29-54 min. Training became the bottleneck instead, and
was highly variable — 182.3 s to 1792.3 s per run (mean 731.5 s). The slowest
runs coincided with heavy *local* contention (a full pytest suite, the pilot
revalidation, repeated 369 MB safetensors loads on the same machine); an
uncontended run took 220.6 s, faster than the smoke.

**Silent stalls, never exceptions.** Across every log: **zero API errors, zero
retries, zero 429/5xx**. The stalls were requests that simply stopped
progressing with no exception raised — precisely PILOT_REPORT §5 trap #2 ("do
not treat silence as progress"). Because nothing raises, exception handling is
the wrong detector; an mtime watchdog is the right one. An earlier "SHARD
ISSUE ... 429" alert in this session was a **false positive** — the monitor's
grep matched `429` inside the dollar amount `$23.6429`.

**Sharding.** After the first 5 runs the remainder was split round-robin
across 3 concurrent processes (round-robin, not contiguous, so an early stop
leaves a bank balanced across tasks and seeds). Concurrency and a hard cap
interact badly if done casually — processes sharing one ledger race, each
reads a stale cumulative, and the guard silently stops guarding. Instead each
shard carried its own ledger, an envelope of plan x 1.01, and a
`prior_spend_usd` including the other shards' envelopes, with the safety
identity asserted before launch and refused otherwise:
`PRIOR 2.7754 + SPENT 2.2371 + ENVELOPES 22.0924 = 27.1049 <= STOP 27.50`.
The kill that preceded resharding was taken **between** runs; the ledger after
it read exactly 5 x ~1.013M tokens, i.e. zero partial waste.

**Shard 2 restart history, and the one refused run.** Shard 2's log went stale
twice and was restarted by the supervisor at 15:04:32 and 15:14:06. A third
firing never came, so the standing §6.3 escalation rule was not triggered.
Those two restarts stranded **428,271 tokens ($0.1891)** of partial-run spend
inside shard 2's ledger — spend that is billed but produced no adapter. That
overage exhausted shard 2's envelope, and its 54th run was **refused** by its
own guard:

```
HARD ABORT: agnews_d2_i2 (whole run, 1,004,561 tok): this shard's projected
spend $7.7784 would exceed its envelope $7.6652. ABORTING.
```

This is the per-shard envelope working as intended — strictly tighter than the
account cap, so it bit while the account still held $0.87 of headroom.
`agnews_d2_i2` was then re-run alone against the true account state
(`--prior-usd 26.6317`, envelope $0.50): 1,004,561 tokens, $0.4436, loss
2.8961 -> 1.5727, exported and remote checkpoint deleted, account
**$27.0753**.

**Export and storage hygiene.** One archive transfer truncated under 6-way
concurrency (`agnews_d0_i1`, RemoteProtocolError at 277,117,538 of
369,223,680 bytes). No retrain was needed: `fetch_and_extract` raises before
the delete step, so the checkpoint survived, and the pilot's §5 finding that
archive builds persist server-side and are cached held — a **single-worker**
retry completed it in 19 s, supporting concurrency as the cause.

Final verification, in this order (local integrity first, remote deletion
confirmed only afterwards):

```
RUN_DIRS                     = 54
ADAPTER_SIZES (distinct)     = {369,216,072: 54}
TENSOR_COUNTS (distinct)     = {506: 54}
MODULE_COUNTS (distinct)     = {253: 54}
INTEGRITY_FAILURES           = NONE      [header parsed AND a tensor decoded per adapter]
REMOTE_CHECKPOINTS_REMAINING = 0         [storage billing goes to zero]
LOCAL_BANK_SIZE              = 19 GB
```

---

## 7. What is verified here, and what is taken on report

**Verified in this session, by me, against artifacts:** every dollar and token
figure (recomputed from the five ledgers and 54 run records at write time);
the live price; the prior-spend audit; the byte- and token-identity of the
data streams; the readout's reproduction of the pilot's published numbers; all
54 adapters' size, tensor count, module count and decodability; zero remaining
remote checkpoints; the merge_lint refusal and the measured bridge-encoding
deltas; zero API errors across all logs.

**Taken on report, not independently re-derived by me:** the Director's ruling
of N = 9 and the exploratory classification; the pilot's own measured
constants that this bank inherits by import (the loss-normalization
measurement, the LoRA-primer LR claim); and the published price table, since
the usage API returns no dollar amounts.

**Outstanding:** full billing reconciliation (66.3% posted at write time), and
the storage line for the deferred-export window.

**Caveats carried forward.** Exploratory; n = 54; one model family; one rank;
one token budget; one recipe. The cross-init contrast is the informative
statistic and rests on 54 scored adapters with 3 init seeds — enough to
separate the two hypotheses cleanly here, not enough to characterize how the
effect scales with init count, rank, or token budget. Canonicalization
attenuates but does not remove the initialization imprint. And the
plain-LOO/controlled-contrast gap in §4 is itself the caution: on a bank with
crossed factors, an unrestricted nearest-neighbour metric can report the
opposite of the truth.

---

## 8. Artifacts

```
scripts/tinker_minibank_data.py           6-task training-text emitter (falco env)
scripts/tinker_minibank_train.py          trainer: independent seeds, account guard, sharding
scripts/tinker_minibank_launch_shards.py  envelope arithmetic + assert-and-refuse launcher
scripts/tinker_minibank_supervise.sh      stall watchdog with the >2-firings escalation rule
scripts/tinker_minibank_signal.py         readout (falco env)
scripts/tinker_minibank_controlled.py     the controlled cross-init contrast
scripts/tinker_minibank_merge_lint.py     bank-format bridge + vertex-disjoint pairing
scripts/tinker_minibank_billing.py        usage-API audit (tinker env)
scripts/tinker_minibank_report.py         typed-core generator
tests/test_tinker_minibank.py             7 acceptance tests
results/tinker-minibank/TYPED_CORE.md     generated typed state
results/tinker-minibank/spend_ledger*.json      5 ledgers
results/tinker-minibank/<run>/run_record.json   54 records
results/tinker-minibank/signal_results.json     plain readout, 3 spaces x 3 labels
results/tinker-minibank/controlled_contrast.json  the decisive figures
results/tinker-minibank/pairwise_distances.json   all 1,431 pairs x 3 spaces
results/tinker-minibank/merge_lint_results.json   27 pairs + refusal probe
results/tinker-minibank/billing_usage.json        usage-API audit
results/tinker-minibank/data/data_manifest.json   18 streams + sha256
results/tinker-minibank/logs/                     smoke, single-process and shard logs
```

Adapter payloads (369,216,072 B each, ~19 GB) and the 202 MB of training text
are gitignored per the repo convention for the Asset-1 bank, the S2 pilots and
the Tinker pilot: regenerable or bulk payloads stay on disk, the measurements
are the deliverable and are tracked.

Test suite with this work added: **838 passed, 1 skipped**.
