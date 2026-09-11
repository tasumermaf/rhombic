# LOCKED: H2-D (lineage distance at matched scale) — Lock Declaration

**LOCKED by the decider (Meridian) on 2026-09-11 on the Director's ruling of
2026-08-04** (decider authority per the PI's 2026-08-04 delegation, the same
authority under which `docs/LOCK_GRANULARITY_2026-08-04.md:3` was declared; plan
of record `C:\falco\docs\MERIDIAN_PROGRAM_PLAN_2026-08-04.md`, framing at
`docs/DECIDER_RULINGS_GRANULARITY_AMBIGUITIES_2026-08-04.md:4-10`). The locked
object is **H2-D only**, on the three measured families, within the registered
card `docs/REGISTRATION_H2_SCALE_2026-07-30.md` (= `PREREG_DRAFT_H2_SCALE_2026-07-21.md`
+ `DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md` Part 4; where they
conflict the rulings govern, `REGISTRATION_H2_SCALE_2026-07-30.md:8`). From this
moment H2-D changes only by dated amendment (L-006), never by silent revision.

The Director's ruling, verbatim:

> "**lock the three measured families for H2-D (lineage distance at matched
> scale), and hold H2-S (within-lineage scale) pending the rank-fraction control
> in §3.** Do not carry Llama-3.1-8B as unmeasured, and do not substitute
> Mistral-7B"
> — `docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:22` (Ask 1)

restated in his Net as "**H2: lock H2-D on the three measured families; hold
H2-S**" (`docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:95`).

**This lock authorizes no bank run.** S1's GPU-day commitment remains the
Director's call (`docs/REGISTRATION_H2_SCALE_2026-07-30.md:14-17`, `:36-37`), and
`results/asset1h2-bank/` is correctly absent from disk (verified 2026-09-11).

**Citation convention.** Every constant below is read from a file in this
repository at **HEAD = `6ca9c4c`** (`6ca9c4c910b904ab465208cbb7ef97ab2beec2a8`),
or from the workstation HF cache where the line says so. Nothing here is restated
from memory or from a prior summary. Three cited files are exceptions, and none
of them carries a locked H2-D constant:

- `scripts/asset1_d1_identifiability.py` (item **A14**, the H2-S padding and
  occupancy clause) and `results/s2-timing-pilots/QUEUE.md` (item **A13**
  sub-item 2) were under active working-tree edit on the date of this
  declaration; their line numbers are cited **at `6ca9c4c`, not at the working
  tree**. See the dated note under §6's table for the rows this affects.
- `results/s2-timing-pilots/RESOLVED_CONFIGS.md` **does not exist at HEAD
  `6ca9c4c`**. It was filed 2026-09-11 by item **A13 sub-item 1** and is
  committed in this same series ahead of this declaration, so its line
  citations resolve at the commit that adds it, not at `6ca9c4c`.

---

## Lock-condition checklist, each against its artifact

| Condition | Status | Evidence |
|---|---|---|
| Lock condition 1 — S2 measured rates published | ✅ 3 of 4 families | `results/s2-timing-pilots/RATES.md:19-24`; condition text at `REGISTRATION_H2_SCALE_2026-07-30.md:36-37`; Director re-derived all nine rates exact (`DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:14`) |
| Lock condition 2 — §3 cost table restated against measured rates | ✅ | `docs/S2_COST_RESTATEMENT_2026-08-04.md:37-44`; self-declared satisfying at `:121` (condition 1's own self-declaration is `:119-120`); recomputed this pass (see §5) |
| Director disposition on H2 | ✅ ruled | `DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:22`, `:95` |
| Ask 2 — model-id provenance: resolved config recorded, not just the string | ✅ **closed** | Ask at `DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:24`; ids at `results/s2-timing-pilots/GATING.md:10-13`; dedicated closure artifact `results/s2-timing-pilots/RESOLVED_CONFIGS.md` (revisions + config.json hashes, filed 2026-09-11 under A13 sub-item 1); resolved configs re-read independently for this lock in §2 below, **agreeing with it value for value**; per-run attestation at `results/s2-timing-pilots/<family>/run_1/TIMING.md:6`, `:31`, `:32` |
| >25% overrun trigger (drop option) | not fired | every measured family under estimate, worst −4.42% (`S2_COST_RESTATEMENT_2026-08-04.md:39-48`) |
| Llama-3.1-8B not carried as unmeasured | ✅ excluded | ruling `:22`; `results/s2-timing-pilots/RATES.md:25` (`llama3.1-8b_mean_min_per_run = —`); no `results/s2-timing-pilots/llama3.1-8b/` directory exists and no weights are cached in either store — only the access probe's `config.json` (verified 2026-09-11; see §8) |
| H2-S excluded from this lock | ✅ HELD | ruling `:22`, `:95`; `AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md:106` |
| H2-V excluded from this lock | ✅ out of scope | corroborating everywhere per S7(d), `REGISTRATION_H2_SCALE_2026-07-30.md:24-26`; counts as its own pre-registered test outside the Holm family (`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:49`) |

---

## 1. What is locked, in one line

The **family roster, the resolved model configurations, the training recipe, the
run counts, the seeds policy, the decision rule, the multiplicity family, the
interlock topology, and the measured cost basis** for H2-D on the three measured
families. The **confirmatory H2-D test set under this lock is P4 alone**
(Gemma-2-2B ↔ Llama-3.2-1B), for the reason given in §7(i).

---

## 2. Family roster and resolved configurations (Director Ask 2)

Model ids are Meridian's resolution of the card's family labels, published at
`results/s2-timing-pilots/GATING.md:10-13` and attested per run at
`results/s2-timing-pilots/<family>/run_1/TIMING.md:6`. The Director confirmed the
two Qwen ids independently from live configs ("hidden 2048, 36 layers" /
"hidden 3584, 28 layers", `DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:24`); the
cached values below match his fetch exactly. `google/gemma-2-2b-it` was the one
id he could not verify (401 anonymously) and placed "on your report" with the
instruction to "confirm it from the workstation cache … and record the resolved
config alongside the id, not just the string" (`:24`). **That condition is closed**
— by the dedicated artifact `results/s2-timing-pilots/RESOLVED_CONFIGS.md` (filed
2026-09-11 under A13 sub-item 1, carrying snapshot revisions and `config.json`
byte hashes) and by the independent re-read below, which agrees with it value for
value. Both records are needed: that file is the closure the Director asked for,
this table is the constant the lock freezes.

Configs read 2026-09-11 for this declaration from
`C:\falco\hf-cache\hub\models--<org>--<name>\snapshots\<revision>\config.json`
(the cache `GATING.md:26` names as required on every model-load step).

| short | model_id | cached snapshot revision | model_type | hidden_size | layers | attn heads | kv heads | head_dim | vocab | dtype |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma2-2b | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | gemma2 | **2304** | **26** | 8 | 4 | 256 | 256000 | bfloat16 |
| qwen2.5-3b | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | qwen2 | 2048 | 36 | 16 | 2 | (128 derived) | 151936 | bfloat16 |
| qwen2.5-7b | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | qwen2 | 3584 | 28 | 28 | 4 | (128 derived) | 152064 | bfloat16 |

Anchor legs, already collected in the Asset-1 bank and not retrained by this card
(roster at `scripts/asset1_bank.py:95-96`):

| short | model_id | cached snapshot revision | hidden_size | layers | attn heads | kv heads | head_dim |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 1536 | 28 | 12 | 2 | (128 derived) |
| llama3.2-1b | `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | 2048 | 16 | 32 | 8 | 64 |

Measured per-family facts from the pilots (each read from that family's
`run_1/TIMING.md`, lines `:31` and `:32`):

| short | base params measured | injected modules | layers × 4 target modules | measured min/run (n=3) |
|---|---|---|---|---|
| gemma2-2b | 2,614,341,888 | 104 | 26 × 4 = 104 ✓ | **75.51** (`RATES.md:19-20`) |
| qwen2.5-3b | 3,085,938,688 | 144 | 36 × 4 = 144 ✓ | **82.86** (`RATES.md:21-22`) |
| qwen2.5-7b | 7,615,616,512 | 112 | 28 × 4 = 112 ✓ | **153.01** (`RATES.md:23-24`) |

`gemma2-2b` also carries `attn_implementation = sdpa` and
`model_dtype = torch.bfloat16` (`gemma2-2b/run_1/TIMING.md:33-34`), peak VRAM
14.07 GB of 48 (`RATES.md:7-9`).

---

## 3. Design constants, frozen at lock

The card pins "Training recipe identical to Asset-1"
(`PREREG_DRAFT_H2_SCALE_2026-07-21.md:55-58`), so every constant below is the
Asset-1 constant, read in source. **Each line is the locked value; the file and
line are where it is read.**

| Constant | Locked value | Source |
|---|---|---|
| `TASKS` | alpaca, code, math, xsum, squad, agnews (6) | `scripts/asset1_bank.py:98` |
| `MAX_STEPS` | 2000 | `scripts/asset1_bank.py:104` |
| `RANK` | 24 | `scripts/asset1_bank.py:105` |
| `N_CHANNELS` | 6 | `scripts/asset1_bank.py:106` |
| `LORA_ALPHA` | 16.0 | `scripts/asset1_bank.py:107` |
| `BRIDGE_MODE` | identity | `scripts/asset1_bank.py:108` |
| `TARGET_MODULES` | q_proj, k_proj, v_proj, o_proj | `scripts/asset1_bank.py:109` |
| `BATCH_SIZE` | 4 | `scripts/asset1_bank.py:110` |
| `GRAD_ACCUM` | 4 (effective batch 16) | `scripts/asset1_bank.py:111-112` |
| cohort tag | `bs4xga4` (mandatory, S3) | `scripts/asset1_bank.py:652`; ruling `DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:47` |
| `LR` | 2e-4 | `scripts/asset1_bank.py:113` |
| `WEIGHT_DECAY` | 0.01 | `scripts/asset1_bank.py:114` |
| `GRAD_CLIP` | 1.0 | `scripts/asset1_bank.py:115` |
| `WARMUP_STEPS` | 100 | `scripts/asset1_bank.py:116` |
| `MAX_LEN` | 512 | `scripts/asset1_bank.py:117` |
| `EVAL_INTERVAL` | 100 | `scripts/asset1_bank.py:118` |
| `EVAL_BATCH_SIZE` | 4 | `scripts/asset1_bank.py:119` |
| `VAL_SEED` | 777 ("NEVER change (locked card)") | `scripts/asset1_datasets.py:61` |
| `VAL_SIZE` | 500 | `scripts/asset1_datasets.py:62` |
| `POOL_CAP` | 40,000 | `scripts/asset1_datasets.py:63` |
| token positions per run | 16,384,000 | `docs/S2_COST_RESTATEMENT_2026-08-04.md:19`; measured per run at `RATES.md:7-15` (`tokens_trained`) |
| `SEED_BASE` | 10,000 | `scripts/asset1_bank.py:101` |
| `DATA_SEED_BASE` | 20,000 | `scripts/asset1_bank.py:102` |
| seed derivation | `seed = SEED_BASE + run_index`, `data_seed = DATA_SEED_BASE + run_index` | `scripts/asset1_bank.py:198-199` |

**Seeds policy, pinned explicitly** (this is a wiring decision the card did not
make and the lock must): the H2 bank uses the **same two bases** with
`run_index` counted **from 0 within its own bank root**, which is a different
root from Asset-1's (`results/asset1h2-bank/`, `PREREG_DRAFT_H2_SCALE_2026-07-21.md:172`;
Asset-1's own index space is 0…479, `scripts/asset1_analysis_io.py:73`). The bank
tag is what distinguishes the two banks, not the seed. **If the builder is ever
made to share one index space across both banks, an offset band must be written
in source before any launch and recorded as a dated amendment** — the S2 pilots
already set the precedent of a disjoint band (`seed = 90001`, `data_seed = 91001`,
`results/s2-timing-pilots/gemma2-2b/run_1/TIMING.md:23-24`).

---

## 4. Arms, run counts, N

Frozen from the registered §3 table (`PREREG_DRAFT_H2_SCALE_2026-07-21.md:67-71`),
with Llama-3.1-8B (row `:70`) struck per the ruling:

| Family | ~Params (card) | Reps | Runs | Role (card `:67-69`) |
|---|---|---|---|---|
| Gemma-2-2B | 2.6B | 20 | **120** | third lineage near anchor scale |
| Qwen2.5-3B | 3.1B | 20 | **120** | within-lineage scale step |
| Qwen2.5-7B | 7.6B | 10 | **60** | within-lineage scale endpoint |
| **three-family total** | | | **300** | = the S1 drop-option bank (`S2_COST_RESTATEMENT_2026-08-04.md:44`) |

Per-direction n varies 60–240 and **is reported with every test** — the
disclosure rule the Director kept twice
(`REGISTRATION_H2_SCALE_2026-07-30.md:17`;
`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:45`; power basis at
`PREREG_DRAFT_H2_SCALE_2026-07-21.md:83-87`).

The anchor legs are **not** retrained: P4's Llama-3.2-1B leg is the Asset-1 bank's
240 llama3.2-1b runs (2 families × 6 tasks × 40 replicates = 480,
`scripts/asset1_bank.py:94-99`; interlock default `EXPECTED_TOTAL_RUNS = 480`,
`scripts/asset1_analysis_io.py:73`).

---

## 5. Rate basis and cost, all measured

**Pilot design (S2, promoted to a precondition,
`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:46`):** nine
timing-only runs, three per accessible family, **excluded from every bank**,
computing **no** H2 statistic, all `status = COMPLETE`
(`docs/S2_COST_RESTATEMENT_2026-08-04.md:16-19`; `results/s2-timing-pilots/RATES.md:7-15`;
per-run provenance `gemma2-2b/run_1/TIMING.md:49-51`). Pilot total **15.57
GPU-hours** (`S2_COST_RESTATEMENT_2026-08-04.md:27`; recomputed from the nine
`wall_clock_min` values in `RATES.md:7-15` = 934.14 min = 15.569 h ✓).

**Restated §3 cost table** (`docs/S2_COST_RESTATEMENT_2026-08-04.md:37-44`),
each row recomputed this pass from `runs × measured_min / 60 / 24`:

| Family | Runs | Measured min/run | Restated GPU-days | Δ vs estimate |
|---|---|---|---|---|
| Gemma-2-2B | 120 | 75.51 | **6.292** (recomputed 6.2925) | −4.42% |
| Qwen2.5-3B | 120 | 82.86 | **6.905** (recomputed 6.9050) | −10.90% |
| Qwen2.5-7B | 60 | 153.01 | **6.375** (recomputed 6.3754) | −33.47% |
| **three-family bank** | **300** | | **19.573 measured** (recomputed 19.5729) | −18.45% vs 24.0 |

**Affine fit, locked as the cost model for any projection:**
`min/run = 34.921 + 15.5352 × params_B`
(`docs/S2_COST_RESTATEMENT_2026-08-04.md:57`), Director-reproduced to the digit
(`DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:14`), recomputed exactly this pass
(34.9208 + 15.5352). Per-B measured 29.04 / 26.73 / 20.13 min/B (`:56`).
**Disclosure attached to the fit:** its x-axis is the card's public-card
approximations 2.6 / 3.1 / 7.6 B (`scripts/s2_cost_restatement.py:32-34`), not
the measured `n_params_base_measured` of §2; refitting on the measured parameter
counts gives 35.029 + 15.492 × params_B, a difference well inside the fit's own
n=3 uncertainty but worth stating rather than leaving implicit
[MATHEMATICAL FACT, computed 2026-09-11].

**Batch geometry (S3):** peak VRAM 14.07 / 11.05 / 20.30 GB of 48
(`RATES.md:7-15`), so the per-family geometry question does not arise for the
measured rows — `bs4xga4` holds for all three and the A1 bit-equivalence
conditions are unchanged (`docs/S2_COST_RESTATEMENT_2026-08-04.md:31-33`).

---

## 6. Decision rule, multiplicity, interlock

**Decision rule — S4 approved with NO constant changes**
(`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:48`;
`REGISTRATION_H2_SCALE_2026-07-30.md:30`):

| Constant / rule | Locked value | Source (at HEAD `6ca9c4c`) |
|---|---|---|
| `H2_ALPHA` | 0.01, one-sided exact binomial | `scripts/asset1_d1_identifiability.py:233` |
| `H2_MARGIN_PP` | 15.0 percentage points | `scripts/asset1_d1_identifiability.py:234` |
| chance | 1 / n_tasks = 1/6 (balanced classes asserted, not assumed) | `scripts/asset1_d1_identifiability.py:1103` (`chance = 1.0 / len(task_names)`), `:1112` (the assert-rather-than-assume comment) |
| both-directions AND | H2 SUPPORTED iff **both** directions satisfy (i) cross-family accuracy NOT above chance at α **and** (ii) within − cross ≥ margin; disagreeing directions reported, never hidden; an empty direction set is NOT supported | `scripts/asset1_d1_identifiability.py:800-825`, `def h2_supported(...)` (rule text in its docstring, `:802-813`) |
| PRIMARY representation | depth-binned SV spectra; probe corroborating; three-way disagreement itself reportable | `PREREG_DRAFT_H2_SCALE_2026-07-21.md:135-139`, `:103-105` |
| headline | shift-controlled; raw descriptive | `PREREG_DRAFT_H2_SCALE_2026-07-21.md:137`, `:151-153` |
| standardizer | `familywise_standardize`, reused verbatim (Director-verified unsupervised) | `scripts/asset1_d1_identifiability.py:710`, `def familywise_standardize(...)`; `PREREG_DRAFT_H2_SCALE_2026-07-21.md:150-152` |
| spectrum dimensionality | `4 × n_depth_bins × sigma_slots`; `n_depth_bins` default 4; `sigma_slots` = the observed rank ⇒ **384 dims at rank 24** | `scripts/asset1_d1_identifiability.py:664` (docstring `spectrum: 4 * n_depth_bins * sigma_slots`), `:1073` (`n_depth_bins: int = 4`), `:1297` (`sigma_slots = fam_rank`) |
| unequal-rank behaviour **(superseded in this same series — see the dated note below)** | **At `6ca9c4c`:** hard raise, "sigma_slots aggregation undefined". **Replaced by A14** (the H2-S pin's padding clause, §8 item 3): pad both families to `sigma_slots = max(rank)`, guard zero-variance padded slots, report **both** views, refuse truncation-only, report padded-slot occupancy. **The equal-rank path is unchanged** — `sigma_slots == top_slots ==` the common rank, `unequal` False, views `("padded",)` only — and equal ranks are the only case this lock can reach, since `RANK = 24` for every locked family (§3). | **Pinned:** `scripts/asset1_d1_identifiability.py:1298-1301` at `6ca9c4c`, the `elif sigma_slots != fam_rank:` raise. **A14's replacement is cited by symbol**, since it has no line numbers at HEAD `6ca9c4c`; line numbers are a snapshot (2026-09-11 11:40 PDT, file sha256 `579b51156c96f6ce…`): `sigma_slots = max(native.values())` in `def resolve_sigma_slots` (`:890`, def `:875`, called `:1767`) · `H2_VIEWS` (`:251`) · `H2S_TRUNCATION_PROHIBITED` (`:265`), enforced by `def assert_not_truncation_only` (`:904`, called `:1315`) · `def padded_slot_occupancy` (`:1062`, applied `:1811`) · equal-rank ⇒ `("padded",)` (`:1816`). Equal-rank regressions: `tests/test_h2s_padding.py::test_equal_ranks_reproduce_the_pre_clause_path`, `::test_slice_top_slots_identity_when_no_padding`, `::test_h2_transfer_default_output_is_byte_for_byte_unchanged` |
| family-identity probe | mandatory per pair, raw and standardized; a pair whose standardized probe does not collapse toward chance must **not** have its transfer number reported as a headline | `DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:50`; `PREREG_DRAFT_H2_SCALE_2026-07-21.md:147-155` |

**Note on this table's Source column (dated 2026-09-11).** Every citation in it
is **pinned to HEAD `6ca9c4c`** and resolves with
`git show 6ca9c4c:scripts/asset1_d1_identifiability.py`. Item **A14**, which
lands in this same commit series ahead of this declaration, inserts a large
block into that file beginning immediately after `6ca9c4c`'s line 238, so
**every pinned citation at line 239 or later sits at a different line in the
working tree**. Readers resolve them **at the pinned SHA**; the pinned numbers
are deliberately **not** renumbered. (Snapshot of the shift, measured 2026-09-11
11:40 PDT while A14 was still under edit, hence a snapshot and not a pinned
fact: +591 / −15 lines, 1,755 → 2,331.) A14 bears on exactly three of the rows
above, and removes no locked value:

- **unequal-rank behaviour** — superseded in substance; row amended above.
- **spectrum dimensionality** — its third citation `:1297` (`sigma_slots =
  fam_rank`) falls inside the block A14 replaces, but **the locked value is
  unchanged**: A14 sets `sigma_slots = max(native.values())`, and every family
  under this lock carries `RANK = 24` (§3), so `max(rank)` *is* the observed
  rank and the spectrum stays 4 × 4 × 24 = 384 dims.
- **standardizer** — `familywise_standardize`'s signature line at `6ca9c4c:711`
  changes, but only to replace the literal `eps = 1e-12` with the named
  `ZERO_VAR_TOL`, which is `1e-12`; the `def` line cited in the row
  (`6ca9c4c:710`) and the function's behaviour are unchanged.

`h2_transfer` likewise gains keyword-only `views` / `slot_layout` parameters
whose defaults reproduce the `6ca9c4c` call exactly
(`tests/test_h2s_padding.py::test_h2_transfer_default_output_is_byte_for_byte_unchanged`);
it carries no row here. Every other line cited in this table is untouched by A14
(verified line-by-line against the diff, 2026-09-11).

Note on reading the rule: `h2_supported` returns SUPPORTED when **transfer
fails** (that is the Asset-1 H2 as registered). H2-D outcome (A) — transfer *at
the band* — is the rule **not** firing. The docstring at `:802-813` is the
authority; nothing in this lock restates it in other words.

**Multiplicity — S5 approved as drafted**
(`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:49`): primary
confirmatory set = four undirected pairs / 8 directed tests, **P1** Qwen1.5B↔Qwen3B,
**P2** Qwen3B↔Qwen7B, **P3** Qwen7B↔Llama-3.1-8B, **P4** Gemma-2-2B↔Llama-3.2-1B;
Holm–Bonferroni at family-wise α = 0.01 across the primary directed tests; every
other pair descriptive, the full 30-directed-pair grid computed and plotted with
no per-pair claims (`PREREG_DRAFT_H2_SCALE_2026-07-21.md:159-168`). H2-V counts
as its own pre-registered test, **outside** the Holm family
(`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:49`).

**Interlock — S9 T2 approved with the tier order frozen at registration**
(`REGISTRATION_H2_SCALE_2026-07-30.md:27-29`;
`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:53`): tier order =
the §3 table order, cheapest first; each tier analyzed only when its families are
COMPLETE; each tier's gate records which tiers were already unblinded when it
fired; any post-unblinding amendment is restricted to **not-yet-unblinded** tiers
(`PREREG_DRAFT_H2_SCALE_2026-07-21.md:176-184`). New bank root
`results/asset1h2-bank/`, never inside `asset1-bank/` (`:172`). Completeness gate
`require_complete_bank(bank_root, allow_partial, expected_total)`
(`scripts/asset1_analysis_io.py:116-135`); its default `expected_total = 480` is
Asset-1's and **must be overridden with the H2 count at wiring time**, noting the
docstring's standing prohibition on exposing that override as a CLI knob on a
real-bank tool (`scripts/asset1_analysis_io.py:127-129`). Phase-V's
`--prereg-locked` gate stands (`PREREG_DRAFT_H2_SCALE_2026-07-21.md:174-179`).

---

## 7. Three decider calls this lock makes explicitly

These are decider acts, not readings of the Director's text. Each is flagged for
his regrade; any of them can be overturned before the tier it affects fires.

### (i) Which H2-D pairs are inside the lock — **P4 only**

The card's §5 and §8 do not line up on which H2-D pairs can carry a claim:

- §5 (`PREREG_DRAFT_H2_SCALE_2026-07-21.md:119-120`) states H2-D as
  "Gemma-2-2B ↔ {Qwen2.5-1.5B, Llama-3.2-1B}; Qwen2.5-7B ↔ Llama-3.1-8B" — three
  undirected pairs.
- §8 (`:159-162`) registers a primary confirmatory set of **four** undirected
  pairs / 8 directed tests — P1 Qwen1.5B↔Qwen3B, P2 Qwen3B↔Qwen7B,
  P3 Qwen7B↔Llama-3.1-8B, P4 Gemma-2-2B↔Llama-3.2-1B — exactly as §6 above
  records it. **Of §5's H2-D pairs, only P3 and P4 carry confirmatory slots in
  §8's four-pair set** (P1 and P2 are the H2-S within-lineage scale pairs,
  `PREREG_DRAFT_H2_SCALE_2026-07-21.md:113-118`); Gemma-2-2B ↔ Qwen2.5-1.5B
  carries none. The Director approved the set as drafted with the instruction
  to "claim only P1-P4"
  (`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:49`).

**Ruling:** §8 governs what may be *claimed*. Gemma-2-2B ↔ Qwen2.5-1.5B is a
registered H2-D pair for interpretation but carries **no confirmatory slot**; it
is one of the descriptive pairs on the transfer-vs-(scale gap, lineage distance)
surface. **P3 is registered but NOT locked**: its Llama-3.1-8B leg has no
measured rate, and the ruling forbids carrying it unmeasured
(`DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:22`). It enters by dated amendment
when its three pilots land.

**Therefore the confirmatory H2-D set under this lock is P4 alone — one
undirected pair, two directed tests.** The §5/§8 tension is recorded here rather
than silently resolved, and is surfaced to the Director as an erratum on the
registered card.

### (ii) The Holm family stays at the registered size — 8 directed tests

Holm is applied at family-wise α = 0.01 over the **registered** primary family
(P1–P4, 8 directed tests) even though only P4's two directions are executable
under this lock; the realized count is recorded alongside. Reasoning: re-sizing
the family down to 2 would *relax* the registered correction and is a revision of
S5, not an application of it; holding the registered size is strictly
conservative (equivalent to the unexecuted tests carrying p = 1); and it
preserves the option for P3 and, if it ever unlocks, H2-S to enter **without**
altering a correction already applied to an unblinded tier — which the T2 rule
would otherwise forbid (`REGISTRATION_H2_SCALE_2026-07-30.md:27-29`). The
alternative (family = tests actually performed) is available to the Director; it
is not what the decider locks, and choosing it after the Gemma tier unblinds
would not be available at all.

### (iii) Rank-fraction is a mandatory per-pair disclosure

The Director cleared H2-D of his §3 confound by width: "Gemma-2-2B vs
Llama-3.2-1B and Qwen-7B vs Llama-3.1-8B are near-matched in width, so lineage
distance is not confounded with rank fraction the same way"
(`DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:44`). Computed from the cached
configs of §2 at the locked `RANK = 24` [MATHEMATICAL FACT, computed 2026-09-11]:

| projection | Gemma-2-2B | Llama-3.2-1B | ratio (P4) |
|---|---|---|---|
| r / hidden_size | 24/2304 = 0.010417 | 24/2048 = 0.011719 | **1.125×** |
| r / q_proj width (heads × head_dim) | 24/2048 = 0.011719 | 24/2048 = 0.011719 | **1.000× (exact)** |
| r / kv_dim (kv_heads × head_dim) | 24/1024 = 0.023438 | 24/512 = 0.046875 | **2.000×** |

P4 is **exactly matched** on the q/o projection width and 1.125× on residual
width — better than the Director's own framing implies. It is **2.0× apart on the
KV projections**, a gap of the same kind he sized on the Qwen ladder ("r/kv_dim
is 0.0938 at both 1.5B and 3B, then halves to 0.0469 at 7B", `:40`) but which he
did not name for P4.

Two reference points, neither locked. The registered-but-unlocked **P3**
(Qwen2.5-7B ↔ Llama-3.1-8B, the latter's config read from the access probe's
cached `config.json`: **hidden_size 4096, `num_attention_heads` 32,
`num_key_value_heads` 8**, with `head_dim` absent from the file and derived as
`hidden_size // num_attention_heads` = 128, so kv_dim = 8 × 128 = 1024)
therefore sits at 24/3584 = 0.006696
vs 24/4096 = 0.005859 on r/hidden — **1.143×**, near-matched as he said — and at
0.046875 vs 0.023438 on r/kv_dim, **2.0×** again. That config was read
2026-09-11 from `F:\AI-Models\huggingface-cache\hub\models--meta-llama--Llama-3.1-8B-Instruct\snapshots\0e9e39f249a16976918f6564b8830bc894c89659\config.json`
and corroborated key-for-key by `results/s2-timing-pilots/RESOLVED_CONFIGS.md`
(keys `llama3.1-8b_hidden_size`, `_num_attention_heads`, `_num_key_value_heads`,
`_head_dim` — cited by key, not by line, because that file is added earlier in
this same series and so carries no line numbers at HEAD `6ca9c4c`). The
descriptive pair Gemma-2-2B ↔ Qwen2.5-1.5B sits at **1.500×** on r/hidden and
**4.000×** on r/kv_dim, between the 1.125× he cleared and the 2.33× on which he
held H2-S; it carries no confirmatory slot under §7(i), which is part of why it
does not disturb this lock.

**Ruling:** this does **not** unlock or re-scope H2-D — the held confound is
scale-*ladder* decay within a lineage, and P4 is a matched-scale cross-lineage
test. But it is pinned as a **mandatory disclosure**: `r/hidden`, `r/q_width` and
`r/kv_dim` are reported per directed pair in every H2-D artifact, and the P4
KV-projection gap of 2.0× is surfaced to the Director in the next packet as a
fact he has not ruled on.

### What the roster lock funds today

The roster, recipe, run counts and cost table are frozen for **all three
measured families** — that is what "lock the three measured families" says, and
striking 240 runs silently would be exactly the revision L-006 forbids. But only
the **Gemma tier** has a locked confirmatory consumer today: **120 runs, 6.292
measured GPU-days**, feeding P4. Qwen2.5-3B and Qwen2.5-7B carry P1/P2 (H2-S,
held) and P3's 7B leg (registered, not locked). **The 19.573 GPU-day figure is
the frozen cost of the registered bank, not an authorized spend.** No bank run is
authorized by this lock.

---

## 8. What is explicitly NOT locked

**H2-S — HELD.** `DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:22`, `:95`;
`AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md:106`. Reason: r/hidden falls 2.33×
across the Qwen scale ladder (0.0156 → 0.0117 → 0.0067), so outcome (B) is
uninterpretable as written (`DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md:38-44`).
The Director's pin is GRANTED on **v2** (`docs/DIRECTOR_GRADES_2026-08-04.md:10-33`;
addendum `AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md:115-142`). The Director's
own count is **"Two conditions on the pin (a third was drafted and withdrawn;
see (b))"** (`DIRECTOR_GRADES_2026-08-04.md:23`) — the amendment's addendum
enumerates three items at `:120`, but item 2 is his **ratification** of a choice
the amendment had already pinned, not a new condition. All three are carried
here as **preconditions of that arm, not as anything this lock satisfies**:

1. *(condition)* **n = 30 runs, not 60** (`DIRECTOR_GRADES_2026-08-04.md:25`;
   `AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md:121-125`).
2. *(ratification, not a condition — this is the drafted-and-withdrawn third)*
   **Probe task alpaca STANDS** "as the amendment pins it"; the Director drafted
   a re-pin to math, his own audit caught it as an inverted reading of the
   record, and he recorded the drafting error rather than removing it
   (`DIRECTOR_GRADES_2026-08-04.md:27`;
   `AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md:126-130`).
3. *(condition)* **The §3 mandatory padding clause, "approved as written, with
   one addition"** (`DIRECTOR_GRADES_2026-08-04.md:29`, item (c); clause at
   `AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md:91-101`, restated as pin
   condition 3 at `:131-134`): pad both families to `sigma_slots = max(rank)`,
   report the contrast on **both** padded and top-24, guard zero-variance padded
   slots, truncation-only **prohibited** — plus **report the padded-slot
   occupancy** (fraction of nonzero mass in slots 25–54 of the rank-54 leg) so a
   null is distinguishable from "the extra slots were never used".
   **Status (dated 2026-09-11): implemented in code, condition still open.**
   The occupancy statistic existed in prose and in zero code as of
   `C:\falco\docs\WS4_STATE_OF_PLAY_2026-09-01.md:233`, and is absent from
   `scripts/asset1_d1_identifiability.py` at HEAD `6ca9c4c` (verified
   2026-09-11). Item **A14**
   (`C:\falco\docs\WS4_STATE_OF_PLAY_2026-09-01.md:615`) implements the whole
   clause — pad-to-max, zero-variance guard, both views, truncation-only
   refused, padded-slot occupancy reported — and **lands in this same commit
   series ahead of this declaration** (working-tree lines cited in
   the dated note under §6's table; its regression suite
   `tests/test_h2s_padding.py` passing 2026-09-11). That closes the clause's *implementation*, not the condition:
   the H2-S arm is HELD and unrun, so nothing has yet been reported under it.
   Neither is **closed by this lock**.

Arm cost, **not authorized**: 3.83 GPU-days expected / 4.46 worst case
(`AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md:74-80`).

**Llama-3.1-8B.** No measured rate and no pilot directory. `159.20 min/run` and
`6.633 GPU-days` are **PROJECTED** and stay labelled as such
(`docs/S2_COST_RESTATEMENT_2026-08-04.md:42`, `:65-69`). **Access is OK** — the
licence gate was accepted and the family probes OK
(`results/s2-timing-pilots/GATING.md:13`, `:17`, probed 2026-08-04, the source of
record; the conflicting stale typed block that
`results/s2-timing-pilots/QUEUE.md` carried at HEAD `6ca9c4c` is logged at
`C:\falco\docs\WS4_STATE_OF_PLAY_2026-09-01.md:239-244` and was corrected under
A13 sub-item 2, earlier in this same series). Stated precisely, because "cached" is easy to get
wrong here [verified 2026-09-11]: the repo's `config.json` alone is on disk in the
**default** HF cache (`F:\AI-Models\huggingface-cache\hub`, revision
`0e9e39f249a16976918f6564b8830bc894c89659`, hidden_size 4096 / 32 layers / 8 kv
heads), which is what a config-only access probe leaves behind
(`results/s2-timing-pilots/GATING.md:3`); **no weights are cached in either
store**, and nothing for this family is in `C:\falco\hf-cache\hub`, the cache
every pilot loads from (`GATING.md:26`). Config corroborated by the parallel Ask-2
closure artifact `results/s2-timing-pilots/RESOLVED_CONFIGS.md`. What is missing
for P3 is therefore the **three timing pilots**, never launched, ~8 GPU-hours
(`docs/S2_COST_RESTATEMENT_2026-08-04.md:68-69`;
`C:\falco\docs\WS4_STATE_OF_PLAY_2026-09-01.md:575`).

**H2-V.** Corroborating everywhere including the anchor pair per S7(d)
(`REGISTRATION_H2_SCALE_2026-07-30.md:24-26`;
`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:51`). Its
`--prereg-locked` Phase-V gate is unaffected, but H2-V carries no part of this
lock and elevation to co-primary remains a future dated amendment conditional on
replication across P1–P4.

**The spectral-tail head-vs-tail contrast.** Pre-declared, "carries no lock and
no multiplicity slot" (`AMENDMENT_H2S_RANK_FRACTION_2026-08-04.md:45-60`,
standing verbatim per `AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md:107`).

**D-aux.** Descriptive-only replication, explicitly non-confirmatory, no lock, no
multiplicity slot (`DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md:55`).

**S1's GPU-day commitment and any bank run**
(`REGISTRATION_H2_SCALE_2026-07-30.md:14-17`, `:36-37`).

**Tinker-substrate arms.** Out of scope of the registration entirely
(`REGISTRATION_H2_SCALE_2026-07-30.md:38-39`).

---

## 9. Carry-forward — named here, not closed here

| Item | Consequence if left open |
|---|---|
| `llama3.1-8b` timing pilots — 3 runs, ~8 GPU-hours, access OK | P3 cannot be locked; the fourth cost row stays PROJECTED; under §7(ii) the amendment admitting P3 must land **before the Gemma tier unblinds** |
| H2-S padding + occupancy clause with a unit test (**A14**) — *code side landed in this same series ahead of this declaration; see §8 item 3* | the clause is now implemented and tested, but pin condition 3 is only **satisfied in execution** when the held arm runs and reports it |
| `expected_total` wiring for `results/asset1h2-bank/` | the completeness interlock would silently check Asset-1's 480 against an H2 bank |
| H2 bank run_index band written in source | §3's seeds pin is otherwise enforced only by this document |
| Cross-bank aggregation for P4 (new Gemma tier × existing Asset-1 llama3.2-1b runs) | unwired; must be resolved before the Gemma tier's gate can fire |
| §5/§8 H2-D pair-set tension on the registered card | erratum owed to the Director (§7(i)) |
| P4 KV-projection rank-fraction gap of 2.0× | disclosure owed to the Director (§7(iii)) |

---

*Lock declared by Meridian as decider on 2026-09-11, on the Director's ruling of
2026-08-04. Every constant above was read from a repository file at HEAD
`6ca9c4c` or from the workstation HF cache on the date of declaration; the cost
and rank-fraction arithmetic was recomputed this pass rather than carried
forward. The Director retains the grader role in full and may regrade any decider
call in §7 before the tier it affects fires. House form after
`docs/LOCK_GRANULARITY_2026-08-04.md`, extended with the per-constant source
column required by the read-every-locked-constant-in-source standard
(`C:\falco\docs\WS4_STATE_OF_PLAY_2026-09-01.md:710`).*
