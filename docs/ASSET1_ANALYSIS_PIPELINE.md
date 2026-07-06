# Asset-1 Analysis Pipeline — Operator Documentation

**Status:** BUILT, TESTED, WAITING FOR THE BANK. 168 tests passing (2026-07-06).
The pre-registration sign-offs are now **DECIDED** — see
`docs/DIRECTOR_DECISIONS_2026-07-06.md` and §4 below. The D1/D2/D3 tools were
updated to match: D1 runs BOTH representations by default, H2 has a pinned
decision rule, D2 runs the decomposition unconditionally with
`permuted_deviation` as THE H3 reference, and D3's primary label rule is a
fixed 5% relative-degradation threshold.
**Spec:** the locked experiment card `asset1_experiment_card_2026-07-03.md`
(Director → Meridian, hypotheses LOCKED; currently held outside the repo at
`C:\Users\Timothy Paul Bielec\Downloads\Telegram Desktop\`).
**Bank:** `results/asset1-bank/` — 480 runs = 40 reps × 6 tasks × 2 families
(qwen2.5-1.5b: 112 modules/run; llama3.2-1b: 64 — module counts are discovered
from each adapter, never hardcoded). The campaign is LIVE; the trainer is
writing into that tree. Nothing below reads it until the manifest says
480/480 COMPLETE.

---

## 1. Purpose

The pre-registration forbids running any classifier, correlation, or summary
statistic over the real bank before all 480 runs are COMPLETE. The entire
D1/D2/D3/D-aux analysis layer was therefore built and validated **against
synthetic fixtures only**, in advance, so that analysis fires the day the bank
completes with zero unblinded development in between. No real adapter, no real
`metrics.json`, and no real manifest statistics were touched during the build.
Every analytical choice the card left open is recorded in §4 and needs
Director sign-off **before** the bank lands — choosing after seeing results
breaks pre-registration.

The pipeline, in dependency order:

| Module | Role | Compute |
|---|---|---|
| `scripts/asset1_analysis_io.py` | Foundation: manifest-driven run enumeration, adapter loading, deterministic feature flattening, gap trajectories, **the interlock**. No CLI of its own — no ungated entry point. | CPU |
| `scripts/asset1_synth.py` | Synthetic bank generator (schema-exact via `asset1_bank.RunSpec`, planted known structure). Refuses to write into any `asset1-bank` path. | CPU |
| `scripts/asset1_d1_identifiability.py` | D1 headline: within-family task identifiability (H1) + cross-family transfer regime contrast (H2) with triviality controls. | CPU only |
| `scripts/asset1_d2_swap.py` | D2 bridge-swap matrix (H3). Stage A = plan + assembly (CPU). Stage B = val-loss evaluation (**GPU, post-bank only**). | CPU / GPU |
| `scripts/asset1_d3_merge.py` | D3 weight-only merge-degradation prediction: features + merge constructor now, labels consumed later. | CPU (labels need GPU evals) |
| `scripts/asset1_daux_gap.py` | D-aux: bridge deviation ↔ generalization gap (pilot r = 0.888 re-verification). | CPU only |

Shared upstream (built earlier, reused not modified):
`scripts/asset1_canonicalize.py` (GL(r)-invariant canonicalization, verified
~1e-13), `scripts/asset1_bank.py` (the trainer — source of truth for schema),
`scripts/asset1_datasets.py` (locked val_seed=777 splits).

## 2. Fire sequence — bank-completion day

Run everything from `C:\falco\rhombic` in the `falco` conda env
(`C:\miniconda3\envs\falco\python.exe`). Every real-bank command below gates
itself on the interlock (§3) and refuses until the manifest shows 480/480
COMPLETE. Prerequisite: the §4 sign-offs are **DECIDED**
(`docs/DIRECTOR_DECISIONS_2026-07-06.md`) — the previously-open defaults
(A2 representation, H2 representation + decision rule, D2 H3 reference +
decomposition, D3 label definition) are now pinned in the tools, so the
analysis fires with those choices baked in and recorded in every output.

**Step 0 — re-verify tooling (CPU, safe any time, touches no real data):**

```
python -m pytest tests/test_asset1_analysis_io.py tests/test_asset1_d1.py tests/test_asset1_d2.py tests/test_asset1_d3_daux.py -q
python scripts/asset1_d1_identifiability.py --synthetic-selftest --out-dir results/asset1-d1-selftest
python scripts/asset1_d3_merge.py --selftest
python scripts/asset1_daux_gap.py --selftest --out-dir results/asset1-daux-selftest
```

(D2 has no selftest flag; its acceptance coverage lives in
`tests/test_asset1_d2.py`.)

**Step 1 — D1 headline (CPU only; the RAM-heavy one):**

```
python scripts/asset1_d1_identifiability.py --out-dir results/asset1-d1
```

Defaults: `--bank-root results/asset1-bank`, `--representation both`
(A2 ADOPTED — raw AND W2T-canonical within each family, both reportable),
`--n-permutations 1000`, `--seed 0`, `--chunk-rows 8` (peak RAM ~0.9 GB at
d≈6.5M). Writes `d1_results.json` + `D1_REPORT.md` under `--out-dir`, plus a
`scratch/` memmap directory that is cleaned per family. H1 per family (both
representations), H2 both directions under both representations with the
family-identity probe and shift-controlled variant, and the pinned H2
decision (`h2_verdict`): PRIMARY = spectrum, corroborating = probe, headline
= the shift-controlled variant, supported iff BOTH directions clear α=0.01
(exact binomial) AND a ≥15pp within-minus-cross accuracy margin. Cosmetic
note: a refused invocation still leaves an empty `--out-dir` behind (mkdir
precedes the interlock in this one tool); ignore or delete.

**Step 2 — D-aux (CPU only; fast — rides along):**

```
python scripts/asset1_daux_gap.py --bank-root results/asset1-bank --out-dir results/asset1-daux
```

Writes `daux_run_table.csv` (one row per run) and `daux_report.json`
(primary `dev_mean` vs `final_gap`, descriptive pairs, per-family and
within-task Simpson's-guard cells, step-0 identity control).

**Step 3 — D2 Stage A: plan + assembly (CPU):**

```
python scripts/asset1_d2_swap.py --bank-root results/asset1-bank --out-dir results/asset1-d2
```

The magnitude/topology decomposition cells now run **UNCONDITIONALLY** by
default (Director OVERRIDE — `docs/DIRECTOR_DECISIONS_2026-07-06.md`); pass
`--no-decomposition` only for exploratory work. Do **not** use `--plan-only`
for the real run: a plan with null `assembled_sha256` silently disables
Stage B's bank-integrity SHA check — full assembly records the digests that
Stage B re-verifies. Writes `d2_swap_plan.json` (**360 evals/family** at the
default K=3 with decomposition on; 180 with `--no-decomposition`). The plan
records `decomposition_policy` (unconditional) and
`h3_structure_reference.status` = APPROVED. `--write-states` persists
assembled `.pt` states under `results/asset1-d2/states/` — optional,
~26 MB each.

**Step 4 — D2 Stage B: evaluation (GPU — the first GPU-burning step; one
command per family):**

```
python scripts/asset1_d2_swap.py --bank-root results/asset1-bank --out-dir results/asset1-d2 --family qwen2.5-1.5b --evaluate --i-have-gpu-and-bank-is-complete
python scripts/asset1_d2_swap.py --bank-root results/asset1-bank --out-dir results/asset1-d2 --family llama3.2-1b --evaluate --i-have-gpu-and-bank-is-complete
```

Loads the base model once per family, re-assembles each eval's state from the
bank, verifies the recorded SHA-256 (hard error if the bank changed since
planning), installs, and computes val_loss on the recipient task's fixed
500-example split (val_seed=777 design; `asset1_datasets.build_dataset` +
`asset1_bank.evaluate_val_loss`). Results checkpoint to
`d2_results_<family>.json` after every eval, but an interrupted run
**restarts that family from eval 1** (the results file is rewritten fresh) —
plan GPU windows accordingly. The H3 structure-reference is DECIDED: the plan
carries `h3_structure_reference.status` = APPROVED 2026-07-06 with
`permuted_deviation` as THE primary reference and the full-entry `permuted`
cell retained as the identity-backbone contrast.

**Step 5 — D3 pair generation + merge emission (CPU):**

```
python scripts/asset1_d3_merge.py --bank-root results/asset1-bank --out-dir results/asset1-d3 --make-pairs <N> --emit-merges
```

`<N>` and the pair design need sign-off first (§4). Default
`--max-run-uses 1` gives vertex-disjoint pairs (the dyadic-dependence-safe
design; `0` = legacy unlimited, exploratory only), `--alpha 0.5` midpoint
merges. Writes `d3_pairs.json` + merged adapter states under
`results/asset1-d3/merges/`.

**Step 6 — D3 label generation (GPU, external to this pipeline):** evaluate
each merged adapter under `merges/` on BOTH endpoint tasks' val splits and
produce a labels file in the documented schema (JSON `{"pairs": [...]}` or
CSV — see the `asset1_d3_merge.py` module docstring). To exercise the
**primary** binarization, carry the per-endpoint metric fields
(`merged_ppl_a`/`native_ppl_a`/`merged_ppl_b`/`native_ppl_b`, and/or the
`*_score_*` task-metric form). Primary label rule is DECIDED: a pair is a
"conflict" if the merge degrades EITHER endpoint by ≥5% relative to that
endpoint's native adapter (perplexity up, or task-metric down); median-split
is secondary/descriptive. The continuous `degradation` field is still
required (feeds the median fallback and the ridge model).

**Step 7 — D3 prediction harness (CPU):**

```
python scripts/asset1_d3_merge.py --bank-root results/asset1-bank --out-dir results/asset1-d3 --labels <labels-file>
```

Writes `d3_report.json`. Labels are binarized by the PRIMARY rule (fixed 5%
relative-degradation threshold, `--label-threshold-rel`, degenerate floor
`--degenerate-min-frac` 0.10); the `binarization` block records the rule
actually used and any degenerate-fallback finding. Group-aware CV numbers
(StratifiedGroupKFold over run-overlap components, cluster bootstrap) are the
per-family headline; naive pair-level CV is reported as an explicitly
anti-conservative secondary block. If the pair graph is one giant component,
no headline is emitted at all.

Ordering rationale: Steps 1–3 are CPU-only and can run immediately and in
parallel with each other; Step 4 is the first GPU commitment; Steps 5–7
follow because D3's labels do not exist until its own GPU evals run.

## 3. The interlock

`asset1_analysis_io.require_complete_bank()` is called by every CLI that can
touch the real bank, before a single adapter is read. It passes only when
`results/asset1-bank/bank_manifest.json` lists **exactly 480 runs, every one
COMPLETE** — a manifest listing any other number of runs, or any
BLOCKED/FAILED/PENDING status, is refused with SystemExit. The manifest (not
the filesystem) is the source of truth: half-written run directories the live
trainer is populating are invisible to analysis until the campaign runner
marks them COMPLETE. `expected_total` is overridable only in code, for
synthetic fixtures and tests; no real-bank CLI exposes it as a knob.

`--allow-partial-bank` exists because the tooling itself must be smoke-testable
against the real tree (does the loader parse a real adapter? does the plan
generator see the manifest?) without waiting three weeks. It prints a loud
multi-line PRE-REGISTRATION WARNING to stderr and marks D1 output
`exploratory_only: true`. Nothing produced under this flag may ever be
reported, quoted, or compared to the locked hypotheses. There is deliberately
no quiet middle ground: if the Director ever authorizes analysis of a
permanently reduced bank (e.g. a family license-blocked), that path also goes
through `--allow-partial-bank` with the warning attached — a design point
itself flagged for confirmation (§4, item G1).

Complementary guards: every tool refuses an `--out-dir` inside the bank tree;
all loads are CPU (`map_location='cpu'`, `weights_only=True`); the synthetic
generator refuses to write into any path containing an `asset1-bank`
component; D2's Stage B additionally requires the explicit
`--i-have-gpu-and-bank-is-complete` acknowledgement and refuses before any
lazy transformers/datasets import.

## 4. DIRECTOR SIGN-OFF SECTION — DECIDED 2026-07-06

Every pre-registration ambiguity the card left open, now **DECIDED** per
`docs/DIRECTOR_DECISIONS_2026-07-06.md` and baked into the tools (each choice
is recorded in the tool's output JSON so runs self-document). The headline
decisions:

- **A2 — representation:** `--representation` default = **both** (raw AND
  W2T-canonical within each family; both reportable).
- **D1 regime-contrast spine:** the distinguishing axis is **label
  granularity / task structure** (W2T's 10k+ fine-grained attribute classes
  vs the 6 coarse tasks here), NOT hub-scale-vs-family — W2T's own
  collections are same-base/same-rank families (W2T Table 6).
- **H2 representation:** PRIMARY = depth-binned SV spectra (`spectrum`);
  probe-projection = corroborating; disagreement is reported.
- **H2 decision rule:** supported iff BOTH directions clear one-sided
  exact-binomial **α = 0.01** AND a **≥ 15 pp** within-minus-cross accuracy
  margin, in the **shift-controlled** representation (raw = descriptive).
- **D2 decomposition:** runs **UNCONDITIONALLY** (+180 evals/family).
- **D2 H3 structure reference:** **`permuted_deviation`** (primary);
  full-entry `permuted` retained as the identity-backbone contrast.
- **D3 label rule:** PRIMARY = **fixed 5% relative-degradation threshold**
  per endpoint; median-split is secondary with a degenerate-balance
  (< 10%) fallback that is reported.

The remaining items below are APPROVED as-is. Their statuses are updated in
place.

### Global

- **G1 — Interlock strictness.** No quiet path for a deliberately reduced
  bank; any non-480 analysis is `--allow-partial-bank` + loud warning.
  Confirm this matches intent.
- **G2 — A2 raw-vs-canonical gating. DECIDED (A2 ADOPTED).**
  `--representation` defaults to **both**: raw AND W2T-canonical within each
  family, both reportable. Canonical is fixed to
  `asset1_canonicalize.feature_vector` variant `'full'` with
  `--proj-dim 16` / `--proj-seed 0`.

### D1 / H1 — DEFAULTS APPROVED as-is

- **D1a — SVM C. APPROVED.** Default 1.0 (sklearn default) via `--svm-c`,
  recorded in every output.
- **D1b — 95% CI type. APPROVED.** Wilson score interval, always carrying
  the caveat that LOO folds are not independent — the permutation p is the
  calibrated inference.
- **D1c — Heterogeneity-guard metric. APPROVED.** Euclidean distance in the
  exact feature space the classifier saw (computed from the Gram); trigger
  at the pilot's 3.7× ratio.
- **D1d — Per-module breakdown has no permutation null. APPROVED.**
  Completeness-only per card D1 item 4 (per-module nulls = 112+ unregistered
  tests needing multiplicity control).

### D1 / H2 — **DECIDED (DIRECTOR_DECISIONS_2026-07-06.md, §6/H2)**

- **H2a — Representation choice. DECIDED.** Both dimension-agnostic
  representations are computed; the depth-binned singular-value spectra of
  the effective update (`spectrum`) is the **PRIMARY** representation
  carrying the H2 claim, and the canonicalize probe-projection route
  (`probe`) is **corroborating**. Disagreement between the two is reported,
  not hidden (`h2_verdict.agreement`).
- **H2b — Aggregation hyperparameters. APPROVED.** `--n-depth-bins 4`,
  sigma_slots = rank (24), bucket = (projection type q/k/v/o, depth bin),
  mean aggregation, empty buckets contribute zeros, depth fraction
  (L + 0.5)/n_layers.
- **H2c — Decision rule. DECIDED (`h2_supported`).** H2 (transfer FAILS) is
  supported iff, for BOTH directions (A→B and B→A) in the shift-controlled
  representation: (i) cross-family accuracy is NOT significantly above chance
  at one-sided exact-binomial **α = 0.01**, AND (ii) within-family accuracy
  minus cross-family accuracy **≥ 15 percentage points**. Each direction's
  accuracy, binomial p, and margin are reported alongside the overall
  verdict. Constants `H2_ALPHA = 0.01` / `H2_MARGIN_PP = 15.0` are recorded
  in output. The shift-controlled variant is the headline; raw is
  descriptive.
- **H2d — Triviality-control protocol (round-1 review fix). APPROVED.** The
  family-identity probe and the shift-controlled (family-standardized)
  transfer variant are part of the pinned protocol; transfer is reported
  under both raw and family-standardized variants and the decision runs on
  the shift-controlled variant.

### D2 — **[required before Stage B GPU evals]**

- **D2a — K.** `--pairs-per-cell` default 3 → 180 evals/family (360 with
  `--decomposition`). The card does not specify K; sign off on 3 or set
  another value before the post-bank run.
- **D2b — H3 structure-destroyed reference. DECIDED: `permuted_deviation`.**
  `permuted_deviation` (I + permute(B − I) — preserves the identity backbone
  and the trained-deviation multiset) is THE primary H3 structure reference;
  the full-entry `permuted` cell (derangement of all C² = 36 entries) is
  retained as the identity-backbone contrast. Both share the same derangement
  per (family, task, slot) so their contrast isolates the backbone effect.
  The plan JSON carries `h3_structure_reference.status` = "APPROVED
  2026-07-06: permuted_deviation" with `primary`/`contrast` fields.
- **D2c — Permuted-baseline reading.** Implemented as permuting the
  RECIPIENT'S OWN bridge (per-task row control). The card's "random/permuted
  bridge" could instead mean permuting each cross-task donor's bridge per
  cell (+T·(T−1)·K = 90 evals/family). Confirm the reading.
- **D2d — Permutation scope.** One derangement over all C² entry positions
  (diagonal included), shared across all modules of the assembled adapter.
  Off-diagonal-only or per-module-independent permutations are defensible
  alternatives; confirm.
- **D2e — Magnitude/topology decomposition.** Two-sided row/col-norm
  factorization D = diag(r)^½ P diag(c)^½ (exact round-trip). Alternatives
  exist (diagonal-vs-off-diagonal split; global norm vs direction). Sign off
  before any contradiction-guard number enters the write-up.
- **D2f — Identity-bridge cell.** An addition beyond the card's two named
  baselines (justified by identity init: measures the total contribution of
  bridge training). Confirm inclusion in the reported matrix.
- **D2g — Recipient reuse.** The same K recipients serve every cell of a task
  row (amortized natives, within-row comparability, at the cost of full cell
  independence). Confirm.
- **D2h — decomposition default. DECIDED: UNCONDITIONAL.** The
  magnitude/topology guard cells run ALWAYS (+180 evals/family), not gated on
  whether a D1/D2 conflict materializes — data-dependent gating is a post-hoc
  forking point. `generate_plan(decomposition=True)` is the default; the CLI
  opt-out is `--no-decomposition` (exploratory only). The plan records
  `decomposition_policy` = unconditional.

### D3 — **[required before labels are generated]**

- **D3a — Pair-selection policy.** The sampler is a deterministic mechanism
  (uniform without replacement over same-family run pairs, seeded,
  vertex-disjoint by default). The reportable design — N, stratification over
  (task_i, task_j) cells, inclusion of same-task pairs as the cross-seed
  reference — is not fixed by the card and must be pre-declared before labels
  exist.
- **D3b — Degradation label definition + binarization. DECIDED (OVERRIDE).**
  Primary binarization = **fixed relative-degradation threshold**: a pair is
  a "conflict" (positive) iff the merge degrades EITHER endpoint task by
  **≥ 5% relative** vs that endpoint's native adapter (perplexity up, or
  task-metric down). The labels schema carries, per pair, the merged metric
  and BOTH natives' metrics per endpoint (`merged_ppl_*` / `native_ppl_*`
  and/or `*_score_*`). Median-split is **secondary/descriptive** only.
  Degenerate fallback: if the 5% rule yields < 10% positives (or < 10%
  negatives), that is reported as a finding and the headline falls back to
  the pre-declared median split — the rule actually used is recorded.
  Constants `THRESHOLD_REL = 0.05` / `DEGENERATE_MIN_FRAC = 0.10`
  (`--label-threshold-rel` / `--degenerate-min-frac`).
- **D3c — Headline AUC pooling.** Fits are per family; a pooled out-of-fold
  AUC is reported as a descriptive summary. Pre-declare which is THE number.
  (Within each family, group-aware CV is the headline basis; naive numbers
  are secondary by construction.)

### D-aux

- **D-auxa — Deviation ↔ gap definition.** Primary pre-registered pair:
  `dev_mean` (mean over modules of ‖bridge_final − I‖_F) vs `final_gap`
  (val − train at the last metrics record, step 2000). `gap_auc`
  (trapezoidal, effectively from step 100 since the step-0 gap is NaN by
  design) is a descriptive alternative. Declare any preferred trajectory
  summary before unblinding.
- **D-auxb — Within-task stratified correlations.** Added as a
  Simpson's-paradox guard, framed as pre-registered-compatible DESCRIPTIVE
  detail; the pooled bank-level correlation remains the claim. Confirm the
  framing.
- **D-auxc — Update-magnitude covariate.** Bridges are confirmed trainable
  (the frozen-bridge fallback is not needed), but mean ‖ΔW‖_F vs gap is still
  computed in the descriptive block. Confirm it may appear
  descriptive-only, or strike it.
- **D-auxd — Interpretive caveat for the D3 write-up.** Invertible bridges
  are a gauge on the update column space, so principal-angle features are
  provably insensitive to bridge numerics; bridge information reaches the
  classifier through magnitude weights and raw distances. Relevant when
  interpreting which feature block drives any AUC gain.

## 5. Synthetic validation summary

**168 tests, all passing** (~50s):

```
python -m pytest tests/test_asset1_analysis_io.py tests/test_asset1_d1.py tests/test_asset1_d2.py tests/test_asset1_d3_daux.py -q
```

All statistics run on synthetic tmp-dir fixtures or in-memory constructions —
nothing reads or writes `results/asset1-bank/`, no HF downloads, no network,
no GPU. `scripts/asset1_synth.py` replicates the real on-disk schema exactly
(manifest fields via `asset1_bank.RunSpec` itself, adapter-state key naming,
config/metrics shapes, COMPLETE markers, bridge npys) while planting known
structure switched by `task_effect`. What the suites and selftests prove:

- **IO/interlock** (`tests/test_asset1_analysis_io.py`): interlock behavior on
  complete/partial/allow-partial/wrong-total/missing-manifest; schema
  round-trip through every IO function; deterministic flattening order and
  layout-slice correctness across mismatched family dims; the synth
  generator's write-safety guards.
- **D1** (`tests/test_asset1_d1.py` + `--synthetic-selftest`): Gram-trick ≡
  explicit linear SVM; permutation-p calibration on null data; planted signal
  detected (accuracy ≫ chance, p below the lock) and null bank at chance;
  cross-family transfer at chance under BOTH H2 variants; the family-identity
  probe fires (>0.9) on families sharing a generative process and differing
  only in dims — the exact triviality scenario from review — and the
  shift-control variant rescues genuinely transferable structure that raw
  transfer misses; the pinned H2 decision rule (`h2_supported`) passes/fails
  each of the two conditions and the both-directions requirement with the
  α = 0.01 / ≥ 15 pp constants honored; `--representation` defaults to
  `both`.
- **D2** (`tests/test_asset1_d2.py`): plan determinism and SHA stability;
  bit-exact recipient A/B preservation with donor bridge installed;
  derangement properties; `permuted_deviation` provably isolates trained
  structure (distance ≤ 2‖D‖_F while the full-entry permutation is O(1)
  backbone-dominated); decomposition exact round-trip and now ON by default
  (unconditional per the Director override); `h3_structure_reference` status
  APPROVED; Stage B import safety (transformers blocked → Stage A unaffected;
  `--evaluate` refused at the gate before any lazy import).
- **D3/D-aux** (`tests/test_asset1_d3_daux.py`, `--selftest` in both tools):
  principal-angle GL(r) gauge invariance to 1e-10; planted-angle selftest
  (full-feature AUC ≥ distance-only + 0.15 and ≥ 0.85); the dependence
  control — naive CV exploits pure run-identity leakage (AUC inflated) while
  group-aware CV sits near chance (≥ +0.15 separation required, group ≤
  0.70); cluster-bootstrap CIs wider than pair-iid under planted dependence;
  the PRIMARY fixed 5% relative-degradation binarization (a pair degrading
  one endpoint ≥ 5% is positive, < 5% negative) with the degenerate-balance
  (< 10%) fallback to median split reported end-to-end; D-aux recovers the
  planted deviation↔gap correlation (r > 0.9 pooled AND within-task) and
  honestly reports "correlation undefined" on the zero-deviation bank.

Canonicalizer invariance (~1e-13) was verified separately in
`tests/test_canonicalize.py`; bank trainer coverage lives in
`tests/test_asset1.py`.

## 6. What remains impossible until post-bank

- **Every real-bank number.** The interlock refuses all D1/D2/D3/D-aux
  real-bank invocations until 480/480 COMPLETE. Nothing exploratory under
  `--allow-partial-bank` is reportable.
- **D2 Stage B.** GPU val-loss evaluation needs the complete bank and the two
  base models (HF download). The H3 reference (D2b) is DECIDED
  (`permuted_deviation`) and the decomposition is unconditional (D2h).
- **D3 labels.** Post-merge degradation labels do not exist until the merged
  adapters emitted in Step 5 are GPU-evaluated. The degradation definition
  and binarization (D3b) are DECIDED (fixed 5% relative-degradation, per
  endpoint) — the GPU evals must emit the per-endpoint metric fields. The
  prediction harness (Step 7) is fully built and idle until then.
- **H2 unblinding.** The analysis runs on completion day. The representation
  (H2a, primary=spectrum) and decision rule (H2c, α=0.01 / ≥15pp, both
  directions) are DECIDED, so the H2 verdict is emitted directly from the
  results — nothing about the rule is chosen after seeing the numbers.
- **The card's result tables.** The card's instruction stands: hypotheses are
  locked; fill result tables only.
