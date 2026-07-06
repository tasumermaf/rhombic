# Asset-1 Analysis Pipeline — Operator Documentation

**Status:** BUILT, TESTED, WAITING FOR THE BANK. 156 tests passing (2026-07-06).
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
COMPLETE. Prerequisite: the §4 sign-offs are resolved — several defaults
(H2 representation, D2 H3 reference, D3 pair policy and label definition)
must be pinned before, not after, the numbers exist.

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

Defaults: `--bank-root results/asset1-bank`, `--representation raw`
(canonical pending A2 approval — if approved, run `--representation both`),
`--n-permutations 1000`, `--seed 0`, `--chunk-rows 8` (peak RAM ~0.9 GB at
d≈6.5M). Writes `d1_results.json` + `D1_REPORT.md` under `--out-dir`, plus a
`scratch/` memmap directory that is cleaned per family. H1 per family, H2
both directions under both representations with the family-identity probe
and shift-controlled variant. Cosmetic note: a refused invocation still
leaves an empty `--out-dir` behind (mkdir precedes the interlock in this one
tool); ignore or delete.

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

Add `--decomposition` only if the Director pre-authorized burning the
contradiction-guard cells unconditionally (§4). Do **not** use `--plan-only`
for the real run: a plan with null `assembled_sha256` silently disables
Stage B's bank-integrity SHA check — full assembly records the digests that
Stage B re-verifies. Writes `d2_swap_plan.json` (180 evals/family at the
default K=3; 360 with `--decomposition`). `--write-states` persists assembled
`.pt` states under `results/asset1-d2/states/` — optional, ~26 MB each.

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
plan GPU windows accordingly. Requires the H3 structure-reference sign-off
first: the plan carries `h3_structure_reference` = PENDING DIRECTOR SIGN-OFF.

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
each merged adapter under `merges/` on the relevant val split(s) and produce
a labels file in the documented schema (JSON `{"pairs": [...]}` or CSV —
see the `asset1_d3_merge.py` module docstring). The degradation definition
is a Director sign-off item; the harness is deliberately agnostic.

**Step 7 — D3 prediction harness (CPU):**

```
python scripts/asset1_d3_merge.py --bank-root results/asset1-bank --out-dir results/asset1-d3 --labels <labels-file>
```

Writes `d3_report.json`. Group-aware CV numbers (StratifiedGroupKFold over
run-overlap components, cluster bootstrap) are the per-family headline; naive
pair-level CV is reported as an explicitly anti-conservative secondary block.
If the pair graph is one giant component, no headline is emitted at all.

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

## 4. DIRECTOR SIGN-OFF SECTION

Every pre-registration ambiguity the card leaves open, with the implemented
default. Items marked **[BEFORE BANK LANDS]** must be pinned before the
480th run completes; the rest before their tool's numbers are unblinded or
enter a write-up.

### Global

- **G1 — Interlock strictness.** No quiet path for a deliberately reduced
  bank; any non-480 analysis is `--allow-partial-bank` + loud warning.
  Confirm this matches intent.
- **G2 — A2 raw-vs-canonical gating.** `--representation` defaults to `raw`
  pending Director A2 approval of canonical mode. Canonical is fixed to
  `asset1_canonicalize.feature_vector` variant `'full'` with
  `--proj-dim 16` / `--proj-seed 0`; confirm those if canonical becomes
  reportable.

### D1 / H1

- **D1a — SVM C.** Card locks "linear SVM" but not C. Default 1.0 (sklearn
  default) via `--svm-c`, recorded in every output. Confirm or pin.
- **D1b — 95% CI type.** Card says "95% CI" without a type; Wilson score
  interval implemented, always carrying the caveat that LOO folds are not
  independent — the permutation p is the calibrated inference. Confirm
  Wilson is acceptable.
- **D1c — Heterogeneity-guard metric.** Euclidean distance in the exact
  feature space the classifier saw (computed from the Gram); trigger at the
  pilot's 3.7× ratio. Confirm.
- **D1d — Per-module breakdown has no permutation null.** Read as
  completeness-only per card D1 item 4 (per-module nulls = 112+ unregistered
  tests needing multiplicity control). Confirm this reading.

### D1 / H2 — **[BEFORE BANK LANDS — highest priority]**

- **H2a — Representation choice.** Raw flattened parameters are dimensionally
  incomparable across families, so two dimension-agnostic representations are
  implemented and both reported: (a) depth-binned singular-value spectra of
  the effective update, (b) the canonicalize probe-projection route. The
  Director must pin which carries the H2 claim before the bank completes.
- **H2b — Aggregation hyperparameters.** `--n-depth-bins 4`, sigma_slots =
  rank (24), bucket = (projection type q/k/v/o, depth bin), mean aggregation,
  empty buckets contribute zeros, depth fraction (L + 0.5)/n_layers. Pin
  pre-unblinding.
- **H2c — Decision rule.** The card says transfer "does not exceed chance"
  with no test statistic. Implemented: accuracy per direction + one-sided
  exact binomial p vs chance, labeled descriptive. Pin the pass/fail
  criterion (e.g. accuracy ≤ 1.5× chance in both directions, or binomial
  p ≥ α) before unblinding.
- **H2d — Triviality-control protocol (round-1 review fix, pinned pre-bank).**
  Both representations carry a family-identity scale signature that would
  make "transfer fails" trivially achievable and hence unfalsifiable. Two
  controls ship as part of the pinned protocol: a family-identity probe
  (linear-SVM CV classifying FAMILY from the H2 representation, with
  per-family mean feature norms) and a shift-controlled transfer variant
  (per-family per-feature z-scoring, unsupervised). Transfer is reported
  under both raw and family-standardized variants; the regime-contrast
  finding is claimable only if transfer stays at chance under the
  shift-controlled variant too. Confirm the protocol.

### D2 — **[required before Stage B GPU evals]**

- **D2a — K.** `--pairs-per-cell` default 3 → 180 evals/family (360 with
  `--decomposition`). The card does not specify K; sign off on 3 or set
  another value before the post-bank run.
- **D2b — H3 structure-destroyed reference.** Two cells ship: `permuted`
  (derangement of all C² = 36 entries — penalty dominated by destroying the
  untrained identity backbone; round-1 finding) and `permuted_deviation`
  (I + permute(B − I) — preserves the identity backbone and the
  trained-deviation multiset; recommended). Both share the same derangement
  per (family, task, slot) so their contrast isolates the backbone effect.
  The plan JSON carries `h3_structure_reference: PENDING DIRECTOR SIGN-OFF`.
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
- **D2h — `--decomposition` default OFF.** The card's contradiction guard is
  conditional on a D1/D2 conflict. Decide pre-run whether to burn the extra
  180 evals/family unconditionally or only if the conflict materializes
  (deterministic either way under the same plan seed).

### D3 — **[required before labels are generated]**

- **D3a — Pair-selection policy.** The sampler is a deterministic mechanism
  (uniform without replacement over same-family run pairs, seeded,
  vertex-disjoint by default). The reportable design — N, stratification over
  (task_i, task_j) cells, inclusion of same-task pairs as the cross-seed
  reference — is not fixed by the card and must be pre-declared before labels
  exist.
- **D3b — Degradation label definition + binarization.** What "degradation"
  is (merged val loss minus mean of the two native val losses? which eval
  set? which alpha?) is fixed at GPU-eval time and needs sign-off. The
  binarization default is a median split (deterministic but data-dependent);
  if a fixed absolute threshold or an explicit `degraded` column is wanted,
  declare it before labels exist.
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

**156 tests, all passing** (44s):

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
  transfer misses.
- **D2** (`tests/test_asset1_d2.py`): plan determinism and SHA stability;
  bit-exact recipient A/B preservation with donor bridge installed;
  derangement properties; `permuted_deviation` provably isolates trained
  structure (distance ≤ 2‖D‖_F while the full-entry permutation is O(1)
  backbone-dominated); decomposition exact round-trip; Stage B import safety
  (transformers blocked → Stage A unaffected; `--evaluate` refused at the
  gate before any lazy import).
- **D3/D-aux** (`tests/test_asset1_d3_daux.py`, `--selftest` in both tools):
  principal-angle GL(r) gauge invariance to 1e-10; planted-angle selftest
  (full-feature AUC ≥ distance-only + 0.15 and ≥ 0.85); the dependence
  control — naive CV exploits pure run-identity leakage (AUC inflated) while
  group-aware CV sits near chance (≥ +0.15 separation required, group ≤
  0.70); cluster-bootstrap CIs wider than pair-iid under planted dependence;
  D-aux recovers the planted deviation↔gap correlation (r > 0.9 pooled AND
  within-task) and honestly reports "correlation undefined" on the
  zero-deviation bank.

Canonicalizer invariance (~1e-13) was verified separately in
`tests/test_canonicalize.py`; bank trainer coverage lives in
`tests/test_asset1.py`.

## 6. What remains impossible until post-bank

- **Every real-bank number.** The interlock refuses all D1/D2/D3/D-aux
  real-bank invocations until 480/480 COMPLETE. Nothing exploratory under
  `--allow-partial-bank` is reportable.
- **D2 Stage B.** GPU val-loss evaluation needs the complete bank, the two
  base models (HF download), and the H3 reference sign-off (D2b).
- **D3 labels.** Post-merge degradation labels do not exist until the merged
  adapters emitted in Step 5 are GPU-evaluated under a signed-off degradation
  definition (D3b). The prediction harness (Step 7) is fully built and idle
  until then.
- **H2 unblinding.** The analysis runs on completion day, but the H2 claim
  cannot be stated until the representation (H2a) and decision rule (H2c)
  are pinned — which must happen before, not after, the results exist.
- **The card's result tables.** The card's instruction stands: hypotheses are
  locked; fill result tables only.
