# E-5 — Coherence-Interpolation Bifurcation Sweep (Pre-registration)

> **Date:** 2026-07-10 (written and committed BEFORE the trainer edit lands and
> BEFORE any run launches)
> **Revision:** v1.1 same day, pre-commit and pre-launch — WL-001 precedent
> citation made precise against `docs/PAPER4_EXPOSURE_CLASSIFICATION.md`
> (rows 27–29), reference bands added to §4, and the design-fork ruling
> recorded in §2. Predictions unchanged in substance.
> **Status:** PRE-REGISTERED — predictions frozen before data collection
> **Provenance:** shortlist item 5 of the 2026-07-07 literature-mapping memo
> (falco-side; pillar item P1-E4). Sanctioned to start on Hermes "if r3 clears
> early" — T-001r3 completed 2026-07-08 with envelope PASS.
> **Hardware:** Hermes RTX 4090 Laptop 16GB, sequential runs, ~3.6 GPU-days.
> **Lane:** Paper 4 / BM battery. Not part of the asset-1 analysis chain; no
> bank interlock applies. Amendments after launch are dated edits, never
> silent revisions (L-006).

## 1. Question

Steersman-guided bridge training lands in discrete regimes — block-diagonal
coherence (endpoint co/cross above 4×10⁴:1 in T-001), hierarchical (~1:1),
collapse (~1e-5:1) — with no intermediate landings observed to date. But no
experiment has *driven* the system between basins with a graded control
parameter. E-5 interpolates the **coherence of the contrastive pair
specification** and asks: is the transition between structured and
unstructured endpoints **all-or-none** (bimodal landings, empty intermediate
band) or **graded** (endpoints track the control parameter smoothly)?

A training-time bifurcation diagram over a spec-coherence axis is a figure
no PEFT paper has published.

## 2. Control parameter — pair-correctness f (exact operationalization)

The n=8 tesseract contrastive spec is 4 co-planar (attract) + 24 cross-planar
(repel) pairs = 28 labeled pairs. For a run at pair-correctness **f**:

1. Select `k = round((1-f) * 28)` pairs uniformly at random (dedicated RNG,
   seeded `run_seed * 100003 + 51`; the draw never touches Python/NumPy/torch
   global RNG streams — training stochasticity is untouched).
2. **Shuffle the labels among the selected pairs** (count-preserving: the
   global 4/24 attract/repel budget is exact at every f). Unselected pairs
   keep their true labels.

Properties, stated in advance:
- **f = 1.0 is a byte-identical no-op** — no RNG consumed, pair lists
  untouched (guard-tested).
- **f = 0** preserves label counts but randomizes the assignment: the 4
  attract pairs are generically no longer a disjoint channel pairing (a
  channel can be attracted to two mutually-repelled partners) — coherence is
  destroyed, not inverted. This is deliberately NOT a deterministic label
  flip, which would hand the optimizer a coherent (merely wrong) target. The
  precedent is WL-001 [PAPER4_EXPOSURE_CLASSIFICATION.md rows 27–29]: a
  coherent random partition of channels crystallized a 3+3 eigenvalue split
  "in the wrong direction" (structure formed on its own spec) while reading
  co/cross ≈ 8.709e-6 against the TRUE spec — coherent-but-wrong specs
  crystallize; only incoherence should prevent crystallization. The f-axis
  must therefore destroy coherence, and structure must be measured against
  the trained spec (§4), not only the true one.

**Design-fork ruling (recorded for the record):** "pair-correctness" here
operates on the **contrastive pair specification** (the co/cross channel-pair
labels of the Steersman loss), not on BM-004 transit-corpus data pairs. The
memo's item 5 specifies *Steersman runs* measured by regime landing
(co/cross, Fiedler), and the two named basins are Steersman-training regimes
produced by pair-spec variation (T-001 true spec vs WL-001/R-001 wrong
specs). The data-pair alternative would additionally require editing the
BM-004 corpus builder, which sits under a Director-approved frozen
pre-registration; this design touches nothing Director-locked.
- A shuffle can fix some labels by chance, so realized spec agreement
  exceeds nominal f slightly at low f. The realized corrupted pair lists and
  realized agreement fraction are recorded in each run's `config.json`.

Trainer support: dated additive edit to `scripts/train_cybernetic.py` adding
`--pair-correctness` (default 1.0), applied where the co/cross pair lists are
computed, with the corruption record written into `config.json` and printed
in the run header. Guard tests prove default-path byte-identity.

## 3. Design

| Axis | Values |
|------|--------|
| pair-correctness f | 0, 0.25, 0.5, 0.75, 1.0 |
| seed | 42, 43, 44 (corruption draw AND training seed — 3 independent draws per f) |
| runs | **15**, sequential |

All other hyperparameters pinned to the T-001 configuration exactly (re-read
from `results/T-001-full-r3/results.json` this session): TinyLlama-1.1B-Chat,
rank 24, n_channels 8, bridge_mode identity, contrastive auto (tesseract),
Steersman default preset, lr 2e-4, batch 2 × grad-accum 8, warmup 200,
max_seq_len 512, alpaca-cleaned, feedback interval 100, checkpoint every 100 —
**except `max_steps = 3000`** (calibration below). No emanation, no dynamic
bridge.

**Step-count calibration [re-read from r3 artifacts, not from memory]:**
r3 (f≡1, seed 42) logged co/cross 1005.7 by step 300, 5088.8 at step 2500,
5144.5 at step 3000, against ~1:1 hierarchical and ~1e-5:1 collapse
references — basin membership is unambiguous at 3000 steps, three orders of
magnitude of separation. Measured rate 6.88 s/step → ~5.7 h/run → ~86 h
total ≈ 3.6 GPU-days (memo budget: 3–5).

**Auxiliary reference:** r3's step-3000 checkpoint (co/cross_true 5144.5,
seed 42, f≡1, 10k-step run truncated at the same step index) is a fourth
f=1 datum at matched depth, at zero cost.

## 4. Endpoints and metrics

- **Primary — co/cross_trained at step 3000:** mean |co-entry| / mean
  |cross-entry| over the bridge matrices, computed against the pair labels
  the run actually optimized (the corrupted spec). Computed post-hoc from
  saved `bridge_final_*.npy` + the `config.json` corruption record,
  aggregated across adapters exactly as the trainer's logged ratio is
  (mean over per-adapter finite ratios).
- **Secondary — co/cross_true:** the trainer's natively logged
  `co_cross_ratio` trajectory (its pair indices are hardcoded to the true
  n=8 spec, independent of training labels — verified in code this session:
  the ratio helper derives pairs from `_coplanar_crossplanar_indices(n)`,
  not from the configured lists).
- **Secondary — fiedler_mean** trajectory (logged), and **mean coupling
  magnitude** mean(|co|, |cross| pooled) at endpoint, to separate collapse
  (magnitudes → 0) from hierarchical (~1:1 with live magnitudes) within the
  unstructured class.
- At f = 1, co/cross_trained ≡ co/cross_true by construction.

**Pre-registered classification bands (endpoint co/cross_trained):**

| Band | Range | Reading |
|------|-------|---------|
| Structured | ≥ 10³ | BD-class coherence on the trained spec |
| **Intermediate** | [10¹, 10³) | the band bistability predicts stays empty |
| Unstructured | < 10¹ | hierarchical or collapse (split by magnitude) |

Why the primary is the *trained*-spec ratio: WL-001 demonstrates that
co/cross_true alone cannot distinguish "no structure formed" from "structure
formed on a different spec" — a wrong-spec crystallization reads ~1e-5 on
the true metric while its eigenvalue split shows full 3+3 structure
[PAPER4_EXPOSURE_CLASSIFICATION.md rows 27–29]. Reference values for the
secondary metrics, re-read from that document this session: fiedler_mean —
BD-regime runs 1.1e-5 (O-001) to 8.93e-5 (H-ch6); WL-001 collapse-class
1.2552e-5; spectral-attractor band 0.0836–0.1019. Fiedler separates the
attractor band from everything else but does NOT separate BD from collapse;
the magnitude metric carries that split. One caveat inherited from the
source: the WL-001/R-001 endpoint depths are ADAPTIVE-GOVERNED (pre-C6b-fix
controller exposure); E-5 runs under the current detector, and its own f=1
and f=0 endpoints re-anchor both basins under current code.

## 5. Pre-registered predictions (frozen 2026-07-10)

- **P1 (bimodality):** at most 2 of 15 runs land in the intermediate band;
  the remainder split across Structured and Unstructured with both occupied.
- **P2 (controls internal to the grid):** all three f=1.0 runs land
  Structured (positive control — if any fails, the sweep is invalid at this
  depth; re-run longer as a dated amendment before any interpretation).
  All three f=0 runs land Unstructured on co/cross_trained (negative
  control — the weaker prior of the two, stated honestly: WL-001 shows
  coherent-wrong specs crystallize, but an f=0 shuffled label set is not a
  partition and carries frustrated triangles; whether crystallization
  survives incoherence is part of what the sweep measures. An f=0 Structured
  landing would be a striking non-predicted outcome, reported as such).
- **P3 (exploratory):** a critical region exists inside f ∈ [0.25, 0.75];
  near it, same-f runs split across basins by seed — the hallmark of
  bistability — rather than co-landing at intermediate values.
- **Falsifier (graded alternative):** ≥ 5 of 15 runs inside the intermediate
  band, or endpoint log-ratios that track f monotonically *through* the band
  (intermediate f landing at intermediate ratios, Spearman across runs with
  the band populated) → the transition is graded, not all-or-none. A graded
  result is publishable closure: the all-or-none reading of the four-regime
  taxonomy would be an artifact of sampling only extreme specs.

No other comparison on this sweep is confirmatory.

## 6. Artifacts

```
results/E-5-bifurcation/
├── sweep_manifest.json        run ledger (status per run, resumable)
├── f{F}_s{SEED}/              one trainer output dir per run (results.json,
│                              bridge_final_*.npy, config.json w/ corruption record)
├── f{F}_s{SEED}.log           full training log per run
├── endpoints.json             post-hoc co/cross_trained + all secondary endpoints
└── RESULTS.md                 bifurcation table + basin classification
```

Runner: `scripts/e5_bifurcation_sweep.py` — sequential chain, manifest
updated after every run (bounded cycles; kill-safe, resumes skipping
COMPLETE), invokes `train_cybernetic.py` via subprocess with the pinned
arguments, computes endpoints after each run.

*Amendments below this line are dated edits.*

---

**Amendment 2026-07-10 (pre-launch, before any run; findings from the
four-lens fresh-context adversarial verification of the implementation —
verdicts 4/4 PASS, 0 blockers, 0 majors):**

1. **NonFinite band added.** A NaN endpoint (diverged/defective bridges) is
   not a basin: it classifies as **NonFinite**, is tallied separately, and
   never enters the P1/P2 counts. An **inf** endpoint (every per-adapter
   cross mean ≤ 1e-12) remains Structured (inf ≥ 10³ is band-consistent),
   but any Structured row whose mean coupling magnitude is < 1e-6 is flagged
   for manual review — the collapsed-bridge-inf signature must not silently
   read as maximal separation.
2. **Realized-agreement scale stated precisely.** At f = 0 the
   count-preserving shuffle over the unbalanced 4/24 label budget leaves
   ≈ 75% of labels fixed in expectation ((4/28)² + (24/28)² ≈ 0.755;
   measured 0.714–0.786 for seeds 42–44). Nominal f is the *coherence*
   axis (how much of the spec is exempt from shuffling), NOT the
   realized-agreement scale; realized agreement is recorded per run and the
   confirmatory analysis must not conflate the two.
3. **r3 auxiliary reference carries the detector caveat.** The r3 step-3000
   datum (§3) was produced by the March trainer on Hermes — its config.json
   lacks `steersman_detector_version`, i.e. detector v1 by definition — so
   it inherits the same ADAPTIVE-GOVERNED caveat as the WL-001/R-001
   endpoints. E-5's own f=1 runs under the current trainer remain the
   operative positive control.
4. **Controller-spec observation recorded.** Steersman Control Law 2 adapts
   the contrastive weight from the *true-spec* co/cross ratio while the
   loss optimizes the *corrupted* spec — effective contrastive pressure
   trajectories therefore vary systematically along the f-axis. This is
   within the frozen design (§3 pins the default preset) and identical
   machinery governed the precedent runs; the confirmatory analysis notes
   it when interpreting basin depths.
5. **Ops guards.** Resuming the sweep with a different `--steps` than the
   manifest records is refused (no mixed-depth sweeps); a defective run dir
   no longer blocks endpoint computation for other runs (errors recorded in
   endpoints.json); `--pair-correctness` outside [0,1] is rejected at the
   CLI.
