# BM Battery — Forward Experiment Plan

> **Date:** July 3, 2026
> **Status:** Pre-registered plan. BM-003 Configs A/B are implementation-complete
> and awaiting deployment; everything else is design.
> **Context:** This plan operationalizes the program's April 2026 architectural
> pivot (see `results/BM-003/PROTOCOL.md`) and the July 2026 re-evaluation.
> Discipline per `LEARNINGS.md` L-006: pass/fail criteria are pre-registered
> before any run; nulls are published.

---

## The question this battery answers

The program's own null results (Exp 2.5, L-001) established that geometric
structure does not *emerge* from generic data under the standard, rotation-
symmetric bridge parameterization. The Steersman line (Papers 3–4) showed
structure can be *programmed* by a controller — for any coherent topology, at
benchmark parity (BM-001), but with no mechanism connecting the geometry to
task performance.

That leaves the original architectural question genuinely open:

> **Does a structural geometric prior — topology fixed by construction, no
> auxiliary losses, the task deciding everything — help, hurt, or do nothing?
> And does the answer change when the data actually carries structure the
> geometry can serve?**

The governing principle (which the program's null results discovered
independently and the equivariance literature states generally): *a symmetry
prior helps exactly when the task is invariant under that symmetry, and costs
you when it is not.* Testing that principle in adapter weight space requires
both factors — the prior AND the matched data. Neither has been tested with
the other. This battery does it factorially.

## Sequence

### BM-000 — Null-model calibration (CPU only; runs first)

No topology metric in this program currently has a chance baseline. Before
any further claims: empirical distributions of co/cross ratio, bridge Fiedler
value, block-diagonal score, and asymmetry ratio over (a) random Gaussian
bridges, (b) random-mask sparse bridges matched to `rd_graph` sparsity,
(c) bridges from standard-LoRA-equivalent training (frozen-identity runs),
and (d) **degree-preserving rewires** — EE-001's control, adopted July 3 as
the battery's canonical null family per the Director's review: every
structured-vs-unstructured comparison reports against the rewire, not only
against the cube. Every metric reported anywhere gets a percentile against
these nulls.
*Deliverable:* `results/BM-000/` with distributions + a reusable
`scripts/bm000_null_model.py`.

### BM-003 — Structural prior vs. control (as protocoled, plus one arm)

Per `results/BM-003/PROTOCOL.md` (April 7, 2026): Qwen2.5-7B-Instruct,
alpaca-cleaned, 10k steps, rank 24, n=6, seed 42, LM loss only.

| Config | Bridge | What it isolates |
|--------|--------|------------------|
| A | Frozen identity (exact standard LoRA) | Control |
| B | `rd_graph`: fixed RD adjacency mask × learnable edge weights | The structural prior |
| C | `rd_graph` on CodeAlpaca, edge weights seeded from B | Topology transfer across tasks |
| F | **Free dense 6×6 bridge, LM loss only** (added Jul 2026) | Any-learnable-bridge effect, no prior, no controller |
| G | **Shuffled-adjacency mask** (amendment 2026-07-07) | Hub attack, trained level — the SVFT-Random-style arm below, now theory-motivated |
| H | **Expander mask (K3,3)** (amendment 2026-07-07) | Hub attack, trained level — maximal-Fiedler motif |

**Amendment 2026-07-07 (dated edit per L-006; Director-approved
2026-07-07, `docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md` A4):**
Configs G/H (exact masks, ±0.5% topology-specificity band with all three
outcomes pre-stated, Config G's disclosed (4,5) overlap with the RD
co-planar set) plus a dissociation eval endpoint with the task-class
assignment **frozen and timestamped 2026-07-07** (workspace-dependent =
{GSM8K-direct}; automatic = {SST-2, MMLU, ARC-C, HellaSwag, WinoGrande})
and a >2pp bridge-ablation criterion with a pre-stated honest null. Full
text: `results/BM-003/PROTOCOL.md` (Amendment section). Null calibration
already run: `results/BM-000b-hub-motifs/` (BM-000b — no untrained motif
ensemble reproduces any trained headline; honest negative 0/48 recorded).

Config F completes the design: {no bridge, free bridge, structured bridge}
under identical LM-only training. (It was run once at 1.5B/2k steps in Exp 1
— parity — but never at battery scale.) Labels D and E are reserved: D is an
internal initialization-variant arm (results not published); E is the revived
BM-002 seeded-transfer question.

**Positioning (added July 3, 2026, from the fourth-expansion sweep,
`docs/LITERATURE_WATCH_2026-07-03.md`):** Config F is architecturally the
**MoSLoRA** configuration (arXiv:2406.11909 — learnable dense mixer between
learnable A/B) and will be cited as such. Config B's nearest prior art is
**SVFT** (arXiv:2405.19597 — fixed sparsity patterns × learnable values,
but between *frozen* SVD factors and with non-semantic patterns);
**SVFT-Random-style matched-parameter random masks** are the required
additional ablation before publication (a shuffled-adjacency arm at equal
edge count). No found work anticipates the full BM-003 combination —
learnable A/B + semantically chosen graph-adjacency mask + LM loss only.
For BM-004, arXiv:2606.01090 (May 31, 2026) is concurrent evidence outside
PEFT that misaligned symmetry priors are actively harmful and aligned ones
pay; cite as concurrent, differentiate on setting (adapters over a
pretrained LM) and prior type (graph adjacency vs. group equivariance).

Pre-registered decision bands (from the BM-003 protocol): B ≥ A on benchmark
mean → the prior is free or better; within −1% → proceed; −1% to −3% →
investigate; worse than −5% → the topology constrains too much.

**Interpretive guard, stated in advance:** a null for B on *generic* data
does not falsify the structural-prior thesis — generic instruction text has
no symmetry for the prior to serve. It bounds the cost of the prior. The
thesis is only tested by BM-004.

### BM-004 — Geometry-matched data (the two-factor test)

Train `rd_graph` and control adapters on purpose-built **paired data that
encodes a geometric relation** (input/target pairs constructed so the mapping
between them respects the lattice symmetry — the paired control-data pattern
from production video-LoRA practice), against the same data shuffled to break
the pairing, and against generic data. 2×2 core: {structural prior, no prior}
× {geometry-matched data, generic data}. **The interaction term is the entire
question.** If the prior helps only on matched data, the governing principle
is confirmed in weight space. If the interaction is null, the original
form-first conjecture for adapter weight space is closed — honestly, by the
right experiment rather than a proxy.

**Three hard requirements on the data arm (adopted July 3 from the
Director's review — without them a matched-arm null is unfalsifiable in the
other direction):**
1. **Operational pre-specification.** Before any run, state formally what
   symmetry/invariance the `rd_graph` mask encodes (the RD face-pair
   coupling structure) and the checkable criterion that makes a dataset
   invariant under it. "Not matched enough" must not be available as a
   post-hoc escape.
2. **Positive-control manipulation check.** Build a synthetic task
   *provably* invariant under the mask's structure and demonstrate the
   prior helps there before any natural-data arm runs. If the prior cannot
   beat baseline on data constructed to be invariant under it, the
   mechanism is broken and no natural-data result is interpretable.
3. **Wrong-symmetry arm.** A permuted-mask (wrong-topology) prior as the
   mandatory negative control — arXiv:2606.01090's finding that misaligned
   priors are *actively harmful* is the methodology for this arm, not
   merely concurrent work; it measures the same interaction term outside
   PEFT.

### BM-002-E — Seeded-bridge transfer (revived)

The abandoned BM-002 third config: TeLoRA on code with Alpaca-seeded bridges,
now runnable via `--seed-bridges` (which, along with `--dataset`, is now
persisted into `config.json` so runs are self-documenting).

## Known issues carried into this battery

1. **Steersman controller stability detection is defective** (identified in
   internal review, April 2026): the adaptive controller can declare STABLE
   while its controlled metric is still moving; related to the Control Law 2
   telemetry bug documented in the 24C-001 recovery. No Steersman arm runs in
   this battery (all arms are LM-loss-only or frozen), but the defect must be
   fixed before any future Steersman run and is noted in Paper 4 limitations.
2. **EE-001** (equal-edge random-graph control, run July 2, 2026,
   `results/EE-001-equal-edge-control/`) narrowed Paper 1's lattice claim to
   the spatially-embeddable graph class. The same discipline applies here:
   BM-000's random-mask baselines are this battery's equivalent control.

## Relationship to the two research tracks

- **Track 1 — the structural prior** (this battery): geometry as a road, not
  a destination. BM-000 → BM-003 → BM-004.
- **Track 2 — the Steersman as its own subject** (Paper 4): topology
  *programmability*, the four-regime taxonomy, homeostatic maintenance of
  programmed structure, and bridge-based diagnostics (task identifiability at
  72.3% pooled LOO; deviation↔generalization-gap r = 0.888). Continues
  independently; nothing in Track 1 depends on it.

*Pre-registered July 3, 2026. Amendments to this plan are dated edits, not
silent revisions.*
