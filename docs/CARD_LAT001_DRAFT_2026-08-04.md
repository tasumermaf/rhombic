# CARD (DRAFT) — LAT-001: Is Diameter the Sufficient Statistic for Continuous-Thought Step Count?

**Status: DRAFT for Director grading — NOT REGISTERED. No GPU run is authorized
by this document.** Drafted 2026-08-04 by Meridian (decider) from LAORA's
acceptance-test design pass (measured basis:
`scripts/lat001_task_graph_sizing.py`, landed 16605d1). Registration follows
the Director's grade + a harness smoke on synthetic minima.

## 0. Re-specification notice

The State-of-Play sketch ("does FCC reduce continuous-thought step count vs
cubic?") is re-specified: **directed shortest-path distance `d` — not topology —
is the primary regressor.** Reason, measured at design time: the natural
DAG-by-linear-functional task construction admits a monotone shortcut that is
*stronger in the treatment arm* (0.768 FCC vs 0.643 cubic mid-size; 0.737 vs
0.580 large), and balancing reachability by per-topology edge-thinning destroys
the manipulation itself. Binary reachability is therefore the wrong endpoint
(FCC reachable fraction 0.99–1.00 → no negatives). The endpoint becomes
**next-hop-on-a-shortest-path over reachable ordered pairs**, which gives every
item ground-truth `d` and removes the label-balance conflict entirely.

## 1. Hypothesis and both outcomes

Per the superposition bound (arXiv:2505.12514: D continuous-thought steps solve
reachability at diameter D), the step-count law should be a function of the
directed distance structure alone.

**Primary pre-registered test:** with continuous-thought budget `c` and directed
distance `d` in the model, the topology main effect and the topology×`d`
interaction are both **null** — `k*(d) = d` regardless of how the graph got its
diameter.

- **Outcome A (null holds):** diameter is the sufficient statistic. The
  FCC-beats-cubic step-count advantage follows as a derived corollary at the
  measured directed-distance ratios (mean directed-distance reductions **34.3%**
  at 216/256 and **36.2%** at 512/500; undirected diameter reductions 33.3% /
  38.1% — both from the landed sizing script). Publishable as the quantitative
  link from lattice topology to reasoning step count.
- **Outcome B (topology matters after conditioning on `d`):** diameter is NOT
  the binding constraint in a learned system — the more interesting result, and
  it publishes as such.

## 2. Design

| Parameter | Value | Basis |
|---|---|---|
| Task | next-hop-on-shortest-path over reachable ordered pairs | §0 |
| Task graph | uniform random orientation of every lattice edge; arc-keep p = 1.0, SAME for all topologies | preserves the contrast (measured) |
| Topologies (5) | cubic-6 · FCC-12 · FCC degree-preserving rewire · random-regular @ FCC degree · random-regular @ cubic degree | 2×2 of degree × arrangement + treatment; rewire moves diameter 10→5 / 13→5 at fixed N, E, degree sequence — the sharpest single test |
| Matched sizes | cubic 27 / FCC 32 · 216 / 256 · 512 / 500 | in-tree verified; **context length caps the ladder at N≈500** (FCC-500 ≈ 10k tokens of edge list) — stated here, not discovered mid-campaign |
| Model | 2-layer transformer, from scratch | matches arXiv:2505.12514 |
| Continuous thoughts | `c` sampled uniformly at train, swept at eval | one model per cell, `c` an eval variable |
| Node IDs | **randomly relabeled — MANDATORY** | coordinates make `d` closed-form; a coordinate-reading model performs no search and the measurement is void |
| LR | 3-point grid spanning an order of magnitude, tuned **per topology** before lock | arXiv:2602.04998 (the LAORA gate) |
| Seeds | 8 per cell (pre-registered floor; pilot variance sets the final N by dated amendment) | |

## 3. Controls (all mandatory)

1. **Node relabeling** (above) — the control that decides whether the
   experiment exists.
2. **Coordinate-recoverable positive control** — same runs with coordinates
   exposed; predicted signature: correctness stops improving with `c`. If the
   signature is absent, the instrument is not measuring search, and nothing
   downstream is interpretable.
3. **FCC degree-preserving rewire** — the matched-degree control (identical
   N/E/degree sequence, collapsed diameter).
4. **Two random-regular arms** — complete the degree × arrangement 2×2
   (EE-001's decomposition pattern applied to step count).
5. **Permutation null** — trained checkpoints scored against permuted
   adjacency; must land at base rate. Evaluation-only, zero training cost.

## 4. Runs and cost

| Component | Runs |
|---|---|
| Per-topology LR grid (1 seed × 3 LRs × 5 topologies × 3 sizes) | 45 |
| Main design (5 topologies × 3 sizes × 8 seeds) | 120 |
| Task-construction / c-regime pilot | 6 |
| Coordinate-shortcut positive control | 6 |
| S2-protocol timing pilots (excluded from bank, compute no LAT statistic) | 3 |
| **Total** | **180** |

**Cost: ≤ 4.375 GPU-days PROJECTED — a CEILING, not an estimate.** Basis: the
S2 affine fit (34.921 + 15.5352·params_B min/run) evaluated at ~0.005B = 35.0
min/run — an extrapolation from 2.6–7.6B checkpoints whose ~35-min intercept is
harness overhead a from-scratch 2-layer model largely does not incur. The three
timing pilots run FIRST and restate this to a measurement (< 2 GPU-h at the
ceiling). Electricity ≈ $9.45 ESTIMATE (300 W, $0.30/kWh — both assumptions).
Rented-equivalent $135–$263 PROJECTED.

## 5. Sequencing and exclusions

- **GPU phase queues AFTER the granularity ladder** (plan of record §3, Track
  A). Harness build + synthetic-minima smokes proceed now (Track B, no GPU
  beyond smokes).
- All runs sit outside every bank root, carry `EXCLUDED_FROM_BANK`, and launch
  behind `gpu_guard` (box-wide arbiter, non-negotiable).
- **No 24-cell arm.** Declined on Obedient Builder grounds (the trainer
  crystallizes any pair-specification; formation is a foregone conclusion),
  plus: degree-8 is an *interior* point on the connectivity axis this
  experiment is establishing, and Paper-4's polytope ratios are frozen pending
  the C6b re-measures. A future 24-cell arm must derive a *differentiating*
  task-side prediction (candidate direction: self-duality ⇒ routing and dual
  routing at identical step cost) and register it separately.

## 6. Provenance and open items

- Motivating fact reproduced in-tree twice: `scripts/fcc_diameter_check.py`
  (7c136ad) — MEAN_ASP 26.6%, MEAN_DIAMETER 35.3%; independently re-run by the
  acceptance-test instance at falco a9be933.
- Design facts (directed-distance survival, shortcut confound, balance failure)
  from `scripts/lat001_task_graph_sizing.py` (16605d1) — seeded, reproducible.
- **Open before registration:** (1) ✅ CLOSED 2026-08-05 — novelty review
  complete (`docs/LAT001_NOVELTY_REVIEW_2026-08-05.md`): **UNCLAIMED**, on a
  27-query sweep plus the complete 44-work citation graph of 2505.12514.
  Two binding boundaries: the diameter→step-count relation is CLAIMED AS
  THEORY by 2505.12514 itself (LAT-001 tests its *sufficiency* in a learned
  model — Outcome B is the falsification of sufficiency); and 2509.22343
  ("Transformers Can Learn Connectivity in Some Graphs but Not Others") is
  the closest existing work and MUST be cited (learnability under discrete
  decoding, graph dimension confounded with degree/diameter/N — not
  step count, not continuous thought). The abstract uses the review's
  pinned sentence verbatim; the review's do-not-say list binds. Residual:
  full-text passes on 2602.01148 and 2510.19753 before submission
  (fragment-only fetches; both make diameter/limit-shaped claims).
  Bonus for §3 control 1: 2509.23365 independently publishes vertex-index
  permutation as a bias control — cite as precedent for mandatory
  relabeling. (2) harness smoke on synthetic minima (in build, this pass);
  (3) the timing pilots' measured rate folded into §4; (4) Director's
  grade of this draft, including his pin on the seed floor and the LR
  grid.
