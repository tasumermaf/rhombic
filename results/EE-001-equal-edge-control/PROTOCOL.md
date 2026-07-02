# EE-001: Equal-Edge-Count Random-Graph Control

**Status:** PRE-REGISTERED — this protocol was written and committed to disk
BEFORE the experiment script was executed (L-006 discipline: define pass/fail
before you have data).

**Date pre-registered:** 2026-07-02
**Script:** `scripts/ee_001_equal_edge_control.py`
**Software:** rhombic v0.3.0, networkx 3.4.2, numpy 2.2.6, scipy 1.15.3
**Python:** C:\miniconda3\envs\falco\python.exe (falco conda env)

---

## Motivation

Paper 1 (Rung 1, `results/rung-1/RESULTS.md`) claims the FCC lattice
(12-neighbor, rhombic dodecahedron Voronoi cell) beats the simple cubic
lattice (6-neighbor) at matched node count: ~2.3–2.5× algebraic connectivity
(Fiedler value), ~30% shorter average paths, ~40% smaller diameter — at
~1.9× edge cost.

**The critique (external review):** a 12-neighbor vs 6-neighbor comparison
partly measures "more edges win." The load-bearing control is an
**equal-edge-count random graph**: what does the *rhombic arrangement* of a
fixed edge budget buy over a *random arrangement* of the same budget?

This experiment supplies that control.

## Hypothesis (pre-registered expectation)

Random regular graphs are near-optimal expanders. A random d-regular graph
has Laplacian Fiedler value λ₂ ≈ d − 2√(d−1) (Alon–Boppana / Friedman), which
for d ≈ 10–12 is O(1) and does NOT decay with N. Any finite-dimensional
lattice has λ₂ = Θ(N^(−2/3)) in 3D. Random graphs also have O(log N)
diameter vs Θ(N^(1/3)) for a 3D lattice.

**We therefore EXPECT the random controls to BEAT the FCC lattice on Fiedler
value and path metrics at matched edge count — decisively, and increasingly
so with N.** The FCC advantage over cubic is not "FCC is the best possible
graph at this edge budget" (it demonstrably is not); the pre-registered
honest claim, if the expectation holds, is:

> FCC is the best **spatially-embeddable** arrangement among those tested.
> Random expanders win on abstract graph metrics but require unbounded-length
> wiring: they cannot be embedded in 3D space with bounded local edge length.
> For physical domains (routing fabrics, memory meshes, spatial data
> structures) the spatial-embeddability constraint is the point, and within
> that constraint the rhombic cell wins. Additionally, part of the raw
> FCC-vs-cubic advantage IS attributable to the larger edge budget — the
> honest decomposition separates "budget effect" from "arrangement effect."

## Pre-registered decision criteria

Let R_f = mean Fiedler(control) / Fiedler(FCC) at matched N and edge budget,
per size. The **degree-preserving rewire of the FCC graph itself** (exact
same degree sequence, randomized wiring) is the primary control; ER-G(N,E)
and random d-regular are secondary.

| Outcome (primary control, all sizes) | Verdict | Interpretation |
|---|---|---|
| R_f > 1.10 | **CONTROL WINS** | Expected. FCC's Fiedler advantage over cubic is partly edge budget, partly geometry-constrained arrangement. Claim narrows to: FCC is the superior *spatially-embeddable* topology. Paper 1 must state this control and the narrowed claim. |
| 0.90 ≤ R_f ≤ 1.10 | **PARITY** | Surprising. The rhombic arrangement extracts expander-class connectivity from a spatial layout; claim strengthens substantially. |
| R_f < 0.90 | **FCC WINS** | Very surprising. Would suggest lattice structure beats random wiring at fixed budget; verify for bugs before believing it. |

Same three-band reading applied to mean shortest path (there, LOWER is
better, so control wins means path ratio < 0.90 vs FCC).

**Both outcomes are publishable.** If CONTROL WINS: Paper 1 gains the control
a reviewer would demand, plus the sharpened spatial-embeddability claim. If
PARITY/FCC WINS: a stronger, stranger result — re-verify, then publish.

Secondary pre-registered check — **budget vs arrangement decomposition**:
- Budget effect  = metric(FCC-rewired-random) vs metric(cubic at matched N)
- Arrangement effect = metric(FCC) vs metric(FCC-rewired-random)
The sum (in log space, approximately) recovers the Paper 1 FCC-vs-cubic gap.

## Methodology — matched EXACTLY to Paper 1

- **Fiedler value:** `networkx.algebraic_connectivity(G, weight=None,
  method='tracemin_lu')` — second-smallest eigenvalue of the **raw
  (unnormalized, unweighted) combinatorial graph Laplacian** L = D − A.
  This is what `rhombic/benchmark.py::_fiedler_value` used for Rung 1.
  NOT the normalized Laplacian. Fallback to default method on failure,
  as in benchmark.py.
- **Mean shortest path:** unweighted BFS. Exact all-pairs
  (`nx.average_shortest_path_length`-equivalent) when N ≤ 1100; for the
  largest size, sampled BFS from 512 fixed-seed source nodes (reported as
  sampled — Paper 1 used exact at ≤4096 but our per-graph count here is 17×
  larger, so sampling keeps runtime sane; sampling error on a mean over
  ≥512×(N−1) distances is negligible for ranking purposes).
- **Diameter:** exact (`nx.diameter`-equivalent, max eccentricity) when
  exact BFS was run; otherwise max eccentricity over the 512 sampled
  sources, reported as a lower-bound proxy and labeled as such.
- **Connectivity guard:** metrics computed only on connected graphs. Random
  samples that come out disconnected are resampled with a fresh seed (up to
  20 tries); the disconnection rate is recorded and reported.

## Configurations

FCC lattice sizes (library: `FCCLattice(m)`, N = 4m³ exactly):

| m | N (FCC) | Cubic reference n | N (cubic) |
|---|---------|-------------------|-----------|
| 4 | 256  | 6  | 216  |
| 6 | 864  | 10 | 1000 |
| 8 | 2048 | 13 | 2197 |

Cubic reference is chosen as `round(N_fcc^(1/3))` per side — the same
matched-node-count convention as Paper 1's `matched_lattices`.

Controls, all matched to the FCC graph of that size:

1. **FCC-rewire (PRIMARY):** degree-preserving randomization of the FCC
   graph itself via `nx.double_edge_swap` with nswap = 5×E (each edge
   expected to be swapped ~10 times). Exact same N, E, and degree sequence;
   only the wiring is randomized. Resample on disconnection.
2. **Random d-regular:** `nx.random_regular_graph(d, N)` with
   d = round(2E/N). Same N; E_regular = N·d/2 (closest regular match —
   exact E match impossible; actual E reported).
3. **Erdős–Rényi G(N, E):** `nx.gnm_random_graph(N, E)` — exact same N
   and E, Poisson-ish degree spread.

**Seeds:** 5 per random model per size (seeds 42, 43, 44, 45, 46 as base;
resampling on disconnection increments a retry offset). Mean ± std reported.

## Runtime budget

< 15 minutes total. Fiedler via tracemin_lu is sparse and fast at N ≤ 2048.
BFS dominates; sampling at the largest size keeps it bounded. If runtime
exceeds budget the largest size is dropped, and that is reported.

## Outputs

- `results/EE-001-equal-edge-control/results.json` — raw numbers
- `results/EE-001-equal-edge-control/RESULTS.md` — table, ratios vs FCC,
  verdict per the pre-registered criteria above
