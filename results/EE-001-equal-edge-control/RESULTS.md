# EE-001 Results: Equal-Edge-Count Random-Graph Control

**Date run:** 2026-07-02
**Protocol:** `PROTOCOL.md` (pre-registered before execution; L-006 discipline)
**Script:** `scripts/ee_001_equal_edge_control.py` — total runtime 60.1 s
**Raw data:** `results.json`
**Software:** rhombic v0.3.0, networkx 3.4.2, numpy 2.2.6, scipy 1.15.3

**Methodology (matched to Paper 1 / Rung 1 exactly):** Fiedler value =
second-smallest eigenvalue of the **raw unnormalized, unweighted
combinatorial Laplacian** via `nx.algebraic_connectivity(method='tracemin_lu')`,
identical to `rhombic/benchmark.py`. Sanity anchor: this run reproduces the
published Rung 1 FCC value at N=864 to four decimal places (0.2492).
Paths/diameter: exact all-pairs BFS at N ≤ 1100; at N = 2048, sampled BFS
from 512 fixed sources (mean path is a sampled estimate; diameter is a
max-eccentricity **lower-bound proxy** — both flagged with † below).
Random models: 5 seeds each, mean ± std. **Zero disconnected samples across
all 45 random graphs** (no repairs needed).

---

## Data

### Size 1 — FCC m=4 (N=256), paths exact

| Graph | N | E | Fiedler | mean path | diameter | Fiedler ×FCC | path ×FCC |
|---|---|---|---|---|---|---|---|
| **FCC lattice** | 256 | 1,176 | 0.5293 | 4.489 | 10 | 1.00 | 1.000 |
| Cubic lattice (ref) | 216 | 540 | 0.2679 | 5.861 | 15 | 0.51 | 1.306 |
| FCC degree-preserving rewire | 256 | 1,176 | 2.2470 ± 0.1265 | 2.737 ± 0.002 | 5.0 ± 0.0 | **4.25** | **0.610** |
| Random 9-regular | 256 | 1,152 | 3.5157 ± 0.0693 | 2.751 ± 0.004 | 4.0 ± 0.0 | **6.64** | **0.613** |
| Erdős–Rényi G(N,E) | 256 | 1,176 | 1.7977 ± 0.5146 | 2.733 ± 0.007 | 5.0 ± 0.0 | **3.40** | **0.609** |

### Size 2 — FCC m=6 (N=864), paths exact

| Graph | N | E | Fiedler | mean path | diameter | Fiedler ×FCC | path ×FCC |
|---|---|---|---|---|---|---|---|
| **FCC lattice** | 864 | 4,356 | 0.2492 | 6.783 | 16 | 1.00 | 1.000 |
| Cubic lattice (ref) | 1,000 | 2,700 | 0.0979 | 9.910 | 27 | 0.39 | 1.461 |
| FCC degree-preserving rewire | 864 | 4,356 | 2.3126 ± 0.1306 | 3.192 ± 0.001 | 5.0 ± 0.0 | **9.28** | **0.471** |
| Random 10-regular | 864 | 4,320 | 4.0658 ± 0.0373 | 3.221 ± 0.001 | 5.0 ± 0.0 | **16.32** | **0.475** |
| Erdős–Rényi G(N,E) | 864 | 4,356 | 1.0370 ± 0.3506 | 3.179 ± 0.002 | 5.6 ± 0.5 | **4.16** | **0.469** |

### Size 3 — FCC m=8 (N=2048), paths sampled†

| Graph | N | E | Fiedler | mean path† | diameter† | Fiedler ×FCC | path ×FCC |
|---|---|---|---|---|---|---|---|
| **FCC lattice** | 2,048 | 10,800 | 0.1439 | 9.093 | ≥22 | 1.00 | 1.000 |
| Cubic lattice (ref) | 2,197 | 6,084 | 0.0581 | 12.882 | ≥36 | 0.40 | 1.417 |
| FCC degree-preserving rewire | 2,048 | 10,800 | 2.4067 ± 0.1069 | 3.527 ± 0.003 | 5.4 ± 0.5 | **16.73** | **0.388** |
| Random 11-regular | 2,048 | 11,264 | 4.7077 ± 0.0109 | 3.487 ± 0.001 | 5.0 ± 0.0 | **32.72** | **0.384** |
| Erdős–Rényi G(N,E) | 2,048 | 10,800 | 1.5183 ± 0.3095 | 3.504 ± 0.004 | 5.8 ± 0.4 | **10.55** | **0.385** |

† Sampled BFS (512 sources); diameter is a lower-bound proxy at this size.

Notes on edge matching: the rewire and ER controls match the FCC edge count
exactly. The random d-regular control uses d = round(2E/N), giving E within
−2.0% to +4.3% of the FCC budget (closest regular match; exact match is
impossible for a regular graph).

---

## Pre-Registered Verdict: CONTROL WINS

Per `PROTOCOL.md`, the primary control is the degree-preserving rewire of
the FCC graph itself (identical N, E, and degree sequence — only the wiring
differs). The pre-registered threshold for CONTROL WINS was R_f > 1.10.

**Measured R_f (rewire/FCC Fiedler): 4.25, 9.28, 16.73 — growing with N.**
Mean-path ratios: 0.610, 0.471, 0.388 (control paths less than half of
FCC's at the largest size). Every control beats the FCC lattice on every
abstract graph metric at every size, usually by an order of magnitude on
Fiedler value. The pre-registered expectation held: random graphs at a
fixed edge budget are near-optimal expanders (Laplacian λ₂ ≈ d − 2√(d−1),
independent of N, vs Θ(N^(−2/3)) for any 3D lattice), and it shows.

**The interpretation that was pre-registered for this outcome now applies:**

1. **The FCC-vs-cubic comparison partly measures edge budget, and Paper 1's
   headline numbers cannot be read as "FCC is a near-optimal graph."** It
   is nowhere near optimal in the space of all graphs at that budget. Its
   own edges, randomly rewired, deliver 4–17× its algebraic connectivity.

2. **The honest claim narrows to spatial embeddability — and that constraint
   is the point.** The FCC lattice uses only nearest-neighbor edges: every
   edge has Euclidean length exactly a/√2 ≈ 0.707a. The rewired control,
   measured on the same node positions, requires mean edge lengths of
   **3.49× / 5.34× / 7.16×** the FCC edge length at the three sizes
   (post-hoc descriptive measurement, seed 42) — and this factor grows as
   N^(1/3), unboundedly. A random expander cannot be embedded in 3D space
   with bounded-length local wiring; a lattice can. For physical domains —
   routing fabrics, memory meshes, sensor networks, spatial data structures,
   anything where an edge is a wire, a pipe, or a physical adjacency — the
   comparison class is spatially-embeddable graphs, and within that class
   the Rung 1 result stands: FCC beats cubic 2.0–2.5× on Fiedler and ~30%
   on paths at matched node count.

3. **Budget vs arrangement decomposition (pre-registered secondary check).**
   Fiedler, per size (m=4 / m=6 / m=8):
   - FCC vs cubic (Paper 1's comparison): 1.98× / 2.55× / 2.48×
   - Rewired-FCC-budget vs cubic (budget + randomness): 8.39× / 23.63× / 41.41×
   - FCC vs rewired-FCC (cost of spatial arrangement): 0.236× / 0.108× / 0.060×

   Read honestly: at a fixed edge budget, *random wiring crushes any
   lattice*, and the gap widens with N. The 2–2.5× FCC-over-cubic advantage
   is real but lives entirely inside the spatially-embeddable regime. The
   "budget effect" as operationalized here conflates extra edges with
   randomization and should not be quoted as a pure budget number; the
   clean statements are the two bullets above.

4. **What Paper 1 must say.** (a) Report this control. (b) Replace any
   implication that 12-connectivity is *per se* superior with the narrowed
   claim: "among spatially-embeddable lattice topologies at matched node
   count, FCC dominates cubic on connectivity, path length, and diameter,
   at ~2× edge cost; unconstrained random graphs at the same budget are
   far better on abstract metrics but require physically unbounded wiring."
   (c) Cite the wire-length numbers as the quantitative form of the
   embeddability argument.

## Caveats

- N=2048 path metrics are sampled (512 sources); diameters at that size are
  lower-bound proxies. Ranking is unaffected — the gaps are order-of-magnitude.
- The random d-regular control's edge count deviates up to 4.3% from the FCC
  budget (regular-graph parity constraint). The rewire control, which matches
  the budget and the degree sequence exactly, is the primary control and shows
  the same qualitative result.
- ER Fiedler has high seed variance (std up to 0.51) because λ₂ is governed by
  the minimum degree; the mean is still ≥3.4× FCC everywhere.
- Wire-length measurement (item 2) is post-hoc descriptive, not pre-registered;
  it quantifies the embeddability argument but decides nothing.
- The FCC-vs-cubic ratio at m=4 here (1.98×) is slightly below Rung 1's
  published 2.31× at its smallest scale because the matched pairs differ
  (Rung 1: FCC 108 vs cubic 125; here: FCC 256 vs cubic 216). Same family,
  no contradiction.

## One-line summary

A fixed edge budget randomly wired beats the FCC lattice by 4–17× on
algebraic connectivity and halves path lengths — so Paper 1's claim
correctly narrows to: **FCC is the best spatially-embeddable arrangement
tested, and spatial embeddability (bounded-length physical wiring, here
7× cheaper at N=2048 and growing) is exactly what physical domains demand.**
