# The Experimental Ladder

Five rungs. Each builds on the last. Each produces a publishable result.

---

## Rung 0: Lattice Library (complete)

Build the substrate. Two lattice topologies — cubic (6-connected) and
FCC/rhombic dodecahedral (12-connected) — implemented as networkx graphs
with matched node counts and bounding volumes.

**Deliverable:** `rhombic/lattice.py`
**Status:** Complete.

---

## Rung 1: Graph Theory Benchmarks (complete)

Measure four fundamental properties on both topologies at multiple scales:

- Average shortest path length (routing efficiency)
- Graph diameter (worst-case latency)
- Algebraic connectivity / Fiedler value (structural robustness)
- Fault tolerance under random node removal (graceful degradation)

**Deliverable:** `rhombic/benchmark.py`, `results/rung-1/`
**Status:** Complete. Results: 30% shorter paths, 40% smaller diameter,
2.4x algebraic connectivity. Scale-invariant.

---

## Rung 2: Spatial Operations (complete)

Both topologies implemented as spatial data structures. Five operations
benchmarked at three scales:

- Nearest-neighbor search (KDTree)
- Range queries (sphere, box)
- Flood fill / region growing
- Spatial hashing and lookup

Results: FCC flood fill reaches **55% more nodes** per hop at 8,000 nodes.
NN queries are **17% faster**. Spatial hash queries are **6% faster**. Range
queries return **24% more nodes** per volume but take 3-5× longer (denser
packing). The routing advantage translates to propagation-dominated workloads
unambiguously. Query-dominated workloads see a density cost.

**Deliverable:** `rhombic/spatial.py`, `results/rung-2/`
**Status:** Complete.

---

## Rung 3: Signal Processing

The Petersen-Middleton theorem (1962) proves FCC sampling is optimal for
bandlimited 3D signals — same samples, higher fidelity, or fewer samples,
same fidelity. The world hasn't adopted it because the cube is the default.

Benchmarks:

- 3D signal reconstruction quality (FCC vs cubic sampling)
- Aliasing resistance at matched sample counts
- Fourier analysis on both lattice topologies

**Deliverable:** `rhombic/signal.py`, `results/rung-3/`

---

## Rung 4: Context Architecture

The ambitious rung. Apply the topology findings to AI context management:

- Token neighborhoods: cubic (fixed window) vs FCC (overlapping, isotropic)
- Retrieval patterns: nearest-neighbor in embedding space on both topologies
- Routing efficiency in multi-agent message passing
- Attention pattern alternatives inspired by lattice connectivity

This is where the research connects to the AI/ML ecosystem and the
Cybernetica argument about computation inheriting Cartesian geometry.

**Deliverable:** `rhombic/context.py`, `results/rung-4/`

---

## Rung 5: Integration and Publication

Synthesize findings across all rungs. Write up the complete argument:

- The cultural genealogy (why the cube persists)
- The empirical evidence (rungs 1-4)
- The cybernetic interpretation (Ashby, Beer, Bateson)
- Practical recommendations for adoption

**Deliverable:** Technical report, blog posts, conference submission.

---

## Design Principles (all rungs)

1. **Cost is always reported alongside benefit.** The FCC lattice uses ~2x
   edges. That cost must be visible in every comparison.

2. **Reproducible by default.** `python -m rhombic.benchmark` reproduces
   every result. No hidden state, no unreported parameters.

3. **Scale invariance tested.** Every metric is measured at multiple scales
   to confirm the advantage is structural, not artifact.

4. **Sparse results are data.** If a rung shows no advantage for FCC, that
   finding is published alongside the wins.
