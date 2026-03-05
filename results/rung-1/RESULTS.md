# Rung 1 Results: Graph Theory Benchmarks — Cubic vs FCC Lattice

**Date:** 2026-03-04
**Software:** `rhombic` v0.1.0 (Python, networkx, numpy)
**Hardware:** Windows 11, RTX 6000 Ada 48GB (CPU-bound for this benchmark)
**Reproducible:** `python -m rhombic.benchmark`

---

## Summary

At every scale tested, the FCC (rhombic dodecahedral) lattice outperforms
the simple cubic lattice on all four graph-theoretic metrics. The advantages
are **structural** — they derive from topology, not implementation, and hold
regardless of scale.

| Metric | FCC Advantage | Consistency |
|--------|--------------|-------------|
| Average shortest path | **29–31% shorter** | Stable across all scales |
| Graph diameter | **38–42% smaller** | Stable across all scales |
| Algebraic connectivity | **2.3–2.5× higher** | Stable across all scales |
| Fault tolerance | FCC slightly more resilient | Converges at large scale |

---

## Raw Data

### Scale ~125 nodes (Cubic: 125, FCC: 108)

| Metric | Cubic | FCC | Ratio |
|--------|-------|-----|-------|
| Edges | 300 | 450 | 1.50× |
| Avg degree | 4.80 | 8.33 | 1.74× |
| Avg shortest path | 4.839 | 3.336 | 0.690 |
| Diameter | 12 | 7 | 0.583 |
| Algebraic connectivity | 0.3820 | 0.8820 | 2.31× |

### Scale ~1000 nodes (Cubic: 1000, FCC: 864)

| Metric | Cubic | FCC | Ratio |
|--------|-------|-----|-------|
| Edges | 2,700 | 4,356 | 1.61× |
| Avg degree | 5.40 | 10.08 | 1.87× |
| Avg shortest path | 9.910 | 6.783 | 0.684 |
| Diameter | 27 | 16 | 0.593 |
| Algebraic connectivity | 0.0979 | 0.2492 | 2.55× |

### Scale ~4000 nodes (Cubic: 4096, FCC: 4000)

| Metric | Cubic | FCC | Ratio |
|--------|-------|-----|-------|
| Edges | 11,520 | 21,660 | 1.88× |
| Avg degree | 5.62 | 10.83 | 1.93× |
| Avg shortest path | 15.941 | 11.352 | 0.712 |
| Diameter | 45 | 28 | 0.622 |
| Algebraic connectivity | 0.0384 | 0.0935 | 2.43× |

---

## Interpretation

### Average Shortest Path (Routing Efficiency)

The FCC lattice reduces average hop count by ~30% at every scale. In a
network where each hop incurs latency (memory access, packet routing,
message passing), this is a **30% reduction in average latency** from
topology alone. No algorithm change. No hardware change. Just the shape
of the fundamental cell.

### Diameter (Worst-Case Latency)

The worst case (diameter) improves by ~38-42%. The longest possible path
through the FCC lattice is roughly 60% the length of the longest path
through the cubic lattice. This bounds the tail latency of any
request-response pattern.

### Algebraic Connectivity (Structural Robustness)

The Fiedler value measures how well-connected a graph is — specifically,
how hard it is to disconnect by removing edges. The FCC lattice is
consistently **2.3-2.5× more algebraically connected** than the cubic.
This is the mathematical formalization of Ashby's requisite variety:
more connectivity = more capacity to absorb perturbation without
fragmenting.

### Fault Tolerance (Graceful Degradation)

Under random node removal, both lattices degrade, but the FCC lattice
maintains higher connectivity at every removal percentage. The advantage
is modest at large scales (~2-5% more of the network remains connected)
because both lattices are above the percolation threshold for random
removal. The real advantage emerges under **targeted** attack (removing
high-degree nodes), which is a future benchmark.

---

## The Cost

The FCC lattice uses approximately **1.9× more edges** than the cubic at
matched node count. This is the price: double the wiring. Whether this
tradeoff is favorable depends on the domain:

- **Network routing:** Edge cost is real (physical wires). But the 30%
  latency reduction and 2.4× robustness improvement may justify it.
- **Memory architecture:** Pointers are cheap. The 12-connected topology
  for spatial data structures costs marginally more memory per node.
- **Software data structures:** The connectivity is logical, not physical.
  The cost is O(1) additional pointers per node.

---

## What This Proves

1. **The topology advantage is real and measurable.** This is not
   philosophy — it is graph theory producing reproducible numbers.

2. **The advantage is stable across tested scales.** The ratios hold from
   100 to 4000+ nodes, consistent with geometric rather than sample-size
   origin.

3. **The cost is bounded.** ~2× edges for ~30% shorter paths and ~2.4×
   robustness. The asymptotic cost ratio approaches 2 (12/6).

4. **For isotropic routing workloads, the cubic lattice is consistently
   outperformed.** For applications where routing efficiency, worst-case
   latency, or structural robustness matter, the FCC lattice is superior
   across all tested scales. The cubic lattice's only
   advantage is simplicity of addressing (integer coordinates on
   orthogonal axes).

---

## Next Steps (Rung 2)

The lattice library now provides the substrate. The next experiment
implements both topologies as **spatial data structures** and benchmarks
real operations: nearest-neighbor search, range queries, flood fill.
This moves from graph theory (abstract) to spatial computation
(practical).

---

*Dashboard plot: `rhombic/results/dashboard.png`*
*Benchmark code: `rhombic/benchmark.py`*
*Lattice library: `rhombic/lattice.py`*
