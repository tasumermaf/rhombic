# Paper 2 Interpretation: What the Numbers Say

---

## The Two Scales

The seven experiments separate cleanly into two registers.

At **lattice scale** (Experiments 1 and 5), the FCC advantage amplifies
monotonically as weight heterogeneity increases. The mechanism is
bottleneck resilience: corpus values create near-disconnections that
the cubic lattice cannot route around. FCC's 12-connected topology
provides alternatives. Direction-based weighting — structurally motivated
by the rhombic dodecahedron's 12 faces — produces the strongest
amplification: the Fiedler ratio nearly triples from 2.3x to 6.1x.

At **single-cell scale** (Experiments 2, 3, 4, 6, 7), the RD shows
measurable structure — bipartite sorting, spectral sensitivity, prime
coherence — but these are characterizations, not unique signatures.
The spectral suppression appears across all tested 24-edge polytopes.
The prime coherence is real but the ordering doesn't transfer from
corpus analysis to vertex labeling.

The single cell is a window into the lattice. The lattice is the
phenomenon.

---

## What Amplifies

The headline: corpus weights push the FCC Fiedler advantage from 2.3x
(Paper 1 baseline) to 6.1x (Paper 2, Experiment 5). This is the
largest effect in the project.

Why direction-based weighting works better than edge-cycling
(Experiment 1's 3.2x): direction weighting assigns the SAME weight
to all edges in a direction pair. In FCC, each of the 6 direction
pairs maps to a pair of opposing rhombic dodecahedron faces. This
means a heavy direction pair creates a uniformly strengthened pathway
through the lattice, while a light pair creates a uniformly weakened
one. Edge cycling distributes weights quasi-randomly across directions,
diluting the bottleneck effect.

The consensus metric (6.69x at scale 125) reveals a scale-dependent
effect. At small scale, FCC's richer local connectivity dominates —
every node has more weighted paths to choose from. At scale 1000,
the advantage inverts because the cubic lattice's longer diameter
means more nodes operate independently during consensus, and the
per-neighbor weight dilution (1/13 vs 1/7) catches up. This matches
the unweighted Paper 1 finding and confirms it is structural.

---

## What the Primes Say

Experiment 3 (simple divisibility count, p=0.30) was underpowered.
Experiment 6 (three-tier scoring, exhaustive enumeration) found the
signal: p = 0.000025.

The difference: Experiment 3 tested whether a TV-optimal assignment
happened to put primes near their divisors. Experiment 6 asked the
right question — given a TV-optimal assignment, which prime-to-vertex
mapping maximizes coherence? The answer: exactly one mapping out of
40,320 achieves the maximum score.

Five of eight primes have direct divisibility hits in their optimal
vertex star:
- 67 -> 134 = 2 x 67
- 29 -> 435 = 15 x 29
- 19 -> 133 = 7 x 19
- 17 -> 136 = 8 x 17
- 11 -> 55 = 5 x 11

The remaining three (23, 31, 89) score through cross-card arithmetic
(sum/difference divisibility) and shared prime factors.

The identity mapping (the determined prime-to-Law ordering from the
isopsephy corpus analysis) scores at the null mean (p = 0.46). This
means the ordering matters but does not transfer to vertex indices.
The vertex indices 0-7 are arbitrary labels on the trivalent vertices
of the RD — they have no intrinsic meaning that should correspond to
the Law ordering. The geometry asks for its own arrangement, and that
arrangement is significant.

---

## What the Spectrum Says

Two findings:

1. **Corpus Fiedler suppression is universal.** All five 24-edge graphs
   show corpus values in the lower tail (0.02% - 5.94%). The
   cuboctahedron is even more extreme than the RD. This means the
   corpus values have a spectral property — high variance, heavy tail —
   that creates bottlenecks regardless of graph topology. It is a
   property of the VALUES, not the GRAPH.

2. **High-symmetry graphs show maximal degeneracy breaking.** The RD
   (Oh symmetry) has only 6 distinct eigenvalues out of 14 under
   uniform weights. Corpus weights break all degeneracies — each
   eigenvalue separates. This is interesting because degeneracy
   breaking under perturbation is a marker of how much "information"
   the perturbation carries relative to the graph's symmetry. Corpus
   values carry maximal information for every high-symmetry graph
   tested.

The spectral distance clustering (RD near random, cuboctahedron
isolated, K(4,6) remote) confirms that the RD's spectrum is generic
rather than special. Its bipartite structure (3,4 degrees) does not
produce a distinctive spectral signature under heterogeneous weights.

---

## The Story for the Paper

Paper 1 showed: FCC beats cubic, 2.3x Fiedler, stable across scales.

Paper 2 shows: structured weights amplify the advantage. The
mechanism is bottleneck resilience. Direction-based weighting —
structurally motivated by the Voronoi cell geometry — produces a
6.1x Fiedler ratio, nearly triple the unweighted baseline.

At single-cell scale, the corpus values show real prime-vertex
coherence (p = 0.000025) but the interpretive ordering doesn't
map to vertex labels. The spectral suppression is universal across
polytopes — a property of the corpus weight distribution, not the
RD specifically. What IS specific to the RD is that it tiles space
and produces the lattice where amplification occurs.

The honest nulls (Experiment 3, identity mapping) are as important
as the positives. They show the limits of single-cell analysis and
prevent overclaiming.

---

## Open Questions

1. **Does the optimal prime-vertex mapping carry interpretive meaning?**
   The geometry chose its own arrangement. That arrangement differs
   from the prime-to-Law mapping determined by cross-card evidence.
   Whether the geometry's preference reveals something about the Laws
   that the corpus analysis missed — or simply reflects graph-theoretic
   structure independent of the Law framework — is the next question.

2. **BCC comparison.** The dual lattice. Its Voronoi cell is the
   truncated octahedron (14 faces, 8-connected). It achieves optimal
   sampling in 3D (Petersen-Middleton). How does it respond to
   heterogeneous weights?

3. **Scale limits of consensus inversion.** At what scale does the
   per-neighbor dilution completely eliminate FCC's consensus advantage?
   The transition between 125 (6.69x FCC advantage) and 1000 (0.73x
   cubic advantage) needs finer resolution.

4. **Multiple seeds.** Single-trial benchmarks with seed=42. Confidence
   intervals from multiple seeds would strengthen the statistical claims,
   though the deterministic reproducibility is itself a feature.
