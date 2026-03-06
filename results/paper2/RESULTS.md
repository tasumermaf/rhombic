# Paper 2 Results: Pure Number Architecture

> Weighted extensions to "The Shape of the Cell." Seven experiments
> testing whether structured edge weights amplify or attenuate the
> FCC topology advantage.
>
> All results reproducible: `cd rhombic && python scripts/run_experiments.py`
> (seed=42, ~190s on consumer hardware)

---

## Experiment 1: Weighted FCC vs Cubic (Lattice Scale)

Cycling edge weights across lattice edges. Four distributions, three scales.

| Scale | Distribution | Fiedler Ratio | Path Ratio | Consensus |
|-------|-------------|---------------|------------|-----------|
| 125 | uniform | 2.31x | 0.69 | 1.00x |
| 125 | random | 2.49x | 0.66 | — |
| 125 | power_law | 2.67x | 0.67 | — |
| 125 | **corpus** | **3.17x** | **0.48** | — |
| 1000 | uniform | 2.55x | 0.68 | — |
| 1000 | random | 2.64x | 0.60 | — |
| 1000 | power_law | 3.06x | 0.64 | — |
| 1000 | **corpus** | **3.11x** | **0.39** | — |
| 8000 | uniform | 2.28x | — | — |
| 8000 | **corpus** | **3.18x** | — | — |

**Finding:** Heterogeneous weights amplify the FCC advantage monotonically.
Corpus weights push the Fiedler ratio from 2.3x (uniform) to 3.2x and
the path advantage from 30% to 60%.

---

## Experiment 2: Optimal Assignment on Rhombic Dodecahedron (Single Cell)

Total variation minimization via simulated annealing (20 restarts, 1000
iterations per restart).

| Distribution | Optimal TV | Random Mean TV | Improvement | p-value |
|-------------|-----------|---------------|-------------|---------|
| random | 8.71 | 17.73 +/- 1.22 | 50.9% | <0.0001 |
| power_law | 6.89 | 10.08 +/- 0.50 | 31.7% | <0.0001 |
| corpus | 8.12 | 13.05 +/- 0.74 | 37.8% | <0.0001 |

**Control:** A random 24-edge graph achieves TV = 6.80 on corpus values
(vs RD's 8.12). The RD's bipartite constraint (every edge connects a
degree-3 to a degree-4 vertex) restricts rather than optimizes the
assignment. The RD sorts because it must, not because it's best at sorting.

---

## Experiment 3: Prime Coherence at Vertex Stars (Single Cell)

TV-optimal edge assignment scored for prime divisibility at 8 trivalent
vertices.

| Metric | Value |
|--------|-------|
| Optimal coherence score | 7 |
| Null mean | 6.06 +/- 0.85 |
| p-value (one-sided) | 0.30 |

**Finding:** Not significant. The test is underpowered: 8 vertices, 24
edges, 8 primes. The combinatorial space is too small for 8 primes to
separate from noise using a simple divisibility count. This motivated the
redesign in Experiment 6.

---

## Experiment 4: Spectral Properties (Single Cell)

Weighted Laplacian spectrum of the RD under four distributions.

| Distribution | Fiedler | Spectral Gap | lambda_max | Distinct Eigenvalues |
|-------------|---------|-------------|-----------|---------------------|
| uniform | 1.4384 | 0.2055 | 7.00 | 6 / 14 |
| random | 0.5364 | 0.1126 | 4.77 | 14 / 14 |
| power_law | 0.1041 | 0.0371 | 2.81 | 14 / 14 |
| corpus | 0.0863 | 0.0314 | 2.75 | 14 / 14 |

Corpus Fiedler percentile vs 10K random weights: **0.08%** (bottom tail).

**Finding:** Corpus weights create near-disconnection bottlenecks. The
uniform spectrum has 6 distinct eigenvalues (high degeneracy from Oh
symmetry); corpus weights break all degeneracies. The Fiedler suppression
identifies the mechanism behind Experiment 1's amplification: FCC routes
around bottlenecks that strangle cubic lattices.

---

## Experiment 5: Direction-Weighted Tessellation (Lattice Scale)

Direction-based weighting maps 24 corpus values to FCC's 6 direction
pairs (4 per pair) vs cubic's 3 direction pairs (8 per pair). Each FCC
direction pair corresponds to two faces of the rhombic dodecahedron.

| Scale | Distribution | Fiedler Ratio | Diffusion Speedup | Consensus Speedup |
|-------|-------------|---------------|-------------------|-------------------|
| 125 | uniform | 2.31x | 2.00x | 1.00x |
| 125 | random | 3.30x | 2.00x | 2.18x |
| 125 | power_law | 3.08x | 2.00x | 4.12x |
| 125 | **corpus** | **5.55x** | 2.00x | **6.69x** |
| 1000 | uniform | 2.55x | 6.00x | 0.93x |
| 1000 | random | 3.65x | 1.40x | 0.78x |
| 1000 | power_law | 3.37x | 1.60x | 0.58x |
| 1000 | **corpus** | **6.11x** | 1.60x | 0.73x |
| 8000 | uniform | 2.28x | — | — |
| 8000 | **corpus** | **5.47x** | — | — |

**Finding:** Direction-based weighting nearly triples the Fiedler advantage
(2.3x -> 6.1x). Consensus speedup peaks at 6.69x (scale 125, corpus) but
inverts at scale 1000 — the advantage is topological at local scale and
washed out by graph diameter at large scale. Diffusion is coarser-grained
(discrete half-step measure) and shows stable 2x regardless of weighting.

---

## Experiment 6: Targeted Prime-Vertex Mapping (Single Cell)

TV-optimal edge assignment, then exhaustive search over all 8! = 40,320
prime-to-vertex mappings. Three-tier scoring: direct divisibility (weight 3),
sum/difference divisibility (weight 2), shared prime factors (weight 1).

| Mapping | Score | p-value | Rank |
|---------|-------|---------|------|
| **Optimal** (geometry-determined) | **35.0** | **0.000025** | 1 / 40,320 |
| Identity (prime[i] -> vertex[i]) | 20.0 | 0.46 | 15,135 / 40,320 |
| Null mean | 19.5 +/- 3.6 | — | — |

Optimal mapping detail:

| Prime | Vertex | Edge Values | Divisible |
|-------|--------|-------------|-----------|
| 67 | 7 | [72, 134, 29] | 134 = 2x67 |
| 29 | 0 | [448, 435, 342] | 435 = 15x29 |
| 19 | 4 | [133, 346, 309] | 133 = 7x19 |
| 17 | 3 | [1296, 136, 64] | 136 = 8x17 |
| 11 | 5 | [78, 386, 55] | 55 = 5x11 |
| 23 | 6 | [94, 153, 240] | — |
| 31 | 2 | [771, 202, 252] | — |
| 89 | 1 | [463, 405, 18] | — |

**Finding:** The optimal mapping is extremely significant (p = 0.000025).
Five of eight primes have direct divisibility hits. The identity mapping
(the determined prime-to-Law ordering) scores at the null mean — the
ordering matters but does not transfer to vertex labeling. The geometry
asks for its own arrangement.

---

## Experiment 7: Spectral Comparison Across Polytopes

Five 24-edge graphs compared under four weight distributions.

### Part A: Spectrum Summary

| Graph | V | E | Degree | Uniform Fiedler | Corpus Fiedler |
|-------|---|---|--------|----------------|---------------|
| RD | 14 | 24 | 3,4 | 1.4384 | 0.0863 |
| Cuboctahedron | 12 | 24 | 4 | 2.0000 | 0.1347 |
| K(4,6) | 10 | 24 | 4,6 | 4.0000 | 0.3886 |
| 3-regular | 16 | 24 | 3 | 0.4565 | 0.0538 |
| Random G(14,24) | 14 | 24 | varies | 0.6997 | 0.0529 |

### Part B: Corpus Fiedler Percentile (vs 10K Random Weights)

| Graph | Percentile |
|-------|-----------|
| RD | 0.08% |
| Cuboctahedron | 0.02% |
| K(4,6) | 0.36% |
| 3-regular | 1.04% |
| Random G(14,24) | 5.94% |

### Part C: Degeneracy Breaking

| Graph | Uniform Distinct / Total | Corpus Distinct / Total |
|-------|------------------------|------------------------|
| RD | 6 / 14 | 14 / 14 |
| Cuboctahedron | 4 / 12 | 12 / 12 |
| K(4,6) | 4 / 10 | 10 / 10 |
| 3-regular | 15 / 16 | 16 / 16 |
| Random | 14 / 14 | 14 / 14 |

### Part D: Spectral Distances (Corpus Weights)

| | RD | Cuboct | K(4,6) | 3-reg | Random |
|---|---|---|---|---|---|
| RD | 0.00 | 3.90 | 4.56 | 3.58 | 0.31 |
| Cuboct | 3.90 | 0.00 | 3.92 | 4.04 | 3.94 |
| K(4,6) | 4.56 | 3.92 | 0.00 | 4.71 | 4.60 |
| 3-reg | 3.58 | 4.04 | 4.71 | 0.00 | 3.59 |
| Random | 0.31 | 3.94 | 4.60 | 3.59 | 0.00 |

**Finding:** Corpus Fiedler suppression is universal across all tested
24-edge polytopes — NOT unique to the RD. The cuboctahedron shows even
stronger suppression (0.02%). High-symmetry graphs (RD, cuboctahedron,
K(4,6)) show maximal degeneracy breaking. RD and random G(14,24) cluster
spectrally (distance 0.31); the cuboctahedron is an isolate.

---

## Raw Result Files

- `experiment_1_weighted_benchmarks.txt`
- `experiment_2_optimal_assignment.txt`
- `experiment_3_prime_coherence.txt`
- `experiment_4_spectral.txt`
- `experiment_5_direction_weighted.txt`
- `experiment_6_prime_vertex.txt`
- `experiment_7_spectral_polytopes.txt`
