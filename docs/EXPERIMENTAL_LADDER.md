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
2.4x algebraic connectivity. Stable across tested scales.

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

## Rung 3: Signal Processing (complete)

In 3D lattice sampling theory, the commonly cited optimality result
identifies BCC (not FCC) as the most efficient sampling geometry for
isotropic bandlimits. Our benchmarks directly measure FCC spatial sampling
with a topology-agnostic reconstructor — empirical measurements, not
derivations from sampling theory.

Benchmarks (density-matched, topology-agnostic RBF reconstruction):

- Frequency sweep from 10% to 100% of cubic Nyquist
- Reconstruction quality (MSE, PSNR) at three scales (216, 512, 1000 samples)
- Isotropy: directional bias in reconstruction error (6 directions)

Results: FCC produces **5-10× lower MSE** in the mid-frequency range (10-60%
Nyquist) and **5-20× more isotropic** reconstruction at all scales tested.
Peak advantage: **+10 dB PSNR** at 1000 samples. The advantage grows with
scale. Above Nyquist, both lattices alias and cubic's rectilinear alignment
accidentally helps with separable signals.

**Deliverable:** `rhombic/signal.py`, `results/rung-3/`
**Status:** Complete.

---

## Rung 4: Context Architecture (complete)

The ambitious rung. Apply the topology findings to AI context management.
Clustered embeddings (128D, 5 clusters) projected to 3D via random orthogonal
projection, assigned to lattice nodes by nearest position. Three benchmarks:

- **Neighborhood recall:** Fraction of an embedding's true 10 nearest neighbors
  (in 128D space) captured by its lattice node's k-hop neighborhood
- **Information diffusion:** Signal pulse propagation from central node via
  max(neighbor × 0.9). Rounds to reach 50% and 80% of lattice
- **Consensus speed:** Laplacian averaging to convergence (ε=0.05)

Results at three scales (125, 500, 1000 nodes):

| Metric | FCC Advantage | Note |
|--------|--------------|------|
| 1-hop recall | **+15-26pp** | 66% vs 51% at 1000 nodes |
| 2-hop recall | **+16pp** | 92% vs 76% at 1000 nodes |
| Diffusion (50% reach) | **1.4-2× faster** | Stable across tested scales |
| Diffusion (80% reach) | **1.6-1.7× faster** | Stable across tested scales |
| Consensus (500 nodes) | **1.58× faster** | Sweet spot |
| Consensus (1000 nodes) | **0.93×** (cubic wins) | Per-neighbor dilution |

The neighborhood recall result is directly relevant to RAG: an FCC-organized
embedding index captures 15-26% more relevant context per lookup. The
consensus result is the most honest — FCC's advantage is real but scale-
dependent for distributed agreement.

**Deliverable:** `rhombic/context.py`, `results/rung-4/`
**Status:** Complete.

---

## Rung 5: Integration and Publication (complete)

Synthesize findings across all rungs into a single publishable document:

- The cultural genealogy (why the cube persists)
- The empirical evidence (rungs 1-4) with cross-domain pattern analysis
- The cybernetic interpretation (Ashby, Beer, Bateson)
- Practical recommendations for adoption (where to use FCC now, where to
  proceed with caution, where to stay cubic)
- Open questions and future work

**Deliverable:** `results/SYNTHESIS.md`
**Status:** Complete.

---

## Paper 2: Weighted Extensions (complete)

Seven experiments testing whether structured edge weights amplify or
attenuate the FCC topology advantage. Two scales: lattice (tessellated
cells) and single-cell (rhombic dodecahedron in isolation).

| Experiment | Question | Finding |
|------------|----------|---------|
| 1. Weighted benchmarks | Does heterogeneity help? | Fiedler 2.3x -> 3.2x under corpus cycling |
| 2. Optimal assignment | Does the RD sort weights? | 37.8% better than random (bipartite constraint) |
| 3. Prime coherence v1 | Primes at vertex stars? | Not significant (p=0.30, underpowered) |
| 4. Spectral properties | Corpus as bottleneck? | Fiedler at 0.08% percentile, full degeneracy breaking |
| 5. Direction weighting | Direction-aware amplification? | **Fiedler 2.3x -> 6.1x**, consensus 6.7x |
| 6. Prime-vertex mapping | Exhaustive prime mapping? | **p=0.000025** (optimal), identity at null |
| 7. Spectral polytopes | RD-specific or universal? | Universal suppression, RD not special |

**Deliverable:** `rhombic/polyhedron.py`, `rhombic/corpus.py`,
`rhombic/assignment.py`, `rhombic/spectral.py` (extended), `rhombic/context.py`
(extended), `scripts/run_experiments.py`, `results/paper2/`

**Status:** Complete. 256 tests, 0 regressions. All 7 experiments
reproducible with `python scripts/run_experiments.py`.

---

## Design Principles (all rungs)

1. **Cost is always reported alongside benefit.** The FCC lattice uses ~2x
   edges. That cost must be visible in every comparison.

2. **Reproducible by default.** `python -m rhombic.benchmark` reproduces
   every result. No hidden state, no unreported parameters.

3. **Scale stability tested.** Every metric is measured at multiple scales
   to confirm the advantage is stable across tested ranges.

4. **Sparse results are data.** If a rung shows no advantage for FCC, that
   finding is published alongside the wins.
