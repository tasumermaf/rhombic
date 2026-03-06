# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-05

### Added

- **Rhombic dodecahedron module** (`polyhedron.py`) — First-class graph object
  for the 14V/24E/12F rhombic dodecahedron (Voronoi cell of FCC lattice).
  Pre-computed combinatorial data, vertex stars, face enumeration, NetworkX export.
- **Isopsephy corpus module** (`corpus.py`) — 24 structured integer edge weights
  from the Falco trump isopsephy corpus, plus 14 Names of Power, 8 tracked primes,
  four weight distributions (uniform, random, power-law, corpus).
- **Optimal assignment module** (`assignment.py`) — Total variation minimization
  via simulated annealing with random restarts. Null distribution generation via
  random permutations. Graph topology comparison.
- **Spectral analysis module** (`spectral.py`) — Weighted Laplacian construction,
  eigenvalue spectrum, Fiedler value, spectral gap computation.
- **Weighted benchmark support** — `_avg_shortest_path`, `_diameter`, `_fiedler_value`
  in `benchmark.py` now accept optional `weight` parameter (backwards-compatible).
- **Experiment runner** (`scripts/run_experiments.py`) — Reproducible 4-experiment
  suite for Paper 2 (seed=42). Outputs to `results/paper2/`.
- 88 new tests (167 total, 0 regressions): `test_polyhedron.py` (24),
  `test_corpus.py` (36), `test_assignment.py` (12), `test_spectral.py` (16).

### Key Results (Paper 2 — Weighted Extensions)

| Experiment | Finding |
|------------|---------|
| Weighted FCC vs Cubic | FCC advantage amplifies under heterogeneous weights (2.3x uniform -> 3.2x corpus) |
| Optimal Assignment | RD sorts weights 37.8% better than random (p<0.0001); bipartite constraint is structural |
| Prime Coherence | Not significant at single-cell scale (p=0.30); test underpowered |
| Spectral Properties | Corpus weights create bottlenecks (Fiedler bottom 0.1%); explains amplification mechanism |

### Fixed (Rounds 4-6 paper remediation)

- **Table 1 direction consistency** — All advantage ratios now read
  "higher is better" with SC baseline values shown explicitly.
- **Table 2 visual separation** — FCC advantages and costs grouped with
  midrule and italic subheadings for scannability.
- **Cybernetics section flagged as interpretive** — Opening sentence
  explicitly marks the subsection as interpretive, not empirical.
- **Isotropy metric highlighted** — Added sentence noting directional
  error variance as a potentially standalone contribution.
- **"nature selects" removed** — Conclusion and all repo files now use
  "close-packed natural structures commonly exhibit" instead of
  teleological phrasing.
- **Conclusion recall figure** — Changed from "7-20pp embedding recall"
  to "15-26pp neighborhood recall" matching the abstract.
- **Rung 4/Index projection comparability** — Added sentence confirming
  FCC advantage is consistent across random orthogonal (Rung 4) and
  PCA (Index) projection methods.
- **README/SYNTHESIS tone calibration** — Synthesis title changed from
  "Why Computation Inherited the Wrong Geometry" to "Cubic vs FCC Lattice
  Topology Across Four Domains." Closing sentence scoped to match paper.
- **Kepler phrasing consistency** — All files now say "proved by Hales
  2005, formally verified 2017" (removed "announced 1998").
- **THESIS.md scoped** — "structurally inferior" qualified with "for the
  isotropic, propagation-dominated workloads tested here."
- **Intro forward reference** — Added cross-reference to "When the Cubic
  Lattice Wins" subsection from the introduction.
- **Reproducibility paragraph rewritten** — Commit hash replaced with
  PyPI version pin and provenance attestation note.

### Added

- Paper figures (Voronoi cells, graph benchmarks) and generation script
- Figure references in paper body (Fig. 1, Fig. 2)
- Fiedler DOI in bibliography

## [0.1.2] - 2026-03-05

### Fixed

- **ORVS citation corrected** — TR number changed from TR-186-2-02-11 (a
  different report) to TR-186-2-01-10; year corrected from 2002 to 2001.
  Also fixed in `signal.py` docstring.
- **Abstract rewritten** — Removed "memory is linear, pixels are square,
  voxels are cubic" (rhetorical-as-factual). Replaced with research framing.
- **Causal overclaims softened** — "sole variable," "entirely attributable,"
  and "only variable" replaced with language acknowledging node-count
  mismatch. "Unexamined default" → "widely adopted default." "Failure mode"
  → "axis-alignment artifact."
- **FCC/HCP consistency** — Introduction and conclusion now both include
  HCP alongside FCC for densest sphere packing claim.
- **Intro genealogy tightened** — "descends from Descartes... through von
  Neumann" → "aligns with Cartesian analytic geometry... and stored-program
  architecture designs."
- **Conclusion rhetorical softening** — "not because it was chosen but
  because it was inherited" → "largely through convention rather than
  demonstrated optimality."
- **Projection method justification** — Added one-sentence explanations
  for why Rung 4 uses random projection and ANN index uses PCA.
- **Full git SHA** — Reproducibility paragraph now uses full 40-character
  commit hash instead of abbreviated 7-character prefix.

## [0.1.1] - 2026-03-05

### Fixed

- **BCC/FCC sampling theory framing** — Correctly attribute 3D sampling
  optimality to BCC (not FCC) per the volume-rendering literature. Rung 3
  results reframed as direct empirical measurements, not derivations from
  Petersen-Middleton theory. Clarified dual-lattice relationship (FCC ↔ BCC
  are reciprocal duals). Fixed `signal.py` docstring: rhombic dodecahedron
  is the real-space Voronoi cell, not the Nyquist region.
- **Kepler conjecture timeline** — Announced 1998, published 2005, formally
  verified 2014–2017. Both FCC and HCP achieve maximal packing density.
- **Bibliography metadata** — Corrected years and DOIs for Theußl (2000),
  Entezari (2004), and Csebfalvi (2010).
- **Universal claims scoped** — "Scale-invariant" → "stable across tested
  scales." "Not optimal" scoped to isotropic, propagation-dominated workloads.
  Applied consistently across all 16 affected files.

### Added

- Reproducibility paragraph in paper (§3): seed, trial count, commit hash
- Node-matching notation: explicit formulas for n_SC and n_FCC
- "When the Cubic Lattice Wins" subsection in paper
- Expanded "Limitations and Threats to Validity" section

### Changed

- Paper §4 title: "Context Architecture" → "Embedding Neighborhood Preservation"

## [0.1.0] - 2026-03-04

### Added

- **Rung 0: Lattice Library** — `CubicLattice` and `FCCLattice` classes with
  matched node counts, interior degree verification, and NetworkX export
- **Rung 1: Graph Theory** — average shortest path, diameter, algebraic
  connectivity, and clustering coefficient benchmarks across three scales
- **Rung 2: Spatial Operations** — flood fill, nearest-neighbor queries,
  range queries, and locality-sensitive hashing benchmarks
- **Rung 3: Signal Processing** — 3D bandlimited signal reconstruction MSE
  and directional isotropy analysis (direct empirical measurement)
- **Rung 4: Context Architecture** — embedding neighbor recall, information
  diffusion rate, and consensus convergence benchmarks
- **Rung 5: Synthesis** — "The Shape of the Cell" complete synthesis document
  with cultural genealogy, cybernetic interpretation, and recommendations
- Visualization module with Vadrashinetal color palette
- Banner generation from library code (`scripts/generate_banner.py`)
- CI workflow (Python 3.10, 3.11, 3.12) with benchmark reproduction
- Security scanning (gitleaks)
- Full community health files (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY)

### Key Results

| Metric | FCC vs Cubic |
|--------|-------------|
| Average shortest path | 30% shorter |
| Graph diameter | 40% smaller |
| Algebraic connectivity | 2.4x higher |
| Flood fill reach | 55% more nodes |
| Signal reconstruction MSE | 5-10x lower |
| Embedding neighbor recall | +15-26pp at 1-hop |
| Edge cost | ~2x more edges |

[0.2.0]: https://github.com/promptcrafted/rhombic/releases/tag/v0.2.0
[0.1.2]: https://github.com/promptcrafted/rhombic/releases/tag/v0.1.2
[0.1.1]: https://github.com/promptcrafted/rhombic/releases/tag/v0.1.1
[0.1.0]: https://github.com/promptcrafted/rhombic/releases/tag/v0.1.0
