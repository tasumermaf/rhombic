# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/promptcrafted/rhombic/releases/tag/v0.1.1
[0.1.0]: https://github.com/promptcrafted/rhombic/releases/tag/v0.1.0
