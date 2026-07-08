# rhombic — Project Identity

You are assisting with **rhombic**, a lattice topology benchmarking library
that compares cubic (6-connected) and FCC/rhombic dodecahedral (12-connected)
lattice topologies. Built by TASUMER MAF.

## The Thesis

The cubic lattice is the unexamined default of computation. This library
measures the alternative and publishes reproducible results showing that
the FCC lattice provides 30% shorter paths, 40% smaller diameter, and
2.4x algebraic connectivity — at the cost of ~2x more edges.

See `docs/THESIS.md` for the full argument.
See `docs/EXPERIMENTAL_LADDER.md` for the 5-rung experimental plan.

## Experimental Ladder

| Rung | Topic | Status |
|------|-------|--------|
| 0 | Lattice Library | Complete |
| 1 | Graph Theory Benchmarks | Complete |
| 2 | Spatial Operations | Complete |
| 3 | Signal Processing | Complete |
| 4 | Context Architecture | Complete |
| P2 | Weighted Extensions (Paper 2) | Draft complete — 7 experiments |
| P3 | The Learnable Bridge (Paper 3) | Draft — 13 experiments; fingerprinting figure corrected Apr 2026 (72.3%, not 84.5%) |
| P4 | The Topology Programmer (Paper 4) | Draft — 4 polytopes, 4 regimes; audit blockers emptied Jul 2026 (T-001 rewritten to three-run evidence; exposure classification in docs/PAPER4_EXPOSURE_CLASSIFICATION.md); adaptive-run re-measures pending post-bank |
| EE | Equal-edge random-graph control (EE-001) | Complete — narrows Paper 1 claim to spatially-embeddable class |
| BM | Benchmark battery (BM-000/000b nulls; BM-003 arms A–H; BM-004 v2) | BM-000/000b complete (hub attack closed at null level); BM-003 G/H + dissociation endpoint and BM-004 v2 pre-registered, Director-approved 2026-07-07; GPU work post-bank, BM-004 gated on the F2 interlock (`scripts/bm004_runner.py`) |
| A1 | Asset-1 adapter bank + D1/D2/D3/D-aux analyses | Campaign LIVE (480 runs, 2 families × 6 tasks × 40 seeds, ETA ~Jul 20–21); all analyses pre-registered and completeness-interlocked; D1 has three representation arms |

Suite: **689 tests** (`pytest -v`). Paper status: Paper 1 complete
(unpublished); Papers 2–4 working drafts. Nothing is submitted or published.
Analyses of the Asset-1 bank are pre-registered; the Director's rulings
(docs/DIRECTOR_DECISIONS_2026-07-06.md, docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md)
bind, and deviations are dated amendments, never silent revisions (L-006).

## How to Add a New Rung

1. Create `rhombic/your_module.py` with benchmarks
2. Add tests in `tests/test_your_module.py`
3. Run benchmarks and save results to `results/rung-N/`
4. Write `RESULTS.md` and `INTERPRETATION.md` in the results directory
5. Update `README.md` with key findings
6. Ensure `python -m rhombic.benchmark` still reproduces all results

## How to Add a New Topology

The library compares cubic and FCC. To add another lattice topology
(BCC, HCP, diamond cubic, etc.):

1. Add a class in `rhombic/lattice.py` following the CubicLattice/FCCLattice pattern
2. Must implement: `__init__(n)`, `to_networkx()`, `stats()`
3. Add to `matched_lattices()` in `benchmark.py`
4. Add unit tests verifying node count, interior degree, and connectivity

## Principles

- **Reproducible by default.** Every result has code that generates it.
- **The geometry is the argument.** The numbers are the evidence.
- **Cost is always reported alongside benefit.** The edge count ratio matters.
- **Sparse results are data, not failure.** If a rung shows no advantage, publish that.
- **Examine every default.** The project's thesis should inform how it's built.

## Color Palette

The visualization palette derives from the 8-Law Weave:

```python
CUBIC_COLOR = '#3D3D6B'    # Fall of Neutral Events (prime 11)
FCC_COLOR = '#B34444'      # Geometric Essence (prime 67)
```

Do not change these without understanding their derivation. See
`rhombic/visualize.py` for the full palette.

## Technical

- Python 3.10+
- Dependencies: numpy, networkx (core); matplotlib (visualization); pytest (dev)
- License: MPL-2.0
- Tests: `pytest -v`
- Benchmark: `python -m rhombic.benchmark`
- Banner: `python scripts/generate_banner.py` (generates from library code)
