# Contributing to rhombic

## Security First

Before contributing, install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Every commit is scanned for accidentally included secrets.

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Install pre-commit hooks (`pre-commit install`)
4. Install dev dependencies: `pip install -e ".[all]"`
5. Make your changes
6. Run tests: `pytest`
7. Commit with a descriptive message
8. Push to your fork and open a Pull Request

## What We're Looking For

- **New lattice topologies.** BCC, hexagonal close-packed, diamond cubic — any tessellation with a clear mathematical characterization.
- **New benchmark metrics.** Betweenness centrality, spectral gap, random walk cover time — anything that measures structural properties.
- **New rungs on the experimental ladder.** Spatial operations, signal processing, context architecture — see `docs/EXPERIMENTAL_LADDER.md`.
- **Performance improvements.** Faster lattice construction, GPU-accelerated metrics.
- **Bug fixes and documentation improvements.**

## What We're Not Looking For

- Benchmarks that don't report cost alongside benefit. The edge count ratio matters.
- Topologies without clear mathematical definition.
- Changes that break reproducibility of existing results.

## Philosophy

- Reproducible by default. Every result has code that generates it.
- The geometry is the argument. The numbers are the evidence.
- Cost is always reported alongside benefit.
- Sparse results are data, not failure.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
