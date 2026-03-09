![rhombic banner](assets/banner.png)

[![PyPI](https://img.shields.io/pypi/v/rhombic)](https://pypi.org/project/rhombic/)
[![CI](https://github.com/promptcrafted/rhombic/actions/workflows/ci.yml/badge.svg)](https://github.com/promptcrafted/rhombic/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/rhombic)](https://pypi.org/project/rhombic/)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

# rhombic

> *The bottleneck is not the processor. It is the shape of the cell.*

A benchmarking library that compares cubic (6-connected) and FCC/rhombic
dodecahedral (12-connected) lattice topologies across graph theory,
spatial operations, and signal processing.

## The Numbers

| Metric | FCC vs Cubic | Scale |
|--------|-------------|-------|
| Average shortest path | **30% shorter** | 125 – 8,000 nodes |
| Graph diameter | **40% smaller** | 125 – 8,000 nodes |
| Algebraic connectivity | **2.4× higher** | 125 – 8,000 nodes |
| Flood fill reach | **55% more nodes** | 125 – 8,000 nodes |
| NN query speed | **17% faster** | 125 – 8,000 nodes |
| Signal reconstruction | **5-10× lower MSE** | 216 – 1,000 samples |
| Reconstruction isotropy | **5-20× more uniform** | 216 – 1,000 samples |
| Embedding neighbor recall | **+15-26pp at 1-hop** | 125 – 1,000 nodes |
| Information diffusion | **1.4-2× faster** | 125 – 1,000 nodes |
| Edge cost | ~2× more edges | (the price) |

These ratios are stable across all tested scales. They hold at every size
tested, consistent with derivation from Voronoi cell geometry rather than
sample size.

![Graph theory dashboard](results/rung-1/dashboard.png)
![Spatial operations dashboard](results/rung-2/dashboard.png)
![Signal processing dashboard](results/rung-3/dashboard.png)
![Context architecture dashboard](results/rung-4/dashboard.png)

### Under Structured Weights (Paper 2)

| Metric | Corpus vs Uniform | Scale |
|--------|-------------------|-------|
| Fiedler ratio (direction-weighted) | 2.3x → **6.1x** | 125 – 8,000 nodes |
| Path advantage | 30% → **60% shorter** | 125 – 1,000 nodes |
| Consensus speedup | 1.0x → **6.7x** | 125 nodes |
| Prime-vertex coherence | **p = 0.000025** | Single cell (40,320 permutations) |

Heterogeneous edge weights amplify the FCC advantage. Direction-based
weighting — mapping structured values to the 6 direction pairs of the
FCC lattice — nearly triples the Fiedler ratio. The mechanism is
bottleneck resilience: FCC routes around suppressed edges that strangle
cubic lattices.

- [Raw data and tables](results/paper2/RESULTS.md)
- [What the numbers mean](results/paper2/INTERPRETATION.md)

## The Question

Computation is built on the cube. Memory is linear. Pixels are square.
Voxels are cubic. Nobody chose this — it accumulated. Descartes gave us
orthogonal coordinates. Von Neumann gave us linear memory. The cubic
lattice is the spatial expression of Cartesian geometry.

Is the cube optimal? This library measures the alternative: the
face-centered cubic lattice, whose Voronoi cells are rhombic
dodecahedra. 12 faces instead of 6. The densest sphere packing in
three dimensions (Kepler, proved by Hales 2005, formally
verified 2017). The lattice that nature
uses for copper, aluminum, and gold.

[Read the full thesis →](docs/THESIS.md)

## Quick Start

```bash
pip install rhombic            # minimal (numpy + networkx)
pip install "rhombic[viz]"     # add matplotlib for plots
pip install "rhombic[all]"     # everything including dev tools
```

Reproduce all results:

```bash
python -m rhombic.benchmark
```

Use in code:

```python
from rhombic.lattice import CubicLattice, FCCLattice

cubic = CubicLattice(n=10)     # 1000 nodes, 6-connected
fcc = FCCLattice(n=6)          # ~864 nodes, 12-connected

# Convert to networkx for any graph analysis
G_cubic = cubic.to_networkx()
G_fcc = fcc.to_networkx()
```

Compare topologies:

```python
from rhombic.lattice import CubicLattice, FCCLattice

cubic, fcc = CubicLattice(5), FCCLattice(5)
print(f"Cubic: {cubic.stats().connectivity}-connected, {cubic.stats().node_count} nodes")
print(f"FCC:   {fcc.stats().connectivity}-connected, {fcc.stats().node_count} nodes")
# Cubic: 6-connected, 125 nodes
# FCC:   12-connected, 500 nodes
```

## Results

### Rung 1: Graph Theory (complete)

Four metrics, three scales, consistent ratios. The FCC lattice outperforms
the cubic lattice on every measure of routing efficiency and structural
robustness. The cost is bounded: ~2× edges for ~30% shorter paths and
~2.4× robustness.

- [Raw data and tables](results/rung-1/RESULTS.md)
- [What the numbers mean](results/rung-1/INTERPRETATION.md)

### Rung 2: Spatial Operations (complete)

The routing advantage translates. FCC flood fill reaches 55% more nodes
per hop. Nearest-neighbor queries are 17% faster. Range queries return
24% more nodes per volume (denser packing). The cost: range query time
scales with density — 3-5× slower for sphere/box queries at 8,000 nodes.

- [Raw data and tables](results/rung-2/RESULTS.md)
- [What the numbers mean](results/rung-2/INTERPRETATION.md)

### Rung 3: Signal Processing (complete)

Direct empirical measurements confirm the FCC advantage. FCC spatial sampling
produces **5-10× lower MSE** and **5-20× more isotropic** reconstruction than
cubic sampling at matched sample counts. The advantage peaks in the mid-frequency range (10-60% of
Nyquist) and grows with scale — from +6 dB at 216 samples to +10 dB at 1,000.
Above Nyquist, both lattices alias and cubic's axis alignment accidentally helps.

- [Raw data and tables](results/rung-3/RESULTS.md)
- [What the numbers mean](results/rung-3/INTERPRETATION.md)

### Rung 4: Context Architecture (complete)

Does the FCC advantage survive when the lattice organizes high-dimensional
embedding data? FCC captures **15-26 more percentage points** of an embedding's
true nearest neighbors at 1-hop. Information diffuses **1.4-2× faster**.
Consensus converges 1.58× faster at moderate scale (500 nodes), though
per-neighbor weight dilution reduces the advantage at 1,000 nodes.

- [Raw data and tables](results/rung-4/RESULTS.md)
- [What the numbers mean](results/rung-4/INTERPRETATION.md)

[Full experimental ladder →](docs/EXPERIMENTAL_LADDER.md)

### FCC Embedding Index (complete)

A proof-of-concept ANN index that organizes high-dimensional embeddings on
lattice topology. At matched node counts, the FCC index captures **+7 to +20
percentage points** more true nearest neighbors at 1-hop than the cubic index.
The only variable is the connectivity pattern.

```python
from rhombic.index import FCCIndex, CubicIndex, brute_force_knn

fcc = FCCIndex.from_target_nodes(dim=384, target_nodes=500).build(embeddings)
results = fcc.query(query_vector, k=10, hops=1)
recall = fcc.recall_at_k(queries, ground_truth, k=10, hops=1)
```

- [Raw data and tables](results/index/RESULTS.md)
- [What the numbers mean](results/index/INTERPRETATION.md)

### Paper 2: Weighted Extensions (complete)

What happens when edges carry heterogeneous weights? Seven experiments across
two scales (lattice and single-cell). The FCC advantage **amplifies** under
structured weights — direction-based corpus weighting pushes the Fiedler ratio
from 2.3x to 6.1x. Prime-vertex coherence is significant at the optimal
mapping (p = 0.000025 vs 40,320 alternatives). Spectral bottleneck creation
is universal across 24-edge polytopes, not RD-specific.

- [Raw data and tables](results/paper2/RESULTS.md)
- [What the numbers mean](results/paper2/INTERPRETATION.md)

### RhombiLoRA Experiments (complete)

Five experiments mapping the lattice topology advantage onto neural
architecture. FCC-topology bridges learn 4.6× more cross-channel coupling
than cubic topology. Bridge matrices encode task identity (84.5% LOO
classification from 28 Q-projection bridges). Bridge interpolation
provides smooth adapter composition with eigenspectrum preservation.

- Exp 1 (1.5B): [Analysis](results/exp1/ANALYSIS.md)
- Exp 2 (7B Alpaca): [Bridge anatomy](results/exp2/bridge_anatomy.md)
- Exp 2.5 (7B Geometric): [Comparison report](results/exp2_5/COMPARISON_REPORT.md) — null on direction, positive on connectivity
- Phase 1A: [Task fingerprints](results/fingerprints/TASK_FINGERPRINT_REPORT.md) — 73.5% 3-way, 84.5% Q-proj
- Phase 2A: [Bridge merging](results/bridge_merge/MERGE_REPORT.md) — smooth interpolation, non-linear in 2/3 pairs
- Cross-phase: [Synthesis](results/CROSS_PHASE_SYNTHESIS.md) — 20 learnings, Q-proj optimal diagnostic

### Synthesis

The complete argument across all four rungs — cultural genealogy, empirical
evidence, cybernetic interpretation, and practical recommendations.

- [The Shape of the Cell: Cubic vs FCC Lattice Topology Across Four Domains](results/SYNTHESIS.md)

## Philosophy

- Reproducible by default. Every result has code that generates it.
- The geometry is the argument. The numbers are the evidence.
- Cost is always reported alongside benefit.
- Sparse results are data, not failure.

## Interactive Agent

`rhombic-agent` — a [Hermes Agent](https://github.com/NousResearch/hermes-agent)
that thinks in 12 dimensions. 9 custom tools + 3 conversational skills. Ask it
to run experiments, generate visualizations, and explain the geometry.

[Hackathon Demo →](https://promptcrafted.github.io/rhombic/)

## Ecosystem

- [Interactive demo](https://huggingface.co/spaces/timotheospaul/rhombic) — try it in your browser
- [The essay](https://timotheospaul.substack.com/p/the-shape-of-the-cell) — the thesis for humans
- [PyPI](https://pypi.org/project/rhombic/) — `pip install rhombic`
- [Full synthesis](results/SYNTHESIS.md) — the complete argument across all four rungs
- [Weighted extensions](results/paper2/RESULTS.md) — what happens under heterogeneous weights

### Paper 3: RhombiLoRA — Neural Adapter Geometry (in progress)

Five experiments complete, paper writeup in progress.

What happens when you apply the lattice topology to LoRA adapters?
**RhombiLoRA** adds a learnable 6×6 coupling matrix — the *bridge* —
between the A and B projections in LoRA, adding 36 parameters per layer.

| Finding | Metric | Value |
|---------|--------|-------|
| Task fingerprinting | LOO SVM accuracy (Q-proj, 28 bridges) | **84.5%** (chance 33.3%) |
| Between > within task | Mann-Whitney U | p < 1e-6 |
| Bridge coupling | FCC (6-ch) vs cubic (3-ch) Fiedler | **4.6×** |
| Adapter composition | Eigenspectrum preservation at all α | cos > 0.999 |
| Merge safety threshold | Cross-task contamination tolerance | 10% |
| Optimal fingerprint size | Q-proj bridges only | **1,008 parameters** |

The bridge does not beat standard LoRA at standard LoRA's own job. It
provides something LoRA cannot: a compact, interpretable diagnostic of
adapter behavior — a 36-parameter summary of what training discovered,
readable without inference or evaluation.

- 20 experimental learnings documented in [LEARNINGS.md](docs/LEARNINGS.md)
- Architecture: [`rhombic.nn`](rhombic/nn/) — `RhombiLoRALinear`, topology, bridge init
- Training harness: [`scripts/train_comparison.py`](scripts/train_comparison.py)

### Papers

- [Paper 1: The Shape of the Cell](paper/rhombic.tex) — four-domain topology comparison (arXiv cs.DS)
- [Paper 2: Pure Number Architecture](paper/rhombic-paper2.tex) — weighted amplification and spectral structure
- Paper 3: The Learnable Bridge (in preparation) — task fingerprinting and adapter composition via structured coupling

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We're looking for new topologies,
new metrics, and new rungs on the experimental ladder.

## License

[MPL-2.0](LICENSE) — Use freely. Modifications to library files shared
back to the commons.

## Built by [Promptcrafted](https://promptcrafted.com)
