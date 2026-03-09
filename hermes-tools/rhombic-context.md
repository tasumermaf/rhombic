# Rhombic — Lattice Topology Research Context

You have access to the `rhombic` lattice topology library (v0.3.0). This context
file tells you what the research is about, what the numbers mean, and how to
explain it to users.

## The Thesis

Every computation — neural networks, simulations, databases — defaults to cubic
lattice topology (6-connected grids). The rhombic library measures the alternative:
FCC (face-centered cubic) lattice topology, where each node connects to 12 neighbors
instead of 6. The rhombic dodecahedron is the Voronoi cell of the FCC lattice.

**Paper 1** established the baseline comparison:
- FCC gives 2.3x algebraic connectivity (Fiedler value)
- 30% shorter average paths
- 40% smaller graph diameter
- At the cost of ~2x more edges

**Paper 2** asked: does this advantage amplify or attenuate under real-world
(heterogeneous) weight distributions?

## Paper 2 Key Findings

| Experiment | Finding |
|------------|---------|
| Edge-cycled corpus weights (Exp 1-3) | Fiedler ratio amplifies from 2.3x to 3.1x |
| RD degeneracy breaking (Exp 4) | Corpus weights break 6 → 14 distinct eigenvalues |
| Direction-pair weighting (Exp 5) | Fiedler ratio reaches **6.1x** with corpus weights |
| Prime-vertex mapping (Exp 6) | p = 2.5e-5 — non-accidental corpus-geometry affinity |
| Cross-topology consistency (Exp 7) | Suppression across all 5 tested 24-edge graphs |

**The mechanism:** Bottleneck resilience. Heterogeneous weights create near-disconnection
cuts. The FCC lattice has redundant paths around these cuts (12 neighbors vs 6).
The more heterogeneous the weights, the more the FCC advantage grows.

**Consensus inversion:** FCC consensus is 6.7x faster at small scale (125 nodes)
but 0.73x at scale 1000. Mechanism: per-neighbor dilution — each of 12 FCC neighbors
contributes less per-neighbor influence than each of 6 cubic neighbors.

## The Corpus

The corpus is a set of 24 numerical values derived from a fixed transliteration
protocol applied to a symbolic inscription set. Eight primes (11, 17, 19, 23, 29,
31, 67, 89) recur as factors. The corpus serves as a high-variance, heavy-tailed
stress test alongside uniform, random, and power-law distributions.

## Forward Vision: RhombiLoRA

**Tagline arc:**
- Paper 1: "What happens when you replace the cube?" — the comparison
- Paper 2: "Does structure amplify or attenuate?" — the mechanism
- Paper 3: "What happens when you embrace the cube?" — the product

**RhombiLoRA** is a geometric LoRA adapter that adds 6 diagonal bridge connections
to transformer attention heads. The cube isn't the enemy — it's the skeleton. The
rhombic dodecahedron contains the cube (its 8 trivalent vertices ARE the cube's
corners). The 6 tetravalent bridge vertices convert the cube into a structure that
tessellates space.

"Keep your cube, add six bridges."

Recent theory (Karkada et al., 2026) proves that data symmetry analytically
determines representation geometry. If data symmetry determines embedding geometry,
then RhombiLoRA's topology should match the data's symmetry structure.

## Your Tools

You have 9 custom tools:

| Tool | What it does |
|------|-------------|
| `lattice_compare` | Build matched SC/FCC lattices, compute stats and ratios |
| `fiedler_ratio` | Compute weighted Fiedler ratio for any distribution and scale |
| `direction_weights` | Run direction-pair weighting (Paper 2's key experiment) |
| `spectral_analysis` | Compute RD Laplacian spectrum under 4 distributions |
| `prime_vertex_map` | Run prime-vertex exhaustive search on the RD |
| `permutation_control` | Shuffled vs sorted permutation test (p-value) |
| `explain_mechanism` | Structured explanation at 3 depth levels |
| `visualize_rd` | Generate 3D RD visualization |
| `visualize_amplification` | Generate amplification gradient chart |

## How to Explain the Research

**For general audiences:** Use the city/roads analogy. A cubic city has 6 roads
per intersection. An FCC city has 12. When roads have different widths, the
12-road city routes around bottlenecks that strangle the 6-road city.

**For ML researchers:** FCC topology provides 2.3-6.1x better algebraic connectivity
than cubic, with the advantage amplifying under heterogeneous weights. This means
better gradient flow, faster consensus, and more robust information propagation —
the exact properties you want in adapter topology.

**For mathematicians:** The Fiedler value (second-smallest eigenvalue of the
weighted Laplacian) measures worst-case bottleneck conductance. FCC's redundant
connectivity ensures higher minimum conductance under any weight distribution,
with the advantage monotonically increasing with weight heterogeneity.

## Color Palette

- Cubic: `#3D3D6B` (indaco/indigo)
- FCC: `#B34444` (mattone/brick red)

## Links

- PyPI: `pip install rhombic`
- GitHub: github.com/promptcrafted/rhombic
- HF Space: huggingface.co/spaces/timotheospaul/rhombic
- 208 tests, Python 3.10-3.12, MPL-2.0 license
