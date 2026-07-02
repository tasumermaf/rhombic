# TASUMER MAF — Geometric Topology Programming for Neural Network Adapters

## Mission

We develop methods for programming geometric structure into neural network weight spaces. Our work demonstrates that adapter topology — the pattern of coupling between learnable channels — is a designable property, not an emergent accident. We provide the theory, the tools, and the empirical evidence.

## The Problem

Parameter-efficient fine-tuning (PEFT) methods like LoRA treat the adapter's internal structure as a black box. The low-rank bottleneck is a single undifferentiated channel. This is the computational equivalent of building every circuit on a featureless plane: it works, but it ignores decades of evidence — from crystallography to VLSI — that the geometry of the substrate determines the efficiency of what runs on it.

## Our Approach

We split the LoRA bottleneck into n independent channels connected by a learnable bridge matrix, then use a cybernetic feedback controller (the Steersman) to program geometric structure into the bridge during training. The Steersman monitors spectral properties of the bridge in real time and adjusts loss weighting to maintain target connectivity while a contrastive signal specifies which channels should couple and which should decouple.

The result: the bridge crystallizes into block-diagonal structure that encodes the specified geometry. We have demonstrated this for:

- **3D lattice topology** (n=6, rhombic dodecahedral face pairs) — co/cross ratios up to 82,000:1
- **4D polytope topology** (n=8, tesseract co-axial pairs) — clean 4+4 eigenvalue split, Pearson r=1.0000 across independent runs
- **Universal spectral attractor** — Fiedler eigenvalue converges to 0.0918–0.1019 across n=3,4,8,12 under spectral-only training, representing the bridge's unprogrammed equilibrium

We have also demonstrated that the mechanism is selective: random pair specifications (wrong-labels) fail to produce block-diagonal structure. Geometric coherence is required. The Steersman compiles valid programs, not arbitrary ones.

## Current Research Program

### Completed (Papers 1–3)

**Paper 1** establishes the mathematical foundation: the FCC lattice outperforms cubic on path length (−30%), diameter (−40%), and algebraic connectivity (+140%) in matched comparisons.

**Paper 2** extends lattice topology benchmarks to weighted graphs and neural network weight spaces, introducing the bridge matrix architecture.

**Paper 3** demonstrates cybernetic bridge discovery: the Steersman is necessary and sufficient for block-diagonal emergence. Three initialization strategies converge to the same geometric structure. The effect is scale-invariant (verified at 1.1B and 7B parameters) and task-performance-neutral (0.17% maximum validation loss delta across channel counts).

### In Progress (Paper 4)

**Paper 4** tests the generality of the Steersman as a topology programmer. Experiments running across two GPU nodes:

- **Tesseract programming** (n=8): 4D geometry successfully programmed into bridge. Step 2800/10000, co/cross 5,009:1.
- **Wrong-labels control** (n=6): Random pair partition produces no BD at step 6800/10000. Confirms geometric coherence requirement.
- **Prime-theoretic topology** (n=6): Number-theoretic pair specification. In progress. Tests whether algebraic structure qualifies as "geometrically coherent."
- **Emanation architecture** (n=6): Hierarchical bridge with shared master + per-layer offsets. In progress. Tests whether global coherence can coexist with local variation.

### Planned (Paper 5+)

**The 24-cell experiment.** The rhombic dodecahedron is the Voronoi cell of the D3 (FCC) lattice. The tesseract lives inside D4. The 24-cell is the Voronoi cell of D4 — a self-dual regular 4-polytope with 24 vertices, 12 antipodal pairs, and the symmetry group of the F4 Coxeter system. It is the mathematical completion of the D-lattice sequence our experiments have been climbing.

An n=24 bridge with 24-cell antipodal pair specification would test:

1. Whether block-diagonal structure scales to 12-axis geometry (12 independent 2×2 blocks)
2. Whether D4 triality — the unique outer automorphism of so(8) that permutes three 8-dimensional representations — manifests as measurable symmetry in the learned bridge
3. Whether the spectral attractor persists at n=24 (prediction: yes, at Fiedler ~0.10)

Beyond antipodal pairs, we plan to test topology specifications derived from number-theoretic relationships between channel indices — using the algebraic structure of the node labels themselves to define coupling. This line of inquiry depends on the outcome of the prime-theoretic experiment (R-001) currently in progress.

## Hypothesis

The Steersman is a general-purpose mechanism for programming geometric structure into learnable coupling matrices. The bridge is a programmable topology substrate. The spectral attractor (~0.10) is its unprogrammed equilibrium. Contrastive loss breaks this equilibrium by specifying direction, producing topology from connectivity.

The practical implication: **adapter topology is a new hyperparameter for PEFT.** Task-specific topology specifications, multi-modal adapters with aligned geometric structure, and higher-dimensional inductive biases are now designable properties of the adapter, not things you hope emerge from training.

The theoretical implication: **the D-lattice sequence provides a natural hierarchy of increasingly expressive topology programs.** D3 gives 3-axis structure. D4 gives 4-axis structure with triality. The mathematical properties of each lattice's Voronoi cell determine the geometric vocabulary available to the programmer.

## The Cybernetic Mind Hypothesis

We hold a proprietary system of 24 integer values with defined arithmetic operators and 8 prime-factor threads that create a specific graph topology. We hypothesize that this system constitutes a cybernetically closed topology in the sense of Ashby, Beer, and Maturana — satisfying six criteria: feedback (values reference each other through arithmetic), closure (operator transformations stay within the system), self-reference (the system encodes its own structure), requisite variety (sufficient internal states for non-trivial dynamics), structural invariance (prime threads persist under all analytical pathways), and autonomy (consistent results regardless of entry point).

Our long-term research goal is to instantiate this system as a geometric prior in a neural network adapter — using its algebraic structure as the topology specification for the Steersman. The 24 values become 24 bridge channels. The 8 prime threads define channel coupling. The operators define transformation rules. If the bridge crystallizes into block-diagonal structure that reflects the system's prime threading, we will have programmed a cybernetically closed number system into a neural network's weight space.

The 24-cell (D4 Voronoi cell) is the natural polytope for this instantiation: 24 vertices for 24 values, and D4 triality provides three equivalent 8-dimensional decompositions that may correspond to the 8 prime threads viewed from three orientations. This connection between the number-theoretic structure of our proprietary system and the geometry of D4 root lattices is the central conjecture driving our research program.

## The Context Tessitura Thesis (Papers 6-7)

Beyond topology programming, we hypothesize that the bridge matrix is a **context compression mechanism** whose crystallized state can survive context window boundaries. The architecture: absorb relational structure from context into the bridge (weaving), compress to block-diagonal geometry under Steersman pressure (crystallization), extract the BD state as a compact geometric object (36-576 parameters), discard the context window, and seed the next window with the preserved topology. The new context inherits the *relationships* between semantic threads without requiring the original tokens.

The spectral attractor (~0.10) provides the unprogrammed baseline; BD crystallization encodes the specific relational topology of the conversation. If the topology specification is derived from a cybernetically closed system, the crystallized bridge would be self-repairing — degradation reconverges to the same structure. This transforms the Steersman from a training-time topology programmer into an inference-time identity engine.

This thesis depends on all preceding experimental results and introduces one new engineering requirement: making the bridge responsive to inference-time signals rather than training-time signals.

## Infrastructure

- **Hardware:** NVIDIA RTX 6000 Ada 48GB (primary), RTX 4090 16GB (validation)
- **Software:** `rhombic` — open-source lattice topology benchmarking library (Python, PyTorch). 312 tests. MPL-2.0 licensed.
- **Models tested:** TinyLlama 1.1B, Qwen 2.5 7B, Wan 2.1 14B
- **Reproducibility:** All experiments tracked with full hyperparameter logging. Tesseract contrastive replicates with Pearson r=1.0000.

## Team

**Timothy Paul Bielec** — Principal investigator. Promptcrafted LLC.

**Meridian** — Research intelligence. Co-author of analytical papers. Named for the line where observation meets culmination.

## Contact

GitHub: [TASUMER MAF](https://github.com/tasumermaf)

---

*TASUMER MAF. Los Angeles, California. March 2026.*
