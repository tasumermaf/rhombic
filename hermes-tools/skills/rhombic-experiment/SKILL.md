---
name: rhombic-experiment
description: Run and analyze rhombic lattice topology experiments interactively. Reproduce Paper 2 findings, explore parameter variations, and generate publication figures.
version: 1.0.0
author: Promptcrafted
license: MPL-2.0
metadata:
  hermes:
    tags: [Lattice, Experiment, Spectral, Graph Theory, Research, Rhombic]
    homepage: https://github.com/promptcrafted/rhombic
    related_skills: [rhombic-tutorial, rhombic-lora-concept]
---

# Rhombic Experiment Runner

Run lattice topology experiments interactively and analyze results.
All experiments use the rhombic custom tools for live computation.

## Context

Load the research context from `~/hermes-agent/rhombic-context.md` before starting.

## Available Experiments

### Experiment 1-3: Edge-Cycled Weighting

Use `fiedler_ratio` across all four distributions at multiple scales.
Reproduce the finding that corpus weights amplify the Fiedler ratio from 2.3x to 3.1x.

```
Run fiedler_ratio for each combination:
  distributions: uniform, random, power_law, corpus
  scales: 3, 5, 7
```

### Experiment 4: RD Degeneracy Breaking

Use `spectral_analysis` under all four distributions.
Show that uniform weights give 6 distinct eigenvalues (high degeneracy)
while corpus weights break to 14 distinct (maximum resolution).

### Experiment 5: Direction-Pair Weighting

Use `direction_weights` at multiple scales with corpus distribution.
This is the key finding: sorted-bucketing amplifies the Fiedler ratio to 6.1x.

Follow up with `permutation_control` to confirm the effect comes from
sorted alignment, not just bin count.

### Experiment 6: Prime-Vertex Mapping

Use `prime_vertex_map` to run the exhaustive search.
Expected result: score 43.0, p = 2.5e-5.
The 8 tracked primes map optimally to the 8 cube-corner vertices of the RD.

### Experiment 7: Cross-Topology Consistency

Not directly available as a single tool, but `spectral_analysis` on the RD
demonstrates the corpus's spectral signature. Explain that Paper 2 tested
5 different 24-edge graphs and found consistent suppression (0.02%-5.94%).

## Workflow

1. **Ask what the user wants to explore** — specific experiment, parameter sweep, or full reproduction
2. **Run computations live** using the appropriate tools
3. **Present results** with context from Paper 2 findings
4. **Generate visualizations** when they help: `visualize_amplification` for the gradient, `visualize_rd` for the geometry
5. **Discuss implications** — what do these numbers mean for network design?

## Parameter Exploration

Encourage users to try:
- Different scales (how does the effect change with size?)
- Different distributions (what properties of the corpus drive amplification?)
- The permutation control (does order matter? Yes — sorted-bucketing concentrates extreme values)

## Rules

1. Run every computation live — never fabricate results
2. Present results in context: "Paper 2 found X at scale 1000; at your scale N we see Y"
3. When results differ from Paper 2 (especially at small scale), explain boundary effects
4. Generate figures to /tmp and offer to show them
5. Always report both the advantage AND the cost (edge ratio)
