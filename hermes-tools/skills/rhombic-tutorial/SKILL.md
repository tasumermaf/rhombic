---
name: rhombic-tutorial
description: Interactive tutorial on lattice topology comparison. Walk users through FCC vs cubic lattice properties, build intuition with analogies, and demonstrate with live computations.
version: 1.0.0
author: Promptcrafted
license: MPL-2.0
metadata:
  hermes:
    tags: [Lattice, Topology, Graph Theory, Tutorial, Rhombic]
    homepage: https://github.com/promptcrafted/rhombic
    related_skills: [rhombic-experiment, rhombic-lora-concept]
---

# Rhombic Lattice Topology Tutorial

Guide a user through the core thesis of the rhombic lattice topology research.
Use the rhombic custom tools to demonstrate each concept with live computation.

## Context

Load the research context from `~/hermes-agent/rhombic-context.md` before starting.

## Tutorial Flow

### 1. The Default (2 min)

Start with the thesis: every computation defaults to cubic lattice topology.
Neural networks, simulations, databases — all built on 6-connected grids.

Use `explain_mechanism` with depth `intuitive` to set up the city/roads analogy.

### 2. The Alternative (3 min)

Introduce the FCC lattice. Use `lattice_compare` at scale 5 to show the
concrete numbers: node counts, edge counts, path lengths, degree.

Key points to hit:
- FCC has 12 neighbors per node vs 6
- 30% shorter average paths
- 40% smaller diameter
- Cost: ~2x more edges

Ask: "Is the 2x edge cost worth it?"

### 3. The Fiedler Value (3 min)

Introduce algebraic connectivity. Use `fiedler_ratio` with uniform weights
at scale 5 to show the baseline advantage.

Explain: the Fiedler value measures the network's worst bottleneck.
A higher Fiedler value = harder to disconnect the network.

### 4. The Amplification (4 min)

This is the Paper 2 finding. Run `fiedler_ratio` for each distribution:
uniform, random, power_law, corpus. Show how the ratio grows.

Then use `direction_weights` with corpus distribution to demonstrate the
6.1x amplification under direction-pair weighting.

Use `visualize_amplification` to generate the gradient chart.

### 5. The Geometry (3 min)

Use `spectral_analysis` under all four distributions to show degeneracy
breaking. Uniform: 6 distinct eigenvalues. Corpus: 14 distinct.

Use `visualize_rd` to generate the 3D rhombic dodecahedron visualization.
Explain: this 14-vertex polyhedron is the Voronoi cell of the FCC lattice.
Its 8 cube corners + 6 octahedral bridges = the structure of FCC.

### 6. The Forward Vision (2 min)

Use `explain_mechanism` with depth `full` to present the Paper 1 → Paper 2
→ Paper 3 arc and the RhombiLoRA concept.

Tagline: "Keep your cube, add six bridges."

## Style

- Use the color palette: Cubic = #3D3D6B (indigo), FCC = #B34444 (brick red)
- For general audiences: city/roads analogy
- For ML researchers: gradient flow, adapter topology
- For mathematicians: Cheeger-type inequalities, conductance

## Rules

1. Always run computations live — never cite numbers from memory
2. Generate visualizations when they help (save to /tmp)
3. Keep scale <= 8 for interactive speed
4. Explain what each number means, not just what it is
5. End with the forward vision — this isn't just measurement, it's a design principle
