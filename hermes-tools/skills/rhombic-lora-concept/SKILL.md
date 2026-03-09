---
name: rhombic-lora-concept
description: Explore the RhombiLoRA concept — a geometric LoRA adapter that adds 6 diagonal bridge connections to transformer attention heads, converting cubic topology to FCC.
version: 1.0.0
author: Promptcrafted
license: MPL-2.0
metadata:
  hermes:
    tags: [LoRA, Transformer, ML, Adapter, Geometry, Rhombic, RhombiLoRA]
    homepage: https://github.com/promptcrafted/rhombic
    related_skills: [rhombic-tutorial, rhombic-experiment]
---

# RhombiLoRA Concept Explorer

Explore the theoretical basis for RhombiLoRA — a geometric LoRA adapter
that leverages FCC lattice topology for improved gradient flow in
transformer architectures.

## Context

Load the research context from `~/hermes-agent/rhombic-context.md` before starting.

## The Three-Paper Arc

| Paper | Question | Answer |
|-------|----------|--------|
| Paper 1 | What happens when you replace the cube? | 2.3x connectivity, 30% shorter paths |
| Paper 2 | Does structure amplify or attenuate? | Amplifies to 6.1x under heterogeneous weights |
| Paper 3 | What happens when you embrace the cube? | RhombiLoRA — keep the cube, add six bridges |

## The Key Insight

The rhombic dodecahedron CONTAINS the cube. Its 8 trivalent vertices ARE
the cube's corners. The 6 tetravalent vertices are the bridges that convert
the cube into a space-filling structure.

In transformer terms: the existing attention head topology is cubic
(6-connected). RhombiLoRA adds 6 diagonal bridge connections per node,
converting it to FCC (12-connected) topology — without replacing any
existing connections.

## Demonstration Workflow

### 1. Show the Geometry

Use `visualize_rd` to generate the 3D visualization.
Point out: red dots = cube corners (existing), blue diamonds = bridges (added).

Use `prime_vertex_map` to show that the 8 cube-corner vertices have special
mathematical properties — the tracked primes map there with p = 2.5e-5.

### 2. Show the Advantage

Use `fiedler_ratio` with corpus weights to show the connectivity advantage.
Use `direction_weights` to show the amplification under structured weights.

Key argument: transformer attention weights are heterogeneous (not uniform).
Paper 2 shows that FCC advantage GROWS with heterogeneity.

### 3. Show the Mechanism

Use `explain_mechanism` at `technical` depth.

The mechanism is bottleneck resilience: when some attention paths carry
more weight than others (which they always do after training), the 6
bridge connections provide alternative routes around bottlenecks.

### 4. The Consensus Inversion

Use `explain_mechanism` at `full` depth to discuss the consensus inversion.
At small scale (125 nodes), FCC consensus is 6.7x faster. At scale 1000,
it's 0.73x (slower). This means RhombiLoRA works best at the attention
head level (small scale), not at the full-model level.

### 5. Theoretical Backing

Reference Karkada et al. (2026): "data symmetry analytically determines
representation geometry." If this is true, then:
- Data with cubic symmetry → cubic attention is sufficient
- Data with higher symmetry → FCC attention should match better
- RhombiLoRA's topology should ADAPT to the data's symmetry structure

## Implementation Sketch

```python
class RhombiLoRALayer(nn.Module):
    """6 diagonal bridge connections per attention head."""

    def __init__(self, d_model, rank=4):
        super().__init__()
        # 6 bridge directions (face-diagonal pairs)
        self.bridges = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, rank, bias=False),
                nn.Linear(rank, d_model, bias=False),
            )
            for _ in range(6)
        ])
        self.scale = 1.0 / 6.0  # Per-bridge scaling

    def forward(self, x):
        bridge_sum = sum(b(x) for b in self.bridges)
        return x + self.scale * bridge_sum
```

This is conceptual — the actual implementation maps the 6 bridge
directions to specific attention head interaction patterns.

## Discussion Topics

- Why 6 bridges? (The RD has exactly 6 tetravalent vertices)
- Why not just increase attention heads? (Topology matters, not just count)
- How does this relate to MoE? (Both route around bottlenecks, different mechanism)
- What about the edge cost? (2x edges for 6.1x connectivity — good trade)
- Could this work with other polytopes? (BCC, HCP — Paper 2 Exp 7 tested consistency)

## Rules

1. Present RhombiLoRA as a research concept, not a finished product
2. Use live computations to back every claim
3. Always show both the advantage AND the cost
4. Reference Karkada et al. as theoretical motivation, not proof
5. The tagline is: "Keep your cube, add six bridges."
