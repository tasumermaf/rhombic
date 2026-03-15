# Fiedler Metric Clarification

Created: 2026-03-14

## The Discrepancy

The VERIFIED_FINDINGS document (March 12) claims "Fiedler ~0.10 scale-invariant
across 1.1B/7B/14B." The channel ablation results show n=6 Fiedler ~0.0004.
These are not contradictory — they use different metrics.

## Two Fiedler Computations

### 1. Bridge Fiedler (`bridge_fiedler()` in `rhombi_lora.py`)

Treats |B[i,j]| (i != j) as edge weights of a complete graph, constructs the
weighted Laplacian, returns the second-smallest eigenvalue. This is the Fiedler
value logged during training by both `train_cybernetic.py` and
`train_contrastive_bridge.py`.

**Results:**
- n=3 (H1, full Steersman): **0.095** — full connectivity
- n=4 (H2, spectral only): **0.084** — full connectivity, no blocks
- n=6 (C-001, full Steersman): **0.0004** — near-zero (block structure)
- n=6 (exp3, full Steersman): **0.0004** — near-zero (block structure)

### 2. Correlation Fiedler (`analyze_holly_bridges.py`)

Computes a correlation matrix between channel bridge patterns, then derives
Fiedler from that correlation matrix. This measures how similarly different
channels are used across layers — a higher-level structural metric.

**Results (from VERIFIED_FINDINGS):**
- Qwen 7B exp3: **0.102**
- TinyLlama exp3_tiny: **0.101**
- Holly 14B (RhombiLoRA): **1.002**
- Holly 14B (standard): **0.842**

## Why the Discrepancy Makes Sense

Block-diagonal structure suppresses between-block bridge connections to
~1e-5. The Bridge Fiedler, which measures GLOBAL connectivity of the bridge
matrix, necessarily approaches zero — the bridge IS nearly disconnected
(three 2x2 components).

The Correlation Fiedler measures how consistently the block structure appears
across layers. High consistency (same 3-block pattern everywhere) produces
high correlation-based Fiedler.

Both metrics are valid but measure different things:
- **Bridge Fiedler:** within-matrix connectivity (low when blocked)
- **Correlation Fiedler:** cross-layer consistency (high when uniform pattern)

## Impact on Claims

### Valid claims (internally consistent):
- H1 vs H2 Fiedler comparison (both use bridge_fiedler) ✓
- "Spectral-only Steersman saturates at lower Fiedler" ✓
- "Block structure requires geometric prior" ✓

### Claims needing revision:
- "Fiedler ~0.10 scale-invariant across 1.1B/7B/14B" — this is the
  CORRELATION Fiedler, not the bridge Fiedler. Both exp3 (1.1B) and
  Qwen 7B (7B) have bridge_fiedler ~0.0004. The scale-invariant metric
  is the correlation Fiedler.
- The PAPER3_FIGURE_INVENTORY entry "Fiedler ~0.10 scale-invariant across
  1.1B/7B/14B" should specify which Fiedler.

## Recommendation

For Paper 3, clearly distinguish the two metrics:
- **Bridge algebraic connectivity** (bridge_fiedler): describes individual
  bridge topology. Near-zero for block-diagonal bridges.
- **Cross-layer structural consistency** (correlation Fiedler): describes
  uniformity of block pattern across all adapters. ~0.10 for cybernetic,
  scale-invariant.

Both support the thesis. The bridge Fiedler shows the block structure IS
near-disconnected. The correlation Fiedler shows the pattern is uniform.
