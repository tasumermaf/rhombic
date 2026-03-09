# Why Geometric Data Should Teach the Bridge: A Gradient Analysis

> **Status:** Theoretical argument, pre-empirical validation.
> **Target:** Paper 3 §3 (Theoretical Grounding)

---

## The Claim

A 6×6 learnable bridge matrix between LoRA's down- and up-projections
will develop direction-pair-specific coupling under training data with
directional structure, but remain near-identity under isotropic data.
The coupling pattern should converge toward the rhombic dodecahedron's
geometric prior: co-planar pairs (4 shared vertices) developing stronger
coupling than cross-planar pairs (2 shared vertices).

## Architecture

Standard LoRA computes: `dW(x) = B @ A @ x`, where A is (r, d_in)
and B is (d_out, r).

RhombiLoRA reshapes the rank-r hidden state into C channels of size
r/C, applies a C×C bridge, then reshapes back:

```
h = A @ x                          # (r,)
h = h.reshape(C, r/C)              # (6, 4) for rank=24, C=6
h = bridge @ h                     # (6, 4) — channel mixing
h = h.reshape(r)                   # (r,)
dW(x) = B @ h                      # (d_out,)
```

The bridge is the **only** addition. With bridge = I₆, this is exactly
standard LoRA. The bridge has 36 parameters per adapter (6² = 36) out of
~135K per adapter (2 × rank × d_model), a 0.03% addition.

## Gradient Flow Through the Bridge

The loss gradient with respect to bridge element B[i,j] is:

```
∂L/∂B[i,j] = Σ_k (∂L/∂h_out[i,k]) · h_in[j,k]
```

where h_in[j,k] is the k-th rank-slice of channel j before the bridge,
and h_out[i,k] is channel i after the bridge, receiving gradient from
the up-projection B matrix.

**Key insight:** B[i,j] receives gradient proportional to the
**co-activation** between channel i (in the up-projection's view)
and channel j (in the down-projection's output). If the training
data creates correlated activation patterns between specific channel
pairs, those bridge elements grow; if activations are uncorrelated,
the bridge remains near-identity.

## Why Isotropic Data Produces Near-Identity Bridges

With Alpaca-cleaned (isotropic instruction data):

1. No systematic correlation between reasoning directions
2. Channel activations are approximately independent
3. Off-diagonal gradients cancel across the batch (positive and
   negative co-activations roughly equal)
4. Bridge stays near I₆ with small random perturbation

**Exp 2 confirms this:** eigenvalues [1.02, 1.04, 1.06, 1.08, 1.10, 1.13],
Gram distance 0.92 from RD coupling, co/cross ratio 1.019, permutation
p = 0.33 (not significant).

## Why Geometric Data Should Differentiate

The geometric dataset is **structured** to create co-planar bias:

### Component 1: Multi-Perspective Reasoning (95% co-planar)
Each example asks for analysis from N perspectives, where the selected
perspectives are drawn from **co-planar pairs** with 95% probability.
This means analytical+empirical (pair 1-2), ethical+practical (3-4),
or creative+systemic (5-6) are co-activated far more often than
analytical+ethical (cross-planar 1-3).

### The Gradient Consequence

When the LLM processes a "compare analytical and empirical viewpoints"
example, the gradient signal flowing through the bridge creates:
- **Strong co-activation** between channels 1 and 2 (analytical ↔ empirical)
- **Weak or absent co-activation** between channels 1 and 3 (analytical ↔ ethical)

Over thousands of co-planar-biased examples, the gradient accumulation
produces:
```
∂L/∂B[1,2] >> ∂L/∂B[1,3]    (co-planar vs cross-planar)
```

The bridge matrix should develop a **block-diagonal bias** — stronger
coupling within co-planar pairs (1-2, 3-4, 5-6) and weaker coupling
across them. This IS the RD coupling pattern (co-planar = 4, cross = 2).

## The Measurement

We define four metrics to detect this convergence:

### 1. Co/Cross Ratio (primary)
```
ratio = mean(|B[i,j]| for co-planar (i,j)) / mean(|B[i,j]| for cross-planar (i,j))
```
Threshold: >1.10 (10% stronger co-planar than cross-planar coupling).
Exp 2 baseline: 1.019 (not significant).

### 2. Gram Alignment (structural)
```
distance = ||B@B^T / ||B@B^T||_F - C_theo / ||C_theo||_F||_F
```
where C_theo is the RD coupling matrix [diag=4, co=4, cross=2].
Measures how closely the learned bridge's self-interaction matches
the theoretical geometric prior. Lower = more aligned.
Exp 2 baseline: 0.9210.

### 3. Permutation Test (statistical)
Shuffle channel labels 10K times, recompute co/cross ratio each time.
P-value = fraction of shuffles ≥ observed ratio.
Significance at p < 0.05.
Exp 2 baseline: p = 0.332 (noise).

### 4. Perturbation Test (robustness)
Zero all co-planar entries, measure Fiedler value retention.
If structure is distributed (co-planar entries contribute but don't
dominate), retention > 50%.
Exp 2 baseline: 68.8% retention (already distributed — expected for
near-identity bridges with small random perturbation).

## The Transient Signal in Exp 2 (Critical Observation)

The Exp 2 co/cross ratio trajectory reveals a **transient directional
peak** that informs the experimental design:

| Step | Co/Cross | Fiedler | Interpretation |
|------|----------|---------|----------------|
| 1000 | 0.929 | 0.018 | Cross-planar stronger (random init) |
| 2000 | 1.023 | 0.017 | Crosses 1.0 |
| **3000** | **1.091** | **0.015** | **Peak — nearly hits 1.10 threshold** |
| 4000 | 1.063 | 0.021 | Starts declining |
| 5000 | 1.043 | 0.024 | Decaying |
| 10000 | 1.019 | 0.040 | Converges toward 1.0 |

The bridge goes through three phases under isotropic data:
1. **Exploration** (0-2K): Random perturbation, no direction preference
2. **Transient bias** (2K-4K): Brief co-planar preference emerges from
   training dynamics (initial weight correlation patterns)
3. **Regularization** (4K+): Isotropic data washes out the bias,
   co/cross converges to 1.0 while Fiedler grows (more connected,
   less directional)

**This is why we chose 3000 steps for Exp 2.5.** The Alpaca run peaks
at step 3000. If geometric data sustains or strengthens the co/cross
ratio at step 3000 (where isotropic data shows 1.091), the dataset
effect is confirmed. Specifically:

- **Exp 2 at 3K: 1.091** → transient, would decay to 1.019
- **Exp 2.5 at 3K: ???** → if >1.10 AND sustained, geometric data
  creates a stable (not transient) directional signal

The Fiedler-vs-co/cross anticorrelation in Exp 2 is itself a finding:
bridges that are MORE connected (higher Fiedler) are LESS directional
(lower co/cross). Geometric data should break this anticorrelation —
producing bridges that are both connected AND directional.

## Predictions

If the geometric dataset teaches directional structure:

| Metric | Exp 2 (Alpaca) | Exp 2.5 (Geometric) — predicted |
|--------|----------------|----------------------------------|
| Co/Cross ratio | 1.019 | >1.10 (pass), possibly >1.20 |
| Permutation p | 0.332 | <0.05 (significant) |
| Gram distance | 0.921 | <0.85 (closer to RD) |
| Eigenvalue spread | 0.107 | >0.30 (learning structure) |
| Perturbation retention | 68.8% | >50% (structure distributed) |

**Pass criteria:** Co/Cross >1.10 AND p < 0.05. If both met, proceed
to Exp 3A-3C (MoE routing, cross-domain, symmetry).

**Null result:** If co/cross ≈ 1.0 and p > 0.05, the dataset does not
create sufficient gradient signal. Options:
1. Increase co-planar bias beyond 95%
2. Pre-train bridge separately (contrastive objective)
3. Increase training steps beyond 3K
4. Reduce bridge learning rate relative to A/B (currently same lr)

## Why This Matters Beyond LoRA

The bridge is a **geometric inductive bias** — it tells the optimizer
that the rank-r hidden space has structure. Standard LoRA treats
rank-r as a flat vector. RhombiLoRA treats it as a lattice of
direction-pair channels with defined coupling.

If training data can teach the bridge to recover the RD coupling
matrix from data alone, this proves:
1. Neural networks can learn lattice topology from structured input
2. Geometric inductive biases create measurable, direction-specific
   parameter organization
3. The information-theoretic advantage of FCC (12-connected, 2.4×
   algebraic connectivity) over cubic (6-connected) is not just
   graph-theoretic — it manifests in learned representations

This positions RhombiLoRA as the first LoRA variant where the
**topology of the hidden space is a measurable, trainable quantity**
rather than an arbitrary vector partition.

---

*Analysis: March 8, 2026. Pre-empirical. To be revised after Exp 2.5
results.*
