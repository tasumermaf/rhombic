# AR-001: Asymmetry Analysis of Bridge Matrices

**Date:** 2026-03-19
**Status:** COMPLETE
**Script:** `scripts/ar001_asymmetry_analysis_v2.py`
**Raw data:** `results/AR-001-raw-results.json`

---

## Question

Why are bridge matrices asymmetric, and does the asymmetry carry structural information?

## Method

For each bridge matrix B (n x n), compute the unique additive decomposition:
- **S = (B + B^T) / 2** — symmetric part (encodes undirected coupling)
- **A = (B - B^T) / 2** — antisymmetric (skew-symmetric) part (encodes directed flow)
- **B = S + A** (orthogonal decomposition; ||B||^2 = ||S||^2 + ||A||^2)

Key metrics:
- **Asymmetry ratio:** ||A||_F / ||B||_F (0 = perfectly symmetric, 1/sqrt(2) ~= 0.707 = maximally asymmetric)
- **Co/cross ratio** on B, S, and A separately (within-block vs cross-block coupling)
- Block structure uses **consecutive 2x2 pairs**: (0,1), (2,3), (4,5), ...

## Experiments Analyzed

| ID | n | Geometry | Model | BD? | Bridges |
|----|---|----------|-------|-----|---------|
| H-ch3 | 3 | None (spectral-only) | TinyLlama-1.1B | No | 88 |
| H-ch4 | 4 | Octahedron | TinyLlama-1.1B | No* | 88 |
| Seed-43 | 6 | RD | TinyLlama-1.1B | Yes | 88 |
| Seed-44 | 6 | RD | TinyLlama-1.1B | Yes | 88 |
| exp3-Qwen | 6 | RD | Qwen2.5-7B | Yes | 112 |
| T-001-full | 8 | Tesseract | TinyLlama-1.1B | Yes | 88 |
| T-001r2 | 8 | Tesseract | TinyLlama-1.1B | Yes | 88 |

*H-ch4 note: Contrastive loss was disabled for n=4 (hardcoded n==6 check in Steersman), so H-ch4 ran spectral-only. Bridge files are from step 600 of 10K (only early checkpoints saved locally). No BD expected or observed.

---

## Finding 1: Asymmetry Is Bimodal — BD Experiments Are 16x More Asymmetric

| Experiment | n | ||A||/||B|| Mean | Std | Min | Max |
|------------|---|------------------|-----|-----|-----|
| H-ch3 | 3 | **0.019** | 0.013 | 0.002 | 0.051 |
| H-ch4 | 4 | **0.014** | 0.008 | 0.001 | 0.031 |
| Seed-43 | 6 | **0.320** | 0.016 | 0.268 | 0.349 |
| Seed-44 | 6 | **0.320** | 0.016 | 0.279 | 0.347 |
| exp3-Qwen | 6 | **0.496** | 0.021 | 0.441 | 0.531 |
| T-001-full | 8 | **0.303** | 0.017 | 0.256 | 0.332 |
| T-001r2 | 8 | **0.319** | 0.019 | 0.269 | 0.351 |

**Interpretation:** Experiments that develop block-diagonal structure (Seed-43/44, exp3-Qwen, T-001-full/r2) have asymmetry ratios of **0.30-0.50**, meaning the antisymmetric part carries 30-50% of the Frobenius norm. Non-BD experiments (H-ch3, H-ch4) are nearly symmetric at ~1.5-2% asymmetry. The Qwen 7B model (exp3-Qwen) is the most asymmetric at ~50%, suggesting larger models develop stronger directional flow.

The distribution is bimodal with no overlap: BD experiments are 16x more asymmetric than non-BD. This is a **necessary feature** of block-diagonal structure, not an artifact.

---

## Finding 2: Symmetrization Catastrophically Destroys the BD Signal

| Experiment | cc(B) | cc(S) | cc(A) | S retains | A retains |
|------------|-------|-------|-------|-----------|-----------|
| H-ch4 | 1.0 | 1.0 | 2.0 | 99% | 196% |
| Seed-43 | **73,309** | 39 | 38 | **0.05%** | 0.05% |
| Seed-44 | **70,201** | 38 | 38 | **0.05%** | 0.05% |
| exp3-Qwen | **33,458** | 66 | 70 | **0.2%** | 0.2% |
| T-001-full | **20,944** | 41 | 40 | **0.2%** | 0.2% |
| T-001r2 | **41,564** | 42 | 42 | **0.1%** | 0.1% |

**This is the headline finding.** Symmetrizing the bridge matrix destroys the co/cross ratio from **tens of thousands** down to **~40** — a loss of **99.95%** of the block-diagonal signal. The antisymmetric part A carries no more co/cross signal than S does (both ~40).

**Why?** The block-diagonal structure is encoded in the **interaction between S and A**, not in either component alone. The BD signal lives in the fact that B[i,j] (upper co-pair entry) is large (~0.8) while B[j,i] (lower co-pair entry) is small (~0.03). When you symmetrize, both become ~0.4 — still large — but now the cross-block entries (which were ~0 in both directions) are ALSO ~0.02 in both directions. The ratio of ~0.4 to ~0.02 gives cc(S) ~ 40, not 70,000.

---

## Finding 3: The Asymmetry IS the BD Structure — Directional Coupling

Within each 2x2 block (i,j), the coupling is **massively asymmetric**:

| Experiment | Mean |B[i,j]| | Mean |B[j,i]| | Ratio |
|------------|-----------------|-----------------|-------|
| Seed-43 | 0.777 | 0.026 | **30:1** |
| T-001r2 | 0.771 | 0.023 | **34:1** |
| H-ch4 | 0.004 | 0.003 | 1:1 |

For BD experiments, the upper-triangle co-pair entries (B[i,j] where i < j) are **~30x larger** than the lower-triangle entries (B[j,i]). This means information flows predominantly from **channel i to channel j** within each block, not bidirectionally.

The sign of B[i,j] is **mixed** across layers/projections (sometimes positive, sometimes negative), meaning the directionality is not a simple scaling but encodes rotation-like dynamics within each 2x2 block.

---

## Finding 4: Positive Correlation Between Asymmetry and BD Strength

| Experiment | Correlation (asym vs log10 cc) | N |
|------------|-------------------------------|---|
| T-001r2 | **+0.72** | 88 |
| T-001-full | **+0.70** | 88 |
| Seed-43 | **+0.63** | 88 |
| Seed-44 | **+0.61** | 88 |
| exp3-Qwen | **+0.51** | 112 |
| H-ch4 | +0.05 (n.s.) | 88 |

Within BD experiments, bridges with **higher asymmetry** have **stronger co/cross ratios**. The correlation is moderate-to-strong (r = 0.51-0.72). This confirms that asymmetry is not noise — it is the mechanism by which BD structure is expressed.

---

## Finding 5: The SVD of A Reveals Rank Structure

The antisymmetric part A of a skew-symmetric matrix has singular values that come in equal pairs (by theorem). The singular values show:

**Seed-43 (n=6, 3 pairs):**
| Pair | Mean SV | % of total |
|------|---------|------------|
| 1-2 | 0.412 | 35.6% |
| 3-4 | 0.386 | 33.4% |
| 5-6 | 0.358 | 31.0% |

**T-001r2 (n=8, 4 pairs):**
| Pair | Mean SV | % of total |
|------|---------|------------|
| 1-2 | 0.415 | 27.2% |
| 3-4 | 0.392 | 25.6% |
| 5-6 | 0.373 | 24.4% |
| 7-8 | 0.348 | 22.8% |

The SVD spectrum is **nearly flat** — all singular value pairs are within ~15% of each other. This means A does not have low-rank structure; the antisymmetric part uses its full rank. The slight decrease from pair 1 to pair n/2 mirrors the layer-depth gradient (early layers have slightly larger A).

---

## Finding 6: Eigenvalues of S Show Clean Two-Cluster Spectrum

The symmetric part S has eigenvalues that cluster into two groups:

**Seed-43 (n=6):** {1.50, 1.47, 1.44} and {0.71, 0.68, 0.66}
**T-001r2 (n=8):** {1.49, 1.47, 1.45, 1.42} and {0.71, 0.69, 0.67, 0.65}
**H-ch3 (n=3):** {1.16, 1.09, 1.04} — no gap, single cluster

For BD experiments, S has n/2 eigenvalues near 1.5 and n/2 near 0.7. The gap between the two clusters is the spectral signature of the 2x2 block structure surviving into S (each 2x2 block contributes one eigenvalue above 1 and one below 1). H-ch3 has no such gap.

---

## Finding 7: Projection Hierarchy in Asymmetry

| Projection | H-ch3 | Seed-43 | T-001r2 | exp3-Qwen |
|------------|-------|---------|---------|-----------|
| k_proj | 0.021 | **0.328** | **0.331** | **0.514** |
| v_proj | 0.016 | **0.329** | **0.330** | **0.504** |
| q_proj | 0.020 | **0.319** | **0.319** | **0.486** |
| o_proj | 0.021 | **0.303** | **0.297** | **0.481** |

k_proj and v_proj are consistently the most asymmetric, followed by q_proj, with o_proj the least asymmetric. This mirrors the known projection hierarchy for BD strength (k_proj >> o_proj). The asymmetry-BD connection holds at the per-projection level.

---

## Finding 8: Asymmetry Gradient by Layer Depth

Both asymmetry and co/cross ratio decrease with layer depth:

**Seed-43:** Layer 0 asymmetry = 0.335, Layer 21 = 0.304 (9% decrease)
**T-001r2:** Layer 0 asymmetry = 0.341, Layer 21 = 0.307 (10% decrease)
**Seed-43:** Layer 0 cc(B) = 176,990, Layer 21 = 58,552 (67% decrease)

This is consistent with the established Layer-Projection BD Gradient finding — early layers develop stronger BD (and therefore stronger asymmetry).

---

## Finding 9: The BD Signal Lives Entirely in the Upper Triangle

Examining the raw matrix entries of Seed-43's highest-cc bridge (Layer 1, k_proj):

| Region | Co-pair mean |abs| | Cross-pair mean |abs| | Co/Cross |
|--------|--------------------|-----------------------|----------|
| **Upper triangle** (B[i,j], i < j) | 0.812 | 0.000002 | **342,739** |
| **Lower triangle** (B[j,i], i < j) | 0.030 | 0.033 | **0.9** |

The block-diagonal signal is concentrated **entirely** in the upper triangle. The lower triangle has small values everywhere — co-pair entries (~0.03) are indistinguishable from cross-pair entries (~0.03). This means:

- The **upper triangle** encodes the block structure with extreme precision (co/cross ~ 300,000+)
- The **lower triangle** is essentially structureless noise at ~0.03 magnitude
- The **diagonal** is close to identity (~1.05 per entry)

When symmetrized, S[i,j] = (B[i,j] + B[j,i])/2 averages a structured value (~0.8 for co, ~0 for cross) with an unstructured value (~0.03 everywhere). The result: co entries drop from 0.8 to 0.42, but cross entries rise from ~0 to ~0.02, destroying the ratio.

---

## Theoretical Interpretation

**The bridge matrix B can be decomposed as B = S + A where S and A are orthogonal components.** In the BD case:

1. **S encodes the coupling magnitude** — how strongly each pair of channels interacts, regardless of direction. S sees the 2x2 blocks but at ~40:1, not 70,000:1.

2. **A encodes the coupling direction** — within each 2x2 block, which channel drives which. A also sees the blocks at comparable strength (~40:1).

3. **The extreme co/cross ratio (~70,000:1) arises from constructive/destructive interference between S and A.** For a co-pair (i,j): B[i,j] = S[i,j] + A[i,j] ~ 0.42 + 0.38 = 0.80 (constructive). B[j,i] = S[i,j] - A[i,j] ~ 0.42 - 0.38 = 0.04 (destructive). For cross-pairs: both S and A are ~0, so B remains ~0 in both directions. The extreme ratio comes from the **near-cancellation** in the destructive direction.

4. **This is a rotation-like structure.** Each 2x2 block of B approximates:
   ```
   [  d,    s+a  ]     [  1.04,   0.80 ]
   [ s-a,    d   ]  ~  [  0.04,   1.04 ]
   ```
   where d ~ 1 (near-identity diagonal), s ~ 0.42 (symmetric coupling), a ~ 0.38 (antisymmetric coupling). The near-equality s ~ a is what produces the extreme upper/lower asymmetry. If a were exactly equal to s, the lower entry would be exactly zero.

5. **Non-BD experiments (H-ch3, H-ch4) have A ~ 0**, so B ~ S and the matrix is nearly symmetric. There is no directional structure because there are no blocks to be directional within.

6. **The energy partition** (squared Frobenius norms): S carries ~89% and A carries ~11% of ||B||^2. Despite A being the smaller component, it is essential — removing it destroys the BD signal.

**Physical interpretation:** The bridge learns **directed information flow** between paired channels. Channel i "writes to" channel j strongly, but channel j barely writes back. This is not a symmetric coupling (like a spring between two masses) but a directed flow (like a valve or diode). The contrastive loss creates the blocks; the gradient dynamics create the directionality within each block.

**Mathematical interpretation:** The bridge is a **near-shear** transformation. In each 2x2 block, the off-diagonal coupling is almost entirely one-directional. The symmetric part S encodes that channels are paired; the antisymmetric part A encodes which channel in each pair is the "driver" and which is the "passenger." Together they produce the extreme BD ratios; alone, each carries only moderate structure.

---

## Key Numbers for the Paper

- BD asymmetry ratio: **0.30-0.50** (vs 0.02 non-BD; 16x)
- Symmetrization destroys **99.95%** of co/cross signal
- Within-block coupling ratio: **30:1** (B[i,j] vs B[j,i])
- Asymmetry-BD correlation: **r = 0.51-0.72** (moderate-strong)
- S eigenvalue gap: {~1.5, ~0.7} two-cluster spectrum in BD
- A SVD: nearly flat, full-rank (no low-rank structure in the antisymmetric part)

---

## Implications

1. **Bridge matrices should NOT be symmetrized** in any analysis or regularization. Symmetrization is catastrophically destructive to the learned structure.

2. **The asymmetry is not noise** — it carries the BD signal and correlates positively with BD strength. It is a **necessary feature**, not an artifact.

3. **The bridge encodes directed information flow**, not symmetric coupling. Future work should investigate what the direction means in terms of the LoRA adaptation's effect on the model's representations.

4. **The 2x2 block structure is a rotation-like structure** where S and A destructively interfere in one direction and constructively in the other. This is reminiscent of creation/annihilation operators in quantum mechanics or the symplectic structure in Hamiltonian mechanics.

5. **For the Qwen 7B model**, asymmetry is ~50% vs ~30% for TinyLlama, suggesting that larger models may develop even more pronounced directional structure. This warrants investigation with more model sizes.
