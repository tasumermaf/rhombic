# Bridge Block-Diagonal Finding — Paper 3 Core Result

Generated: 2026-03-13T12:05
Updated: 2026-03-13T21:30 — temporal emergence analysis (27,056 checkpoint bridges)

## Discovery

The cybernetic feedback loop (Steersman) causes 6×6 bridge matrices to
self-organize into 3 independent 2×2 co-planar blocks aligned with the
rhombic dodecahedron's 3 coordinate axes.

## Evidence

### Universal across cybernetic experiments

| Dataset | Model | Bridges | Block-Diagonal | Co/Cross Ratio |
|---------|-------|---------|---------------|----------------|
| C-001 (identity init) | TinyLlama 1.1B | 88 | **100%** | **27:1** |
| C-002 (geometric init) | TinyLlama 1.1B | 88 | **100%** | **16:1** |
| exp3_tinyllama | TinyLlama 1.1B | 88 | **100%** | **39:1** |
| exp3 | Qwen 7B | 112 | **100%** | **67:1** |

### Absent in all non-cybernetic experiments

| Dataset | Model | Bridges | Block-Diagonal | Co/Cross Ratio |
|---------|-------|---------|---------------|----------------|
| exp1 learnable | Qwen 7B | 56 | 0% | 1:1 |
| exp2 FCC | Qwen 7B | 112 | 2% | 1:1 |
| exp2.5 geometric | Qwen 7B | 112 | 0% | 1:1 |
| fingerprint code | Qwen 7B | 112 | 0% | 1:1 |
| fingerprint math | Qwen 7B | 112 | 2% | 1:1 |

### Temporal emergence (27,056 checkpoint bridges across 272 steps)

All three cybernetic experiments lock to 100% block-diagonal by step 200.
Structure emerges identically at both 1.1B and 7B scale.

**Qwen 7B (exp3) — 14,560 bridges, 130 steps:**

| Step | Co-planar | Cross-planar | Ratio | BD% |
|------|-----------|-------------|-------|-----|
| 0 | 0.000000 | 0.000000 | — | 0% |
| 100 | 0.006171 | 0.000442 | 13.9:1 | 99% |
| 200 | 0.022113 | 0.000092 | 239.9:1 | **100%** |
| 500 | 0.078074 | 0.000027 | 2,857:1 | 100% |
| 1000 | 0.149090 | 0.000021 | 6,935:1 | 100% |
| 5000 | 0.678595 | 0.000106 | 6,432:1 | 100% |
| 10000 | 1.315761 | 0.000132 | 9,968:1 | 100% |
| 12900 | 1.514943 | 0.000083 | **18,248:1** | 100% |

**TinyLlama 1.1B (exp3_tinyllama) — 8,888 bridges, 101 steps:**

| Step | Co-planar | Cross-planar | Ratio | BD% |
|------|-----------|-------------|-------|-----|
| 0 | 0.000000 | 0.000000 | — | 0% |
| 100 | 0.012412 | 0.000425 | 29.2:1 | **100%** |
| 1000 | 0.162484 | 0.000027 | 5,972:1 | 100% |
| 5000 | 0.586834 | 0.000080 | 7,352:1 | 100% |
| 10000 | 0.778412 | 0.000021 | **37,929:1** | 100% |

**TinyLlama 1.1B (C-001) — 3,608 bridges, 41 steps:**

| Step | Co-planar | Cross-planar | Ratio | BD% |
|------|-----------|-------------|-------|-----|
| 0 | 0.000000 | 0.000000 | — | 0% |
| 100 | 0.006614 | 0.000690 | 9.6:1 | **100%** |
| 500 | 0.081432 | 0.000027 | 3,033:1 | 100% |
| 4000 | 0.480126 | 0.000088 | **5,438:1** | 100% |

**Key observations:**
- Cross-planar coupling saturates at < 0.001 across all experiments
- Co-planar coupling grows monotonically (linear in Qwen, sub-linear in TinyLlama)
- Peak ratios: 18,248:1 (Qwen 7B), 37,929:1 (TinyLlama exp3), 5,438:1 (C-001 at step 4000, still growing)
- TinyLlama achieves higher ratios because cross-planar is more suppressed (~0.00002 vs ~0.00008)

### Pair identity and axis symmetry

The SAME three co-planar pairs dominate in every experiment:
- **(0,1)** = ±x axis faces
- **(2,3)** = ±y axis faces
- **(4,5)** = ±z axis faces

**100% of top-3 off-diagonal entries** are co-planar in ALL cybernetic bridges.
**0% of top-3 off-diagonal entries** are co-planar in ALL non-cybernetic bridges.
Both identity-init and geometric-init converge to identical pair topology.

**Axis symmetry:** All three axes develop nearly identical coupling strength:

| Experiment | x-axis (0,1) | y-axis (2,3) | z-axis (4,5) | Max deviation |
|-----------|-------------|-------------|-------------|---------------|
| Qwen 7B (exp3) | 1.508 | 1.521 | 1.522 | 0.9% |
| TinyLlama (exp3_tinyllama) | 0.775 | 0.781 | 0.779 | 0.8% |
| C-001 (identity init) | 0.475 | 0.483 | 0.482 | 1.6% |

The RD's three coordinate axes are treated symmetrically — no axis is preferred.

**Coupling polarity:** Signs are ~50/50 positive/negative across layers at all
three axes. The Steersman constrains TOPOLOGY (which channels couple) but leaves
POLARITY to the optimizer. Different transformer layers couple co-planar faces in
different directions, but always with the same magnitude. The block-diagonal
structure constrains the geometry; the optimizer determines the dynamics.

## Magnitude comparison across scale (final state)

| | Co-planar mean | Cross-planar mean | Ratio | Coupling |
|-|---------------|------------------|-------|----------|
| TinyLlama 1.1B (C-001, step 4000) | 0.480 | 0.000088 | 5,438:1 | Moderate |
| TinyLlama 1.1B (exp3_tinyllama, step 10000) | 0.778 | 0.000021 | 37,929:1 | Strong |
| Qwen 7B (exp3, step 12900) | 1.515 | 0.000083 | 18,248:1 | Very strong |
| Qwen 7B (non-cyber, exp1) | 0.009 | 0.008 | 1.1:1 | Negligible |

Larger models develop stronger absolute co-planar coupling (1.515 vs 0.778)
while cross-planar remains uniformly suppressed below 0.001 in all
cybernetic experiments. The three orders of magnitude between co-planar
and cross-planar couplings are maintained across both model scales.

## Mechanism

The Steersman applies three feedback signals:
1. **Contrastive loss** — penalizes bridges that don't separate channels
2. **Spectral loss** — penalizes bridges with low Fiedler eigenvalue
3. **Bridge LR scaling** — modulates bridge learning rate based on feedback

The combination of spectral + contrastive pressure creates a selection
effect: the bridge must BOTH separate and connect. The geometric structure
of the RD (6 faces grouping naturally into 3 axis pairs) provides the
minimum-energy solution — 3 independent coupling blocks that each connect
exactly 2 co-planar faces.

## Implications

### 1. The Steersman is an architectural selector, not a regularizer

It doesn't merely prevent overfitting — it discovers the 3-axis coordinate
structure inherent in the 6-face rhombic dodecahedron geometry. Without
Steersman: generic coupling matrix. With Steersman: geometric object.

### 2. n=6 is justified despite effective DOF = 9

The 6-channel bridge provides the search space (36 parameters) from which
the Steersman selects 9 effective parameters (3 diagonal deviations + 6
co-planar off-diagonal). Starting with n=3 would deny the Steersman the
ability to discover WHICH pairings matter — it would force all-to-all
coupling instead of discovering independent axis blocks.

### 3. n=3 vs n=6 are topologically different

A 3×3 bridge has 6 off-diagonal entries where every channel couples to
every other channel. A 6×6 block-diagonal bridge has 6 active off-diagonal
entries, but channels 0 and 1 couple ONLY to each other, never to 2-5.
The block-diagonal structure enables independent coupling per axis —
different axes can develop different coupling strengths and signs.

### 4. Scale invariance across 3 model families

Block-diagonal structure appears identically at TinyLlama 1.1B and
Qwen 7B (both cybernetic). The 3-axis self-organization is an
architectural property, not a model-capacity artifact.

**Holly Battery (Wan 2.1 14B, video diffusion, NON-cybernetic):**
RhombiLoRA co/cross ratio = 1.071:1 (essentially uniform). Standard
LoRA control = 1.033:1. Delta = +0.038 — negligible. Even at 14B scale
in a completely different modality, without the Steersman the bridge
remains a generic coupling matrix. This confirms the Steersman is the
causal factor, not model scale or domain.

### 5. Channel ablation prediction

H1-H5 experiments (all cybernetic on TinyLlama) will likely show:
- n=3 wins on raw efficiency (fewer params, same val loss)
- n=6 shows block-diagonal (validating the geometry thesis)
- n=8 and n=12 show diminishing returns (extra DOF remain unused)

The correct interpretation: n=3 APPROXIMATES what the Steersman discovers
from n=6. The Steersman reveals the 3-axis structure. An n=3 bridge assumes
it a priori. The scientific value is in the discovery, not the efficiency.

## For Paper 3

This finding should be a central result. The narrative:
1. Standard LoRA adds learnable parameters with no structural constraint
2. RhombiLoRA adds a bridge with geometric initialization
3. The Steersman (cybernetic training) causes the bridge to self-organize
4. The emergent structure matches the RD's 3-axis coordinate geometry
5. This is evidence that the geometry is not arbitrary — the training
   dynamics discover it independently

---

*Analysis based on 27,748 bridges (692 final-state + 27,056 checkpoint)
across 9 experiments, 272 training steps, 2 model scales, 2 init modes,
with and without cybernetic training.*
