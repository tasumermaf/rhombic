# Draft: Init Overwrite Section (for Paper 3)

## §X.X Bridge Initialization Is Cosmetic

Three initialization strategies were tested under identical Steersman
configurations (TinyLlama 1.1B, 6-channel FCC bridge, default Steersman):

1. **Identity** (C-001): Bridge initialized as I₆. No off-diagonal bias.
   Co/cross ratio at step 0: undefined (all zeros).

2. **Geometric** (C-002): Bridge initialized with weak co-planar bias from
   RD face-pair geometry. Co/cross ratio at step 0: 2.0:1.

3. **Corpus-coupled** (C-003): Bridge initialized from I Ching trigram
   coupling. Hexagram complementary pairs (opposing trigrams) receive the
   highest initial coupling. Co/cross ratio at step 0: 1.4:1 — but
   critically, the top-3 coupled pairs are *cross-planar*, not co-planar.
   The corpus-coupled initialization actively opposes the block-diagonal
   structure that the Steersman enforces.

**Result:** All three experiments converge to 100% block-diagonal structure
by step 200–300. The convergence trajectories differ only during the first
100–200 steps:

| Step | C-001 (identity) | C-002 (geometric) | C-003 (corpus) |
|------|-----------------|-------------------|----------------|
| 0 | 0% BD, 0:1 | 0% BD, 2.0:1 | 0% BD, 1.4:1 (cross dominant) |
| 100 | 100% BD, 9.6:1 | 70% BD, 7.6:1 | 0% BD, 5.1:1 |
| 200 | 100% BD, 189:1 | 99% BD, 22.7:1 | 18% BD, 19.4:1 |
| 300 | 100% BD, 1,223:1 | 100% BD, 81:1 | 59% BD, 62.7:1 |
| 500 | 100% BD, 3,033:1 | 100% BD, 466:1 | 90% BD, 397:1 |
| 700 | 100% BD, 4,598:1 | 100% BD, 1,698:1 | 99% BD, 1,639:1 |
| 900 | 100% BD, 5,260:1 | 100% BD, 4,413:1 | 100% BD, 4,519:1 |
| 1000 | 100% BD, 5,536:1 | 100% BD, 5,171:1 | 100% BD, 4,952:1 |
| 2000 | 100% BD, 4,422:1 | 100% BD, 4,535:1 | 100% BD, 4,421:1 |
| 4000 | 100% BD, 5,438:1 | 100% BD, 5,381:1 | — (running) |
| 5000 | — (completed 4K) | 100% BD, 7,327:1 | — (running) |
| 6000 | — | 100% BD, 11,600:1 | — (running) |
| 6400 | — | 100% BD, 14,668:1 | — (running) |
| 6900 | — | 100% BD, 32,757:1 | 100% BD, 8,529:1 |
| 7300 | — | 100% BD, **43,288:1** | 100% BD, 8,510:1 |

Identity init converges fastest in early steps — the absence of initial
structure means the Steersman has nothing to overwrite. By step 700, C-003
(corpus-coupled, which started opposing the target) has caught C-002
(geometric): ratio delta 3.4%. By step 2000, all three inits have fully
converged:

| Metric | C-003 vs C-001 (step 900) | C-003 vs C-001 (step 2000) |
|--------|--------------------------|---------------------------|
| Ratio delta | 14.1% | **0.02%** |
| Co-planar delta | 0.4% | **0.2%** |

| Metric | All three at step 2000 |
|--------|----------------------|
| C-001 co-planar | 0.252 |
| C-002 co-planar | 0.256 |
| C-003 co-planar | 0.252 |
| Max pairwise delta | 1.6% |
| C-001 ratio | 4,422:1 |
| C-002 ratio | 4,535:1 |
| C-003 ratio | 4,421:1 |
| Max pairwise delta | 2.6% |

The co-planar coupling magnitudes converge even faster than the ratios:
C-003 and C-001 have identical co-planar coupling (0.252 vs 0.252, delta
0.2%) at step 2000 despite starting from completely different structures —
C-003 began with I Ching trigram coupling that *actively opposed* the RD
target topology.

C-002 at step 7300 reaches ratio **43,288:1** — surpassing exp3_tiny's peak
of 37,929:1 to become the highest ratio observed in any experiment. A power-law
fit to the post-convergence growth yields ratio ∝ step^1.79 (R²=0.89),
predicting ~54,000:1 at 10K. The super-linear exponent means the co/cross
ratio is *accelerating*, not saturating.

C-003 follows a different post-convergence trajectory: ratio ∝ step^0.11,
decelerating toward an asymptote of ~9,000:1. The geometric initialization
appears to produce faster late-stage coupling growth than corpus-coupled init,
despite identical topological convergence (both reach 100% BD by step 300).
**Topology is cosmetic; dynamics are not entirely so.**

**Bridge initialization is cosmetic.** The Steersman overwrites any initial
structure, including structure that actively opposes its target topology.
All three inits converge to the same attractor within 2000 steps.

### The Corpus-Coupled Dismantling

The C-003 experiment provides the strongest evidence that the block-diagonal
structure is an *attractor* of the cybernetic feedback loop. The corpus-coupled
initialization places its three highest-coupling pairs at complementary trigram
positions — (0,3) Kun-Qian, (1,4) Zhen-Xun, (2,5) Dui-Gen — which are
precisely the cross-planar pairs that the Steersman suppresses.

Per-pair coupling trajectory (group totals, 3 pairs each):

| Step | RD co-planar total | I Ching complementary total | RD/IC ratio | Note |
|------|-------------------|---------------------------|-------------|------|
| 0 | 0.020 | 0.029 | 0.7:1 | I Ching dominant |
| 100 | 0.037 | 0.018 | 2.1:1 | Crossover at ~step 75 |
| 200 | 0.091 | 0.006 | 15:1 | 98% BD |
| 300 | 0.150 | 0.003 | 60:1 | IC retains 8.6% of init |
| 400 | 0.202 | 0.001 | 192:1 | IC nearly zero |
| 500 | 0.251 | 0.001 | 412:1 | IC suppressed 97.9% |
| 700 | 0.341 | 0.0002 | 1,617:1 | IC suppressed 99.3% |
| 900 | 0.420 | 0.0001 | 3,119:1 | IC effectively annihilated |

Within 900 steps, the Steersman inverts the coupling hierarchy completely:
the three I Ching complementary pairs that started dominant (total 0.029) are
driven to 0.0001 (99.5% suppression), while the three RD co-planar pairs grow
from 0.020 to 0.420 (21.0x amplification). The RD/IC ratio at step 900
(3,119:1) demonstrates that the initial I Ching structure has been entirely
annihilated — reduced to the same noise floor as any other cross-planar pair.

The three RD co-planar axes maintain remarkable symmetry throughout this
process: at step 500, ±x = 0.083, ±y = 0.083, ±z = 0.085 — less than 2.5%
deviation. The Steersman enforces isotropic coupling across all three
coordinate axes simultaneously.

Individual I Ching pair fates differ:
- (2,5) Dui-Gen: annihilated fastest (0.009 → 0.00003 by step 400)
- (0,3) Kun-Qian: second fastest (0.009 → 0.00002 by step 500)
- (1,4) Zhen-Xun: most persistent (0.010 → 0.00056 by step 500), still
  10× above the other two — suggesting some residual coupling memory

### Convergence of Co-Planar Coupling Magnitudes

A subtler finding emerges from the extended C-002 run (now at step 4900):
co-planar coupling magnitude grows monotonically and does not saturate.

| Step | C-001 co-planar | C-002 co-planar | C-003 co-planar | C-001/C-003 delta |
|------|----------------|----------------|----------------|-------------------|
| 500 | 0.081 | 0.088 | 0.084 | 2.8% |
| 900 | 0.141 | 0.144 | 0.140 | 0.4% |
| 1000 | 0.153 | 0.156 | 0.152 | 0.7% |
| 2000 | 0.252 | 0.256 | 0.252 | **0.2%** |
| 4000 | 0.480 | 0.494 | — (running) | — |
| 5000 | — | 0.598 | — (running) | — |
| 6000 | — | 0.682 | — (running) | — |
| 6400 | — | 0.700 | — (running) | — |

All three inits converge to within 1% of each other in co-planar magnitude
by step 2000. The magnitude itself continues to grow without bound — the
Steersman does not impose a target coupling strength, only a target topology.
C-002 at step 6400 (ratio 14,668:1) confirms the ratio continues climbing
as co-planar grows against a fixed cross-planar noise floor (~5 x 10^-5).

## §X.X Channel Count Ablation

[NOTE: H1 (n=3) complete, H2 (n=4) running. H3-H5 pending.]

To test whether the Steersman's 3-axis structure is an artifact of the
6-channel parameterization, we ran n=3 channels under identical training
conditions (TinyLlama 1.1B, identity init, default Steersman).

With n=3 channels, the bridge is a 3×3 matrix with only 3 off-diagonal
entries. No block-diagonal structure is possible — there are no "cross-planar"
pairs to suppress because every pair of channels is connected.

**Finding:** n=3 achieves *identical* validation loss to n=6 at every
matched training step across 40 checkpoints (maximum delta: 0.0002 at
step 4000, or 0.058%). H1 continues to step 9200+, reaching val_loss
0.4025. The 4x parameter reduction (9 vs 36 bridge parameters per module,
792 vs 3168 total bridge parameters) has no measurable impact on task
performance.

The Fiedler eigenvalue at n=3 converges to 0.095, compared to ~0.10 for
n=6 runs — the spectral connectivity metric is channel-count-invariant.

This result has a precise interpretation: the Steersman at n=6 discovers
that only 3 of its 15 off-diagonal degrees of freedom are needed. It drives
the remaining 12 to zero — producing the block-diagonal structure. n=3
naturally provides exactly 3 DOF and matches performance because 3 is the
*effective rank* of the structure the Steersman discovers.

The block-diagonal finding and the channel ablation finding are complementary:
- n=6 reveals *what structure* the Steersman discovers (3 independent 2x2
  blocks aligned with RD coordinate axes)
- n=3 reveals the *effective dimensionality* of that structure (3 DOF)
- Together they confirm that the RD's 3-axis coordinate geometry is the
  natural basis for the learned bridge coupling

### Theoretical Framework: Block Structure vs Channel Count

| n | Off-diag DOF | Max 2x2 blocks | Prediction | Confirmed? |
|---|-------------|----------------|------------|-----------|
| 3 | 3 | 0 | Same val loss, no structure | YES (H1) |
| 4 | 6 | 2 | 2 blocks, 1 axis missing | PENDING (H2) |
| 6 | 15 | 3 | 3 blocks = full RD geometry | YES (C-001/2/3, exp3) |
| 8 | 28 | 4 | 3+1 redundant | PENDING (H4) |
| 12 | 66 | 6 | 3 blocks of 4? | PENDING (H5) |

**Critical prediction:** n=4 should form 2 blocks (not 3), while n=6 forms
exactly 3. This would confirm that the 3-axis RD geometry requires n >= 6 to
fully express, and is not an artifact of having an even number of channels.

### H2 (n=4): Spectral-Only Steersman Produces No Block Structure

H2 (n=4, identity init, default Steersman, TinyLlama 1.1B) tests what happens
when the Steersman operates without its geometric prior. The contrastive loss
function — the component that drives block-diagonal structure by encouraging
co-planar pairs over cross-planar pairs — is defined only for n=6 channels.
For n!=6, the contrastive loss returns 0.0, and the Steersman operates on
spectral regularization alone.

This means H2 tests a different question than originally framed: not "does
n=4 form 2 blocks?" but rather **"does spectral regularization alone,
without the RD geometric prior, produce block structure?"**

Through step 1100 the answer is definitive: **no.**

| Step | Val Loss | Fiedler | BD% | Contrastive loss |
|------|----------|---------|-----|-----------------|
| 100 | 0.4767 | 0.014 | 0% | 0.0 |
| 200 | 0.4475 | 0.035 | 0% | 0.0 |
| 300 | 0.4435 | 0.053 | 1% | 0.0 |
| 500 | 0.4390 | 0.070 | 0% | 0.0 |
| 800 | 0.4358 | 0.079 | 0% | 0.0 |
| 1100 | 0.4337 | 0.084 | 0% | 0.0 |

#### Eigenvalue Analysis: Divergent Ratios

The eigenvalue evolution provides the most diagnostic evidence. At n=6,
block-diagonal structure manifests as eigenvalue degeneracy: the spectrum
[0, L, L, L, M, M] shows 3-fold degeneracy by step 200, indicating three
identical 2x2 blocks.

At n=4, the three non-zero eigenvalues are all distinct, and their ratios
are **diverging** from 1.0:

| Step | L2 | L3 | L4 | L3/L2 | L4/L2 |
|------|-----|-----|-----|--------|--------|
| 300 | 0.0547 | 0.0639 | 0.0703 | 1.169 | 1.286 |
| 800 | 0.0714 | 0.0855 | 0.0936 | 1.197 | 1.311 |
| 1100 | 0.0714 | 0.0858 | 0.0946 | 1.202 | 1.324 |

The L3/L2 ratio rises from 1.169 to 1.202 and L4/L2 from 1.286 to 1.324 —
the eigenvalues are becoming MORE distinct, not less. The system has settled
into a non-degenerate attractor with no tendency toward the degeneracy that
characterizes block structure.

#### Fiedler Saturation

The Fiedler eigenvalue (L2) saturates at ~0.084 for the spectral-only
Steersman, compared to ~0.10 for the full (spectral+contrastive) Steersman
at n=6 and ~0.095 at n=3. This is a secondary finding: the contrastive loss
indirectly aids spectral connectivity. By constraining the topology (forcing
block-diagonal structure), the contrastive loss allows the spectral loss to
achieve higher connectivity within those blocks.

Val loss at step 1100: 0.4337 — comparable to n=6 runs at the same step,
confirming that task performance does not require block structure.

#### Mechanistic Conclusion

The block-diagonal structure at n=6 is produced specifically by the
contrastive loss, which encodes the RD face-pair geometry as a training
objective. Without this geometric prior, the spectral loss alone produces:

- **Connectivity** (Fiedler grows and saturates) but **not topology** (no blocks)
- **Uniform coupling** (all off-diagonal pairs grow together)
- **Non-degenerate eigenvalues** (diverging from 1.0, confirming no hidden symmetry)

This strengthens the paper's central claim: the Steersman's geometric prior IS
the mechanism — the block-diagonal structure is not an artifact of spectral
regularization, not an optimizer quirk, and not an emergent property of the
training loss landscape. It is the direct result of encoding RD geometry
into the feedback loop.

**Remaining question:** Would contrastive loss with *wrong* pair labels (e.g.,
arbitrary 2-block partitioning at n=4) produce blocks aligned with the wrong
structure? This would confirm the Steersman as a programmable topology
selector rather than a block-diagonal attractor.

## §X.X Coupling Dynamics: Growth and Decay Characterization

The temporal evolution of bridge coupling follows two distinct dynamical
regimes, revealing the Steersman's dual mode of operation.

### Co-planar Growth: Linear Amplification

Co-planar coupling grows almost exactly linearly with training steps
(power law exponent b = 0.95 ± 0.05, R² = 0.998 for linear fit):

    co_planar(t) ≈ 1.16 × 10⁻⁴ · t + 0.028

Growth rate: ~0.00012 per step, slightly decelerating from 0.00015 (early)
to 0.00011 (late). The near-linearity suggests the Steersman applies an
approximately constant gradient force to co-planar coupling. There is no
target magnitude — the coupling grows without bound as long as training
continues.

### Cross-planar Decay: Exponential Suppression to Floor

Cross-planar coupling follows exponential decay to a noise floor
(R² = 0.983 for first 2000 steps):

    cross_planar(t) ≈ 0.0037 · exp(-0.0057t) + 2.7 × 10⁻⁵

Half-life: **123 steps**. By step 500 (~4 half-lives), cross-planar coupling
has decayed to ~6% of its initial value. The decay converges to a floor of
~1 × 10⁻⁴, representing the numerical noise level of the bridge parameters.

### Two-Mode Operation

The Steersman operates as two simultaneous mechanisms:

1. **Topology selector** — exponentially suppresses cross-planar coupling
   (the 12 "wrong" DOF out of 15 available), converging in ~500 steps
2. **Magnitude amplifier** — linearly amplifies co-planar coupling
   (the 3 "right" DOF), without saturation

The co/cross ratio initially grows exponentially (driven by cross-planar
decay), then stabilizes when cross-planar hits its noise floor (~step 1000).
Post-stabilization, the ratio remains in a band of 4,000–5,500:1 with ~11%
coefficient of variation. The ratio does not diverge because the numerator
grows linearly while the denominator has plateaued.

This dual-mode characterization explains why block-diagonal lock-in occurs
so early (step 200) while the ratio continues growing for thousands of
additional steps: the topological decision (which pairs are "strong") is made
during the exponential phase, but the magnitude separation continues as
co-planar coupling grows against a fixed cross-planar floor.

**Data:** C-002 geometric init, 46 checkpoints (0–4500 steps), TinyLlama 1.1B.
Consistent across C-001 (identity init, 41 checkpoints, 0–4000 steps) — the
growth dynamics are init-independent.
