# Paper 3: The Learnable Bridge — Outline (SCOPED v2.0)

**Title:** The Learnable Bridge: Cybernetic Feedback Discovers Rhombic
Dodecahedral Geometry in Multi-Channel LoRA

**Target:** NeurIPS / ICML (workshop or main track), arXiv cs.LG

**Scope:** This paper makes ONE claim: cybernetic feedback (the Steersman)
discovers block-diagonal structure aligned with RD coordinate geometry.
The contrastive loss encoding face-pair topology is necessary and sufficient.
All data is in hand. Topology programming beyond RD (tesseract, wrong-labels,
resonance, emanation) moves to Paper 4. Cross-modal transit and dynamic
bridges move to Paper 5.

**Page target:** 8-10 pages NeurIPS format + appendix

---

## Abstract

Multi-channel LoRA partitions a low-rank adapter's bottleneck into n parallel
channels coupled through a learnable n × n bridge matrix. We introduce the
Steersman, a cybernetic feedback mechanism encoding a geometric prior based
on the rhombic dodecahedron's 3-axis coordinate system. Across 12 experiments
spanning 1.1B–14B parameters, 3 model architectures, and 3 initialization
strategies, the Steersman at n=6 produces 100% block-diagonal bridge
structure — three independent 2×2 blocks — while non-cybernetic training
produces 0%. Block-diagonal lock-in occurs within 200 steps and is
initialization-independent (including initializations that actively oppose the
target topology). Channel ablation across n ∈ {3, 4, 6, 8} isolates the
mechanism: the contrastive loss is necessary and sufficient. Spectral-only
training drives connectivity (Fiedler → 0.09) but not topology (0% BD).
Task performance is orthogonal to bridge structure (0.17% max val loss delta).
The effective dimensionality is 3: n=3 matches n=6 performance with 4× fewer
bridge parameters.

## 1. Introduction

- Multi-channel LoRA: partitioning rank into coupled channels
- The bridge matrix as a new, compact, interpretable object of study
- Paper 1–2 recap: FCC topology advantages, amplification under weights
- This paper's question: can the bridge DISCOVER geometric structure?
- Contributions (5):
  1. Block-diagonal emergence under cybernetic training (100% vs 0%)
  2. Initialization independence (cosmetic init — 3 strategies converge)
  3. Coupling dynamics (exponential decay + linear growth, t½ = 123 steps)
  4. Contrastive loss as necessary and sufficient mechanism
  5. Effective dimensionality = 3 (n=3 = n=6 in performance)

## 2. Background and Related Work

- LoRA (Hu et al., 2021) and multi-rank extensions
- Multi-head / multi-channel PEFT
- Cybernetic optimization: second-order feedback, Ashby's requisite variety
- Block-diagonal structure in neural networks
- The rhombic dodecahedron in computational geometry (Papers 1-2)

## 3. Method

### 3.1 RhombiLoRA Architecture
- n-channel bridge matrix B (n × n, learnable, per-module)
- Channel size = rank / n_channels
- Forward pass: y = B @ lora_A(x) reshaped through bridge → lora_B

### 3.2 The Steersman
- Three components: contrastive loss, spectral loss, adaptive LR scaling
- Feedback interval: 100 steps
- Three control laws: connectivity trending, directionality, stability

### 3.3 RD Geometry
- 12 faces, 3 coordinate axes, 6 face pairs
- Co-planar (3 pairs) vs cross-planar (12 pairs)
- Contrastive loss: maximize co-planar coupling, minimize cross-planar

## 4. Block-Diagonal Emergence

### 4.1 Core Finding
- Table: 12 experiments (6 cybernetic n=6, 6 non-cybernetic)
- 100% BD, 42,500+ bridges, 2 model scales (cybernetic)
- 0% BD, 570 bridges, 3 model scales (non-cybernetic)
- **Figure 1:** Architecture diagram
- **Figure 2:** Temporal emergence (all cybernetic runs overlaid)

### 4.2 Lock-in Dynamics
- BD by step 200 across all experiments
- Cross-planar: exponential decay, t½ = 123 steps
- Co-planar: linear growth, ~0.00012/step
- Three phases: selection (0-200), growth (200-1000), plateau (1000+)
- **Figure 3:** Coupling dynamics with exponential/linear fits

### 4.3 Initialization Independence
- Identity (no bias), geometric (weak co-planar), corpus-coupled (opposes target)
- Corpus-coupled: I Ching complementary suppressed 99.5% in 900 steps
- All three converge: ratio band 64,168:1–71,337:1, val loss 0.12% delta
- **Figure 4:** Three-way convergence
- **Figure 5:** Corpus-coupled dismantling (per-pair trajectory)

### 4.4 Axis Symmetry
- <2.5% deviation between 3 co-planar axes
- ~50/50 positive/negative coupling signs
- Topology constrained, dynamics free

## 5. Channel Ablation

### 5.1 Contrastive Loss Is the Mechanism
- n=6 (contrastive active): Bridge Fiedler 0.00009, co/cross 70,404:1
- n={3,4,8} (contrastive disabled): Bridge Fiedler ~0.09, co/cross ~1:1
- 1,020× bifurcation in spectral connectivity
- Val loss identical: 0.4015–0.4022 (0.17% max delta)
- **Figure 6:** Definitive ablation (val loss + Fiedler + co/cross, 4 panels)

### 5.2 Spectral Attractor
- Fiedler converges to 0.0918–0.0951 across n={3,4,8}: 3.5% band
- Channel-count-invariant property of spectral-only training
- The spectral loss creates a universal connectivity target

### 5.3 Effective Dimensionality
- n=3 matches n=6 val loss (max delta 0.058%)
- n=3 uses 4× fewer bridge params (9 vs 36 per module)
- 3 DOF from 15 available: effective dimensionality = 3 = number of RD axes

## 6. Scale Invariance

### 6.1 Cross-Scale Results
- TinyLlama 1.1B: 100% BD, ratios up to 82,854:1
- Qwen 7B: 100% BD, ratios up to 22,477:1
- Holly Battery (Wan 2.1 14B, non-cybernetic): 0% BD, 1.07:1
- **Figure 7:** Per-layer coupling comparison (3 scales)

### 6.2 Two Fiedler Metrics
- Bridge Fiedler (within-matrix): near-zero for BD, ~0.09 for non-BD
- Correlation Fiedler (cross-layer): ~0.10 scale-invariant
- **Figure 8:** Bridge heatmap comparison (BD vs non-BD)

### 6.3 Holly Null Result
- 14B video diffusion, 3.8% better loss, 50% smaller checkpoint
- 0% BD without Steersman → Steersman is causal, not scale/architecture

## 7. Discussion

### 7.1 Why Block-Diagonal?
- BD as sharp attractor in loss landscape
- 3D structure in 15D parameter space
- Exponential suppression vs linear growth → basin of attraction

### 7.2 What Does the Structure Mean?
- 3 independent coupling axes per module
- Positive/negative polarity → inhibition/excitation per axis
- Connection to attention head specialization

### 7.3 Practical Implications
- n=3 for efficiency, n=6 for interpretability
- Cybernetic training as a general approach to discovering minimal structure
- Bridge as a structural fingerprint (prior work: 84.5% task classification)

### 7.4 Limitations
- Single task family (instruction-following)
- Single bridge architecture (dense square)
- No formal null model for BD emergence probability
- Contrastive loss defined only for n=6 in this paper (generalization → Paper 4)

## 8. Conclusion

The learnable bridge, under cybernetic feedback, discovers the 3-axis
coordinate geometry of the rhombic dodecahedron. The structure is not
imposed — it emerges from a feedback loop monitoring spectral connectivity.
Initialization is cosmetic. Scale is irrelevant. Task performance is
orthogonal. The contrastive loss encoding face-pair geometry is necessary
and sufficient. Multi-channel LoRA has structural preferences that standard
training does not surface; cybernetic training reveals them.

A companion paper (in preparation) extends these findings to arbitrary
topologies — 4D, number-theoretic, and random — establishing the Steersman
as a general-purpose topology programmer.

## Appendices

### A. Experimental Details
- Full hyperparameter tables for all 12 experiments
- Training curves

### B. Pre-Registered Predictions
- Channel ablation decision tree (written before results)

### C. Mathematical Characterization
- Coupling dynamics fit parameters
- Block detection algorithm
- Fiedler eigenvalue computation details

## Figures (10 total)

| Fig | Content | Status | Source File |
|-----|---------|--------|-------------|
| 1 | RhombiLoRA architecture + Steersman diagram | NEW (schematic) | — |
| 2 | Temporal emergence (all cybernetic runs) | DONE | fig_all_cybernetic_temporal.png |
| 3 | Coupling dynamics with exp/linear fits | DONE | fig_temporal_emergence.png |
| 4 | Three-way init convergence | DONE | fig_init_convergence_comprehensive.png |
| 5 | Corpus-coupled dismantling | DONE | fig_corpus_dismantling_full.png |
| 6 | Channel ablation (val loss + Fiedler + co/cross) | DONE | fig_full_ablation.png |
| 7 | Per-layer coupling (3 scales) | DONE | fig_per_layer_coupling.png |
| 8 | Bridge heatmap (BD vs non-BD) | DONE | fig_heatmap_comparison.png |
| 9 | Eigenvalue spectra (n=6 BD vs n=4/8 non-BD) | DONE | channel-ablation/fig_h2_eigenvalue_divergence.png |
| 10 | Per-module coupling (q/k/v/o) | DONE | fig_per_module_coupling_c002.png |

**9/10 figures already generated.** Only Fig 1 (architecture schematic) needs creation.

---

*Scoped March 15, 2026. Previous outline attempted to hold 3+ papers of content.
This version ships with existing data. Tesseract, wrong-labels, resonance,
emanation → Paper 4. Dynamic bridge, transit → Paper 5.*
