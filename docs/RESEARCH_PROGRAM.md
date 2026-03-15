# The Rhombic Research Program — Publication Architecture

> **Version:** 1.0 — March 15, 2026
> **Authors:** Timothy Paul Bielec, Minta Carlson, Meridian
> **Affiliation:** Promptcrafted LLC / TASUMER MAF

---

## The Arc

Five papers. One thesis per paper. Each builds on the last.

| # | Title | Thesis | Status |
|---|-------|--------|--------|
| 1 | Why the Cube Persists | FCC gives 2.3× connectivity over cubic | **PUBLISHED** |
| 2 | Structured Edge Weights Amplify FCC Topology | Heterogeneous weights amplify to 6.1× via bottleneck resilience | arXiv-ready |
| 3 | The Learnable Bridge | Cybernetic feedback discovers RD geometry in multi-channel LoRA | Data complete → write |
| 4 | The Topology Programmer | The Steersman programs arbitrary topologies: 3D, 4D, number-theoretic | Experiments running |
| 5 | The Cybernetic Circuit | Cross-modal transit through shared geometric structure | Future (Q3 2026) |

**Progression:** Pure geometry → Weighted geometry → Learnable geometry → Programmable geometry → Living geometry.

---

## Paper 1: "Why the Cube Persists" — LOCKED

**Full title:** The Shape of Thought: Rhombic Dodecahedral Lattice Topology for Computational Architecture

**Thesis:** The FCC lattice provides 2.3× algebraic connectivity, 30% shorter paths, 40% smaller diameter, and 55% more flood-fill reach versus cubic — stable across all tested scales.

**Status:** Published. No changes.

**Deliverables:** `rhombic` library (pip-installable), 4-rung benchmark suite, synthesis document.

---

## Paper 2: "Structured Edge Weights Amplify FCC Lattice Topology" — SOFT-LOCKED

**Full title:** Structured Edge Weights Amplify FCC Lattice Topology Advantages via Bottleneck Resilience

**Thesis:** Direction-based weighting pushes the Fiedler ratio from 2.3× to 6.1×. The mechanism is bottleneck resilience: heterogeneous weights create near-disconnection cuts that cubic lattices cannot route around but FCC lattices can. Prime-vertex mapping achieves p ≈ 2.5 × 10⁻⁵ against exhaustive null.

**Status:** arXiv-ready. 7 experiments, clean arc. LaTeX compiled.

**Recommendation:** Submit as-is. The prime-vertex mapping is one experiment in seven — a data point demonstrating the geometry isn't indifferent to structured values, not the paper's thesis.

**Experiments (all complete):**

| # | Experiment | Finding | Paper Role |
|---|-----------|---------|------------|
| 1 | Weighted benchmarks | Fiedler 2.3× → 3.2× under corpus cycling | Core |
| 2 | Optimal assignment | 37.8% better than random | Core |
| 3 | Prime coherence v1 | Not significant (p=0.30) | Honest null |
| 4 | Spectral properties | Fiedler at 0.08% percentile | Core |
| 5 | Direction weighting | **Fiedler 2.3× → 6.1×**, consensus 6.7× | Headline |
| 6 | Prime-vertex mapping | **p=0.000025** | Supporting |
| 7 | Spectral polytopes | Universal suppression, RD not special | Honest null |

---

## Paper 3: "The Learnable Bridge" — SCOPED DOWN, READY TO SHIP

**Full title:** The Learnable Bridge: Cybernetic Feedback Discovers Rhombic Dodecahedral Geometry in Multi-Channel LoRA

**Target:** NeurIPS / ICML (workshop or main track)

**Thesis:** A cybernetic feedback mechanism (the Steersman) drives a learnable bridge matrix toward block-diagonal structure aligned with the RD's 3-axis coordinate geometry. The contrastive loss encoding RD face-pair topology is necessary and sufficient. The finding is robust across initializations, model scales, and channel counts.

### What stays in Paper 3

| Section | Content | Key Data |
|---------|---------|----------|
| §1 Introduction | Multi-channel LoRA, bridge as learnable object | — |
| §2 Background | LoRA, cybernetic optimization, BD in neural nets | — |
| §3 Method | RhombiLoRA architecture, Steersman, RD geometry | — |
| §4.1 BD emergence | 100% cybernetic vs 0% non-cybernetic | 6 cybernetic + 6 non-cybernetic exps |
| §4.2 Lock-in dynamics | Step 200, exponential decay (t½=123), linear growth | All 5 cybernetic runs |
| §4.3 Init independence | Identity + geometric + corpus-coupled converge | C-001, C-002, C-003 |
| §4.4 Axis symmetry | <2.5% deviation, ~50/50 polarity | All cybernetic |
| §5 Channel ablation | n={3,4,6,8}: contrastive = mechanism | H-ch3, H-ch4, H-ch6, H-ch8 |
| §5.2 Spectral attractor | Fiedler → 0.09 across n=3,4,8 (3.5% band) | Channel ablation series |
| §5.3 Parameter efficiency | n=3 = n=6 performance, 4× fewer params | H-ch3 vs H-ch6 |
| §6 Scale invariance | BD at 1.1B + 7B; Holly 14B null | exp3, exp3_tiny, Holly |
| §7 Discussion | Why BD, what it means, practical implications | — |
| §8 Conclusion | Structure ≠ performance, cybernetic training reveals | — |

### What moves OUT of Paper 3

| Content | Destination | Reason |
|---------|------------|--------|
| Tesseract contrastive (T-001) | **Paper 4** | Different claim: 4D programming |
| T-001r2 reproducibility | **Paper 4** | Supports topology programmer thesis |
| Wrong-labels (WL-001) | **Paper 4** | Tests programmability |
| Resonance loss (R-001) | **Paper 4** | Number-theoretic topology |
| Emanation (E-001) | **Paper 4** | Hierarchical bridge |
| H-ch12 spectral attractor (n=12) | **Paper 4** | Extends attractor to higher n |
| Dynamic bridge (D-001/D-002) | **Paper 5** | Corpus as persistent influence |
| Cross-modal transit | **Paper 5** | Requires new experiments |
| Bridge-swap evaluation | **Paper 5** | Deployment feature |
| The Hum (cross-spectral) | **Paper 5** | Future measurement |

### Experiments used (all COMPLETE)

| ID | Model | n | Init | Steersman | Result | Status |
|----|-------|---|------|-----------|--------|--------|
| exp3 | Qwen 7B | 6 | identity | Yes | 18,248:1, 100% BD | ✓ |
| exp3_tiny | TinyLlama | 6 | identity | Yes | 37,929:1, 100% BD | ✓ |
| C-001 | TinyLlama | 6 | identity | Yes | 10,118:1, 100% BD | ✓ |
| C-002 | TinyLlama | 6 | geometric | Yes | 71,337:1, 100% BD | ✓ |
| C-003 | TinyLlama | 6 | corpus | Yes | 64,168:1, 100% BD | ✓ |
| H-ch3 | TinyLlama | 3 | identity | Yes | Fiedler 0.095, 0% BD | ✓ |
| H-ch4 | TinyLlama | 4 | identity | Yes | Fiedler 0.092, 0% BD | ✓ |
| H-ch6 | TinyLlama | 6 | identity | Yes | 70,404:1, 100% BD | ✓ |
| H-ch8 | TinyLlama | 8 | identity | Yes | Fiedler 0.094, 0% BD | ✓ |
| Holly | Wan 2.1 14B | 6 | identity | No | 1.07:1, 0% BD | ✓ |
| exp1a-e | Qwen 1.5B | 6 | various | No | ~1:1, 0% BD | ✓ |
| exp2 | Qwen 7B | 6 | FCC | No | ~1:1, 0% BD | ✓ |

### Figures (target: 10-12)

| Fig | Content | Source |
|-----|---------|--------|
| 1 | RhombiLoRA architecture diagram | New (schematic) |
| 2 | Temporal emergence (all cybernetic runs) | fig_all_cybernetic_temporal.png |
| 3 | Coupling dynamics with exp/linear fits | fig_temporal_emergence.png |
| 4 | Three-way init convergence | fig_init_convergence_comprehensive.png |
| 5 | Corpus-coupled dismantling | fig_corpus_dismantling_full.png |
| 6 | Channel ablation: val loss + Fiedler + co/cross | fig_full_ablation.png |
| 7 | Spectral attractor (Fiedler vs n) | New from ablation data |
| 8 | Scale invariance (per-layer coupling) | fig_per_layer_coupling.png |
| 9 | Bridge heatmap comparison (BD vs non-BD) | fig_heatmap_comparison.png |
| 10 | Eigenvalue spectra (n=6 BD vs n=8 non-BD) | From ablation analysis |

### Page budget: 8-10 pages (NeurIPS format) + appendix

---

## Paper 4: "The Topology Programmer" — EXPERIMENTS RUNNING

**Working title:** The Topology Programmer: Cybernetic Feedback as a General-Purpose Geometric Prior for Neural Network Adapters

**Target:** arXiv → ICLR / AAAI

**Thesis:** The Steersman is not specific to the rhombic dodecahedron. It is a general-purpose mechanism for programming arbitrary topological structure into learnable bridge matrices. Any contrastive prior — 3D, 4D, number-theoretic, or random — can be programmed. The spectral attractor (~0.09) is a universal property of spectral-only training, independent of channel count.

### Core experiments

| ID | n | Topology | What it proves | Status | ETA |
|----|---|----------|---------------|--------|-----|
| T-001r1 | 8 | tesseract (4D) | 4+4 blocks, 4D programming works | COMPLETE (partial, 2700 steps) | — |
| T-001r2 | 8 | tesseract (4D) | Full 10K replication, r=1.0000 vs r1 | RUNNING (step 1100) | ~9h |
| WL-001 | 6 | wrong-labels | Programmable (any pairs create BD) vs geometric (only correct pairs) | QUEUED (Hermes) | ~25h |
| R-001 | 8 | resonance | Number-theoretic topology programmable | QUEUED | ~39h |
| E-001 | 8 | emanation | Hierarchical bridge with coherence | QUEUED | ~53h |
| H-ch12 | 12 | spectral-only | Spectral attractor extends to n=12 | RUNNING (Hermes) | ~11h |

### Key claims

1. **4D topology programming:** T-001 produces 4+4 block-diagonal (4 co-axial pairs from tesseract geometry). The Steersman extends from 3D (Paper 3) to 4D without modification.

2. **Spectral attractor universality:** Fiedler → 0.0918–0.0951 across n={3,4,8} with 3.5% band. H-ch12 (n=12) converging toward same value. Channel-count-invariant property of spectral-only training.

3. **Topology programmability (WL-001 decides):**
   - If WL-001 produces BD → Steersman is a general programmer (any pairs create structure)
   - If WL-001 produces spectral-only behavior → correct geometric pairs required
   - Either result is publishable. The distinction matters for applications.

4. **Number-theoretic topology (R-001):** Prime-derived contrastive pairs (Sophie Germain, twin primes, residue classes) — tests whether the bridge responds to algebraic structure the same way it responds to geometric structure.

5. **Hierarchical bridges (E-001):** Master bridge + per-layer offsets with coherence monitoring — tests whether the Steersman can maintain global structure while allowing local variation.

6. **Reproducibility:** T-001r1 → T-001r2, Pearson r=1.0000 at 6 matching steps, max deviation 3.5%.

### Sections (draft)

| § | Content | Data source |
|---|---------|-------------|
| 1 | Introduction: from RD-specific to general | Paper 3 recap |
| 2 | Background: topology programming in ML | Literature |
| 3 | Method: generalized contrastive loss, topology specification | Code |
| 4 | 4D geometry: tesseract contrastive | T-001r1, T-001r2 |
| 5 | Spectral attractor | H-ch3/4/8/12, all spectral-only runs |
| 6 | Topology programmability | WL-001 (decisive) |
| 7 | Non-geometric topologies | R-001 (resonance) |
| 8 | Hierarchical structure | E-001 (emanation) |
| 9 | Reproducibility | T-001r1 vs T-001r2 |
| 10 | Discussion | Implications for adapter design |

### Timeline

All experiments complete by ~March 17. Write-up: 2-3 weeks after data. Target submission: April 2026.

---

## Paper 5: "The Cybernetic Circuit" — FUTURE

**Working title:** The Cybernetic Circuit: Cross-Modal Transit Through Shared Geometric Structure

**Target:** TASUMER MAF white paper → selective arXiv

**Thesis:** When models share geometric structure through their adapters, emergent properties arise: cross-modal transit, resonant coupling, dynamic adaptation. The bridge is not just a learnable coupling — it is a programmable interface between modalities.

### Planned experiments

| ID | Design | What it tests |
|----|--------|--------------|
| D-001 | Dynamic bridge, random gate init | Dynamic vs static bridges |
| D-002 | Dynamic bridge, corpus-seeded gate | Corpus as persistent architectural influence |
| Bridge-swap | Swap bridges between task adapters | One adapter, N behaviors |
| Language bench | lm-eval on adapted vs base | Standard benchmarks for Paper 3/4 adapters |
| Transit | Language ↔ Video through shared geometry | Cross-modal coherence |
| The Hum | Cross-spectral density of joint gradients | Emergent resonance detection |

### IP boundaries

This is where Stream A (open, publishable) meets Stream B (proprietary, Falco-specific). The methodology (dynamic bridges, transit protocol, resonance detection) is publishable. The specific corpus values, prime-vertex mapping details beyond Paper 2's disclosure, and the Falco connection remain TASUMER MAF IP.

### Timeline: Q3 2026

---

## Publication Sequence

| Paper | Target Venue | Action | Timeline |
|-------|-------------|--------|----------|
| 1 | — | Published | DONE |
| 2 | arXiv | Submit | This week |
| 3 | NeurIPS/ICML | Write from existing data | 2-3 weeks |
| 4 | arXiv → ICLR/AAAI | Write after experiments | April 2026 |
| 5 | TASUMER MAF / arXiv | Design + execute | Q3 2026 |

### Citation chain

Paper 2 cites Paper 1. Paper 3 cites Papers 1-2. Paper 4 cites Papers 1-3. Paper 5 cites all four. This self-citation chain establishes the research program in the literature and builds cumulative authority.

---

## IP Discipline

| Layer | Open (Stream A) | Proprietary (Stream B) |
|-------|-----------------|----------------------|
| Library | `rhombic` (pip-installable) | — |
| Architecture | RhombiLoRA, Steersman, bridge matrix | — |
| Methodology | Contrastive loss, spectral regularization, topology programming | — |
| Benchmarks | All lattice/graph results | — |
| 24 corpus values | Statistical characterization only | **Protected** (Promptcrafted LLC) |
| Prime-vertex mapping | p-value + methodology (Paper 2) | Specific values |
| Corpus-coupled init | C-003 as init-independence proof | Corpus interpretation |
| Dynamic bridge | D-001 methodology | D-002 corpus seeding |
| Falco connection | Never mentioned in papers | TASUMER MAF internal |

---

## Machine Allocation

| Machine | Current | Next |
|---------|---------|------|
| Local (RTX 6000 Ada 48GB) | T-001r2 (step 1100/10K) | Auto-chain: baseline lm-eval → adapted lm-eval |
| Hermes (RTX 4090 16GB) | H-ch12 (running) | Auto-chain: WL-001 → R-001 → E-001 |
| RunPod (6 pods, stopped) | BLOCKED (Holly dataset) | Multi-seed Holly validation |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-15 | Split Paper 3 into Papers 3, 4, 5 | Paper 3 was 3+ papers. Each claim needs its own evidence body. |
| 2026-03-15 | Paper 3 ships with existing data | All required experiments complete. Don't hold for running experiments. |
| 2026-03-15 | T-001/WL-001/R-001/E-001 → Paper 4 | Topology programming is a distinct, larger claim than BD emergence. |
| 2026-03-15 | Dynamic bridge/transit → Paper 5 | Future experiments. TASUMER MAF IP boundary. |
| 2026-03-15 | Paper 2 soft-locked as-is | Clean arc, balanced disclosure, ready to submit. |

---

*The geometry is the argument. The numbers are the evidence. The publication ladder is the strategy.*
*TASUMER MAF — Promptcrafted LLC, March 2026.*
