# RhombiLoRA Competitive Landscape — Paper 3 Positioning

> **Status:** Updated March 8, 2026 (literature search refresh)
> **Sources:** EMNLP 2024, ICLR 2024-2025, ACL 2024-2025, ICML 2024, NeurIPS 2024

---

## The Unique Position

**RhombiLoRA is the first LoRA variant that places a learnable dense coupling
matrix between A and B while BOTH outer projections also remain learnable.**

The closest prior art (LoRA-XS, EMNLP 2024) uses a similar r×r bridge but
freezes A and B. RhombiLoRA trains A, B, and bridge jointly, enabling the
bridge to serve as an interpretable diagnostic (task fingerprinting, overfitting
detection) rather than just a compression mechanism.

**No existing method uses adapter internal structure as a task fingerprint or
overfitting diagnostic.** This is a genuine gap in the literature.

---

## Critical Prior Art: LoRA-XS

**Banaei et al., "LoRA-XS: Low-Rank Adaptation with Extremely Small Number of
Parameters" (EMNLP 2024, arXiv 2405.17604)**

| Feature | LoRA-XS | RhombiLoRA |
|---------|---------|------------|
| Bridge matrix | r × r (same as rank) | 6 × 6 (fixed, geometry-motivated) |
| Outer projections | **Frozen** (SVD-initialized) | **Learnable** |
| Purpose of bridge | Parameter compression | Cross-rank coupling |
| Total trainable params | r² per layer | A + B + bridge per layer |
| Bridge interpretability | Generic compression | Geometric structure (6-fold RD) |
| Diagnostic use | None | Task fingerprint, overfit detection |

**Differentiation strategy:** LoRA-XS answers "how few parameters can you
train?" RhombiLoRA answers "what does the coupling between A and B encode?"
The questions are complementary. LoRA-XS's frozen outer projections prevent
the bridge from serving as a diagnostic because A and B don't adapt to the
task — only the bridge does. In RhombiLoRA, the bridge captures the coupling
pattern that emerges when A and B jointly learn, which is why it differentiates
by task (Phase 1A early signal: p = 6.66e-10) and potentially by training
phase (Phase 3A).

---

## Methods That Modify the A-to-B Pathway

| Method | Venue | Mechanism | A/B Status | Measures Geometry? | RhombiLoRA Differentiator |
|--------|-------|-----------|------------|-------------------|--------------------------|
| **LoRA-XS** | EMNLP 2024 | r×r learnable matrix between A and B | **Frozen** (SVD) | No | Both A and B learnable; bridge as diagnostic, not compression |
| **LoRA-Mini** | AAAI WS 2025 | Inner/outer decomposition, train inner only | **Frozen** (outer) | No | Same distinction as LoRA-XS — inner-only training |
| **LoRAN** | EMNLP 2024 Findings | Fixed non-linear activation (Sinter) between A and B | Learnable | No | Fixed non-linearity vs learnable coupling |
| **MELoRA** | ACL 2024 | Block-diagonal mini-LoRAs (channel independence) | Learnable | No | Enforces independence; we learn cross-channel coupling |
| **VeRA** | ICLR 2024 | Frozen random A/B, learnable scaling vectors | **Frozen** (random) | No | Diagonal scaling vs dense coupling |

## Methods That Decompose/Restructure the Update

| Method | Venue | Mechanism | RhombiLoRA Differentiator |
|--------|-------|-----------|--------------------------|
| **DoRA** | ICML 2024 (Oral) | Magnitude/direction decomposition of W | Operates on frozen weight, not adapter; **composable** with RhombiLoRA |
| **MoRA** | arXiv 2405.12130 | Replaces A/B with square matrix M | No factored structure at all |
| **LoRMA** | ACL Findings 2025 | Multiplicative: W' = BA × W₀ | Fixed permutation for rank inflation, not learnable coupling |
| **TLoRA** | arXiv 2501.08727 | Learnable transform on W₀ before adaptation | Transform on pretrained weight, not between A and B |
| **Localized LoRA** | arXiv 2506.00236 | Independent low-rank adapters per weight block | Off-diagonal interactions via separate factors, not shared bridge |

## Adapter Merging (Phase 2A Context)

| Method | Venue | Mechanism | Relevance |
|--------|-------|-----------|-----------|
| **Task Arithmetic** | ICLR 2023 | Add/subtract task vectors (full weight) | Baseline for bridge-level merging comparison |
| **TIES-Merging** | NeurIPS 2023 | Trim, elect signs, merge disjoint | Resolves sign conflicts; bridge merging avoids this by construction |
| **AdaMerging** | ICLR 2024 | Learned merge coefficients via entropy | Layer-wise coefficients; bridge offers per-adapter merge point |
| **DARE** | NeurIPS 2024 | Random pruning + rescaling before merge | Sub-weight pruning; bridge offers structured sub-weight decomposition |
| **TSV-Merge** | arXiv 2508.03999, 2025 | SVD of per-layer LoRA, merge via task singular vectors | **Closest to bridge-level merging** — sub-weight decomposition |

**Bridge merging advantage:** Merge at 36 parameters (bridge) while preserving
task-specific A/B projections. No existing method offers this decomposition.

## Adapter Fingerprinting (Phase 1A Context)

| Method | Venue | Mechanism | Relevance |
|--------|-------|-----------|-----------|
| **Learning on LoRAs** | arXiv 2410.04207 | ML on LoRA weight spaces, GL(r)-equivariant | Proves adapter structure encodes task info; uses full weights |
| **Weight-space Backdoor Detection** | arXiv 2602.15195 | 5 spectral metrics from LoRA matrices | 97% accuracy detecting backdoors from spectral features |
| **SpecLoRA** | arXiv 2505.23099 | Task-relevant directions = dominant singular values | Spectral analysis of LoRA for rank allocation |
| **LoRAuter** | arXiv 2601.21795 | Task-indexed adapter routing | Spectral routing has limited signal |

**Bridge fingerprinting advantage:** The bridge is a compact (36-parameter),
interpretable summary of coupling structure. Existing approaches use raw spectral
statistics of full A/B matrices. No prior work uses a structured intermediate
matrix as a fingerprint.

## Overfitting Detection (Phase 3A Context)

**No prior work uses adapter internal metrics to detect overfitting.** The
closest is T-LoRA (arXiv 2507.05964, diffusion-specific) and LoRA Dropout
(arXiv 2404.09610, theoretical analysis using train/val loss, not adapter
structure). This is a genuine gap. Phase 3A would be novel.

## MoE-Based LoRA Routing

| Method | Venue | Mechanism | RhombiLoRA Differentiator |
|--------|-------|-----------|--------------------------|
| **X-LoRA** | arXiv 2402.07148 | Token-level gating across LoRA experts | Cross-adapter routing, not within-adapter coupling |
| **L-MoE** | arXiv 2510.17898 | Lightweight gating over LoRA experts | Same — routing between, not coupling within |
| **LoRA-Mixer** | arXiv 2507.00029 | Serial attention routing of task LoRA experts | Cross-expert coordination |
| **HoRA** | arXiv 2510.04295 | Cross-head coupling via hypernetworks | Coupling across heads, not within adapter |

## Strongest Novelty Claims (Ordered)

1. **Bridge as diagnostic** (fingerprinting + overfit detection) — no prior work
2. **Joint A-B-bridge learning** — LoRA-XS freezes A/B; we train all three
3. **Fixed bridge dimension independent of rank** — 6×6 not r×r
4. **Geometric motivation for bridge structure** — entirely novel in PEFT
5. **Sub-weight merging through bridge decomposition** — novel merger strategy
6. **Composability** — orthogonal to DoRA, AdaLoRA, MoE methods

## Papers That Must Be Cited

### In Related Work (detailed comparison required)
- LoRA-XS (Banaei et al., EMNLP 2024, arXiv 2405.17604) — **closest prior art**
- LoRA-Mini (Singh et al., AAAI WS 2025, arXiv 2411.15804)
- VeRA (Kopiczko et al., ICLR 2024, arXiv 2310.11454)
- MELoRA (Ren et al., ACL 2024, arXiv 2402.17263) — ablation baseline
- DoRA (Liu et al., ICML 2024, arXiv 2402.09353) — composable
- LoRAN (Li et al., EMNLP 2024, arXiv 2509.21870)
- Learning on LoRAs (Putterman et al., 2024, arXiv 2410.04207)

### In Introduction/Background (brief mention)
- Localized LoRA (Barazandeh, 2025, arXiv 2506.00236)
- LoRMA (Bihany et al., ACL Findings 2025, arXiv 2506.07621)
- MoRA (Jiang et al., 2024, arXiv 2405.12130)
- RandLoRA (Albert et al., ICLR 2025, arXiv 2502.00987)
- SHiRA (Bhardwaj et al., NeurIPS 2024, arXiv 2406.13175)
- Weight-space Backdoor Detection (Luo et al., 2026, arXiv 2602.15195)
- TSV-Merge (Marczak et al., 2025, arXiv 2508.03999)

---

*Updated March 8, 2026 after comprehensive literature search.*
