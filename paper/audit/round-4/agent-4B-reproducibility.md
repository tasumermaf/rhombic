# Reproducibility Audit — Paper 3: The Learnable Bridge

**Auditor:** Agent 4B (adversarial reproducibility audit)
**Date:** 2026-03-15
**Scope:** Can an independent researcher reproduce the core claims from the paper alone?

---

## Executive Summary

The paper is **substantially reproducible** for its core claims. Hyperparameters are fully specified in both the main text and Appendix A. The Steersman algorithm is specified with enough precision to reimplement. Hardware is documented. The code is publicly available. The main gaps are: (1) the corpus-coupled initialization depends on proprietary data that cannot be distributed, (2) several Steersman implementation details visible in the code are not in the paper, and (3) the Holly Battery experiment uses a different optimizer (Prodigy) whose hyperparameters are not specified.

**Verdict: 0 CRITICAL, 3 MAJOR, 7 MINOR, 4 POLISH gaps.**

---

## 1. Hyperparameters

**Rating: COMPLETE in appendix, ADEQUATE in main text.**

The appendix (Table A.1 / Tab. `\ref{tab:hyperparams}`) specifies all standard training hyperparameters:

| Parameter | Specified? | Value |
|-----------|-----------|-------|
| Optimizer | Yes | AdamW |
| Weight decay | Yes | 0.01 |
| Learning rate | Yes | 2e-4 |
| LR schedule | Yes | Cosine decay after linear warmup |
| Warmup steps | Yes | 200 |
| Batch size | Yes | 2 |
| Gradient accumulation | Yes | 8 (effective BS = 16) |
| Max gradient norm | Yes | 1.0 |
| LoRA rank | Yes | 24 |
| LoRA alpha | Yes | 16 |
| LoRA dropout | Yes | 0.0 |
| Target modules | Yes | W_Q, W_K, W_V, W_O |
| Dataset | Yes | alpaca-cleaned |
| Max sequence length | Yes | 512 |
| Precision | Yes | bfloat16 |
| Gradient checkpointing | Yes | Enabled |
| Random seed | Yes | 42 |
| Steersman interval | Yes | 100 steps |

**Gaps:**

- **[MINOR] AdamW betas not specified.** The paper says "AdamW" but does not state whether the PyTorch default betas (0.9, 0.999) are used or custom values. Standard defaults are a safe assumption, but explicit specification would be better.

- **[MINOR] Tokenizer padding strategy not specified.** The code shows `tokenizer.pad_token = tokenizer.eos_token` — a common default, but not stated in the paper. For sequence length 512 on alpaca-cleaned, this could affect batch composition.

- **[MAJOR] Holly Battery optimizer mismatch.** The main text mentions the Holly Battery achieves "3.8% lower loss" and uses "Prodigy optimizer" (mentioned in passing on line describing the experiment as "Wan 2.1 14B, Prodigy optimizer, no Steersman"). The Prodigy optimizer has its own distinct hyperparameters (d0, weight decay schedule, etc.) that are nowhere specified. An independent researcher cannot reproduce the Holly Battery experiment without these. The per-experiment table in the appendix lists Holly's steps as "10 epochs" but does not list the optimizer or its parameters.

- **[MINOR] Holly Battery steps as "10 epochs."** All other experiments specify step counts. Holly specifies epochs. The number of steps per epoch (depends on dataset size and batch size for the video diffusion task) is not stated. Furthermore, the Holly Battery uses a video diffusion model (Wan 2.1), which presumably uses a different dataset than alpaca-cleaned — this is never stated.

---

## 2. Steersman Algorithm

**Rating: REPRODUCIBLE with effort. All control laws specified; some implementation details require code reference.**

The paper specifies the three-component loss (Eq. 3):

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + w_c \mathcal{L}_{\text{con}} + w_s \mathcal{L}_{\text{spec}}$$

### Contrastive Loss (Eq. 4)

Fully specified. The formula is clear: mean absolute co-planar coupling minus mean absolute cross-planar coupling, negated, averaged over all adapters. The partition into 3 co-planar and 12 cross-planar pairs is geometrically derived and documented.

### Spectral Loss (Eq. 5)

Fully specified. Mean squared deviation of per-adapter Fiedler eigenvalue from target. The Laplacian construction (absolute off-diagonal values as edge weights) and eigenvalue computation method (`torch.linalg.eigvalsh`) are stated.

### Steersman Control Laws

All three control laws are described in Section 3.2 with the specific threshold values:

| Parameter | Value | Source |
|-----------|-------|--------|
| Initial w_c | 0.1 | Main text + Appendix |
| Max w_c | 0.5 | Main text + Appendix |
| Initial w_s | 0.05 | Main text + Appendix |
| Max w_s | 0.2 | Main text + Appendix |
| Target Fiedler (initial) | 0.1 | Main text + Appendix |
| Fiedler decline threshold | -0.001 | Main text + Appendix |
| Stagnation threshold | 0.02 | Main text + Appendix |
| Deviation growth threshold | 0.05 | Main text + Appendix |
| Min bridge LR multiplier | 0.1 | Main text + Appendix |
| Sliding window | 5 measurements | Appendix only |

**Gaps:**

- **[MINOR] Control law gain constants.** The paper states that when the Fiedler trend is declining, w_s is "increased proportionally." The code (line 333) shows the actual formula is `boost = min(abs(fiedler_trend) * 10.0, max_spectral - current_weight)`. The gain factor of 10.0 is not in the paper. Similarly, the directionality control law increases w_c by `min(0.02, max - current)` per intervention (code line 363) — the 0.02 increment is not specified in the paper. The stability law uses `dampen = max(0.8, 1.0 - deviation_trend)` (code line 390) — the 0.8 floor is not in the paper.

- **[MINOR] Decay dynamics for w_c and w_s.** The paper says w_s "decays slowly toward its base value" when Fiedler trend is positive, and w_c "decays" when ratio exceeds 1.2. The actual decay rates are in the code but not the paper. The paper says w_s decays toward 0.05; the code decays toward `base_spectral * 0.5 = 0.025`.

- **[MINOR] Spectral target adaptation.** The paper says the initial target is 0.1 and is "adapted upward during training to track the observed mean Fiedler value." The tracking rate (0.1 in the appendix) is specified, but the precise update rule (`target += rate * (observed - target)` only when observed > target) is only in the code.

- **[POLISH] Co/cross ratio threshold for directionality law.** The paper says w_c increases when the ratio is "near unity and stagnating (trend magnitude < 0.02)." The code additionally checks `co_cross < 1.1` (line 362), and the ratio threshold for decay is `co_cross > 1.2` (line 372). The 1.1 check is not in the paper; the 1.2 value is mentioned.

**Assessment:** A careful reimplementation from the paper alone would produce qualitatively correct behavior. The specific gain constants, decay rates, and secondary thresholds affect the speed and smoothness of convergence but not the final outcome (the paper's own convergence analysis shows lock-in by step 200 regardless of initialization). The qualitative control logic — boost when metric stagnates, decay when metric is healthy, bound within limits — is fully communicated.

---

## 3. Hardware

**Rating: FULLY SPECIFIED.**

The appendix provides a hardware table mapping each experiment to a specific GPU:

- Local: RTX 6000 Ada 48 GB (C-001/2/3, exp3, exp3_tiny)
- Hermes: RTX 4090 16 GB (H1-H4)
- RunPod: A100 80 GB (Holly)

Wall times are provided per experiment (approximate). This is sufficient.

No gaps.

---

## 4. Model Details

**Rating: ADEQUATE for core experiments; GAP for Holly.**

Models are specified by their Hugging Face identifiers:
- TinyLlama-1.1B-Chat (22 layers)
- Qwen2.5-7B-Instruct (28 layers)
- Wan-2.1-14B (40 layers) — for Holly Battery

The adapter configuration (rank 24, alpha 16, target modules W_Q/W_K/W_V/W_O) is specified. Bridge parameter counts per channel configuration are tabulated.

**Gaps:**

- **[POLISH] Hugging Face model revision/commit not pinned.** The paper references "TinyLlama-1.1B-Chat" and "Qwen2.5-7B-Instruct" but does not pin a specific revision hash. Model weights on HF can be updated. This is standard practice in most ML papers but worth noting.

- **[MAJOR] Holly Battery adapter targets unclear.** The Holly experiment uses Wan 2.1, a video diffusion model. The paper states LoRA targets W_Q/W_K/W_V/W_O across all experiments — but video diffusion transformers have different layer structures than causal LMs. The paper mentions "40 layers" but does not specify which modules within the Wan architecture receive adapters. The alpaca-cleaned dataset is clearly inappropriate for video diffusion fine-tuning, so Holly must use a different dataset (and likely different sequence length, precision, and other settings), but none of this is specified.

---

## 5. Data

**Rating: ADEQUATE for core experiments; GAP for Holly.**

The alpaca-cleaned dataset is specified for all text experiments. Sequence length (512) and tokenization strategy are stated. The dataset is publicly available.

**Gaps:**

- **[MAJOR] Holly Battery training data not specified.** The Holly experiment fine-tunes a video diffusion model. The training data, preprocessing, and objective function (not causal LM loss) must differ from the text experiments, but none of this is documented.

- **[POLISH] alpaca-cleaned split.** The paper does not state whether a train/validation split is used or what the split ratio is. Validation loss is reported, implying a held-out set exists, but the split is not specified. The code likely creates one, but the paper should state it.

---

## 6. Initialization Strategies

**Rating: WELL SPECIFIED for identity and geometric; PARTIALLY SPECIFIED for corpus-coupled.**

### Identity

Fully specified: M = I_6. No ambiguity.

### Geometric

Fully specified: M = I_6 + 0.01 * C_hat, where C_hat is the normalized co-planar coupling matrix from the RD face-sharing geometry. The paper describes the perturbation magnitude (0.01) and the geometric derivation.

### Corpus-Coupled

The paper states: "Off-diagonal entries are set according to I Ching complementary trigram pairings, producing a bridge that initially favors cross-planar coupling (I Ching/RD ratio 1.4:1 at initialization)."

**Gaps:**

- **[MINOR] Corpus-coupled initialization is proprietary.** The code shows that `corpus_coupled_matrix()` depends on `corpus_private.json`, which is not distributed with the public package. The initialization combines hexagram coupling, geometric coupling, and thread density — three multiplicative factors whose derivation requires proprietary corpus values. An independent researcher cannot reproduce the exact C-003 initialization.

  However, this gap is **mitigated** by the paper's own findings: C-003 converges to the same attractor as C-001 (identity) and C-002 (geometric) within 200 steps. The initialization is cosmetic to the final result. An adversarial initialization with cross-planar bias at 1.4:1 could be constructed independently (e.g., random cross-planar perturbation of the same magnitude) without the specific trigram coupling.

- **[POLISH] I Ching trigram-to-channel mapping not specified.** The code maps 6 trigrams to 6 channels (Kun, Zhen, Dui, Qian, Xun, Gen), with coupling computed line-by-line (+1/3 for matching, -1/3 for complementary). The specific mapping and the coupling formula are in the code but not in the paper. Again, since initialization is shown to be cosmetic, this is not critical.

---

## 7. Metrics

**Rating: WELL DEFINED.**

All reported metrics are clearly defined:

- **Co-planar/cross-planar ratio (rho):** Eq. 6 in the paper. Mean absolute co-planar coupling divided by mean absolute cross-planar coupling.

- **Block-diagonal classification:** Two criteria: rho > 10 AND all cross-planar entries < 1e-3. Stated in Section 3.3 and Appendix A.6. The appendix notes that typical convergence values far exceed these thresholds.

- **Bridge Fiedler:** Second-smallest eigenvalue of the weighted Laplacian constructed from absolute off-diagonal bridge entries. Defined in Section 5.2.

- **Correlation Fiedler:** Fiedler eigenvalue of the pairwise Pearson correlation matrix of flattened bridge vectors across layers. Defined in Section 5.2.

- **Spectral gap:** Mentioned in the ablation (Section 4, "spectral gap at n=6 converges to ~0.00006"). Not formally defined in the paper body, but the concept is standard (lambda_3 - lambda_2 for the bridge Laplacian).

- **Validation loss:** Standard causal LM cross-entropy loss on held-out data. Not explicitly defined but universally understood.

- **Co-planar spread / ratio spread / val loss spread:** Defined in Appendix A.7 as maximum pairwise deltas expressed as percentages.

- **BD%:** Percentage of bridges classified as block-diagonal. Clearly defined.

- **Half-life:** Stated as 123 +/- 8 steps from least-squares fit across three experiments. The fitting method (exponential decay model) could be more explicit.

**Gaps:**

None that would impede reproduction. All metrics are computable from the bridge matrices, which are the primary data.

---

## 8. Code Availability

**Rating: GOOD with one significant limitation.**

The paper provides:
- GitHub URL: `https://github.com/promptcrafted/rhombic`
- PyPI package: `https://pypi.org/project/rhombic/` (v0.3.0, 312 tests)
- Reproduction instruction: "install rhombic and run the training scripts with seed=42"

The code repository contains:
- `rhombic/nn/rhombi_lora.py` — the RhombiLoRA architecture
- `rhombic/nn/topology.py` — bridge initialization and geometric derivations
- `scripts/train_cybernetic.py` — the full cybernetic training loop including the Steersman class
- `scripts/train_contrastive_bridge.py` — contrastive loss pair indices
- 312 tests

**Gaps:**

- **[MINOR] Reproduction instructions are minimal.** "Install rhombic and run the training scripts with seed=42" does not specify which script, which arguments, or which model. A dedicated `scripts/reproduce_paper3.sh` or equivalent would help. The training script docstring shows example commands but with different step counts than the paper's experiments.

- The corpus-coupled initialization code depends on `corpus_private.json`, which is proprietary. The code handles this gracefully (raises a clear error if absent), and the paper's findings do not depend on this specific initialization (convergence is initialization-independent). The identity and geometric initializations — which produce the same final result — are fully public.

---

## Gap Summary

| # | Category | Gap | Severity | Impact on Core Claims |
|---|----------|-----|----------|----------------------|
| 1 | Holly optimizer | Prodigy optimizer hyperparameters not specified | MAJOR | Cannot reproduce Holly; Holly is a null result / control, not a core claim |
| 2 | Holly data | Training data for video diffusion not specified | MAJOR | Same as above |
| 3 | Holly modules | Adapter target modules for Wan 2.1 not specified | MAJOR | Same as above |
| 4 | Steersman gains | Proportional gain constants (10.0x, 0.02 increment, 0.8 floor) not in paper | MINOR | Qualitative behavior reproducible; exact trajectory may differ |
| 5 | Steersman decay | w_s/w_c decay rates and base multipliers not in paper | MINOR | Same as #4 |
| 6 | Spectral target | Precise update rule for target adaptation partially specified | MINOR | Appendix gives tracking rate; code gives full rule |
| 7 | AdamW betas | Not explicitly stated | MINOR | Standard defaults assumed |
| 8 | Tokenizer | Padding strategy not stated | MINOR | Standard practice |
| 9 | Corpus init | Requires proprietary data | MINOR | Paper proves initialization is cosmetic; identity/geometric produce same result |
| 10 | Reproduction | No dedicated reproduction script or step-by-step guide | MINOR | Code exists; assembly required |
| 11 | Model revisions | HF model commits not pinned | POLISH | Standard omission |
| 12 | Train/val split | Not specified for alpaca-cleaned | POLISH | Standard practice |
| 13 | I Ching mapping | Trigram-to-channel mapping not in paper | POLISH | Initialization cosmetic |
| 14 | Half-life fit | Exponential decay model not formally specified | POLISH | Standard fitting |

---

## Assessment of Impact on Core Claims

The paper makes six numbered contributions. Here is the reproducibility assessment for each:

1. **Block-diagonal emergence (100% cybernetic, 0% non-cybernetic):** FULLY REPRODUCIBLE. All hyperparameters, loss functions, and metrics specified. The C-001/C-002/H3 experiments use identity/geometric initialization with public code.

2. **Initialization independence:** SUBSTANTIALLY REPRODUCIBLE. Identity and geometric experiments are fully reproducible. The corpus-coupled experiment (C-003) requires proprietary data, but an adversarial initialization with similar cross-planar bias could be independently constructed.

3. **Coupling dynamics (half-life, linear growth):** REPRODUCIBLE. These are measured from the bridge matrices during training, which the code logs at every checkpoint.

4. **Channel count as effective rank:** FULLY REPRODUCIBLE. The ablation (H1-H4) uses identity initialization on TinyLlama with fully specified hyperparameters.

5. **Contrastive loss as mechanism:** FULLY REPRODUCIBLE. Same as #4 — the ablation design is clean and all parameters specified.

6. **Scale consistency (1.1B to 7B):** REPRODUCIBLE. TinyLlama and Qwen 7B experiments are fully specified.

The Holly Battery result (14B null control) is the least reproducible experiment but is also the least critical — it provides additional non-cybernetic control evidence complementing the better-controlled Qwen-family comparison (exp2/exp2.5 vs. exp3).

---

## Recommendations

1. **Add Holly Battery details** to the appendix: Prodigy optimizer configuration, training data, target modules for Wan 2.1, and number of training steps (not just epochs). Even though Holly is a control experiment, under-specification of controls undermines the comparison.

2. **Add a reproduction guide** — either a script or an appendix section with exact command lines for the key experiments (C-001, C-002, H3, and the H1-H4 ablation).

3. **Specify the Steersman gain constants** in the appendix. The control law descriptions in the main text are qualitatively clear but quantitatively incomplete. Adding the gain factor (10x), the w_c increment (0.02), the dampen floor (0.8), and the w_s base decay multiplier (0.5x base) to Appendix A.2 would close this gap with four additional rows in the table.

4. **State the alpaca-cleaned train/val split** explicitly.

5. **Pin model revisions** in the appendix (HuggingFace commit hashes).

---

*Audit completed 2026-03-15. The paper's core experimental claims are reproducible from the information provided, with the Holly Battery experiment being the notable exception. The Steersman's implementation has minor quantitative gaps that would not prevent a competent reimplementation from reaching the same qualitative conclusions.*
