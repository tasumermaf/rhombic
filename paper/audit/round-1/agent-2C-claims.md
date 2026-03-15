# Round 2C: Claim-Evidence Alignment Audit — Paper 3

Auditor: Opus 4.6 (adversarial claims agent)
Date: 2026-03-15
Paper: `rhombic-paper3.tex` (The Learnable Bridge)
Scope: Causal claims, universality claims, negative claims, statistical claims, and deferred F-01-13 investigation.

---

## F-02C-01: "Necessary and Sufficient" Overstates Ablation Evidence

- **Severity:** MAJOR
- **Locations:** Abstract (line 63), Introduction (lines 148, 177, 223), Conclusion (lines 1131-1132)
- **Claim:** "The contrastive loss encoding RD face-pair geometry is necessary and sufficient" for block-diagonal structure.
- **Evidence presented:** n=6 (contrastive active) produces 100% BD; n=3,4,8 (contrastive disabled) produce 0% BD.
- **Problem — sufficiency:** The contrastive loss was tested only at n=6. "Sufficient" would require demonstrating that the contrastive loss, when applied to ANY valid channel count with appropriate pair labels, produces block structure. The paper itself acknowledges this in the Limitations section (lines 1066-1071): "The ablation isolates the contrastive loss as the mechanism but does not test whether an n=4 or n=8 contrastive loss (using a different geometric prior) would produce block structure at those channel counts." This directly contradicts the "sufficient" claim. The wrong-labels experiment (WL-001) that would test programmability is queued but not run (per EXPERIMENT_QUEUE.md).
- **Problem — necessity:** The claim is better supported. Removing the contrastive loss (at n=3,4,8) eliminates BD. But these experiments also change the channel count simultaneously. The confound is acknowledged in the Limitations but not in the Abstract/Introduction/Conclusion where the claim appears.
- **Resolution:** Downgrade to "the contrastive loss is the mechanism that produces block-diagonal structure at n=6" or "is necessary and sufficient at n=6." The universal quantifier is not earned. Alternatively, run contrastive loss at n=4 or n=8 with appropriate pair definitions.

---

## F-02C-02: "Universal" Applied to 6 Experiments, 2 Scales, 1 Task

- **Severity:** MODERATE
- **Locations:** Introduction (line 144, "Block-diagonal emergence is universal"), Contribution list (line 209, "universal property of cybernetic multi-channel LoRA training")
- **Claim:** Block-diagonal emergence is a "universal property."
- **Evidence:** 6 cybernetic experiments, all at n=6, 2 model scales (1.1B, 7B), 1 task family (instruction following on alpaca-cleaned), 1 optimizer (AdamW), 1 dataset.
- **Problem:** "Universal" in scientific usage means "holds under all conditions." The paper tests under highly constrained conditions: single task, single dataset, single optimizer family, single rank (24), and crucially only n=6 for the contrastive case. The Limitations section (line 1057) acknowledges "single task family" as a limitation. "Universal" is too strong for 6 experiments in a narrow experimental space.
- **Resolution:** Replace "universal" with "robust" or "consistent across all tested configurations." The finding IS strong — 100% across 6 experiments with 42,500+ bridges — but "universal" implies a scope the evidence does not cover.

---

## F-02C-03: "The Steersman Is the Causal Factor" — Confound Not Fully Isolated

- **Severity:** MODERATE
- **Location:** Line 998
- **Claim:** "The Holly result rules out model scale, dataset, and optimizer as confounders. The Steersman is the causal factor."
- **Evidence:** Holly Battery (Wan 2.1, 14B, no Steersman) produces 0% BD. Cybernetic experiments produce 100%.
- **Problem:** Holly Battery differs from the cybernetic experiments in MULTIPLE dimensions: model architecture (video diffusion vs. text LLM), task (video generation vs. instruction following), dataset, optimizer (Prodigy vs. AdamW), AND absence of Steersman. The paper claims Holly "rules out model scale, dataset, and optimizer" — but it would need to vary these factors independently while holding others constant to make that claim. Holly actually introduces all these confounds simultaneously. A proper causal claim would require: same model + same task + same optimizer + Steersman on/off. The TinyLlama experiments (e.g., exp3_tiny with Steersman vs. a hypothetical TinyLlama-without-Steersman at n=6) would be the right control.
- **Partial mitigation:** The non-cybernetic experiments exp1a-e, exp2, exp2.5 (all without Steersman) DO use the same Qwen family and instruction following, providing a more controlled comparison to exp3 (Qwen 7B with Steersman). But the text attributes the causal ruling-out specifically to Holly, which is the weakest comparison due to multiple simultaneous differences.
- **Resolution:** Reframe Holly as "additional evidence consistent with the Steersman being the causal factor" rather than claiming Holly alone rules out confounders. The strongest causal evidence comes from the Qwen exp2/exp2.5 (no Steersman) vs. exp3 (Steersman) comparison, where model family, task, and dataset are held constant.

---

## F-02C-04: "Block-Diagonal Structure Requires the Geometric Prior" — Channel/Prior Confound

- **Severity:** MODERATE
- **Location:** Section 4.2 header (line 827), repeated in body
- **Claim:** "Block-Diagonal Structure Requires the Geometric Prior"
- **Evidence:** n=6 with contrastive → 100% BD; n=3,4,8 without contrastive → 0% BD.
- **Problem:** The experimental design confounds channel count with contrastive loss availability. The contrastive loss is defined only for n=6. For n!=6, it returns zero. So the comparison is not "n=6 with prior" vs. "n=6 without prior" — it is "n=6 with prior" vs. "different n without prior." The paper acknowledges this confound in Limitations (lines 1066-1071) but states the section header as if the confound does not exist.
- **What would resolve it:** Running n=6 with spectral loss only (contrastive disabled) while keeping n=6. This experiment appears to not have been run.
- **Resolution:** The header claim should be softened to "Block-Diagonal Structure Emerges Only When the Geometric Prior Is Active" or the n=6-spectral-only control should be added.

---

## F-02C-05: "Contrastive Topology IS the Structure Signal" — Circular Reasoning Risk

- **Severity:** MODERATE
- **Location:** DRAFT_INIT_OVERWRITE_SECTION.md line 264 (source), reflected in paper's Discussion §6.1 and throughout
- **Claim:** The contrastive loss is the sole mechanism producing block-diagonal structure, and block-diagonal structure is entirely determined by the geometric prior.
- **Evidence:** When contrastive loss is active (n=6), BD emerges. When inactive (n!=6), BD does not emerge.
- **Problem:** There is a subtle circularity. The contrastive loss is DEFINED as: reward co-planar coupling, penalize cross-planar coupling. Block-diagonal structure is DEFINED as: high co-planar coupling, low cross-planar coupling. The loss function directly optimizes for the metric used to measure the outcome. This is not an empirical discovery so much as a demonstration that a loss function achieves what it optimizes for. The paper frames this as "the bridge discovers the geometry," but the geometry was injected via the loss function. The genuinely interesting finding is not that BD emerges (that is nearly tautological given the loss), but rather: (a) lock-in speed, (b) initialization independence, (c) the 3-DOF effective rank, and (d) task performance orthogonality. The paper should distinguish between the tautological component (contrastive loss produces what it optimizes) and the non-trivial components.
- **Resolution:** Acknowledge the relationship between the loss function and the measured outcome. Reframe the contribution: the finding is not that the contrastive loss produces BD (that is by design) but that BD is an attractor (robust to initialization), fast (200 steps), does not harm task performance, and reveals effective dimensionality. Some of this framing already exists but is inconsistent — the paper sometimes presents BD emergence as a discovery and sometimes as expected.

---

## F-02C-06: "Scale-Invariant" Based on 2 Cybernetic Scales

- **Severity:** MINOR
- **Locations:** Abstract (line 197), Contribution (line 228), Section heading (line 919)
- **Claim:** "The finding is scale-invariant."
- **Evidence:** TinyLlama 1.1B and Qwen 7B both show BD emergence under cybernetic training. Holly 14B (non-cybernetic) shows no BD.
- **Problem:** Only 2 model scales were tested with cybernetic training (1.1B, 7B). "Invariant" implies tested and confirmed across a continuum. Two data points cannot establish invariance. The 14B result is non-cybernetic only, as the Limitations section acknowledges (lines 1078-1081). The Correlation Fiedler convergence to ~0.10 at both scales is suggestive but not definitive.
- **Additionally:** The Qwen 7B run (exp3) used "an earlier code version with less aggressive contrastive weighting" (line 928). This means the two cybernetic scale points were not run under identical conditions, introducing a confound into the scale comparison.
- **Resolution:** "Scale-consistent across 1.1B–7B" is defensible. "Scale-invariant" is not, especially given the code-version difference. The paper should flag the Qwen exp3 code difference as a limitation of the scale comparison, not just of the ratio comparison.

---

## F-02C-07: "Task Performance Is Independent of Channel Count" — Limited Range

- **Severity:** MINOR
- **Location:** Section header (line 818), Abstract (line 71)
- **Claim:** Task performance is independent of channel count / orthogonal to bridge topology.
- **Evidence:** Validation loss 0.4010-0.4022 across n=3,4,6 (0.17% max delta). n=3 vs n=6 delta max 0.058%.
- **Problem:** Four channel counts (3, 4, 6, 8) were tested, all at rank 24. "Independent" is a strong claim. The claim is better stated as "insensitive to channel count across n in {3,4,6,8}" — the range is narrow. Very large n (where channel size s=r/n becomes very small) might behave differently. The claim is also made at a single rank (24) and single task. H4 (n=8) val loss is listed as "---" in Table 1 (line 604), meaning the data may be incomplete.
- **Partial mitigation:** 0.17% delta across 4 channel counts IS strong evidence of insensitivity within this range.
- **Resolution:** Minor wording adjustment: "insensitive to channel count across n in {3,4,6,8}" rather than "independent." The H4 missing val loss should be noted or filled.

---

## F-02C-08: "The Attractor Is Absolute" — Unfalsifiable as Stated

- **Severity:** MINOR
- **Location:** Line 616
- **Claim:** "The attractor is absolute."
- **Evidence:** 100% BD rate across all cybernetic n=6 experiments from step 200 onward.
- **Problem:** "Absolute" means "without exception or qualification." This is a claim about all possible cybernetic n=6 experiments, not just the ones run. Six experiments with consistent results is strong but finite evidence. A different learning rate, dataset, model architecture, or rank could potentially destabilize the attractor. The paper would need to demonstrate robustness across a much wider hyperparameter sweep to claim absoluteness.
- **Resolution:** "The attractor is robust across all tested conditions" captures the evidence without overreach.

---

## F-02C-09: Absence of Evidence Treated as Evidence of Absence (Non-Cybernetic = 0%)

- **Severity:** LOW
- **Location:** Throughout (lines 57, 146-147, 577, 618-619)
- **Claim:** Non-cybernetic training produces 0% block-diagonal structure.
- **Evidence:** 6 non-cybernetic experiments, 570 final-state bridges, none BD.
- **Problem:** The non-cybernetic experiments were run at different hyperparameters, different code versions, and in some cases different architectures. The claim is empirically well-supported for the specific configurations tested. However, it is conceivable that non-cybernetic training with certain hyperparameter choices (e.g., very high weight decay differentially applied to cross-planar entries, or specific initialization strategies) could produce block-diagonal structure accidentally. The paper presents the 0% as definitive rather than empirical.
- **Mitigation:** This is largely a wording issue. The evidence is strong. But the paper's framing ("produces 0%") could be read as a universal negative claim.
- **Resolution:** No action strictly required, but consider "produced 0% across all non-cybernetic configurations tested."

---

## F-02C-10: "Effective Dimensionality Is Exactly 3" — Measurement Precision

- **Severity:** LOW
- **Location:** Abstract (line 68), Section 4.4 (line 896), Conclusion (line 1135)
- **Claim:** "The effective dimensionality of the discovered structure is exactly 3."
- **Evidence:** At n=6, 12 of 15 off-diagonal entries are driven to ~0 (order 10^-5), leaving 3 active entries. n=3 (with 3 off-diagonal pairs) matches performance.
- **Problem:** "Exactly 3" is stated as a mathematical fact, but it is an empirical observation. The 12 suppressed entries are at ~10^-5, not literally zero. The effective dimensionality is approximately 3. The n=3 performance match supports this but does not establish mathematical exactness — n=2 was not tested (n=2 would have only 1 off-diagonal pair, testing whether the effective dimensionality might be even lower).
- **Resolution:** Minor: "effectively 3" or "converges to 3" rather than "exactly 3." The finding is strong; the wording is slightly stronger than the evidence.

---

## F-02C-11: Statistical Claims — No Formal Tests Reported

- **Severity:** MODERATE
- **Locations:** Throughout
- **Claim:** Various comparative claims (performance "within 0.17%", "identical validation loss", ratios "converge to a narrow band", etc.)
- **Evidence:** Point estimates reported without confidence intervals, standard errors, or formal statistical tests.
- **Problem:** The paper reports no p-values for any comparison except the inherited p=0.474 from Paper 2 (line 119). Key claims that would benefit from formal testing:
  - "Task performance is independent of channel count" — ANOVA or equivalent across n=3,4,6,8
  - "Initialization is cosmetic" — formal convergence test on final-state distributions
  - "Scale-invariant" — formal comparison of structure metrics across scales
  - The 123±8 step half-life (line 657) reports uncertainty but no derivation method
- **Mitigation:** The effect sizes are enormous (70,000:1 vs 1:1 ratios, 100% vs 0% BD rates), making formal statistical testing arguably unnecessary for the primary findings. The performance comparisons (0.17% delta) are the claims most in need of formal testing.
- **Resolution:** For the primary structural claims (BD/non-BD), the binary classification with N=42,500+ is self-evidently significant. For performance comparisons, add confidence intervals or at minimum note the number of independent measurements contributing to each val loss estimate.

---

## F-02C-12: k > v > q > o Hierarchy Claimed as "Architectural Invariant"

- **Severity:** LOW
- **Location:** Line 937
- **Claim:** "The ordering k > v > q > o is consistent across scales, suggesting an architectural invariant."
- **Evidence:** Two scales (TinyLlama, Qwen 7B) show the same module-type ordering.
- **Problem:** Two data points establish consistency, not invariance. "Invariant" implies a general property of transformer architectures. Different attention mechanisms (e.g., grouped-query attention, multi-query attention) could plausibly alter this ordering.
- **Resolution:** "Consistent across the two scales tested" is accurate. "Suggesting an architectural invariant" is appropriately hedged ("suggesting") but should note N=2.

---

## F-02C-13: Resolution of Deferred Finding F-01-13 (IC/RD Initial Ratio 2.9:1)

- **Severity:** MODERATE (elevated from MINOR per F-01-12/F-01-13)
- **Location:** Paper lines 718-720
- **Claim:** "At step 0, I Ching complementary pairs have coupling magnitude 0.029, while RD co-planar pairs start at 0.010—a 2.9:1 disadvantage."
- **Source data (DRAFT_INIT_OVERWRITE_SECTION.md):**
  - Line 95: Header reads "Per-pair coupling trajectory (**group totals**, 3 pairs each)"
  - Line 99: Step 0 row: RD co-planar total = 0.020, I Ching complementary total = 0.029
  - Lines 121-123: Individual IC pairs: (2,5)=0.009, (0,3)=0.009, (1,4)=0.010. Sum = 0.028, consistent with total ≈ 0.029.
- **Analysis:**
  - The source explicitly labels the table columns as "group totals" — meaning 0.029 and 0.020 are sums across 3 pairs each.
  - The paper says "per-pair coupling magnitudes" and then quotes 0.029 and 0.010.
  - If 0.029 is the IC group total, the per-pair IC average is 0.029/3 ≈ 0.0097.
  - The RD group total is 0.020 (from source), not 0.010. Per-pair RD average would be 0.020/3 ≈ 0.0067.
  - The paper's 0.010 for RD does NOT match either the group total (0.020) or the per-pair average (0.0067).
  - The 2.9:1 ratio (0.029/0.010) does not match the ratio from group totals: 0.029/0.020 = 1.45:1 (which matches the "I Ching/RD ratio 1.4:1" stated elsewhere in the paper at line 556 and in the source at line 16).
  - **There is a genuine inconsistency.** The paper uses IC=0.029 (group total) and RD=0.010 (neither group total nor per-pair average), producing a ratio of 2.9:1 that contradicts the 1.4:1 ratio reported in the Method section (line 556) for the same initialization.
  - The draft section 04 (04_block_diagonal_emergence.md, line 55) repeats the same 0.029/0.010/2.9:1 claim, indicating the error propagated from the original draft into the paper.
- **Resolution:** CONFIRMED DATA INCONSISTENCY. Either:
  (a) 0.029 and 0.020 are group totals → ratio is 1.45:1, consistent with the 1.4:1 reported elsewhere. Paper's 0.010 and 2.9:1 are wrong.
  (b) 0.029 and 0.010 are per-pair values → ratio is 2.9:1, but then the source table header "group totals" is mislabeled, and the 1.4:1 in Method is wrong.
  The individual IC pair values (0.009, 0.009, 0.010) sum to 0.028 ≈ 0.029, confirming that 0.029 IS a group total. Therefore interpretation (a) is correct. The RD total is 0.020, not 0.010. The ratio is ~1.45:1, not 2.9:1.

---

## F-02C-14: "Discovers" vs. "Is Trained To Produce" — Framing Inconsistency

- **Severity:** LOW (philosophical but affects reader interpretation)
- **Locations:** Title ("Discovers"), Abstract (line 47), Conclusion (line 1118), throughout
- **Claim:** "The learnable bridge discovers rhombic dodecahedral geometry."
- **Evidence:** The bridge converges to block-diagonal structure when trained with a loss function that explicitly encodes RD face-pair geometry.
- **Problem:** "Discovers" implies the bridge independently arrives at a geometric structure without being told what it is. In reality, the Steersman's contrastive loss directly encodes the RD face-pair partition. The bridge does not discover the geometry — it is trained to reproduce it. The genuinely surprising aspects (speed, robustness, effective rank) are separate from discovery.
- **Mitigation:** The paper does explain the mechanism clearly in §3.2. The reader who reads the method section understands that the geometry is injected. But the title and abstract framing as "discovery" is potentially misleading.
- **Resolution:** This is a framing choice with trade-offs. A more precise title might be "Cybernetic Feedback Stabilizes Rhombic Dodecahedral Geometry..." The current framing is defensible if "discovers" is read as "the optimizer finds the intended structure rather than some other minimum," but a referee may challenge it.

---

## F-02C-15: Holly Battery Optimization — Confounded Comparison

- **Severity:** LOW
- **Location:** Lines 621-626
- **Claim:** Holly Battery achieves "3.8% lower loss than the non-RhombiLoRA baseline while using 9.15 GB less VRAM and producing 6% faster inference."
- **Evidence:** Not detailed in this paper; cited from prior work/experiment.
- **Problem:** These performance comparisons (loss, VRAM, speed) are presented without specifying the baseline configuration. Is it standard LoRA with the same rank? Full fine-tuning? The comparison is tangential to Paper 3's thesis (which is about bridge structure, not performance) but is stated as fact without citation or specification.
- **Resolution:** Either specify the baseline or cite the source for these numbers. Minor issue since Holly's role in this paper is as a structural control, not a performance benchmark.

---

## Summary

| Severity | Count | Finding IDs |
|----------|-------|-------------|
| MAJOR | 1 | F-02C-01 |
| MODERATE | 5 | F-02C-02, F-02C-03, F-02C-04, F-02C-05, F-02C-11, F-02C-13 |
| MINOR | 3 | F-02C-06, F-02C-07, F-02C-08 |
| LOW | 4 | F-02C-09, F-02C-10, F-02C-12, F-02C-14, F-02C-15 |

**Note:** F-02C-13 resolves deferred finding F-01-13. The IC/RD initial ratio inconsistency is CONFIRMED: the paper reports 2.9:1 using mixed units (IC group total vs. an unexplained RD value of 0.010), while the source data yields 1.45:1 using consistent group totals — matching the 1.4:1 reported elsewhere in the same paper.

### Critical Pattern

The paper's strongest claims ("necessary and sufficient," "universal," "causal," "invariant," "absolute," "exactly") consistently use language that overstates the evidence by one degree. The underlying findings are genuinely strong — 100% vs 0% across 42,500+ bridges is remarkable. But the language claims mathematical certainty where the evidence provides robust empirical consistency. A systematic pass replacing absolute qualifiers with empirical ones would strengthen the paper by preempting referee objections without weakening the actual findings.

### The Circularity Issue (F-02C-05)

The most structurally important finding is the circularity between the contrastive loss definition and the block-diagonal metric. The paper would be significantly strengthened by explicitly acknowledging: "The contrastive loss is designed to produce block-diagonal structure. That it succeeds is expected. The non-trivial findings are: (1) convergence speed, (2) initialization independence, (3) effective dimensionality, and (4) task-performance orthogonality." This reframing would make the contribution clearer and the paper more defensible.

### The Missing Control (F-02C-04)

The single most impactful experiment that could be added is n=6 with contrastive loss disabled (spectral-only). This would directly test whether the geometric prior is necessary at the same channel count, eliminating the n/contrastive confound.
