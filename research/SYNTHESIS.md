# Research Suite — Cross-Cutting Synthesis

> What the collected literature (March–July 2026 sweeps; ~100 assessed works)
> establishes about this program's position. Sources: `docs/LITERATURE_WATCH.md`
> (35 papers, three expansions), the non-English/Chinese landscape sweep
> (34 primary papers + 4 surveys), the TeLoRA competitive scouts (Mar 26–27),
> the Karkada evaluation, the external-implementations survey, the Nemotron
> engineering research, and the Director's representation-alignment map
> (June 2026). Individual assessments live verbatim on the cards in `papers/`.
> **Dated judgments — the newest full sweep is July 2026.**

## 1. The claims ledger (REVISED by the July 3, 2026 fourth-expansion sweep)

> The fourth expansion (`docs/LITERATURE_WATCH_2026-07-03.md`) materially
> revised the April picture. Current state:

1. **Learnable bridge between learnable A and B — STRONG FORM DEAD.**
   MoSLoRA (EMNLP 2024) is a learnable dense mixer between learnable A/B;
   StelLA (NeurIPS 2025 Spotlight) a full learnable r×r between learnable
   Stiefel factors; AdaLoRA a learnable diagonal middle; BoRA learnable
   diagonal block bridges. All four prior sweeps missed MoSLoRA. What
   survives: *what the bridge does* — programmable/structural topology,
   emergent BD, diagnostics — not where it sits.
2. **Contrastive objectives on *internal* adapter structure** — still clean.
3. **Polytope geometry as an adapter design parameter** — still clean
   (zero results, English and Chinese, re-verified July 2026).
4. **Emergent block-diagonal from geometric pair specification** — still
   clean; all block-structure work remains imposed (add MELoRA, GraLoRA,
   HyperAdapt to the imposed table).
5. **Cybernetic closed-loop feedback on adapter internals — SURVIVES
   NARROWLY.** FlexLoRA (ICLR 2026) closes a spectral-entropy→rank-action
   loop; ours must be stated as feedback controlling *topology/update
   shaping*, not rank budgets. Space crowding fast.
6. **Adapter internals as diagnostics — PARTIALLY ERODED.** The weight-space
   community entered PEFT (W2T, Mar 2026); adapter spectra read training
   objectives linearly (arXiv:2604.08844); training-time merge-conflict
   prediction is claimed (arXiv:2606.19549). Still unclaimed: overfit
   detection from adapter internals, and weight-only post-hoc merge
   prediction. **1B's window is weeks-to-months.**

## 2. The threats (ranked; revised July 2026)

- **MoSLoRA** (EMNLP 2024) + **StelLA** (NeurIPS 2025 Spotlight) +
  **AdaLoRA** + **BoRA** — the three-factor lineage that kills claim 1's
  strong form. Every bridge-adjacent sentence in Papers 3–4 must be written
  against this set. MoSLoRA is also the correct dense-bridge baseline
  (= BM battery Config F); SVFT-Random is the matched-params fixed-mask
  ablation for BM-003.
- **W2T + the weight-space-learning community** (Mar 2026 onward) — entered
  PEFT at hub scale; partially preempts routing/retrieval; their
  flattened-baseline collapse vs our 72.3% linear result is a regime
  contrast to claim, not hide. **2606.19549** preempts training-time merge
  prediction. 1B must move fast and be framed weight-only/post-hoc.
- **FlexLoRA** (ICLR 2026) — nearest claim-2 collision (spectral loop on
  rank allocation). The Steersman claim is stated as topology programming
  under feedback, distinct from budget allocation.
- **DiaBlo** (ICLR 2026) and **Block-Diagonal LoRA** (NeurIPS 2025) —
  vocabulary collision, not mechanism collision; keep the "imposed vs.
  emergent" paragraph everywhere.
- **"Learning Rate Matters"** (Microsoft, Feb 2026) — the reason the claim
  is diagnostics at benchmark parity, never loss improvement.
- **UWSH/Share (JHU group)** — most likely group to run the Tier-3
  principal-angle experiment next; **2606.01090** is concurrent with
  BM-004's thesis outside PEFT. Both are speed threats, not scoops — the
  specific experiments remain unclaimed as of July 3, 2026.

## 3. The supports

- **Block-Recurrent ViT** (ICLR 2026) — BD structure emerges spontaneously in
  transformer representational similarity: independent evidence the motif is
  a natural phenomenon.
- **Shuttleworth et al.** ("intruder dimensions") — spectral analysis of
  adapter weights reveals real training phenomena; validates weight-structure
  diagnostics as a category.
- **Mallinar et al.** (grokking via AGOP, ICML 2025) — structured weight
  patterns emerge **when the task demands it**, even in non-neural models.
- **Karkada et al.** — data symmetry analytically determines representation
  geometry.
  Together, Mallinar + Karkada are the published theoretical foundation for
  the program's two-factor experiment (BM-004): structure should appear
  exactly when the data carries the symmetry the prior encodes.
- **Tam & Dunson** — differentiable Fiedler prior art (regularizer); our
  diagnostic use is downstream and properly cited.
- **Olsen et al.** (SGD → spectra, Dyson Brownian motion) — candidate theory
  for the universal spectral attractor (Fiedler ≈ 0.09).
- **Equivariant DL** (Bronstein; NequIP; Allegro) — "form-first computation"
  already works where the geometry is the physics; the governing principle —
  *a symmetry prior helps exactly when the task is invariant under it* — is
  the program's organizing sentence, discovered independently by our nulls.

## 4. The Chinese and non-English landscape

Extremely active, organized on three axes: (1) parameter compression via
frozen-outer bridge matrices; (2) block-diagonal for serving efficiency;
(3) spectral methods for redundancy (SeLoRA, Spectral Surgery; BSLoRA
Zhejiang/Fudan; SMoA Northeastern; StructLoRA's inter-layer graph
coordinator — closest cousin, wrong scale). Two comprehensive Chinese-led
surveys found **nothing** overlapping the program's mechanism — strong
negative evidence, since those surveys would have captured Chinese-language
work on bridge topology if it existed. Standing residue: NLPCC/CCL
proceedings are not English-indexed; corporate labs (DAMO, ByteDance, Noah's
Ark, DeepSeek) are unassessable from outside. Verdict of the April sweep:
*"Claims to hedge: none."*

## 5. The representation-alignment cluster (Director's map, June 2026)

The cross-modal ambition lives in an active named field: the Platonic
Representation Hypothesis (convergent representations across modalities),
vec2vec (unsupervised embedding translation), with 2026 critiques showing
alignment degrades at scale and third modalities sit near-orthogonal until
explicitly coupled. The decisive negative for naive adapter transfer:
**LoRAs do not transfer between video-diffusion variants because singular
subspaces mismatch** — the same wall as the program's own nulls, located
precisely in *spectral* compatibility. The falsifiable next step is the
principal-angle measurement between modality adapters' subspaces.

## 6. The nulls as compass (our own, kept beside the external record)

Exp 2.5 (co/cross 1.002, p = 0.474) and L-001 (rank-basis rotational
symmetry) mark the boundary: weight space has no spatial metric privileging
any lattice **under a rotation-symmetric parameterization on generic data**.
The structural mask (`rd_graph`) breaks the parameterization symmetry;
geometry-matched data supplies the missing invariance. What remains untested
is their interaction — which is BM-004, and the reason this suite's most
load-bearing external cards are Mallinar, Karkada, and the equivariance
anchors.

## 7. Engineering references (Nemotron era)

The competition research produced a durable training playbook whose external
anchors are on cards tagged `nemotron-engineering`: prompt-loss-weight ≈ 0.1
beats completion-only for short answers (Shi et al. 2024); TRL
completion-loss tokenization bugs; GRPO recipes (low LR, KL=0 for RLVR,
graduated rewards); hybrid Mamba/MoE adapter pitfalls (PEFT out_proj bypass;
router auto-exclusion under all-linear).
