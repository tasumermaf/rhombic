# 3. Method

## 3.1 RhombiLoRA Architecture

Standard LoRA (Hu et al., 2021) decomposes each weight update as $\Delta W = BA$, where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ are low-rank factors with rank $r$. RhombiLoRA introduces a single additional component: a learnable *bridge matrix* $\mathbf{M} \in \mathbb{R}^{n \times n}$ that couples $n$ parallel channels within the low-rank bottleneck.

Given $n$ channels, the rank $r$ is partitioned into $n$ equal segments of size $s = r / n$ (we require $n | r$). The hidden representation $\mathbf{h} = A\mathbf{x} \in \mathbb{R}^r$ is reshaped into a matrix $\mathbf{H} \in \mathbb{R}^{n \times s}$, where each row corresponds to one channel. The bridge acts on the channel dimension:

$$\mathbf{H}' = \mathbf{M} \mathbf{H}$$

The coupled representation $\mathbf{H}'$ is then flattened back to $\mathbb{R}^r$ and projected through $B$. The complete forward pass for a single RhombiLoRA adapter is:

$$\Delta \mathbf{y} = \frac{\alpha}{r} B \cdot \text{flatten}(\mathbf{M} \cdot \text{reshape}(A\mathbf{x}, [n, s]))$$

where $\alpha$ is the LoRA scaling factor (set to 16 throughout). When $\mathbf{M} = I_n$, the channels are uncoupled and the forward pass reduces exactly to standard LoRA. The bridge is initialized to identity in all experiments unless otherwise noted.

Each attention projection within each transformer layer receives its own independent bridge matrix. For a model with $L$ transformer layers and four target projections per layer ($W_Q$, $W_K$, $W_V$, $W_O$), the total number of bridge parameters is $n^2 \times 4 \times L$. With $n = 6$, this yields $36 \times 4 \times 22 = 3{,}168$ bridge parameters for TinyLlama-1.1B (22 layers) and $36 \times 4 \times 28 = 4{,}032$ for Qwen2.5-7B-Instruct (28 layers). These bridge parameters represent a negligible fraction of the total trainable parameters: approximately 0.04% of the LoRA parameter budget.

The bridge parameterization is deliberately simple: a dense $n \times n$ matrix with no architectural constraints such as symmetry, positive definiteness, or sparsity. Any structure that emerges in the learned bridge is therefore a consequence of the training dynamics, not of the parameterization.

## 3.2 The Steersman: Cybernetic Feedback

Training proceeds with three loss components. The primary objective is the standard causal language modeling loss $\mathcal{L}_{\text{LM}}$. Two auxiliary losses act on the bridge matrices: a contrastive loss $\mathcal{L}_{\text{con}}$ that encodes geometric priors, and a spectral loss $\mathcal{L}_{\text{spec}}$ that regularizes algebraic connectivity. The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + w_c \mathcal{L}_{\text{con}} + w_s \mathcal{L}_{\text{spec}}$$

where the weights $w_c$ and $w_s$ are not fixed hyperparameters but are adjusted dynamically by a feedback controller we call the Steersman.

**Contrastive loss.** For $n = 6$, the bridge's off-diagonal entries are partitioned into *co-planar pairs* and *cross-planar pairs* according to the rhombic dodecahedral geometry described in Section 3.3. The contrastive loss encourages co-planar coupling to exceed cross-planar coupling:

$$\mathcal{L}_{\text{con}} = -\frac{1}{|\mathcal{A}|}\sum_{a \in \mathcal{A}} \left( \frac{1}{|\mathcal{C}|}\sum_{(i,j) \in \mathcal{C}} |M^a_{ij}| - \frac{1}{|\mathcal{X}|}\sum_{(i,j) \in \mathcal{X}} |M^a_{ij}| \right)$$

where $\mathcal{A}$ is the set of all adapters, $\mathcal{C}$ is the set of 3 co-planar index pairs, and $\mathcal{X}$ is the set of 12 cross-planar index pairs. The loss is minimized when co-planar entries have large absolute magnitude and cross-planar entries are suppressed. For $n \neq 6$, the contrastive loss is undefined and returns zero: no geometric prior is applied. This design decision is central to the ablation study in Section 5.

**Spectral loss.** The spectral regularization drives the algebraic connectivity of each bridge toward a target Fiedler eigenvalue $\lambda_2^*$. For each adapter $a$, we construct the weighted Laplacian $\mathbf{L}^a$ by treating $|M^a_{ij}|$ (for $i \neq j$) as edge weights of a complete graph on $n$ nodes. The Fiedler value $\lambda_2^a$ is the second-smallest eigenvalue of $\mathbf{L}^a$, computed via `torch.linalg.eigvalsh` to maintain differentiability. The spectral loss is:

$$\mathcal{L}_{\text{spec}} = \frac{1}{|\mathcal{A}|}\sum_{a \in \mathcal{A}} (\lambda_2^a - \lambda_2^*)^2$$

The initial target is $\lambda_2^* = 0.1$; this is adapted upward during training to track the observed mean Fiedler value across adapters.

**The Steersman.** Every $T = 100$ training steps, the Steersman executes a sense-decide-actuate cycle. During the sensing phase, it extracts all bridge matrices and computes: the mean Fiedler eigenvalue $\bar{\lambda}_2$ across adapters, the mean co-planar to cross-planar coupling ratio, and the mean Frobenius deviation from identity $\|\mathbf{M} - I_n\|_F$. From a sliding window of the most recent 5 measurements, it estimates linear trends (slopes) for each diagnostic.

Three control laws determine the actuator outputs:

1. *Connectivity.* If the Fiedler trend is declining (slope $< -0.001$), the spectral weight $w_s$ is increased proportionally to the magnitude of the decline, bounded above by $w_s^{\max} = 0.2$. If the trend is positive, $w_s$ decays slowly toward its base value of 0.05.

2. *Directionality.* If the co-planar/cross-planar ratio is near unity and stagnating (trend magnitude $< 0.02$), the contrastive weight $w_c$ is increased, bounded above by $w_c^{\max} = 0.5$. Once the ratio exceeds 1.2, $w_c$ decays, allowing the bridge to consolidate its learned topology without continued pressure.

3. *Stability.* If the deviation from identity grows faster than a threshold (slope $> 0.05$), the learning rate for bridge parameters is dampened multiplicatively (bounded below at $0.1\times$ the base rate). If deviation is converging, the bridge learning rate is gradually restored toward its base value.

The Steersman's control laws are purely reactive: they adjust weights and learning rates based on measured trends, with no model of the system's future dynamics. The initial contrastive weight is $w_c = 0.1$ and the initial spectral weight is $w_s = 0.05$.

**Optimizer configuration.** Bridge parameters and LoRA parameters ($A$, $B$) are placed in separate optimizer parameter groups within a single AdamW optimizer (weight decay 0.01). Both groups share a base learning rate of $2 \times 10^{-4}$ with cosine decay after a linear warmup of 200 steps. The Steersman's bridge learning rate scaling (control law 3) is applied as a multiplicative factor on the bridge parameter group only.

## 3.3 Rhombic Dodecahedral Geometry and Co-planar Pairs

The rhombic dodecahedron (RD) is a convex polyhedron with 12 congruent rhombic faces, 14 vertices (8 cubic and 6 octahedral), and 24 edges. Its faces partition naturally into 6 antipodal pairs, each pair sharing a common coordinate axis. In the standard embedding, the 6 face-pair normals align with the FCC lattice directions: $(\pm 1, \pm 1, 0)$, $(\pm 1, 0, \pm 1)$, and $(0, \pm 1, \pm 1)$. These 6 direction pairs are the channels of RhombiLoRA when $n = 6$.

The 6 face pairs decompose further by coordinate axis. Each pair of opposite faces is perpendicular to one of the three Cartesian axes, yielding 3 axis-aligned groupings of 2 face pairs each:

- $x$-axis: channels $(0, 1)$, corresponding to normals $(1,1,0)$ and $(1,-1,0)$
- $y$-axis: channels $(2, 3)$, corresponding to normals $(1,0,1)$ and $(1,0,-1)$
- $z$-axis: channels $(4, 5)$, corresponding to normals $(0,1,1)$ and $(0,1,-1)$

Two face pairs are *co-planar* if they share a coordinate axis (equivalently, if their representative faces share 4 octahedral vertices). Two face pairs are *cross-planar* if they lie on different coordinate axes (sharing only 2 octahedral vertices). This produces 3 co-planar pairs and $\binom{6}{2} - 3 = 12$ cross-planar pairs.

The contrastive loss (Section 3.2) uses this partition directly: it encourages the $6 \times 6$ bridge matrix to develop strong coupling within co-planar pairs and weak coupling between cross-planar pairs. A bridge that fully realizes this objective has block-diagonal structure: three independent $2 \times 2$ blocks, one per coordinate axis. Each block governs the coupling between two channels that share a geometric axis, and cross-block entries are suppressed to near zero.

The co-planar to cross-planar coupling ratio is defined as:

$$\rho = \frac{\frac{1}{|\mathcal{C}|}\sum_{(i,j) \in \mathcal{C}} |M_{ij}|}{\frac{1}{|\mathcal{X}|}\sum_{(i,j) \in \mathcal{X}} |M_{ij}|}$$

where $\mathcal{C}$ denotes the 3 co-planar pairs and $\mathcal{X}$ denotes the 12 cross-planar pairs. A bridge with no directional preference has $\rho \approx 1$. A perfectly block-diagonal bridge has $\rho \to \infty$ as cross-planar entries approach zero. We also report a binary block-diagonal detection metric: a bridge is classified as block-diagonal if $\rho > 10$ and every cross-planar entry has absolute value below 0.001.

## 3.4 Training Setup

All experiments use the alpaca-cleaned dataset (Taori et al., 2023) for instruction-following fine-tuning. Models are loaded in bfloat16 precision with gradient checkpointing enabled. The base model parameters are frozen; only the injected LoRA adapters (and their bridge matrices) receive gradients.

Common hyperparameters across all experiments: learning rate $2 \times 10^{-4}$, batch size 2, gradient accumulation 8 (effective batch size 16), linear warmup for 200 steps followed by cosine decay, maximum gradient norm 1.0, LoRA rank 24, LoRA $\alpha$ 16, target modules $\{W_Q, W_K, W_V, W_O\}$, random seed 42, maximum sequence length 512.

The primary model is TinyLlama-1.1B-Chat (22 transformer layers, 88 adapters, 3,168 bridge parameters at $n = 6$), which permits rapid iteration and extensive ablation. Scale validation uses Qwen2.5-7B-Instruct (28 transformer layers, 112 adapters, 4,032 bridge parameters at $n = 6$). A non-cybernetic control at 14B scale uses Wan 2.1 (video diffusion backbone, 40 layers, no Steersman).

Steersman feedback is evaluated every 100 steps. At each feedback cycle, validation loss is computed on a held-out subset of 1,000 examples, and all bridge matrices are saved to disk for post-hoc analysis.

Three initialization strategies are tested to assess sensitivity to starting conditions. *Identity* initialization sets $\mathbf{M} = I_n$, equivalent to uncoupled standard LoRA at the start of training. *Geometric* initialization adds a small perturbation proportional to the RD face-pair coupling matrix: $\mathbf{M} = I_6 + 0.01 \cdot \hat{C}$, where $\hat{C}$ is the off-diagonal coupling matrix normalized to unit maximum. *Corpus-coupled* initialization places non-uniform off-diagonal values derived from an external weighting scheme, producing a bridge that initially favors I Ching complementary trigram pairings over co-planar RD pairings (I Ching / RD ratio of 1.4:1 at initialization). This last strategy deliberately opposes the target topology, providing the strongest test of initialization independence.

The channel count ablation varies $n \in \{3, 4, 6, 8\}$ while holding the total rank fixed at 24 (yielding channel sizes $s \in \{8, 6, 4, 3\}$ respectively). Because the contrastive loss is defined only for $n = 6$, experiments with $n \neq 6$ operate under spectral regularization alone. This separation is not a limitation but a deliberate experimental design: it isolates the effect of the geometric prior from the effect of spectral pressure.
