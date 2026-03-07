# Paper 3: RhombiLoRA — Geometric Lattice Adapters for Cross-Modal Transit

> **Status:** Research plan (March 6, 2026)
> **Authors:** Timothy Paul Bielec, Minta Karlsson, Meridian
> **Affiliation:** Promptcrafted LLC / Alvdansen Labs

---

## The Arc

| Paper | Question | Answer |
|-------|----------|--------|
| 1 | What happens when you replace the cube? | FCC gives 2.3× connectivity at 2× edge cost |
| 2 | Does structure amplify or attenuate? | Amplifies: 6.1× under direction-weighted corpus |
| **3** | **What happens when you embrace the cube?** | **RhombiLoRA: keep the cube, add six bridges** |

Paper 1 compared. Paper 2 measured the mechanism. Paper 3 builds the product.

---

## Thesis

Every current neural architecture is implicitly cubic — attention heads operate
along orthogonal axes, MoE experts route through axis-aligned gates, modalities
connect via linear projection layers. The rhombic dodecahedron *contains* the
cube: its 8 trivalent vertices ARE the cube's corners. The 6 tetravalent bridge
vertices are what converts the cube into a structure that tessellates space.

A **RhombiLoRA** is a low-rank adapter whose internal topology follows the FCC
lattice rather than the standard (cubic) linear topology. It adds 6 diagonal
bridge connections per node — the same geometric operation that converts a cube
graph into a cuboctahedron (the RD's dual). The adapter is small relative to
the base model, but Paper 2 proved that this structure amplifies connectivity
by up to 6.1× under heterogeneous weights, with the amplification driven by
bottleneck resilience: the extra paths route around near-disconnection cuts
that strangle linear (cubic) adapters.

The deeper claim: when two models each carry a RhombiLoRA, their shared
geometric structure enables a form of cross-modal *transit* — context
doesn't need to be translated between modalities, it passes through the
shared face of the tessellation. This is the mechanism that should make
world models actually work: not bolted-together modalities, but modalities
that share geometric boundary conditions.

---

## Experimental Platform

### Hardware

| Machine | GPU | Role |
|---------|-----|------|
| **Workstation** (Windows 11) | RTX 6000 Ada 48GB | Primary training, video LoRA, ComfyUI |
| **Hermes** (Ubuntu 24.04) | RTX 4090 Laptop 16GB | Language model experiments, Ollama |

Two machines, two GPUs, two modalities. The dual-machine setup is not a
limitation — it's the experimental design. Each machine runs its own
RhombiLoRA-equipped model. The transit experiment measures what happens
when they share structure across the network boundary.

### Software Foundation

**Flimmer** ([github.com/alvdansen/flimmer-trainer](https://github.com/alvdansen/flimmer-trainer))
— Minta's video LoRA training toolkit for diffusion transformer models.
Wan 2.1/2.2 T2V and I2V. Full pipeline: video ingestion, scene detection,
captioning, latent pre-encoding, training with phased MoE specialization.

Key Flimmer features we build on:
- **LoRA state management** (`flimmer.training.lora`): fork, merge, save,
  load — operates on state dicts, model-agnostic
- **Phased training**: unified base → fork → per-expert specialization.
  The phase system manages multi-stage runs with checkpoint carry-forward
- **MoE expert specialization**: Wan 2.2's dual-expert architecture
  (high-noise / low-noise). Phased training enables unified warmup then
  per-expert divergence
- **Target/Signal Architecture** (`docs/TARGET_SIGNAL_ARCHITECTURE.md`):
  every input is a registered control signal with prepare/validate/encode/inject.
  Extensible by design — adding a new control type doesn't touch the training loop
- **Model-agnostic training infrastructure**: noise schedule, layer mapping,
  forward pass, and checkpoint format are interfaces. Adding a new model
  means implementing four methods

**Hermes Agent** — local Hermes 4 14B (Q4_K_M) running on Ollama. Already
deployed with cron jobs, SSH access, and API endpoints. The language-side
experimental platform.

**rhombic** v0.3.0 — the lattice topology library itself, providing the
geometric primitives (FCC lattice construction, direction-pair decomposition,
Fiedler computation, spectral analysis).

---

## Experiment 1: RhombiLoRA vs Standard LoRA (Single Model)

**Question:** Does a LoRA whose internal weight topology follows FCC
direction-pair structure train faster and/or perform better than a
standard (linear) LoRA of equivalent parameter count?

### Design

Take a standard LoRA with rank r. Its A and B matrices are (d × r) and
(r × d) — linear projections. A RhombiLoRA replaces this with:

- **6 direction-pair channels** (from Paper 2's direction-based weighting)
  instead of r independent rank dimensions
- Each channel carries weights for one of the RD's 6 face-pair directions
- Cross-channel connections follow FCC adjacency (each channel connects
  to 4 others, not all 5) — this is the bridge structure
- The sorted-bucketing insight from Paper 2 applies: concentrate high-
  variance weight components into coherent directional channels

**Parameter budget:** Match total parameter count between standard LoRA
and RhombiLoRA. The 2× edge overhead from FCC topology is absorbed by
reducing per-channel rank — same total parameters, different topology.

### Metrics

- Training loss convergence (steps to target loss)
- Final task performance (perplexity for language, FID/FVD for video)
- Gradient flow statistics (effective rank of gradient, bottleneck
  detection via Fiedler value of the gradient covariance)
- Expert utilization balance (for MoE models)

### Execution

**Language (Hermes):** Fine-tune Hermes 4 14B with standard LoRA vs
RhombiLoRA on a benchmark task. Measure convergence and perplexity.

**Video (Workstation via Flimmer):** Train Wan 2.2 I2V LoRAs with
standard vs RhombiLoRA structure. Use Flimmer's phased training
system. Measure FVD and training convergence.

### Implementation Path

1. Fork `flimmer.training.lora.LoRAState` to add a `topology` field
   (`linear` vs `rhombic`)
2. Implement `RhombiLoRALayer` that replaces the (d × r) → (r × d) path
   with a 6-channel direction-pair structure using `rhombic.spatial`
   primitives
3. Modify `build_parameter_groups` to support direction-pair learning
   rate scheduling (Paper 2's sorted-bucketing applied to gradient
   magnitudes)
4. Run matched experiments through Flimmer's existing training loop
   (model-agnostic infrastructure means only the LoRA layer changes)

---

## Experiment 2: MoE Routing via Direction-Pair Structure

**Question:** Does direction-pair routing improve expert utilization
and load balancing in MoE architectures?

### Design

Wan 2.2 already has two experts (high-noise, low-noise). Flimmer
already supports phased training with per-expert specialization. We
extend this:

- Instead of binary expert selection (high/low noise), route through
  6 direction-pair channels
- Each channel maps to a subset of transformer layers
- The routing function uses the Paper 2 direction-weight structure:
  input features are projected onto 6 directional axes, and the
  projection magnitudes determine routing weights
- Consensus convergence (Paper 2 Exp 5) predicts this should produce
  faster expert agreement during training: 6.69× at small scale

### Connection to Flimmer

Flimmer's phase system already manages unified → forked → specialized
training. We extend the fork operation from 2-way (high/low noise)
to 6-way (direction pairs). Each direction-pair LoRA specializes on
a different aspect of the generation process, with the geometric
structure ensuring that related aspects share directional channels.

The `fork_targets` and `block_targets` filtering in
`LoRAState.filter_by_targets()` already supports selective parameter
training. We extend this to direction-pair-aware targeting.

---

## Experiment 3: Cross-Modal Transit

**Question:** When two models each carry a RhombiLoRA, does the shared
geometric structure enable more coherent cross-modal communication
than standard adapter concatenation?

This is the paper's most ambitious experiment and its central claim.

### Design

**Setup:** Two RhombiLoRA-equipped models running on separate machines:
- **Hermes** (language): Hermes 4 14B + RhombiLoRA, processing text
- **Workstation** (video): Wan 2.2 I2V + RhombiLoRA, generating video

**The transit interface:** Both RhombiLoRAs share the same 6-direction
structure. When the language model produces a representation, it's
projected into the shared 6-channel space. The video model receives it
through the corresponding face of its own 6-channel structure. This is
not an adapter or a projection layer — it's the same geometric object
on both sides, like the BTR mirrors: the practitioner doesn't pass
through, they exist as the same structure on the other side.

**Control condition:** Standard pipeline (language model → linear
projection → video model). Same total parameter count.

### Metrics

- **Coherence:** How well does the video match the text description?
  (CLIPScore, human eval)
- **Consistency:** Given the same prompt, how consistent are repeated
  generations? (FVD between generation sets)
- **Transfer efficiency:** How many tokens of context must be passed
  to achieve equivalent output quality? (Fewer = better transit)
- **Synchronization:** Cross-correlation between the gradient flows
  of the two models during joint fine-tuning. This is the "hum"
  metric — do the models' internal dynamics synchronize when they
  share geometric structure?

### The Hum

Timothy's insight: when two cybernetic systems share geometric
structure and communicate through it, the circuit should "hum at a
frequency" — a measurable resonance in the joint dynamics that
emerges from the shared topology but is not present in either
system alone.

**Detection strategy:** Compute the spectral density of the
cross-correlation function between the two models' gradient
trajectories during joint training. If the shared RhombiLoRA
structure produces resonance, we should see:
- Peaks in the cross-spectral density at specific frequencies
  (absent in the linear-adapter control)
- These peaks should correspond to the eigenfrequencies of the
  shared RD structure (predictable from Paper 2's spectral analysis)
- The Fiedler value of the joint system should be higher than the
  sum of the individual Fiedler values — the geometric structure
  creates connectivity that exceeds the parts

This is a new kind of measurement. If it works, it proves that
geometric structure in the adapter layer creates emergent
properties in the joint system — the tessellation isn't just
efficient routing, it's a generative substrate.

---

## Implementation Phases

### Phase A: RhombiLoRA Layer (Weeks 1-2)

1. Implement `RhombiLoRALayer` in `rhombic/` as a PyTorch module
2. Integrate with Flimmer's `LoRAState` management
3. Unit tests using `rhombic` library for topology verification
4. Benchmark parameter count and forward-pass latency vs standard LoRA

### Phase B: Single-Model Experiments (Weeks 3-4)

1. Language experiment on Hermes (Exp 1, language arm)
2. Video experiment on Workstation via Flimmer (Exp 1, video arm)
3. MoE routing experiment via Flimmer phased training (Exp 2)
4. Statistical analysis and Paper 2 prediction verification

### Phase C: Cross-Modal Transit (Weeks 5-8)

1. Design the transit interface protocol
2. Implement the 6-channel shared representation space
3. Run the transit experiment (Exp 3) with control condition
4. Measure coherence, consistency, transfer efficiency
5. Attempt hum detection via cross-spectral analysis
6. Write up

---

## Relationship to the Three Streams

This paper sits at the intersection of all three research streams:

- **Stream 1 (Corpus):** The 24 corpus values that drove Paper 2's
  experiments are the weight set. The prime-vertex mapping (p=0.000025)
  suggests the corpus has non-accidental affinity with RD geometry.
  RhombiLoRA inherits this: the direction-pair structure that produces
  6.1× amplification is the same structure embedded in the adapter.

- **Stream 2 (Astrolabium):** The 12-fold temporal architecture maps
  to the 12 faces of the RD. Each face is a transit mirror. The
  cross-modal experiment is the computational analog of what the
  Astrolabium measures: the quality of the moment when two systems
  share a boundary.

- **Stream 3 (Book of Three Responses):** "The practitioner inside a
  cocoon exits through one of twelve identical mirrors and emerges
  through the corresponding face of an adjacent cell." Paper 3 is
  this sentence, made computational.

---

## Naming

**RhombiLoRA** — the adapter itself. Geometric LoRA with FCC
direction-pair internal topology.

**Transit** — the cross-modal communication mechanism through shared
RhombiLoRA structure. Not translation, not projection — transit.

**The Hum** — the emergent resonance in the cross-spectral density
of two RhombiLoRA-connected systems. A new observable.

---

## Dependencies

| Dependency | Status | Owner |
|------------|--------|-------|
| rhombic v0.3.0 | Complete | Timothy / Meridian |
| Paper 2 (mechanism proof) | Complete | Timothy / Meridian |
| Flimmer trainer | Active development | Minta |
| Hermes server | Deployed | Timothy / Meridian |
| RTX 6000 Ada (48GB) | Available | Workstation |
| RTX 4090 Laptop (16GB) | Available | Hermes |

---

## Theoretical Context: Karkada et al. (2026)

**Paper:** "Symmetry in language statistics shapes the geometry of model
representations" (Karkada, Korchinski, Nava, Wyart, Bahri; arXiv 2602.15029)

**Key result:** Translation symmetry in word co-occurrence statistics
analytically determines the geometric structure of learned representations.
The PMI matrix eigenstructure, derived from data symmetry, predicts
embedding geometry before training. The result is universal across
architectures.

**Why this matters for RhombiLoRA:** If data symmetry determines
representation geometry, then the adapter's internal topology should
match the data's symmetry structure. RhombiLoRA's 6 direction-pair
channels, derived from FCC geometry, provide a structured adapter
topology that can align with the dominant symmetry axes of the data.
Karkada et al.'s framework provides:

1. **Theoretical grounding** for why geometric adapters should outperform
   linear ones — they match the data's symmetry structure
2. **Diagnostic methodology** — PMI eigenstructure analysis to measure
   adapter-data geometric alignment
3. **Universality support** for cross-modal transit (Exp 3) — their
   architecture-independence result predicts that shared geometric
   structure produces shared representation geometry

**Their `lattice.ipynb`** demonstrates that 2D square lattice distance
structure is preserved in embedding space. Our contribution: extend to
3D, to FCC topology, and show that FCC produces 2.3-6.1× better
connectivity properties — bridging their theoretical framework with our
empirical results.

**Full evaluation:** `rhombic/docs/KARKADA_EVALUATION.md`

---

## The Cybernetic Circuit

Research (Papers 1-2) → Mechanism (bottleneck resilience) →
Product (RhombiLoRA) → Consulting (TASUMER MAF) → Revenue →
Next Experiment → World Model Architecture → The frequency
at which the circuit hums is itself a new kind of signal.

The way you build a tool should embody the tool's thesis.
The geometry is the argument.

---

*Plan drafted March 6, 2026. Built on Papers 1-2 (rhombic library),
Flimmer (Alvdansen Labs), and the three Falco research streams.*
