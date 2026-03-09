# Experiment 3 Expanded: Geometric Transit Architecture

> **Date:** March 8, 2026
> **Authors:** Timothy Paul Bielec, Meridian
> **Status:** Design document. Incorporates Exp 2 findings.
> **Scope:** Three interconnected experiments + dataset architecture +
> compute infrastructure plan.

---

## The Problem Exp 2 Revealed

Exp 2 proved that FCC topology produces 1.73× more cross-channel coupling
than cubic at 7B scale. But the co-planar vs cross-planar signal collapsed
to 1.02× — barely above noise. The bridge learns THAT mixing helps, but
doesn't learn structured directional preference.

**Why:** Alpaca-cleaned is isotropic — it contains no signal that would
reward one direction of mixing over another. The bridge cannot learn
directional structure from directionless data.

**The insight (Timothy):** The training data itself must encode the geometric
principles we want the model to learn. We need to teach the model HOW to use
the bridge, not just give it a bridge and hope gradient flow finds the
structure.

This reframes everything. Experiments 3A-3C each require purpose-built
training data alongside architectural innovation.

---

## The Paired Training Paradigm (Proven at Moonvalley)

Before designing the experiments, we document an existing technique from
Promptcrafted's production work that directly informs the architecture.

### What We Already Built

Timothy and Minta trained a character LoRA for Moonvalley's Marey video
model (Flux-30B-control-v2) using a **two-stage paired data strategy:**

**Stage 1 — Character LoRA:** Standard video LoRA training on Peter Rabbit
IP. The model learns the character's appearance, proportions, and motion
patterns. This is a conventional LoRA — `train_data/` only.

**Stage 2 — Paired Control Training:** The trained character LoRA continues
training with PAIRED data:
- `train_data/` = Peter Rabbit character videos (the target output)
- `control_data/` = corresponding pose skeleton videos (the control signal)
- `use_control_data: True` — activates paired mode
- `task_id: 'video_condition'`, `cond_type: 'pose'`

The model learns to map FROM the pose skeleton TO the character-specific
output. The critical insight: by training the controlnet on a CUSTOM
skeleton derived from the character (not a generic human skeleton), the
control signal is aligned to the character's specific anatomy. The pose
data drives Marey's internal pose control system, but now that system
understands Peter Rabbit's body, not a human body.

**Config (from production):**
```yaml
dataset:
  use_control_data: True
  default_values:
    train:
      task_id: ['video_condition', 'override']
      cond_type: ['pose', 'override']
```

Rank 64, alpha 64, lr 6e-5, 200 epochs, checkpoint every 10, gradient
checkpointing enabled. Trained on Moonvalley's cloud infrastructure.

### Why This Is Transit

The paired training strategy is structurally identical to what we're
building with RhombiLoRA:

| Marey Paired Training | RhombiLoRA Transit |
|----------------------|-------------------|
| `train_data/` (character video) | Model A's representation space |
| `control_data/` (pose skeleton) | Model B's representation space |
| LoRA learns the mapping | Bridge learns the mapping |
| Generic learned projection | **Geometric** (6-channel FCC) projection |
| Custom skeleton per character | Shared topology per modality pair |

The difference: Marey's paired training uses a generic learned mapping
(the LoRA itself). RhombiLoRA paired training uses the 6-channel bridge to
structure that mapping — the FCC topology constrains HOW the transit is
learned, not just THAT it's learned.

### Application to Each Experiment

**Exp 3A (MoE Expert Bridge):** The Flimmer fork-and-specialize workflow
IS a paired training setup. After forking:
- High-noise expert's `train_data` = noisy generation targets
- Low-noise expert's `train_data` = fine detail targets
- The BRIDGE between them is the geometric analog of the controlnet —
  it learns the mapping between two representational domains
- **Extension:** Add a Phase 4 where the merged experts train on paired
  data: `train_data/` = final output, `control_data/` = the other expert's
  intermediate representations. The bridge learns to coordinate experts
  the same way Peter Rabbit's controlnet learns to coordinate pose and
  character.

**Exp 3B (Cross-Modal Transit):** This is EXACTLY paired training:
- `train_data/` = target modality output (image from Flux Dev)
- `control_data/` = source modality conditioning (text from Qwen)
- Phase 1: independent training (character LoRA equivalent)
- Phase 2: paired training (controlnet equivalent)
- The bridge geometry provides the custom skeleton — shared topology that
  constrains the mapping

**Exp 3C (LGM):** Each expert pair connected by a face can use the paired
training paradigm. Expert A's intermediate output becomes Expert B's
control signal, and vice versa. The 12 faces of the RD define 12 paired
training relationships per expert. This scales the Marey paradigm from
one pair to a full tessellated network.

**Geometric Dataset:** Component 3 (Cross-Domain Transit) adopts the
paired format directly:
```
geometric_transit_data/
├── train_data/    # Target representation (e.g., code implementation)
│   ├── 001.txt
│   └── 002.txt
├── control_data/  # Source representation (e.g., math definition)
│   ├── 001.txt
│   └── 002.txt
```
The bridge learns which direction-pair channels carry the transit signal
for each domain pairing. Different domain pairings (math↔code vs
math↔intuition) should activate different bridge directions — producing
the co-planar signal that Alpaca couldn't.

---

## Three Experiments

### 3A: MoE Expert Bridge Training (Wan 2.2)

**Question:** When MoE experts are trained separately with RhombiLoRA, does
the shared bridge topology provide a natural alignment basis for merging?

**Architecture:**

Flimmer already implements the three-phase MoE workflow:
```
Phase 1 (full_noise):  Unified warmup → single LoRA trains on all noise levels
Phase 2 (high_noise):  Fork → high-noise expert specializes (composition, motion)
Phase 3 (low_noise):   Fork → low-noise expert specializes (texture, detail)
Merge:                 merge_experts() → inference-ready merged LoRA
```

We extend this with RhombiLoRA:
```
Phase 1 (full_noise):  Unified RhombiLoRA warmup → bridge learns base coupling
Phase 2 (high_noise):  Fork → expert bridge diverges toward motion structure
Phase 3 (low_noise):   Fork → expert bridge diverges toward texture structure
Merge:                 Bridge-aligned merge → face symmetry constrains alignment
```

**The face symmetry mechanism:**

The RD's 12 faces form 6 direction pairs. Each expert's bridge is a 6×6
matrix that learns its own coupling pattern. After separate training:

- High-noise bridge: coupling reflects coarse compositional structure
- Low-noise bridge: coupling reflects fine textural structure

The SHARED topology (both are 6×6 with the same direction-pair indexing)
means the bridges are expressed in the same geometric basis. Standard
`merge_experts()` concatenates state dicts with different prefixes. We
extend this: the bridge matrices provide an alignment signal.

**Bridge-aligned merge options:**

1. **Direct merge (baseline):** Concatenate as Flimmer does today. Each
   expert keeps its own bridge. The shared indexing provides implicit
   alignment.

2. **Averaged bridge:** Compute the Fréchet mean of the two expert bridges.
   Use this as a shared bridge for both experts at inference. Tests whether
   the geometric structure is complementary or redundant.

3. **Joint fine-tune:** After separate training, freeze the base LoRA params
   and fine-tune ONLY the bridges jointly. The bridges learn to coordinate
   through shared gradient flow. This is the "hum" — the resonance between
   two systems sharing geometric structure.

**Metrics:**
- FVD (Fréchet Video Distance) of generated videos vs ground truth
- Expert utilization balance during inference
- Bridge Fiedler of individual and merged bridges
- Cross-spectral density between expert gradient trajectories during
  joint fine-tune phase

**Integration with Flimmer:**

The changes to Flimmer are minimal:
- `LoRAState` gets a `topology` field: `'linear'` | `'rhombic'`
- `fork()` deep-copies the bridge along with A/B matrices
- `merge_experts()` gains an `alignment` parameter: `'none'` | `'average'` | `'joint'`
- `build_parameter_groups()` adds bridge params at bridge-specific LR
- No changes to the training loop — RhombiLoRALinear is a drop-in replacement
  for nn.Linear in the LoRA injection layer

**Phase 4 — Paired Bridge Training (the Marey extension):**

After separate expert training (Phases 2-3) and initial merge, add a
paired training phase that teaches the bridges to coordinate:

```yaml
# Extend Flimmer project.yaml
phases:
  # ... Phases 1-3 as above ...
  - type: full_noise
    name: "Paired Bridge Alignment"
    overrides:
      use_control_data: True
      # train_data = final high-quality outputs
      # control_data = intermediate expert representations
      # ONLY bridge params are trainable (freeze A/B)
      freeze_lora_ab: true
      task_id: ['expert_coordination', 'override']
```

This is the Peter Rabbit controlnet move: after the model knows what each
expert does independently, teach the bridge HOW the experts should talk to
each other using paired examples of "this is what one expert produces" →
"this is what the coordinated output should look like."

**Compute:** Wan 2.2 14B fits on the A6000 (48GB) with gradient checkpointing
and bf16. Flimmer already handles this. No cloud compute needed for 3A.

---

### 3B: Cross-Modal Transit (LLM × Image Model)

**Question:** When two models in different modalities each carry a
RhombiLoRA, does the shared geometric structure produce measurable
resonance in joint gradient dynamics?

**Architecture:**

```
                    Shared 6-channel space
                         ╔═══════╗
                         ║ FACE  ║
    ┌──────────┐         ║ ALIGN ║         ┌──────────┐
    │ Qwen 7B  │──bridge─╢       ╟─bridge──│ Flux Dev │
    │ + RhombiLoRA      ║       ║          │ + RhombiLoRA
    └──────────┘         ╚═══════╝         └──────────┘
       Text                                   Image
```

**Phase 1 — Independent training (character LoRA equivalent):**
- Qwen 7B + RhombiLoRA on geometric reasoning dataset (see §Dataset)
- Flux Dev + RhombiLoRA on image generation (style/concept LoRA)
- Each model learns its own bridge coupling pattern
- This is the Peter Rabbit Stage 1: learn the domain first

**Phase 2 — Paired bridge training (controlnet equivalent):**
- Freeze both base models and both LoRA A/B matrices
- Connect the two RhombiLoRAs through their shared 6-channel space
- Fine-tune ONLY the bridges on paired text→image data:
  - `train_data/` = target images (what Flux should produce)
  - `control_data/` = text descriptions (what Qwen understands)
- The bridge geometry is the custom skeleton — it constrains how the
  text→image mapping is learned through the 6 FCC channels

**Phase 3 — Transit measurement:**
- Record bridge gradient trajectories during Phase 2
- Measure cross-spectral density of bridge gradient trajectories

**Control condition:**
- Same setup but with standard LoRA (no bridge) + a learned linear
  projection layer between the two LoRAs. Match total parameter count.

**The hum detection protocol:**

During joint bridge training, record the gradient of each bridge matrix
at every step. This produces two time series of 6×6 matrices:
- G_text(t) = ∂L/∂bridge_text at step t
- G_image(t) = ∂L/∂bridge_image at step t

Flatten each to 36-dimensional vectors. Compute cross-spectral density:
```python
from scipy.signal import csd
freqs, Pxy = csd(G_text_flat, G_image_flat, fs=1.0, nperseg=256)
```

If the shared geometric structure produces resonance, Pxy should show peaks
at frequencies corresponding to the RD's eigenvalues (known from Paper 2's
spectral analysis: the Laplacian eigenvalues of the 12-node RD graph).

**Compute:**
- Qwen 7B: A6000 (48GB) — fits easily, proven by Exp 2
- Flux Dev: A6000 (48GB) — fits with quantization (~12GB)
- Joint training: requires both models loaded simultaneously. At ~20GB
  each with INT8 quantization, this is tight on 48GB but potentially
  feasible with aggressive memory management. Alternatively, use gradient
  checkpointing + offloading, or move to cloud for the joint phase.
- If A6000 can't hold both: RunPod H100 80GB (~$3.50/hr) for the joint phase.
  ~8h of joint training = ~$30. Very reasonable.

---

### 3C: Voronoi Cell MoE — The Large Geometric Model (LGM)

**Question:** Can the RD tessellation provide the routing architecture for
a Mixture of Experts model where expert communication follows the FCC
lattice?

**The vision:**

Standard MoE routing: a gating network selects top-k experts from a flat
pool. Experts don't talk to each other. Communication happens only through
the shared backbone.

Geometric MoE routing: experts are arranged on the vertices of a lattice.
Each expert communicates with its geometric neighbors through shared faces.
Routing = selecting a region of the lattice, not a flat subset.

```
Standard MoE:      Token → Gate → top-k experts → sum → output
                   (experts are independent, unstructured)

Geometric MoE:     Token → Spatial projection → lattice region → neighbor
                   communication through shared faces → output
                   (experts form a tessellated space)
```

**The RD tessellation as MoE architecture:**

The FCC lattice's Voronoi cells are rhombic dodecahedra. Each RD has 12
faces shared with 12 neighbors. In the MoE interpretation:

- Each Voronoi cell = one expert
- Each face = a communication channel between adjacent experts
- The 6 direction pairs of each face correspond to the 6 channels of
  the RhombiLoRA bridge

An 8-expert LGM would place experts at the 8 vertices of a cube (the
trivalent vertices of the RD — the SAME geometry that encodes the Laws).
Each expert connects to 3 neighbors (trivalent). A 14-expert LGM uses
all vertices of the RD (8 trivalent + 6 tetravalent).

**Routing mechanism:**

Input features are projected onto the 6 directional axes of the FCC lattice
(the same direction-pair decomposition from Paper 2). The projection
magnitudes determine which region of the lattice is activated — not top-k
from a flat pool, but a spatial neighborhood in the tessellated space.

```python
# Geometric routing
directions = fcc_direction_pairs()  # (6, 3) — canonical direction vectors
projections = input @ directions.T   # (batch, 6) — directional components
# Activate the Voronoi cell whose center is nearest in 6D projection space
activated_cell = voronoi_nearest(projections, cell_centers)
# Route through the cell AND its face-sharing neighbors
expert_outputs = [experts[i](input) for i in activated_cell.neighborhood]
# Weight by face-sharing proximity (RhombiLoRA bridge weights)
output = bridge_weighted_sum(expert_outputs, activated_cell.bridges)
```

**Sphere packing connection:**

FCC is the densest sphere packing in 3D (Kepler conjecture, Hales 2005).
Each sphere touches 12 neighbors. The Voronoi cells of this packing ARE
rhombic dodecahedra. The LGM uses this:

- Maximum density = maximum expert coverage of the representation space
- 12-fold coordination = each expert has 12 communication channels
- Face-sharing = shared boundary between expert domains
- Tessellation = no gaps in coverage (every point in representation space
  belongs to exactly one expert)

**This is the architectural thesis:** cubic lattice MoE (standard) has
6-fold coordination. FCC lattice MoE has 12-fold coordination. Paper 2
proved the connectivity advantage. The LGM makes it the architecture.

**Naming:**

| Name | Expansion | Emphasis |
|------|-----------|---------|
| **LGM** | Large Geometric Model | The geometry IS the architecture |
| **LRM** | Large Rhombic Model | The specific geometry (RD/FCC) |
| **GeoPack** | Geometric Packing Model | The sphere packing connection |

Timothy to decide. "LGM" has the best ring and the clearest thesis.

**Compute:** This is beyond A6000 territory. A meaningful LGM experiment
needs:
- 8-14 experts, each a small transformer (1-3B params)
- Total: 8-42B params
- Training: multi-GPU (4-8× H100 80GB or equivalent)
- Estimated cost on RunPod: 8× H100 cluster for 48h = ~$1,344
  (at $3.50/hr/GPU)

This is grant/sponsorship territory, not personal GPU territory. See §Compute
Infrastructure below.

---

## Dataset Architecture: Teaching Geometry Through Data

### The Problem

Exp 2's co-planar signal collapsed because Alpaca-cleaned is isotropic.
The bridge learns generic mixing because the data rewards generic mixing.
To teach directional structure, the data must CONTAIN directional structure.

### Design Principles

1. **Multi-perspective reasoning maps to direction pairs.** Each of the 6
   direction pairs corresponds to a reasoning axis. A task that requires
   integrating multiple perspectives teaches the bridge to couple specific
   channels.

2. **Expert-decomposable tasks map to MoE routing.** Tasks where different
   subtasks naturally belong to different experts teach the model to route
   through specific cells.

3. **Cross-domain synthesis maps to transit.** Tasks requiring translation
   between domains (code↔language, math↔intuition, text↔image description)
   teach the bridge to align modalities through shared faces.

4. **Symmetry-structured tasks map to face symmetry.** Tasks with internal
   symmetry groups (palindromes, group theory, geometric transformations)
   teach the bridge to recognize and leverage symmetry.

### Dataset Components

#### Component 1: Multi-Perspective Reasoning (for co-planar signal)

Tasks that require explicit reasoning from 2-6 different perspectives,
then synthesis. The perspectives map to direction pairs.

**Format:**
```json
{
  "instruction": "Analyze this situation from the perspectives of [6 roles].
                  Then synthesize a recommendation that accounts for all six.",
  "perspectives": ["engineering", "ethics", "economics", "user_experience",
                    "legal", "environmental"],
  "input": "[scenario]",
  "synthesis": "[integrated recommendation]"
}
```

**Sources:**
- Existing multi-stakeholder decision-making datasets
- Generated scenarios requiring 6-way perspective integration
- Cross-disciplinary review tasks (e.g., "evaluate this proposal from
  technical, financial, legal, ethical, environmental, and user perspectives")

**Expected effect:** The 6 perspectives create natural directional structure
in the gradient. The bridge should learn to couple perspectives that
frequently co-occur in synthesis (co-planar) more strongly than those that
are independent (cross-planar).

#### Component 2: Expert-Decomposable Tasks (for MoE routing)

Tasks with explicit subtask decomposition where different subtasks require
different capabilities.

**Format:**
```json
{
  "instruction": "Solve this problem.",
  "decomposition": [
    {"subtask": "parse the mathematical expression", "expert": "formal"},
    {"subtask": "identify the physical intuition", "expert": "spatial"},
    {"subtask": "compute the numerical result", "expert": "arithmetic"}
  ],
  "input": "[problem]",
  "output": "[solution integrating all subtasks]"
}
```

**Sources:**
- Multi-step math/physics problems with explicit step decomposition
- Programming tasks requiring design + implementation + testing
- Research synthesis requiring literature review + analysis + writing

#### Component 3: Cross-Domain Transit (for bridge alignment)

Paired examples where the SAME concept is expressed in different
representational formats. The bridge should learn to transit between
them through shared structure.

**Format:**
```json
{
  "domain_a": "The derivative measures instantaneous rate of change.",
  "domain_b": "def derivative(f, x, h=1e-7): return (f(x+h) - f(x)) / h",
  "domain_c": "A ball rolls faster and faster down a hill — the derivative
               tells you how fast 'faster' is getting.",
  "transit_task": "Given the mathematical definition, produce both the code
                   implementation and the intuitive explanation."
}
```

**Sources:**
- Math concept ↔ code implementation ↔ intuitive explanation triples
- Scientific paper abstract ↔ methodology ↔ lay summary
- Architecture diagram ↔ code ↔ natural language description

#### Component 4: Geometric Reasoning (for geometric awareness)

Tasks requiring explicit spatial, topological, or geometric reasoning.
These teach the model that geometry is a first-class reasoning domain.

**Sources:**
- Graph theory problems (connectivity, paths, coloring)
- Spatial reasoning benchmarks
- Topology puzzles
- Symmetry group identification
- Tessellation and packing problems

#### Component 5: Symmetry-Structured Data (for face symmetry learning)

Tasks with explicit symmetry properties that mirror the RD's symmetry group.

**Sources:**
- Palindromic structures (linguistic, mathematical)
- Group theory exercises
- Transformation-invariant tasks (rotational, reflective, translational)
- Music theory (inversions, retrogrades, transpositions)

### Dataset Size Estimates

| Component | Examples | Source Strategy |
|-----------|---------|----------------|
| Multi-perspective | 10,000 | Generated + curated |
| Expert-decomposable | 15,000 | Existing benchmarks + augmentation |
| Cross-domain transit | 8,000 | Paired corpus construction |
| Geometric reasoning | 5,000 | Existing benchmarks (GSM-Hard, MATH, spatial) |
| Symmetry-structured | 3,000 | Generated |
| **Total** | **~41,000** | |

Comparable in size to Alpaca-cleaned (52K). Can be mixed with Alpaca at
ratios from 10% (supplement) to 100% (replacement) to measure the effect
of geometric data density on bridge learning.

### Validation: Does the Data Teach the Geometry?

**Prediction:** Training RhombiLoRA on the geometric dataset should produce:
1. Higher Fiedler values than Alpaca training (the bridge finds more structure)
2. Stronger co-planar/cross-planar differentiation (the 6 perspectives
   create directional signal)
3. The co-planar preference should match the data's correlation structure
   (perspectives that frequently co-occur → co-planar coupling)

**Test:** Train identical RhombiLoRA configs on (a) Alpaca, (b) geometric
dataset, (c) 50/50 mix. Compare bridge evolution curves.

---

## Compute Infrastructure

### What Fits on the A6000 (48GB)

| Experiment | VRAM | Feasible? |
|------------|------|-----------|
| 3A: Wan 2.2 MoE RhombiLoRA (single expert) | ~24-32GB | Yes |
| 3A: Wan 2.2 merged inference | ~28-36GB | Yes (INT8) |
| 3B: Qwen 7B RhombiLoRA training | ~22GB | Yes (proven Exp 2) |
| 3B: Flux Dev RhombiLoRA training | ~14GB | Yes |
| 3B: Joint bridge fine-tune (both loaded) | ~34-40GB | Tight, likely yes with offloading |
| 3C: LGM (8 experts × 1.5B) | ~96GB | No |
| 3C: LGM (8 experts × 3B) | ~192GB | No |

### Cloud Compute Options

| Provider | GPU | VRAM | $/hr | Best For |
|----------|-----|------|------|----------|
| RunPod | H100 SXM | 80GB | ~$3.50 | Single-GPU overflow |
| RunPod | 8× H100 | 640GB | ~$28 | LGM training |
| Lambda Labs | H200 | 141GB | ~$3.50 | Large single-model |
| Lambda Labs | 8× H200 | 1.1TB | ~$28 | Full LGM |
| fal.ai | Serverless A100 | 80GB | per-second | Inference, not training |
| Replicate | Various | Various | per-second | Inference, not training |
| **NVIDIA DGX Cloud** | H100 clusters | — | Sponsorship | Full LGM development |
| **Google TPU Research** | TPU v5e | — | Research credits | Alternative to GPU |

### Sponsorship Strategy

The LGM concept is genuinely novel — sphere-packing-derived MoE architecture
with face-sharing expert communication. This has:

1. **Academic publishability:** Novel architecture with geometric theory
2. **Industry relevance:** MoE routing is a hot topic (Mixtral, DeepSeek)
3. **Clear differentiation:** Nobody else is using lattice topology for routing
4. **Measurable claims:** Fiedler value, connectivity ratio, the hum

**Potential sponsors:**
- **NVIDIA Inception Program** — Promptcrafted is already eligible as a startup.
  DGX Cloud credits for research.
- **Google Cloud Research Credits** — TPU access for academic research.
  The geometric MoE paper would qualify.
- **Lambda Labs Research Program** — H200 cluster time for novel architectures.
- **Hugging Face** — Community compute grants for open-source research.
  rhombic is already on PyPI and HF Spaces.
- **Direct sponsorship from MoE researchers** — DeepSeek, Mistral, etc.
  might be interested in geometric routing research.

**The pitch:** "We have Papers 1-2 proving FCC lattice provides 2-6×
connectivity advantage over cubic. We have Exp 1-2 proving this translates
to learned cross-channel coupling in LoRA adapters. The next step is
geometric MoE — sphere-packing-derived expert routing with face-sharing
communication. We need H100/H200 cluster time to train the first LGM."

---

## Implementation Sequence

### Phase A: Foundation (Weeks 1-2, A6000 only)

1. Build the geometric training dataset (Components 1-5)
   - Component 3 uses the Marey paired format: `train_data/` + `control_data/`
   - Generate paired transit data (math↔code, concept↔image-description)
2. **Exp 2.5:** Re-run Exp 2 with geometric dataset vs Alpaca
   - Same configs (FCC r24, cubic r24, standard r24)
   - Measure whether co-planar signal strengthens with geometric data
   - If co-planar ratio goes from 1.02× to >1.10×, the data is teaching
     directional structure — proceed to 3A-3C
3. Integrate RhombiLoRALinear into Flimmer's LoRA injection layer
   - Add `topology` field to LoRAState
   - Extend `fork()` and `merge_experts()` for bridge handling
   - Add `freeze_lora_ab` option for paired bridge-only training phases
   - Add bridge-specific parameter groups in `build_parameter_groups()`
   - Implement `use_control_data: True` equivalent for bridge pair alignment

### Phase B: MoE Expert Bridge (Weeks 3-4, A6000)

4. **Exp 3A:** Wan 2.2 MoE with RhombiLoRA
   - Phase 1: unified warmup
   - Phase 2-3: expert fork with bridge divergence tracking
   - Merge: compare direct, averaged, and joint bridge fine-tune
   - Measure FVD improvement over standard LoRA MoE

### Phase C: Cross-Modal Transit (Weeks 5-6, A6000 + possibly cloud)

5. **Exp 3B:** Qwen RhombiLoRA + Flux Dev RhombiLoRA
   - Independent training on matched domain data
   - Joint bridge fine-tune on paired text→image tasks
   - Hum detection via cross-spectral density
   - Compare to linear projection control

### Phase D: LGM Design + Sponsorship (Weeks 7-8)

6. Finalize LGM architecture design
7. Write sponsorship proposals for compute access
8. If compute available: prototype 8-expert LGM on H100 cluster
9. **Exp 3C:** Geometric MoE vs standard MoE on matched benchmark

### Phase E: Paper (Weeks 9-10)

10. Write Paper 3 covering all experiments
11. Include dataset as an open-source contribution
12. Release updated rhombic library with nn.LGM module

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Geometric dataset doesn't strengthen co-planar signal | Medium | High | Test with Exp 2.5 before committing. Multiple component designs provide fallbacks. |
| Joint bridge fine-tune doesn't produce the hum | Medium | High | The hum is the most speculative prediction. Failure here is publishable as a null result. Bridge alignment still works without resonance. |
| Wan 2.2 + RhombiLoRA doesn't improve FVD | Low | Medium | FVD might not capture the bridge benefit. Add qualitative evaluation (human study). |
| Can't get cloud compute for LGM | Medium | Medium | LGM is Phase D — Papers 1-3 stand without it. Sponsorship strategy has multiple paths. |
| Flimmer integration introduces bugs | Low | Low | Flimmer has test infrastructure. RhombiLoRALinear is a drop-in replacement. |

---

## Relationship to the Three Streams

**Stream 1 (Corpus):** The 24 corpus values remain the weight set for
validation experiments. The prime-vertex mapping (p=0.000025) validates
that the RD geometry has non-accidental affinity with the corpus. The
geometric training dataset extends this: teaching a model to reason
geometrically is teaching it the structure the corpus encodes.

**Stream 2 (Astrolabium):** The 12-fold temporal architecture maps to
the 12 faces of the RD. The LGM's Voronoi cell MoE is the computational
instantiation of the Astrolabium's temporal navigation — each cell a moment,
each face a transit between moments.

**Stream 3 (Three Responses):** "The practitioner inside a cocoon exits
through one of twelve identical mirrors and emerges through the
corresponding face of an adjacent cell." The LGM IS this sentence. Each
expert is a cell. Each face is a mirror. Transit is routing.

---

## Provenance: The Paired Training Pattern

The paired training paradigm documented here originates from production
work by Timothy Paul Bielec and Minta Carlson (Promptcrafted LLC /
Alvdansen Labs) for Moonvalley's Marey video model, February 2025. The
Peter Rabbit character LoRA + controlnet pairing demonstrated that a
two-stage strategy (learn the domain, then learn the mapping) produces
superior control alignment compared to joint training from scratch.

Raw configs and documentation:
`C:\Users\Timothy Paul Bielec\Desktop\MV Assets\Paired_Training\`

The extension to geometric topology (RhombiLoRA bridge as structured
controlnet) is [ANALYTICAL CONTRIBUTION] — the insight that the bridge
matrix is the geometric analog of the controlnet mapping, and that FCC
topology constrains this mapping in ways that generic learned projections
cannot.

---

*Design document drafted March 8, 2026. Updated March 8: paired training
paradigm section added from Promptcrafted/Moonvalley production work.
Built on Exp 1-2 findings, Flimmer infrastructure, Marey paired training,
and the three Falco research streams.*
