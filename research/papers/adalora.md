# AdaLoRA


**Tags:** peripheral, rank-allocation, composable, peft-landscape

## Program assessments

### From `C:/falco/rhombic/docs/competitive_landscape.md`

**Verdict:** Named only as composable/orthogonal method — no citation details given in source · **Threat:** low · **Confidence:** medium

**Composability** — orthogonal to DoRA, AdaLoRA, MoE methods

### From `['C:\\falco\\rhombic\\orvad-research-telora-competitive-intel.md', 'C:\\falco\\rhombic\\orvad-research-telora-competitive-intel-update3.md', 'C:\\falco\\rhombic\\results\\research-scout-2026-03-26.md']`

**Verdict:** Mentioned as an established structured-variant baseline in HuggingFace PEFT and as part of the rank-allocation trajectory

**HuggingFace PEFT library** has no topology-aware LoRA implementations. AdaLoRA, CorDA, PiSSA, MiSS are the structured variants. TeLoRA would be first geometry-based entry.
---
AdaLoRA | Adaptive rank via SVD | Established
---
The field's trajectory is toward (a) better rank allocation (GoRA, IGU-LoRA, AdaLoRA), (b) manifold-aware optimization (OrthoGeoLoRA, StelLA, Stiefel-LoRA), and (c) extreme parameter reduction via frozen outer matrices (LoRA-XS, LoRA-SB, VeRA).
