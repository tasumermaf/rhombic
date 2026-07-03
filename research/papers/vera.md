# VeRA


**Authors:** Kopiczko, Blankevoort, Asano  
**Institution:** Qualcomm + Univ. Amsterdam  
**Venue:** ICLR 2024  
**Date:** 2024  
**Tags:** bridge-lineage, peripheral, peft-landscape, frozen-matrices

## Program assessments

### From `C:\falco\rhombic\orvad-research-non-english-lora.md`

**Verdict:** Bridge-matrix lineage member; frozen random projections with diagonal scaling vectors; already cited · **Threat:** MEDIUM (as lineage, claim 1) · **Cite in:** already cited · **Confidence:** HIGH

Five papers now place a trainable matrix between frozen A and B projections, forming a recognizable research lineage. None keeps A and B learnable; none constrains the bridge geometrically.
---
| **VeRA** | Kopiczko, Blankevoort, Asano | Qualcomm + Univ. Amsterdam | ICLR 2024 | Diagonal (d,b vectors) | Frozen (random) | Diagonal only |
---
**TeLoRA differentiation:** All five freeze A and B. TeLoRA trains A, B, and bridge jointly, meaning the bridge captures emergent coupling patterns rather than serving as a compressed proxy for what frozen projections would do. The bridge-as-diagnostic (task fingerprinting, overfitting detection) is unique to TeLoRA precisely because A and B adapt alongside it.
---
**Must-cite:** VeRA (already cited).

### From `['C:\\falco\\rhombic\\orvad-research-telora-competitive-intel-update3.md', 'C:\\falco\\rhombic\\results\\research-scout-2026-03-26.md']`

**Verdict:** Mentioned as an established PEFT method and as part of the extreme-parameter-reduction trajectory (frozen outer matrices)

VeRA | Shared random matrices + per-layer scaling | Established
---
The field's trajectory is toward (a) better rank allocation (GoRA, IGU-LoRA, AdaLoRA), (b) manifold-aware optimization (OrthoGeoLoRA, StelLA, Stiefel-LoRA), and (c) extreme parameter reduction via frozen outer matrices (LoRA-XS, LoRA-SB, VeRA).
