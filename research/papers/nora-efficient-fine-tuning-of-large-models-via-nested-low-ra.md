# NoRA: Efficient Fine-Tuning of Large Models via Nested Low-Rank Adaptation

> [link](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Efficient_Fine-Tuning_of_Large_Models_via_Nested_Low-Rank_Adaptation_ICCV_2025_paper.pdf)

**Authors:** Li et al. (10 authors)  
**Institution:** HKUST (Hong Kong)  
**Venue:** ICCV 2025  
**Date:** 2025  
**Tags:** bridge-lineage, chinese-institution, nested-lora

## Program assessments

### From `C:\falco\rhombic\orvad-research-non-english-lora.md`

**Verdict:** Bridge-matrix lineage member (nested inner LoRA, frozen outer); shows Chinese community engagement with middle-matrix architecture · **Threat:** MEDIUM (as lineage, claim 1) · **Cite in:** competitive_landscape.md lineage table (MEDIUM priority) · **Confidence:** HIGH

Five papers now place a trainable matrix between frozen A and B projections, forming a recognizable research lineage. None keeps A and B learnable; none constrains the bridge geometrically.
---
| **NoRA** | Li et al. (10 authors) | HKUST (Hong Kong) | ICCV 2025 | Inner LoRA (serial) | Frozen outer (AwSVD) | Nested low-rank, no geometric |
---
**TeLoRA differentiation:** All five freeze A and B. TeLoRA trains A, B, and bridge jointly, meaning the bridge captures emergent coupling patterns rather than serving as a compressed proxy for what frozen projections would do. The bridge-as-diagnostic (task fingerprinting, overfitting detection) is unique to TeLoRA precisely because A and B adapt alongside it.
---
**Must-cite:** NoRA (recommend citing — HKUST, shows Chinese community engagement with this architecture).
---
| **NoRA** (Li et al., ICCV 2025, HKUST) | **MEDIUM** | Chinese institution, nested LoRA, shows awareness of middle-matrix concept |
---
**Expand the LoRA-XS lineage table** in competitive_landscape.md to include EDoRA and NoRA.
