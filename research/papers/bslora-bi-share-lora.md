# BSLoRA (Bi-Share LoRA)

> [link](https://icml.cc/virtual/2025/poster/45733)

**Authors:** Yuhua Zhou, Ruifeng Li, Changhai Zhou, Fei Yang, Aimin Pan  
**Institution:** Zhejiang University, Zhejiang Lab, Fudan University  
**Venue:** ICML 2025  
**Date:** 2025  
**Tags:** chinese-institution, parameter-sharing

## Program assessments

### From `C:\falco\rhombic\orvad-research-non-english-lora.md`

**Verdict:** Parameter sharing mechanism, not coupling; no bridge, no spectral structure — different axis from TeLoRA · **Threat:** Distant · **Cite in:** optional (MEDIUM priority in recommendations table) · **Confidence:** HIGH (via ICML poster)

Intra-layer and inter-layer parameter sharing. Uses 44.59% of LoRA parameters for comparable/better performance. Sharing mechanism, not coupling. No bridge, no spectral structure.
---
**Threat level:** All distant. BSLoRA (sharing), SMoA (modulation), StructLoRA (inter-layer coordination), CLoRA (subspace regulation) — none addresses intra-bridge topology.
---
**Must-cite:** BSLoRA and SMoA are optional.
---
**Parameter sharing across LoRA modules** (BSLoRA, Zhejiang/Fudan) — reducing parameters through intra/inter-layer sharing. Different axis from TeLoRA but active area.
---
| **BSLoRA** (Zhou et al., ICML 2025, Zhejiang/Fudan) | **MEDIUM** | Chinese institution, structured LoRA at top venue |
---
References BSLoRA for block-structured approaches.
