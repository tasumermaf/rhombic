# CLoRA (contrastive LoRA composition)


**Venue:** ICCV 2025  
**Date:** 2025  
**Tags:** contrastive, diffusion-lora

## Program assessments

### From `C:\falco\rhombic\orvad-research-non-english-lora.md`

**Verdict:** Contrastive objective between adapters (image generation), never within an adapter's internal structure · **Threat:** None · **Confidence:** HIGH (located and characterized)

All contrastive LoRA methods found apply contrastive objectives **between** adapters or **between** adapter and base model, never **within** an adapter's internal structure.
---
**CLoRA** (ICCV 2025): Contrastive composition of multiple LoRAs for image generation
---
TeLoRA's contrastive loss operates **within the bridge** — co-axial vs. cross-axial channel pairs defined by polytope geometry. This is a fundamentally different application of contrastive learning. No prior work applies contrastive objectives to internal adapter structure.
---
**Threat level:** None. Different application entirely.
