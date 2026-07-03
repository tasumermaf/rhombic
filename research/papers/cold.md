# CoLD


**Venue:** KDD 2025  
**Date:** 2025  
**Tags:** contrastive, decoding

## Program assessments

### From `C:\falco\rhombic\orvad-research-non-english-lora.md`

**Verdict:** Contrastive decoding between LoRA-adapted and base model, not within adapter internals · **Threat:** None · **Confidence:** HIGH (located and characterized)

All contrastive LoRA methods found apply contrastive objectives **between** adapters or **between** adapter and base model, never **within** an adapter's internal structure.
---
**CoLD** (KDD 2025): Contrastive decoding between LoRA-adapted and base model distributions
---
TeLoRA's contrastive loss operates **within the bridge** — co-axial vs. cross-axial channel pairs defined by polytope geometry. This is a fundamentally different application of contrastive learning. No prior work applies contrastive objectives to internal adapter structure.
---
**Threat level:** None. Different application entirely.
