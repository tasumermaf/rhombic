# LoRAGen: Structure-Aware Weight Space Learning for LoRA Generation

> [link](https://openreview.net/forum?id=mrafO7aTYj)

**Authors:** Hao Huang et al.  
**Venue:** OpenReview  
**Date:** October 2025  
**Tags:** weight-generation, module-heterogeneity

## Program assessments

### From `C:\falco\rhombic\docs\LITERATURE_WATCH.md`

**Verdict:** MODERATE for Paper 4 · **Cite in:** Paper 4 if discussing module-level heterogeneity. · **Confidence:** MEDIUM -- abstract and architecture details read

**Relevance: MODERATE for Paper 4.** LoRAGen's finding that LoRA weight distributions are module-heterogeneous supports our layer-projection gradient finding (k_proj >> v_proj for BD formation). Their module-aware decoder is motivated by the same observation: different projection types behave differently. Different application (generation vs. training), convergent structural insight.
