# LoRA-XS


**Authors:** Banaei et al.  
**Venue:** EMNLP 2024  
**Date:** 2024  
**Tags:** sweep-2026-07, bridge-lineage, claim-1, frozen-outers, frozen-matrices, collaboration-lead, competitive-landscape, peripheral

## Program assessments

### From `docs/LITERATURE_WATCH_2026-07-03.md`

**Verdict:** claim-1 lineage (frozen-outers family) · **Threat:** low · **Cite in:** claim-1 lineage · **Confidence:** medium

MoSLoRA (2406.11909) · StelLA (2510.01938) · AdaLoRA (2303.10512) · BoRA (2508.06953) · Lily (2407.09946, routed A–B coupling) · CeRA (2602.22911, nonlinear) · CRMA (2606.00382) · GraphLoRA-Rec (2606.07526, trainable message-passing in the LoRA pathway, imposed data-graph) · TLoRA/ID-LoRA/LoRA-XS family (frozen outers).

### From `['C:\\falco\\rhombic\\results\\research-scout-2026-03-26.md', 'C:\\falco\\rhombic\\results\\research-scout-2026-03-27.md']`

**Verdict:** Already cited and differentiated; bridge-lineage ancestor (r x r trainable between frozen A/B) · **Threat:** LOW (Claim 1 closest-competitor family) · **Cite in:** Already cited (Papers 3-4 Related Work) · **Confidence:** HIGH

LoRA-XS (EMNLP 2024) | r x r trainable between frozen A/B | None | Already cited. Differentiated.
---
**Key gap in the field:** Every "bridge" paper (LoRA-XS, LoRA-SB, TLoRA) treats the middle matrix as a generic learnable parameter. None constrains it by graph structure, and none uses the bridge as a diagnostic.
---
**Banaei et al.** (LoRA-XS) | Our bridge generalizes their r x r concept. Joint paper comparing frozen-outer vs learnable-outer bridges could be compelling. | LOW (competitive overlap)
---
Semantic Scholar API returned ~60 citing papers. Scanned for 2026 entries. Notable new citers (all from Feb 2026, not overnight) [...] No new March 26-27 papers citing LoRA-XS detected.

### From `C:\falco\rhombic\docs\LITERATURE_WATCH.md`

**Verdict:** Competitive landscape entry

| **LoRA-XS** | r x r dense | Frozen A/B | No | No | No | EMNLP 2024 |
