# Jamba (Mamba-Transformer hybrid — RMSNorm stabilization precedent)


**Institution:** AI21 Labs  
**Tags:** nemotron-engineering, mamba-hybrid, rmsnorm, peripheral

## Program assessments

### From `C:\falco\rhombic\competition\RESEARCH_GRPO.md`

**Verdict:** architectural-precedent · **Threat:** none · **Confidence:** medium

**RMSNorm is used throughout.** Jamba (a prior Mamba-Transformer hybrid) documented
that Mamba layers suffer from large activation values causing loss spikes, and RMSNorm
on internal activations was required to stabilize training. This is likely why the
`rmsnorm_fn` patch matters.
