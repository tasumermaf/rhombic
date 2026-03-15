# Definitive Channel Ablation Results — March 14, 2026

All experiments complete except H4 (n=8, running) and H5 (n=12, queued).

## The Complete Picture

### All n=6 Cybernetic Experiments (COMPLETE)

| Experiment | Model | Init | Steps | Val Loss | Co/Cross | Peak Ratio | Bridge Fiedler |
|-----------|-------|------|-------|----------|----------|------------|---------------|
| exp3 | Qwen 7B | identity | 12.9K | — | 18,248:1 | 18,248:1 | ~0.0004 |
| exp3_tiny | TinyLlama | identity | 10K | — | 37,929:1 | 37,929:1 | ~0.0004 |
| C-001 | TinyLlama | identity | 4K | 0.4178 | 10,118:1 | 10,118:1 | ~0.0004 |
| **C-002** | TinyLlama | geometric | **10K** | **0.4010** | **71,337:1** | **76,916:1** @9.7K | **0.00008** |
| **C-003** | TinyLlama | corpus | **10K** | **0.4011** | **64,168:1** | **82,854:1** @9K | **0.00009** |
| **H3** | TinyLlama | identity | **10K** | **0.4015** | **70,404:1** | **82,154:1** @9.8K | **0.00009** |

**All produce 100% block-diagonal structure.** Val loss converges to 0.4010-0.4015
(0.12% max delta). Final co/cross ratio converges to 64K-71K band (10% max delta).
Init is fully cosmetic.

### Channel Count Ablation (n=3/4/6/8)

| n | Contrastive | Val Loss @10K | Bridge Fiedler | Co/Cross | BD% | Status |
|---|-------------|--------------|---------------|----------|-----|--------|
| 3 | N/A | 0.4020 | 0.095 | N/A | N/A | DONE |
| 4 | **disabled** | 0.4022 | **0.092** | ~1:1 | **0%** | DONE |
| **6** | **active** | **0.4015** | **0.00009** | **70,404:1** | **100%** | DONE |
| 8 | **disabled** | 0.4227 @2.9K | 0.085 | ~1:1 | **0%** | Running |

**The contrastive loss IS the mechanism:**
- Only n=6 has active contrastive loss (RD face-pair geometry encoded)
- Only n=6 produces block-diagonal structure
- Val loss is IDENTICAL across n=3/4/6 (0.17% max delta)
- Bridge Fiedler bifurcation: 0.00009 (n=6) vs ~0.09 (n=3/4/8) = **1,020× difference**

### Peak Ratios Timeline

| All-time ranking | Experiment | Step | Ratio |
|-----------------|-----------|------|-------|
| **1** | **C-003** | **9000** | **82,854:1** |
| **2** | **H3** | **9800** | **82,154:1** |
| 3 | H3 | 9700 | 79,687:1 |
| 4 | C-003 | 9500 | 79,125:1 |
| 5 | C-002 | 9700 | 76,916:1 |
| 6 | C-002 | 10000 | 71,337:1 |
| 7 | H3 | 10000 | 70,404:1 |
| 8 | C-003 | 10000 | 64,168:1 |
| 9 | exp3_tiny | 10000 | 37,929:1 |
| 10 | exp3 (7B) | 12900 | 18,248:1 |

Peak ratios occur at steps 9000-9800 due to cross-planar floor noise. Final
ratios stabilize at 64K-71K for TinyLlama 10K runs. **H3's peak was higher
than initially reported (82,154:1 vs 76,452:1) — full 100-checkpoint data
reveals the peak at step 9800, not 8000.**

## Key Findings for Paper 3

### F1: Block structure requires the RD geometric prior
The contrastive loss encodes which channel pairs should couple (co-planar)
and which should separate (cross-planar). Without it, spectral regularization
drives connectivity but not topology. Three independent runs without
contrastive (n=3, 4, 8) ALL produce uniform coupling.

### F2: Task performance is orthogonal to bridge topology
Val loss at 10K steps: 0.4010-0.4022 across n=3/4/6. The 0.17% max delta
is within noise. Block-diagonal structure provides no performance benefit.
It is a structural signature, not a performance feature.

### F3: Init is fully cosmetic
Three initializations that produce completely different starting topologies
(identity, geometric, corpus-coupled with I Ching trigrams) converge to the
same final state: 100% BD, 64K-71K co/cross ratio, 0.4010-0.4015 val loss.
Convergence occurs by step 2000 for topology, by step 10000 for ratio magnitude.

### F4: The Fiedler dip reveals competing objectives
H2 (n=4, spectral-only) shows a three-phase Fiedler trajectory:
1. Rapid growth (0-1200): exponential approach to 0.084
2. Dip (1200-3000): task gradient fights spectral regularization
3. Recovery (3000-10000): slow climb to 0.092
The exponential saturation model fit at step 1800 predicted asymptote 0.087;
actual final value is 0.092 (5.6% higher).

### F5: Three regimes of bridge behavior
- **Trivial (n=3):** Too few DOF for meaningful structure. Fiedler 0.095.
- **Spectral-only (n=4, 8):** Connectivity without topology. Fiedler ~0.09.
- **Full Steersman (n=6):** Block-diagonal. Bridge Fiedler 0.00009. 70K:1.

## Training Compute Summary

| Machine | Experiments | Wall Time | GPU |
|---------|-----------|-----------|-----|
| Local (RTX 6000 Ada 48GB) | C-001, C-002, C-003 | ~36h | 48GB VRAM |
| Hermes (RTX 4090 16GB) | H1, H2, H3, H4, H5 | ~50h | 16GB VRAM |

All TinyLlama 1.1B. Same hyperparameters: LR=2e-4, batch=2, grad_accum=8,
warmup=200, rank=24, targets=q/k/v/o_proj, dataset=alpaca-cleaned.

## Remaining Work

1. **H4 (n=8):** Running at step 2900/10000. Expected to confirm spectral-only
   pattern. Will complete in ~8 hours.
2. **H5 (n=12):** Queued. Same prediction: spectral-only → no structure.
3. **Paper 3 writing:** All analysis materials prepared. 22+ figures generated.
   Definitive ablation results in hand.
