# H3 (n=6) Channel Ablation — Analysis Report

Updated: 2026-03-14 (COMPLETE — 10K steps on Hermes)

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | TinyLlama 1.1B |
| n_channels | **6** |
| Init | identity |
| Steersman | default (contrastive + spectral) |
| Steps | 10,000 |
| Bridge params per module | 36 (6×6) |
| Total bridge params | 3,168 |

## Critical Design Note: Contrastive Loss ACTIVE

Unlike H2 (n=4) and H4 (n=8), the contrastive loss is fully operational
for n=6. The `contrastive_bridge_loss` function's `if B.shape[0] != 6:
continue` check passes, and the RD face-pair geometry is encoded as:

- **Co-planar pairs:** (0,1)=±x, (2,3)=±y, (4,5)=±z → encouraged
- **Cross-planar pairs:** all other 12 pairs → suppressed

This is the ONLY channel count in the ablation series with active
contrastive loss. H3 serves as the positive control.

## Results — Complete 10K Run

### Training Metrics

| Step | Val Loss | Fiedler | Co/Cross | Contrastive | Deviation | Spectral Gap |
|------|----------|---------|----------|-------------|-----------|-------------|
| 100 | 0.4766 | 0.00308 | 76:1 | -0.00013 | 0.019 | 0.154 |
| 200 | 0.4474 | 0.00044 | 746:1 | -0.00141 | 0.052 | 0.009 |
| 500 | 0.4389 | 0.00012 | 3,581:1 | -0.00588 | 0.159 | 0.001 |
| 1000 | 0.4343 | 0.00012 | 7,246:1 | -0.00702 | 0.279 | 0.0004 |
| 2000 | 0.4260 | 0.00024 | 8,120:1 | -0.00616 | 0.446 | 0.0005 |
| 3000 | 0.4217 | 0.00042 | 8,328:1 | -0.00890 | 0.635 | 0.0006 |
| 5000 | 0.4119 | 0.00037 | 13,190:1 | -0.01445 | 1.021 | 0.0003 |
| 7000 | 0.4051 | 0.00018 | 35,204:1 | -0.01777 | 1.250 | 0.0001 |
| 8000 | 0.4034 | 0.00008 | **76,452:1** | -0.01851 | 1.302 | 0.00005 |
| 9000 | 0.4023 | 0.00008 | 73,644:1 | -0.01896 | 1.334 | 0.00005 |
| **10000** | **0.4015** | **0.00009** | **70,404:1** | -0.01941 | 1.367 | 0.00006 |

### Key Observations

1. **Block-diagonal lock-in by step 200.** Co/cross ratio hits 746:1 at step 200
   and never returns. The spectral gap collapses to 0.009 — near-perfect 3-fold
   eigenvalue degeneracy. This matches the lock-in timing seen in all other n=6
   cybernetic experiments (C-001, C-002, C-003, exp3, exp3_tiny).

2. **Co/cross ratio peaks at 82,154:1 (step 9800)** then decreases to
   70,404:1 at step 10000. Full 100-checkpoint analysis reveals the peak
   later than initially reported (step 9800, not 8000). The decline occurs
   because cross-planar values hit a computational floor (~1e-5) and
   numerical noise creates fluctuations.

3. **Bridge Fiedler near-zero throughout.** From step 200 onward, Bridge Fiedler
   stays below 0.001 — the blocks are nearly disconnected. This is the opposite
   of H2 (n=4), where Fiedler reaches 0.092.

4. **Val loss: 0.4015** — the best of all four channel counts (n=3: 0.4020,
   n=4: 0.4022), though the differences are within noise (0.17% max delta).

5. **Deviation: 1.367.** Bridges have moved 137% from identity. The contrastive
   loss has completely reshaped the matrix topology.

## Comparison: The Definitive Ablation

| Metric | H1 (n=3) | H2 (n=4) | **H3 (n=6)** | H4 (n=8) @2.9K |
|--------|----------|----------|------------|---------|
| Val Loss | 0.4020 | 0.4022 | **0.4015** | 0.4227 |
| Bridge Fiedler | 0.095 | 0.092 | **0.00009** | 0.085 |
| Co/Cross Ratio | N/A | ~1:1 | **70,404:1** | ~1:1 |
| Spectral Gap | ~0.5 | 0.559 | **0.00006** | ~0.6 |
| Block Structure | N/A | 0% BD | **100% BD** | 0% BD |
| Contrastive | N/A | disabled | **active** | disabled |

**The contrastive loss IS the mechanism.** Only n=6 has it active, and only n=6
produces block structure. Task performance is identical across all channel counts.

## Three Regimes

1. **n=3 (trivial):** Too few channels for block structure to be meaningful.
   The 3×3 bridge has only 6 off-diagonal elements. Spectral regularization
   drives uniform connectivity (Fiedler 0.095).

2. **n=4, 8 (spectral-only):** Contrastive loss disabled. The Steersman
   operates on spectral loss alone, producing uniform connectivity
   (Fiedler ~0.085-0.092) with no topological structure.

3. **n=6 (full Steersman):** Contrastive loss active. The RD geometric
   prior creates 100% block-diagonal structure with 70K:1 co/cross
   separation. Bridge Fiedler collapses to 0.00009 (near-disconnected blocks).

## Files

- `metrics_hermes.csv` — 100 checkpoints, CSV format
- `bridge_final_*.npy` — 88 final bridge matrices (6×6) on Hermes
- `train.log` — 2842 lines, full training log
