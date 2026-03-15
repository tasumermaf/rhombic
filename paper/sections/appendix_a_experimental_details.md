# Appendix A: Experimental Details

## A.1 Common Hyperparameters

All experiments share the following configuration unless otherwise noted:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Base learning rate | $2 \times 10^{-4}$ |
| LR schedule | Cosine decay after linear warmup |
| Warmup steps | 200 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Max gradient norm | 1.0 |
| LoRA rank ($r$) | 24 |
| LoRA $\alpha$ | 16 |
| LoRA dropout | 0.0 |
| Target modules | $W_Q, W_K, W_V, W_O$ |
| Dataset | alpaca-cleaned |
| Max sequence length | 512 |
| Precision | bfloat16 |
| Gradient checkpointing | Enabled |
| Random seed | 42 |
| Steersman feedback interval | 100 steps |

## A.2 Steersman Hyperparameters

| Parameter | Value |
|-----------|-------|
| Initial contrastive weight ($w_c$) | 0.1 |
| Max contrastive weight ($w_c^{\max}$) | 0.5 |
| Initial spectral weight ($w_s$) | 0.05 |
| Max spectral weight ($w_s^{\max}$) | 0.2 |
| Target Fiedler ($\lambda_2^*$, initial) | 0.1 |
| Fiedler decline threshold | $-0.001$ |
| Stagnation threshold | 0.02 |
| Deviation growth threshold | 0.05 |
| Min bridge LR multiplier | 0.1 |
| Sliding window size | 5 measurements |

## A.3 Per-Experiment Configuration

| ID | Model | Layers | $n$ | Init | Steps | Machine | Wall Time |
|----|-------|--------|-----|------|-------|---------|-----------|
| exp3 | Qwen2.5-7B-Instruct | 28 | 6 | identity | 12,900 | Local (RTX 6000 Ada 48GB) | ~36h |
| exp3\_tiny | TinyLlama-1.1B-Chat | 22 | 6 | identity | 10,000 | Local | ~12h |
| C-001 | TinyLlama-1.1B-Chat | 22 | 6 | identity | 4,000 | Local | ~5h |
| C-002 | TinyLlama-1.1B-Chat | 22 | 6 | geometric | 10,000 | Local | ~12h |
| C-003 | TinyLlama-1.1B-Chat | 22 | 6 | corpus | 10,000 | Local | ~12h |
| H1 | TinyLlama-1.1B-Chat | 22 | 3 | identity | 10,000 | Hermes (RTX 4090 16GB) | ~10h |
| H2 | TinyLlama-1.1B-Chat | 22 | 4 | identity | 10,000 | Hermes | ~10h |
| H3 | TinyLlama-1.1B-Chat | 22 | 6 | identity | 10,000 | Hermes | ~10h |
| H4 | TinyLlama-1.1B-Chat | 22 | 8 | identity | 10,000 | Hermes | ~10h |
| Holly | Wan-2.1-14B | 40 | 6 | identity | 10 epochs | RunPod (A100 80GB) | ~12h |

## A.4 Bridge Parameter Counts

| $n$ | Channel size ($s = r/n$) | Bridge params/module ($n^2$) | Modules (TinyLlama) | Total bridge params |
|-----|--------------------------|------------------------------|---------------------|---------------------|
| 3 | 8 | 9 | 88 | 792 |
| 4 | 6 | 16 | 88 | 1,408 |
| 6 | 4 | 36 | 88 | 3,168 |
| 8 | 3 | 64 | 88 | 5,632 |

Bridge parameters as fraction of total LoRA parameters: 0.04% ($n = 6$, TinyLlama).

## A.5 Hardware

| Machine | GPU | VRAM | Role |
|---------|-----|------|------|
| Local workstation | NVIDIA RTX 6000 Ada Generation | 48 GB | Primary experiments (C-001, C-002, C-003, exp3, exp3\_tiny) |
| Hermes server | NVIDIA RTX 4090 | 16 GB | Channel ablation (H1–H4) |
| RunPod cloud | NVIDIA A100 | 80 GB | Holly Battery (14B) |

## A.6 Block-Diagonal Detection

A bridge matrix $\mathbf{M}$ is classified as block-diagonal if both conditions hold:

1. $\rho > 10$ (co-planar/cross-planar ratio exceeds 10:1)
2. $\max_{(i,j) \in \mathcal{X}} |M_{ij}| < 10^{-3}$ (all cross-planar entries below threshold)

The threshold values (10:1 ratio, $10^{-3}$ maximum) are conservative: typical cybernetic bridges at convergence have $\rho > 60{,}000$:1 and cross-planar maximum $< 10^{-5}$. The classification is robust to threshold choice across several orders of magnitude.

## A.7 Convergence Metrics

Convergence of the three initialization strategies (C-001, C-002, C-003) is assessed at each checkpoint by computing:

- **Co-planar spread**: Maximum pairwise delta in mean co-planar coupling magnitude, expressed as percentage of the mean
- **Ratio spread**: Maximum pairwise delta in co-planar/cross-planar ratio, expressed as percentage of the mean
- **Val loss spread**: Maximum pairwise delta in validation loss

At step 1,000: co-planar spread = 4.3%, ratio spread = 4.3%. At step 2,000: co-planar spread = 1.6%, ratio spread = 2.6%. At step 5,000: co-planar spread = 2.7%. The three initializations are functionally indistinguishable by step 1,000.
