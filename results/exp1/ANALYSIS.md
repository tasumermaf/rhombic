# Experiment 1: RhombiLoRA vs Standard LoRA

## Setup

**Model:** Qwen2.5-1.5B-Instruct (1.55B parameters)
**Dataset:** Alpaca-cleaned (52K instruction-following examples)
**Training:** 2000 steps, batch 4, gradient accumulation 4, lr 2e-4
**GPU:** NVIDIA RTX 6000 Ada (48GB)

## Five Configurations

| # | Config | Rank | Channels | Bridge | Trainable Params |
|---|--------|------|----------|--------|-----------------|
| 1 | Standard LoRA | 24 | 6 | Frozen identity | 3,268,608 |
| 2 | RhombiLoRA frozen | 24 | 6 | Frozen identity | 3,268,608 |
| 3 | RhombiLoRA learnable | 24 | 6 | Learnable (I init) | 3,270,624 |
| 4 | Standard LoRA | 48 | 6 | Frozen identity | 6,537,216 |
| 5 | RhombiLoRA cubic | 24 | 3 | Learnable (I init) | 3,269,112 |

## Result 1: Identity Equivalence Confirmed

Configs 1 and 2 produce nearly identical loss curves (delta < 0.0002 at
every checkpoint). RhombiLoRA with frozen identity bridge IS standard LoRA.

## Result 2: Doubling Rank Does Not Help

Config 4 (rank 48, 2x parameter budget) converges to the SAME loss as
rank 24. The bottleneck at this training duration is not rank — it's
training signal per step. This is important context: the bridge provides
a DIFFERENT kind of capacity (cross-channel mixing) rather than MORE
capacity (higher rank).

| Step | Std r24 | Std r48 | RhombiLoRA r24 |
|------|---------|---------|----------------|
| 500  | 0.7217  | 0.7315  | 0.7214         |
| 1000 | 0.3770  | 0.3769  | 0.3768         |
| 1500 | 0.3722  | 0.3719  | 0.3722         |
| 2000 | 0.3761  | 0.3763  | 0.3760         |

## Result 3: The Bridge Learns Structured Coupling

Config 3's bridge deviates monotonically from identity, developing
increasing algebraic connectivity (Fiedler value) over training:

| Step | Fiedler | Deviation |
|------|---------|-----------|
| 0    | 0.0000  | 0.0000    |
| 500  | 0.0265  | 0.0358    |
| 1000 | 0.0323  | 0.0434    |
| 1500 | 0.0365  | 0.0501    |
| 2000 | 0.0373  | 0.0507    |

The primary learned behavior is diagonal self-amplification (~0.009 per
channel). Off-diagonal cross-channel mixing is an order of magnitude
smaller (~0.001) but consistently present and growing.

## Result 4: 6 Channels > 3 Channels (The Topology Effect)

The 6-channel bridge (FCC topology) develops **4.6x higher Fiedler value**
and **5.3x higher deviation** from identity than the 3-channel bridge
(cubic topology), with identical parameter budget and loss.

|                    | 6-ch (FCC) | 3-ch (cubic) | Ratio |
|--------------------|-----------|-------------|-------|
| Diagonal growth    | +0.0087   | +0.0088     | 1.0x  |
| Mean |off-diag|    | 0.001004  | 0.000604    | 1.7x  |
| Final Fiedler      | 0.0373    | 0.0082      | 4.6x  |
| Final Deviation    | 0.0507    | 0.0095      | 5.3x  |

Both develop the same diagonal self-amplification (topology-independent).
But the 6-channel bridge finds and uses MORE cross-channel coupling. The
optimizer discovers that 6 channels offer more useful mixing pathways than
3 — the FCC topology provides better algebraic connectivity, matching
the Paper 2 result.

## Result 5: Co-planar vs Cross-planar Coupling (Weak Signal)

Within the 6-channel bridge, co-planar direction pairs (predicted by RD
geometry to share 4 octahedral vertices) develop 1.36x stronger coupling
than cross-planar pairs (sharing only 2 vertices). Correlation between
the geometric prediction and learned structure: r = 0.31.

The signal is in the predicted direction but modest. At 2000 steps with
36 bridge parameters per adapter, the signal-to-noise ratio is low.
Individual adapters vary (std = 0.009) more than the mean off-diagonal
coupling (0.001). A longer training run or more focused experiment may
reveal this more clearly.

## Summary

1. RhombiLoRA with frozen bridge = standard LoRA (verified)
2. The bridge learns structured coupling, not random noise
3. 6-channel (FCC) topology learns 4.6x more cross-channel mixing than
   3-channel (cubic) topology — at zero additional parameter cost
4. Doubling rank provides no benefit at this training duration; the bridge
   provides a qualitatively different kind of capacity
5. The learned coupling structure shows a weak (1.36x, r=0.31) alignment
   with the RD geometric prediction — promising but requires longer training

**The thesis holds:** keep your cube, add six bridges. The geometry of the
rhombic dodecahedron provides a natural channel structure for cross-rank
mixing in LoRA adapters.
