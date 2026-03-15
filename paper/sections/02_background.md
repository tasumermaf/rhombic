# 2. Background and Related Work

## 2.1 Low-Rank Adaptation

LoRA (Hu et al., 2021) injects trainable low-rank decompositions into frozen pretrained models, reducing the number of trainable parameters by orders of magnitude while maintaining competitive task performance. The method has been extended along several axes: adaptive rank allocation (AdaLoRA; Zhang et al., 2023), weight decomposition (DoRA; Liu et al., 2024), and multi-task composition (LoRAHub; Huang et al., 2023). All standard LoRA variants treat the rank dimensions as a homogeneous vector space — individual rank dimensions are interchangeable under rotation, and the adapter provides no compact representation of inter-dimensional coupling.

## 2.2 Multi-Channel and Multi-Head Parameter-Efficient Methods

Several works partition the adapter's parameter space into parallel components. Multi-head LoRA (Wang et al., 2024) splits the rank into heads that attend to different input subspaces. Mixture-of-LoRA (Li et al., 2024) routes inputs through multiple LoRA experts. These methods introduce structural decomposition but do not include learnable inter-component coupling — each head or expert operates independently, and the outputs are combined through fixed aggregation (concatenation, addition, or gating).

RhombiLoRA (Bielec, 2026a) introduces the bridge matrix as an explicit parameterization of inter-channel coupling. Paper 2 in this series established that the bridge learns task-discriminative structure under standard (non-cybernetic) training: bridge fingerprints classify task type with 84.5% leave-one-out accuracy, and bridge interpolation preserves eigenspectrum structure at all mixing ratios. However, without structured feedback, the bridge develops uniform coupling with no topological preference (co-planar/cross-planar ratio 1.002:1). The present work introduces the Steersman to provide that structured feedback.

## 2.3 Cybernetic Optimization

Cybernetic control theory (Wiener, 1948; Ashby, 1956) formalizes feedback systems that maintain stability through sense-decide-actuate loops. In machine learning, second-order feedback has appeared in learning rate scheduling (Smith, 2017), loss landscape analysis (Li et al., 2018), and meta-learning (Andrychowicz et al., 2016). The Steersman differs from learned meta-optimizers in that its control laws are fixed (not learned), its state space is low-dimensional (3 scalar diagnostics), and its actuation is limited to weight adjustments and learning rate scaling. This simplicity is deliberate: we aim to demonstrate that even minimal feedback is sufficient to drive strong structural emergence.

## 2.4 Block-Diagonal Structure in Neural Networks

Block-diagonal weight matrices arise naturally in modular networks (Clune et al., 2013; Amer and Maul, 2019), where architectural constraints enforce modularity. The lottery ticket hypothesis (Frankle and Carlin, 2019) identifies sparse subnetworks within dense networks; when the surviving connections cluster, the effective weight matrix is approximately block-diagonal. Mixture-of-experts architectures (Fedus et al., 2022; Lepikhin et al., 2021) achieve block-diagonal routing through learned gating functions. In all these cases, the block structure is either architecturally imposed (modular networks), a consequence of pruning (lottery tickets), or mediated by a separate routing mechanism (MoE gating). The present work demonstrates block-diagonal emergence in a dense, unconstrained matrix through feedback-driven training dynamics alone.

## 2.5 The Rhombic Dodecahedron

The rhombic dodecahedron (RD) is the Voronoi cell of the face-centered cubic (FCC) lattice (Conway and Sloane, 1999). Its 12 faces tessellate $\mathbb{R}^3$ without gaps, and its dual is the cuboctahedron. The RD's 6 face pairs partition naturally by coordinate axis, yielding the 3-fold symmetry that motivates the contrastive loss in Section 3.2. In prior work (Bielec, 2026a), we established that 12-connected FCC lattices provide 30% shorter paths and 2.4$\times$ higher algebraic connectivity than 6-connected cubic lattices at comparable spatial resolution. The present work uses the RD's coordinate geometry as a structural prior for multi-channel coupling, not as a spatial lattice.
