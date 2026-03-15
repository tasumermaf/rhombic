# Figure References for Paper 3

## Paper Figures (mapped to sections)

| Fig | File | Section | Content |
|-----|------|---------|---------|
| 1 | `fig_all_6_cybernetic_definitive.png` | §4.1, §4.2 | All 6 cybernetic experiments: co/cross ratio (log) over training steps + early lock-in zoom. Shows convergence to 64K-71K band and 82,854:1 peak. |
| 2 | `fig_init_convergence_final.png` | §4.3 | Four-panel init convergence proof: co/cross ratio, val loss, ratio delta (%), Bridge Fiedler. All 4 n=6 runs (C-001, C-002, C-003, H3). |
| 3 | `fig_corpus_dismantling_full.png` | §4.3 | Corpus-coupled dismantling: per-pair coupling trajectories showing IC→RD crossover at step ~75 and 99.5% suppression by step 900. |
| 4 | `fig_bridge_heatmap_c002_vs_exp2.png` | §4.1, §6.1 | 2×4 panel: cybernetic (C-002) vs non-cybernetic (exp2) bridge heatmaps per module. Visual BD vs near-identity contrast. |
| 5 | `channel-ablation/fig_channel_ablation_definitive.png` | §5.2 | THE ablation figure. Three panels: val loss (n=3/4/6), Bridge Fiedler (log, 1,020× gap), co/cross ratio (n=6 only, peak 82,154:1). |
| 6 | `channel-ablation/fig_h2_eigenvalue_divergence.png` | §5.2.3 | H2 eigenvalue ratios diverging from degeneracy — no block structure at n=4. |
| 7 | `channel-ablation/fig_fiedler_saturation_comparison.png` | §5.2.2 | Fiedler saturation across n=3/4/6 + val loss comparison. |
| 8 | `fig_per_layer_coupling.png` | §6.1 | Per-layer co-planar coupling: Qwen 7B vs TinyLlama. |
| 9 | `fig_per_module_coupling_c002.png` | §6.1 | Per-module coupling hierarchy: k_proj > v_proj > q_proj > o_proj. |

## Supplementary Figures

| Fig | File | Content |
|-----|------|---------|
| S1 | `fig_temporal_emergence.png` | Original 3-experiment temporal emergence (exp3, exp3_tiny, C-001) |
| S2 | `fig_early_emergence.png` | First 1000 steps zoomed (3 experiments) |
| S3 | `fig_init_convergence_comprehensive.png` | Earlier 3-init convergence (partial data) |
| S4 | `fig_init_growth_analysis.png` | Power-law extrapolation (later shown unreliable) |
| S5 | `fig_heatmap_comparison.png` | 4-panel cybernetic vs non-cybernetic heatmaps |
| S6 | `fig_heatmap_cybernetic.png` | Single cybernetic bridge (Qwen 7B L14 q_proj) |

## Key Numbers Referenced in Figures

- Peak ratio: 82,854:1 (C-003, step 9000)
- Convergence band: 64,168:1 – 71,337:1
- Bridge Fiedler (n=6): 0.00009
- Bridge Fiedler (n=3/4/8): 0.085–0.095
- Bifurcation: 1,020×
- Val loss range: 0.4010–0.4022 (0.17% max delta)
- Lock-in step: 200
- Cross-planar half-life: 123 steps
