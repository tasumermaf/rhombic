"""Generate the Rung 4 context architecture dashboard."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rhombic.context import run_context_benchmark

# 8-Law Weave palette
CUBIC_COLOR = '#3D3D6B'
FCC_COLOR = '#B34444'

def main():
    # Run benchmarks at three scales
    scales = [125, 500, 1000]
    results = []
    for scale in scales:
        n_embed = min(scale, 500)
        print(f"Running context benchmark at scale ~{scale}...")
        r = run_context_benchmark(target_nodes=scale, n_embeddings=n_embed)
        results.append(r)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Rung 4: Context Architecture — Cubic vs FCC',
                 fontsize=14, fontweight='bold', y=0.98)

    # ── Row 1: Neighborhood Recall ────────────────────────────────

    # Panel 1: Recall at 1-hop across scales
    ax = axes[0, 0]
    scales_actual = [r.cubic_nodes for r in results]
    recall_c = [r.recall[0].cubic_recall for r in results]
    recall_f = [r.recall[0].fcc_recall for r in results]
    x = np.arange(len(scales_actual))
    w = 0.35
    ax.bar(x - w/2, recall_c, w, color=CUBIC_COLOR, label='Cubic')
    ax.bar(x + w/2, recall_f, w, color=FCC_COLOR, label='FCC')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scales_actual])
    ax.set_ylabel('Recall@1-hop')
    ax.set_xlabel('Lattice nodes')
    ax.set_title('Embedding Neighbor Recall (1-hop)')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Panel 2: Recall by hop depth (largest scale)
    ax = axes[0, 1]
    r = results[-1]
    hops = [rec.hops for rec in r.recall]
    rc = [rec.cubic_recall for rec in r.recall]
    rf = [rec.fcc_recall for rec in r.recall]
    ax.plot(hops, rc, 'o-', color=CUBIC_COLOR, label='Cubic', linewidth=2)
    ax.plot(hops, rf, 's-', color=FCC_COLOR, label='FCC', linewidth=2)
    ax.set_xlabel('Hops')
    ax.set_ylabel('Recall')
    ax.set_title(f'Recall vs Hops ({r.cubic_nodes} nodes)')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Panel 3: Neighborhood size by hop depth
    ax = axes[0, 2]
    sc = [rec.cubic_neighborhood_size for rec in r.recall]
    sf = [rec.fcc_neighborhood_size for rec in r.recall]
    ax.plot(hops, sc, 'o-', color=CUBIC_COLOR, label='Cubic', linewidth=2)
    ax.plot(hops, sf, 's-', color=FCC_COLOR, label='FCC', linewidth=2)
    ax.set_xlabel('Hops')
    ax.set_ylabel('Avg neighborhood size')
    ax.set_title('Neighborhood Size vs Hops')
    ax.legend()

    # ── Row 2: Diffusion + Consensus ─────────────────────────────

    # Panel 4: Diffusion curves (largest scale)
    ax = axes[1, 0]
    r = results[-1]
    rounds = np.arange(1, len(r.diffusion.cubic_curve) + 1)
    ax.plot(rounds, r.diffusion.cubic_curve, color=CUBIC_COLOR,
            label='Cubic', linewidth=2)
    ax.plot(rounds, r.diffusion.fcc_curve, color=FCC_COLOR,
            label='FCC', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Fraction of lattice reached')
    ax.set_title(f'Information Diffusion ({r.cubic_nodes} nodes)')
    ax.legend()

    # Panel 5: Diffusion speedup across scales
    ax = axes[1, 1]
    speedup_50 = [r.diffusion.rounds_to_50_cubic / max(1, r.diffusion.rounds_to_50_fcc)
                  for r in results]
    speedup_80 = [r.diffusion.rounds_to_80_cubic / max(1, r.diffusion.rounds_to_80_fcc)
                  for r in results]
    ax.bar(x - w/2, speedup_50, w, color=CUBIC_COLOR, alpha=0.7, label='50% reach')
    ax.bar(x + w/2, speedup_80, w, color=FCC_COLOR, alpha=0.7, label='80% reach')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scales_actual])
    ax.set_ylabel('FCC speedup (×)')
    ax.set_xlabel('Lattice nodes')
    ax.set_title('Diffusion Speedup')
    ax.legend()

    # Panel 6: Consensus convergence curves (medium scale)
    ax = axes[1, 2]
    r = results[1]  # medium scale
    max_show = min(100, len(r.consensus.cubic_curve), len(r.consensus.fcc_curve))
    rounds_c = np.arange(1, max_show + 1)
    ax.semilogy(rounds_c, r.consensus.cubic_curve[:max_show],
                color=CUBIC_COLOR, label='Cubic', linewidth=2)
    ax.semilogy(rounds_c, r.consensus.fcc_curve[:max_show],
                color=FCC_COLOR, label='FCC', linewidth=2)
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='ε=0.05')
    ax.set_xlabel('Round')
    ax.set_ylabel('Max deviation from mean')
    ax.set_title(f'Consensus Convergence ({r.cubic_nodes} nodes)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('results/rung-4/dashboard.png', dpi=150, bbox_inches='tight')
    print(f"Dashboard saved to results/rung-4/dashboard.png")


if __name__ == '__main__':
    main()
