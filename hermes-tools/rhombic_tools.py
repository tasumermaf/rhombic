#!/usr/bin/env python3
"""
Rhombic Lattice Topology Tools for Hermes Agent.

Custom tools wrapping the `rhombic` library (v0.3.0) for interactive
lattice topology exploration, experimentation, and visualization.
"""

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _check_rhombic() -> bool:
    try:
        import rhombic
        return True
    except ImportError:
        return False


def _graph_to_spec(g):
    """Convert networkx graph to (n_vertices, edges, edge_list) for spectral API."""
    n = g.number_of_nodes()
    edges = list(g.edges())
    return n, edges


# ── Tool Handlers ────────────────────────────────────────────────────────────

def lattice_compare_handler(args: dict, **kwargs) -> str:
    """Compare cubic (SC) and FCC lattices at a given scale."""
    from rhombic.lattice import CubicLattice, FCCLattice
    import networkx as nx

    scale = int(args.get("scale", 5))

    cubic = CubicLattice(scale)
    fcc = FCCLattice(scale)
    cs = cubic.stats()
    fs = fcc.stats()

    cg = cubic.to_networkx()
    fg = fcc.to_networkx()

    c_path = nx.average_shortest_path_length(cg)
    f_path = nx.average_shortest_path_length(fg)

    result = {
        "scale": scale,
        "cubic": {
            "nodes": cs.node_count, "edges": cs.edge_count,
            "connectivity": cs.connectivity,
            "avg_degree": round(float(cs.avg_degree), 2),
            "avg_path_length": round(c_path, 4),
        },
        "fcc": {
            "nodes": fs.node_count, "edges": fs.edge_count,
            "connectivity": fs.connectivity,
            "avg_degree": round(float(fs.avg_degree), 2),
            "avg_path_length": round(f_path, 4),
        },
        "ratios": {
            "node_ratio": round(fs.node_count / cs.node_count, 2),
            "edge_ratio": round(fs.edge_count / cs.edge_count, 2),
            "path_ratio_fcc_over_sc": round(f_path / c_path, 4),
        },
        "summary": (
            f"At scale {scale}: FCC has {fs.node_count} nodes vs cubic {cs.node_count}. "
            f"FCC paths are {round((1 - f_path/c_path)*100, 1)}% shorter. "
            f"Edge cost: {round(fs.edge_count / cs.edge_count, 1)}x."
        ),
    }
    return json.dumps(result, indent=2)


def fiedler_ratio_handler(args: dict, **kwargs) -> str:
    """Compute weighted Fiedler ratio (FCC/SC) for a given weight distribution and scale."""
    from rhombic.lattice import CubicLattice, FCCLattice
    from rhombic.spectral import fiedler_value
    from rhombic.corpus import weight_distributions
    import numpy as np

    scale = int(args.get("scale", 5))
    distribution = args.get("distribution", "uniform")

    cubic = CubicLattice(scale)
    fcc = FCCLattice(scale)
    cg = cubic.to_networkx()
    fg = fcc.to_networkx()

    rng = np.random.default_rng(42)

    def make_weights(g, dist):
        n_edges = g.number_of_edges()
        if dist == "uniform":
            return np.ones(n_edges).tolist()
        elif dist == "random":
            return (rng.random(n_edges) + 0.01).tolist()
        elif dist == "power_law":
            return (rng.pareto(2.0, n_edges) + 0.01).tolist()
        elif dist == "corpus":
            from rhombic.corpus import edge_values
            vals = edge_values()
            return [float(vals[i % len(vals)]) for i in range(n_edges)]
        else:
            return np.ones(n_edges).tolist()

    cn, ce = _graph_to_spec(cg)
    fn, fe = _graph_to_spec(fg)

    cw = make_weights(cg, distribution)
    fw = make_weights(fg, distribution)

    c_fied = fiedler_value(cn, ce, cw)
    f_fied = fiedler_value(fn, fe, fw)
    ratio = f_fied / c_fied if c_fied > 0 else float('inf')

    result = {
        "scale": scale,
        "distribution": distribution,
        "cubic_fiedler": round(float(c_fied), 6),
        "fcc_fiedler": round(float(f_fied), 6),
        "fiedler_ratio": round(float(ratio), 4),
        "summary": (
            f"Fiedler ratio (FCC/SC) at scale {scale} with {distribution} weights: "
            f"{ratio:.2f}x. FCC algebraic connectivity is {ratio:.1f}x higher."
        ),
    }
    return json.dumps(result, indent=2)


def direction_weights_handler(args: dict, **kwargs) -> str:
    """Run direction-pair weighting experiment (Paper 2, Experiment 5)."""
    from rhombic.lattice import CubicLattice, FCCLattice
    from rhombic.corpus import direction_weights, edge_values
    from rhombic.spectral import fiedler_value
    import numpy as np

    scale = int(args.get("scale", 5))
    distribution = args.get("distribution", "corpus")

    cubic = CubicLattice(scale)
    fcc = FCCLattice(scale)
    cg = cubic.to_networkx()
    fg = fcc.to_networkx()

    rng = np.random.default_rng(42)
    ev = edge_values()  # The 24 corpus values

    # Direction-pair weighting for FCC: sort values into 6 direction buckets
    fw = direction_weights(ev, 6)
    # Extend to full FCC edge count by cycling
    fn, fe = _graph_to_spec(fg)
    n_fcc_edges = fg.number_of_edges()
    fw_full = [float(fw[i % len(fw)]) for i in range(n_fcc_edges)]
    f_fied = fiedler_value(fn, fe, fw_full)

    # Edge-cycled weighting for SC (baseline comparison)
    cn, ce = _graph_to_spec(cg)
    n_sc_edges = cg.number_of_edges()
    cw = [float(ev[i % len(ev)]) for i in range(n_sc_edges)]
    c_fied = fiedler_value(cn, ce, cw)

    ratio = f_fied / c_fied if c_fied > 0 else float('inf')

    result = {
        "scale": scale,
        "distribution": distribution,
        "method": "direction_pair_weighting",
        "cubic_fiedler": round(float(c_fied), 6),
        "fcc_fiedler": round(float(f_fied), 6),
        "fiedler_ratio": round(float(ratio), 4),
        "baseline_ratio": 2.31,
        "amplification": f"{ratio:.2f}x vs 2.31x baseline",
        "summary": (
            f"Direction-weighted Fiedler ratio at scale {scale} ({distribution}): "
            f"{ratio:.2f}x. Paper 1 baseline was 2.31x."
        ),
    }
    return json.dumps(result, indent=2)


def spectral_analysis_handler(args: dict, **kwargs) -> str:
    """Compute weighted Laplacian spectrum of the RD under specified distribution."""
    from rhombic.polyhedron import RhombicDodecahedron
    from rhombic.spectral import spectrum as compute_spectrum
    from rhombic.corpus import edge_values, weight_distributions
    import numpy as np

    distribution = args.get("distribution", "corpus")
    rd = RhombicDodecahedron()
    g = rd.to_networkx()
    n, edges = _graph_to_spec(g)
    n_edges = g.number_of_edges()
    rng = np.random.default_rng(42)

    if distribution == "uniform":
        w = [1.0] * n_edges
    elif distribution == "random":
        w = (rng.random(n_edges) + 0.01).tolist()
    elif distribution == "power_law":
        w = (rng.pareto(2.0, n_edges) + 0.01).tolist()
    elif distribution == "corpus":
        ev = edge_values()
        w = [float(v) for v in ev[:n_edges]]
    else:
        w = [1.0] * n_edges

    spec = compute_spectrum(n, edges, w)
    spectrum_list = [round(float(v), 4) for v in sorted(spec)]
    distinct = len(set(round(v, 3) for v in spec))
    fiedler = spectrum_list[1] if len(spectrum_list) > 1 else 0.0

    result = {
        "graph": "rhombic_dodecahedron",
        "vertices": n, "edges": n_edges,
        "distribution": distribution,
        "spectrum": spectrum_list,
        "fiedler_value": fiedler,
        "distinct_eigenvalues": distinct,
        "total_eigenvalues": len(spectrum_list),
        "summary": (
            f"RD spectrum under {distribution} weights: {distinct}/{len(spectrum_list)} "
            f"distinct eigenvalues. Fiedler value: {fiedler:.4f}. "
            f"{'Degeneracy preserved.' if distinct <= 6 else 'Degeneracy broken by heterogeneous weights.'}"
        ),
    }
    return json.dumps(result, indent=2)


def prime_vertex_map_handler(args: dict, **kwargs) -> str:
    """Run prime-vertex mapping optimization on the RD (Paper 2, Experiment 6)."""
    from rhombic.polyhedron import RhombicDodecahedron
    from rhombic.corpus import edge_values, TRACKED_PRIMES
    from rhombic.assignment import optimal_prime_assignment, null_prime_scores
    import numpy as np

    rd = RhombicDodecahedron()
    g = rd.to_networkx()
    edges = list(g.edges())
    ev = edge_values()
    trivalent = list(rd.trivalent_vertices)
    primes = list(TRACKED_PRIMES)

    # Build vertex_edges map: vertex -> list of edge indices
    vertex_edges = {}
    for i, (u, v) in enumerate(edges):
        vertex_edges.setdefault(u, []).append(i)
        vertex_edges.setdefault(v, []).append(i)

    mapping, score = optimal_prime_assignment(vertex_edges, ev, primes, trivalent)

    # Compute p-value via exhaustive null
    null = null_prime_scores(vertex_edges, ev, primes, trivalent)
    p_val = float(np.mean(null >= score))

    result = {
        "graph": "rhombic_dodecahedron",
        "primes": primes,
        "trivalent_vertices": trivalent,
        "best_score": round(float(score), 4),
        "best_mapping": {str(k): v for k, v in mapping.items()},
        "p_value": p_val if p_val > 0 else f"<= {1/len(null):.1e}",
        "n_permutations": len(null),
        "summary": (
            f"Prime-vertex mapping: best score {score:.1f}, "
            f"p = {p_val:.2e}. The optimal mapping places 8 tracked primes "
            f"at the 8 trivalent vertices of the RD (the cube's corners)."
        ),
    }
    return json.dumps(result, indent=2)


def permutation_control_handler(args: dict, **kwargs) -> str:
    """Run shuffled vs sorted permutation test for direction-pair weighting.

    Tests whether amplification comes from SORTED bucketing (alignment) or
    merely from having 6 bins. The sorted baseline uses direction_weights()
    which sorts then buckets. The shuffled null assigns values to direction
    buckets in RANDOM order (no sorting), then averages per bucket.
    """
    from rhombic.lattice import FCCLattice
    from rhombic.corpus import direction_weights, edge_values
    from rhombic.spectral import fiedler_value
    import numpy as np

    n_trials = int(args.get("trials", 200))
    scale = int(args.get("scale", 5))

    fcc = FCCLattice(scale)
    fg = fcc.to_networkx()
    fn, fe = _graph_to_spec(fg)
    n_fcc_edges = fg.number_of_edges()
    ev = edge_values()
    n_dirs = 6
    per_group = len(ev) // n_dirs

    # Sorted baseline (direction_weights sorts internally, then buckets)
    fw_sorted = direction_weights(ev, n_dirs)
    fw_sorted_full = [float(fw_sorted[i % len(fw_sorted)]) for i in range(n_fcc_edges)]
    fiedler_sorted = fiedler_value(fn, fe, fw_sorted_full)

    # Shuffled trials: assign values to buckets in RANDOM order (no sorting)
    rng = np.random.default_rng(42)
    shuffled_fiedlers = []
    for _ in range(n_trials):
        vals = list(ev)
        rng.shuffle(vals)
        # Manual bucketing WITHOUT sorting — the key difference
        buckets = []
        for i in range(n_dirs):
            start = i * per_group
            end = start + per_group if i < n_dirs - 1 else len(vals)
            buckets.append(float(np.mean(vals[start:end])))
        fw_shuf_full = [float(buckets[j % len(buckets)]) for j in range(n_fcc_edges)]
        shuffled_fiedlers.append(fiedler_value(fn, fe, fw_shuf_full))

    shuffled_arr = np.array(shuffled_fiedlers)
    p_value = float(np.mean(shuffled_arr >= fiedler_sorted))

    result = {
        "scale": scale,
        "n_trials": n_trials,
        "sorted_fiedler": round(float(fiedler_sorted), 6),
        "shuffled_mean": round(float(shuffled_arr.mean()), 6),
        "shuffled_std": round(float(shuffled_arr.std()), 6),
        "p_value": p_value if p_value > 0 else f"<= {1/n_trials}",
        "summary": (
            f"Permutation control at scale {scale}: sorted Fiedler = {fiedler_sorted:.4f}, "
            f"shuffled mean = {shuffled_arr.mean():.4f} +/- {shuffled_arr.std():.4f}. "
            f"p = {p_value if p_value > 0 else f'<= {1/n_trials}'}. "
            f"{'Alignment matters.' if p_value < 0.05 else 'No significant difference.'}"
        ),
    }
    return json.dumps(result, indent=2)


def explain_mechanism_handler(args: dict, **kwargs) -> str:
    """Return structured explanation of the bottleneck resilience mechanism."""
    depth = args.get("depth", "intuitive")

    explanations = {
        "intuitive": {
            "title": "Why FCC Beats Cubic",
            "explanation": (
                "Imagine a city where every intersection connects to 6 roads (a cubic grid). "
                "Now imagine one where every intersection connects to 12 roads (the FCC lattice). "
                "When all roads are equal, having more connections helps but not dramatically. "
                "The real difference shows up when roads have different widths. In the cubic city, "
                "if one crucial road narrows, the whole neighborhood gets cut off — there are only "
                "5 alternative routes. In the FCC city, there are 11 alternatives. The FCC lattice "
                "is resilient to bottlenecks because it has redundant paths in every direction."
            ),
            "key_number": "2.3x connectivity advantage with uniform weights, up to 6.1x with structured weights",
            "tagline": "Keep your cube, add six bridges.",
        },
        "technical": {
            "title": "Bottleneck Resilience Mechanism",
            "explanation": (
                "The Fiedler value (algebraic connectivity) is a spectral measure of "
                "bottleneck vulnerability. Cheeger-type inequalities relate it to the graph's "
                "conductance. In a cubic lattice, each node has 6 neighbors along 3 axis-aligned "
                "directions. In the FCC lattice, each node has 12 neighbors along 6 face-diagonal "
                "directions. Under uniform weights, FCC's Fiedler value is 2.3x higher. Under "
                "heterogeneous weights, the advantage amplifies because: (1) sorted-bucketing "
                "concentrates extreme values into coherent directional channels, and (2) FCC's "
                "redundant connectivity routes around low-weight channels."
            ),
            "key_numbers": {
                "uniform_ratio": 2.31,
                "corpus_edge_cycled": 3.11,
                "corpus_direction_weighted": 6.11,
                "prime_vertex_p": 2.5e-5,
                "total_tests": 208,
            },
            "mechanism": "Bottleneck resilience: redundant paths around near-disconnection cuts",
        },
        "full": {
            "title": "Complete Mechanism — Paper 1 to Paper 2",
            "paper1": (
                "Paper 1 established the baseline: FCC gives 2.3x algebraic connectivity, "
                "30% shorter paths, 40% smaller diameter — at 2x edge cost."
            ),
            "paper2_findings": [
                "Exp 1-3: Edge-cycled corpus weights amplify to 3.1x",
                "Exp 4: Corpus breaks RD degeneracy (6 to 14 distinct eigenvalues)",
                "Exp 5: Direction-pair weighting reaches 6.1x Fiedler ratio",
                "Exp 6: Prime-vertex mapping p = 2.5e-5",
                "Exp 7: Suppression consistent across 5 different 24-edge graphs",
            ],
            "consensus_inversion": (
                "FCC consensus 6.7x faster at 125 nodes but 0.73x at 1000. "
                "Mechanism: per-neighbor dilution."
            ),
            "forward_vision": (
                "RhombiLoRA: a geometric LoRA adapter that adds 6 diagonal bridge "
                "connections to transformer attention heads. The cube isn't the enemy — "
                "it's the skeleton. The rhombic dodecahedron contains the cube. "
                "Keep your cube, add six bridges."
            ),
        },
    }

    content = explanations.get(depth, explanations["intuitive"])
    return json.dumps(content, indent=2)


def visualize_rd_handler(args: dict, **kwargs) -> str:
    """Generate a rhombic dodecahedron visualization and save to file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from rhombic.polyhedron import RhombicDodecahedron, CUBIC_VERTICES, OCTAHEDRAL_VERTICES
        import numpy as np

        output_dir = args.get("output_dir", "/tmp")
        filename = args.get("filename", "rd_visualization.png")
        filepath = os.path.join(output_dir, filename)

        rd = RhombicDodecahedron()
        g = rd.to_networkx()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Draw edges
        for u, v in g.edges():
            uc = CUBIC_VERTICES[u] if u < 8 else OCTAHEDRAL_VERTICES[u - 8]
            vc = CUBIC_VERTICES[v] if v < 8 else OCTAHEDRAL_VERTICES[v - 8]
            ax.plot([uc[0], vc[0]], [uc[1], vc[1]], [uc[2], vc[2]],
                    color='#666666', linewidth=0.8, alpha=0.6)

        # Draw trivalent vertices (cube corners) in FCC color
        for i in range(8):
            c = CUBIC_VERTICES[i]
            ax.scatter(*c, color='#B34444', s=60, zorder=5)

        # Draw tetravalent vertices (octahedral) in cubic color
        for i in range(6):
            c = OCTAHEDRAL_VERTICES[i]
            ax.scatter(*c, color='#3D3D6B', s=80, zorder=5, marker='D')

        ax.set_title('Rhombic Dodecahedron\n8 cube corners (red) + 6 octahedral bridges (blue)',
                      fontsize=11)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

        plt.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return json.dumps({
            "status": "success",
            "filepath": filepath,
            "summary": f"Rhombic dodecahedron visualization saved to {filepath}",
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def visualize_amplification_handler(args: dict, **kwargs) -> str:
    """Generate the amplification gradient bar chart (Paper 2, Figure 3)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        output_dir = args.get("output_dir", "/tmp")
        filename = args.get("filename", "amplification_gradient.png")
        filepath = os.path.join(output_dir, filename)

        labels = ['Uniform', 'Random', 'Power-law', 'Corpus']
        ratios_ec = [2.55, 2.64, 3.06, 3.11]
        ratios_dw = [2.55, 3.65, 3.37, 6.11]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width/2, ratios_ec, width, color='#3D3D6B',
               label='Edge-cycled', alpha=0.85)
        ax.bar(x + width/2, ratios_dw, width, color='#B34444',
               label='Direction-weighted', alpha=0.85)

        for i, v in enumerate(ratios_dw):
            ax.text(x[i] + width/2, v + 0.08, f'{v:.1f}x',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#B34444')

        ax.set_ylabel('Fiedler ratio (FCC / Cubic)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 7.5)
        ax.axhline(y=2.31, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.text(3.6, 2.31 + 0.1, 'Paper 1 baseline (2.3x)',
                fontsize=7, color='grey', ha='right')
        ax.legend(fontsize=9)
        ax.set_title('Fiedler ratio amplification at scale 1,000')

        plt.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return json.dumps({
            "status": "success",
            "filepath": filepath,
            "summary": f"Amplification gradient chart saved to {filepath}",
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ── Registry ────────────────────────────────────────────────────────────────

from tools.registry import registry

LATTICE_COMPARE_SCHEMA = {
    "name": "lattice_compare",
    "description": "Compare cubic (SC) and FCC lattice topologies at a given scale. Returns node/edge counts, connectivity, path lengths, and ratios.",
    "parameters": {
        "type": "object",
        "properties": {
            "scale": {
                "type": "integer",
                "description": "Lattice dimension (default 5). Cubic has n^3 nodes, FCC ~4n^3. Keep <= 10 for fast results.",
                "default": 5,
            }
        },
        "required": [],
    },
}

FIEDLER_RATIO_SCHEMA = {
    "name": "fiedler_ratio",
    "description": "Compute weighted Fiedler ratio (FCC/SC algebraic connectivity) for a weight distribution and scale. Higher = more FCC advantage.",
    "parameters": {
        "type": "object",
        "properties": {
            "scale": {"type": "integer", "description": "Lattice dimension (default 5, keep <= 8)", "default": 5},
            "distribution": {"type": "string", "enum": ["uniform", "random", "power_law", "corpus"], "default": "uniform"},
        },
        "required": [],
    },
}

DIRECTION_WEIGHTS_SCHEMA = {
    "name": "direction_weights",
    "description": "Run direction-pair weighting experiment (Paper 2 key finding). Shows how structured weight assignment amplifies FCC advantage from 2.3x to up to 6.1x.",
    "parameters": {
        "type": "object",
        "properties": {
            "scale": {"type": "integer", "description": "Lattice dimension (default 5)", "default": 5},
            "distribution": {"type": "string", "enum": ["uniform", "random", "power_law", "corpus"], "default": "corpus"},
        },
        "required": [],
    },
}

SPECTRAL_ANALYSIS_SCHEMA = {
    "name": "spectral_analysis",
    "description": "Compute weighted Laplacian spectrum of the rhombic dodecahedron (14V, 24E) under different weight distributions. Shows degeneracy breaking.",
    "parameters": {
        "type": "object",
        "properties": {
            "distribution": {"type": "string", "enum": ["uniform", "random", "power_law", "corpus"], "default": "corpus"},
        },
        "required": [],
    },
}

PRIME_VERTEX_MAP_SCHEMA = {
    "name": "prime_vertex_map",
    "description": "Run prime-vertex mapping on the rhombic dodecahedron. Finds optimal assignment of 8 primes to 8 cube-corner vertices. Paper 2 found p = 2.5e-5.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

PERMUTATION_CONTROL_SCHEMA = {
    "name": "permutation_control",
    "description": "Run shuffled vs sorted permutation test. Tests whether amplification comes from alignment or just bin count.",
    "parameters": {
        "type": "object",
        "properties": {
            "trials": {"type": "integer", "description": "Number of permutation trials (default 200)", "default": 200},
            "scale": {"type": "integer", "description": "Lattice dimension (default 5)", "default": 5},
        },
        "required": [],
    },
}

EXPLAIN_MECHANISM_SCHEMA = {
    "name": "explain_mechanism",
    "description": "Structured explanation of the bottleneck resilience mechanism. Depth: 'intuitive', 'technical', or 'full'.",
    "parameters": {
        "type": "object",
        "properties": {
            "depth": {"type": "string", "enum": ["intuitive", "technical", "full"], "default": "intuitive"},
        },
        "required": [],
    },
}

VISUALIZE_RD_SCHEMA = {
    "name": "visualize_rd",
    "description": "Generate a 3D visualization of the rhombic dodecahedron and save as PNG.",
    "parameters": {
        "type": "object",
        "properties": {
            "output_dir": {"type": "string", "default": "/tmp"},
            "filename": {"type": "string", "default": "rd_visualization.png"},
        },
        "required": [],
    },
}

VISUALIZE_AMPLIFICATION_SCHEMA = {
    "name": "visualize_amplification",
    "description": "Generate the amplification gradient bar chart (Paper 2 Figure 3).",
    "parameters": {
        "type": "object",
        "properties": {
            "output_dir": {"type": "string", "default": "/tmp"},
            "filename": {"type": "string", "default": "amplification_gradient.png"},
        },
        "required": [],
    },
}

for name, schema, handler in [
    ("lattice_compare", LATTICE_COMPARE_SCHEMA, lattice_compare_handler),
    ("fiedler_ratio", FIEDLER_RATIO_SCHEMA, fiedler_ratio_handler),
    ("direction_weights", DIRECTION_WEIGHTS_SCHEMA, direction_weights_handler),
    ("spectral_analysis", SPECTRAL_ANALYSIS_SCHEMA, spectral_analysis_handler),
    ("prime_vertex_map", PRIME_VERTEX_MAP_SCHEMA, prime_vertex_map_handler),
    ("permutation_control", PERMUTATION_CONTROL_SCHEMA, permutation_control_handler),
    ("explain_mechanism", EXPLAIN_MECHANISM_SCHEMA, explain_mechanism_handler),
    ("visualize_rd", VISUALIZE_RD_SCHEMA, visualize_rd_handler),
    ("visualize_amplification", VISUALIZE_AMPLIFICATION_SCHEMA, visualize_amplification_handler),
]:
    registry.register(
        name=name,
        toolset="rhombic",
        schema=schema,
        handler=handler,
        check_fn=_check_rhombic,
        requires_env=[],
        is_async=False,
        description=schema["description"],
    )
