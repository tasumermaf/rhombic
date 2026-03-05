"""
rhombic — Interactive demo of FCC vs Cubic lattice topology.

Three tabs:
  1. The Numbers — headline results + live Rung 1 benchmark
  2. Embedding Recall — FCC vs Cubic ANN index comparison
  3. The Thesis — condensed argument + links
"""

import time
import numpy as np
import networkx as nx
import gradio as gr
import plotly.graph_objects as go

# ── Colors (Vadrashinetal 8-Law Weave) ──────────────────────────────
CUBIC_COLOR = '#3D3D6B'   # Fall of Neutral Events (11)
FCC_COLOR = '#B34444'      # Geometric Essence (67)


# ── Lattice builders (self-contained — no rhombic import needed) ────

FCC_OFFSETS = np.array([
    [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],
    [1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],
    [0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1],
], dtype=np.float64) * 0.5


def build_cubic(n):
    """Build cubic lattice with n^3 nodes."""
    positions = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                positions.append((x, y, z))
    pos = np.array(positions, dtype=np.float64)
    pos_to_idx = {tuple(p): i for i, p in enumerate(positions)}

    edges = []
    for i, (x, y, z) in enumerate(positions):
        for dx, dy, dz in [(1,0,0),(0,1,0),(0,0,1)]:
            nb = (x+dx, y+dy, z+dz)
            if nb in pos_to_idx:
                edges.append((i, pos_to_idx[nb]))
    return pos, edges


def build_fcc(n):
    """Build FCC lattice with ~4n^3 nodes."""
    basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
    positions = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                for b in basis:
                    positions.append(np.array([x,y,z], dtype=np.float64) + b)
    pos = np.array(positions)

    # Deduplicate
    _, unique_idx = np.unique(np.round(pos, 8), axis=0, return_index=True)
    pos = pos[np.sort(unique_idx)]

    # Find edges via distance (nearest neighbor distance = 0.5*sqrt(2))
    from scipy.spatial import KDTree
    tree = KDTree(pos)
    nn_dist = 0.5 * np.sqrt(2)
    pairs = tree.query_pairs(nn_dist + 0.01)
    edges = list(pairs)
    return pos, edges


def lattice_to_nx(positions, edges):
    G = nx.Graph()
    G.add_nodes_from(range(len(positions)))
    G.add_edges_from(edges)
    return G


# ── Tab 1: The Numbers ─────────────────────────────────────────────

HEADLINE_TABLE = """
| Domain | FCC Advantage | Cost |
|--------|--------------|------|
| Graph routing | **30% shorter paths**, 2.4x algebraic connectivity | ~2x edges |
| Spatial operations | **55% more flood fill** reach, 17% faster NN | 3-5x range query time |
| Signal processing | **5-10x lower** reconstruction MSE | Same sample count |
| Context architecture | **+15-26pp** embedding recall at 1-hop | ~2x neighborhood size |

*These ratios are stable across all tested scales, consistent with
derivation from the geometry rather than the sample.*
"""


def run_live_benchmark(target_nodes):
    """Run a quick Rung 1 benchmark at the given scale."""
    target_nodes = int(target_nodes)
    t0 = time.time()

    # Build lattices
    n_cubic = max(2, round(target_nodes ** (1/3)))
    n_fcc = max(2, round((target_nodes / 4) ** (1/3)))

    c_pos, c_edges = build_cubic(n_cubic)
    f_pos, f_edges = build_fcc(n_fcc)

    Gc = lattice_to_nx(c_pos, c_edges)
    Gf = lattice_to_nx(f_pos, f_edges)

    cn, fn = Gc.number_of_nodes(), Gf.number_of_nodes()
    ce, fe = Gc.number_of_edges(), Gf.number_of_edges()

    # Metrics
    c_path = nx.average_shortest_path_length(Gc) if nx.is_connected(Gc) else None
    f_path = nx.average_shortest_path_length(Gf) if nx.is_connected(Gf) else None

    c_diam = nx.diameter(Gc) if nx.is_connected(Gc) else None
    f_diam = nx.diameter(Gf) if nx.is_connected(Gf) else None

    try:
        c_fiedler = nx.algebraic_connectivity(Gc, method='tracemin_lu')
    except Exception:
        c_fiedler = None
    try:
        f_fiedler = nx.algebraic_connectivity(Gf, method='tracemin_lu')
    except Exception:
        f_fiedler = None

    elapsed = time.time() - t0

    # Build results table
    rows = []
    rows.append(f"| Nodes | {cn} | {fn} |")
    rows.append(f"| Edges | {ce} | {fe} | ")
    if c_path and f_path:
        ratio = (c_path - f_path) / c_path * 100
        rows.append(f"| Avg shortest path | {c_path:.2f} | {f_path:.2f} | **{ratio:.0f}% shorter** |")
    if c_diam and f_diam:
        ratio = (c_diam - f_diam) / c_diam * 100
        rows.append(f"| Diameter | {c_diam} | {f_diam} | **{ratio:.0f}% smaller** |")
    if c_fiedler and f_fiedler:
        ratio = f_fiedler / c_fiedler
        rows.append(f"| Algebraic connectivity | {c_fiedler:.3f} | {f_fiedler:.3f} | **{ratio:.1f}x higher** |")

    table = "| Metric | Cubic | FCC | FCC Advantage |\n|--------|-------|-----|---------------|\n"
    table += "\n".join(rows)
    table += f"\n\n*Computed in {elapsed:.2f}s*"

    # Bar chart
    metrics = []
    cubic_vals = []
    fcc_vals = []

    if c_path and f_path:
        metrics.append("Avg Path")
        cubic_vals.append(c_path)
        fcc_vals.append(f_path)
    if c_diam and f_diam:
        metrics.append("Diameter")
        cubic_vals.append(c_diam)
        fcc_vals.append(f_diam)
    if c_fiedler and f_fiedler:
        metrics.append("Alg. Connectivity")
        cubic_vals.append(c_fiedler)
        fcc_vals.append(f_fiedler)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Cubic (6-connected)', x=metrics, y=cubic_vals,
        marker_color=CUBIC_COLOR, text=[f"{v:.2f}" for v in cubic_vals],
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        name='FCC (12-connected)', x=metrics, y=fcc_vals,
        marker_color=FCC_COLOR, text=[f"{v:.2f}" for v in fcc_vals],
        textposition='auto'
    ))
    fig.update_layout(
        barmode='group',
        title=f"Rung 1: Graph Theory at ~{target_nodes} nodes",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=40),
    )

    return table, fig


# ── Tab 2: Embedding Recall ────────────────────────────────────────

def generate_clustered_embeddings(n_vectors, dim, n_clusters=30, seed=42):
    """Generate synthetic embeddings with overlapping clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    assignments = rng.integers(0, n_clusters, size=n_vectors)
    noise = rng.standard_normal((n_vectors, dim)) * 0.3
    embeddings = centers[assignments] + noise
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype(np.float32)


def build_adjacency(edges, n_nodes):
    adj = {i: [] for i in range(n_nodes)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def flood_fill(adj, start, hops):
    visited = {start}
    frontier = {start}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for nb in adj.get(node, []):
                if nb not in visited:
                    next_frontier.add(nb)
                    visited.add(nb)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def run_index_benchmark(target_nodes, n_queries=50, dim=128, max_hops=3):
    """Run the FCC vs Cubic embedding index comparison."""
    target_nodes = int(target_nodes)

    # Generate embeddings
    n_vectors = max(200, target_nodes * 2)
    embeddings = generate_clustered_embeddings(n_vectors, dim, n_clusters=20, seed=42)

    queries = embeddings[:n_queries]
    corpus = embeddings[n_queries:]

    # Ground truth via brute-force cosine
    e_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    sim = q_norm @ e_norm.T
    k = 10
    gt = np.argsort(-sim, axis=1)[:, :k]

    results = {}

    for topology in ["Cubic", "FCC"]:
        # Build lattice
        if topology == "Cubic":
            n = max(2, round(target_nodes ** (1/3)))
            pos, edges = build_cubic(n)
        else:
            n = max(2, round((target_nodes / 4) ** (1/3)))
            pos, edges = build_fcc(n)

        n_nodes = len(pos)
        adj = build_adjacency(edges, n_nodes)

        # PCA to 3D
        mean = corpus.mean(axis=0)
        centered = corpus - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_comp = Vt[:3].T
        projected = centered @ pca_comp

        # Scale to lattice bbox
        lmin, lmax = pos.min(axis=0), pos.max(axis=0)
        lext = lmax - lmin
        lext = np.where(lext > 0, lext, 1.0)

        pmin, pmax = projected.min(axis=0), projected.max(axis=0)
        pext = pmax - pmin
        pext = np.where(pext > 0, pext, 1.0)

        margin = 0.1
        scaled = lmin + lext * (margin + (projected - pmin) / pext * (1 - 2*margin))

        # Assign to lattice nodes
        from scipy.spatial import KDTree
        tree = KDTree(pos)
        _, assignments = tree.query(scaled)

        # Store vectors at nodes
        from collections import defaultdict
        node_vecs = defaultdict(list)
        for vec_idx, node_idx in enumerate(assignments):
            node_vecs[node_idx].append(vec_idx)

        # Query
        hop_recalls = {}
        for hops in range(1, max_hops + 1):
            recalls = []
            for qi in range(len(queries)):
                q = queries[qi]
                q_centered = q - mean
                q_3d = q_centered @ pca_comp
                q_scaled = lmin + lext * (margin + (q_3d - pmin) / pext * (1 - 2*margin))
                _, node_idx = tree.query(q_scaled)

                neighborhood = flood_fill(adj, int(node_idx), hops)
                candidates = []
                for nd in neighborhood:
                    candidates.extend(node_vecs.get(nd, []))

                if not candidates:
                    recalls.append(0.0)
                    continue

                cands = np.array(candidates)
                retrieved = set(cands[:100]) if len(cands) > 100 else set(cands)

                # Rank by cosine in original space for top-k
                cand_vecs = corpus[cands]
                c_n = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-10)
                q_n = q / (np.linalg.norm(q) + 1e-10)
                sims = c_n @ q_n
                top_k_idx = np.argsort(-sims)[:k]
                retrieved = set(cands[top_k_idx])

                true_set = set(gt[qi][:k])
                recalls.append(len(retrieved & true_set) / max(len(true_set), 1))

            hop_recalls[hops] = float(np.mean(recalls))

        results[topology] = {
            "nodes": n_nodes,
            "hop_recalls": hop_recalls,
            "positions": pos,
            "assignments": assignments,
            "projected_scaled": scaled,
        }

    # Build recall comparison chart
    hops_list = list(range(1, max_hops + 1))
    fig_recall = go.Figure()
    fig_recall.add_trace(go.Bar(
        name=f"Cubic ({results['Cubic']['nodes']} nodes)",
        x=[f"{h}-hop" for h in hops_list],
        y=[results['Cubic']['hop_recalls'][h] for h in hops_list],
        marker_color=CUBIC_COLOR,
        text=[f"{results['Cubic']['hop_recalls'][h]:.3f}" for h in hops_list],
        textposition='auto',
    ))
    fig_recall.add_trace(go.Bar(
        name=f"FCC ({results['FCC']['nodes']} nodes)",
        x=[f"{h}-hop" for h in hops_list],
        y=[results['FCC']['hop_recalls'][h] for h in hops_list],
        marker_color=FCC_COLOR,
        text=[f"{results['FCC']['hop_recalls'][h]:.3f}" for h in hops_list],
        textposition='auto',
    ))
    fig_recall.update_layout(
        barmode='group',
        title=f"Recall@10 — FCC vs Cubic at ~{target_nodes} nodes",
        yaxis_title="Recall@10",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=40),
    )

    # Delta text
    fcc_1hop = results['FCC']['hop_recalls'][1]
    cubic_1hop = results['Cubic']['hop_recalls'][1]
    delta = fcc_1hop - cubic_1hop
    delta_text = f"## FCC advantage at 1-hop: **+{delta*100:.1f} percentage points**\n\n"
    delta_text += f"FCC recall: {fcc_1hop:.3f} ({results['FCC']['nodes']} nodes) vs "
    delta_text += f"Cubic recall: {cubic_1hop:.3f} ({results['Cubic']['nodes']} nodes)\n\n"
    delta_text += "*Same vectors, same PCA, same cosine ranking. The only variable is the connectivity pattern.*"

    # 3D scatter of lattice assignments (FCC)
    fcc_pos = results['FCC']['positions']
    fcc_scaled = results['FCC']['projected_scaled']
    fig_3d = go.Figure()

    # Lattice nodes (faint)
    fig_3d.add_trace(go.Scatter3d(
        x=fcc_pos[:, 0], y=fcc_pos[:, 1], z=fcc_pos[:, 2],
        mode='markers',
        marker=dict(size=2, color=FCC_COLOR, opacity=0.15),
        name='FCC lattice nodes',
    ))

    # Embedded vectors
    fig_3d.add_trace(go.Scatter3d(
        x=fcc_scaled[:, 0], y=fcc_scaled[:, 1], z=fcc_scaled[:, 2],
        mode='markers',
        marker=dict(size=3, color='#FFD700', opacity=0.6),
        name='Embedded vectors',
    ))

    fig_3d.update_layout(
        title="PCA projection mapped onto FCC lattice",
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
            aspectmode='cube',
        ),
        template="plotly_white",
        height=500,
        margin=dict(t=50, b=20, l=20, r=20),
    )

    return delta_text, fig_recall, fig_3d


# ── Tab 3: The Thesis ──────────────────────────────────────────────

THESIS_TEXT = """
# The Shape of the Cell

Computation is built on the cube. Memory is linear. Pixels are square.
Voxels are cubic. Nobody chose this — it accumulated. Descartes gave us
orthogonal coordinates. Von Neumann gave us linear memory. The cubic
lattice is the spatial expression of Cartesian geometry.

**Is the cube optimal?** This library measures the alternative: the
face-centered cubic lattice, whose Voronoi cells are rhombic dodecahedra.
12 faces instead of 6. The densest sphere packing in three dimensions
(Kepler, proved by Hales 2005, formally verified 2017). The lattice commonly exhibited by copper,
aluminum, and gold.

## The Evidence

| Metric | FCC vs Cubic |
|--------|-------------|
| Average shortest path | **30% shorter** |
| Graph diameter | **40% smaller** |
| Algebraic connectivity | **2.4x higher** |
| Flood fill reach | **55% more nodes** |
| NN query speed | **17% faster** |
| Signal reconstruction | **5-10x lower MSE** |
| Embedding neighbor recall | **+15-26pp at 1-hop** |
| Information diffusion | **1.4-2x faster** |
| Edge cost | ~2x more edges (the price) |

These ratios are **stable across all tested scales**, consistent with
derivation from the geometry rather than the sample.

## The Cost

The FCC lattice uses ~2x more edges. This is the price: double the wiring.
Whether this tradeoff is favorable depends on the domain. In software data
structures, an edge is a pointer — 8 bytes. In a memory hierarchy, an edge
is an adjacency relationship that costs nothing until traversed, at which
point you arrive 30% sooner.

## The Cybernetic Bridge

W. Ross Ashby's Law of Requisite Variety: a system must have at least as
much variety in its responses as exists in the perturbations it faces. A
cubic cell absorbs perturbation along 6 axes. A rhombic dodecahedral cell
absorbs it along 12. The algebraic connectivity ratio (2.4x, not 2x) shows
the advantage compounds — the isotropy of the FCC lattice produces more than
the sum of its additional connections.

Stafford Beer would recognize this: the difference between a fragile hierarchy
(6-connected, easily partitioned) and a viable system (12-connected, isotropic,
resistant to fragmentation).

---

**Reproduce everything:** `pip install rhombic && python -m rhombic.benchmark`

- [GitHub](https://github.com/promptcrafted/rhombic)
- [PyPI](https://pypi.org/project/rhombic/)
- [Full synthesis](https://github.com/promptcrafted/rhombic/blob/main/results/SYNTHESIS.md)

*Built by [Promptcrafted](https://promptcrafted.com).
The geometry is the argument. The numbers are the evidence.*
"""


# ── Build the Gradio app ───────────────────────────────────────────

with gr.Blocks(
    title="rhombic — FCC vs Cubic Lattice Topology",
    theme=gr.themes.Base(),
    css="""
    .main-header { text-align: center; margin-bottom: 1em; }
    .main-header h1 { color: #B34444; }
    """
) as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>rhombic</h1>
        <p><em>The bottleneck is not the processor. It is the shape of the cell.</em></p>
    </div>
    """)

    with gr.Tabs():
        # ── Tab 1: The Numbers ──
        with gr.TabItem("The Numbers"):
            gr.Markdown(HEADLINE_TABLE)
            gr.Markdown("---")
            gr.Markdown("### Live Benchmark")
            gr.Markdown("Move the slider to run a Rung 1 graph theory benchmark at any scale.")

            with gr.Row():
                slider = gr.Slider(
                    minimum=27, maximum=1000, value=125, step=1,
                    label="Target node count",
                )
                run_btn = gr.Button("Run benchmark", variant="primary")

            results_md = gr.Markdown()
            results_plot = gr.Plot()

            run_btn.click(
                fn=run_live_benchmark,
                inputs=[slider],
                outputs=[results_md, results_plot],
            )

        # ── Tab 2: Embedding Recall ──
        with gr.TabItem("Embedding Recall"):
            gr.Markdown("""
### FCC Embedding Index

A proof-of-concept ANN index that organizes high-dimensional embeddings on
lattice topology. At matched node counts, the FCC index captures more true
nearest neighbors at each hop than the cubic index.
The only variable is the connectivity pattern: 12 neighbors (FCC) vs 6 (cubic).
            """)

            with gr.Row():
                idx_slider = gr.Slider(
                    minimum=50, maximum=500, value=125, step=25,
                    label="Target lattice nodes",
                )
                idx_btn = gr.Button("Run comparison", variant="primary")

            idx_delta = gr.Markdown()
            idx_recall_plot = gr.Plot()
            idx_3d_plot = gr.Plot()

            idx_btn.click(
                fn=run_index_benchmark,
                inputs=[idx_slider],
                outputs=[idx_delta, idx_recall_plot, idx_3d_plot],
            )

        # ── Tab 3: The Thesis ──
        with gr.TabItem("The Thesis"):
            gr.Markdown(THESIS_TEXT)


if __name__ == "__main__":
    demo.launch()
