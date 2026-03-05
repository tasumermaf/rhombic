"""
Rung 4: Context architecture benchmarks — lattice topology meets AI.

Current AI context management is rectilinear: token windows are 1D slices,
RAG retrieval uses flat top-k, multi-agent routing is hub-and-spoke,
attention is either O(n^2) or windowed (local, cubic). What happens when
you organize high-dimensional data on FCC topology instead?

Three benchmarks:

1. Neighborhood Recall — the RAG metric. Given N embeddings quantized to
   a lattice, what fraction of an embedding's true nearest neighbors fall
   within its lattice k-hop neighborhood? FCC's 12 connections should
   capture more relevant context per hop.

2. Information Diffusion — the attention metric. How many rounds of
   neighbor-averaging until a query node accumulates X% of total
   relevance? FCC's denser connectivity should propagate faster.

3. Consensus Speed — the multi-agent metric. N agents with state vectors
   average with their lattice neighbors each round. How many rounds to
   convergence? Models distributed AI agreement.

The question: do the Rung 1-3 advantages (30% shorter paths, 55% more
flood fill reach, 5-10x signal quality) translate when the lattice
organizes high-dimensional AI data?
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import KDTree
from rhombic.lattice import CubicLattice, FCCLattice
from rhombic.spatial import build_adjacency, flood_fill
from rhombic.benchmark import matched_lattices


# ── Result types ─────────────────────────────────────────────────────


@dataclass
class RecallResult:
    """Neighborhood recall at a single k-hop depth."""
    hops: int
    cubic_recall: float      # fraction of true k-NN captured by lattice neighborhood
    fcc_recall: float
    cubic_neighborhood_size: float  # avg nodes in k-hop neighborhood
    fcc_neighborhood_size: float


@dataclass
class DiffusionResult:
    """Information diffusion speed comparison."""
    rounds_to_50_cubic: int
    rounds_to_50_fcc: int
    rounds_to_80_cubic: int
    rounds_to_80_fcc: int
    cubic_curve: list[float]   # cumulative relevance captured per round
    fcc_curve: list[float]


@dataclass
class ConsensusResult:
    """Multi-agent consensus speed comparison."""
    cubic_rounds: int         # rounds to converge within epsilon
    fcc_rounds: int
    cubic_curve: list[float]  # max deviation from mean per round
    fcc_curve: list[float]


@dataclass
class ContextResult:
    """Full Rung 4 benchmark results."""
    target_nodes: int
    cubic_nodes: int
    fcc_nodes: int
    embedding_dim: int
    n_embeddings: int

    # Neighborhood recall at multiple hop depths
    recall: list[RecallResult]

    # Information diffusion
    diffusion: DiffusionResult

    # Consensus
    consensus: ConsensusResult

    elapsed_seconds: float


# ── Embedding generation ─────────────────────────────────────────────


def generate_embeddings(n: int, dim: int, n_clusters: int = 5,
                        seed: int = 42) -> np.ndarray:
    """Generate clustered embeddings simulating real token distributions.

    Real embeddings aren't uniform — they cluster around semantic topics.
    This generates n_clusters Gaussian clusters in dim-dimensional space
    with varying spread, producing a realistic distribution for testing
    neighborhood recall.
    """
    rng = np.random.default_rng(seed)
    points_per_cluster = n // n_clusters
    remainder = n - points_per_cluster * n_clusters

    embeddings = []
    for i in range(n_clusters):
        center = rng.standard_normal(dim) * 3.0
        spread = rng.uniform(0.3, 1.5)
        count = points_per_cluster + (1 if i < remainder else 0)
        cluster = center + rng.standard_normal((count, dim)) * spread
        embeddings.append(cluster)

    return np.vstack(embeddings)


def project_to_3d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """Project high-dimensional embeddings to 3D via random projection.

    Johnson-Lindenstrauss: random projection approximately preserves
    pairwise distances. This is how we map embedding space to lattice space.
    """
    rng = np.random.default_rng(seed)
    dim = embeddings.shape[1]
    # Random projection matrix (orthogonalized for quality)
    proj = rng.standard_normal((dim, 3))
    proj, _ = np.linalg.qr(proj)
    projected = embeddings @ proj[:, :3]

    # Normalize to [0, 1]³
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    extent = maxs - mins
    extent = np.where(extent > 0, extent, 1.0)
    return (projected - mins) / extent


# ── Lattice assignment ───────────────────────────────────────────────


def assign_to_lattice(points_3d: np.ndarray,
                      lattice_positions: np.ndarray) -> np.ndarray:
    """Assign each 3D point to its nearest lattice node.

    Returns array of lattice node indices, one per input point.
    """
    tree = KDTree(lattice_positions)
    _, indices = tree.query(points_3d)
    return indices


# ── Benchmark 1: Neighborhood Recall ─────────────────────────────────


def neighborhood_recall(embeddings: np.ndarray, points_3d: np.ndarray,
                        lattice_positions: np.ndarray,
                        adjacency: dict[int, list[int]],
                        assignments: np.ndarray,
                        k_true: int = 10,
                        max_hops: int = 3) -> list[RecallResult]:
    """Measure how well lattice neighborhoods capture embedding neighbors.

    For each embedding:
    1. Find its k_true nearest neighbors in original embedding space
    2. Find its lattice node's h-hop neighborhood (h = 1, 2, ..., max_hops)
    3. Recall = fraction of true neighbors whose lattice nodes are in
       the h-hop lattice neighborhood

    This is the RAG metric: does the lattice topology capture semantic
    proximity?
    """
    n = len(embeddings)
    # True nearest neighbors in embedding space
    embed_tree = KDTree(embeddings)
    _, true_nn = embed_tree.query(embeddings, k=k_true + 1)
    # Drop self (first column)
    true_nn = true_nn[:, 1:]

    results = []
    for hops in range(1, max_hops + 1):
        recalls = []
        neighborhood_sizes = []

        for i in range(n):
            # Lattice neighborhood of this point's assigned node
            lattice_node = assignments[i]
            lattice_neighbors = flood_fill(adjacency, lattice_node, hops)
            neighborhood_sizes.append(len(lattice_neighbors))

            # Which of the true nearest neighbors are in the lattice neighborhood?
            true_neighbor_nodes = set(assignments[j] for j in true_nn[i])
            captured = len(true_neighbor_nodes & lattice_neighbors)
            recall = captured / len(true_neighbor_nodes) if true_neighbor_nodes else 0.0
            recalls.append(recall)

        results.append(RecallResult(
            hops=hops,
            cubic_recall=0.0,   # filled by caller
            fcc_recall=0.0,     # filled by caller
            cubic_neighborhood_size=0.0,
            fcc_neighborhood_size=0.0,
        ))
        # Store raw values for the caller to assign to the right topology
        results[-1]._raw_recall = float(np.mean(recalls))
        results[-1]._raw_size = float(np.mean(neighborhood_sizes))

    return results


# ── Benchmark 2: Information Diffusion ───────────────────────────────


def information_diffusion(adjacency: dict[int, list[int]],
                          source_node: int,
                          n_nodes: int,
                          max_rounds: int = 30) -> list[float]:
    """Measure how fast a signal pulse spreads through the lattice.

    Start: signal = 1.0 at source_node, 0.0 everywhere else.
    Each round: node[i] = max(node[i], max(neighbors[j] * decay))
    where decay = 0.9 per hop. This models information propagation
    without conservation — we care about REACH, not redistribution.

    Track: fraction of nodes that have received signal > 0.01.
    FCC's 12 connections should reach more nodes per round than cubic's 6.
    """
    signal = np.zeros(n_nodes)
    signal[source_node] = 1.0

    curve = []
    threshold = 0.01
    for _ in range(max_rounds):
        new_signal = signal.copy()
        for node in range(n_nodes):
            if signal[node] > threshold:
                for nb in adjacency.get(node, []):
                    propagated = signal[node] * 0.9
                    if propagated > new_signal[nb]:
                        new_signal[nb] = propagated
        signal = new_signal
        reached = float(np.sum(signal > threshold) / n_nodes)
        curve.append(reached)
    return curve


def _rounds_to_threshold(curve: list[float], threshold: float) -> int:
    """Find first round where curve exceeds threshold."""
    for i, v in enumerate(curve):
        if v >= threshold:
            return i + 1
    return len(curve)


# ── Benchmark 3: Consensus Speed ─────────────────────────────────────


def consensus_speed(adjacency: dict[int, list[int]],
                    initial_states: np.ndarray,
                    epsilon: float = 0.05,
                    max_rounds: int = 500) -> tuple[int, list[float]]:
    """Measure rounds to consensus via Laplacian averaging.

    Each round: new[i] = mean(state[i], state[neighbors])
    This is the standard Laplacian consensus update. Each node replaces
    its state with the average of itself and ALL its neighbors, weighted
    equally per neighbor.

    The convergence rate is determined by the spectral gap of the
    averaging matrix, which relates directly to the Fiedler value.
    FCC's 2.4x higher algebraic connectivity should yield faster consensus.

    Convergence when max deviation from global mean < epsilon.
    """
    n = len(initial_states)
    # Normalize initial states to zero mean, unit variance
    state = initial_states.copy().astype(np.float64)
    std = state.std()
    if std > 1e-15:
        state = (state - state.mean()) / std
    global_mean = float(state.mean())  # should be ~0

    deviations = []
    converged_at = max_rounds

    for round_num in range(max_rounds):
        max_dev = float(np.max(np.abs(state - global_mean)))
        deviations.append(max_dev)

        if max_dev < epsilon and converged_at == max_rounds:
            converged_at = round_num + 1
            # Keep running to fill the curve for visualization
            if round_num > converged_at + 20:
                # Pad remaining with final value
                deviations.extend([max_dev] * (max_rounds - round_num - 1))
                break

        # Standard Laplacian: new[i] = mean(self + all neighbors)
        new_state = np.zeros_like(state)
        for node in range(n):
            neighbors = adjacency.get(node, [])
            total = state[node]
            for nb in neighbors:
                total += state[nb]
            new_state[node] = total / (len(neighbors) + 1)
        state = new_state

    return converged_at, deviations


# ── Main benchmark runner ────────────────────────────────────────────


def run_context_benchmark(target_nodes: int = 500,
                          embedding_dim: int = 128,
                          n_embeddings: int = 200,
                          seed: int = 42) -> ContextResult:
    """Run the full Rung 4 context architecture benchmark.

    1. Generate clustered embeddings in d-dim space
    2. Project to 3D
    3. Assign to both cubic and FCC lattices
    4. Measure neighborhood recall, diffusion, consensus on both
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Generate lattices
    cubic, fcc = matched_lattices(target_nodes)
    adj_c = build_adjacency(cubic.edges, cubic.node_count)
    adj_f = build_adjacency(fcc.edges, fcc.node_count)

    # Generate embeddings and project to 3D
    embeddings = generate_embeddings(n_embeddings, embedding_dim, seed=seed)
    points_3d = project_to_3d(embeddings, seed=seed)

    # Scale 3D points to match lattice bounding boxes
    pos_c = cubic.positions
    pos_f = fcc.positions

    def scale_to_lattice(pts, lattice_pos):
        bb_min = lattice_pos.min(axis=0)
        bb_max = lattice_pos.max(axis=0)
        extent = bb_max - bb_min
        extent = np.where(extent > 0, extent, 1.0)
        # Add margin to keep points interior
        margin = 0.1
        return bb_min + extent * (margin + pts * (1 - 2 * margin))

    pts_for_cubic = scale_to_lattice(points_3d, pos_c)
    pts_for_fcc = scale_to_lattice(points_3d, pos_f)

    # Assign embeddings to lattice nodes
    assign_c = assign_to_lattice(pts_for_cubic, pos_c)
    assign_f = assign_to_lattice(pts_for_fcc, pos_f)

    # ── Benchmark 1: Neighborhood Recall ─────────────────────────
    k_true = min(10, n_embeddings - 1)
    max_hops = 3

    recall_c = neighborhood_recall(
        embeddings, pts_for_cubic, pos_c, adj_c, assign_c,
        k_true=k_true, max_hops=max_hops)
    recall_f = neighborhood_recall(
        embeddings, pts_for_fcc, pos_f, adj_f, assign_f,
        k_true=k_true, max_hops=max_hops)

    # Merge into RecallResult objects
    recall_results = []
    for rc, rf in zip(recall_c, recall_f):
        recall_results.append(RecallResult(
            hops=rc.hops,
            cubic_recall=rc._raw_recall,
            fcc_recall=rf._raw_recall,
            cubic_neighborhood_size=rc._raw_size,
            fcc_neighborhood_size=rf._raw_size,
        ))

    # ── Benchmark 2: Information Diffusion ───────────────────────
    # Pulse from a central node — how fast does signal reach the lattice?
    center_c = np.argmin(np.linalg.norm(
        pos_c - pos_c.mean(axis=0), axis=1))
    center_f = np.argmin(np.linalg.norm(
        pos_f - pos_f.mean(axis=0), axis=1))

    curve_c = information_diffusion(adj_c, center_c, cubic.node_count)
    curve_f = information_diffusion(adj_f, center_f, fcc.node_count)

    diffusion = DiffusionResult(
        rounds_to_50_cubic=_rounds_to_threshold(curve_c, 0.50),
        rounds_to_50_fcc=_rounds_to_threshold(curve_f, 0.50),
        rounds_to_80_cubic=_rounds_to_threshold(curve_c, 0.80),
        rounds_to_80_fcc=_rounds_to_threshold(curve_f, 0.80),
        cubic_curve=curve_c,
        fcc_curve=curve_f,
    )

    # ── Benchmark 3: Consensus Speed ─────────────────────────────
    # Initialize agents with random states
    init_c = rng.standard_normal(cubic.node_count)
    init_f = rng.standard_normal(fcc.node_count)

    rounds_c, dev_c = consensus_speed(adj_c, init_c)
    rounds_f, dev_f = consensus_speed(adj_f, init_f)

    consensus = ConsensusResult(
        cubic_rounds=rounds_c,
        fcc_rounds=rounds_f,
        cubic_curve=dev_c,
        fcc_curve=dev_f,
    )

    elapsed = time.perf_counter() - t0

    return ContextResult(
        target_nodes=target_nodes,
        cubic_nodes=cubic.node_count,
        fcc_nodes=fcc.node_count,
        embedding_dim=embedding_dim,
        n_embeddings=n_embeddings,
        recall=recall_results,
        diffusion=diffusion,
        consensus=consensus,
        elapsed_seconds=elapsed,
    )


def print_context_result(r: ContextResult):
    """Print context benchmark results."""
    print(f"\n{'='*72}")
    print(f"CONTEXT ARCHITECTURE BENCHMARK ({r.elapsed_seconds:.1f}s)")
    print(f"{'='*72}")
    print(f"Lattice nodes: Cubic={r.cubic_nodes}, FCC={r.fcc_nodes}")
    print(f"Embeddings: {r.n_embeddings} points in {r.embedding_dim}D")
    print()

    # Recall
    print("NEIGHBORHOOD RECALL (fraction of true k-NN captured)")
    print(f"{'Hops':>6} {'Cubic Recall':>14} {'FCC Recall':>14} "
          f"{'Δ':>8} {'Cubic Size':>12} {'FCC Size':>12}")
    print(f"{'-'*68}")
    for rec in r.recall:
        delta = rec.fcc_recall - rec.cubic_recall
        print(f"{rec.hops:>6} {rec.cubic_recall:>14.3f} {rec.fcc_recall:>14.3f} "
              f"{delta:>+8.3f} {rec.cubic_neighborhood_size:>12.1f} "
              f"{rec.fcc_neighborhood_size:>12.1f}")

    # Diffusion
    print(f"\nINFORMATION DIFFUSION (rounds to threshold)")
    print(f"{'Threshold':>12} {'Cubic':>8} {'FCC':>8} {'Speedup':>10}")
    print(f"{'-'*40}")
    print(f"{'50%':>12} {r.diffusion.rounds_to_50_cubic:>8} "
          f"{r.diffusion.rounds_to_50_fcc:>8} "
          f"{r.diffusion.rounds_to_50_cubic / max(1, r.diffusion.rounds_to_50_fcc):>10.2f}x")
    print(f"{'80%':>12} {r.diffusion.rounds_to_80_cubic:>8} "
          f"{r.diffusion.rounds_to_80_fcc:>8} "
          f"{r.diffusion.rounds_to_80_cubic / max(1, r.diffusion.rounds_to_80_fcc):>10.2f}x")

    # Consensus
    print(f"\nCONSENSUS SPEED (rounds to convergence, ε=0.05)")
    print(f"  Cubic: {r.consensus.cubic_rounds} rounds")
    print(f"  FCC:   {r.consensus.fcc_rounds} rounds")
    speedup = r.consensus.cubic_rounds / max(1, r.consensus.fcc_rounds)
    print(f"  Speedup: {speedup:.2f}x")


def run_context_suite(scales: list[int] | None = None) -> list[ContextResult]:
    """Run context benchmarks at multiple scales."""
    if scales is None:
        scales = [125, 500, 1000]

    results = []
    for scale in scales:
        n_embed = min(scale, 500)  # cap embeddings for tractability
        print(f"\nRunning context benchmark at scale ~{scale} "
              f"({n_embed} embeddings)...")
        r = run_context_benchmark(target_nodes=scale, n_embeddings=n_embed)
        print_context_result(r)
        results.append(r)

    return results


if __name__ == "__main__":
    run_context_suite()
