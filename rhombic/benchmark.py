"""
Rung 1: Graph theory benchmarks comparing cubic and FCC lattices.

Metrics:
  - Average shortest path length
  - Graph diameter
  - Algebraic connectivity (Fiedler value)
  - Fault tolerance (random node removal)
  - Degree distribution

All benchmarks use matched node counts for fair comparison.
"""

from __future__ import annotations

import time
import numpy as np
import networkx as nx
from dataclasses import dataclass
from rhombic.lattice import CubicLattice, FCCLattice


@dataclass
class BenchmarkResult:
    """Results from a single benchmark comparison."""
    scale: int                       # approximate node count
    cubic_nodes: int
    fcc_nodes: int
    cubic_edges: int
    fcc_edges: int
    cubic_avg_degree: float
    fcc_avg_degree: float
    cubic_avg_path: float | None     # None if too large to compute
    fcc_avg_path: float | None
    cubic_diameter: int | None
    fcc_diameter: int | None
    cubic_fiedler: float | None
    fcc_fiedler: float | None
    cubic_fault_curve: list[float]   # fraction connected vs fraction removed
    fcc_fault_curve: list[float]
    elapsed_seconds: float


def matched_lattices(target_nodes: int) -> tuple[CubicLattice, FCCLattice]:
    """Create cubic and FCC lattices with approximately matched node counts.

    The cubic lattice has n³ nodes. The FCC lattice has ~4m³ nodes.
    We choose n and m so both are close to target_nodes.
    """
    n_cubic = max(2, round(target_nodes ** (1/3)))
    # FCC has ~4m³ nodes, so m ≈ (target/4)^(1/3)
    n_fcc = max(2, round((target_nodes / 4) ** (1/3)))

    cubic = CubicLattice(n_cubic)
    fcc = FCCLattice(n_fcc)
    return cubic, fcc


def _avg_shortest_path(G: nx.Graph) -> float | None:
    """Average shortest path length. Returns None if graph is disconnected."""
    if not nx.is_connected(G):
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    if G.number_of_nodes() < 2:
        return None
    return nx.average_shortest_path_length(G)


def _diameter(G: nx.Graph) -> int | None:
    """Graph diameter. Returns None if disconnected."""
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    if G.number_of_nodes() < 2:
        return None
    return nx.diameter(G)


def _fiedler_value(G: nx.Graph) -> float | None:
    """Algebraic connectivity (second-smallest eigenvalue of Laplacian).

    Higher = more robust connectivity. Returns None if too small or disconnected.
    """
    if not nx.is_connected(G):
        return None
    if G.number_of_nodes() < 3:
        return None
    try:
        return nx.algebraic_connectivity(G, method='tracemin_lu')
    except Exception:
        try:
            return nx.algebraic_connectivity(G)
        except Exception:
            return None


def _fault_tolerance(G: nx.Graph, steps: int = 20) -> list[float]:
    """Measure fraction of nodes in largest connected component as
    random nodes are progressively removed.

    Returns a list of (fraction_remaining_connected) at each removal step.
    Steps are evenly spaced from 0% to 50% node removal.
    """
    n = G.number_of_nodes()
    if n < 10:
        return [1.0] * steps

    nodes = list(G.nodes())
    rng = np.random.default_rng(42)
    rng.shuffle(nodes)

    results = []
    G_copy = G.copy()
    remove_per_step = max(1, n // (2 * steps))  # remove up to 50%

    for step in range(steps):
        # Measure current largest CC fraction
        if G_copy.number_of_nodes() == 0:
            results.append(0.0)
            continue
        largest_cc = max(nx.connected_components(G_copy), key=len)
        results.append(len(largest_cc) / n)

        # Remove nodes
        start = step * remove_per_step
        end = min(start + remove_per_step, len(nodes))
        to_remove = [nd for nd in nodes[start:end] if nd in G_copy]
        G_copy.remove_nodes_from(to_remove)

    return results


def run_benchmark(target_nodes: int, compute_paths: bool = True) -> BenchmarkResult:
    """Run full benchmark comparison at a given scale.

    Parameters
    ----------
    target_nodes : int
        Approximate number of nodes for each lattice.
    compute_paths : bool
        If False, skip expensive path computations (for large lattices).
    """
    t0 = time.time()

    cubic, fcc = matched_lattices(target_nodes)
    Gc = cubic.to_networkx()
    Gf = fcc.to_networkx()

    cs = cubic.stats()
    fs = fcc.stats()

    # Path-based metrics (expensive for large graphs)
    cubic_avg_path = None
    fcc_avg_path = None
    cubic_diam = None
    fcc_diam = None

    if compute_paths:
        cubic_avg_path = _avg_shortest_path(Gc)
        fcc_avg_path = _avg_shortest_path(Gf)
        cubic_diam = _diameter(Gc)
        fcc_diam = _diameter(Gf)

    # Algebraic connectivity
    cubic_fiedler = _fiedler_value(Gc)
    fcc_fiedler = _fiedler_value(Gf)

    # Fault tolerance
    cubic_fault = _fault_tolerance(Gc)
    fcc_fault = _fault_tolerance(Gf)

    elapsed = time.time() - t0

    return BenchmarkResult(
        scale=target_nodes,
        cubic_nodes=cs.node_count,
        fcc_nodes=fs.node_count,
        cubic_edges=cs.edge_count,
        fcc_edges=fs.edge_count,
        cubic_avg_degree=cs.avg_degree,
        fcc_avg_degree=fs.avg_degree,
        cubic_avg_path=cubic_avg_path,
        fcc_avg_path=fcc_avg_path,
        cubic_diameter=cubic_diam,
        fcc_diameter=fcc_diam,
        cubic_fiedler=cubic_fiedler,
        fcc_fiedler=fcc_fiedler,
        cubic_fault_curve=cubic_fault,
        fcc_fault_curve=fcc_fault,
        elapsed_seconds=elapsed,
    )


def print_result(r: BenchmarkResult):
    """Print a single benchmark result as a formatted comparison."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: ~{r.scale} target nodes ({r.elapsed_seconds:.1f}s)")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Cubic':>12} {'FCC':>12} {'Ratio':>8}")
    print(f"{'-'*62}")
    print(f"{'Nodes':<30} {r.cubic_nodes:>12} {r.fcc_nodes:>12} {r.fcc_nodes/r.cubic_nodes:>8.2f}")
    print(f"{'Edges':<30} {r.cubic_edges:>12} {r.fcc_edges:>12} {r.fcc_edges/max(1,r.cubic_edges):>8.2f}")
    print(f"{'Avg degree':<30} {r.cubic_avg_degree:>12.2f} {r.fcc_avg_degree:>12.2f} {r.fcc_avg_degree/r.cubic_avg_degree:>8.2f}")

    if r.cubic_avg_path is not None and r.fcc_avg_path is not None:
        print(f"{'Avg shortest path':<30} {r.cubic_avg_path:>12.3f} {r.fcc_avg_path:>12.3f} {r.fcc_avg_path/r.cubic_avg_path:>8.3f}")
    if r.cubic_diameter is not None and r.fcc_diameter is not None:
        print(f"{'Diameter':<30} {r.cubic_diameter:>12} {r.fcc_diameter:>12} {r.fcc_diameter/r.cubic_diameter:>8.3f}")
    if r.cubic_fiedler is not None and r.fcc_fiedler is not None:
        print(f"{'Algebraic connectivity':<30} {r.cubic_fiedler:>12.4f} {r.fcc_fiedler:>12.4f} {r.fcc_fiedler/r.cubic_fiedler:>8.2f}")

    # Fault tolerance summary: fraction connected after 25% and 50% removal
    mid = len(r.cubic_fault_curve) // 2
    end = len(r.cubic_fault_curve) - 1
    if mid > 0 and end > 0:
        print(f"{'Connected @ 25% removed':<30} {r.cubic_fault_curve[mid]:>12.3f} {r.fcc_fault_curve[mid]:>12.3f}")
        print(f"{'Connected @ ~50% removed':<30} {r.cubic_fault_curve[end]:>12.3f} {r.fcc_fault_curve[end]:>12.3f}")


def run_suite(scales: list[int] | None = None) -> list[BenchmarkResult]:
    """Run benchmarks at multiple scales."""
    if scales is None:
        scales = [125, 1000, 8000]

    results = []
    for scale in scales:
        compute_paths = scale <= 10000  # skip expensive paths for large graphs
        print(f"\nRunning benchmark at scale ~{scale}...")
        r = run_benchmark(scale, compute_paths=compute_paths)
        print_result(r)
        results.append(r)

    return results


if __name__ == "__main__":
    run_suite()
