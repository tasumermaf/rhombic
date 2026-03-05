"""
Rung 2: Spatial operation benchmarks comparing cubic and FCC lattices.

Metrics:
  - Nearest-neighbor search (brute force + lattice-aware)
  - Range query: sphere and box
  - Flood fill / region growing
  - Spatial hashing: point-to-cell lookup

The question: does the 30% routing advantage from Rung 1 translate
to faster spatial computation? If yes, the FCC lattice is a drop-in
replacement for cubic grids in game engines, physics sims, medical imaging.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import KDTree
from rhombic.lattice import CubicLattice, FCCLattice


@dataclass
class SpatialResult:
    """Results from a single spatial benchmark comparison."""
    scale: int
    cubic_nodes: int
    fcc_nodes: int

    # Nearest-neighbor search (average over N query points)
    cubic_nn_time: float          # seconds for N queries
    fcc_nn_time: float
    cubic_nn_accuracy: float      # fraction finding true nearest
    fcc_nn_accuracy: float

    # Range query: sphere
    cubic_sphere_time: float
    fcc_sphere_time: float
    cubic_sphere_count: float     # avg nodes returned per query
    fcc_sphere_count: float

    # Range query: box
    cubic_box_time: float
    fcc_box_time: float
    cubic_box_count: float
    fcc_box_count: float

    # Flood fill
    cubic_flood_time: float
    fcc_flood_time: float
    cubic_flood_reached: int      # nodes reached within radius
    fcc_flood_reached: int

    # Spatial hashing
    cubic_hash_build_time: float
    fcc_hash_build_time: float
    cubic_hash_query_time: float
    fcc_hash_query_time: float

    elapsed_seconds: float


# ── Spatial operations ───────────────────────────────────────────────


def nearest_neighbor_search(positions: np.ndarray, query_points: np.ndarray
                            ) -> tuple[np.ndarray, float]:
    """Find nearest lattice node for each query point using KDTree.

    Returns (indices, elapsed_seconds).
    """
    tree = KDTree(positions)
    t0 = time.perf_counter()
    _, indices = tree.query(query_points)
    elapsed = time.perf_counter() - t0
    return indices, elapsed


def range_query_sphere(positions: np.ndarray, center: np.ndarray,
                       radius: float) -> np.ndarray:
    """Return indices of all nodes within radius of center."""
    dists = np.linalg.norm(positions - center, axis=1)
    return np.where(dists <= radius)[0]


def range_query_box(positions: np.ndarray, box_min: np.ndarray,
                    box_max: np.ndarray) -> np.ndarray:
    """Return indices of all nodes within axis-aligned box."""
    mask = np.all((positions >= box_min) & (positions <= box_max), axis=1)
    return np.where(mask)[0]


def flood_fill(adjacency: dict[int, list[int]], start: int,
               max_hops: int) -> set[int]:
    """BFS flood fill from start node, up to max_hops steps.

    Returns set of reached node indices.
    """
    visited = {start}
    frontier = {start}
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def build_adjacency(edges: list[tuple[int, int]], node_count: int
                    ) -> dict[int, list[int]]:
    """Build adjacency list from edge list."""
    adj: dict[int, list[int]] = {i: [] for i in range(node_count)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


class SpatialHash:
    """Simple uniform grid spatial hash for point-to-cell lookup."""

    def __init__(self, positions: np.ndarray, cell_size: float):
        self.cell_size = cell_size
        self.positions = positions
        self.cells: dict[tuple, list[int]] = {}

        t0 = time.perf_counter()
        for idx, pos in enumerate(positions):
            key = self._hash(pos)
            if key not in self.cells:
                self.cells[key] = []
            self.cells[key].append(idx)
        self.build_time = time.perf_counter() - t0

    def _hash(self, pos: np.ndarray) -> tuple:
        return tuple((pos / self.cell_size).astype(int))

    def query(self, point: np.ndarray) -> list[int]:
        """Return all node indices in the same cell as point."""
        key = self._hash(point)
        return self.cells.get(key, [])

    def query_neighborhood(self, point: np.ndarray) -> list[int]:
        """Return all node indices in the cell and its 26 neighbors."""
        base = (point / self.cell_size).astype(int)
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (base[0]+dx, base[1]+dy, base[2]+dz)
                    result.extend(self.cells.get(key, []))
        return result


# ── Benchmark runner ─────────────────────────────────────────────────


def run_spatial_benchmark(target_nodes: int, n_queries: int = 1000,
                          seed: int = 42) -> SpatialResult:
    """Run full spatial benchmark comparison at a given scale."""
    from rhombic.lattice import CubicLattice, FCCLattice
    from rhombic.benchmark import matched_lattices

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    cubic, fcc = matched_lattices(target_nodes)
    pos_c = cubic.positions
    pos_f = fcc.positions

    # Bounding box for generating random query points
    bb_min = np.minimum(pos_c.min(axis=0), pos_f.min(axis=0))
    bb_max = np.maximum(pos_c.max(axis=0), pos_f.max(axis=0))
    extent = bb_max - bb_min

    # Generate shared random query points within bounding volume
    query_points = rng.uniform(bb_min, bb_max, size=(n_queries, 3))

    # ── 1. Nearest-neighbor search ───────────────────────────────
    # Build KDTrees (not timed — we time query only)
    tree_c = KDTree(pos_c)
    tree_f = KDTree(pos_f)

    t_c = time.perf_counter()
    dist_c, idx_c = tree_c.query(query_points)
    cubic_nn_time = time.perf_counter() - t_c

    t_f = time.perf_counter()
    dist_f, idx_f = tree_f.query(query_points)
    fcc_nn_time = time.perf_counter() - t_f

    # Accuracy: brute-force verify a sample
    sample_size = min(100, n_queries)
    sample_idx = rng.choice(n_queries, sample_size, replace=False)
    cubic_correct = 0
    fcc_correct = 0
    for si in sample_idx:
        qp = query_points[si]
        true_c = np.argmin(np.linalg.norm(pos_c - qp, axis=1))
        true_f = np.argmin(np.linalg.norm(pos_f - qp, axis=1))
        if idx_c[si] == true_c:
            cubic_correct += 1
        if idx_f[si] == true_f:
            fcc_correct += 1

    # ── 2. Range query: sphere ───────────────────────────────────
    # Use radius = 15% of bounding box diagonal
    diag = np.linalg.norm(extent)
    radius = diag * 0.15
    n_range = min(100, n_queries)
    centers = query_points[:n_range]

    t_c = time.perf_counter()
    sphere_counts_c = []
    for c in centers:
        result = range_query_sphere(pos_c, c, radius)
        sphere_counts_c.append(len(result))
    cubic_sphere_time = time.perf_counter() - t_c

    t_f = time.perf_counter()
    sphere_counts_f = []
    for c in centers:
        result = range_query_sphere(pos_f, c, radius)
        sphere_counts_f.append(len(result))
    fcc_sphere_time = time.perf_counter() - t_f

    # ── 3. Range query: box ──────────────────────────────────────
    box_half = extent * 0.15  # 30% of extent per axis
    t_c = time.perf_counter()
    box_counts_c = []
    for c in centers:
        result = range_query_box(pos_c, c - box_half, c + box_half)
        box_counts_c.append(len(result))
    cubic_box_time = time.perf_counter() - t_c

    t_f = time.perf_counter()
    box_counts_f = []
    for c in centers:
        result = range_query_box(pos_f, c - box_half, c + box_half)
        box_counts_f.append(len(result))
    fcc_box_time = time.perf_counter() - t_f

    # ── 4. Flood fill ────────────────────────────────────────────
    # Flood fill from a central node, same hop count for both
    adj_c = build_adjacency(cubic.edges, cubic.node_count)
    adj_f = build_adjacency(fcc.edges, fcc.node_count)

    # Find the node nearest to the center of the bounding box
    center = (bb_min + bb_max) / 2
    start_c = np.argmin(np.linalg.norm(pos_c - center, axis=1))
    start_f = np.argmin(np.linalg.norm(pos_f - center, axis=1))

    # Use hop count = cube root of node count (comparable depth)
    max_hops = max(2, int(round(target_nodes ** (1/3) / 2)))

    t_c = time.perf_counter()
    reached_c = flood_fill(adj_c, start_c, max_hops)
    cubic_flood_time = time.perf_counter() - t_c

    t_f = time.perf_counter()
    reached_f = flood_fill(adj_f, start_f, max_hops)
    fcc_flood_time = time.perf_counter() - t_f

    # ── 5. Spatial hashing ───────────────────────────────────────
    # Cell size = mean nearest-neighbor distance
    cell_size_c = cubic.spacing if hasattr(cubic, 'spacing') else 1.0
    cell_size_f = fcc.a if hasattr(fcc, 'a') else 1.0

    hash_c = SpatialHash(pos_c, cell_size_c)
    hash_f = SpatialHash(pos_f, cell_size_f)

    # Time N queries
    t_c = time.perf_counter()
    for qp in query_points:
        hash_c.query_neighborhood(qp)
    cubic_hash_query_time = time.perf_counter() - t_c

    t_f = time.perf_counter()
    for qp in query_points:
        hash_f.query_neighborhood(qp)
    fcc_hash_query_time = time.perf_counter() - t_f

    elapsed = time.perf_counter() - t0

    return SpatialResult(
        scale=target_nodes,
        cubic_nodes=cubic.node_count,
        fcc_nodes=fcc.node_count,
        cubic_nn_time=cubic_nn_time,
        fcc_nn_time=fcc_nn_time,
        cubic_nn_accuracy=cubic_correct / sample_size,
        fcc_nn_accuracy=fcc_correct / sample_size,
        cubic_sphere_time=cubic_sphere_time,
        fcc_sphere_time=fcc_sphere_time,
        cubic_sphere_count=np.mean(sphere_counts_c),
        fcc_sphere_count=np.mean(sphere_counts_f),
        cubic_box_time=cubic_box_time,
        fcc_box_time=fcc_box_time,
        cubic_box_count=np.mean(box_counts_c),
        fcc_box_count=np.mean(box_counts_f),
        cubic_flood_time=cubic_flood_time,
        fcc_flood_time=fcc_flood_time,
        cubic_flood_reached=len(reached_c),
        fcc_flood_reached=len(reached_f),
        cubic_hash_build_time=hash_c.build_time,
        fcc_hash_build_time=hash_f.build_time,
        cubic_hash_query_time=cubic_hash_query_time,
        fcc_hash_query_time=fcc_hash_query_time,
        elapsed_seconds=elapsed,
    )


def print_spatial_result(r: SpatialResult):
    """Print a single spatial benchmark result."""
    print(f"\n{'='*70}")
    print(f"SPATIAL BENCHMARK: ~{r.scale} target nodes ({r.elapsed_seconds:.1f}s)")
    print(f"{'='*70}")
    print(f"Nodes: Cubic={r.cubic_nodes}, FCC={r.fcc_nodes}")
    print()

    print(f"{'Metric':<35} {'Cubic':>12} {'FCC':>12} {'Ratio':>8}")
    print(f"{'-'*67}")

    # Nearest neighbor
    print(f"{'NN query time (s)':<35} {r.cubic_nn_time:>12.6f} {r.fcc_nn_time:>12.6f} {r.fcc_nn_time/max(1e-9,r.cubic_nn_time):>8.2f}")
    print(f"{'NN accuracy':<35} {r.cubic_nn_accuracy:>12.3f} {r.fcc_nn_accuracy:>12.3f}")

    # Sphere range query
    print(f"{'Sphere query time (s)':<35} {r.cubic_sphere_time:>12.6f} {r.fcc_sphere_time:>12.6f} {r.fcc_sphere_time/max(1e-9,r.cubic_sphere_time):>8.2f}")
    print(f"{'Sphere avg count':<35} {r.cubic_sphere_count:>12.1f} {r.fcc_sphere_count:>12.1f} {r.fcc_sphere_count/max(1,r.cubic_sphere_count):>8.2f}")

    # Box range query
    print(f"{'Box query time (s)':<35} {r.cubic_box_time:>12.6f} {r.fcc_box_time:>12.6f} {r.fcc_box_time/max(1e-9,r.cubic_box_time):>8.2f}")
    print(f"{'Box avg count':<35} {r.cubic_box_count:>12.1f} {r.fcc_box_count:>12.1f} {r.fcc_box_count/max(1,r.cubic_box_count):>8.2f}")

    # Flood fill
    print(f"{'Flood fill time (s)':<35} {r.cubic_flood_time:>12.6f} {r.fcc_flood_time:>12.6f} {r.fcc_flood_time/max(1e-9,r.cubic_flood_time):>8.2f}")
    print(f"{'Flood fill reached':<35} {r.cubic_flood_reached:>12} {r.fcc_flood_reached:>12} {r.fcc_flood_reached/max(1,r.cubic_flood_reached):>8.2f}")

    # Spatial hashing
    print(f"{'Hash build time (s)':<35} {r.cubic_hash_build_time:>12.6f} {r.fcc_hash_build_time:>12.6f} {r.fcc_hash_build_time/max(1e-9,r.cubic_hash_build_time):>8.2f}")
    print(f"{'Hash query time (s)':<35} {r.cubic_hash_query_time:>12.6f} {r.fcc_hash_query_time:>12.6f} {r.fcc_hash_query_time/max(1e-9,r.cubic_hash_query_time):>8.2f}")


def run_spatial_suite(scales: list[int] | None = None) -> list[SpatialResult]:
    """Run spatial benchmarks at multiple scales."""
    if scales is None:
        scales = [125, 1000, 8000]

    results = []
    for scale in scales:
        print(f"\nRunning spatial benchmark at scale ~{scale}...")
        r = run_spatial_benchmark(scale)
        print_spatial_result(r)
        results.append(r)

    return results


if __name__ == "__main__":
    run_spatial_suite()
