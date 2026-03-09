# Rhombic v0.3.0 — Complete API Reference

Generated from live introspection on Kallisti. 208/208 tests passing.

**Install**: `pip install -e .` (from repo root, Python 3.10+)
**Requires**: numpy, scipy, networkx
**Optional**: matplotlib (for visualize module)

---

## rhombic.lattice

```python
from rhombic.lattice import CubicLattice, FCCLattice, LatticeStats

CubicLattice(n: int, spacing: float = 1.0)
  .edge_directions() -> list[np.ndarray]
  .edges() -> list[tuple[int, int]]
  .node_count() -> int
  .positions() -> np.ndarray
  .stats() -> LatticeStats
  .to_networkx() -> nx.Graph
  # Properties: .adjacency, .dim

FCCLattice(n: int, a: float = 1.0)
  .edge_directions() -> list[np.ndarray]
  .edges() -> list[tuple[int, int]]
  .node_count() -> int
  .positions() -> np.ndarray
  .stats() -> LatticeStats
  .to_networkx() -> nx.Graph
  # Properties: .adjacency, .dim

LatticeStats (dataclass):
  .n_nodes: int
  .n_edges: int
  .degree: int
  .diameter: int
  .positions: np.ndarray
```

**Common mistakes**: No `compare_lattices()` function. Use `rhombic.benchmark.matched_lattices()` + `run_benchmark()` instead.

---

## rhombic.spectral

```python
from rhombic.spectral import (
    fiedler_value, spectrum, weighted_laplacian,
    spectral_gap, compare_spectra, spectrum_summary,
    SpectrumComparison, SpectrumSummary
)

fiedler_value(
    n_vertices: int,
    edges: list[tuple[int, int]],
    edge_weights: list[float] | np.ndarray | None = None
) -> float

spectrum(
    n_vertices: int,
    edges: list[tuple[int, int]],
    edge_weights: list[float] | np.ndarray | None = None
) -> np.ndarray

weighted_laplacian(
    n_vertices: int,
    edges: list[tuple[int, int]],
    edge_weights: list[float] | np.ndarray | None = None
) -> np.ndarray

spectral_gap(
    n_vertices: int,
    edges: list[tuple[int, int]],
    edge_weights: list[float] | np.ndarray | None = None
) -> float

compare_spectra(
    graphs: dict[str, dict],
    distributions: dict[str, list[float]]
) -> list[SpectrumComparison]

spectrum_summary(
    graph_name: str,
    n_vertices: int,
    edges: list[tuple[int, int]],
    edge_weights: list[float] | np.ndarray | None = None,
    tol: float = 1e-10
) -> SpectrumSummary
```

**Common mistakes**:
- `weighted_fiedler()` does NOT exist — use `fiedler_value()` with `edge_weights` param
- All functions take `(n_vertices, edges, edge_weights)` — NOT `(graph, weights)`

---

## rhombic.corpus

```python
from rhombic.corpus import (
    TRUMP_VALUES,       # dict[24 entries] — card name → numeric value
    TRUMP_NAMES,        # dict[24 entries] — card name → display name
    TRUMP_INSCRIPTIONS, # dict[24 entries] — card name → inscription text
    NAMES_OF_POWER,     # dict[14 entries]
    TRACKED_PRIMES,     # list[8 entries]
    CANONICAL_EDGE_ORDER, # list[24 entries]
    CorpusStats,
    direction_weights,
    edge_values,
    weight_distributions,
    corpus_stats,
    prime_factors,
    prime_factor_set,
    prime_membership,
    shared_factors,
)

direction_weights(values: list[float], n_directions: int) -> list[float]
edge_values() -> list[int]
weight_distributions(seed: int = 42) -> dict[str, list[float]]
corpus_stats() -> CorpusStats
prime_factors(value: int) -> list[int]
prime_factor_set(value: int) -> set[int]
prime_membership(value: int, prime: int) -> bool
shared_factors(v1: int, v2: int) -> set[int]
```

**Common mistakes**:
- `CORPUS_VALUES` does NOT exist — use `TRUMP_VALUES`
- `direction_pair_weights` is NOT in spatial — it's `direction_weights` in corpus

---

## rhombic.benchmark

```python
from rhombic.benchmark import (
    matched_lattices, run_benchmark, run_suite, BenchmarkResult
)

matched_lattices(target_nodes: int) -> tuple[CubicLattice, FCCLattice]

run_benchmark(
    target_nodes: int = 500,
    compute_paths: bool = True
) -> BenchmarkResult

run_suite(
    scales: list[int] | None = None
) -> list[BenchmarkResult]

BenchmarkResult (dataclass):
  .cubic: LatticeStats
  .fcc: LatticeStats
  .fiedler_cubic: float
  .fiedler_fcc: float
  .spectral_gap_cubic: float
  .spectral_gap_fcc: float
  .fault_tolerance_cubic: float
  .fault_tolerance_fcc: float
  # ... plus diameter, path length fields
```

**Common mistakes**: `compare_lattices()` does NOT exist. Use `matched_lattices()` to get size-matched pairs, then `run_benchmark()` for full comparison.

---

## rhombic.assignment

```python
from rhombic.assignment import (
    optimal_assignment, optimal_prime_assignment,
    total_variation, compare_graphs, prime_vertex_score
)

optimal_assignment(
    values: list[float],
    graph: nx.Graph,
    maximize: bool = True
) -> tuple[dict, float]

optimal_prime_assignment(
    values: dict[str, int],
    graph: nx.Graph,
    primes: list[int] | None = None,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2
) -> dict[int, str]

total_variation(
    assignment: dict,
    graph: nx.Graph,
    values: list[float]
) -> float

compare_graphs(
    values: list[float],
    graphs: dict[str, nx.Graph],
    maximize: bool = True
) -> dict[str, dict]

prime_vertex_score(
    card_value: int,
    neighbors: list[int],
    primes: list[int]
) -> float
```

**Common mistakes**: `prime_vertex_search()` does NOT exist — use `optimal_prime_assignment()`

---

## rhombic.polyhedron

```python
from rhombic.polyhedron import RhombicDodecahedron, cuboctahedron_graph

RhombicDodecahedron()
  .degree() -> int
  .face_edges(face: tuple) -> list[tuple[int, int]]
  .neighbors(vertex: int) -> list[int]
  .to_networkx() -> nx.Graph
  .vertex_star(vertex: int) -> list[tuple]
  # Properties:
  .edges -> list[tuple[int, int]]
  .faces -> list[tuple]
  .vertices -> list[int]

cuboctahedron_graph() -> nx.Graph
```

**Common mistakes**:
- No `.graph` property — use `.to_networkx()` to get the NetworkX graph
- `plot_rd_3d()` does NOT exist in visualize module

---

## rhombic.spatial

```python
from rhombic.spatial import (
    SpatialHash, flood_fill, nearest_neighbor_search,
    range_query_sphere, range_query_box
)

SpatialHash(cell_size: float = 1.0)
  .insert(point_id: int, position: np.ndarray)
  .query_sphere(center: np.ndarray, radius: float) -> list[int]
  .query_box(lo: np.ndarray, hi: np.ndarray) -> list[int]

flood_fill(
    adjacency: dict[int, list[int]],
    start: int,
    max_hops: int
) -> set[int]

nearest_neighbor_search(
    positions: np.ndarray,
    query: np.ndarray,
    k: int = 1
) -> np.ndarray

range_query_sphere(
    positions: np.ndarray,
    center: np.ndarray,
    radius: float
) -> np.ndarray

range_query_box(
    positions: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray
) -> np.ndarray
```

---

## rhombic.visualize

Requires matplotlib.

```python
from rhombic.visualize import (
    plot_diameter_comparison,
    plot_fault_tolerance,
    plot_fiedler_comparison,
    plot_path_comparison,
    plot_summary_dashboard,
)

plot_diameter_comparison(results: list[BenchmarkResult], save_path: str | None = None)
plot_fault_tolerance(results: list[BenchmarkResult], save_path: str | None = None)
plot_fiedler_comparison(results: list[BenchmarkResult], save_path: str | None = None)
plot_path_comparison(results: list[BenchmarkResult], save_path: str | None = None)
plot_summary_dashboard(results: list[BenchmarkResult], save_path: str | None = None)
```

All plot functions accept a list of `BenchmarkResult` from `run_suite()`.

**Common mistakes**: `plot_rd_3d()` does NOT exist. No 3D plotting in this module.

---

## rhombic.context

Context-aware retrieval benchmarking — compares information diffusion and consensus convergence on cubic vs FCC lattice topologies.

```python
from rhombic.context import (
    run_context_benchmark, run_context_suite, print_context_result,
    ContextResult, RecallResult, DiffusionResult, ConsensusResult,
    assign_to_lattice, consensus_speed, weighted_consensus_speed,
    information_diffusion, weighted_information_diffusion,
    neighborhood_recall, generate_embeddings, project_to_3d,
)

run_context_benchmark(
    target_nodes: int = 500,
    embedding_dim: int = 128,
    n_embeddings: int = 200,
    seed: int = 42
) -> ContextResult

run_context_suite(scales: list[int] | None = None) -> list[ContextResult]

print_context_result(r: ContextResult)

assign_to_lattice(
    points_3d: np.ndarray,
    lattice_positions: np.ndarray
) -> np.ndarray

consensus_speed(
    adjacency: dict[int, list[int]],
    initial_states: np.ndarray,
    epsilon: float = 0.05,
    max_rounds: int = 500
) -> tuple[int, list[float]]

weighted_consensus_speed(
    adjacency: dict[int, list[int]],
    edge_weight: dict[tuple[int, int], float],
    initial_states: np.ndarray,
    epsilon: float = 0.05,
    max_rounds: int = 500
) -> tuple[int, list[float]]

information_diffusion(
    adjacency: dict[int, list[int]],
    source_node: int,
    n_nodes: int,
    max_rounds: int = 30
) -> list[float]

weighted_information_diffusion(
    adjacency: dict[int, list[int]],
    edge_weight: dict[tuple[int, int], float],
    source_node: int,
    n_nodes: int,
    max_rounds: int = 30
) -> list[float]

neighborhood_recall(
    embeddings: np.ndarray,
    points_3d: np.ndarray,
    lattice_positions: np.ndarray,
    adjacency: dict[int, list[int]],
    assignments: np.ndarray,
    k_true: int = 10,
    max_hops: int = 3
) -> list[RecallResult]

generate_embeddings(
    n: int, dim: int,
    n_clusters: int = 5,
    seed: int = 42
) -> np.ndarray

project_to_3d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray
```

---

## rhombic.signal

Signal processing benchmarking — compares sampling reconstruction quality on cubic vs FCC grids.

```python
from rhombic.signal import (
    run_signal_benchmark, run_signal_suite, print_signal_result,
    SignalResult, FrequencySweepPoint,
    cubic_samples, fcc_samples, density_matched_samples,
    directional_signal, isotropic_signal,
    reconstruct, compute_mse, compute_psnr,
)

run_signal_benchmark(
    target_count: int = 500,
    n_eval: int = 800,
    seed: int = 42
) -> SignalResult

run_signal_suite(target_counts: list[int] | None = None) -> list[SignalResult]

print_signal_result(r: SignalResult)

cubic_samples(n: int) -> np.ndarray
fcc_samples(n: int) -> np.ndarray
density_matched_samples(target_count: int) -> tuple[np.ndarray, np.ndarray]

directional_signal(points: np.ndarray, direction: np.ndarray, freq: float) -> np.ndarray
isotropic_signal(points: np.ndarray, freq: float) -> np.ndarray

reconstruct(
    sample_pos: np.ndarray,
    sample_vals: np.ndarray,
    eval_pos: np.ndarray
) -> np.ndarray

compute_mse(true: np.ndarray, recon: np.ndarray) -> float
compute_psnr(true: np.ndarray, recon: np.ndarray) -> float
```

---

## rhombic.index

High-dimensional lattice-based approximate nearest neighbor search.

```python
from rhombic.index import (
    LatticeIndex, CubicIndex, FCCIndex, brute_force_knn
)

LatticeIndex(dim: int, n_cells_per_side: int = 5)
  .build(embeddings: np.ndarray) -> LatticeIndex
  .query(q: np.ndarray, k: int = 10, hops: int = 1) -> np.ndarray
  .recall_at_k(queries: np.ndarray, ground_truth: np.ndarray, k: int = 10, hops: int = 1) -> float

CubicIndex(dim: int, n_cells_per_side: int = 5)  # inherits LatticeIndex
  .from_target_nodes(dim: int, target_nodes: int) -> CubicIndex  # classmethod

FCCIndex(dim: int, n_cells_per_side: int = 5)   # inherits LatticeIndex
  .from_target_nodes(dim: int, target_nodes: int) -> FCCIndex    # classmethod

brute_force_knn(embeddings: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray
```

---

## Quick Fix Guide for Hermes Tools

Meridian's `rhombic_tools.py` has these known mismatches:

| Wrong (in rhombic_tools.py) | Correct |
|---|---|
| `compare_lattices(...)` | `matched_lattices(n)` + `run_benchmark(n)` |
| `weighted_fiedler(graph, weights)` | `fiedler_value(n_vertices, edges, edge_weights)` |
| `CORPUS_VALUES` | `TRUMP_VALUES` |
| `prime_vertex_search(...)` | `optimal_prime_assignment(...)` |
| `plot_rd_3d(...)` | Not available — use `to_networkx()` + custom plotting |
| `RhombicDodecahedron().graph` | `RhombicDodecahedron().to_networkx()` |
| `direction_pair_weights` (from spatial) | `direction_weights` (from corpus) |
| `spectral.spectrum(graph, weights)` | `spectral.spectrum(n_vertices, edges, edge_weights)` |

---
*Generated 2026-03-06 from rhombic v0.3.0, commit 4d2224a, 208/208 tests passing*
