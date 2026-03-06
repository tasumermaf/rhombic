"""
Optimal edge-weight assignment on graphs via total variation minimization.

Given a set of weights and a graph, find the assignment of weights to edges
that minimizes total variation: the sum over all vertices of pairwise absolute
differences between weights on incident edges.

Lower TV means similar values cluster on adjacent edges — the topology
naturally "sorts" the weights.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def total_variation(vertex_edges: dict[int, list[int]],
                    edge_weights: list[float] | np.ndarray) -> float:
    """Compute total variation of an edge-weight assignment.

    TV = sum over all vertices v of:
         sum over pairs (i, j) of edges incident to v of |w_i - w_j|

    Parameters
    ----------
    vertex_edges : dict mapping vertex index -> list of edge indices
    edge_weights : weight assigned to each edge (indexed by edge position)
    """
    w = np.asarray(edge_weights, dtype=np.float64)
    tv = 0.0
    for v, incident in vertex_edges.items():
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                tv += abs(w[incident[i]] - w[incident[j]])
    return tv


def _swap_tv_delta(vertex_edges: dict[int, list[int]],
                   edge_weights: np.ndarray,
                   edge_to_vertices: list[list[int]],
                   e1: int, e2: int) -> float:
    """Compute the change in TV if edges e1 and e2 swap weights.

    Only vertices incident to e1 or e2 are affected.
    Returns new_tv - old_tv (negative means improvement).
    """
    affected_verts = set()
    for v in edge_to_vertices[e1]:
        affected_verts.add(v)
    for v in edge_to_vertices[e2]:
        affected_verts.add(v)

    # Compute local TV before swap
    old_local = 0.0
    for v in affected_verts:
        incident = vertex_edges[v]
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                old_local += abs(edge_weights[incident[i]] - edge_weights[incident[j]])

    # Swap
    edge_weights[e1], edge_weights[e2] = edge_weights[e2], edge_weights[e1]

    # Compute local TV after swap
    new_local = 0.0
    for v in affected_verts:
        incident = vertex_edges[v]
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                new_local += abs(edge_weights[incident[i]] - edge_weights[incident[j]])

    # Swap back
    edge_weights[e1], edge_weights[e2] = edge_weights[e2], edge_weights[e1]

    return new_local - old_local


def optimal_assignment(vertex_edges: dict[int, list[int]],
                       edges: list[tuple[int, int]],
                       weights: list[float],
                       n_restarts: int = 100,
                       max_iters: int = 5000,
                       seed: int = 42) -> tuple[list[float], float]:
    """Find edge-weight assignment minimizing total variation.

    Uses simulated annealing with random restarts (24! ~ 6.2e23 makes
    brute force infeasible).

    Parameters
    ----------
    vertex_edges : dict mapping vertex -> list of incident edge indices
    edges : list of (u, v) edge pairs
    weights : the values to assign (one per edge)
    n_restarts : number of random restart attempts
    max_iters : SA iterations per restart
    seed : random seed for reproducibility

    Returns
    -------
    (best_assignment, best_tv) : the best weight ordering found and its TV
    """
    rng = np.random.default_rng(seed)
    n = len(weights)
    w = np.array(weights, dtype=np.float64)

    # Precompute edge -> incident vertices
    edge_to_vertices: list[list[int]] = [[] for _ in range(n)]
    for v, incident in vertex_edges.items():
        for e_idx in incident:
            edge_to_vertices[e_idx].append(v)

    best_assignment = w.copy()
    best_tv = total_variation(vertex_edges, best_assignment)

    for restart in range(n_restarts):
        # Random initial permutation
        current = w.copy()
        rng.shuffle(current)
        current_tv = total_variation(vertex_edges, current)

        # SA schedule
        temp = current_tv * 0.5  # initial temperature
        cooling = 0.995

        for it in range(max_iters):
            # Random swap
            e1, e2 = rng.choice(n, size=2, replace=False)
            delta = _swap_tv_delta(vertex_edges, current, edge_to_vertices, e1, e2)

            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                current[e1], current[e2] = current[e2], current[e1]
                current_tv += delta

            temp *= cooling

        if current_tv < best_tv:
            best_tv = current_tv
            best_assignment = current.copy()

    return best_assignment.tolist(), float(best_tv)


def random_assignment_tv(vertex_edges: dict[int, list[int]],
                         weights: list[float],
                         n_samples: int = 10000,
                         seed: int = 42) -> np.ndarray:
    """Compute TV distribution under random permutations (null model).

    Returns array of n_samples TV values.
    """
    rng = np.random.default_rng(seed)
    w = np.array(weights, dtype=np.float64)
    tvs = np.empty(n_samples)

    for i in range(n_samples):
        perm = w.copy()
        rng.shuffle(perm)
        tvs[i] = total_variation(vertex_edges, perm)

    return tvs


@dataclass
class AssignmentResult:
    """Results from optimal assignment experiment."""
    graph_name: str
    n_edges: int
    optimal_tv: float
    random_mean_tv: float
    random_std_tv: float
    p_value: float             # fraction of random samples <= optimal
    best_assignment: list[float]


def compare_graphs(graphs: dict[str, dict],
                   weights: list[float],
                   n_restarts: int = 100,
                   n_null_samples: int = 10000,
                   seed: int = 42) -> list[AssignmentResult]:
    """Compare optimal assignment across multiple graphs.

    Parameters
    ----------
    graphs : dict mapping name -> {'vertex_edges': dict, 'edges': list}
    weights : the values to assign
    n_restarts : SA restarts per graph
    n_null_samples : random permutations for null distribution
    seed : random seed

    Returns
    -------
    List of AssignmentResult, one per graph.
    """
    results = []
    for name, gdata in graphs.items():
        ve = gdata['vertex_edges']
        ed = gdata['edges']

        best_assign, best_tv = optimal_assignment(
            ve, ed, weights, n_restarts=n_restarts, seed=seed)

        null_tvs = random_assignment_tv(ve, weights, n_samples=n_null_samples, seed=seed)
        p_value = float(np.mean(null_tvs <= best_tv))

        results.append(AssignmentResult(
            graph_name=name,
            n_edges=len(ed),
            optimal_tv=best_tv,
            random_mean_tv=float(null_tvs.mean()),
            random_std_tv=float(null_tvs.std()),
            p_value=p_value,
            best_assignment=best_assign,
        ))

    return results
