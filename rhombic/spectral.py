"""
Weighted Laplacian spectral analysis for graphs.

The weighted Laplacian L_w encodes the graph's connectivity and edge weights:
  L_w[i,j] = -w_{ij}           if (i,j) is an edge
  L_w[i,i] = sum of weights of edges incident to i

Eigenvalues are real and non-negative (positive semidefinite for connected graphs).
The Fiedler value (second smallest eigenvalue) measures algebraic connectivity —
higher means more robust information flow.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def weighted_laplacian(n_vertices: int,
                       edges: list[tuple[int, int]],
                       edge_weights: list[float] | np.ndarray | None = None) -> np.ndarray:
    """Construct the weighted Laplacian matrix.

    Parameters
    ----------
    n_vertices : number of vertices
    edges : list of (u, v) pairs
    edge_weights : weight for each edge; defaults to all 1.0

    Returns
    -------
    L : (n_vertices, n_vertices) symmetric positive semidefinite matrix
    """
    L = np.zeros((n_vertices, n_vertices), dtype=np.float64)

    if edge_weights is None:
        weights = [1.0] * len(edges)
    else:
        weights = list(edge_weights)

    for idx, (u, v) in enumerate(edges):
        w = weights[idx]
        L[u, v] -= w
        L[v, u] -= w
        L[u, u] += w
        L[v, v] += w

    return L


def spectrum(n_vertices: int,
             edges: list[tuple[int, int]],
             edge_weights: list[float] | np.ndarray | None = None) -> np.ndarray:
    """Sorted eigenvalues of the weighted Laplacian.

    Returns eigenvalues in ascending order.
    """
    L = weighted_laplacian(n_vertices, edges, edge_weights)
    eigenvalues = np.linalg.eigvalsh(L)
    return np.sort(eigenvalues)


def fiedler_value(n_vertices: int,
                  edges: list[tuple[int, int]],
                  edge_weights: list[float] | np.ndarray | None = None) -> float:
    """Second smallest eigenvalue of the weighted Laplacian.

    Positive for connected graphs. Measures algebraic connectivity.
    """
    eigs = spectrum(n_vertices, edges, edge_weights)
    return float(eigs[1])


def spectral_gap(n_vertices: int,
                 edges: list[tuple[int, int]],
                 edge_weights: list[float] | np.ndarray | None = None) -> float:
    """Ratio lambda_2 / lambda_max.

    A larger spectral gap indicates faster mixing / consensus convergence.
    """
    eigs = spectrum(n_vertices, edges, edge_weights)
    lambda_2 = eigs[1]
    lambda_max = eigs[-1]
    if lambda_max < 1e-10:
        return 0.0
    return float(lambda_2 / lambda_max)


@dataclass
class SpectrumComparison:
    """Spectral comparison between two graphs under different weight distributions."""
    distribution_name: str
    graph1_name: str
    graph2_name: str
    graph1_fiedler: float
    graph2_fiedler: float
    graph1_gap: float
    graph2_gap: float
    graph1_spectrum: list[float]
    graph2_spectrum: list[float]


def compare_spectra(graphs: dict[str, dict],
                    distributions: dict[str, list[float]]) -> list[SpectrumComparison]:
    """Compare spectra of multiple graphs under multiple weight distributions.

    Parameters
    ----------
    graphs : dict mapping name -> {'n_vertices': int, 'edges': list}
    distributions : dict mapping name -> list of edge weights

    Returns
    -------
    List of SpectrumComparison for each (distribution, graph pair).
    """
    graph_names = list(graphs.keys())
    results = []

    for dist_name, weights in distributions.items():
        spectra = {}
        fiedlers = {}
        gaps = {}

        for gname, gdata in graphs.items():
            nv = gdata['n_vertices']
            ed = gdata['edges']
            # Weights may need truncation/extension if graphs differ in edge count
            n_edges = len(ed)
            w = weights[:n_edges] if len(weights) >= n_edges else weights + [1.0] * (n_edges - len(weights))

            s = spectrum(nv, ed, w)
            spectra[gname] = s.tolist()
            fiedlers[gname] = float(s[1]) if len(s) > 1 else 0.0
            gaps[gname] = float(s[1] / s[-1]) if s[-1] > 1e-10 else 0.0

        # Produce pairwise comparisons
        for i in range(len(graph_names)):
            for j in range(i + 1, len(graph_names)):
                g1, g2 = graph_names[i], graph_names[j]
                results.append(SpectrumComparison(
                    distribution_name=dist_name,
                    graph1_name=g1,
                    graph2_name=g2,
                    graph1_fiedler=fiedlers[g1],
                    graph2_fiedler=fiedlers[g2],
                    graph1_gap=gaps[g1],
                    graph2_gap=gaps[g2],
                    graph1_spectrum=spectra[g1],
                    graph2_spectrum=spectra[g2],
                ))

    return results


# ── Extended spectral analysis ──────────────────────────────────────


def eigenvalue_multiplicity_pattern(
        eigenvalues: np.ndarray,
        tol: float = 1e-6) -> list[tuple[float, int]]:
    """Group eigenvalues by approximate equality and count multiplicities.

    Returns list of (representative_value, multiplicity) sorted ascending.
    """
    if len(eigenvalues) == 0:
        return []
    eigs = np.sort(eigenvalues)
    groups: list[tuple[float, int]] = []
    current_val = eigs[0]
    current_count = 1

    for i in range(1, len(eigs)):
        if abs(eigs[i] - current_val) < tol:
            current_count += 1
        else:
            groups.append((float(current_val), current_count))
            current_val = eigs[i]
            current_count = 1
    groups.append((float(current_val), current_count))
    return groups


def spectral_distance(spec1: np.ndarray, spec2: np.ndarray) -> float:
    """L2 distance between two spectra (Euclidean in eigenvalue space).

    If spectra differ in length, the shorter one is zero-padded.
    Both are sorted ascending before comparison.
    """
    s1 = np.sort(np.asarray(spec1))
    s2 = np.sort(np.asarray(spec2))
    n = max(len(s1), len(s2))
    padded1 = np.zeros(n)
    padded2 = np.zeros(n)
    padded1[:len(s1)] = s1
    padded2[:len(s2)] = s2
    return float(np.linalg.norm(padded1 - padded2))


@dataclass
class SpectrumSummary:
    """Summary statistics for a single graph's spectrum."""
    graph_name: str
    n_vertices: int
    n_edges: int
    fiedler: float
    spectral_gap: float
    lambda_max: float
    n_distinct_eigenvalues: int
    multiplicity_pattern: list[tuple[float, int]]
    full_spectrum: list[float]


def spectrum_summary(
        graph_name: str,
        n_vertices: int,
        edges: list[tuple[int, int]],
        edge_weights: list[float] | np.ndarray | None = None,
        tol: float = 1e-6) -> SpectrumSummary:
    """Compute comprehensive spectral summary for a graph."""
    eigs = spectrum(n_vertices, edges, edge_weights)
    fv = float(eigs[1]) if len(eigs) > 1 else 0.0
    lmax = float(eigs[-1])
    gap = fv / lmax if lmax > 1e-10 else 0.0
    pattern = eigenvalue_multiplicity_pattern(eigs, tol=tol)

    return SpectrumSummary(
        graph_name=graph_name,
        n_vertices=n_vertices,
        n_edges=len(edges),
        fiedler=fv,
        spectral_gap=gap,
        lambda_max=lmax,
        n_distinct_eigenvalues=len(pattern),
        multiplicity_pattern=pattern,
        full_spectrum=eigs.tolist(),
    )
