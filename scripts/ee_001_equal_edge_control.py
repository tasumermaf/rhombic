"""
EE-001: Equal-edge-count random-graph control for Paper 1 (FCC vs cubic).

Pre-registered protocol: results/EE-001-equal-edge-control/PROTOCOL.md
(written BEFORE this script was executed — L-006 discipline).

For each FCC lattice size, builds:
  - FCC lattice (the Paper 1 subject)
  - Cubic lattice at matched node count (the Paper 1 baseline, reference)
  - FCC-rewire: degree-preserving double-edge-swap randomization of the FCC
    graph itself (PRIMARY control: exact same degree sequence)
  - Random d-regular graph, d = round(2E/N) (closest regular match)
  - Erdos-Renyi G(N, E) (exact same N and E)

Metrics match Paper 1 (rhombic/benchmark.py) exactly:
  - Fiedler: nx.algebraic_connectivity(G, method='tracemin_lu') on the raw
    unnormalized, unweighted combinatorial Laplacian
  - Mean shortest path: unweighted BFS; exact for N <= EXACT_BFS_LIMIT,
    else sampled from SAMPLE_SOURCES fixed sources (flagged in output)
  - Diameter: exact when exact BFS is run; else max eccentricity over the
    sampled sources (lower-bound proxy, flagged)

Usage:
    python scripts/ee_001_equal_edge_control.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

# Make the rhombic package importable when run from repo root or scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rhombic.lattice import CubicLattice, FCCLattice  # noqa: E402

# ── Configuration (per PROTOCOL.md) ─────────────────────────────────
FCC_SIZES = [4, 6, 8]          # unit cells per side -> N = 4*m^3
SEEDS = [42, 43, 44, 45, 46]   # 5 seeds per random model
EXACT_BFS_LIMIT = 1100         # exact all-pairs BFS at or below this N
SAMPLE_SOURCES = 512           # BFS sources when sampling
MAX_RESAMPLE = 20              # retries for disconnected random samples
REWIRE_SWAP_FACTOR = 5         # nswap = factor * E
OUT_DIR = REPO_ROOT / "results" / "EE-001-equal-edge-control"


# ── Metrics (Paper 1 methodology) ───────────────────────────────────

def fiedler_value(G: nx.Graph) -> float | None:
    """Raw-Laplacian algebraic connectivity, exactly as rhombic/benchmark.py."""
    if not nx.is_connected(G) or G.number_of_nodes() < 3:
        return None
    try:
        return float(nx.algebraic_connectivity(G, weight=None,
                                               method="tracemin_lu"))
    except Exception:
        try:
            return float(nx.algebraic_connectivity(G, weight=None))
        except Exception:
            return None


def path_metrics(G: nx.Graph, rng: np.random.Generator) -> dict:
    """Mean shortest path + diameter (exact or sampled, per protocol)."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    exact = n <= EXACT_BFS_LIMIT
    if exact:
        sources = nodes
    else:
        idx = rng.choice(n, size=min(SAMPLE_SOURCES, n), replace=False)
        sources = [nodes[i] for i in idx]

    total, count, ecc_max = 0, 0, 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(G, s)
        vals = list(lengths.values())
        total += sum(vals)
        count += len(vals) - 1  # exclude the zero self-distance
        ecc = max(vals)
        if ecc > ecc_max:
            ecc_max = ecc

    return {
        "mean_path": total / count if count else None,
        "diameter": ecc_max,
        "path_exact": exact,
    }


def measure(G: nx.Graph, name: str, rng: np.random.Generator) -> dict:
    t0 = time.time()
    out = {
        "name": name,
        "N": G.number_of_nodes(),
        "E": G.number_of_edges(),
        "fiedler": fiedler_value(G),
    }
    out.update(path_metrics(G, rng))
    out["measure_seconds"] = round(time.time() - t0, 2)
    return out


# ── Control constructors (resample-on-disconnect) ───────────────────

def connected_sample(builder, base_seed: int, label: str) -> tuple[nx.Graph, int]:
    """Call builder(seed) until the result is connected. Returns (G, n_failures)."""
    failures = 0
    for attempt in range(MAX_RESAMPLE):
        G = builder(base_seed + 1000 * attempt)
        if nx.is_connected(G):
            return G, failures
        failures += 1
    raise RuntimeError(f"{label}: no connected sample in {MAX_RESAMPLE} tries "
                       f"(base_seed={base_seed})")


def build_fcc_rewire(fcc_graph: nx.Graph, seed: int) -> nx.Graph:
    G = fcc_graph.copy()
    E = G.number_of_edges()
    nx.double_edge_swap(G, nswap=REWIRE_SWAP_FACTOR * E,
                        max_tries=REWIRE_SWAP_FACTOR * E * 20, seed=seed)
    return G


# ── Main ─────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    results = {"protocol": "EE-001", "sizes": []}
    disconnect_counts = {"fcc_rewire": 0, "random_regular": 0, "erdos_renyi": 0}

    for m in FCC_SIZES:
        print(f"\n=== FCC m={m} ===", flush=True)
        fcc = FCCLattice(m)
        Gf = fcc.to_networkx()
        N, E = Gf.number_of_nodes(), Gf.number_of_edges()

        n_cubic = max(2, round(N ** (1 / 3)))
        Gc = CubicLattice(n_cubic).to_networkx()

        rng = np.random.default_rng(7)  # fixed sampling RNG per size

        size_entry = {
            "fcc_m": m,
            "fcc": measure(Gf, f"FCC m={m}", rng),
            "cubic": measure(Gc, f"Cubic n={n_cubic}", rng),
            "controls": {},
        }
        print(f"  FCC   N={N} E={E} fiedler={size_entry['fcc']['fiedler']:.4f}",
              flush=True)
        print(f"  Cubic N={Gc.number_of_nodes()} E={Gc.number_of_edges()} "
              f"fiedler={size_entry['cubic']['fiedler']:.4f}", flush=True)

        d_reg = round(2 * E / N)
        if (N * d_reg) % 2 == 1:
            d_reg += 1  # N*d must be even for a regular graph

        control_builders = {
            "fcc_rewire": lambda s: build_fcc_rewire(Gf, s),
            "random_regular": lambda s: nx.random_regular_graph(d_reg, N, seed=s),
            "erdos_renyi": lambda s: nx.gnm_random_graph(N, E, seed=s),
        }

        for cname, builder in control_builders.items():
            runs = []
            for seed in SEEDS:
                G, fails = connected_sample(builder, seed, cname)
                disconnect_counts[cname] += fails
                r = measure(G, f"{cname} m={m} seed={seed}", rng)
                r["seed"] = seed
                r["disconnected_resamples"] = fails
                runs.append(r)
                print(f"  {cname:<15} seed={seed} N={r['N']} E={r['E']} "
                      f"fiedler={r['fiedler']:.4f} mean_path={r['mean_path']:.3f} "
                      f"({r['measure_seconds']}s)", flush=True)
            size_entry["controls"][cname] = {
                "d_regular": d_reg if cname == "random_regular" else None,
                "runs": runs,
                "fiedler_mean": float(np.mean([r["fiedler"] for r in runs])),
                "fiedler_std": float(np.std([r["fiedler"] for r in runs])),
                "mean_path_mean": float(np.mean([r["mean_path"] for r in runs])),
                "mean_path_std": float(np.std([r["mean_path"] for r in runs])),
                "diameter_mean": float(np.mean([r["diameter"] for r in runs])),
                "diameter_std": float(np.std([r["diameter"] for r in runs])),
                "E_mean": float(np.mean([r["E"] for r in runs])),
            }

        results["sizes"].append(size_entry)

    results["disconnected_resample_counts"] = disconnect_counts
    results["total_seconds"] = round(time.time() - t_start, 1)
    results["config"] = {
        "fcc_sizes": FCC_SIZES, "seeds": SEEDS,
        "exact_bfs_limit": EXACT_BFS_LIMIT, "sample_sources": SAMPLE_SOURCES,
        "rewire_swap_factor": REWIRE_SWAP_FACTOR,
        "fiedler_method": "nx.algebraic_connectivity raw Laplacian, tracemin_lu",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}  (total {results['total_seconds']}s)", flush=True)
    print(f"Disconnected resamples: {disconnect_counts}", flush=True)


if __name__ == "__main__":
    main()
