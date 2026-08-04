# -*- coding: utf-8 -*-
"""LAT-001 sizing v2 -- CORRECT matched-N pairs (FCCLattice(n) = 4n^3).

Two candidate task-graph constructions compared:
  (A) random orientation + fixed arc-keep prob p  (same p across topologies)
  (B) DAG orientation by a generic linear functional w.x  (+ shortcut baseline)

Reports whether the undirected diameter contrast survives into the DIRECTED
distances that actually set the continuous-thought step count.
"""
import random
import networkx as nx
import numpy as np

from rhombic.lattice import CubicLattice, FCCLattice

SEED = 42
# matched-N pairs: cubic n^3 vs FCC 4m^3
PAIRS = [(3, 2), (6, 4), (8, 5)]        # 27/32, 216/256, 512/500


def rewire(g, seed):
    h = g.copy()
    nx.double_edge_swap(h, nswap=20 * h.number_of_edges(),
                        max_tries=400 * h.number_of_edges(), seed=seed)
    return h


def dregular(n, deg, seed):
    if (n * deg) % 2:
        deg += 1
    return nx.random_regular_graph(deg, n, seed=seed)


def rand_orient(g, p, seed):
    rng = random.Random(seed)
    d = nx.DiGraph(); d.add_nodes_from(g.nodes())
    for u, v in g.edges():
        if rng.random() < 0.5:
            u, v = v, u
        if rng.random() < p:
            d.add_edge(u, v)
    return d


def dag_orient(g, seed):
    """Orient by a generic linear functional on node coords; ties broken by index."""
    rng = random.Random(seed)
    w = np.array([1.0, np.pi / 3, np.e / 5])
    key = {}
    for i, nd in enumerate(g.nodes()):
        c = np.array(nd, dtype=float) if isinstance(nd, tuple) and len(nd) == 3 \
            else np.array([i, 0.0, 0.0])
        key[nd] = float(w @ c) + 1e-9 * rng.random()
    d = nx.DiGraph(); d.add_nodes_from(g.nodes())
    for u, v in g.edges():
        d.add_edge(u, v) if key[u] < key[v] else d.add_edge(v, u)
    return d, key


def profile(d, n_src, seed, key=None):
    rng = random.Random(seed)
    nodes = list(d.nodes())
    srcs = rng.sample(nodes, min(n_src, len(nodes)))
    dists, npairs, nreach, shortcut_ok = [], 0, 0, 0
    for s in srcs:
        sp = nx.single_source_shortest_path_length(d, s)
        for t in nodes:
            if t == s:
                continue
            npairs += 1
            r = t in sp
            if r:
                nreach += 1
                dists.append(sp[t])
            if key is not None:
                pred = key[t] > key[s]
                shortcut_ok += (pred == r)
    a = np.array(dists) if dists else np.array([0])
    return dict(reach=nreach / npairs, mean=float(a.mean()), mx=int(a.max()),
                p95=float(np.percentile(a, 95)),
                shortcut=(shortcut_ok / npairs) if key is not None else None)


def undir(g):
    return (g.number_of_nodes(), g.number_of_edges(),
            2 * g.number_of_edges() / g.number_of_nodes(),
            nx.diameter(g), nx.average_shortest_path_length(g))


for cn, fm in PAIRS:
    gc, gf = CubicLattice(cn).to_networkx(), FCCLattice(fm).to_networkx()
    n = gf.number_of_nodes()
    tops = [("cubic-6", gc), ("FCC-12", gf),
            ("FCC-rewire", rewire(gf, SEED)),
            ("rand-reg(FCCdeg)", dregular(n, int(round(2 * gf.number_of_edges() / n)), SEED)),
            ("rand-reg(cubicdeg)", dregular(gc.number_of_nodes(),
                                            int(round(2 * gc.number_of_edges() / gc.number_of_nodes())), SEED))]
    print(f"\n=== matched pair: cubic N={gc.number_of_nodes()} vs FCC N={n} ===")
    print(f"{'topology':<20} {'N':>4} {'E':>5} {'deg':>4} {'D':>3} {'ASP':>6} "
          f"|| {'A:p=1.0':>22} | {'A:p=0.6':>22} | {'B:DAG (+shortcut)':>30}")
    for lab, g in tops:
        N, E, dg, D, asp = undir(g)
        row = f"{lab:<20} {N:>4} {E:>5} {dg:>4.1f} {D:>3} {asp:>6.3f} ||"
        for p in (1.0, 0.6):
            pr = profile(rand_orient(g, p, SEED), 24, SEED)
            row += (f" reach{pr['reach']:.2f} mean{pr['mean']:5.2f} max{pr['mx']:>3}"
                    f" p95{pr['p95']:5.1f} |")
        dd, key = dag_orient(g, SEED)
        pr = profile(dd, 24, SEED, key=key)
        row += (f" reach{pr['reach']:.2f} mean{pr['mean']:5.2f} max{pr['mx']:>3}"
                f" p95{pr['p95']:5.1f} shortcut{pr['shortcut']:.3f}")
        print(row)
