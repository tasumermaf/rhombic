# -*- coding: utf-8 -*-
"""LAT-001 tasks — graphs, orientation, relabeling, examples, encoding.

Implementer A file per ``lat001/SPEC.md`` (§2 task definition, §3
tokenization, §7.1 pinned signatures). train.py / evaluate.py touch graphs
only through this module; model.py never imports it — the sole shared
surface is ``lat001/common.py`` (frozen).

Pipeline order is pinned by SPEC §2.3 and is load-bearing:

    build undirected graph -> orient -> enumerate reachable pairs, d,
    next-hop sets, and the train/eval split ON ORIGINAL IDS -> apply the
    relabel permutation to arcs, queries, and answer sets -> encode tokens.

Because pairs, splits, and per-example arc-shuffle seeds are all computed
pre-relabel, examples are exactly paired across relabelings of the same
graph — the McNemar lever for the invariance check (SPEC §6.3).

Determinism: every RNG in this file is a ``random.Random`` seeded through
``common.derive_seed`` with a pinned tag (SPEC §7.1) — ``"graph"``,
``"orient"``, ``"split"``, ``"stratify"``, ``f"relabel:{tag}"``,
``f"shuf:{uid}"`` — except ``permute_arcs``, whose caller passes an
already-derived ``perm_seed`` (SPEC §6.2). No unseeded RNG anywhere.
"""
from __future__ import annotations

import hashlib
import random
from typing import Sequence

import networkx as nx
import torch  # §7.1 pins encode_batch -> LongTensor / make_targets -> FloatTensor

from rhombic.lattice import CubicLattice, FCCLattice
from lat001.common import (
    BOS,
    EDGE_SEP,
    N_MAX,
    QUERY,
    SIZE_PAIRS,
    TOPOLOGIES,
    VOCAB_SIZE,
    Example,
    TaskConfig,
    TaskData,
    coord_token,
    derive_seed,
    node_token,
)

# Topology arms with lattice coordinates — the only arms where the
# coordinate-exposed positive control is meaningful (SPEC §2.5).
_LATTICE_ARMS = ("cubic6", "fcc12")


# ── Graph constructions (SPEC §2.2) ──────────────────────────────────
# The three functions below are VERBATIM from
# scripts/lat001_task_graph_sizing.py @ 16605d1 (the landed constructions).
# Do not edit them here — tests/test_lat001.py T-1 enforces body parity by
# loading the script and comparing edge/arc sets at the script's seed.

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


def build_undirected_graph(topology: str, size_index: int, seed: int) -> nx.Graph:
    """Undirected task graph for one arm (SPEC §2.2 table).

    ``seed`` is passed RAW to rewire/dregular — exactly as the sizing script
    uses its literal SEED — so T-1 can hold these internals against the
    script at seed 42. The pinned ``"graph"`` tag derivation happens at the
    build_task call site, not here.
    """
    cn, fm = SIZE_PAIRS[size_index]
    if topology == "cubic6":
        return CubicLattice(cn).to_networkx()
    if topology == "fcc12":
        return FCCLattice(fm).to_networkx()
    if topology == "fcc_rewire":
        return rewire(FCCLattice(fm).to_networkx(), seed)
    if topology == "rr_fccdeg":
        gf = FCCLattice(fm).to_networkx()
        n = gf.number_of_nodes()
        # int(round(mean degree)) matches the sizing script's expression
        return dregular(n, int(round(2 * gf.number_of_edges() / n)), seed)
    if topology == "rr_cubicdeg":
        gc = CubicLattice(cn).to_networkx()
        n = gc.number_of_nodes()
        return dregular(n, int(round(2 * gc.number_of_edges() / n)), seed)
    raise ValueError(f"unknown topology {topology!r}; expected one of {TOPOLOGIES}")


# ── Distance / apportionment helpers ─────────────────────────────────

def _dist_to_target(pred: list[list[int]], t: int) -> dict[int, int]:
    """Directed distance d(s, t) for every s that reaches t.

    One reverse-BFS from the target over predecessor lists (SPEC §2.1) —
    the same pass yields both d and, by filtering successors, next_hops.
    """
    dist = {t: 0}
    frontier = [t]
    while frontier:
        nxt = []
        for x in frontier:
            for w in pred[x]:
                if w not in dist:
                    dist[w] = dist[x] + 1
                    nxt.append(w)
        frontier = nxt
    return dist


def _stratified_cap(universe: list, pool: list[int], cap: int, seed: int) -> list[int]:
    """Cap the eval pool at ``cap`` indices, proportionally stratified by d.

    Largest-remainder apportionment with a min-1 floor per non-empty d
    bucket. The floor is a deliberate guard on top of SPEC §2.4's plain
    "proportional": d is the primary regressor, and a strictly proportional
    draw can zero out a rare-d bucket, blanking that column of the (c,d)
    matrix. FLAGGED in the implementation report, not silent.

    Output order: ascending d, sample order within bucket — deterministic.
    """
    by_d: dict[int, list[int]] = {}
    for i in pool:
        by_d.setdefault(universe[i][2], []).append(i)
    ds = sorted(by_d)
    total = len(pool)

    quota = {d: min(len(by_d[d]), max(1, (len(by_d[d]) * cap) // total)) for d in ds}
    surplus = cap - sum(quota.values())
    if surplus > 0:
        # Distribute leftover slots by largest fractional part (ties toward
        # smaller d). Terminates: total > cap guarantees a bucket with room.
        frac_order = sorted(ds, key=lambda d: (-((len(by_d[d]) * cap) % total), d))
        k = 0
        while surplus > 0:
            d = frac_order[k % len(frac_order)]
            if quota[d] < len(by_d[d]):
                quota[d] += 1
                surplus -= 1
            k += 1
    while surplus < 0:
        # min-1 floors overshot the cap (needs more d-buckets than cap —
        # unreachable at N <= 512; handled for completeness): shave the
        # largest quotas first, never below the floor of 1.
        d = max(ds, key=lambda dd: (quota[dd], -dd))
        if quota[d] <= 1:
            break
        quota[d] -= 1
        surplus += 1

    rng = random.Random(seed)  # pinned tag "stratify", derived by caller
    picked: list[int] = []
    for d in ds:
        bucket = by_d[d]
        picked.extend(bucket if quota[d] >= len(bucket) else rng.sample(bucket, quota[d]))
    return picked


def _integer_coords(topology: str, size_index: int,
                    relabel: tuple[int, ...], n_nodes: int
                    ) -> tuple[tuple[int, int, int], ...]:
    """Integer lattice coordinates indexed by RELABELED node id (SPEC §2.5).

    cubic: round(pos / spacing), range 0..7.  FCC: round(pos / (0.5 a)),
    range 0..10.  Both fit COORD_0..COORD_31 (coord_token range-checks).
    """
    cn, fm = SIZE_PAIRS[size_index]
    if topology == "cubic6":
        lat, scale = CubicLattice(cn), None
        scale = lat.spacing
    elif topology == "fcc12":
        lat = FCCLattice(fm)
        scale = 0.5 * lat.a
    else:  # unreachable — build_task guards before calling
        raise ValueError(f"no lattice coordinates for topology {topology!r}")
    if lat.node_count != n_nodes:
        raise RuntimeError("lattice rebuild disagrees with task node count")
    coords: list[tuple[int, int, int]] = [(0, 0, 0)] * n_nodes
    for orig in range(n_nodes):
        pos = lat.positions[orig]
        coords[relabel[orig]] = tuple(int(round(float(p) / scale)) for p in pos)
    return tuple(coords)


def _fingerprint(n_nodes: int, arcs: list[tuple[int, int]], config: TaskConfig) -> str:
    """sha256 over (n_nodes, sorted arcs, config fields) — SPEC §5 interlock."""
    cfg_fields = (config.topology, config.size_index, config.seed,
                  config.relabel_tag, config.expose_coordinates,
                  config.eval_fraction, config.max_eval_pairs,
                  config.max_train_pairs)
    payload = repr((n_nodes, sorted(arcs), cfg_fields))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Public API (SPEC §7.1) ───────────────────────────────────────────

def build_task(config: TaskConfig) -> TaskData:
    """Build one cell: graph, orientation, examples, splits, relabeling."""
    if config.topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {config.topology!r}; expected one of {TOPOLOGIES}")
    if not 0 <= config.size_index < len(SIZE_PAIRS):
        raise ValueError(f"size_index {config.size_index} outside [0, {len(SIZE_PAIRS)})")
    if config.expose_coordinates and config.topology not in _LATTICE_ARMS:
        raise ValueError(
            f"expose_coordinates is valid only for {_LATTICE_ARMS} (SPEC §2.5); "
            f"got {config.topology!r}")

    # 1. Undirected graph. Pinned "graph" tag derived here (see the
    #    build_undirected_graph docstring for why the seed passes raw below).
    g = build_undirected_graph(config.topology, config.size_index,
                               derive_seed(config.seed, "graph"))
    n_nodes = g.number_of_nodes()
    if n_nodes > N_MAX:
        raise ValueError(f"{n_nodes} nodes exceeds N_MAX={N_MAX}")

    # 2. Orientation — uniform random, arc-keep p = 1.0, SAME for all arms
    #    (card §2: no thinning; the contrast is preserved by construction).
    dg = rand_orient(g, 1.0, derive_seed(config.seed, "orient"))
    arcs_orig = list(dg.edges())  # orientation-output order — shared across relabelings
    m_arcs = len(arcs_orig)

    # 3. Reachable ordered pairs, d, next-hop sets — ORIGINAL ids (SPEC §2.3).
    succ: list[list[int]] = [[] for _ in range(n_nodes)]
    pred: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in arcs_orig:
        succ[u].append(v)
        pred[v].append(u)
    # Canonical target-major enumeration order (pre-shuffle, pre-relabel).
    universe: list[tuple[int, int, int, tuple[int, ...]]] = []
    for t in range(n_nodes):
        dist = _dist_to_target(pred, t)
        for s in range(n_nodes):
            d = dist.get(s)
            if s == t or d is None:
                continue  # unreachable or d = 0 — universe is d >= 1 only
            hops = tuple(u for u in succ[s] if dist.get(u) == d - 1)
            universe.append((s, t, d, hops))  # hops non-empty for reachable pairs

    # 4. Split (tag "split") — seeded shuffle, eval_fraction held out.
    order = list(range(len(universe)))
    random.Random(derive_seed(config.seed, "split")).shuffle(order)
    n_eval_pool = int(round(len(universe) * config.eval_fraction))
    eval_idx = order[:n_eval_pool]
    train_idx = order[n_eval_pool:][: config.max_train_pairs]

    # 5. Eval cap — proportional stratified by d (tag "stratify").
    if len(eval_idx) > config.max_eval_pairs:
        eval_idx = _stratified_cap(universe, eval_idx, config.max_eval_pairs,
                                   derive_seed(config.seed, "stratify"))

    # 6. Relabel (tag f"relabel:{tag}") — MANDATORY (card §2: coordinates
    #    make d closed-form; without relabeling the measurement is void).
    perm = list(range(n_nodes))
    random.Random(derive_seed(config.seed, f"relabel:{config.relabel_tag}")).shuffle(perm)
    relabel = tuple(perm)  # relabel[orig_id] = new_id
    arcs = [(relabel[u], relabel[v]) for u, v in arcs_orig]

    def _mk(i: int) -> Example:
        s, t, d, hops = universe[i]
        uid = f"{s}-{t}"  # ORIGINAL ids — pairs examples across relabelings
        return Example(
            src=relabel[s], tgt=relabel[t],
            next_hops=tuple(sorted(relabel[u] for u in hops)),
            d=d, uid=uid,
            shuffle_seed=derive_seed(config.seed, f"shuf:{uid}"))

    train = [_mk(i) for i in train_idx]
    eval_ = [_mk(i) for i in eval_idx]

    coords = (_integer_coords(config.topology, config.size_index, relabel, n_nodes)
              if config.expose_coordinates else None)
    d_max = max(max((ex.d for ex in train), default=0),
                max((ex.d for ex in eval_), default=0))
    # Constant prompt length within a cell (SPEC §3.2): 3m+4, +5N with coords.
    seq_len = 3 * m_arcs + 4 + (5 * n_nodes if coords is not None else 0)

    return TaskData(config=config, n_nodes=n_nodes, m_arcs=m_arcs, arcs=arcs,
                    train=train, eval=eval_, d_max=d_max, seq_len=seq_len,
                    relabel=relabel, coords=coords,
                    fingerprint=_fingerprint(n_nodes, arcs, config))


def encode_example(task: TaskData, ex: Example) -> list[int]:
    """Lazy token encoding, SPEC §3.2 layouts.

    Standard:   BOS ([u][v] <e>) x m  <Q> [src] [tgt]           (3m + 4)
    Coordinate: BOS ([node][cx][cy][cz] <e>) x N  (arcs)  query (5N + 3m + 4)

    Arc order: a per-example shuffle of ARC INDICES seeded by
    ex.shuffle_seed. The index permutation is uid-keyed and relabel-free,
    so it is identical across relabelings of the same graph — only the
    token values change (SPEC §3.2).
    """
    arc_order = list(range(task.m_arcs))
    random.Random(ex.shuffle_seed).shuffle(arc_order)

    toks = [BOS]
    if task.coords is not None:
        # Coordinate block ordered by relabeled node id ascending (SPEC §3.2).
        for nid in range(task.n_nodes):
            cx, cy, cz = task.coords[nid]
            toks += [node_token(nid), coord_token(cx), coord_token(cy),
                     coord_token(cz), EDGE_SEP]
    for i in arc_order:
        u, v = task.arcs[i]
        toks += [node_token(u), node_token(v), EDGE_SEP]  # tail, head, <e>
    toks += [QUERY, node_token(ex.src), node_token(ex.tgt)]
    # Prompt ends at [tgt]: no <A>, no thought slots — model.forward appends
    # those (SPEC §3.2/§4.2). Ground-truth answers never appear in the input.
    return toks


def encode_batch(task: TaskData, examples: Sequence[Example]) -> torch.LongTensor:
    """[B, T] LongTensor of prompts (pinned return type, SPEC §7.1)."""
    if not examples:
        return torch.empty((0, task.seq_len), dtype=torch.long)
    rows = [encode_example(task, ex) for ex in examples]
    if any(len(r) != task.seq_len for r in rows):
        # Within a cell all prompts share one length — anything else is a bug.
        raise ValueError("encoded lengths disagree with task.seq_len")
    return torch.tensor(rows, dtype=torch.long)


def make_targets(task: TaskData, examples: Sequence[Example]) -> torch.FloatTensor:
    """[B, VOCAB_SIZE] uniform soft labels over next_hops (SPEC §2.1).

    Uniform over ALL valid next hops — relabeling-equivariant; no arbitrary
    canonical-hop choice exists anywhere in the pipeline.
    """
    out = torch.zeros((len(examples), VOCAB_SIZE), dtype=torch.float32)
    for b, ex in enumerate(examples):
        p = 1.0 / len(ex.next_hops)
        for h in ex.next_hops:
            out[b, node_token(h)] = p
    return out


def permute_arcs(task: TaskData, perm_seed: int) -> TaskData:
    """Permutation-null probe (SPEC §6.2): arcs through a seeded pi != id;
    queries and answer sets keep their ORIGINAL labels.

    Evaluation-only. The returned TaskData deliberately keeps the base
    task's fingerprint / relabel / coords: it is a scoring probe of the
    same underlying cell, and the checkpoint interlock references the
    training graph (judgment call, flagged in the implementation report).
    Never train on the output of this function.
    """
    n = task.n_nodes
    identity = list(range(n))
    rng = random.Random(perm_seed)
    perm = identity[:]
    rng.shuffle(perm)
    while perm == identity:  # pi != identity is pinned; reshuffle deterministically
        rng.shuffle(perm)
    arcs = [(perm[u], perm[v]) for u, v in task.arcs]
    return TaskData(config=task.config, n_nodes=n, m_arcs=task.m_arcs,
                    arcs=arcs, train=task.train, eval=task.eval,
                    d_max=task.d_max, seq_len=task.seq_len,
                    relabel=task.relabel, coords=task.coords,
                    fingerprint=task.fingerprint)


# ── Self-checks (pure functions, CPU, seconds) ───────────────────────
# Run as:  python -m lat001.tasks   (from C:/falco/rhombic)

if __name__ == "__main__":
    import sys

    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        print(f"{name} = {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    print("=== SELFCHECK: lat001.tasks ===")

    for topo in ("cubic6", "fcc12"):
        task = build_task(TaskConfig(topo, 0, 42))
        D = nx.DiGraph()
        D.add_nodes_from(range(task.n_nodes))
        D.add_edges_from(task.arcs)
        R = D.reverse(copy=True)

        # 1. Every emitted example vs independent networkx BFS.
        dist_cache: dict[int, dict[int, int]] = {}
        ok = True
        for ex in task.train + task.eval:
            if ex.tgt not in dist_cache:
                dist_cache[ex.tgt] = dict(
                    nx.single_source_shortest_path_length(R, ex.tgt))
            dist = dist_cache[ex.tgt]
            succ_s = set(D.successors(ex.src))
            expect = {u for u in succ_s if dist.get(u) == ex.d - 1}
            ok &= dist.get(ex.src) == ex.d and ex.d >= 1
            ok &= bool(ex.next_hops) and set(ex.next_hops) == expect
        n_ex = len(task.train) + len(task.eval)
        check(f"{topo}: {n_ex} examples match independent BFS", ok)

        # 2. Universe completeness: every reachable ordered pair, once.
        n_reach = sum(
            1 for t in D.nodes
            for s in nx.single_source_shortest_path_length(R, t)
            if s != t)
        check(f"{topo}: universe complete ({n_reach} reachable pairs)",
              n_ex == n_reach)

        # 3. Standard token layout 3m+4.
        toks = encode_example(task, task.eval[0])
        check(f"{topo}: standard layout 3m+4",
              len(toks) == task.seq_len == 3 * task.m_arcs + 4
              and toks[0] == BOS and toks[-3] == QUERY)

        # 4. Full-rebuild determinism (bitwise).
        task2 = build_task(TaskConfig(topo, 0, 42))
        check(f"{topo}: rebuild bitwise-deterministic",
              task2.fingerprint == task.fingerprint
              and task2.arcs == task.arcs
              and [e.uid for e in task2.eval] == [e.uid for e in task.eval]
              and encode_example(task2, task2.eval[0]) == toks)

    # 5. Relabel pairing: tags 1 vs 2, same seed, sigma = relabel_b o relabel_a^-1.
    ta = build_task(TaskConfig("cubic6", 0, 42, relabel_tag=1))
    tb = build_task(TaskConfig("cubic6", 0, 42, relabel_tag=2))
    sigma = [0] * ta.n_nodes
    for orig in range(ta.n_nodes):
        sigma[ta.relabel[orig]] = tb.relabel[orig]
    check("relabel tags 1/2: exactly paired via sigma",
          [e.uid for e in ta.eval] == [e.uid for e in tb.eval]
          and [(sigma[u], sigma[v]) for u, v in ta.arcs] == tb.arcs
          and all((sigma[a.src], sigma[a.tgt]) == (b.src, b.tgt)
                  and {sigma[h] for h in a.next_hops} == set(b.next_hops)
                  and a.shuffle_seed == b.shuffle_seed
                  for a, b in zip(ta.eval, tb.eval)))

    # 6. Coordinate control: 5N+3m+4 layout, coords populated, arm guard.
    tc = build_task(TaskConfig("cubic6", 0, 42, expose_coordinates=True))
    toks_c = encode_example(tc, tc.eval[0])
    ok = (len(toks_c) == tc.seq_len == 5 * tc.n_nodes + 3 * tc.m_arcs + 4
          and tc.coords is not None and len(tc.coords) == tc.n_nodes)
    try:
        build_task(TaskConfig("rr_fccdeg", 0, 42, expose_coordinates=True))
        ok = False  # must raise
    except ValueError:
        pass
    check("coordinate control: 5N+3m+4 layout + arm guard", ok)

    # 7. encode_batch / make_targets shapes; target rows sum to 1.
    xb = encode_batch(ta, ta.eval[:4])
    tg = make_targets(ta, ta.eval[:4])
    check("encode_batch [B,T] / make_targets rows sum to 1",
          tuple(xb.shape) == (4, ta.seq_len)
          and tuple(tg.shape) == (4, VOCAB_SIZE)
          and bool(torch.allclose(tg.sum(dim=1), torch.ones(4))))

    # 8. Permutation-null plumbing: arcs move, everything else stays.
    from collections import Counter
    tp = permute_arcs(ta, perm_seed=7)
    out_deg = lambda arcs: sorted(Counter(u for u, _ in arcs).values())
    check("permute_arcs: arcs permuted, labels/fingerprint/degrees stable",
          tp.arcs != ta.arcs
          and tp.eval is ta.eval
          and tp.fingerprint == ta.fingerprint
          and out_deg(tp.arcs) == out_deg(ta.arcs)
          and permute_arcs(ta, perm_seed=7).arcs == tp.arcs)

    print(f"ALL = {'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
