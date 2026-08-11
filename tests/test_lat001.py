# -*- coding: utf-8 -*-
"""Tests for the LAT-001 harness (lat001/SPEC.md §9).

ONE file, three comment-bannered sections (SPEC §1):

    SECTION A — tasks.py        T-1 graph parity, T-2 example correctness,
                                T-3 relabel pairing  (+ two small A extras)
    SECTION B — model.py        T-4                  [Implementer B appends]
    SECTION C — train/evaluate  T-5, T-6             [Implementer C appends]

All tests CPU, all seeded. Runtime note: T-1 executes the sizing script once
(its module-level sweep runs on load — stdout suppressed, module cached), the
single heaviest item in Section A.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import networkx as nx
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:  # robust regardless of pytest rootdir config
    sys.path.insert(0, str(_REPO))

from rhombic.lattice import CubicLattice, FCCLattice  # noqa: E402
from lat001 import tasks  # noqa: E402
from lat001.common import (  # noqa: E402
    BOS,
    COORD_BASE,
    EDGE_SEP,
    N_MAX,
    NODE_BASE,
    QUERY,
    SIZE_PAIRS,
    TOPOLOGIES,
    TaskConfig,
    node_id,
    node_token,
)

# =====================================================================
# SECTION A — tasks.py (Implementer A): T-1, T-2, T-3
# =====================================================================

_SIZING = None


def _sizing_script():
    """Load scripts/lat001_task_graph_sizing.py per SPEC §9 T-1.

    spec_from_file_location executes the module — including its module-level
    sizing sweep. That is the pinned loading mechanism; we suppress its
    stdout and cache the module so the cost is paid once per test session.
    """
    global _SIZING
    if _SIZING is None:
        path = _REPO / "scripts" / "lat001_task_graph_sizing.py"
        spec = importlib.util.spec_from_file_location("lat001_sizing_script", path)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
        _SIZING = mod
    return _SIZING


def _edge_set(g: nx.Graph) -> frozenset:
    return frozenset(frozenset((u, v)) for u, v in g.edges())


def test_t1_graph_parity():
    """T-1: tasks.py internals == sizing-script constructions, size 0, seed 42.

    build_undirected_graph passes its seed RAW to rewire/dregular (see its
    docstring), so calling it with the script's literal seed 42 must
    reproduce the script's graphs exactly. Orientation parity is checked at
    the function level: tasks.rand_orient vs script.rand_orient, seed 42.
    """
    sc = _sizing_script()
    cn, fm = SIZE_PAIRS[0]

    # Script-side constructions, exactly as written in the script's main loop.
    gc = CubicLattice(cn).to_networkx()
    gf = FCCLattice(fm).to_networkx()
    n = gf.number_of_nodes()
    script_graphs = {
        "cubic6": gc,
        "fcc12": gf,
        "fcc_rewire": sc.rewire(gf, 42),
        "rr_fccdeg": sc.dregular(n, int(round(2 * gf.number_of_edges() / n)), 42),
        "rr_cubicdeg": sc.dregular(
            gc.number_of_nodes(),
            int(round(2 * gc.number_of_edges() / gc.number_of_nodes())), 42),
    }

    for topo in TOPOLOGIES:
        g_tasks = tasks.build_undirected_graph(topo, 0, 42)
        g_script = script_graphs[topo]
        assert _edge_set(g_tasks) == _edge_set(g_script), f"edge set mismatch: {topo}"
        # Oriented arc parity (p = 1.0, the pinned orientation regime).
        d_tasks = tasks.rand_orient(g_tasks, 1.0, 42)
        d_script = sc.rand_orient(g_script, 1.0, 42)
        assert set(d_tasks.edges()) == set(d_script.edges()), f"arc set mismatch: {topo}"


def test_t2_example_correctness_cubic27():
    """T-2: every emitted example verified against fresh networkx BFS;
    token layouts match SPEC §3.2 exactly."""
    task = tasks.build_task(TaskConfig("cubic6", 0, 42))
    assert task.n_nodes == 27

    D = nx.DiGraph()
    D.add_nodes_from(range(task.n_nodes))
    D.add_edges_from(task.arcs)
    R = D.reverse(copy=True)

    dist_cache: dict[int, dict[int, int]] = {}
    for ex in task.train + task.eval:
        if ex.tgt not in dist_cache:
            dist_cache[ex.tgt] = dict(nx.single_source_shortest_path_length(R, ex.tgt))
        dist = dist_cache[ex.tgt]
        assert ex.d >= 1 and dist.get(ex.src) == ex.d
        succ = set(D.successors(ex.src))
        expect = {u for u in succ if dist.get(u) == ex.d - 1}
        assert ex.next_hops, "next_hops empty for a reachable pair"
        assert set(ex.next_hops) == expect
        assert set(ex.next_hops) <= succ

    # Universe completeness: every reachable ordered pair (s != t) emitted
    # exactly once (no caps bind at N = 27).
    n_reach = sum(
        1 for t in D.nodes
        for s in nx.single_source_shortest_path_length(R, t)
        if s != t)
    assert len(task.train) + len(task.eval) == n_reach

    # Standard layout: BOS ([u][v]<e>) x m <Q> [src] [tgt]  — length 3m+4.
    m = task.m_arcs
    for ex in (task.train[0], task.eval[0]):
        toks = tasks.encode_example(task, ex)
        assert len(toks) == 3 * m + 4 == task.seq_len
        assert toks[0] == BOS and toks[-3] == QUERY
        assert toks[-2] == node_token(ex.src) and toks[-1] == node_token(ex.tgt)
        body = toks[1:-3]
        arcs_seen = set()
        for i in range(0, len(body), 3):
            u_tok, v_tok, sep = body[i:i + 3]
            assert sep == EDGE_SEP
            arcs_seen.add((node_id(u_tok), node_id(v_tok)))
        assert arcs_seen == set(task.arcs)

    # Coordinate-control layout: 5N + 3m + 4; block ordered by relabeled id,
    # values equal the lattice coordinates mapped through the relabel.
    taskc = tasks.build_task(TaskConfig("cubic6", 0, 42, expose_coordinates=True))
    N = taskc.n_nodes
    toks = tasks.encode_example(taskc, taskc.eval[0])
    assert len(toks) == 5 * N + 3 * taskc.m_arcs + 4 == taskc.seq_len
    lat = CubicLattice(SIZE_PAIRS[0][0])
    inv = {new: orig for orig, new in enumerate(taskc.relabel)}
    for i in range(N):
        rec = toks[1 + 5 * i: 1 + 5 * (i + 1)]
        assert rec[0] == node_token(i) and rec[4] == EDGE_SEP
        got = tuple(t - COORD_BASE for t in rec[1:4])
        pos = lat.positions[inv[i]]
        assert got == tuple(int(round(float(p) / lat.spacing)) for p in pos)
        assert got == taskc.coords[i]

    # Coordinate control is valid ONLY for the lattice arms (SPEC §2.5).
    with pytest.raises(ValueError):
        tasks.build_task(TaskConfig("fcc_rewire", 0, 42, expose_coordinates=True))


def test_t3_relabel_pairing():
    """T-3: same seed, tags 1 vs 2 — identical uid sequences, identical d per
    uid, arc lists related by a node permutation; tokens differ only by
    node-token substitution (shared arc-index shuffle)."""
    ta = tasks.build_task(TaskConfig("cubic6", 0, 42, relabel_tag=1))
    tb = tasks.build_task(TaskConfig("cubic6", 0, 42, relabel_tag=2))

    assert [e.uid for e in ta.train] == [e.uid for e in tb.train]
    assert [e.uid for e in ta.eval] == [e.uid for e in tb.eval]
    assert [e.d for e in ta.train + ta.eval] == [e.d for e in tb.train + tb.eval]

    # sigma = relabel_b o relabel_a^-1 maps A-names to B-names, index-aligned.
    sigma = [0] * ta.n_nodes
    for orig in range(ta.n_nodes):
        sigma[ta.relabel[orig]] = tb.relabel[orig]
    assert [(sigma[u], sigma[v]) for u, v in ta.arcs] == tb.arcs

    for ea, eb in zip(ta.train + ta.eval, tb.train + tb.eval):
        assert (sigma[ea.src], sigma[ea.tgt]) == (eb.src, eb.tgt)
        assert {sigma[h] for h in ea.next_hops} == set(eb.next_hops)
        assert ea.shuffle_seed == eb.shuffle_seed  # uid-keyed, relabel-free

    # Token streams: same length, same positions, node tokens substituted
    # through sigma, every non-node token identical (SPEC §3.2).
    for ea, eb in zip(ta.eval[:5], tb.eval[:5]):
        xa = tasks.encode_example(ta, ea)
        xb = tasks.encode_example(tb, eb)
        assert len(xa) == len(xb)
        for a, b in zip(xa, xb):
            if NODE_BASE <= a < NODE_BASE + N_MAX:
                assert b == node_token(sigma[node_id(a)])
            else:
                assert a == b


def test_ta_extra_determinism():
    """Section A extra: full rebuild of one cell is bitwise-deterministic."""
    t1 = tasks.build_task(TaskConfig("fcc12", 0, 42))
    t2 = tasks.build_task(TaskConfig("fcc12", 0, 42))
    assert t1.fingerprint == t2.fingerprint
    assert t1.arcs == t2.arcs
    assert [e.uid for e in t1.train] == [e.uid for e in t2.train]
    assert [e.uid for e in t1.eval] == [e.uid for e in t2.eval]
    assert (tasks.encode_example(t1, t1.eval[0])
            == tasks.encode_example(t2, t2.eval[0]))


def test_ta_extra_permute_arcs():
    """Section A extra: permutation-null probe — arcs permuted, queries and
    answer sets untouched, fingerprint preserved (interlock references the
    training graph), deterministic under perm_seed."""
    from collections import Counter
    base = tasks.build_task(TaskConfig("cubic6", 0, 42))
    perm = tasks.permute_arcs(base, perm_seed=7)
    assert perm.arcs != base.arcs
    assert perm.eval is base.eval and perm.train is base.train
    assert perm.fingerprint == base.fingerprint
    out_deg = lambda arcs: sorted(Counter(u for u, _ in arcs).values())
    assert out_deg(perm.arcs) == out_deg(base.arcs)
    assert tasks.permute_arcs(base, perm_seed=7).arcs == perm.arcs
    assert tasks.permute_arcs(base, perm_seed=8).arcs != perm.arcs


def test_ta_extra_stratified_cap():
    """Section A extra: force `_stratified_cap` to engage (SPEC §2.4).

    The cap fires only above max_eval_pairs, so no smoke cell and no other
    test reaches it (harness review 2026-08-11, F-3): pools are 140/192
    against the 5000 default. Dropping the cap to 50 on cubic6/0 exercises
    the apportionment, the min-1 floor, and the surplus loop. max_eval_pairs
    feeds no seed derivation, so the uncapped build IS the pre-cap pool.
    """
    pool = tasks.build_task(TaskConfig("cubic6", 0, 42))
    capped = tasks.build_task(TaskConfig("cubic6", 0, 42, max_eval_pairs=50))
    assert len(pool.eval) > 50, "cap would not engage — pool too small"
    assert len(capped.eval) == 50

    # min-1 floor: every d present pre-cap survives with >= 1 eval item.
    ds_pool = {ex.d for ex in pool.eval}
    ds_capped = {ex.d for ex in capped.eval}
    assert ds_pool <= ds_capped
    assert set(ex.d for ex in capped.eval) <= ds_pool   # invents no strata

    # Seeded: two identical builds give the identical eval list.
    again = tasks.build_task(TaskConfig("cubic6", 0, 42, max_eval_pairs=50))
    assert [(e.uid, e.d) for e in capped.eval] == [(e.uid, e.d) for e in again.eval]


# =====================================================================
# SECTION B — model.py (Implementer B): T-4
# =====================================================================
# Ported from the model.py __main__ self-checks, which pytest never
# executes (harness review 2026-08-11, B-1), plus the liveness assertion
# from that review's N-2: nothing else in this suite or the §8 smoke
# asserts that the latent budget reaches the answer logits at all.
# SCOPE: liveness here is forward-value only. A .detach() at model.py:192
# leaves every forward value bit-identical, so it is NOT caught by this
# test — catching it needs the second half of the review's proposed SA-6
# (non-zero gradient w.r.t. the first latent state), which is a SPEC §8
# amendment and deliberately not added in this pass.

import torch  # noqa: E402

from lat001 import model as model_lib  # noqa: E402
from lat001.common import SMOKE_MODEL, VOCAB_SIZE  # noqa: E402


def test_t4_model_forward_and_grad():
    """T-4: forward shape [B, 550] for c in {0, 1, 5}; the latent loop
    extends the sequence by exactly c+1 positions; gradients reach the
    token embedding through a c=3 backward; the loop is LIVE (logits move
    with c)."""
    torch.manual_seed(0)
    m = model_lib.ContinuousThoughtTransformer(SMOKE_MODEL)
    B, T = 3, 17
    tokens = torch.randint(0, VOCAB_SIZE, (B, T))

    # Forward shape + sequence extension. ln_f fires once per _encode: the
    # c latent forwards see lengths T .. T+c-1, the final <A> forward sees
    # T+c+1 — c+1 encodes, ending exactly c+1 positions past the prompt.
    seen_lengths: list[int] = []
    hook = m.ln_f.register_forward_hook(
        lambda mod, inp, out: seen_lengths.append(out.size(1)))
    try:
        for c in (0, 1, 5):
            seen_lengths.clear()
            with torch.no_grad():
                logits = m(tokens, c)
            assert logits.shape == (B, VOCAB_SIZE), logits.shape
            assert len(seen_lengths) == c + 1, (c, seen_lengths)
            assert seen_lengths[-1] == T + c + 1, (c, seen_lengths)
            assert seen_lengths == [T + j for j in range(c)] + [T + c + 1], \
                (c, seen_lengths)
    finally:
        hook.remove()

    # Gradients reach the token embedding through the whole unrolled loop.
    m.zero_grad(set_to_none=True)
    m(tokens, 3).sum().backward()
    g = m.tok_emb.weight.grad
    assert g is not None
    assert float(g.abs().sum()) > 0.0

    # Liveness (review N-2): the latent state must reach the <A> logits.
    # Measured ~3e-2 on a trained checkpoint; this model is untrained, so
    # the bar is only "not bitwise-identical", well above fp32 noise.
    m.eval()
    with torch.no_grad():
        base = m(tokens, 0)
        for k in (1, 2, 5):
            delta = float((m(tokens, k) - base).abs().max())
            assert delta > 1e-6, (k, delta)


# =====================================================================
# SECTION C — train/evaluate (Implementer C): T-5, T-6
# =====================================================================
# One module-scoped 50-step trained model on cubic-27 is shared by T-5
# and T-6 (SPEC §9 runtime target). Checkpoints go to pytest's tmp dir —
# transient test artifacts, not results (lat001/results/ is for results).
# c_max=2 for the test TrainConfig keeps the unrolled latent loop cheap;
# the §8 smoke (python -m lat001.evaluate --smoke) exercises the pinned
# c_max = d_max + 2 policy.

import math  # noqa: E402

import numpy as np  # noqa: E402

from lat001 import evaluate as ev  # noqa: E402
from lat001 import train as train_lib  # noqa: E402
from lat001.common import SMOKE_MODEL, TrainConfig  # noqa: E402


@pytest.fixture(scope="module")
def trained_cubic27(tmp_path_factory):
    """(cfg, task, TrainResult, model) — 50 steps, SMOKE_MODEL, seed 42."""
    cfg = TaskConfig(topology="cubic6", size_index=0, seed=42)
    task = tasks.build_task(cfg)
    tcfg = TrainConfig(steps=50, batch_size=16, lr=1e-3, c_min=0, c_max=2,
                       seed=42,
                       checkpoint_dir=str(tmp_path_factory.mktemp("ckpt")))
    result = train_lib.train(task, SMOKE_MODEL, tcfg)
    model, _ = train_lib.load_checkpoint(result.checkpoint_path,
                                         expect_fingerprint=task.fingerprint)
    return cfg, task, result, model


def test_t5_train_smoke(trained_cubic27):
    """T-5: loss finite + decreasing trend; checkpoint written and
    reloadable; fingerprint interlock rejects a mismatched task."""
    _cfg, task, result, _model = trained_cubic27
    assert len(result.losses) == 50
    assert all(math.isfinite(x) for x in result.losses)
    assert float(np.mean(result.losses[-10:])) < float(np.mean(result.losses[:10]))
    assert Path(result.checkpoint_path).exists()
    # reloadable with the matching fingerprint (the fixture already did);
    # a mismatched fingerprint must be refused (SPEC §5)
    with pytest.raises(ValueError):
        train_lib.load_checkpoint(result.checkpoint_path,
                                  expect_fingerprint="0" * 64)


def test_t6_eval_matrix_plumbing(trained_cubic27):
    """T-6a: matrix shapes, NaN/count consistency, JSON round-trip,
    bitwise determinism of two identical evaluate calls."""
    _cfg, task, _result, model = trained_cubic27
    c_vals = (0, 1, 2)
    r1 = ev.evaluate(model, task, c_vals, batch_size=32)
    assert r1.c_values == c_vals
    assert r1.d_values == tuple(range(1, r1.d_values[-1] + 1))
    assert r1.acc_by_c_d.shape == (len(c_vals), len(r1.d_values))
    assert r1.counts_by_c_d.shape == r1.acc_by_c_d.shape
    assert r1.overall_by_c.shape == (len(c_vals),)
    nan_mask = np.isnan(r1.acc_by_c_d)
    assert (r1.counts_by_c_d[nan_mask] == 0).all()   # NaN exactly where count 0
    assert (r1.counts_by_c_d[~nan_mask] > 0).all()
    assert ((r1.overall_by_c >= 0) & (r1.overall_by_c <= 1)).all()
    # counts are a property of the eval split — constant across c rows
    assert (r1.counts_by_c_d == r1.counts_by_c_d[0]).all()
    # determinism: identical call is bitwise-equal (SPEC §9 T-6)
    r2 = ev.evaluate(model, task, c_vals, batch_size=32)
    assert np.array_equal(r1.overall_by_c, r2.overall_by_c)
    assert np.array_equal(r1.acc_by_c_d, r2.acc_by_c_d, equal_nan=True)


def test_t6_save_results_roundtrip(trained_cubic27, tmp_path):
    """T-6b: save_results round-trips through strict JSON (NaN -> null)."""
    import json

    _cfg, task, _result, model = trained_cubic27
    r1 = ev.evaluate(model, task, (0, 1), batch_size=32)
    path = str(tmp_path / "eval.json")
    ev.save_results(r1, path)
    loaded = ev.load_results(path)
    assert loaded["type"] == "EvalResult"
    assert loaded == json.loads(json.dumps(ev._to_jsonable(r1)))


def test_t6_null_and_invariance_statistics(trained_cubic27):
    """T-6c: permutation_null and relabeling_invariance return finite
    statistics on a 50-step model; null is seed-deterministic."""
    cfg, task, _result, model = trained_cubic27
    nr = ev.permutation_null(model, task, c=1, n_perms=3, seed=0)
    assert nr.n_perms == 3
    assert nr.n == 3 * len(task.eval)
    assert math.isfinite(nr.z)
    assert 0.0 <= nr.p <= 1.0
    assert 0.0 <= nr.acc_model <= 1.0
    assert 0.0 <= nr.acc_baseline <= 1.0
    nr2 = ev.permutation_null(model, task, c=1, n_perms=3, seed=0)
    assert nr2.z == nr.z and nr2.acc_model == nr.acc_model   # seeded determinism

    ir = ev.relabeling_invariance(model, cfg, relabel_tags=(1, 2), c=1,
                                  max_pairs=80)
    assert ir.n_pairs == 80                     # capped below the 140-item split
    assert ir.b + ir.c_dis <= ir.n_pairs
    assert 0.0 < ir.p <= 1.0
    assert 0.0 <= ir.acc_a <= 1.0 and 0.0 <= ir.acc_b <= 1.0
    # tags colliding with the training tag must be refused (SPEC §6.3)
    with pytest.raises(ValueError):
        ev.relabeling_invariance(model, cfg, relabel_tags=(0, 2), c=1,
                                 max_pairs=10)
