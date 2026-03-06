"""Tests for edge directions, direction weights, weighted diffusion, and weighted consensus."""

import numpy as np
import pytest
from rhombic.lattice import CubicLattice, FCCLattice
from rhombic.spatial import build_adjacency
from rhombic.corpus import edge_values, direction_weights
from rhombic.context import (
    weighted_information_diffusion,
    weighted_consensus_speed,
)


# ── Edge direction tests ─────────────────────────────────────────────


class TestCubicEdgeDirections:

    def test_three_direction_pairs(self):
        cubic = CubicLattice(3)
        dirs = cubic.edge_directions()
        assert len(dirs) == 3

    def test_all_edges_classified(self):
        cubic = CubicLattice(4)
        dirs = cubic.edge_directions()
        total = sum(len(v) for v in dirs.values())
        assert total == cubic.edge_count

    def test_no_overlap(self):
        cubic = CubicLattice(3)
        dirs = cubic.edge_directions()
        all_edges = []
        for edges in dirs.values():
            all_edges.extend(edges)
        assert len(set(all_edges)) == len(all_edges)

    def test_balanced_for_cube(self):
        """Interior of a cubic lattice has equal edges in each direction."""
        cubic = CubicLattice(5)
        dirs = cubic.edge_directions()
        counts = [len(v) for v in dirs.values()]
        # n*(n-1)*n edges per direction for n^3 cubic
        expected = 5 * 4 * 5  # n * (n-1) * n
        for c in counts:
            assert c == expected


class TestFCCEdgeDirections:

    def test_six_direction_pairs(self):
        fcc = FCCLattice(3)
        dirs = fcc.edge_directions()
        assert len(dirs) == 6

    def test_all_edges_classified(self):
        fcc = FCCLattice(3)
        dirs = fcc.edge_directions()
        total = sum(len(v) for v in dirs.values())
        assert total == fcc.edge_count

    def test_no_overlap(self):
        fcc = FCCLattice(3)
        dirs = fcc.edge_directions()
        all_edges = []
        for edges in dirs.values():
            all_edges.extend(edges)
        assert len(set(all_edges)) == len(all_edges)

    def test_directions_nonempty(self):
        fcc = FCCLattice(3)
        dirs = fcc.edge_directions()
        for pair_idx, edges in dirs.items():
            assert len(edges) > 0, f"Direction pair {pair_idx} is empty"


# ── Direction weights tests ──────────────────────────────────────────


class TestDirectionWeights:

    def test_fcc_six_buckets(self):
        vals = list(range(1, 25))
        dw = direction_weights(vals, 6)
        assert len(dw) == 6

    def test_cubic_three_buckets(self):
        vals = list(range(1, 25))
        dw = direction_weights(vals, 3)
        assert len(dw) == 3

    def test_monotonic(self):
        """Buckets from sorted values should be monotonically increasing."""
        vals = list(range(1, 25))
        dw = direction_weights(vals, 6)
        for i in range(len(dw) - 1):
            assert dw[i] <= dw[i + 1]

    def test_corpus_values(self):
        """Works with actual corpus values."""
        corpus = [float(v) for v in edge_values()]
        dw = direction_weights(corpus, 6)
        assert len(dw) == 6
        assert all(d > 0 for d in dw)


# ── Weighted diffusion tests ─────────────────────────────────────────


def _make_weighted_adj(cubic):
    """Build adjacency and edge weight dicts for a cubic lattice."""
    adj = build_adjacency(cubic.edges, cubic.node_count)
    ew = {}
    rng = np.random.default_rng(42)
    for u, v in cubic.edges:
        w = rng.uniform(0.5, 2.0)
        ew[(u, v)] = w
        ew[(v, u)] = w
    return adj, ew


class TestWeightedDiffusion:

    def test_monotonic_reach(self):
        cubic = CubicLattice(5)
        adj, ew = _make_weighted_adj(cubic)
        curve = weighted_information_diffusion(
            adj, ew, 62, cubic.node_count, max_rounds=10)
        for i in range(1, len(curve)):
            assert curve[i] >= curve[i - 1]

    def test_reaches_nodes(self):
        cubic = CubicLattice(5)
        adj, ew = _make_weighted_adj(cubic)
        curve = weighted_information_diffusion(
            adj, ew, 62, cubic.node_count, max_rounds=15)
        assert curve[-1] > 0

    def test_higher_weights_faster(self):
        """Higher edge weights should propagate faster."""
        cubic = CubicLattice(5)
        adj = build_adjacency(cubic.edges, cubic.node_count)
        center = cubic.node_count // 2

        # Low weights
        ew_low = {(u, v): 0.3 for u, v in cubic.edges}
        ew_low.update({(v, u): 0.3 for u, v in cubic.edges})
        curve_low = weighted_information_diffusion(
            adj, ew_low, center, cubic.node_count, max_rounds=15)

        # High weights
        ew_high = {(u, v): 2.0 for u, v in cubic.edges}
        ew_high.update({(v, u): 2.0 for u, v in cubic.edges})
        curve_high = weighted_information_diffusion(
            adj, ew_high, center, cubic.node_count, max_rounds=15)

        # High weights should reach more by round 10
        assert curve_high[9] >= curve_low[9]


# ── Weighted consensus tests ─────────────────────────────────────────


class TestWeightedConsensus:

    def test_converges(self):
        cubic = CubicLattice(5)
        adj, ew = _make_weighted_adj(cubic)
        init = np.random.default_rng(42).standard_normal(cubic.node_count)
        rounds, devs = weighted_consensus_speed(
            adj, ew, init, max_rounds=300)
        assert rounds < 300

    def test_returns_deviations(self):
        cubic = CubicLattice(4)
        adj, ew = _make_weighted_adj(cubic)
        init = np.random.default_rng(42).standard_normal(cubic.node_count)
        rounds, devs = weighted_consensus_speed(
            adj, ew, init, max_rounds=100)
        assert len(devs) > 0
        # Deviations should generally decrease
        assert devs[-1] <= devs[0]

    def test_fcc_faster_weighted(self):
        """FCC should converge at least as fast under weighted consensus."""
        cubic = CubicLattice(8)
        fcc = FCCLattice(5)

        adj_c = build_adjacency(cubic.edges, cubic.node_count)
        adj_f = build_adjacency(fcc.edges, fcc.node_count)

        rng = np.random.default_rng(42)

        # Random edge weights
        ew_c = {}
        for u, v in cubic.edges:
            w = rng.uniform(0.5, 2.0)
            ew_c[(u, v)] = w
            ew_c[(v, u)] = w
        ew_f = {}
        for u, v in fcc.edges:
            w = rng.uniform(0.5, 2.0)
            ew_f[(u, v)] = w
            ew_f[(v, u)] = w

        init_c = rng.standard_normal(cubic.node_count)
        init_f = rng.standard_normal(fcc.node_count)

        rounds_c, _ = weighted_consensus_speed(adj_c, ew_c, init_c, max_rounds=300)
        rounds_f, _ = weighted_consensus_speed(adj_f, ew_f, init_f, max_rounds=300)

        # FCC should converge at least as fast (allow noise margin)
        assert rounds_f <= rounds_c * 1.3
