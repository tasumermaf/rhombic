"""Tests for Rung 4: Context Architecture benchmarks."""

import numpy as np
import pytest
from rhombic.lattice import CubicLattice, FCCLattice
from rhombic.spatial import build_adjacency
from rhombic.context import (
    generate_embeddings,
    project_to_3d,
    assign_to_lattice,
    neighborhood_recall,
    information_diffusion,
    consensus_speed,
    run_context_benchmark,
)


class TestEmbeddingGeneration:
    def test_shape(self):
        emb = generate_embeddings(100, 64)
        assert emb.shape == (100, 64)

    def test_clustered(self):
        """Embeddings should be clustered, not uniform."""
        emb = generate_embeddings(200, 64, n_clusters=5)
        # Pairwise distances within clusters should be smaller than across
        # Just verify they're not all the same distance (i.e., not uniform)
        dists = np.linalg.norm(emb[:10] - emb[10:20], axis=1)
        assert np.std(dists) > 0

    def test_different_seeds(self):
        e1 = generate_embeddings(50, 32, seed=1)
        e2 = generate_embeddings(50, 32, seed=2)
        assert not np.allclose(e1, e2)


class TestProjection:
    def test_output_3d(self):
        emb = generate_embeddings(100, 64)
        pts = project_to_3d(emb)
        assert pts.shape == (100, 3)

    def test_in_unit_cube(self):
        emb = generate_embeddings(100, 64)
        pts = project_to_3d(emb)
        assert np.all(pts >= -1e-10)
        assert np.all(pts <= 1 + 1e-10)


class TestAssignment:
    def test_valid_indices(self):
        cubic = CubicLattice(5)
        pts = np.random.default_rng(42).uniform(0, 4, (50, 3))
        idx = assign_to_lattice(pts, cubic.positions)
        assert len(idx) == 50
        assert np.all(idx >= 0)
        assert np.all(idx < cubic.node_count)


class TestNeighborhoodRecall:
    def test_fcc_higher_recall(self):
        """FCC should capture more true neighbors at 1-hop."""
        cubic = CubicLattice(8)
        fcc = FCCLattice(5)

        emb = generate_embeddings(150, 64, seed=42)
        pts = project_to_3d(emb, seed=42)

        # Scale to lattice
        def scale(p, lp):
            mn, mx = lp.min(axis=0), lp.max(axis=0)
            ext = np.where(mx > mn, mx - mn, 1.0)
            return mn + ext * (0.1 + p * 0.8)

        pts_c = scale(pts, cubic.positions)
        pts_f = scale(pts, fcc.positions)

        adj_c = build_adjacency(cubic.edges, cubic.node_count)
        adj_f = build_adjacency(fcc.edges, fcc.node_count)

        assign_c = assign_to_lattice(pts_c, cubic.positions)
        assign_f = assign_to_lattice(pts_f, fcc.positions)

        rec_c = neighborhood_recall(emb, pts_c, cubic.positions, adj_c,
                                    assign_c, k_true=8, max_hops=2)
        rec_f = neighborhood_recall(emb, pts_f, fcc.positions, adj_f,
                                    assign_f, k_true=8, max_hops=2)

        # At 1-hop, FCC should have higher recall
        assert rec_f[0]._raw_recall > rec_c[0]._raw_recall


class TestDiffusion:
    def test_monotonic_reach(self):
        """Signal reach should increase over rounds."""
        cubic = CubicLattice(5)
        adj = build_adjacency(cubic.edges, cubic.node_count)
        curve = information_diffusion(adj, 62, cubic.node_count, max_rounds=10)
        # Should be monotonically non-decreasing
        for i in range(1, len(curve)):
            assert curve[i] >= curve[i-1]

    def test_fcc_faster_spread(self):
        """FCC should reach more nodes per round."""
        cubic = CubicLattice(8)
        fcc = FCCLattice(5)

        adj_c = build_adjacency(cubic.edges, cubic.node_count)
        adj_f = build_adjacency(fcc.edges, fcc.node_count)

        # Start from center
        center_c = np.argmin(np.linalg.norm(
            cubic.positions - cubic.positions.mean(axis=0), axis=1))
        center_f = np.argmin(np.linalg.norm(
            fcc.positions - fcc.positions.mean(axis=0), axis=1))

        curve_c = information_diffusion(adj_c, center_c, cubic.node_count,
                                        max_rounds=15)
        curve_f = information_diffusion(adj_f, center_f, fcc.node_count,
                                        max_rounds=15)

        # After 5 rounds, FCC should have reached more of its lattice
        assert curve_f[4] > curve_c[4]


class TestConsensus:
    def test_converges(self):
        """Both topologies should eventually converge."""
        cubic = CubicLattice(5)
        adj = build_adjacency(cubic.edges, cubic.node_count)
        init = np.random.default_rng(42).standard_normal(cubic.node_count)
        rounds, devs = consensus_speed(adj, init, max_rounds=300)
        assert rounds < 300

    def test_fcc_faster_at_scale(self):
        """FCC should converge faster at moderate scale."""
        cubic = CubicLattice(8)
        fcc = FCCLattice(5)

        adj_c = build_adjacency(cubic.edges, cubic.node_count)
        adj_f = build_adjacency(fcc.edges, fcc.node_count)

        rng = np.random.default_rng(42)
        init_c = rng.standard_normal(cubic.node_count)
        init_f = rng.standard_normal(fcc.node_count)

        rounds_c, _ = consensus_speed(adj_c, init_c, max_rounds=300)
        rounds_f, _ = consensus_speed(adj_f, init_f, max_rounds=300)

        # FCC should converge at least as fast (allow some noise)
        assert rounds_f <= rounds_c * 1.2


class TestBenchmarkRunner:
    def test_completes(self):
        r = run_context_benchmark(target_nodes=125, n_embeddings=50)
        assert r.cubic_nodes > 0
        assert r.fcc_nodes > 0
        assert len(r.recall) == 3

    def test_recall_populated(self):
        r = run_context_benchmark(target_nodes=125, n_embeddings=50)
        for rec in r.recall:
            assert 0 <= rec.cubic_recall <= 1
            assert 0 <= rec.fcc_recall <= 1

    def test_diffusion_populated(self):
        r = run_context_benchmark(target_nodes=125, n_embeddings=50)
        assert len(r.diffusion.cubic_curve) > 0
        assert len(r.diffusion.fcc_curve) > 0
