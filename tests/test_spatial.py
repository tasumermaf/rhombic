"""Tests for Rung 2: Spatial operations."""

import numpy as np
import pytest
from rhombic.lattice import CubicLattice, FCCLattice
from rhombic.spatial import (
    nearest_neighbor_search,
    range_query_sphere,
    range_query_box,
    flood_fill,
    build_adjacency,
    SpatialHash,
    run_spatial_benchmark,
)


# ── Nearest-neighbor search ──────────────────────────────────────────

class TestNearestNeighbor:
    def test_finds_exact_node(self):
        """Query at a node position should return that node."""
        cubic = CubicLattice(5)
        # Query at node 0's position
        query = cubic.positions[0:1]
        idx, _ = nearest_neighbor_search(cubic.positions, query)
        assert idx[0] == 0

    def test_accuracy(self):
        """KDTree should agree with brute force."""
        cubic = CubicLattice(5)
        rng = np.random.default_rng(42)
        queries = rng.uniform(0, 4, size=(50, 3))
        idx, _ = nearest_neighbor_search(cubic.positions, queries)
        # Brute force check
        for i, qp in enumerate(queries):
            true_nn = np.argmin(np.linalg.norm(cubic.positions - qp, axis=1))
            assert idx[i] == true_nn


# ── Range queries ────────────────────────────────────────────────────

class TestRangeQueries:
    def test_sphere_contains_center_node(self):
        """Sphere centered on a node should contain that node."""
        cubic = CubicLattice(5)
        center = cubic.positions[62]  # middle-ish node
        result = range_query_sphere(cubic.positions, center, 0.5)
        assert 62 in result

    def test_sphere_radius_zero(self):
        """Radius 0 at a node position returns just that node."""
        cubic = CubicLattice(5)
        center = cubic.positions[0]
        result = range_query_sphere(cubic.positions, center, 0.0)
        assert len(result) == 1
        assert result[0] == 0

    def test_sphere_large_radius(self):
        """Large enough radius should capture all nodes."""
        cubic = CubicLattice(3)
        center = cubic.positions.mean(axis=0)
        result = range_query_sphere(cubic.positions, center, 100.0)
        assert len(result) == cubic.node_count

    def test_box_contains_nodes(self):
        """Box covering the whole lattice should return all nodes."""
        cubic = CubicLattice(3)
        result = range_query_box(
            cubic.positions,
            np.array([-1.0, -1.0, -1.0]),
            np.array([3.0, 3.0, 3.0])
        )
        assert len(result) == cubic.node_count

    def test_box_empty(self):
        """Box outside the lattice should return nothing."""
        cubic = CubicLattice(3)
        result = range_query_box(
            cubic.positions,
            np.array([10.0, 10.0, 10.0]),
            np.array([11.0, 11.0, 11.0])
        )
        assert len(result) == 0


# ── Flood fill ───────────────────────────────────────────────────────

class TestFloodFill:
    def test_zero_hops(self):
        """Zero hops should return just the start node."""
        cubic = CubicLattice(3)
        adj = build_adjacency(cubic.edges, cubic.node_count)
        result = flood_fill(adj, 0, 0)
        assert result == {0}

    def test_one_hop_cubic(self):
        """One hop from interior node in cubic should reach 6 neighbors + self."""
        cubic = CubicLattice(5)
        adj = build_adjacency(cubic.edges, cubic.node_count)
        # Node at (2,2,2) should be interior
        center_idx = None
        for idx, pos in enumerate(cubic.positions):
            if tuple(pos) == (2.0, 2.0, 2.0):
                center_idx = idx
                break
        assert center_idx is not None
        result = flood_fill(adj, center_idx, 1)
        assert len(result) == 7  # self + 6 neighbors

    def test_fcc_more_reach(self):
        """FCC flood fill should reach more nodes than cubic at same hops."""
        cubic = CubicLattice(8)
        fcc = FCCLattice(5)

        adj_c = build_adjacency(cubic.edges, cubic.node_count)
        adj_f = build_adjacency(fcc.edges, fcc.node_count)

        # Start from node nearest to center of bounding box (interior)
        center_c = cubic.positions.mean(axis=0)
        center_f = fcc.positions.mean(axis=0)
        start_c = int(np.argmin(np.linalg.norm(cubic.positions - center_c, axis=1)))
        start_f = int(np.argmin(np.linalg.norm(fcc.positions - center_f, axis=1)))

        reached_c = flood_fill(adj_c, start_c, 3)
        reached_f = flood_fill(adj_f, start_f, 3)

        # FCC should reach more from interior (12-connected vs 6-connected)
        assert len(reached_f) > len(reached_c)


# ── Spatial hashing ──────────────────────────────────────────────────

class TestSpatialHash:
    def test_node_in_own_cell(self):
        """Every node should be findable in its own cell."""
        cubic = CubicLattice(3)
        sh = SpatialHash(cubic.positions, 1.0)
        for idx, pos in enumerate(cubic.positions):
            cell_contents = sh.query(pos)
            assert idx in cell_contents

    def test_neighborhood_contains_neighbors(self):
        """Neighborhood query should include adjacent cells."""
        cubic = CubicLattice(5)
        sh = SpatialHash(cubic.positions, 1.0)
        # Query at center — neighborhood should return multiple nodes
        center = cubic.positions[62]
        neighbors = sh.query_neighborhood(center)
        assert len(neighbors) > 1


# ── Integration ──────────────────────────────────────────────────────

def test_spatial_benchmark_runs():
    """Smoke test: spatial benchmark completes at small scale."""
    result = run_spatial_benchmark(target_nodes=125, n_queries=50)
    assert result.cubic_nodes > 0
    assert result.fcc_nodes > 0
    assert result.cubic_nn_accuracy > 0.9
    assert result.fcc_nn_accuracy > 0.9
    assert result.fcc_flood_reached >= result.cubic_flood_reached
