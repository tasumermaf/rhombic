"""Unit tests for lattice construction."""

import pytest
from rhombic.lattice import CubicLattice, FCCLattice


class TestCubicLattice:

    def test_node_count(self):
        """Cubic lattice with n nodes per side has n^3 total nodes."""
        for n in (2, 3, 5, 7):
            lattice = CubicLattice(n)
            assert lattice.node_count == n ** 3

    def test_interior_degree_6(self):
        """Interior nodes of a cubic lattice have exactly 6 neighbors."""
        G = CubicLattice(5).to_networkx()
        # Interior nodes: positions where all coords are in [1, 3]
        interior = [
            n for n, d in G.nodes(data=True)
            if all(1.0 <= c <= 3.0 for c in d['pos'])
        ]
        assert len(interior) > 0
        for n in interior:
            assert G.degree(n) == 6

    def test_connected(self):
        """Cubic lattice is a single connected component."""
        import networkx as nx
        G = CubicLattice(4).to_networkx()
        assert nx.is_connected(G)

    def test_edge_count_bounds(self):
        """Edge count should be roughly 3n^2(n-1) for n^3 cubic lattice."""
        n = 5
        lattice = CubicLattice(n)
        expected = 3 * n * n * (n - 1)
        assert lattice.edge_count == expected

    def test_stats(self):
        lattice = CubicLattice(3)
        stats = lattice.stats()
        assert stats.name.startswith("Cubic")
        assert stats.node_count == 27
        assert stats.connectivity == 6


class TestFCCLattice:

    def test_node_count_approx(self):
        """FCC lattice with n unit cells per side has approximately 4n^3 nodes."""
        for n in (2, 3, 4):
            lattice = FCCLattice(n)
            # Deduplication at boundaries means slightly fewer than 4n^3
            assert lattice.node_count <= 4 * n ** 3
            assert lattice.node_count >= n ** 3  # at least as many as cubic

    def test_interior_degree_12(self):
        """Interior nodes of an FCC lattice have exactly 12 neighbors."""
        lattice = FCCLattice(5)
        G = lattice.to_networkx()
        positions = lattice.positions
        # Interior: nodes far from boundaries
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)
        margin = lattice.a * 1.0
        interior = [
            n for n, d in G.nodes(data=True)
            if all(bbox_min[i] + margin <= d['pos'][i] <= bbox_max[i] - margin
                   for i in range(3))
        ]
        assert len(interior) > 0, "No interior nodes found"
        for n in interior:
            assert G.degree(n) == 12, f"Node {n} has degree {G.degree(n)}, expected 12"

    def test_connected(self):
        """FCC lattice is a single connected component."""
        import networkx as nx
        G = FCCLattice(3).to_networkx()
        assert nx.is_connected(G)

    def test_stats(self):
        lattice = FCCLattice(3)
        stats = lattice.stats()
        assert stats.name.startswith("FCC")
        assert stats.connectivity == 12


class TestMatchedLattices:

    def test_comparable_node_counts(self):
        """matched_lattices produces lattices with roughly similar node counts."""
        from rhombic.benchmark import matched_lattices
        cubic, fcc = matched_lattices(1000)
        ratio = fcc.node_count / cubic.node_count
        assert 0.5 <= ratio <= 2.0, f"Node count ratio {ratio} too far from 1.0"

    def test_both_connected(self):
        """Both matched lattices are connected."""
        import networkx as nx
        from rhombic.benchmark import matched_lattices
        cubic, fcc = matched_lattices(125)
        assert nx.is_connected(cubic.to_networkx())
        assert nx.is_connected(fcc.to_networkx())
