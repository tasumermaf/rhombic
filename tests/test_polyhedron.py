"""Tests for polyhedron graph objects (RD + cuboctahedron)."""

import pytest
import networkx as nx
from rhombic.polyhedron import RhombicDodecahedron, cuboctahedron_graph


@pytest.fixture
def rd():
    return RhombicDodecahedron()


class TestCombinatorics:
    """Verify the fundamental combinatorial properties."""

    def test_vertex_count(self, rd):
        assert len(rd.vertices) == 14

    def test_edge_count(self, rd):
        assert len(rd.edges) == 24

    def test_face_count(self, rd):
        assert len(rd.faces) == 12

    def test_euler_formula(self, rd):
        """V - E + F = 2 for any convex polyhedron."""
        V = len(rd.vertices)
        E = len(rd.edges)
        F = len(rd.faces)
        assert V - E + F == 2

    def test_trivalent_count(self, rd):
        assert len(rd.trivalent_vertices) == 8

    def test_tetravalent_count(self, rd):
        assert len(rd.tetravalent_vertices) == 6

    def test_trivalent_degrees(self, rd):
        """All cubic vertices have degree 3."""
        for v in rd.trivalent_vertices:
            assert rd.degree(v) == 3

    def test_tetravalent_degrees(self, rd):
        """All octahedral vertices have degree 4."""
        for v in rd.tetravalent_vertices:
            assert rd.degree(v) == 4

    def test_total_incidences(self, rd):
        """Sum of degrees = 2 * |E|. Also: 8*3 + 6*4 = 48 = 2*24."""
        total = sum(rd.degree(v) for v in range(14))
        assert total == 2 * len(rd.edges)
        assert total == 48

    def test_edges_are_bipartite(self, rd):
        """Every edge connects a cubic vertex to an octahedral vertex."""
        for u, v in rd.edges:
            types = {u < 8, v < 8}
            assert types == {True, False}, f"Edge ({u}, {v}) is not bipartite"


class TestFaces:
    """Verify face structure."""

    def test_each_face_has_four_vertices(self, rd):
        for face in rd.faces:
            assert len(face) == 4

    def test_each_face_has_two_cubic_two_octahedral(self, rd):
        for face in rd.faces:
            cubic = sum(1 for v in face if v < 8)
            octahedral = sum(1 for v in face if v >= 8)
            assert cubic == 2
            assert octahedral == 2

    def test_face_edges_count(self, rd):
        """Each face should have 4 bounding edges."""
        for f_idx in range(12):
            fe = rd.face_edges(f_idx)
            assert len(fe) == 4, f"Face {f_idx} has {len(fe)} edges, expected 4"

    def test_all_edges_covered_by_faces(self, rd):
        """Each edge borders exactly 2 faces."""
        edge_face_count = [0] * 24
        for f_idx in range(12):
            for e_idx in rd.face_edges(f_idx):
                edge_face_count[e_idx] += 1
        for e_idx, count in enumerate(edge_face_count):
            assert count == 2, f"Edge {e_idx} appears in {count} faces"


class TestStars:
    """Verify vertex star queries."""

    def test_trivalent_star_size(self, rd):
        for v in rd.trivalent_vertices:
            assert len(rd.vertex_star(v)) == 3

    def test_tetravalent_star_size(self, rd):
        for v in rd.tetravalent_vertices:
            assert len(rd.vertex_star(v)) == 4

    def test_star_edges_are_incident(self, rd):
        """Every edge in vertex_star(v) is incident to v."""
        for v in range(14):
            for e_idx in rd.vertex_star(v):
                u, w = rd.edges[e_idx]
                assert v in (u, w), f"Edge {e_idx} ({u},{w}) not incident to {v}"


class TestNetworkX:
    """Verify NetworkX conversion."""

    def test_connected(self, rd):
        G = rd.to_networkx()
        assert nx.is_connected(G)

    def test_node_count(self, rd):
        G = rd.to_networkx()
        assert G.number_of_nodes() == 14

    def test_edge_count(self, rd):
        G = rd.to_networkx()
        assert G.number_of_edges() == 24

    def test_weighted_graph(self, rd):
        weights = list(range(1, 25))
        G = rd.to_networkx(edge_weights=weights)
        for u, v, data in G.edges(data=True):
            assert 'weight' in data
            assert data['weight'] >= 1

    def test_vertex_types(self, rd):
        G = rd.to_networkx()
        cubic_count = sum(1 for _, d in G.nodes(data=True) if d['vertex_type'] == 'cubic')
        oct_count = sum(1 for _, d in G.nodes(data=True) if d['vertex_type'] == 'octahedral')
        assert cubic_count == 8
        assert oct_count == 6

    def test_diameter(self, rd):
        """Rhombic dodecahedron graph has diameter 4."""
        G = rd.to_networkx()
        assert nx.diameter(G) == 4


class TestRepr:
    def test_repr(self, rd):
        assert "14" in repr(rd)
        assert "24" in repr(rd)
        assert "12" in repr(rd)


# ── Cuboctahedron tests ──────────────────────────────────────────────


class TestCuboctahedron:

    def test_vertex_count(self):
        G = cuboctahedron_graph()
        assert G.number_of_nodes() == 12

    def test_edge_count(self):
        G = cuboctahedron_graph()
        assert G.number_of_edges() == 24

    def test_four_regular(self):
        """Every vertex has degree 4."""
        G = cuboctahedron_graph()
        for v in G.nodes():
            assert G.degree(v) == 4

    def test_connected(self):
        G = cuboctahedron_graph()
        assert nx.is_connected(G)
