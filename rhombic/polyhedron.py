"""
Polyhedra as first-class graph objects.

RhombicDodecahedron: 14 vertices (8 cubic/trivalent + 6 octahedral/tetravalent),
24 edges, 12 faces. Voronoi cell of the FCC lattice.

cuboctahedron_graph: 12 vertices, 24 edges, 4-regular Archimedean solid.
Same edge count as RD but different vertex count and degree sequence.
"""

from __future__ import annotations

import math
import networkx as nx


# ── Fixed combinatorial data ──────────────────────────────────────────

# 8 cubic vertices (trivalent, degree 3) — indices 0-7
CUBIC_VERTICES: list[tuple[int, int, int]] = [
    ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),
    (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1),
]

# 6 octahedral vertices (tetravalent, degree 4) — indices 8-13
OCTAHEDRAL_VERTICES: list[tuple[int, int, int]] = [
    ( 2,  0,  0), (-2,  0,  0),
    ( 0,  2,  0), ( 0, -2,  0),
    ( 0,  0,  2), ( 0,  0, -2),
]

ALL_VERTICES: list[tuple[int, int, int]] = CUBIC_VERTICES + OCTAHEDRAL_VERTICES

def _build_edges() -> list[tuple[int, int]]:
    """Compute edges: cubic-octahedral pairs at distance sqrt(3).

    For vertices at (±1,±1,±1) and (±2,0,0)/(0,±2,0)/(0,0,±2),
    the nearest cubic-octahedral distance is sqrt(3). Each cubic vertex
    has 3 octahedral neighbors; each octahedral vertex has 4 cubic neighbors.
    """
    edges = []
    for ci in range(8):
        cx, cy, cz = ALL_VERTICES[ci]
        for oi in range(8, 14):
            ox, oy, oz = ALL_VERTICES[oi]
            dx, dy, dz = cx - ox, cy - oy, cz - oz
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq == 3:  # exact integer arithmetic
                edges.append((ci, oi))
    return edges


def _build_faces(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Compute the 12 rhombic faces.

    Each face is a rhombus with 4 vertices: 2 cubic + 2 octahedral,
    alternating around the boundary. Two vertices share a face if they
    are connected by a path of length 2 through the face's other vertices.
    """
    # Build adjacency
    adj: dict[int, set[int]] = {i: set() for i in range(14)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    faces = []
    seen: set[frozenset[int]] = set()

    # Each face has exactly 2 cubic and 2 octahedral vertices.
    # Two cubic vertices c1, c2 and two octahedral vertices o1, o2
    # form a face when c1-o1, c1-o2, c2-o1, c2-o2 are all edges.
    for c1 in range(8):
        oct_neighbors_c1 = adj[c1]
        for c2 in range(c1 + 1, 8):
            oct_neighbors_c2 = adj[c2]
            shared_oct = oct_neighbors_c1 & oct_neighbors_c2
            if len(shared_oct) == 2:
                o1, o2 = sorted(shared_oct)
                face_key = frozenset([c1, c2, o1, o2])
                if face_key not in seen:
                    seen.add(face_key)
                    faces.append(sorted([c1, c2, o1, o2]))

    return faces


# Pre-compute once at module level
_EDGES = _build_edges()
_FACES = _build_faces(_EDGES)


class RhombicDodecahedron:
    """The rhombic dodecahedron: 14 vertices, 24 edges, 12 rhombic faces.

    This is a fixed combinatorial object, not parameterized by size.
    """

    def __init__(self):
        self.vertices: list[tuple[int, int, int]] = list(ALL_VERTICES)
        self.edges: list[tuple[int, int]] = list(_EDGES)
        self.faces: list[list[int]] = [list(f) for f in _FACES]

        # Precompute adjacency for star queries
        self._adj: dict[int, list[int]] = {i: [] for i in range(14)}
        for u, v in self.edges:
            self._adj[u].append(v)
            self._adj[v].append(u)

        # Precompute edge-to-index mapping
        self._edge_index: dict[tuple[int, int], int] = {}
        for idx, (u, v) in enumerate(self.edges):
            self._edge_index[(u, v)] = idx
            self._edge_index[(v, u)] = idx

        # Precompute vertex-to-edge incidence
        self._vertex_edges: dict[int, list[int]] = {i: [] for i in range(14)}
        for idx, (u, v) in enumerate(self.edges):
            self._vertex_edges[u].append(idx)
            self._vertex_edges[v].append(idx)

    @property
    def trivalent_vertices(self) -> list[int]:
        """The 8 cubic vertex indices (degree 3)."""
        return list(range(8))

    @property
    def tetravalent_vertices(self) -> list[int]:
        """The 6 octahedral vertex indices (degree 4)."""
        return list(range(8, 14))

    def vertex_star(self, v: int) -> list[int]:
        """Edge indices incident to vertex v."""
        return list(self._vertex_edges[v])

    def face_edges(self, f: int) -> list[int]:
        """Edge indices bounding face f."""
        face_verts = self.faces[f]
        result = []
        for i, u in enumerate(face_verts):
            for v in face_verts[i + 1:]:
                key = (u, v)
                if key in self._edge_index:
                    result.append(self._edge_index[key])
        return result

    def degree(self, v: int) -> int:
        """Degree of vertex v."""
        return len(self._adj[v])

    def neighbors(self, v: int) -> list[int]:
        """Neighbor vertices of v."""
        return list(self._adj[v])

    def to_networkx(self, edge_weights: dict[int, float] | list[float] | None = None) -> nx.Graph:
        """Convert to a NetworkX graph.

        Parameters
        ----------
        edge_weights : dict or list or None
            If provided, assigns 'weight' attribute to each edge.
            Dict maps edge index -> weight. List provides weights in edge order.
        """
        G = nx.Graph()
        for idx, coords in enumerate(self.vertices):
            vtype = "cubic" if idx < 8 else "octahedral"
            G.add_node(idx, pos=coords, vertex_type=vtype)

        for idx, (u, v) in enumerate(self.edges):
            attrs = {}
            if edge_weights is not None:
                if isinstance(edge_weights, dict):
                    attrs['weight'] = edge_weights.get((u, v), edge_weights.get(idx, 1.0))
                else:
                    attrs['weight'] = edge_weights[idx]
            G.add_edge(u, v, **attrs)

        return G

    def __repr__(self) -> str:
        return f"RhombicDodecahedron(V=14, E=24, F=12)"


# ── Cuboctahedron ───────────────────────────────────────────────────

# 12 vertices: all permutations of (0, ±1, ±1)
_CUBOCT_VERTICES: list[tuple[int, int, int]] = [
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
]


def _build_cuboct_edges() -> list[tuple[int, int]]:
    """Edges connect vertices at distance sqrt(2)."""
    edges = []
    n = len(_CUBOCT_VERTICES)
    for i in range(n):
        x1, y1, z1 = _CUBOCT_VERTICES[i]
        for j in range(i + 1, n):
            x2, y2, z2 = _CUBOCT_VERTICES[j]
            dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            if dist_sq == 2:
                edges.append((i, j))
    return edges


_CUBOCT_EDGES = _build_cuboct_edges()


def cuboctahedron_graph() -> nx.Graph:
    """Cuboctahedron: 12 vertices, 24 edges, 4-regular Archimedean solid.

    Same edge count as the rhombic dodecahedron but different structure:
    all vertices have degree 4 (vs bipartite 3/4 for RD), 12 vertices
    (vs 14 for RD), NOT a space-filling Voronoi cell.
    """
    G = nx.Graph()
    for idx, coords in enumerate(_CUBOCT_VERTICES):
        G.add_node(idx, pos=coords)
    G.add_edges_from(_CUBOCT_EDGES)
    return G
