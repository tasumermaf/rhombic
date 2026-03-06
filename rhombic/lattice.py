"""
Core lattice implementations: Cubic (6-connected) and FCC (12-connected).

Both lattices fill the same bounding volume with comparable node counts.
The cubic lattice places nodes at integer coordinates on a simple grid.
The FCC lattice places nodes at the positions of a face-centered cubic
crystal — every corner plus every face center of the cubic unit cell.

The FCC lattice is the Voronoi dual of the rhombic dodecahedral tessellation:
each FCC node sits at the center of a rhombic dodecahedron, and two nodes
are neighbors iff their Voronoi cells share a face. 12 faces = 12 neighbors.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Iterator


# ── FCC neighbor offsets ─────────────────────────────────────────────
# In the conventional cubic unit cell, FCC sites are at:
#   (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
# Each site has 12 nearest neighbors at distance a/√2 (a = lattice param).
# The 12 offset vectors (in units of 0.5a):
FCC_NEIGHBOR_OFFSETS = np.array([
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
], dtype=np.float64) * 0.5

# Cubic neighbor offsets (6-connected)
CUBIC_NEIGHBOR_OFFSETS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.float64)


@dataclass
class LatticeStats:
    """Summary statistics for a lattice instance."""
    name: str
    node_count: int
    edge_count: int
    connectivity: int          # neighbors per interior node
    density: float             # nodes per unit volume
    avg_degree: float          # actual mean degree (boundary effects)
    bounding_box: tuple        # (xmin, ymin, zmin, xmax, ymax, zmax)


class CubicLattice:
    """Simple cubic lattice: nodes at integer coordinates, 6-connected.

    Parameters
    ----------
    n : int
        Nodes per side. Total nodes = n³.
    spacing : float
        Distance between adjacent nodes. Default 1.0.
    """

    def __init__(self, n: int, spacing: float = 1.0):
        self.n = n
        self.spacing = spacing
        self._build()

    def _build(self):
        n = self.n
        # Node positions: (i, j, k) * spacing for i,j,k in [0, n-1]
        ijk = np.mgrid[0:n, 0:n, 0:n].reshape(3, -1).T  # shape (n³, 3)
        self.positions = ijk.astype(np.float64) * self.spacing
        self.node_count = len(self.positions)

        # Build coordinate -> index map
        self._coord_to_idx: dict[tuple, int] = {}
        for idx, (i, j, k) in enumerate(ijk):
            self._coord_to_idx[(int(i), int(j), int(k))] = idx

        # Build edges
        edges = []
        for idx, (i, j, k) in enumerate(ijk):
            for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                ni, nj, nk = int(i+di), int(j+dj), int(k+dk)
                neighbor = self._coord_to_idx.get((ni, nj, nk))
                if neighbor is not None:
                    edges.append((idx, neighbor))
        self.edges = edges
        self.edge_count = len(edges)

    def to_networkx(self) -> nx.Graph:
        """Convert to networkx Graph with position attributes."""
        G = nx.Graph()
        for idx, pos in enumerate(self.positions):
            G.add_node(idx, pos=tuple(pos))
        G.add_edges_from(self.edges)
        return G

    def stats(self) -> LatticeStats:
        bbox = self.positions.min(axis=0)
        bbox_max = self.positions.max(axis=0)
        volume = np.prod(bbox_max - bbox) if np.all(bbox_max > bbox) else 0.0
        G = self.to_networkx()
        degrees = [d for _, d in G.degree()]
        return LatticeStats(
            name=f"Cubic {self.n}³",
            node_count=self.node_count,
            edge_count=self.edge_count,
            connectivity=6,
            density=self.node_count / volume if volume > 0 else float('inf'),
            avg_degree=np.mean(degrees),
            bounding_box=(*bbox, *bbox_max),
        )

    def edge_directions(self) -> dict[int, list[int]]:
        """Classify edges by direction pair.

        Cubic lattice has 3 direction pairs: x(0), y(1), z(2).
        Returns dict mapping direction index to list of edge indices.
        """
        directions: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for idx, (u, v) in enumerate(self.edges):
            diff = self.positions[v] - self.positions[u]
            axis = int(np.argmax(np.abs(diff)))
            directions[axis].append(idx)
        return directions

    def __repr__(self):
        return f"CubicLattice(n={self.n}, nodes={self.node_count}, edges={self.edge_count})"


class FCCLattice:
    """Face-centered cubic lattice: 12-connected.

    Each conventional cubic unit cell contains 4 FCC sites:
      (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    scaled by the lattice parameter `a`.

    The spacing is chosen so that the bounding volume matches a cubic
    lattice of equivalent extent, enabling fair comparison.

    Parameters
    ----------
    n : int
        Unit cells per side. Total nodes ≈ 4n³.
    a : float
        Lattice parameter (unit cell side length). Default 1.0.
    """

    # The 4 basis vectors within each unit cell (fractional coordinates)
    BASIS = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])

    def __init__(self, n: int, a: float = 1.0):
        self.n = n
        self.a = a
        self._build()

    def _build(self):
        n, a = self.n, self.a

        # Generate all sites: for each unit cell (i,j,k), place 4 basis atoms
        positions = []
        coord_to_idx: dict[tuple, int] = {}
        idx = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    origin = np.array([i, j, k], dtype=np.float64)
                    for b, basis in enumerate(self.BASIS):
                        pos = (origin + basis) * a
                        # Use a hashable key: round to avoid float issues
                        key = tuple(np.round(pos / (a * 0.5)).astype(int))
                        if key not in coord_to_idx:
                            coord_to_idx[key] = idx
                            positions.append(pos)
                            idx += 1

        self.positions = np.array(positions)
        self.node_count = len(positions)
        self._coord_to_idx = coord_to_idx

        # Build edges: for each node, check all 12 FCC neighbor offsets
        edges_set: set[tuple[int, int]] = set()
        for node_idx, pos in enumerate(positions):
            for offset in FCC_NEIGHBOR_OFFSETS:
                neighbor_pos = pos + offset * self.a
                key = tuple(np.round(neighbor_pos / (self.a * 0.5)).astype(int))
                neighbor_idx = coord_to_idx.get(key)
                if neighbor_idx is not None and neighbor_idx != node_idx:
                    edge = (min(node_idx, neighbor_idx), max(node_idx, neighbor_idx))
                    edges_set.add(edge)

        self.edges = list(edges_set)
        self.edge_count = len(self.edges)

    def to_networkx(self) -> nx.Graph:
        """Convert to networkx Graph with position attributes."""
        G = nx.Graph()
        for idx, pos in enumerate(self.positions):
            G.add_node(idx, pos=tuple(pos))
        G.add_edges_from(self.edges)
        return G

    def stats(self) -> LatticeStats:
        bbox = self.positions.min(axis=0)
        bbox_max = self.positions.max(axis=0)
        volume = np.prod(bbox_max - bbox) if np.all(bbox_max > bbox) else 0.0
        G = self.to_networkx()
        degrees = [d for _, d in G.degree()]
        return LatticeStats(
            name=f"FCC {self.n}³ ({self.node_count} nodes)",
            node_count=self.node_count,
            edge_count=self.edge_count,
            connectivity=12,
            density=self.node_count / volume if volume > 0 else float('inf'),
            avg_degree=np.mean(degrees),
            bounding_box=(*bbox, *bbox_max),
        )

    def edge_directions(self) -> dict[int, list[int]]:
        """Classify edges by direction pair.

        FCC lattice has 12 neighbor directions forming 6 antipodal pairs.
        Returns dict mapping pair index (0-5) to list of edge indices.

        Pair 0: (1,1,0)/(-1,-1,0)   Pair 1: (1,-1,0)/(-1,1,0)
        Pair 2: (1,0,1)/(-1,0,-1)   Pair 3: (1,0,-1)/(-1,0,1)
        Pair 4: (0,1,1)/(0,-1,-1)   Pair 5: (0,1,-1)/(0,-1,1)
        """
        canonical_map = {
            (1, 1, 0): 0, (-1, -1, 0): 0,
            (1, -1, 0): 1, (-1, 1, 0): 1,
            (1, 0, 1): 2, (-1, 0, -1): 2,
            (1, 0, -1): 3, (-1, 0, 1): 3,
            (0, 1, 1): 4, (0, -1, -1): 4,
            (0, 1, -1): 5, (0, -1, 1): 5,
        }
        half_a = self.a * 0.5
        directions: dict[int, list[int]] = {i: [] for i in range(6)}
        for idx, (u, v) in enumerate(self.edges):
            diff = self.positions[v] - self.positions[u]
            rounded = tuple(int(round(d / half_a)) for d in diff)
            pair_idx = canonical_map.get(rounded)
            if pair_idx is not None:
                directions[pair_idx].append(idx)
        return directions

    def __repr__(self):
        return f"FCCLattice(n={self.n}, nodes={self.node_count}, edges={self.edge_count})"
