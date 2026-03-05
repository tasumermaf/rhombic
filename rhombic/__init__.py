"""
rhombic — Lattice topology library for the Continental computation experiment.

Two lattice topologies, same node count, same bounding volume.
The only variable is the fundamental cell: cube vs rhombic dodecahedron.
"""

from rhombic.lattice import CubicLattice, FCCLattice
from rhombic.index import FCCIndex, CubicIndex

__all__ = ["CubicLattice", "FCCLattice", "FCCIndex", "CubicIndex"]
__version__ = "0.1.2"
