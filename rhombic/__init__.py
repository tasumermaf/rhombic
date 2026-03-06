"""
rhombic — Lattice topology library for the Continental computation experiment.

Two lattice topologies, same node count, same bounding volume.
The only variable is the fundamental cell: cube vs rhombic dodecahedron.
"""

from rhombic.lattice import CubicLattice, FCCLattice
from rhombic.index import FCCIndex, CubicIndex
from rhombic.polyhedron import RhombicDodecahedron
from rhombic.corpus import TRUMP_VALUES, NAMES_OF_POWER, TRACKED_PRIMES, edge_values

__all__ = [
    "CubicLattice", "FCCLattice", "FCCIndex", "CubicIndex",
    "RhombicDodecahedron",
    "TRUMP_VALUES", "NAMES_OF_POWER", "TRACKED_PRIMES", "edge_values",
]
__version__ = "0.3.0"
