"""
rhombic — Lattice topology library for the Continental computation experiment.

Two lattice topologies, same node count, same bounding volume.
The only variable is the fundamental cell: cube vs rhombic dodecahedron.
"""

from rhombic.lattice import CubicLattice, FCCLattice

__all__ = ["CubicLattice", "FCCLattice"]
__version__ = "0.1.0"
