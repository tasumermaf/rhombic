"""Bridge topology derived from rhombic dodecahedron face-sharing geometry.

The RD's 12 faces form 6 antipodal pairs, each corresponding to one FCC
direction pair. Two direction pairs couple through shared octahedral vertices:
co-planar pairs share 4, cross-planar pairs share 2.

This module computes that coupling from the polyhedron's combinatorial data,
producing the geometric prior for the RhombiLoRA bridge matrix.
"""

from __future__ import annotations

import numpy as np

from rhombic.polyhedron import RhombicDodecahedron, ALL_VERTICES


def direction_pair_coupling() -> np.ndarray:
    """Compute 6x6 coupling matrix from RD face-sharing geometry.

    Each face is assigned to one of 6 direction pairs by its outward normal.
    coupling[i,j] = number of shared octahedral vertices between direction
    pair i and direction pair j.

    Returns
    -------
    coupling : (6, 6) ndarray
        Symmetric matrix. Diagonal = 4, co-planar = 4, cross-planar = 2.
    """
    rd = RhombicDodecahedron()

    # 6 canonical direction pair representatives (unit vectors).
    # Order matches FCCLattice.edge_directions():
    #   0: (1,1,0)  1: (1,-1,0)  2: (1,0,1)
    #   3: (1,0,-1) 4: (0,1,1)   5: (0,1,-1)
    canonical_dirs = np.array([
        [1, 1, 0], [1, -1, 0],
        [1, 0, 1], [1, 0, -1],
        [0, 1, 1], [0, 1, -1],
    ], dtype=np.float64)
    canonical_dirs /= np.linalg.norm(canonical_dirs, axis=1, keepdims=True)

    # Assign each face to a direction pair via face-center normal
    face_to_pair: list[int] = []
    for face_verts in rd.faces:
        coords = np.array([ALL_VERTICES[v] for v in face_verts], dtype=np.float64)
        center = coords.mean(axis=0)
        center_unit = center / np.linalg.norm(center)
        # Match to canonical direction (or its antipode)
        dots = np.abs(canonical_dirs @ center_unit)
        face_to_pair.append(int(np.argmax(dots)))

    # For each direction pair, collect octahedral vertex indices
    pair_oct_verts: dict[int, set[int]] = {i: set() for i in range(6)}
    for face_idx, pair_idx in enumerate(face_to_pair):
        for v in rd.faces[face_idx]:
            if v >= 8:  # octahedral vertices are indices 8-13
                pair_oct_verts[pair_idx].add(v)

    # Build coupling matrix: shared octahedral vertex count
    coupling = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            coupling[i, j] = len(pair_oct_verts[i] & pair_oct_verts[j])

    return coupling


def bridge_init(n_channels: int = 6, mode: str = 'identity') -> np.ndarray:
    """Create a bridge initialization matrix.

    Parameters
    ----------
    n_channels : int
        Number of channels. Default 6 (FCC direction pairs).
    mode : str
        'identity' — I_n (standard LoRA behavior at init).
        'geometric' — I_6 + eps * normalized_coupling. Requires n_channels=6.

    Returns
    -------
    bridge : (n_channels, n_channels) ndarray
    """
    if mode == 'identity':
        return np.eye(n_channels, dtype=np.float64)
    elif mode == 'geometric':
        if n_channels != 6:
            raise ValueError("'geometric' mode requires n_channels=6")
        coupling = direction_pair_coupling()
        # Extract off-diagonal coupling as perturbation
        off_diag = coupling.copy()
        np.fill_diagonal(off_diag, 0.0)
        max_val = off_diag.max()
        if max_val > 0:
            off_diag /= max_val
        eps = 0.01
        return np.eye(6, dtype=np.float64) + eps * off_diag
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'identity' or 'geometric'.")
