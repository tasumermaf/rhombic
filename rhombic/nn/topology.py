"""Bridge topology derived from rhombic dodecahedron face-sharing geometry.

The RD's 12 faces form 6 antipodal pairs, each corresponding to one FCC
direction pair. Two direction pairs couple through shared octahedral vertices:
co-planar pairs share 4, cross-planar pairs share 2.

This module computes that coupling from the polyhedron's combinatorial data,
producing the geometric prior for the RhombiLoRA bridge matrix.
"""

from __future__ import annotations

from typing import Callable

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
        'corpus' — Diagonal scaled by corpus direction weights + geometric
                   coupling. The proprietary weight distribution that Paper 2
                   showed amplifies Fiedler ratio from 2.3x to 6.1x.
                   Requires n_channels=6 and corpus_private.json.
        'corpus_coupled' — I_6 + corpus-derived off-diagonal coupling.
                   Hexagram × geometric × thread_density on off-diag,
                   identity on diagonal. Corrects L-026 (weights on
                   off-diagonal, not diagonal). Requires n_channels=6
                   and corpus_private.json.

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
    elif mode == 'corpus':
        if n_channels != 6:
            raise ValueError("'corpus' mode requires n_channels=6")
        from rhombic.corpus import corpus_available, edge_values, direction_weights
        if not corpus_available():
            raise ValueError(
                "'corpus' mode requires corpus_private.json. "
                "See rhombic.corpus for details."
            )
        # Sorted-bucketed direction weights from Paper 2 Experiment 5
        values = [float(v) for v in edge_values()]
        dir_weights = direction_weights(values, n_directions=6)
        # Normalize to mean=1 so the bridge starts near identity scale
        mean_w = sum(dir_weights) / len(dir_weights)
        normed = [w / mean_w for w in dir_weights]
        # Diagonal: corpus-weighted channels
        bridge = np.diag(np.array(normed, dtype=np.float64))
        # Off-diagonal: geometric coupling (same as 'geometric' mode)
        coupling = direction_pair_coupling()
        off_diag = coupling.copy()
        np.fill_diagonal(off_diag, 0.0)
        max_val = off_diag.max()
        if max_val > 0:
            off_diag /= max_val
        eps = 0.01
        bridge += eps * off_diag
        return bridge
    elif mode == 'corpus_coupled':
        if n_channels != 6:
            raise ValueError("'corpus_coupled' mode requires n_channels=6")
        from rhombic.corpus import corpus_available, edge_values, corpus_coupled_matrix
        if not corpus_available():
            raise ValueError(
                "'corpus_coupled' mode requires corpus_private.json. "
                "See rhombic.corpus for details."
            )
        values = edge_values()
        return corpus_coupled_matrix(values)
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            "Use 'identity', 'geometric', 'corpus', or 'corpus_coupled'."
        )


def create_emanation_bridge(
    n_channels: int = 6,
    bridge_mode: str = 'identity',
    num_layers: int = 1,
) -> tuple:
    """Create a shared master bridge and a factory for per-layer projections.

    Emanation architecture: a single master bridge (n x n) is shared across
    all layers. Each layer gets a layer_proj parameter that modulates the
    master bridge via sigmoid gating. The effective per-layer bridge is:

        effective_bridge = master_bridge * 2 * sigmoid(layer_proj)

    With layer_proj initialized to zeros, sigmoid(0) = 0.5 and 2 * 0.5 = 1.0,
    so the initial effective bridge equals the master bridge exactly.

    Parameters
    ----------
    n_channels : int
        Number of bridge channels. Default 6 (RD direction pairs).
    bridge_mode : str
        Initialization mode for the master bridge. Any mode accepted by
        bridge_init(): 'identity', 'geometric', 'corpus', 'corpus_coupled'.
    num_layers : int
        Number of layers (informational only; the factory can be called
        any number of times). Default 1.

    Returns
    -------
    master_bridge : torch.nn.Parameter
        The shared master bridge parameter. Pass this to each
        RhombiLoRALinear(master_bridge=...) constructor.
    make_layer_proj : callable
        Factory function taking no arguments, returning an nn.Parameter
        of shape (n_channels, n_channels) initialized to zeros (identity-
        preserving). Provided for convenience when building layers outside
        of RhombiLoRALinear.
    """
    import torch
    import torch.nn as nn

    bridge_np = bridge_init(n_channels, mode=bridge_mode)
    master_bridge = nn.Parameter(torch.from_numpy(bridge_np).float())

    def make_layer_proj() -> nn.Parameter:
        """Create a layer projection parameter (zeros = identity-preserving)."""
        return nn.Parameter(torch.zeros(n_channels, n_channels))

    return master_bridge, make_layer_proj
