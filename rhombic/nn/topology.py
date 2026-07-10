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


def rd_adjacency_mask(n_channels: int = 6) -> np.ndarray:
    """Return the RD-derived adjacency mask for graph convolution bridge.

    For n=6 (FCC direction pairs), the coupling structure is:
      - Diagonal: 1.0 (self-connection)
      - Co-planar pairs (coupling=4): 1.0 (strong RD-adjacent)
      - Cross-planar pairs (coupling=2): 0.5 (weak RD-adjacent)

    The mask is FIXED (non-learnable). Learnable edge weights multiply it.
    Topology is structural by construction, not an optimization target.

    Parameters
    ----------
    n_channels : int
        Must be 6 for geometric meaning. Other values get a fully-connected
        mask (identity + uniform off-diagonal) as a fallback.

    Returns
    -------
    mask : (n_channels, n_channels) ndarray
        Fixed topology mask. Values in {0.5, 1.0}.
    """
    if n_channels != 6:
        # Fallback: fully connected with uniform off-diagonal
        mask = np.ones((n_channels, n_channels), dtype=np.float64)
        return mask

    coupling = direction_pair_coupling()
    mask = np.eye(n_channels, dtype=np.float64)
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            # Co-planar (coupling=4) → 1.0, Cross-planar (coupling=2) → 0.5
            mask[i, j] = coupling[i, j] / 4.0
    return mask


def shuffled_rd_adjacency_mask(n_channels: int = 6, seed: int = 42) -> np.ndarray:
    """Wrong-symmetry twin of ``rd_adjacency_mask`` (BM-004 prereg §6 / hard fix F3).

    A seeded relabeling of the RD channels applied simultaneously to the mask's
    rows and columns — ``mask[perm][:, perm]`` — so the OFF-DIAGONAL pattern is
    permuted while the diagonal (1.0), the off-diagonal edge count, and the
    weight multiset ({1.0 co-planar, 0.5 cross-planar}) are all preserved
    exactly. Permutations that are automorphisms of the RD relation (they leave
    the mask invariant) are REJECTED and redrawn — the mask-level mirror of
    ``bm004_transit_data.label_permutation_is_geometric``'s rejection discipline
    (BM-000 rewire-null pattern) — so the returned mask genuinely DIFFERS from
    ``rd_adjacency_mask``: a misaligned prior, not the same prior relabeled.

    Same conventions as ``rd_adjacency_mask``: symmetric, values in {0.5, 1.0},
    diagonal 1.0. Deterministic in ``seed`` (numpy ``default_rng``).

    Parameters
    ----------
    n_channels : int
        Must be 6. The n != 6 fallback mask is uniform (fully connected) and has
        no wrong-symmetry twin — every permutation is an automorphism — so this
        raises rather than loop forever.
    seed : int
        Seed for the permutation draw (the trainer's ``--seed``).

    Returns
    -------
    mask : (n_channels, n_channels) ndarray
        A permuted copy of the RD mask, guaranteed not equal to it.
    """
    if n_channels != 6:
        raise ValueError(
            "'shuffled_rd' requires n_channels=6 (the RD direction pairs); "
            "the n != 6 fallback mask is uniform and has no wrong-symmetry twin."
        )
    base = rd_adjacency_mask(n_channels)
    rng = np.random.default_rng(seed)
    # Reject relation automorphisms (perms that leave the mask unchanged), the
    # mask-level analog of label_permutation_is_geometric: keep the first draw
    # whose conjugation actually moves the off-diagonal pattern.
    for _ in range(1000):
        perm = rng.permutation(n_channels)
        shuffled = base[np.ix_(perm, perm)]
        if not np.array_equal(shuffled, base):
            return shuffled
    raise RuntimeError(  # pragma: no cover — unreachable for the RD mask
        "no non-automorphic permutation of the RD mask found in 1000 draws")


def bridge_init(n_channels: int = 6, mode: str = 'identity',
                seed: int = 42) -> np.ndarray:
    """Create a bridge initialization matrix.

    Parameters
    ----------
    n_channels : int
        Number of channels. Default 6 (FCC direction pairs).
    mode : str
        'identity' — I_n (standard LoRA behavior at init).
        'rd_graph' — I_n + 0.1*(rd_adjacency_mask - I): the initial edge weights
                   that multiply the fixed RD topology mask.
        'shuffled_rd' — same convention over the seeded wrong-symmetry twin
                   (shuffled_rd_adjacency_mask): I_n + 0.1*(shuffled_mask - I).
                   Uses ``seed`` (BM-004 hard fix F3). Requires n_channels=6.
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
    elif mode == 'rd_graph':
        # RD graph convolution: topology-weighted initialization.
        # Diagonal=1.0, co-planar=0.1, cross-planar=0.05.
        # The rd_adjacency_mask provides the fixed topology;
        # these are the INITIAL edge weights that multiply it.
        return np.eye(n_channels, dtype=np.float64) + 0.1 * (
            rd_adjacency_mask(n_channels) - np.eye(n_channels, dtype=np.float64)
        )
    elif mode == 'shuffled_rd':
        # Wrong-symmetry twin of rd_graph (BM-004 hard fix F3): the same
        # I + 0.1*(mask - I) convention over the SHUFFLED mask instead of the
        # RD mask, so the initial edge weights multiply a fixed misaligned
        # topology. Seeded from the trainer's --seed.
        return np.eye(n_channels, dtype=np.float64) + 0.1 * (
            shuffled_rd_adjacency_mask(n_channels, seed=seed)
            - np.eye(n_channels, dtype=np.float64)
        )
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
            "Use 'identity', 'rd_graph', 'shuffled_rd', 'geometric', "
            "'corpus', or 'corpus_coupled'."
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
