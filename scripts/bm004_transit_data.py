"""BM-004 paired-transit corpus builder — pure polytope geometry, public math.

Pre-registration: docs/BM004_PREREGISTRATION_v2_2026-07-07.md (read it first).
Plan lineage: docs/BM_BATTERY_PLAN.md ("BM-004"), docs/MERIDIAN_RESEARCH_POSITION_2026-07-02.md
section 3.2 (the never-run fair test of the original conception).

What this builds
----------------
Training/eval text corpora over the rhombic-dodecahedral cell adjacency:
FCC lattice cells (integer triples with even coordinate sum, the D3 lattice),
each with 12 face-neighbors at the (+-1,+-1,0)-type offsets. Four artifacts:

  (i)   transit-walk sequences   — tokenized cell/face random-walk paths (ABS frame)
  (ii)  paired samples           — the same walk in two frames (EGO => ABS);
                                   translating between frames requires applying
                                   the transit relation step-by-step: the frame
                                   pair IS the geometric-relation signal (the
                                   paired control-data pattern)
  (iii) articulation variant     — walks interrupted mid-sequence; the model is
                                   trained to STATE the local adjacency relation
                                   (counterfactual-reflection template, workspace
                                   paper import WS-4 — see pre-registration)
  (iv)  shuffled-adjacency twin  — the negative-control corpus (see below)

Design decisions (recorded here AND in the output manifest; all flagged for
Director sign-off in the pre-registration's pinned-defaults table)
------------------------------------------------------------------
D1. FCC adjacency from first principles: cells = {(x,y,z) in Z^3 : x+y+z even};
    the 12 neighbor offsets are the full orbit of (1,1,0) under signed
    permutation, ordered to match rhombic/lattice.py FCC_NEIGHBOR_OFFSETS
    (integer units, that array x2) and, at the direction-pair level,
    rhombic/nn/topology.py canonical_dirs — so face labels F00..F11 map onto
    the rd_graph mask's 6 channels by a fixed, documented table
    (face_channel_map). No import from the library: property tests verify the
    construction independently.
D2. Shuffled twin = PER-CELL scrambled face->offset assignment (a fresh seeded
    permutation of the 12 offsets at every cell). A single GLOBAL label
    permutation was considered and REJECTED: for a token-embedding LM a global
    relabeling is gauge-equivalent to the matched corpus (the model just learns
    permuted embeddings), so it controls nothing. Per-cell scrambling preserves
    every graph-level statistic (same neighbor SET, degree 12, symmetric,
    same token frequencies, same walk lengths) while destroying translation
    invariance of the face->displacement rule — the relation can only be
    learned as per-cell LOOKUP, never as one reusable rule. That is exactly
    the contrast BM-004's transfer/swap endpoints need.
D3. Text grammar (version bm004-v2, exact; parsers below are the reference
    implementation):
      ABS walk:      ABS CELL x y z F## CELL x y z ... F## CELL x y z END
      EGO walk:      EGO START x y z F## F## ... END
      Paired:        PAIR <ego-text> => <abs-text>
      Articulation:  <abs-prefix ... F##> INTERRUPT STATE FROM CELL x y z
                     EXIT F## OFFSET dx dy dz ARRIVE CELL x y z ENTER F##
                     RESUME CELL x y z <abs-continuation> END
    The STATE clause contains geometry facts only (offset arithmetic + entry
    face); it is valid in both arms (in the shuffled arm it states the
    scrambled-but-actual relation). No corpus-derived values anywhere —
    IP boundary holds by construction.
D4. Walk faces are drawn uniformly over the 12 faces (immediate backtracking
    allowed — uniform is the simplest pre-specifiable null; any non-uniform
    walk policy would be an unregistered forking point).
D5. Determinism: all randomness from numpy.random.default_rng seeded with
    integer lists [seed, stream_tag, arm_index, record_index] (per-cell
    permutations: [seed, 44, x+SHIFT, y+SHIFT, z+SHIFT]). Output files are
    byte-deterministic in their arguments; no wall clock anywhere.
D6. Defaults: seed=20260707, walk_len=24, box_radius=8 (start-cell sampling
    box; coordinates may drift up to walk_len beyond it),
    articulation_fraction=0.075 — a fraction OF PAIRED SEQUENCES:
    n_articulation = round(f * n_pairs). That is the denominator the July 7
    workspace-mapping memo states its 5-10% band in ("~5-10% of paired
    sequences"; 0.075 is the midpoint). It is NOT a fraction of the arm's
    total records (at the pinned 1:1 mixture and 40k+40k production sizes,
    3,000 articulation records = 3.61% of the 83,000-record E4 arm; a
    total-records denominator at 7.5% would put articulation at 16.2% of
    pairs, OUTSIDE the memo band — denominator pinned 2026-07-07, review
    fix). Builder-default n_walks=n_pairs=2000 (production sizes are pinned
    in the pre-registration, not here — cheap by default).
D7. The corpus-level invariance criterion (pre-registration hard fix F1) is
    the JOINT action: relabel face tokens by a Gram-automorphism pi AND
    transform all cell coordinates by induced_cell_map(pi), the unique
    signed-permutation matrix g with g @ OFFSETS[f] == OFFSETS[pi[f]].
    Token-only relabeling is NOT an invariance of ABS/PAIR records (it maps
    them off the support) — see the F1 joint-equivariance tests.

Safety
------
Refuses to write into any path containing an 'asset1-bank' component (the
LIVE campaign tree) and refuses to clobber existing non-empty directories
that do not carry this builder's manifest marker. CPU/numpy only — torch is
never imported; CUDA cannot be touched.

Usage
-----
    python scripts/bm004_transit_data.py --out-dir results/BM-004/data-dev
    (all knobs have documented defaults; --seed defaults to 20260707)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# ── Geometry constants (first principles; ordering documented in D1) ──

# 12 FCC nearest-neighbor offsets, integer units. Order matches
# rhombic/lattice.py FCC_NEIGHBOR_OFFSETS (x0.5 there) — verified by tests.
OFFSETS = np.array([
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
], dtype=np.int64)

N_FACES = 12
N_CHANNELS = 6

# Canonical direction-pair representatives — same order as
# rhombic/nn/topology.py canonical_dirs (the rd_graph mask's channel order).
CANONICAL_DIRS = np.array([
    [1, 1, 0], [1, -1, 0],
    [1, 0, 1], [1, 0, -1],
    [0, 1, 1], [0, 1, -1],
], dtype=np.int64)

SEED_DEFAULT = 20260707
STREAM_WALKS, STREAM_PAIRS, STREAM_ARTIC, STREAM_CELLPERM = 11, 22, 33, 44
COORD_SHIFT = 1 << 20          # SeedSequence entries must be non-negative
ARM_INDEX = {"matched": 0, "shuffled": 1}

GRAMMAR_VERSION = "bm004-v2"
MANIFEST_NAME = "BM004_TRANSIT_DATA_MANIFEST.json"


# ── Face tables (derived, not hand-typed; tests re-derive them) ──────


def antipode_map(offsets: np.ndarray = OFFSETS) -> np.ndarray:
    """a[i] = j such that offsets[j] == -offsets[i]. Fixed-point-free
    involution (every RD face has exactly one opposite face)."""
    amap = np.full(len(offsets), -1, dtype=np.int64)
    for i in range(len(offsets)):
        for j in range(len(offsets)):
            if np.array_equal(offsets[j], -offsets[i]):
                amap[i] = j
                break
    if (amap < 0).any() or (amap == np.arange(len(offsets))).any() \
            or not (amap[amap] == np.arange(len(offsets))).all():
        raise AssertionError("antipode map is not a fixed-point-free involution")
    return amap


def face_channel_map(offsets: np.ndarray = OFFSETS) -> np.ndarray:
    """c[i] = rd_graph bridge channel (0..5) for face i: the index of the
    canonical direction pair containing +-offsets[i]. Antipodal faces share
    a channel — the mask's 6 channels are the 6 antipodal face pairs."""
    cmap = np.full(len(offsets), -1, dtype=np.int64)
    for i, off in enumerate(offsets):
        for c, d in enumerate(CANONICAL_DIRS):
            if np.array_equal(off, d) or np.array_equal(off, -d):
                cmap[i] = c
                break
    if (cmap < 0).any():
        raise AssertionError("face has no canonical direction pair")
    return cmap


ANTIPODE = antipode_map()
FACE_CHANNEL = face_channel_map()


def relation_matrix(offsets: np.ndarray = OFFSETS) -> np.ndarray:
    """12x12 Gram matrix R[i,j] = offsets[i] . offsets[j] (values in
    {-2,-1,0,1,2}). Encodes antipodality (R=-2) and co/cross-planarity —
    the label-level shadow of the rd_graph mask's coupling structure."""
    return offsets @ offsets.T


def label_permutation_is_geometric(perm: np.ndarray,
                                   offsets: np.ndarray = OFFSETS) -> bool:
    """True iff perm is an automorphism of the geometric relation, i.e.
    R[perm][:, perm] == R. The 48 octahedral symmetries induce exactly such
    permutations (faithfully — 48 distinct perms; property-tested). This
    identifies WHICH label permutations qualify for the matched-arm
    invariance criterion (hard fix F1 in the pre-registration); the
    invariance action itself is the JOINT one — perm on face tokens together
    with induced_cell_map(perm) on all cell coordinates (design note D7)."""
    R = relation_matrix(offsets)
    p = np.asarray(perm, dtype=np.int64)
    return bool(np.array_equal(R[np.ix_(p, p)], R))


def induced_cell_map(perm: np.ndarray,
                     offsets: np.ndarray = OFFSETS) -> np.ndarray:
    """The unique 3x3 signed-permutation (integer orthogonal) matrix g_pi
    with g_pi @ offsets[f] == offsets[perm[f]] for every face f — the
    cell-coordinate half of the F1 joint invariance action (design note D7).

    Exists precisely when perm is a relation-matrix automorphism (raises
    ValueError otherwise): a Gram-preserving map on a spanning set extends
    uniquely to an orthogonal transform of R^3, and the symmetry group of
    the 12-offset set is the octahedral group O_h (order 48), whose elements
    are signed permutation matrices. Construction: solve on three linearly
    independent offsets, then verify on all twelve.
    """
    p = np.asarray(perm, dtype=np.int64)
    if not label_permutation_is_geometric(p, offsets):
        raise ValueError(
            "permutation is not a relation-matrix automorphism; "
            "no induced cell map exists (F1: only the 48 geometric "
            "permutations act on the corpus)")
    basis = [0, 2, 4]                     # (1,1,0), (-1,1,0), (1,0,1) — independent
    B = offsets[basis].T.astype(np.float64)
    target = offsets[p[basis]].T.astype(np.float64)
    g = np.rint(target @ np.linalg.inv(B)).astype(np.int64)
    if not np.array_equal(g @ offsets.T, offsets[p].T):
        raise AssertionError("induced cell map failed verification on all offsets")
    if not np.array_equal(g @ g.T, np.eye(3, dtype=np.int64)):
        raise AssertionError("induced cell map is not orthogonal")
    return g


# ── Transit geometry (matched vs shuffled adjacency) ─────────────────


class TransitGeometry:
    """RD/FCC transit adjacency.

    mode='matched':  translation-invariant rule — exiting cell c through
        face f arrives at c + OFFSETS[f], entering through ANTIPODE[f].
    mode='shuffled': per-cell scrambled face->offset assignment (design
        decision D2). Same neighbor sets, degree 12, symmetric returns —
        but the face->displacement rule differs cell to cell, so no single
        reusable relation exists.
    """

    def __init__(self, mode: str = "matched", seed: int = SEED_DEFAULT):
        if mode not in ("matched", "shuffled"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.seed = seed
        self._perm_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _cell_perm(self, cell: tuple[int, int, int]) -> np.ndarray:
        key = tuple(int(v) for v in cell)
        if key not in self._perm_cache:
            entries = [self.seed, STREAM_CELLPERM] + \
                      [int(v) + COORD_SHIFT for v in key]
            if min(entries) < 0:
                raise ValueError(f"cell coordinate out of seedable range: {key}")
            rng = np.random.default_rng(entries)
            self._perm_cache[key] = rng.permutation(N_FACES)
        return self._perm_cache[key]

    def face_offset(self, cell: tuple[int, int, int], face: int) -> np.ndarray:
        if self.mode == "matched":
            return OFFSETS[face]
        return OFFSETS[self._cell_perm(cell)[face]]

    def step(self, cell: tuple[int, int, int], face: int
             ) -> tuple[tuple[int, int, int], int]:
        """Exit `cell` through `face`; return (arrival_cell, entry_face)."""
        off = self.face_offset(cell, face)
        nxt = (int(cell[0] + off[0]), int(cell[1] + off[1]),
               int(cell[2] + off[2]))
        if self.mode == "matched":
            return nxt, int(ANTIPODE[face])
        # entry face g at nxt satisfies perm_nxt[g] == index of -off
        k = int(ANTIPODE[self._cell_perm(cell)[face]])
        perm_nxt = self._cell_perm(nxt)
        g = int(np.where(perm_nxt == k)[0][0])
        return nxt, g

    def neighbors(self, cell: tuple[int, int, int]
                  ) -> list[tuple[int, tuple[int, int, int]]]:
        """All 12 (face, arrival_cell) pairs from `cell`."""
        return [(f, self.step(cell, f)[0]) for f in range(N_FACES)]


# ── Walk generation ──────────────────────────────────────────────────


def sample_start_cell(rng: np.random.Generator, box_radius: int
                      ) -> tuple[int, int, int]:
    """Uniform lattice cell (even coordinate sum) in the sampling box."""
    while True:
        c = rng.integers(-box_radius, box_radius + 1, size=3)
        if int(c.sum()) % 2 == 0:
            return (int(c[0]), int(c[1]), int(c[2]))


def generate_walk(geom: TransitGeometry, rng: np.random.Generator,
                  walk_len: int, box_radius: int
                  ) -> tuple[list[tuple[int, int, int]], list[int], list[int]]:
    """Uniform random transit walk. Returns (cells, faces, entry_faces) with
    len(cells) == walk_len + 1 and len(faces) == len(entry_faces) == walk_len."""
    cells = [sample_start_cell(rng, box_radius)]
    faces: list[int] = []
    entries: list[int] = []
    for _ in range(walk_len):
        f = int(rng.integers(0, N_FACES))
        nxt, entry = geom.step(cells[-1], f)
        faces.append(f)
        entries.append(entry)
        cells.append(nxt)
    return cells, faces, entries


# ── Text emission (grammar bm004-v2; see D3) ─────────────────────────


def cell_tok(cell: tuple[int, int, int]) -> str:
    return f"CELL {cell[0]} {cell[1]} {cell[2]}"


def face_tok(face: int) -> str:
    return f"F{face:02d}"


def abs_text(cells: list[tuple[int, int, int]], faces: list[int]) -> str:
    parts = ["ABS", cell_tok(cells[0])]
    for f, c in zip(faces, cells[1:]):
        parts.append(face_tok(f))
        parts.append(cell_tok(c))
    parts.append("END")
    return " ".join(parts)


def ego_text(start: tuple[int, int, int], faces: list[int]) -> str:
    parts = ["EGO", "START", str(start[0]), str(start[1]), str(start[2])]
    parts.extend(face_tok(f) for f in faces)
    parts.append("END")
    return " ".join(parts)


def pair_text(cells: list[tuple[int, int, int]], faces: list[int]) -> str:
    return f"PAIR {ego_text(cells[0], faces)} => {abs_text(cells, faces)}"


def articulation_text(geom: TransitGeometry,
                      cells: list[tuple[int, int, int]], faces: list[int],
                      interrupt_index: int) -> str:
    """ABS walk interrupted at transition `interrupt_index` (0-based); the
    STATE clause articulates that single transition, then the walk resumes."""
    k = interrupt_index
    if not 0 <= k < len(faces):
        raise ValueError(f"interrupt_index {k} out of range")
    off = geom.face_offset(cells[k], faces[k])
    _, entry = geom.step(cells[k], faces[k])
    parts = ["ABS", cell_tok(cells[0])]
    for i in range(k):
        parts.append(face_tok(faces[i]))
        parts.append(cell_tok(cells[i + 1]))
    parts.append(face_tok(faces[k]))
    parts.extend([
        "INTERRUPT", "STATE",
        "FROM", cell_tok(cells[k]),
        "EXIT", face_tok(faces[k]),
        "OFFSET", str(int(off[0])), str(int(off[1])), str(int(off[2])),
        "ARRIVE", cell_tok(cells[k + 1]),
        "ENTER", face_tok(entry),
        "RESUME", cell_tok(cells[k + 1]),
    ])
    for i in range(k + 1, len(faces)):
        parts.append(face_tok(faces[i]))
        parts.append(cell_tok(cells[i + 1]))
    parts.append("END")
    return " ".join(parts)


# ── Reference parsers (the eval harness will reuse these) ────────────


def parse_abs(text: str) -> tuple[list[tuple[int, int, int]], list[int]]:
    """Inverse of abs_text. Raises ValueError on malformed input."""
    toks = text.split()
    if toks[0] != "ABS" or toks[-1] != "END":
        raise ValueError("not an ABS walk")
    i, cells, faces = 1, [], []
    while i < len(toks) - 1:
        if toks[i] == "CELL":
            cells.append((int(toks[i + 1]), int(toks[i + 2]), int(toks[i + 3])))
            i += 4
        elif toks[i].startswith("F"):
            faces.append(int(toks[i][1:]))
            i += 1
        else:
            raise ValueError(f"unexpected token {toks[i]!r}")
    return cells, faces


def parse_ego(text: str) -> tuple[tuple[int, int, int], list[int]]:
    """Inverse of ego_text."""
    toks = text.split()
    if toks[:2] != ["EGO", "START"] or toks[-1] != "END":
        raise ValueError("not an EGO walk")
    start = (int(toks[2]), int(toks[3]), int(toks[4]))
    faces = [int(t[1:]) for t in toks[5:-1]]
    return start, faces


def parse_pair(text: str) -> tuple[str, str]:
    """Split a PAIR record into (ego_text, abs_text)."""
    if not text.startswith("PAIR "):
        raise ValueError("not a PAIR record")
    ego, sep, abs_ = text[5:].partition(" => ")
    if not sep:
        raise ValueError("PAIR record missing '=>'")
    return ego, abs_


def parse_articulation_state(text: str) -> dict:
    """Extract the STATE clause fields from an articulation record."""
    toks = text.split()
    i = toks.index("STATE")
    if toks[i + 1] != "FROM" or toks[i + 2] != "CELL":
        raise ValueError("malformed STATE clause")
    return {
        "from": (int(toks[i + 3]), int(toks[i + 4]), int(toks[i + 5])),
        "exit": int(toks[i + 7][1:]),          # after 'EXIT'
        "offset": (int(toks[i + 9]), int(toks[i + 10]), int(toks[i + 11])),
        "arrive": (int(toks[i + 14]), int(toks[i + 15]), int(toks[i + 16])),
        "enter": int(toks[i + 18][1:]),        # after 'ENTER'
    }


def unroll_ego(geom: TransitGeometry, start: tuple[int, int, int],
               faces: list[int]) -> list[tuple[int, int, int]]:
    """Apply the transit relation step-by-step (EGO -> ABS decoding)."""
    cells = [start]
    for f in faces:
        cells.append(geom.step(cells[-1], f)[0])
    return cells


# ── Corpus builder ───────────────────────────────────────────────────


def _guard_out_dir(out_dir: Path) -> None:
    """Refuse the live campaign tree and refuse clobbering foreign dirs."""
    resolved = out_dir.resolve()
    if "asset1-bank" in resolved.parts:
        raise ValueError(
            f"REFUSING to write into {resolved}: the path contains an "
            f"'asset1-bank' component — that is the LIVE campaign tree and "
            f"is write-protected.")
    if resolved.exists():
        if any(resolved.iterdir()) and not (resolved / MANIFEST_NAME).exists():
            raise ValueError(
                f"REFUSING to write into existing non-empty directory "
                f"{resolved}: it does not carry {MANIFEST_NAME}, so it is "
                f"not a previous BM-004 corpus. Choose an empty/new directory.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def build_corpus(out_dir: str | Path,
                 n_walks: int = 2000,
                 n_pairs: int = 2000,
                 articulation_fraction: float = 0.075,
                 walk_len: int = 24,
                 box_radius: int = 8,
                 seed: int = SEED_DEFAULT,
                 arms: tuple[str, ...] = ("matched", "shuffled")) -> dict:
    """Write the BM-004 paired-transit corpus. Returns the manifest dict.

    Files per arm (JSONL, one record per line, byte-deterministic):
        {arm}_walks.jsonl, {arm}_pairs.jsonl, {arm}_articulation.jsonl
    Plus BM004_TRANSIT_DATA_MANIFEST.json recording every parameter,
    default, face table, and file hash (runs self-document).
    """
    if min(n_walks, n_pairs, walk_len) < 1 or box_radius < 1:
        raise ValueError("n_walks, n_pairs, walk_len, box_radius must be >= 1")
    if not 0.0 <= articulation_fraction <= 1.0:
        raise ValueError("articulation_fraction must be in [0, 1]")
    for arm in arms:
        if arm not in ARM_INDEX:
            raise ValueError(f"unknown arm {arm!r}")

    out_dir = Path(out_dir)
    _guard_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Denominator pinned to PAIRED sequences (D6; pre-registration section 2):
    # the mapping memo's 5-10% band is stated as a fraction of paired
    # sequences, and that — not the arm total, not n_walks — is the pinned
    # semantics. (2026-07-07 review fix: was round(f * n_walks), which
    # coincides only at the 1:1 mixture.)
    n_articulation = max(1, round(articulation_fraction * n_pairs))
    files: dict[str, dict] = {}

    for arm in arms:
        a = ARM_INDEX[arm]
        geom = TransitGeometry(mode=arm, seed=seed)

        rows = []
        for i in range(n_walks):
            rng = np.random.default_rng([seed, STREAM_WALKS, a, i])
            cells, faces, _ = generate_walk(geom, rng, walk_len, box_radius)
            rows.append({"id": i, "arm": arm, "kind": "walk",
                         "text": abs_text(cells, faces),
                         "meta": {"start": list(cells[0]), "faces": faces}})
        path = out_dir / f"{arm}_walks.jsonl"
        _write_jsonl(path, rows)
        files[path.name] = {"n_records": len(rows), "sha256": _sha256(path)}

        rows = []
        for i in range(n_pairs):
            rng = np.random.default_rng([seed, STREAM_PAIRS, a, i])
            cells, faces, _ = generate_walk(geom, rng, walk_len, box_radius)
            rows.append({"id": i, "arm": arm, "kind": "pair",
                         "text": pair_text(cells, faces),
                         "meta": {"start": list(cells[0]), "faces": faces}})
        path = out_dir / f"{arm}_pairs.jsonl"
        _write_jsonl(path, rows)
        files[path.name] = {"n_records": len(rows), "sha256": _sha256(path)}

        rows = []
        for i in range(n_articulation):
            rng = np.random.default_rng([seed, STREAM_ARTIC, a, i])
            cells, faces, _ = generate_walk(geom, rng, walk_len, box_radius)
            k = int(rng.integers(0, walk_len))
            rows.append({"id": i, "arm": arm, "kind": "articulation",
                         "text": articulation_text(geom, cells, faces, k),
                         "meta": {"start": list(cells[0]), "faces": faces,
                                  "interrupt_index": k}})
        path = out_dir / f"{arm}_articulation.jsonl"
        _write_jsonl(path, rows)
        files[path.name] = {"n_records": len(rows), "sha256": _sha256(path)}

    manifest = {
        "generator": "scripts/bm004_transit_data.py",
        "grammar_version": GRAMMAR_VERSION,
        "preregistration": "docs/BM004_PREREGISTRATION_v2_2026-07-07.md",
        "params": {
            "n_walks": n_walks, "n_pairs": n_pairs,
            "articulation_fraction": articulation_fraction,
            "n_articulation": n_articulation,
            "walk_len": walk_len, "box_radius": box_radius,
            "seed": seed, "arms": list(arms),
        },
        "mixture_accounting": {
            "articulation_denominator": "n_pairs",
            "note": "articulation_fraction is a fraction of PAIRED sequences "
                    "(the July 7 mapping memo band is stated as '~5-10% of "
                    "paired sequences'), NOT of the arm's total records. "
                    "See pre-registration section 2 for the pinned "
                    "production accounting.",
            "n_records_per_arm_without_articulation": n_walks + n_pairs,
            "n_records_per_arm_with_articulation":
                n_walks + n_pairs + n_articulation,
            "articulation_share_of_pairs": n_articulation / n_pairs,
            "articulation_share_of_arm_total":
                n_articulation / (n_walks + n_pairs + n_articulation),
        },
        "invented_defaults_for_director_signoff": [
            "seed=20260707; walk_len=24; box_radius=8 (start-cell box)",
            "articulation_fraction=0.075 OF PAIRED SEQUENCES (midpoint of "
            "the memo's 5-10%-of-paired-sequences band; denominator pinned "
            "to n_pairs — at production 1:1 sizes this is 3.61% of the E4 "
            "arm's total records; a total-records denominator at 7.5% would "
            "land at 16.2% of pairs, outside the memo band)",
            "walk policy: uniform over 12 faces, backtracking allowed (D4)",
            "text grammar bm004-v2 (D3): ABS/EGO/PAIR/INTERRUPT-STATE-RESUME",
            "shuffled twin = per-cell face->offset scramble (D2; global "
            "label permutation rejected as gauge-equivalent for a "
            "token-embedding LM)",
            "builder-default corpus sizes n_walks=n_pairs=2000 (production "
            "sizes pinned in the pre-registration, not here)",
        ],
        "geometry": {
            "cells": "integer triples with even coordinate sum (D3 lattice)",
            "offsets": OFFSETS.tolist(),
            "antipode_map": ANTIPODE.tolist(),
            "face_channel_map": FACE_CHANNEL.tolist(),
            "relation_matrix": relation_matrix().tolist(),
        },
        "files": files,
    }
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the BM-004 paired-transit corpus (pure RD/FCC "
                    "geometry, public math). Deterministic; refuses the "
                    "live asset1-bank tree; CPU/numpy only.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-walks", type=int, default=2000)
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--articulation-fraction", type=float, default=0.075,
                        help="fraction of PAIRED sequences (n_articulation = "
                             "round(f * n_pairs)); the memo band's "
                             "denominator, NOT the arm total (D6)")
    parser.add_argument("--walk-len", type=int, default=24)
    parser.add_argument("--box-radius", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--arms", nargs="+", default=["matched", "shuffled"],
                        choices=["matched", "shuffled"])
    args = parser.parse_args()

    manifest = build_corpus(
        args.out_dir, n_walks=args.n_walks, n_pairs=args.n_pairs,
        articulation_fraction=args.articulation_fraction,
        walk_len=args.walk_len, box_radius=args.box_radius,
        seed=args.seed, arms=tuple(args.arms))
    total = sum(f["n_records"] for f in manifest["files"].values())
    print(f"BM-004 corpus written: {args.out_dir} "
          f"({total} records across {len(manifest['files'])} files; "
          f"grammar {manifest['grammar_version']}, seed {args.seed})")


if __name__ == "__main__":
    main()
