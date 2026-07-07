"""Property tests for scripts/bm004_transit_data.py (BM-004 corpus builder).

Verifies, per the pre-registration (docs/BM004_PREREGISTRATION_v2_2026-07-07.md):
  1. Adjacency correctness — degree 12, all offsets at squared distance 2,
     lattice parity preserved, symmetric returns, translation invariance
     (matched arm) — re-derived independently of rhombic/lattice.py.
  2. Face tables — antipode involution, 6 channels of 2 antipodal faces,
     channel order matches topology.py canonical_dirs.
  3. Walk validity — every emitted ABS step is a real transit step.
  4. Pair-frame consistency — EGO decodes to exactly the ABS cells.
  5. Shuffled-twin properties — degree/neighbor-set preservation, symmetric
     returns, translation invariance BROKEN (the designed break).
  6. Articulation-variant structure — STATE clause states the true relation.
  6b. Articulation mixture accounting — denominator is PAIRED sequences
      (round(f * n_pairs)), manifest self-documents both shares
      (2026-07-07 review fix).
  6c. F1 joint-equivariance criterion — the 48 octahedral automorphisms,
      induced_cell_map roundtrip, token-only relabeling breaks the support,
      joint (pi, g_pi) action preserves it (2026-07-07 review fix).
  7. Determinism — same seed => byte-identical files; different seed differs.
  8. Safety guard — refuses asset1-bank paths.

CPU/numpy only; no torch, no CUDA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bm004_transit_data as td  # noqa: E402

SEED = td.SEED_DEFAULT
CELLS_UNDER_TEST = [(0, 0, 0), (1, 1, 0), (2, 0, 0), (-3, 1, 0), (1, -2, 1)]


# ── 1. Adjacency correctness ─────────────────────────────────────────


def test_offsets_are_the_twelve_fcc_neighbors():
    """First-principles check: OFFSETS is exactly the orbit of (1,1,0) under
    signed coordinate permutation — 12 distinct vectors, |v|^2 == 2, even sum."""
    assert td.OFFSETS.shape == (12, 3)
    assert len({tuple(o) for o in td.OFFSETS}) == 12
    assert (np.sum(td.OFFSETS ** 2, axis=1) == 2).all()
    assert (td.OFFSETS.sum(axis=1) % 2 == 0).all()
    expected = {tuple(v) for v in
                [(a * p[0], b * p[1], c * p[2])
                 for p in [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
                 for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
                if sum(abs(x) for x in v) == 2}
    assert {tuple(o) for o in td.OFFSETS} == expected


@pytest.mark.parametrize("mode", ["matched", "shuffled"])
def test_degree_twelve_distinct_neighbors(mode):
    geom = td.TransitGeometry(mode=mode, seed=SEED)
    for cell in CELLS_UNDER_TEST:
        nbrs = geom.neighbors(cell)
        assert len(nbrs) == 12
        arrival_cells = {c for _, c in nbrs}
        assert len(arrival_cells) == 12          # all distinct
        for _, c in nbrs:
            d = np.array(c) - np.array(cell)
            assert int(np.sum(d ** 2)) == 2      # nearest-neighbor distance
            assert sum(c) % 2 == 0               # stays on the lattice


@pytest.mark.parametrize("mode", ["matched", "shuffled"])
def test_adjacency_symmetry_return_through_entry_face(mode):
    """Exiting via f and re-exiting via the entry face returns to the start,
    entering through the original exit face — in BOTH arms."""
    geom = td.TransitGeometry(mode=mode, seed=SEED)
    for cell in CELLS_UNDER_TEST:
        for f in range(12):
            nxt, entry = geom.step(cell, f)
            back, back_entry = geom.step(nxt, entry)
            assert back == cell
            assert back_entry == f


def test_translation_invariance_matched():
    """neighbors(c + t) == neighbors(c) + t with identical face labels, for
    lattice translations t (even coordinate sum)."""
    geom = td.TransitGeometry(mode="matched", seed=SEED)
    for t in [(2, 0, 0), (1, 1, 0), (-1, 1, 2)]:
        assert sum(t) % 2 == 0
        for cell in CELLS_UNDER_TEST:
            shifted = (cell[0] + t[0], cell[1] + t[1], cell[2] + t[2])
            for f in range(12):
                nxt, entry = geom.step(cell, f)
                nxt_s, entry_s = geom.step(shifted, f)
                assert nxt_s == (nxt[0] + t[0], nxt[1] + t[1], nxt[2] + t[2])
                assert entry_s == entry


def test_translation_invariance_broken_in_shuffled_twin():
    """The designed break (design decision D2): the face->offset rule must
    differ across cells in the shuffled arm."""
    geom = td.TransitGeometry(mode="shuffled", seed=SEED)
    offsets_by_cell = [
        tuple(tuple(geom.face_offset(cell, f)) for f in range(12))
        for cell in CELLS_UNDER_TEST
    ]
    assert len(set(offsets_by_cell)) > 1


# ── 2. Face tables ───────────────────────────────────────────────────


def test_antipode_map_involution():
    a = td.antipode_map()
    assert (a != np.arange(12)).all()            # fixed-point-free
    assert (a[a] == np.arange(12)).all()         # involution
    for i in range(12):
        assert np.array_equal(td.OFFSETS[a[i]], -td.OFFSETS[i])


def test_face_channel_map_six_antipodal_pairs():
    cmap = td.face_channel_map()
    a = td.antipode_map()
    assert set(cmap.tolist()) == set(range(6))
    for ch in range(6):
        faces = np.where(cmap == ch)[0]
        assert len(faces) == 2
        assert a[faces[0]] == faces[1]           # each channel = one antipodal pair
    # channel order matches topology.py canonical_dirs
    for i, off in enumerate(td.OFFSETS):
        d = td.CANONICAL_DIRS[cmap[i]]
        assert np.array_equal(off, d) or np.array_equal(off, -d)


def test_geometric_permutation_criterion():
    """Identity is a relation automorphism; swapping two faces from different
    channels is not (the F1 invariance criterion actually discriminates)."""
    assert td.label_permutation_is_geometric(np.arange(12))
    perm = np.arange(12)
    # faces 0 (channel of (1,1,0)) and 4 (channel of (1,0,1)): dot = 1,
    # swapping them scrambles the relation matrix.
    perm[0], perm[4] = perm[4], perm[0]
    assert not td.label_permutation_is_geometric(perm)


# ── 3-6. Corpus content ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("bm004") / "corpus"
    manifest = td.build_corpus(out, n_walks=40, n_pairs=30,
                               articulation_fraction=0.1, walk_len=12,
                               box_radius=4, seed=SEED)
    return out, manifest


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("arm", ["matched", "shuffled"])
def test_walk_validity(corpus, arm):
    """Every ABS record is a genuine transit walk under its arm's geometry."""
    out, _ = corpus
    geom = td.TransitGeometry(mode=arm, seed=SEED)
    rows = _read_jsonl(out / f"{arm}_walks.jsonl")
    assert len(rows) == 40
    for row in rows:
        cells, faces = td.parse_abs(row["text"])
        assert len(cells) == 13 and len(faces) == 12
        for i, f in enumerate(faces):
            nxt, _ = geom.step(cells[i], f)
            assert nxt == cells[i + 1]


@pytest.mark.parametrize("arm", ["matched", "shuffled"])
def test_pair_frame_consistency(corpus, arm):
    """EGO frame decodes (via the arm's own adjacency) to exactly the ABS
    frame — the pair signal is the transit relation and nothing else."""
    out, _ = corpus
    geom = td.TransitGeometry(mode=arm, seed=SEED)
    for row in _read_jsonl(out / f"{arm}_pairs.jsonl"):
        ego, abs_ = td.parse_pair(row["text"])
        start, ego_faces = td.parse_ego(ego)
        cells, abs_faces = td.parse_abs(abs_)
        assert ego_faces == abs_faces
        assert cells[0] == start
        assert td.unroll_ego(geom, start, ego_faces) == cells


def test_shuffled_twin_degree_and_neighbor_set_preservation():
    """The shuffled twin preserves the neighbor SET (hence degree 12 and all
    graph-level statistics); only the face-label semantics are scrambled."""
    matched = td.TransitGeometry(mode="matched", seed=SEED)
    shuffled = td.TransitGeometry(mode="shuffled", seed=SEED)
    for cell in CELLS_UNDER_TEST:
        m = {c for _, c in matched.neighbors(cell)}
        s = {c for _, c in shuffled.neighbors(cell)}
        assert m == s
        assert len(s) == 12


@pytest.mark.parametrize("arm", ["matched", "shuffled"])
def test_articulation_variant_structure(corpus, arm):
    """INTERRUPT/STATE/RESUME present; the STATE clause states the TRUE
    relation of its arm (offset arithmetic + entry face); matched-arm entry
    face is the geometric antipode; walk resumes validly to END."""
    out, manifest = corpus
    geom = td.TransitGeometry(mode=arm, seed=SEED)
    rows = _read_jsonl(out / f"{arm}_articulation.jsonl")
    # Denominator is n_pairs (paired sequences), per prereg section 2:
    # round(0.1 * 30) = 3 — NOT round(0.1 * 40) (walks) and NOT of the total.
    assert len(rows) == manifest["params"]["n_articulation"] == 3
    for row in rows:
        text = row["text"]
        for kw in ("INTERRUPT", "STATE", "RESUME"):
            assert kw in text
        st = td.parse_articulation_state(text)
        # offset arithmetic is stated correctly
        assert tuple(np.array(st["from"]) + np.array(st["offset"])) == st["arrive"]
        # the stated offset and entry face are the arm's actual relation
        assert tuple(geom.face_offset(st["from"], st["exit"])) == st["offset"]
        nxt, entry = geom.step(st["from"], st["exit"])
        assert nxt == st["arrive"] and entry == st["enter"]
        if arm == "matched":
            assert st["enter"] == int(td.ANTIPODE[st["exit"]])
        # the resumed walk is itself valid: strip STATE clause, re-verify
        pre, _, post = text.partition(" INTERRUPT")
        resumed = post.split(" RESUME ", 1)[1]
        cells, faces = td.parse_abs(pre + " " + resumed
                                    if resumed.startswith("CELL")
                                    else pre + " END")
        for i, f in enumerate(faces):
            assert geom.step(cells[i], f)[0] == cells[i + 1]


# ── 6b. Articulation mixture accounting (2026-07-07 review fix) ──────
# Regression for the prereg/code mismatch: prereg section 2 originally read
# "92.5% walks+pairs / 7.5% articulation records" (a total-records
# denominator) while the builder divided by n_walks — and the mapping memo's
# band is stated in a THIRD denominator, paired sequences. The pinned
# semantics is the memo's: n_articulation = round(f * n_pairs).


def test_articulation_denominator_is_paired_sequences(tmp_path):
    """With n_walks != n_pairs the count must follow n_pairs, and the
    manifest must self-document the denominator and both shares."""
    m = td.build_corpus(tmp_path / "d", n_walks=50, n_pairs=20,
                        articulation_fraction=0.1, walk_len=4, box_radius=3,
                        seed=SEED, arms=("matched",))
    assert m["params"]["n_articulation"] == 2            # round(0.1 * 20)
    mix = m["mixture_accounting"]
    assert mix["articulation_denominator"] == "n_pairs"
    assert mix["n_records_per_arm_without_articulation"] == 70
    assert mix["n_records_per_arm_with_articulation"] == 72
    assert mix["articulation_share_of_pairs"] == pytest.approx(0.1)
    assert mix["articulation_share_of_arm_total"] == pytest.approx(2 / 72)


def test_production_mixture_arithmetic():
    """The prereg section-2 production numbers, verified as pure arithmetic:
    f=0.075 of 40,000 pairs = 3,000 articulation records = 3.61% of the
    83,000-record E4 arm — inside the memo's 5-10%-of-paired band; the
    rejected total-records reading (7.5% of total) would be 16.2% of pairs,
    OUTSIDE the band."""
    n_walks = n_pairs = 40_000
    n_art = round(0.075 * n_pairs)
    assert n_art == 3_000
    total = n_walks + n_pairs + n_art
    assert total == 83_000
    assert 0.05 <= n_art / n_pairs <= 0.10                # in the memo band
    assert n_art / total == pytest.approx(0.0361, abs=5e-4)
    n_art_total_reading = round(0.075 / 0.925 * (n_walks + n_pairs))
    assert not 0.05 <= n_art_total_reading / n_pairs <= 0.10   # out of band


# ── 6c. F1 joint-equivariance criterion (2026-07-07 review fix) ──────
# Regression for the prereg hard-fix-F1 misstatement: token-only face
# relabeling is NOT an invariance of the corpus; the true invariance is the
# JOINT action (pi on face tokens, induced_cell_map(pi) on cell coordinates).


def _signed_permutation_matrices() -> list[np.ndarray]:
    """All 48 3x3 signed permutation matrices — the octahedral group O_h,
    the symmetry group of the 12-offset set."""
    from itertools import permutations, product
    mats = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            g = np.zeros((3, 3), dtype=np.int64)
            for r, (c, s) in enumerate(zip(perm, signs)):
                g[r, c] = s
            mats.append(g)
    return mats


def _induced_face_perm(g: np.ndarray) -> np.ndarray:
    """pi with OFFSETS[pi[f]] == g @ OFFSETS[f] (g maps the offset set to
    itself for every signed permutation matrix)."""
    images = (g @ td.OFFSETS.T).T
    pi = np.full(12, -1, dtype=np.int64)
    for f in range(12):
        match = np.where((td.OFFSETS == images[f]).all(axis=1))[0]
        assert len(match) == 1
        pi[f] = match[0]
    return pi


def test_all_48_octahedral_symmetries_induce_geometric_permutations():
    """Exactly 48 signed-permutation matrices; each induces a distinct
    (faithful) label permutation; every one passes the Gram criterion."""
    mats = _signed_permutation_matrices()
    assert len(mats) == 48
    perms = []
    for g in mats:
        pi = _induced_face_perm(g)
        assert td.label_permutation_is_geometric(pi)
        perms.append(tuple(pi))
    assert len(set(perms)) == 48


def test_induced_cell_map_roundtrip_and_rejection():
    """induced_cell_map recovers g_pi from pi for all 48 automorphisms
    (existence + uniqueness) and rejects non-geometric permutations."""
    for g in _signed_permutation_matrices():
        pi = _induced_face_perm(g)
        assert np.array_equal(td.induced_cell_map(pi), g)
    bad = np.arange(12)
    bad[0], bad[4] = bad[4], bad[0]      # cross-channel swap, non-geometric
    with pytest.raises(ValueError, match="not a relation-matrix automorphism"):
        td.induced_cell_map(bad)


def test_face_relabeling_alone_breaks_every_nonidentity_automorphism():
    """The test that would have caught the F1 misstatement: applying a
    geometric pi to face tokens ALONE (cells unchanged) maps an ABS walk off
    the support for EVERY non-identity automorphism, on a walk that uses all
    twelve faces. Token-only relabeling discriminates nothing and preserves
    nothing; it is not the invariance action."""
    geom = td.TransitGeometry(mode="matched", seed=SEED)
    faces = list(range(12)) * 2          # deterministic walk using every face
    cells = td.unroll_ego(geom, (0, 0, 0), faces)
    n_checked = 0
    for g in _signed_permutation_matrices():
        pi = _induced_face_perm(g)
        if np.array_equal(pi, np.arange(12)):
            continue
        relabeled = [int(pi[f]) for f in faces]
        still_valid = all(geom.step(cells[i], f)[0] == cells[i + 1]
                          for i, f in enumerate(relabeled))
        assert not still_valid
        n_checked += 1
    assert n_checked == 47


def test_joint_action_preserves_walk_validity():
    """The corrected F1 statement: relabeling faces by pi AND transforming
    all cells by g_pi maps valid matched walks to valid matched walks, with
    entry faces transforming by pi as well (transit-step equivariance:
    g_pi(c + OFFSETS[f]) = g_pi c + OFFSETS[pi(f)])."""
    geom = td.TransitGeometry(mode="matched", seed=SEED)
    rng = np.random.default_rng([SEED, 98])
    cells, faces, entries = td.generate_walk(geom, rng, walk_len=12,
                                             box_radius=4)
    for g in _signed_permutation_matrices():
        pi = _induced_face_perm(g)
        new_cells = [tuple(int(v) for v in g @ np.asarray(c, dtype=np.int64))
                     for c in cells]
        new_faces = [int(pi[f]) for f in faces]
        for i, f in enumerate(new_faces):
            nxt, entry = geom.step(new_cells[i], f)
            assert nxt == new_cells[i + 1]
            assert entry == int(pi[entries[i]])


def test_joint_action_preserves_start_box_and_parity():
    """g_pi is a signed coordinate permutation: it maps the start-cell
    sampling box [-r, r]^3 to itself and preserves lattice parity, so the
    start-cell distribution (uniform on box intersect lattice) is
    g_pi-invariant — completing the joint-invariance argument in F1."""
    r = 4
    rng = np.random.default_rng([SEED, 97])
    starts = [td.sample_start_cell(rng, r) for _ in range(50)]
    for g in _signed_permutation_matrices():
        for c in starts:
            gc = g @ np.asarray(c, dtype=np.int64)
            assert (np.abs(gc) <= r).all()               # box preserved
            assert int(gc.sum()) % 2 == 0                # parity preserved


# ── 7. Determinism ───────────────────────────────────────────────────


def test_determinism_same_seed_byte_identical(tmp_path):
    m1 = td.build_corpus(tmp_path / "a", n_walks=15, n_pairs=10,
                         walk_len=6, box_radius=3, seed=SEED)
    m2 = td.build_corpus(tmp_path / "b", n_walks=15, n_pairs=10,
                         walk_len=6, box_radius=3, seed=SEED)
    assert m1["files"] == m2["files"]            # sha256 per file identical
    for name in m1["files"]:
        assert (tmp_path / "a" / name).read_bytes() == \
               (tmp_path / "b" / name).read_bytes()


def test_determinism_seed_sensitivity(tmp_path):
    m1 = td.build_corpus(tmp_path / "a", n_walks=15, n_pairs=10,
                         walk_len=6, box_radius=3, seed=SEED)
    m3 = td.build_corpus(tmp_path / "c", n_walks=15, n_pairs=10,
                         walk_len=6, box_radius=3, seed=SEED + 1)
    hashes1 = {n: f["sha256"] for n, f in m1["files"].items()}
    hashes3 = {n: f["sha256"] for n, f in m3["files"].items()}
    assert hashes1 != hashes3


def test_manifest_self_documents_defaults(corpus):
    """Guardrail: every invented default is recorded in the JSON output."""
    out, manifest = corpus
    on_disk = json.loads((out / td.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk["params"] == manifest["params"]
    assert on_disk["invented_defaults_for_director_signoff"]
    assert on_disk["grammar_version"] == td.GRAMMAR_VERSION
    assert on_disk["geometry"]["face_channel_map"] == td.FACE_CHANNEL.tolist()


# ── 8. Safety guard ──────────────────────────────────────────────────


def test_guard_refuses_asset1_bank_paths(tmp_path):
    with pytest.raises(ValueError, match="asset1-bank"):
        td.build_corpus(tmp_path / "asset1-bank" / "anything",
                        n_walks=1, n_pairs=1, walk_len=2)


def test_guard_refuses_foreign_nonempty_dir(tmp_path):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "unrelated.txt").write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="not a previous BM-004 corpus"):
        td.build_corpus(target, n_walks=1, n_pairs=1, walk_len=2)
