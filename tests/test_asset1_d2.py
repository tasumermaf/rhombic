"""Tests for the Asset 1 D2 cross-task bridge-swap tool.

Covers: plan determinism (same seed -> identical plan, sha-stable across
regenerations), plan structure (eval-count formula, unique ids, recipient
reuse, family filter, insufficient-runs errors), assembly correctness on a
synthetic bank (recipient lora_A/lora_B preserved bit-exact, donor bridge
installed, module-name matching, family mixing raises), permuted-bridge
properties (same entry multiset, not equal to original, no fixed points,
seeded-reproducible), the permuted_deviation reference (round-1 review
fix: identity backbone preserved, trained-deviation multiset preserved,
distance to native bounded by 2||D||_F while the full-entry permutation
is dominated by identity-backbone destruction; shared sigma with the
permuted cell; Director sign-off flag in the plan), the magnitude/
topology decomposition (exact round-trip, self-transplant identity,
manual-formula cross-check), the CLI interlock + out-dir guards, penalty
computation, and Stage B import safety (transformers blocked -> plan
path unaffected, --evaluate refused at the gate before any lazy import).

Pre-registration hygiene: every statistic below runs on SYNTHETIC fixtures
generated in tmp dirs. Nothing reads or writes results/asset1-bank/, no HF
downloads, no network, no GPU (CUDA is never initialized).
"""

from __future__ import annotations

import importlib.abc
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

torch = pytest.importorskip("torch")

import asset1_analysis_io as aio  # noqa: E402
import asset1_canonicalize as canon  # noqa: E402
import asset1_d2_swap as d2  # noqa: E402
import asset1_synth as synth  # noqa: E402

K = 2   # pairs per cell used throughout these tests


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    """2 families x 3 tasks x 4 reps = 24 runs, task_effect = 1.0.

    n_channels = 2 -> 2x2 bridges (derangements of 4 entries exist);
    trained-like bridges (non-identity) so the permuted baseline is
    well-defined. Family geometries differ (module count and dims), which
    is what makes the family-mixing assembly test meaningful.
    """
    root = tmp_path_factory.mktemp("synth-d2") / "bank"
    info = synth.make_synthetic_bank(
        root, n_families=2, n_tasks=3, n_reps=4, n_layers=1,
        d_model=12, rank=4, n_channels=2, task_effect=1.0, seed=5)
    return root, info


@pytest.fixture(scope="module")
def plan(bank):
    """Assembled plan (SHA-256 filled) over the module-scoped bank."""
    root, _ = bank
    p = d2.generate_plan(root, seed=3, pairs_per_cell=K)
    d2.assemble_plan_states(root, p)
    return p


def _evals(plan_dict, kind=None, family=None):
    out = plan_dict["evals"]
    if kind is not None:
        out = [e for e in out if e["kind"] == kind]
    if family is not None:
        out = [e for e in out if e["family_short"] == family]
    return out


def _adapter_for(root, ev, which="recipient"):
    if which == "recipient":
        task, idx = ev["task_recipient"], ev["recipient_run_index"]
    else:
        task, idx = ev["task_donor"], ev["donor_run_index"]
    return aio.load_adapter(
        root / ev["family_short"] / task / f"run_{idx:03d}")


def _assemble(root, ev):
    return d2.assemble_eval(root, ev, d2._AdapterCache())


# ── Plan generation ─────────────────────────────────────────────────


def test_plan_determinism_same_seed(bank):
    root, _ = bank
    p1 = d2.generate_plan(root, seed=3, pairs_per_cell=K)
    p2 = d2.generate_plan(root, seed=3, pairs_per_cell=K)
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


def test_plan_differs_with_seed(bank):
    root, _ = bank
    p1 = d2.generate_plan(root, seed=3, pairs_per_cell=K)
    p2 = d2.generate_plan(root, seed=4, pairs_per_cell=K)
    assert json.dumps(p1, sort_keys=True) != json.dumps(p2, sort_keys=True)


def test_plan_eval_counts(bank):
    root, info = bank
    p = d2.generate_plan(root, seed=3, pairs_per_cell=K)
    T = len(info["tasks"])
    n_fam = 2
    counts = p["eval_counts"]
    assert counts[d2.KIND_NATIVE] == n_fam * T * K
    assert counts[d2.KIND_IDENTITY] == n_fam * T * K
    assert counts[d2.KIND_PERMUTED] == n_fam * T * K
    assert counts[d2.KIND_PERMUTED_DEV] == n_fam * T * K
    assert counts[d2.KIND_CROSS_SEED] == n_fam * T * K
    assert counts[d2.KIND_CROSS_TASK] == n_fam * T * (T - 1) * K
    assert counts["total"] == len(p["evals"]) == n_fam * (5 * T + T * (T - 1)) * K
    assert d2.KIND_MAGNITUDE not in counts        # decomposition off


def test_plan_carries_h3_reference_signoff_flag(bank):
    """The plan must flag that the H3 structure-destroyed reference is
    unpinned between 'permuted' and 'permuted_deviation' until Director
    sign-off (round-1 review fix — pre-registration hygiene)."""
    root, _ = bank
    p = d2.generate_plan(root, seed=3, pairs_per_cell=K)
    ref = p["h3_structure_reference"]
    assert set(ref["candidates"]) == {d2.KIND_PERMUTED,
                                      d2.KIND_PERMUTED_DEV}
    assert ref["recommended"] == d2.KIND_PERMUTED_DEV
    assert "DIRECTOR SIGN-OFF" in ref["status"].upper()
    assert "identity" in ref["rationale"]


def test_plan_decomposition_counts(bank):
    root, info = bank
    p = d2.generate_plan(root, seed=3, pairs_per_cell=K, decomposition=True)
    T = len(info["tasks"])
    assert p["eval_counts"][d2.KIND_MAGNITUDE] == 2 * T * (T - 1) * K
    assert p["eval_counts"][d2.KIND_TOPOLOGY] == 2 * T * (T - 1) * K
    # Guard cells reuse the exact cross-task pairs
    ct = {(e["family_short"], e["task_recipient"], e["task_donor"],
           e["recipient_run_index"], e["donor_run_index"], e["pair_slot"])
          for e in _evals(p, d2.KIND_CROSS_TASK)}
    mag = {(e["family_short"], e["task_recipient"], e["task_donor"],
            e["recipient_run_index"], e["donor_run_index"], e["pair_slot"])
           for e in _evals(p, d2.KIND_MAGNITUDE)}
    assert ct == mag


def test_plan_unique_eval_ids(plan):
    ids = [e["eval_id"] for e in plan["evals"]]
    assert len(ids) == len(set(ids))


def test_plan_recipients_reused_across_cells(plan, bank):
    _, info = bank
    fam = info["families"][0]["short"]
    for task in info["tasks"]:
        expected = set(plan["recipients_by_family_task"][fam][task])
        for kind in (d2.KIND_NATIVE, d2.KIND_IDENTITY, d2.KIND_PERMUTED,
                     d2.KIND_PERMUTED_DEV, d2.KIND_CROSS_SEED,
                     d2.KIND_CROSS_TASK):
            got = {e["recipient_run_index"] for e in plan["evals"]
                   if e["family_short"] == fam and e["kind"] == kind
                   and e["task_recipient"] == task}
            assert got == expected, (task, kind)


def test_plan_family_filter(bank):
    root, info = bank
    fam = info["families"][1]["short"]
    p = d2.generate_plan(root, seed=3, pairs_per_cell=K, families=[fam])
    assert p["families"] == [fam]
    assert {e["family_short"] for e in p["evals"]} == {fam}
    with pytest.raises(ValueError, match="unknown families"):
        d2.generate_plan(root, seed=3, families=["nope"])


def test_plan_never_mixes_families(plan):
    """Donor and recipient of every eval belong to the same family (single
    family_short field), and cross-seed donors differ from recipients."""
    for e in _evals(plan, d2.KIND_CROSS_SEED):
        assert e["task_donor"] == e["task_recipient"]
        assert e["donor_run_index"] != e["recipient_run_index"]
    for e in _evals(plan, d2.KIND_CROSS_TASK):
        assert e["task_donor"] != e["task_recipient"]


def test_plan_insufficient_runs_raises(tmp_path):
    root = tmp_path / "bank"
    synth.make_synthetic_bank(root, n_families=1, n_tasks=2, n_reps=1,
                              n_layers=1, d_model=8, rank=2, n_channels=2,
                              task_effect=1.0, seed=0)
    with pytest.raises(ValueError, match="pairs_per_cell"):
        d2.generate_plan(root, seed=0, pairs_per_cell=1)   # cross-seed needs 2


def test_plan_records_valid_permutations(plan, bank):
    _, info = bank
    C = 2
    for kind in (d2.KIND_PERMUTED, d2.KIND_PERMUTED_DEV):
        for e in _evals(plan, kind):
            perm = e["permutation"]
            assert sorted(perm) == list(range(C * C))
            assert all(p != i for i, p in enumerate(perm))  # derangement
            assert all(isinstance(p, int) for p in perm)    # json-native
            assert e["perm_seed"] is not None
    for kind in (d2.KIND_NATIVE, d2.KIND_IDENTITY, d2.KIND_CROSS_TASK,
                 d2.KIND_CROSS_SEED):
        assert all(e["permutation"] is None for e in _evals(plan, kind))


def test_permuted_kinds_share_sigma_per_slot(plan):
    """'permuted' and 'permuted_deviation' of the same (family, task,
    slot, recipient) carry the SAME derangement — the design decision
    that makes their pairwise contrast isolate the identity-backbone
    effect (module docstring)."""
    def _key(e):
        return (e["family_short"], e["task_recipient"],
                e["recipient_run_index"], e["pair_slot"])

    full = {_key(e): e["permutation"]
            for e in _evals(plan, d2.KIND_PERMUTED)}
    dev = {_key(e): e["permutation"]
           for e in _evals(plan, d2.KIND_PERMUTED_DEV)}
    assert full.keys() == dev.keys() and len(full) > 0
    for key in full:
        assert full[key] == dev[key], key


# ── Assembly correctness ────────────────────────────────────────────


def test_cross_task_assembly_preserves_recipient_ab(plan, bank):
    root, _ = bank
    ev = _evals(plan, d2.KIND_CROSS_TASK)[0]
    recipient = _adapter_for(root, ev, "recipient")
    donor = _adapter_for(root, ev, "donor")
    state = _assemble(root, ev)
    assert set(state) == set(recipient)
    for name in state:
        assert torch.equal(state[name]["lora_A"], recipient[name]["lora_A"])
        assert torch.equal(state[name]["lora_B"], recipient[name]["lora_B"])
        assert torch.equal(state[name]["bridge"], donor[name]["bridge"])
        assert not torch.equal(state[name]["bridge"],
                               recipient[name]["bridge"])
        for f in ("scaling", "n_channels", "rank"):
            assert torch.equal(state[name][f], recipient[name][f])


def test_native_assembly_is_the_recipient(plan, bank):
    root, _ = bank
    ev = _evals(plan, d2.KIND_NATIVE)[0]
    recipient = _adapter_for(root, ev, "recipient")
    state = _assemble(root, ev)
    assert d2.state_sha256(state) == d2.state_sha256(recipient)


def test_cross_seed_assembly(plan, bank):
    root, _ = bank
    ev = _evals(plan, d2.KIND_CROSS_SEED)[0]
    donor = _adapter_for(root, ev, "donor")
    recipient = _adapter_for(root, ev, "recipient")
    state = _assemble(root, ev)
    for name in state:
        assert torch.equal(state[name]["bridge"], donor[name]["bridge"])
        assert torch.equal(state[name]["lora_A"], recipient[name]["lora_A"])


def test_identity_assembly(plan, bank):
    root, _ = bank
    ev = _evals(plan, d2.KIND_IDENTITY)[0]
    state = _assemble(root, ev)
    eye = torch.eye(2, dtype=torch.float32)
    for name in state:
        assert torch.equal(state[name]["bridge"], eye)


def test_permuted_assembly_properties(plan, bank):
    root, _ = bank
    ev = _evals(plan, d2.KIND_PERMUTED)[0]
    recipient = _adapter_for(root, ev, "recipient")
    state = _assemble(root, ev)
    state2 = _assemble(root, ev)                       # reproducible
    for name in state:
        orig = recipient[name]["bridge"].numpy()
        perm = state[name]["bridge"].numpy()
        assert not np.array_equal(perm, orig)
        assert sorted(perm.reshape(-1).tolist()) == sorted(
            orig.reshape(-1).tolist())                 # same entry multiset
        assert np.array_equal(perm, state2[name]["bridge"].numpy())
        assert torch.equal(state[name]["lora_A"], recipient[name]["lora_A"])
        assert torch.equal(state[name]["lora_B"], recipient[name]["lora_B"])


def test_permuted_deviation_assembly_properties(plan, bank):
    """The corrected reference: bridge == I + perm(B - I) per module —
    identity backbone preserved exactly, trained-deviation entry multiset
    preserved, lora_A/lora_B untouched, reproducible."""
    root, _ = bank
    ev = _evals(plan, d2.KIND_PERMUTED_DEV)[0]
    recipient = _adapter_for(root, ev, "recipient")
    state = _assemble(root, ev)
    state2 = _assemble(root, ev)                       # reproducible
    perm = np.asarray(ev["permutation"], dtype=np.int64)
    for name in state:
        orig = recipient[name]["bridge"].numpy()
        new = state[name]["bridge"].numpy()
        eye = np.eye(orig.shape[0], dtype=orig.dtype)
        assert not np.array_equal(new, orig)
        # exact reconstruction: I + permuted deviation (float32 storage
        # of I + d then re-subtraction of I costs the low bits of
        # diagonal-resident entries — tolerance, not bit-equality)
        expected = eye + (orig - eye).reshape(-1)[perm].reshape(orig.shape)
        np.testing.assert_allclose(new, expected, atol=1e-6)
        # trained-deviation entry multiset preserved (NOT the raw entries)
        np.testing.assert_allclose(
            np.sort((new - eye).reshape(-1)),
            np.sort((orig - eye).reshape(-1)), atol=1e-6)
        assert np.array_equal(new, state2[name]["bridge"].numpy())
        assert torch.equal(state[name]["lora_A"], recipient[name]["lora_A"])
        assert torch.equal(state[name]["lora_B"], recipient[name]["lora_B"])


def test_family_mixing_raises(bank):
    root, info = bank
    fam0, fam1 = (f["short"] for f in info["families"])
    task = info["tasks"][0]
    rec0 = next(iter(aio.iter_runs(root, family=fam0, task=task)))
    rec1 = next(iter(aio.iter_runs(root, family=fam1, task=task)))
    a0 = aio.load_adapter(rec0["run_dir"])
    a1 = aio.load_adapter(rec1["run_dir"])
    with pytest.raises(ValueError, match="family mixing"):
        d2.assemble_swapped_state(a0, a1, d2.KIND_CROSS_TASK)


def test_module_name_mismatch_raises(bank):
    root, info = bank
    fam = info["families"][0]["short"]
    task = info["tasks"][0]
    recs = list(aio.iter_runs(root, family=fam, task=task))
    a = aio.load_adapter(recs[0]["run_dir"])
    b = aio.load_adapter(recs[1]["run_dir"])
    name = sorted(b)[0]
    b["renamed_module"] = b.pop(name)
    with pytest.raises(ValueError, match="module-name sets differ"):
        d2.assemble_swapped_state(a, b, d2.KIND_CROSS_TASK)


def test_unknown_kind_and_missing_permutation_raise(bank):
    root, info = bank
    fam = info["families"][0]["short"]
    rec = next(iter(aio.iter_runs(root, family=fam)))
    a = aio.load_adapter(rec["run_dir"])
    with pytest.raises(ValueError, match="unknown swap kind"):
        d2.assemble_swapped_state(a, a, "bogus")
    with pytest.raises(ValueError, match="requires a permutation"):
        d2.assemble_swapped_state(a, a, d2.KIND_PERMUTED)
    with pytest.raises(ValueError, match="requires a permutation"):
        d2.assemble_swapped_state(a, a, d2.KIND_PERMUTED_DEV)


def test_sha_stable_across_regeneration(bank, plan):
    root, _ = bank
    p2 = d2.generate_plan(root, seed=3, pairs_per_cell=K)
    d2.assemble_plan_states(root, p2)
    sha1 = {e["eval_id"]: e["assembled_sha256"] for e in plan["evals"]}
    sha2 = {e["eval_id"]: e["assembled_sha256"] for e in p2["evals"]}
    assert sha1 == sha2
    assert all(v is not None for v in sha1.values())


def test_written_state_roundtrips_through_bank_loader(bank, tmp_path):
    """--write-states output uses the bank's flat key format and reloads
    through the production loader (asset1_canonicalize)."""
    root, _ = bank
    p = d2.generate_plan(root, seed=3, pairs_per_cell=K,
                         families=["synthfam0"])
    p["evals"] = p["evals"][:2]
    d2.assemble_plan_states(root, p, write_states_dir=tmp_path / "states")
    for ev in p["evals"]:
        path = tmp_path / "states" / f"{ev['eval_id']}.pt"
        assert path.exists()
        assert ev["state_file"] == f"states/{path.name}"
        reloaded = canon.load_adapter_modules(path)
        rebuilt = _assemble(root, ev)
        assert d2.state_sha256(reloaded) == ev["assembled_sha256"]
        assert d2.state_sha256(rebuilt) == ev["assembled_sha256"]


# ── Permutation helpers ─────────────────────────────────────────────


def test_derangement_properties():
    rng = np.random.default_rng(0)
    perm = d2.draw_derangement(rng, 36)
    assert sorted(perm.tolist()) == list(range(36))
    assert not np.any(perm == np.arange(36))
    p1 = d2.draw_derangement(np.random.default_rng([1, 2, 3]), 36)
    p2 = d2.draw_derangement(np.random.default_rng([1, 2, 3]), 36)
    assert np.array_equal(p1, p2)
    with pytest.raises(ValueError):
        d2.draw_derangement(rng, 1)


def test_permute_bridge_mapping():
    b = np.arange(4.0).reshape(2, 2)          # entries 0,1,2,3
    perm = [1, 0, 3, 2]                        # a derangement of 4
    out = d2.permute_bridge(b, perm)
    assert out.tolist() == [[1.0, 0.0], [3.0, 2.0]]   # new[i] = old[perm[i]]
    with pytest.raises(ValueError, match="length"):
        d2.permute_bridge(b, [1, 0, 2])
    with pytest.raises(ValueError, match="not a permutation"):
        d2.permute_bridge(b, [1, 1, 3, 2])


def test_permute_bridge_degenerate_raises():
    const = np.full((2, 2), 7.0)               # entry-constant matrix
    with pytest.raises(ValueError, match="degenerate"):
        d2.permute_bridge(const, [1, 0, 3, 2])


def test_permute_bridge_deviation_mapping():
    """new = I + perm(B - I), exact; validation mirrors permute_bridge."""
    b = np.eye(2) + np.array([[0.0, 0.1], [0.2, 0.3]])
    perm = [1, 0, 3, 2]                        # derangement of 4
    out = d2.permute_bridge_deviation(b, perm)
    # deviation entries [0.0, 0.1, 0.2, 0.3] -> [0.1, 0.0, 0.3, 0.2]
    expected = np.eye(2) + np.array([[0.1, 0.0], [0.3, 0.2]])
    np.testing.assert_allclose(out, expected, atol=0)
    with pytest.raises(ValueError, match="length"):
        d2.permute_bridge_deviation(b, [1, 0, 2])
    with pytest.raises(ValueError, match="not a permutation"):
        d2.permute_bridge_deviation(b, [1, 1, 3, 2])
    with pytest.raises(ValueError, match="square"):
        d2.permute_bridge_deviation(np.zeros((2, 3)), [1, 0, 3, 2, 5, 4])
    # exactly-identity bridge: D = 0 -> result equals original -> raise
    with pytest.raises(ValueError, match="degenerate"):
        d2.permute_bridge_deviation(np.eye(2), [1, 0, 3, 2])


def test_permuted_deviation_isolates_trained_structure():
    """THE ROUND-1 REGRESSION TEST: for a trained-like bridge
    B = I + eps * noise, the full-entry permutation's distance from
    native is dominated by scattering the identity backbone (O(1),
    independent of eps), while the deviation permutation's distance is
    bounded by 2 * ||D||_F (O(eps)) — i.e. the old 'permuted' cell
    destroys far more than the trained signal it claims to isolate."""
    rng = np.random.default_rng(0)
    C, eps = 6, 0.01
    D = eps * rng.standard_normal((C, C))
    B = np.eye(C) + D
    perm = d2.draw_derangement(np.random.default_rng([0, 44, 0, 0, 0]),
                               C * C)
    d_full = np.linalg.norm(d2.permute_bridge(B, perm) - B)
    d_dev = np.linalg.norm(d2.permute_bridge_deviation(B, perm) - B)
    dev_norm = np.linalg.norm(D)
    # corrected reference: bounded by the trained-signal magnitude
    assert d_dev <= 2.0 * dev_norm + 1e-12
    # legacy reference: dominated by the identity backbone, NOT the
    # trained deviation — an order of magnitude beyond the signal scale
    assert d_full > 10.0 * dev_norm
    # and the backbone effect does not vanish as eps -> 0
    B_tiny = np.eye(C) + 1e-6 * rng.standard_normal((C, C))
    assert np.linalg.norm(d2.permute_bridge(B_tiny, perm) - B_tiny) > 1.0


# ── Magnitude / topology decomposition ──────────────────────────────


def test_decompose_recompose_roundtrip():
    rng = np.random.default_rng(42)
    for c in (2, 6):
        B = np.eye(c) + 0.3 * rng.standard_normal((c, c))
        r, cvec, P = d2.bridge_decompose(B)
        back = d2.bridge_recompose(r, cvec, P)
        np.testing.assert_allclose(back, B, atol=1e-12)


def test_decompose_identity_bridge():
    B = np.eye(6)
    r, c, P = d2.bridge_decompose(B)
    assert np.all(r == 0) and np.all(c == 0) and np.all(P == 0)
    np.testing.assert_allclose(d2.bridge_recompose(r, c, P), B, atol=0)


def test_transplant_self_is_identity_op():
    rng = np.random.default_rng(7)
    B = np.eye(6) + 0.5 * rng.standard_normal((6, 6))
    np.testing.assert_allclose(d2.transplant_magnitude(B, B), B, atol=1e-12)
    np.testing.assert_allclose(d2.transplant_topology(B, B), B, atol=1e-12)


def test_transplant_matches_manual_formula():
    rng = np.random.default_rng(9)
    R = np.eye(4) + 0.4 * rng.standard_normal((4, 4))
    D = np.eye(4) + 0.4 * rng.standard_normal((4, 4))
    dR, dD = R - np.eye(4), D - np.eye(4)
    rR = np.linalg.norm(dR, axis=1)
    cR = np.linalg.norm(dR, axis=0)
    rD = np.linalg.norm(dD, axis=1)
    cD = np.linalg.norm(dD, axis=0)
    P_D = dD / (np.sqrt(rD)[:, None] * np.sqrt(cD)[None, :])
    expected_topo = np.eye(4) + np.sqrt(rR)[:, None] * P_D * np.sqrt(cR)[None, :]
    np.testing.assert_allclose(d2.transplant_topology(R, D), expected_topo,
                               atol=1e-12)
    P_R = dR / (np.sqrt(rR)[:, None] * np.sqrt(cR)[None, :])
    expected_mag = np.eye(4) + np.sqrt(rD)[:, None] * P_R * np.sqrt(cD)[None, :]
    np.testing.assert_allclose(d2.transplant_magnitude(R, D), expected_mag,
                               atol=1e-12)


def test_transplants_differ_from_plain_swap():
    rng = np.random.default_rng(11)
    R = np.eye(4) + 0.4 * rng.standard_normal((4, 4))
    D = np.eye(4) + 0.4 * rng.standard_normal((4, 4))
    mag = d2.transplant_magnitude(R, D)
    topo = d2.transplant_topology(R, D)
    for out in (mag, topo):
        assert not np.allclose(out, R)
        assert not np.allclose(out, D)
    assert not np.allclose(mag, topo)


def test_decompose_rejects_nonsquare():
    with pytest.raises(ValueError):
        d2.bridge_decompose(np.zeros((2, 3)))


# ── Penalty computation ─────────────────────────────────────────────


def _row(kind, task, r_idx, val, **kw):
    return {"eval_id": f"{kind}-{task}-{r_idx}", "kind": kind,
            "task_recipient": task, "task_donor": kw.get("task_donor"),
            "recipient_run_index": r_idx,
            "donor_run_index": kw.get("donor_run_index"),
            "pair_slot": 0, "val_loss": val}


def test_compute_penalties():
    rows = [
        _row(d2.KIND_NATIVE, "a", 0, 1.0),
        _row(d2.KIND_CROSS_TASK, "a", 0, 1.5, task_donor="b",
             donor_run_index=7),
        _row(d2.KIND_IDENTITY, "a", 0, 0.9),
    ]
    pens = d2.compute_penalties(rows)
    by_kind = {p["kind"]: p for p in pens}
    assert by_kind[d2.KIND_CROSS_TASK]["penalty"] == pytest.approx(0.5)
    assert by_kind[d2.KIND_IDENTITY]["penalty"] == pytest.approx(-0.1)
    assert by_kind[d2.KIND_CROSS_TASK]["native_val_loss"] == 1.0


def test_compute_penalties_missing_native_raises():
    rows = [_row(d2.KIND_CROSS_TASK, "a", 3, 1.5, task_donor="b",
                 donor_run_index=7)]
    with pytest.raises(ValueError, match="no native val_loss"):
        d2.compute_penalties(rows)


# ── CLI: interlock, guards, plan output ─────────────────────────────


def test_cli_interlock_refuses_incomplete_total(bank, tmp_path):
    root, _ = bank   # 24 runs, all COMPLETE — but expected_total is 480
    with pytest.raises(SystemExit, match="REFUSING"):
        d2.main(["--bank-root", str(root), "--out-dir", str(tmp_path / "o"),
                 "--plan-only"])


def test_cli_plan_only_with_allow_partial(bank, tmp_path, capsys):
    root, _ = bank
    out = tmp_path / "out"
    d2.main(["--bank-root", str(root), "--out-dir", str(out),
             "--plan-only", "--allow-partial-bank", "--seed", "3",
             "--pairs-per-cell", str(K)])
    captured = capsys.readouterr()
    assert "PRE-REGISTRATION WARNING" in captured.err
    assert "total" in captured.out
    plan = json.loads((out / d2.PLAN_FILENAME).read_text(encoding="utf-8"))
    assert plan["eval_counts"]["total"] == len(plan["evals"])
    # plan-only must not have touched adapters
    assert all(e["assembled_sha256"] is None for e in plan["evals"])


def test_cli_full_assembly_fills_shas(bank, tmp_path):
    root, _ = bank
    out = tmp_path / "out"
    d2.main(["--bank-root", str(root), "--out-dir", str(out),
             "--allow-partial-bank", "--seed", "3",
             "--pairs-per-cell", str(K), "--family", "synthfam0"])
    plan = json.loads((out / d2.PLAN_FILENAME).read_text(encoding="utf-8"))
    assert plan["families"] == ["synthfam0"]
    assert all(e["assembled_sha256"] is not None for e in plan["evals"])


def test_cli_out_dir_guard_bank_component(bank, tmp_path):
    root, _ = bank
    with pytest.raises(SystemExit, match="asset1-bank"):
        d2.main(["--bank-root", str(root),
                 "--out-dir", str(tmp_path / "asset1-bank" / "d2"),
                 "--plan-only", "--allow-partial-bank"])


def test_cli_out_dir_guard_inside_bank_root(bank):
    root, _ = bank
    with pytest.raises(SystemExit, match="inside"):
        d2.main(["--bank-root", str(root),
                 "--out-dir", str(root / "d2-out"),
                 "--plan-only", "--allow-partial-bank"])


# ── Stage B gates + import safety ───────────────────────────────────


class _BlockHeavyImports(importlib.abc.MetaPathFinder):
    """Simulate an environment without transformers/datasets installed."""

    BLOCKED = ("transformers", "datasets")

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(f"{fullname} blocked by test")
        return None


@pytest.fixture()
def no_transformers():
    saved = {name: sys.modules.pop(name) for name in list(sys.modules)
             if name.split(".")[0] in _BlockHeavyImports.BLOCKED}
    blocker = _BlockHeavyImports()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_stage_a_import_safe_without_transformers(bank, tmp_path,
                                                  no_transformers, capsys):
    """The plan path never imports transformers — Stage A works in an
    environment where the import would fail."""
    root, _ = bank
    out = tmp_path / "out"
    d2.main(["--bank-root", str(root), "--out-dir", str(out),
             "--plan-only", "--allow-partial-bank",
             "--pairs-per-cell", str(K)])
    assert (out / d2.PLAN_FILENAME).exists()
    with pytest.raises(ImportError):        # the blocker really blocks
        import transformers  # noqa: F401


def test_evaluate_refused_without_gate_flag(bank, tmp_path, no_transformers):
    """--evaluate without --i-have-gpu-and-bank-is-complete exits at the
    gate BEFORE any lazy transformers import (no ImportError leaks)."""
    root, _ = bank
    with pytest.raises(SystemExit, match="i-have-gpu-and-bank-is-complete"):
        d2.main(["--bank-root", str(root), "--out-dir", str(tmp_path / "o"),
                 "--evaluate", "--family", "synthfam0",
                 "--allow-partial-bank"])


def test_evaluate_requires_family(bank, tmp_path):
    root, _ = bank
    with pytest.raises(SystemExit, match="--family"):
        d2.main(["--bank-root", str(root), "--out-dir", str(tmp_path / "o"),
                 "--evaluate", "--i-have-gpu-and-bank-is-complete",
                 "--allow-partial-bank"])


def test_evaluate_requires_existing_plan(bank, tmp_path):
    root, _ = bank
    with pytest.raises(SystemExit, match="Stage A first"):
        d2.main(["--bank-root", str(root), "--out-dir", str(tmp_path / "o"),
                 "--evaluate", "--family", "synthfam0",
                 "--i-have-gpu-and-bank-is-complete",
                 "--allow-partial-bank"])


def test_module_import_has_no_heavy_deps():
    """asset1_d2_swap must be importable with transformers absent — its
    module-level namespace holds only stdlib + numpy/torch + the analysis
    IO layer (transformers/datasets/asset1_bank are Stage-B lazy imports)."""
    assert "transformers" not in d2.__dict__
    assert "datasets" not in d2.__dict__
    assert "asset1_bank" not in d2.__dict__
    assert "bank_mod" not in d2.__dict__
