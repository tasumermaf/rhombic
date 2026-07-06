"""Asset 1 — D2 cross-task bridge-swap matrix (locked experiment card, H3).

H3 is directional with opposite predictions. Substrate hypothesis: swapping
a task-j-trained bridge onto task-i-trained A/B costs about the cross-seed
baseline. Task-specific hypothesis: the cross-task penalty is large and
scales with the D1 between-task distance. This tool builds the full
(task_i A/B x task_j bridge) swap matrix per family, with every baseline
IN THE SAME MATRIX, evaluated on task_i's fixed 500-example validation
split. Penalty = swapped_val_loss - native_val_loss per cell.

Two cleanly separated stages
----------------------------
STAGE A (CPU, pre-bank buildable): deterministic swap-plan generation and
adapter assembly. Reads adapters only through asset1_analysis_io (manifest-
driven, COMPLETE runs only), writes ONLY under --out-dir (never under the
bank), and is gated by require_complete_bank() like every analysis CLI.

STAGE B (GPU, post-bank ONLY): evaluation harness. Loads the base model
once per family, installs each assembled adapter state, and computes
val_loss on the recipient task's fixed val split, reusing the bank's own
conventions (asset1_datasets.build_dataset with the locked val_seed=777
design — the val split is invariant to data_seed by construction — and
asset1_bank.evaluate_val_loss with EVAL_BATCH_SIZE / MAX_LEN). Every
GPU-touching function carries an explicit docstring gate; transformers /
datasets are imported LAZILY inside Stage B functions only, so importing
this module never touches HF. The CLI refuses --evaluate without the
explicit --i-have-gpu-and-bank-is-complete flag. One command per family:

    python scripts/asset1_d2_swap.py --bank-root results/asset1-bank \
        --out-dir results/asset1-d2 --family qwen2.5-1.5b \
        --evaluate --i-have-gpu-and-bank-is-complete

Matrix design (kinds)
---------------------
For each family, tasks T (real bank: 6) and K = --pairs-per-cell run pairs
per cell (default 3):

    native      : recipient's own unmodified adapter — the penalty reference.
    cross_task  : recipient (task i) lora_A/lora_B + donor (task j != i)
                  bridge. The H3 cell.
    cross_seed  : donor bridge from a DIFFERENT RUN of the SAME task —
                  the substrate reference (pilot penalty ~ -1.9%).
    permuted    : the recipient's OWN bridge with a seeded fixed-point-free
                  permutation of ALL its C x C entries. The FULL-ENTRY
                  permutation preserves the bridge's entry multiset but —
                  because trained bridges are I + small deviation
                  (bridge_mode='identity' init) — it scatters the ~1.0
                  identity diagonal off-diagonal, so its penalty is
                  dominated by destroying the UNTRAINED identity backbone,
                  not the trained arrangement (round-1 review finding).
                  Retained for comparison; see 'permuted_deviation'.
    permuted_deviation : the corrected structure-destroyed reference
                  (round-1 review fix). Permute only the DEVIATION
                  D = B - I and reconstruct I + permute(D): the identity
                  backbone is preserved exactly and the TRAINED-deviation
                  entry multiset is preserved, so this cell isolates
                  destruction of the trained arrangement. Shares the SAME
                  derangement sigma as the 'permuted' cell of its
                  (family, task, slot), so the pairwise contrast between
                  the two cells isolates exactly the identity-backbone
                  effect. WHICH of the two cells is the pre-registered H3
                  structure-destroyed reference REQUIRES DIRECTOR SIGN-OFF
                  BEFORE STAGE B ('permuted_deviation' recommended); the
                  plan JSON carries this flag ('h3_structure_reference').
                  (Design decision: permuting the recipient's own bridge
                  rather than a cross-task donor's keeps this a per-task
                  row control; flagged to the Director.)
    identity    : bridge = I. bridge_mode is 'identity' at init, so trained
                  deviation-from-identity IS the signal being swapped;
                  this cell measures the total contribution of bridge
                  training to the native loss.
    cross_task_magnitude / cross_task_topology (--decomposition only):
                  the contradiction-guard cells — transplant ONLY the
                  donor's magnitude profile or ONLY its normalized pattern
                  (see decomposition below), on the same (recipient, donor)
                  pairs as the cross_task cells.

Recipient reuse (design decision): for each task i the SAME K recipient
runs are used for every donor task j and every baseline kind. This
amortizes native evaluations (K per task, not K per cell) and makes all
cells of a matrix row directly comparable on identical recipients.

Evaluation count per family: native/identity/permuted/permuted_deviation/
cross_seed = 5*T*K, cross_task = T*(T-1)*K, decomposition adds
2*T*(T-1)*K. Real bank (T=6, K=3): 180 evals per family (360 with
--decomposition); both families: 360 (720). The plan prints these totals.

Permuted-bridge definitions (exact)
-----------------------------------
One permutation sigma of the C^2 entry positions per (family, task, slot),
drawn from numpy default_rng([seed, 44, family_idx, task_idx, slot]) by
rejection sampling: permutations are drawn until sigma has NO fixed point
(a derangement — "excluding any trivial fixed points"; acceptance rate
~ 1/e). The SAME sigma is applied to every module's bridge in the
assembled adapter, and the SAME sigma is shared by the 'permuted' and
'permuted_deviation' cells of a slot (deliberate: their pairwise contrast
then isolates the identity-backbone effect for that exact rearrangement):

    permuted           : new_flat[i] = old_flat[sigma[i]], reshaped C x C.
    permuted_deviation : D = B - I; new = I + reshape(D_flat[sigma]).

The permutation is recorded in the plan (data-independent, so --plan-only
and full assembly record identical plans). Degenerate case: if the
transformed matrix equals the original (possible only for degenerate
entry multisets — e.g. an exactly-identity bridge, where D = 0; impossible
for generically valued trained bridges), assembly raises rather than
silently producing a native duplicate.

Magnitude / topology decomposition (contradiction guard)
--------------------------------------------------------
Let B be a C x C bridge and D = B - I its deviation from identity (the
trained signal). Define

    r_i = ||D[i, :]||_2   (row norms),   c_j = ||D[:, j]||_2  (col norms)
    P   = diag(r)^{-1/2} @ D @ diag(c)^{-1/2}     (0 where r_i or c_j = 0)

The SCALE component is (r, c) — the per-row/per-col deviation norms. The
PATTERN component is P — the two-sided-normalized structure of the
deviation. The factorization is exact by construction:

    D = diag(r)^{1/2} @ P @ diag(c)^{1/2},   so  B = I + sqrt(r) P sqrt(c)^T
    (elementwise: D_ij = sqrt(r_i) * P_ij * sqrt(c_j))

and rows/cols with zero deviation round-trip exactly (P entries are 0
there). Transplants used by the guard cells:

    transplant_magnitude(recipient R, donor D):
        I + diag(r_D)^{1/2} @ P_R @ diag(c_D)^{1/2}
        — donor's scale profile carried on the recipient's pattern.
    transplant_topology(recipient R, donor D):
        I + diag(r_R)^{1/2} @ P_D @ diag(c_R)^{1/2}
        — donor's pattern carried at the recipient's scale profile.

Note: the transplant is a DEFINED operation, exact in the round-trip sense
(transplant with donor == recipient reproduces the original bridge to
float precision) but not norm-preserving in general — the row/col norms of
the transplanted deviation are not exactly the donor's (P rows are not
unit vectors under the two-sided normalization). This is documented, not
hidden; the guard compares penalties, not norms.

Determinism
-----------
Every stochastic choice derives from numpy default_rng seeded with integer
lists [seed, stream_tag, indices] (stream tags: recipients 11, cross-task
donors 22, cross-seed donors 33, permutations 44; family/task indices are
positions in the manifest campaign lists). No wall clock, no entropy. The
plan JSON is fully deterministic in (bank contents, seed, flags); assembled
state SHA-256 digests are recorded per eval for post-bank integrity checks
(Stage B re-assembles and verifies before evaluating).

Safety
------
* THE INTERLOCK: main() calls require_complete_bank() before anything
  else touches the bank; --allow-partial-bank maps to allow_partial=True
  and prints the loud PRE-REGISTRATION WARNING (tooling checks only).
* Never writes under the bank: --out-dir is refused if it contains an
  'asset1-bank' path component or lies inside --bank-root.
* CPU-only in Stage A; all loads are map_location='cpu' via
  asset1_analysis_io / asset1_canonicalize.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asset1_analysis_io import (  # noqa: E402
    iter_runs,
    load_adapter,
    load_manifest,
    require_complete_bank,
)

# ── Locked constants ────────────────────────────────────────────────

PLAN_FILENAME = "d2_swap_plan.json"
STATES_DIRNAME = "states"
DEFAULT_PAIRS_PER_CELL = 3     # K — see module docstring for eval counts

KIND_NATIVE = "native"
KIND_CROSS_TASK = "cross_task"
KIND_CROSS_SEED = "cross_seed"
KIND_PERMUTED = "permuted"
KIND_PERMUTED_DEV = "permuted_deviation"
KIND_IDENTITY = "identity"
KIND_MAGNITUDE = "cross_task_magnitude"
KIND_TOPOLOGY = "cross_task_topology"

ALL_KINDS = frozenset({
    KIND_NATIVE, KIND_CROSS_TASK, KIND_CROSS_SEED, KIND_PERMUTED,
    KIND_PERMUTED_DEV, KIND_IDENTITY, KIND_MAGNITUDE, KIND_TOPOLOGY,
})
# Kinds that require a recorded entry-position permutation.
_PERMUTATION_KINDS = frozenset({KIND_PERMUTED, KIND_PERMUTED_DEV})
# Kinds whose donor adapter is a DIFFERENT run than the recipient.
_FOREIGN_DONOR_KINDS = frozenset({
    KIND_CROSS_TASK, KIND_CROSS_SEED, KIND_MAGNITUDE, KIND_TOPOLOGY,
})

# rng stream tags (documented in the module docstring)
_STREAM_RECIPIENTS = 11
_STREAM_CT_DONORS = 22
_STREAM_CS_DONORS = 33
_STREAM_PERMUTATION = 44

_MAX_DERANGEMENT_ATTEMPTS = 10_000
_STATE_FIELD_ORDER = ("lora_A", "lora_B", "bridge",
                      "scaling", "n_channels", "rank")
_ADAPTER_CACHE_SIZE = 8        # LRU cap: ~8 x 26 MB on the real bank


# ── Permutation (structure-destroyed baseline) ──────────────────────


def draw_derangement(rng: np.random.Generator, n: int) -> np.ndarray:
    """First fixed-point-free permutation of range(n) from ``rng``.

    Rejection sampling (acceptance ~ 1/e); deterministic given the rng
    state. Raises after _MAX_DERANGEMENT_ATTEMPTS (cannot trigger for
    n >= 2 in practice).
    """
    if n < 2:
        raise ValueError(f"no derangement exists for n={n}")
    idx = np.arange(n)
    for _ in range(_MAX_DERANGEMENT_ATTEMPTS):
        perm = rng.permutation(n)
        if not np.any(perm == idx):
            return perm
    raise RuntimeError(
        f"failed to draw a derangement of {n} elements in "
        f"{_MAX_DERANGEMENT_ATTEMPTS} attempts — rng stream defect?")


def permute_bridge(bridge: np.ndarray, permutation) -> np.ndarray:
    """Apply an entry-position permutation: new_flat[i] = old_flat[perm[i]].

    Raises if the permutation length does not match the entry count or if
    the permuted matrix equals the original (degenerate entry multiset,
    e.g. an exactly-identity bridge — see module docstring).
    """
    b = np.asarray(bridge)
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.shape != (b.size,):
        raise ValueError(
            f"permutation length {perm.shape} does not match bridge entry "
            f"count {b.size} (bridge shape {b.shape})")
    if sorted(perm.tolist()) != list(range(b.size)):
        raise ValueError("permutation is not a permutation of the entry "
                         "positions")
    out = b.reshape(-1)[perm].reshape(b.shape)
    if np.array_equal(out, b):
        raise ValueError(
            "permuted bridge equals the original — degenerate entry "
            "multiset (e.g. an exactly-identity bridge); the permuted "
            "baseline is undefined for this adapter")
    return out


def permute_bridge_deviation(bridge: np.ndarray, permutation) -> np.ndarray:
    """Permute ONLY the deviation from identity: I + permute(B - I).

    The corrected structure-destroyed reference (round-1 review fix):
    trained bridges are I + small deviation, so permuting ALL entries
    (permute_bridge) scatters the untrained ~1.0 identity diagonal
    off-diagonal and its penalty conflates init-structure destruction
    with trained-structure destruction. This variant preserves the
    identity backbone exactly and preserves the TRAINED-deviation entry
    multiset, isolating destruction of the trained arrangement.

    Same permutation convention as permute_bridge
    (new_flat[i] = old_flat[perm[i]], applied to D = B - I). Raises if
    the bridge is not square, if the permutation is invalid, or if the
    result equals the original bridge (degenerate deviation multiset,
    e.g. an exactly-identity bridge where D = 0).
    """
    b = np.asarray(bridge)
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError(f"bridge must be square 2-D, got shape {b.shape}")
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.shape != (b.size,):
        raise ValueError(
            f"permutation length {perm.shape} does not match bridge entry "
            f"count {b.size} (bridge shape {b.shape})")
    if sorted(perm.tolist()) != list(range(b.size)):
        raise ValueError("permutation is not a permutation of the entry "
                         "positions")
    eye = np.eye(b.shape[0], dtype=b.dtype)
    delta = b - eye
    out = eye + delta.reshape(-1)[perm].reshape(b.shape)
    if np.array_equal(out, b):
        raise ValueError(
            "deviation-permuted bridge equals the original — degenerate "
            "deviation multiset (e.g. an exactly-identity bridge); the "
            "permuted_deviation baseline is undefined for this adapter")
    return out


# ── Magnitude / topology decomposition (contradiction guard) ────────


def bridge_decompose(bridge) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact scale/pattern factorization of the deviation-from-identity.

    Returns (r, c, P) with r/c the row/col L2 norms of D = B - I and
    P = diag(r)^{-1/2} @ D @ diag(c)^{-1/2} (0 where r_i or c_j is 0), so
    that D = diag(r)^{1/2} @ P @ diag(c)^{1/2} exactly. float64.
    """
    B = np.asarray(bridge, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"bridge must be square 2-D, got shape {B.shape}")
    delta = B - np.eye(B.shape[0])
    r = np.linalg.norm(delta, axis=1)
    c = np.linalg.norm(delta, axis=0)
    inv_rs = np.divide(1.0, np.sqrt(r), out=np.zeros_like(r), where=r > 0)
    inv_cs = np.divide(1.0, np.sqrt(c), out=np.zeros_like(c), where=c > 0)
    P = inv_rs[:, None] * delta * inv_cs[None, :]
    return r, c, P


def bridge_recompose(r: np.ndarray, c: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Inverse of bridge_decompose: B = I + diag(r)^{1/2} P diag(c)^{1/2}."""
    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    delta = np.sqrt(r)[:, None] * P * np.sqrt(c)[None, :]
    return np.eye(P.shape[0]) + delta


def transplant_magnitude(recipient_bridge, donor_bridge) -> np.ndarray:
    """Donor's scale profile (row/col deviation norms) on the recipient's
    pattern: I + diag(r_D)^{1/2} @ P_R @ diag(c_D)^{1/2}. float64."""
    r_d, c_d, _ = bridge_decompose(donor_bridge)
    _, _, p_r = bridge_decompose(recipient_bridge)
    return bridge_recompose(r_d, c_d, p_r)


def transplant_topology(recipient_bridge, donor_bridge) -> np.ndarray:
    """Donor's pattern at the recipient's scale profile:
    I + diag(r_R)^{1/2} @ P_D @ diag(c_R)^{1/2}. float64."""
    r_r, c_r, _ = bridge_decompose(recipient_bridge)
    _, _, p_d = bridge_decompose(donor_bridge)
    return bridge_recompose(r_r, c_r, p_d)


# ── Plan generation (Stage A, CPU) ──────────────────────────────────


def _runs_by_task(bank_root: Path, family_short: str) -> dict[str, list[dict]]:
    """COMPLETE run records of one family grouped by task (manifest order)."""
    groups: dict[str, list[dict]] = {}
    for rec in iter_runs(bank_root, family=family_short):
        groups.setdefault(rec["task"], []).append(rec)
    return groups


def _campaign_lists(manifest: dict, bank_root: Path
                    ) -> tuple[list[dict], list[str]]:
    """(families [{model, short}], tasks) from the manifest campaign block,
    with a defensive fallback derived from the run entries."""
    camp = manifest.get("campaign", {})
    families = camp.get("families")
    tasks = camp.get("tasks")
    if not families:
        seen: dict[str, str] = {}
        for e in manifest["runs"]:
            seen.setdefault(e["family_short"], e["family"])
        families = [{"model": m, "short": s} for s, m in sorted(seen.items())]
    if not tasks:
        tasks = sorted({e["task"] for e in manifest["runs"]})
    return families, list(tasks)


def _eval_entry(family_short: str, kind: str, task_i: str, slot: int,
                recipient_idx: int, task_j: str | None = None,
                donor_idx: int | None = None,
                permutation: list[int] | None = None,
                perm_seed: list[int] | None = None) -> dict:
    eval_id = f"{family_short}__{kind}__{task_i}__r{recipient_idx:03d}"
    if donor_idx is not None:
        eval_id += f"__from_{task_j}__d{donor_idx:03d}"
    eval_id += f"__k{slot}"
    return {
        "eval_id": eval_id,
        "kind": kind,
        "family_short": family_short,
        "task_recipient": task_i,      # val split evaluated on
        "task_donor": task_j,
        "recipient_run_index": int(recipient_idx),
        "donor_run_index": None if donor_idx is None else int(donor_idx),
        "pair_slot": int(slot),
        "perm_seed": perm_seed,
        "permutation": permutation,
        "assembled_sha256": None,      # filled by assemble_plan_states
        "state_file": None,            # filled when --write-states
    }


def generate_plan(bank_root: str | Path, *, seed: int = 0,
                  pairs_per_cell: int = DEFAULT_PAIRS_PER_CELL,
                  families: list[str] | None = None,
                  decomposition: bool = False) -> dict:
    """Build the deterministic D2 swap plan (no adapter files are read).

    Enumeration is manifest-driven (COMPLETE runs only). All pairing
    randomness derives from ``seed`` through the documented rng streams;
    two invocations with identical bank contents and arguments produce
    byte-identical plans. Run-dir information is NOT stored — Stage A/B
    re-derive run dirs from the locked <family_short>/<task>/run_<idx:03d>
    layout.

    ``families``: family_short names to include (None = all campaign
    families). Every cell requires at least ``pairs_per_cell`` COMPLETE
    runs (and >= 2 for the cross-seed baseline); a clear error names the
    deficient cell otherwise.
    """
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if pairs_per_cell < 1:
        raise ValueError("pairs_per_cell must be >= 1")
    K = int(pairs_per_cell)
    bank_root = Path(bank_root)
    manifest = load_manifest(bank_root)
    fam_entries, tasks = _campaign_lists(manifest, bank_root)
    fam_shorts_all = [f["short"] for f in fam_entries]

    if families is None:
        selected = list(fam_shorts_all)
    else:
        unknown = [f for f in families if f not in fam_shorts_all]
        if unknown:
            raise ValueError(
                f"unknown families {unknown}; campaign families: "
                f"{fam_shorts_all}")
        selected = list(families)

    evals: list[dict] = []
    recipients_by_family_task: dict[str, dict[str, list[int]]] = {}
    n_channels_by_family: dict[str, int] = {}

    for family_short in selected:
        fam_idx = fam_shorts_all.index(family_short)
        groups = _runs_by_task(bank_root, family_short)
        missing = [t for t in tasks if t not in groups]
        if missing:
            raise ValueError(
                f"family {family_short!r} has no COMPLETE runs for tasks "
                f"{missing} — cannot build the swap matrix (partial bank?)")

        run_indices = {t: [r["run_index"] for r in groups[t]] for t in tasks}
        for t in tasks:
            n_avail = len(run_indices[t])
            if n_avail < K or n_avail < 2:
                raise ValueError(
                    f"cell ({family_short}, {t}) has only {n_avail} COMPLETE "
                    f"run(s); need >= max(pairs_per_cell={K}, 2). Lower "
                    f"--pairs-per-cell or wait for more COMPLETE runs.")

        cfg = groups[tasks[0]][0]["config"]
        if cfg is None or "n_channels" not in cfg:
            raise ValueError(
                f"config.json missing/incomplete for the first COMPLETE run "
                f"of ({family_short}, {tasks[0]}) — cannot determine "
                f"n_channels for the permutation baseline")
        n_channels = int(cfg["n_channels"])
        n_channels_by_family[family_short] = n_channels

        # Recipients: the SAME K runs per task across all cells (docstring).
        recipients: dict[str, list[int]] = {}
        for t_idx, t in enumerate(tasks):
            rng = np.random.default_rng(
                [seed, _STREAM_RECIPIENTS, fam_idx, t_idx])
            pos = rng.choice(len(run_indices[t]), size=K, replace=False)
            recipients[t] = [int(run_indices[t][p]) for p in pos]
        recipients_by_family_task[family_short] = recipients

        for i_idx, t_i in enumerate(tasks):
            # Row baselines: native, identity, permuted, permuted_deviation
            # (self-donor). The two permuted kinds share the SAME sigma per
            # slot so their contrast isolates the identity-backbone effect
            # (module docstring; round-1 review fix).
            for k, r_idx in enumerate(recipients[t_i]):
                evals.append(_eval_entry(family_short, KIND_NATIVE,
                                         t_i, k, r_idx))
                evals.append(_eval_entry(family_short, KIND_IDENTITY,
                                         t_i, k, r_idx))
                perm_seed = [seed, _STREAM_PERMUTATION, fam_idx, i_idx, k]
                perm = draw_derangement(np.random.default_rng(perm_seed),
                                        n_channels * n_channels)
                for perm_kind in (KIND_PERMUTED, KIND_PERMUTED_DEV):
                    evals.append(_eval_entry(
                        family_short, perm_kind, t_i, k, r_idx,
                        task_j=t_i, donor_idx=r_idx,
                        permutation=[int(p) for p in perm],
                        perm_seed=perm_seed))

            # Cross-seed baseline: same task, different run.
            rng_cs = np.random.default_rng(
                [seed, _STREAM_CS_DONORS, fam_idx, i_idx])
            for k, r_idx in enumerate(recipients[t_i]):
                pool = [x for x in run_indices[t_i] if x != r_idx]
                donor = int(pool[int(rng_cs.integers(len(pool)))])
                evals.append(_eval_entry(
                    family_short, KIND_CROSS_SEED, t_i, k, r_idx,
                    task_j=t_i, donor_idx=donor))

            # Cross-task cells (+ decomposition guard cells on same pairs).
            for j_idx, t_j in enumerate(tasks):
                if j_idx == i_idx:
                    continue
                rng_ct = np.random.default_rng(
                    [seed, _STREAM_CT_DONORS, fam_idx, i_idx, j_idx])
                donor_pos = rng_ct.choice(len(run_indices[t_j]), size=K,
                                          replace=False)
                for k in range(K):
                    r_idx = recipients[t_i][k]
                    d_idx = int(run_indices[t_j][donor_pos[k]])
                    evals.append(_eval_entry(
                        family_short, KIND_CROSS_TASK, t_i, k, r_idx,
                        task_j=t_j, donor_idx=d_idx))
                    if decomposition:
                        for kind in (KIND_MAGNITUDE, KIND_TOPOLOGY):
                            evals.append(_eval_entry(
                                family_short, kind, t_i, k, r_idx,
                                task_j=t_j, donor_idx=d_idx))

    ids = [e["eval_id"] for e in evals]
    if len(ids) != len(set(ids)):
        raise RuntimeError("internal error: eval_id collision in plan")

    counts = dict(sorted(Counter(e["kind"] for e in evals).items()))
    counts["total"] = len(evals)
    return {
        "generator": "scripts/asset1_d2_swap.py",
        "plan_version": 2,
        "h3_structure_reference": {
            "candidates": [KIND_PERMUTED, KIND_PERMUTED_DEV],
            "recommended": KIND_PERMUTED_DEV,
            "status": "PENDING DIRECTOR SIGN-OFF — required before "
                      "Stage B",
            "rationale": (
                "Trained bridges are I + small deviation "
                "(bridge_mode='identity' init). The full-entry "
                "'permuted' cell scatters the untrained identity "
                "diagonal, so its penalty conflates init-structure "
                "destruction with trained-structure destruction and "
                "biases the H3 reference scale upward (round-1 review "
                "finding). 'permuted_deviation' permutes only "
                "D = B - I, preserving the identity backbone and the "
                "trained-deviation entry multiset. Both cells share "
                "one sigma per slot; both are evaluated; the Director "
                "must pin the H3 reference before any Stage B "
                "evaluation is run."),
        },
        "bank_root": str(bank_root),
        "seed": int(seed),
        "pairs_per_cell": K,
        "decomposition": bool(decomposition),
        "families": selected,
        "families_full": [f for f in fam_entries if f["short"] in selected],
        "tasks": tasks,
        "n_channels_by_family": n_channels_by_family,
        "recipients_by_family_task": recipients_by_family_task,
        "eval_counts": counts,
        "evals": evals,
    }


# ── Adapter assembly (Stage A, CPU) ─────────────────────────────────


def assemble_swapped_state(recipient: dict, donor: dict, kind: str,
                           permutation=None) -> dict[str, dict[str, torch.Tensor]]:
    """Assemble one swapped adapter: recipient lora_A/lora_B + transformed
    donor bridge, per module (matched by name; families never mixed).

    ``recipient`` / ``donor`` are asset1_analysis_io.load_adapter outputs
    ({module: {field: tensor}}); for self-donor kinds (native, identity,
    permuted) pass the recipient as donor. Raises ValueError on module-set
    or shape mismatches (the family-mixing guard: Qwen and Llama differ in
    module count and dimensions, so any cross-family pairing fails here).
    """
    if kind not in ALL_KINDS:
        raise ValueError(f"unknown swap kind {kind!r}; valid: "
                         f"{sorted(ALL_KINDS)}")
    if kind in _PERMUTATION_KINDS and permutation is None:
        raise ValueError(f"kind {kind!r} requires a permutation")
    if set(recipient) != set(donor):
        only_r = sorted(set(recipient) - set(donor))[:3]
        only_d = sorted(set(donor) - set(recipient))[:3]
        raise ValueError(
            f"module-name sets differ between recipient and donor "
            f"(family mixing?): only-recipient={only_r} only-donor={only_d}")

    out: dict[str, dict[str, torch.Tensor]] = {}
    for name in sorted(recipient):
        r_entry, d_entry = recipient[name], donor[name]
        for field in ("lora_A", "lora_B"):
            if r_entry[field].shape != d_entry[field].shape:
                raise ValueError(
                    f"{name}.{field} shape mismatch: recipient "
                    f"{tuple(r_entry[field].shape)} vs donor "
                    f"{tuple(d_entry[field].shape)} (family mixing?)")
        r_bridge, d_bridge = r_entry["bridge"], d_entry["bridge"]
        if r_bridge.shape != d_bridge.shape:
            raise ValueError(
                f"{name}.bridge shape mismatch: recipient "
                f"{tuple(r_bridge.shape)} vs donor {tuple(d_bridge.shape)}")

        if kind == KIND_NATIVE:
            new_bridge = r_bridge.clone()
        elif kind in (KIND_CROSS_TASK, KIND_CROSS_SEED):
            new_bridge = d_bridge.clone()
        elif kind == KIND_IDENTITY:
            new_bridge = torch.eye(r_bridge.shape[0], dtype=r_bridge.dtype)
        elif kind in _PERMUTATION_KINDS:
            transform = (permute_bridge if kind == KIND_PERMUTED
                         else permute_bridge_deviation)
            try:
                permuted = transform(r_bridge.detach().cpu().numpy(),
                                     permutation)
            except ValueError as e:
                raise ValueError(f"module {name!r}: {e}") from e
            new_bridge = torch.from_numpy(np.ascontiguousarray(permuted))
        elif kind == KIND_MAGNITUDE:
            new_bridge = torch.from_numpy(transplant_magnitude(
                r_bridge.detach().cpu().numpy(),
                d_bridge.detach().cpu().numpy()).astype(np.float32))
        else:  # KIND_TOPOLOGY
            new_bridge = torch.from_numpy(transplant_topology(
                r_bridge.detach().cpu().numpy(),
                d_bridge.detach().cpu().numpy()).astype(np.float32))

        entry = {
            "lora_A": r_entry["lora_A"].clone(),
            "lora_B": r_entry["lora_B"].clone(),
            "bridge": new_bridge,
        }
        for field in ("scaling", "n_channels", "rank"):
            if field in r_entry:
                entry[field] = r_entry[field].clone()
        out[name] = entry
    return out


def state_sha256(state: dict[str, dict[str, torch.Tensor]]) -> str:
    """Canonical SHA-256 of an assembled state: sorted module order, fixed
    field order, dtype/shape headers + raw little-endian bytes."""
    h = hashlib.sha256()
    for name in sorted(state):
        entry = state[name]
        for field in _STATE_FIELD_ORDER:
            if field not in entry:
                continue
            arr = np.ascontiguousarray(entry[field].detach().cpu().numpy())
            h.update(f"{name}.{field}|{arr.dtype.str}|{arr.shape}|".encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def state_to_flat(state: dict[str, dict[str, torch.Tensor]]
                  ) -> dict[str, torch.Tensor]:
    """Flatten to the bank's on-disk key format ('<module>.<field>') so a
    saved assembled state round-trips through
    asset1_canonicalize.load_adapter_modules."""
    flat: dict[str, torch.Tensor] = {}
    for name, entry in state.items():
        for field in _STATE_FIELD_ORDER:
            if field in entry:
                flat[f"{name}.{field}"] = entry[field]
    return flat


class _AdapterCache:
    """Tiny LRU over load_adapter — recipients stay hot within a matrix
    row while cross-task donors (used once each) rotate through."""

    def __init__(self, maxsize: int = _ADAPTER_CACHE_SIZE):
        self.maxsize = maxsize
        self._data: OrderedDict[Path, dict] = OrderedDict()

    def get(self, run_dir: Path) -> dict:
        if run_dir in self._data:
            self._data.move_to_end(run_dir)
            return self._data[run_dir]
        adapter = load_adapter(run_dir)
        self._data[run_dir] = adapter
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)
        return adapter


def _run_dir(bank_root: Path, family_short: str, task: str,
             run_index: int) -> Path:
    """The locked bank layout — the manifest's absolute run_dir strings are
    machine-specific and deliberately not trusted (foundation decision)."""
    return Path(bank_root) / family_short / task / f"run_{run_index:03d}"


def assemble_eval(bank_root: str | Path, ev: dict,
                  cache: _AdapterCache) -> dict:
    """Assemble the adapter state for one plan eval entry."""
    bank_root = Path(bank_root)
    recipient = cache.get(_run_dir(bank_root, ev["family_short"],
                                   ev["task_recipient"],
                                   ev["recipient_run_index"]))
    if ev["kind"] in _FOREIGN_DONOR_KINDS:
        donor = cache.get(_run_dir(bank_root, ev["family_short"],
                                   ev["task_donor"], ev["donor_run_index"]))
    else:
        donor = recipient
    return assemble_swapped_state(recipient, donor, ev["kind"],
                                  permutation=ev.get("permutation"))


def assemble_plan_states(bank_root: str | Path, plan: dict,
                         write_states_dir: str | Path | None = None) -> dict:
    """Assemble every eval's state, fill assembled_sha256 (and state_file
    when writing) in place, and return the plan. CPU-only; reads the bank,
    writes only under ``write_states_dir``."""
    bank_root = Path(bank_root)
    cache = _AdapterCache()
    states_dir = None
    if write_states_dir is not None:
        states_dir = Path(write_states_dir)
        states_dir.mkdir(parents=True, exist_ok=True)
    for ev in plan["evals"]:
        state = assemble_eval(bank_root, ev, cache)
        ev["assembled_sha256"] = state_sha256(state)
        if states_dir is not None:
            path = states_dir / f"{ev['eval_id']}.pt"
            torch.save(state_to_flat(state), path)
            ev["state_file"] = f"{STATES_DIRNAME}/{path.name}"
    return plan


# ── Penalty computation (pure; shared by Stage B and post-analysis) ─


def compute_penalties(rows: list[dict]) -> list[dict]:
    """penalty = swapped val_loss - native val_loss, joined per
    (task_recipient, recipient_run_index). ``rows`` are eval entries
    augmented with 'val_loss'. Raises if a swapped row has no native
    reference (the plan always provides one per recipient)."""
    natives: dict[tuple[str, int], float] = {}
    for row in rows:
        if row["kind"] == KIND_NATIVE:
            key = (row["task_recipient"], row["recipient_run_index"])
            natives[key] = float(row["val_loss"])
    out: list[dict] = []
    for row in rows:
        if row["kind"] == KIND_NATIVE:
            continue
        key = (row["task_recipient"], row["recipient_run_index"])
        if key not in natives:
            raise ValueError(
                f"no native val_loss for recipient {key} — cannot compute "
                f"penalty for {row['eval_id']}")
        out.append({
            "eval_id": row["eval_id"],
            "kind": row["kind"],
            "task_recipient": row["task_recipient"],
            "task_donor": row.get("task_donor"),
            "recipient_run_index": row["recipient_run_index"],
            "donor_run_index": row.get("donor_run_index"),
            "pair_slot": row.get("pair_slot"),
            "val_loss": float(row["val_loss"]),
            "native_val_loss": natives[key],
            "penalty": float(row["val_loss"]) - natives[key],
        })
    return out


# ── Stage B — GPU evaluation harness (POST-BANK ONLY) ───────────────
#
# Nothing below runs unless the operator passes BOTH --evaluate and
# --i-have-gpu-and-bank-is-complete. transformers / datasets / asset1_bank
# are imported lazily INSIDE these functions so importing this module never
# touches HF or the heavyweight training stack.


def _load_family_model(model_id: str, device):
    """GPU GATE — POST-BANK ONLY. Loads tokenizer + bf16 base model on
    ``device`` (the bank's dtype convention), frozen and in eval mode.
    Lazy-imports transformers; never call from tests or pre-bank code."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


def _install_state(injected: dict, safe_to_dotted: dict[str, str],
                   state: dict) -> None:
    """GPU GATE — POST-BANK ONLY. Copy an assembled state into the injected
    RhombiLoRALinear modules (bridge_mode='identity' -> standard mode, so
    the stored EFFECTIVE bridge is exactly the .bridge parameter)."""
    if set(safe_to_dotted) != set(state):
        raise ValueError(
            "assembled state modules do not match the injected modules — "
            "wrong family for this plan?")
    with torch.no_grad():
        for safe, entry in state.items():
            lora = injected[safe_to_dotted[safe]]
            lora.lora_A.data.copy_(entry["lora_A"].to(lora.lora_A.dtype))
            lora.lora_B.data.copy_(entry["lora_B"].to(lora.lora_B.dtype))
            lora.bridge.data.copy_(entry["bridge"].to(lora.bridge.dtype))


def evaluate_plan(bank_root: str | Path, out_dir: str | Path, plan: dict,
                  family_short: str, device_str: str = "cuda") -> dict:
    """GPU GATE — POST-BANK ONLY. Evaluate every plan eval of one family.

    Loads the base model ONCE, then for each eval: re-assembles the state
    from the bank (verifying the recorded SHA-256 — the bank must not have
    changed since planning), installs it, and computes val_loss on the
    recipient task's FIXED validation split (asset1_datasets.build_dataset
    'val'; invariant to data_seed by the locked val_seed=777 design) using
    asset1_bank.evaluate_val_loss with the bank's EVAL_BATCH_SIZE / MAX_LEN.
    Results are checkpointed to d2_results_<family>.json after every eval;
    penalties are appended at the end via compute_penalties.
    """
    from datetime import datetime, timezone       # results provenance only
    from torch.utils.data import DataLoader       # lazy — Stage B only

    import asset1_bank as bank_mod                # lazy — heavy deps
    from asset1_datasets import build_dataset     # lazy — HF datasets

    bank_root = Path(bank_root)
    out_dir = Path(out_dir)
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[asset1-d2] REFUSING: --evaluate requested on "
                         "cuda but no CUDA device is available")

    fam_evals = [e for e in plan["evals"]
                 if e["family_short"] == family_short]
    if not fam_evals:
        raise SystemExit(f"[asset1-d2] plan contains no evals for family "
                         f"{family_short!r} (families: {plan['families']})")
    fam_entry = next(f for f in plan["families_full"]
                     if f["short"] == family_short)

    cache = _AdapterCache()
    first_state = assemble_eval(bank_root, fam_evals[0], cache)
    first_entry = next(iter(first_state.values()))
    rank = int(first_entry["rank"].item())
    n_channels = int(first_entry["n_channels"].item())

    tokenizer, model = _load_family_model(fam_entry["model"], device)
    injected = bank_mod.inject_rhombi_lora(
        model, rank, n_channels, bank_mod.TARGET_MODULES)
    safe_to_dotted = {n.replace(".", "_"): n for n in injected}

    val_loaders: dict[str, DataLoader] = {}

    def _val_loader(task: str) -> DataLoader:
        if task not in val_loaders:
            ds = build_dataset(task, tokenizer, "val", data_seed=0,
                               max_len=bank_mod.MAX_LEN,
                               pool_cap=bank_mod.POOL_CAP)
            val_loaders[task] = DataLoader(
                ds, batch_size=bank_mod.EVAL_BATCH_SIZE, shuffle=False,
                num_workers=0, pin_memory=True)
        return val_loaders[task]

    results_path = out_dir / f"d2_results_{family_short}.json"
    results: dict = {
        "generator": "scripts/asset1_d2_swap.py",
        "family_short": family_short,
        "model": fam_entry["model"],
        "plan_seed": plan["seed"],
        "pairs_per_cell": plan["pairs_per_cell"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "rows": [],
        "penalties": None,
    }

    def _flush() -> None:
        results_path.write_text(json.dumps(results, indent=2),
                                encoding="utf-8")

    for n, ev in enumerate(fam_evals, 1):
        state = assemble_eval(bank_root, ev, cache)
        sha = state_sha256(state)
        plan_sha = ev.get("assembled_sha256")
        if plan_sha and sha != plan_sha:
            raise RuntimeError(
                f"assembled-state SHA mismatch for {ev['eval_id']}: plan "
                f"{plan_sha[:12]}... vs now {sha[:12]}... — "
                f"bank contents changed since planning; regenerate the plan")
        if not plan_sha:
            print(f"WARNING: {ev['eval_id']} carries no assembled_sha256 "
                  f"(plan generated with --plan-only) — bank-integrity "
                  f"check SKIPPED; recording sha_verified=false",
                  file=sys.stderr)
        _install_state(injected, safe_to_dotted, state)
        val_loss = bank_mod.evaluate_val_loss(
            model, _val_loader(ev["task_recipient"]), device)
        results["rows"].append({**{k: ev[k] for k in (
            "eval_id", "kind", "task_recipient", "task_donor",
            "recipient_run_index", "donor_run_index", "pair_slot")},
            "assembled_sha256": sha, "sha_verified": bool(plan_sha),
            "val_loss": val_loss})
        print(f"  [{n}/{len(fam_evals)}] {ev['eval_id']} | "
              f"val_loss={val_loss:.4f}", flush=True)
        _flush()

    results["penalties"] = compute_penalties(results["rows"])
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    _flush()
    print(f"Stage B complete for {family_short}: {len(fam_evals)} evals -> "
          f"{results_path}", flush=True)
    return results


# ── CLI ─────────────────────────────────────────────────────────────


def _guard_out_dir(out_dir: Path, bank_root: Path) -> None:
    """Refuse any output location that could write into the live bank."""
    resolved = out_dir.resolve()
    if "asset1-bank" in resolved.parts:
        raise SystemExit(
            f"[asset1-d2] REFUSING: --out-dir {resolved} contains an "
            f"'asset1-bank' path component — that is the LIVE campaign "
            f"tree and is write-protected")
    bank_resolved = bank_root.resolve()
    if resolved == bank_resolved or bank_resolved in resolved.parents:
        raise SystemExit(
            f"[asset1-d2] REFUSING: --out-dir {resolved} lies inside "
            f"--bank-root {bank_resolved} — analyses never write into "
            f"the bank")


def _print_eval_counts(plan: dict) -> None:
    print("D2 swap plan eval counts:")
    for kind, n in plan["eval_counts"].items():
        if kind == "total":
            continue
        print(f"  {kind:<24} {n}")
    print(f"  {'total':<24} {plan['eval_counts']['total']}")
    per_fam = Counter(e["family_short"] for e in plan["evals"])
    for fam, n in sorted(per_fam.items()):
        print(f"  (family {fam}: {n} evals)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Asset 1 D2 — cross-task bridge-swap matrix. Stage A "
                    "(CPU) builds the deterministic swap plan and assembles "
                    "adapter states; Stage B (--evaluate, GPU, post-bank "
                    "only) computes the val-loss matrix.")
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--family", type=str, default=None,
                        help="family_short to restrict to (Stage A default: "
                             "all families; REQUIRED for --evaluate — one "
                             "command per family)")
    parser.add_argument("--seed", type=int, default=0,
                        help="master seed for all pairing/permutation "
                             "randomness (default 0)")
    parser.add_argument("--pairs-per-cell", type=int,
                        default=DEFAULT_PAIRS_PER_CELL,
                        help=f"K donor/recipient pairs per matrix cell "
                             f"(default {DEFAULT_PAIRS_PER_CELL}; real bank "
                             f"at K=3 -> 180 evals/family, 360 with "
                             f"--decomposition)")
    parser.add_argument("--decomposition", action="store_true",
                        help="add the magnitude-only / topology-only "
                             "contradiction-guard cells (2x T*(T-1)*K more "
                             "evals per family)")
    parser.add_argument("--plan-only", action="store_true",
                        help="write the plan without loading any adapter "
                             "(assembled_sha256 stays null)")
    parser.add_argument("--write-states", action="store_true",
                        help="also save every assembled state as a .pt "
                             "under <out-dir>/states/")
    parser.add_argument("--allow-partial-bank", action="store_true",
                        help="EXPLORATORY ONLY: bypass the pre-registration "
                             "interlock with a loud warning")
    parser.add_argument("--evaluate", action="store_true",
                        help="Stage B: run the GPU evaluation harness over "
                             "an existing plan (post-bank only)")
    parser.add_argument("--i-have-gpu-and-bank-is-complete",
                        action="store_true",
                        help="explicit operator acknowledgement required by "
                             "--evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Stage B device (default cuda)")
    args = parser.parse_args(argv)

    _guard_out_dir(args.out_dir, args.bank_root)

    if args.evaluate:
        # Gate BEFORE any bank access or lazy heavyweight import.
        if not args.i_have_gpu_and_bank_is_complete:
            raise SystemExit(
                "[asset1-d2] REFUSING: --evaluate is the GPU Stage B "
                "harness and runs only post-bank. Pass "
                "--i-have-gpu-and-bank-is-complete to acknowledge.")
        if args.family is None:
            raise SystemExit(
                "[asset1-d2] --family is required for --evaluate "
                "(Stage B runs one command per family)")

    # THE INTERLOCK — every path that can touch the real bank goes
    # through this gate (asset1_analysis_io.require_complete_bank).
    require_complete_bank(args.bank_root,
                          allow_partial=args.allow_partial_bank)

    if args.evaluate:
        plan_path = args.out_dir / PLAN_FILENAME
        if not plan_path.exists():
            raise SystemExit(
                f"[asset1-d2] no {PLAN_FILENAME} under {args.out_dir} — "
                f"run Stage A first")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        evaluate_plan(args.bank_root, args.out_dir, plan, args.family,
                      device_str=args.device)
        return

    # ── Stage A ──
    families = None if args.family is None else [args.family]
    plan = generate_plan(args.bank_root, seed=args.seed,
                         pairs_per_cell=args.pairs_per_cell,
                         families=families,
                         decomposition=args.decomposition)
    _print_eval_counts(plan)

    if not args.plan_only:
        states_dir = (args.out_dir / STATES_DIRNAME
                      if args.write_states else None)
        assemble_plan_states(args.bank_root, plan,
                             write_states_dir=states_dir)
        print("Assembled all states"
              + (f"; .pt files under {states_dir}" if states_dir else
                 " (SHA-256 recorded; no .pt files — pass --write-states "
                 "to persist them)"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / PLAN_FILENAME
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Plan written to {plan_path}")
    print("Stage B (post-bank, GPU) — one command per family, e.g.:")
    for fam in plan["families"]:
        print(f"  python scripts/asset1_d2_swap.py --bank-root "
              f"{args.bank_root} --out-dir {args.out_dir} --family {fam} "
              f"--evaluate --i-have-gpu-and-bank-is-complete")


if __name__ == "__main__":
    main()
