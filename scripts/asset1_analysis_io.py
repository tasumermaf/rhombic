"""Asset 1 — shared analysis IO layer (FOUNDATION module for D1/D2/D3/D-aux).

Every downstream analysis tool imports run enumeration, adapter loading,
feature flattening, and metrics parsing from HERE, so the pre-registration
interlock and the deterministic feature ordering cannot drift between
analyses.

Design decisions
----------------
1.  THE MANIFEST IS THE SOURCE OF TRUTH. ``iter_runs`` walks
    ``bank_manifest.json`` (written atomically by ``asset1_bank.py``), never
    the filesystem. A half-written run directory that the live trainer is
    populating RIGHT NOW is invisible to analysis until the campaign runner
    marks it COMPLETE in the manifest. The manifest's per-entry ``run_dir``
    string is an absolute path from the writing machine, so run paths are
    re-derived from ``bank_root`` + the locked layout
    ``<family_short>/<task>/run_<idx:03d>`` (asset1_bank.generate_manifest).

2.  THE INTERLOCK (require_complete_bank). The experiment card locks the
    D1-D3/D-aux analyses to the FULL 480-run bank. Any CLI that can point at
    the real bank must call ``require_complete_bank()`` before touching a
    single adapter; it raises SystemExit unless the manifest shows all
    ``expected_total`` runs COMPLETE. Downstream CLIs expose an explicit
    ``--allow-partial-bank`` flag that maps to ``allow_partial=True``, which
    prints a loud PRE-REGISTRATION WARNING to stderr and continues — for
    tooling smoke checks only, never for reportable numbers. This module
    deliberately has NO CLI of its own so there is no ungated entry point.

3.  READ-ONLY AND CPU-ONLY. Nothing in this module writes, moves, or deletes
    anything anywhere; the campaign tree is live. All tensor loading is
    ``map_location="cpu"`` with ``weights_only=True`` — the bank's
    ``adapter_state.pt`` is a flat dict of plain tensors (asset1_bank.py
    save loop), which torch 2.6's weights_only unpickler accepts; this is
    already proven in production by ``asset1_canonicalize.load_adapter_modules``
    (reused here as the single source of truth for key parsing).

4.  DETERMINISTIC FEATURE ORDER. ``flatten_features`` concatenates modules
    in sorted-name order and tensors within a module in the fixed order
    lora_A, lora_B, bridge (row-major flattening). ``feature_layout`` is
    computed by the SAME generator, so its (module, tensor, slice) map can
    never disagree with the vector. Raw parameters are flattened as stored —
    no scaling multiplication, no normalization — matching the pre-registered
    "raw all-modules adapter parameters" D1 headline. Nothing here depends on
    module COUNT (Qwen 112, Llama 64): the module set is discovered from the
    adapter file itself.

5.  NaN CONVENTION. metrics.json records train_loss = null at step 0 (no
    training has happened yet). ``load_gap_trajectory`` maps null -> np.nan
    so the arrays stay rectangular; D-aux consumers compute the gap as
    ``val - train`` and get NaN at step 0, which is the honest value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asset1_canonicalize import load_adapter_modules  # noqa: E402

# ── Locked constants ────────────────────────────────────────────────

EXPECTED_TOTAL_RUNS = 480          # the locked experiment card (2026-07-03)
REAL_BANK_ROOT = REPO_ROOT / "results" / "asset1-bank"
MANIFEST_NAME = "bank_manifest.json"

# Fixed tensor order within a module: (include-key, adapter-state field).
_TENSOR_ORDER: tuple[tuple[str, str], ...] = (
    ("A", "lora_A"),
    ("B", "lora_B"),
    ("bridge", "bridge"),
)
_VALID_INCLUDE = frozenset(k for k, _ in _TENSOR_ORDER)


# ── Manifest ────────────────────────────────────────────────────────


def load_manifest(bank_root: str | Path) -> dict:
    """Parse ``<bank_root>/bank_manifest.json``.

    Raises FileNotFoundError with a clear message if the manifest is
    missing — a bank without a manifest is not a bank.
    """
    path = Path(bank_root) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} under {Path(bank_root)} — not an Asset-1 "
            f"bank root (the manifest is written by asset1_bank.py)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "runs" not in manifest or not isinstance(manifest["runs"], list):
        raise ValueError(f"{path} has no 'runs' list — corrupt manifest")
    return manifest


def run_status_counts(manifest: dict) -> dict[str, int]:
    """Status -> count, recomputed from the run entries (NOT the manifest's
    own cached ``status_counts`` block, which could in principle be stale)."""
    counts: dict[str, int] = {}
    for entry in manifest["runs"]:
        s = entry["status"]
        counts[s] = counts.get(s, 0) + 1
    return dict(sorted(counts.items()))


def require_complete_bank(bank_root: str | Path,
                          allow_partial: bool = False,
                          expected_total: int = EXPECTED_TOTAL_RUNS) -> dict:
    """THE INTERLOCK — refuse to analyze a partial bank (pre-registration).

    Passes only when the manifest lists exactly ``expected_total`` runs and
    every one of them has status COMPLETE. Otherwise raises SystemExit with
    a clear refusal — unless ``allow_partial=True`` (a downstream CLI's
    explicit ``--allow-partial-bank`` flag), in which case a loud multi-line
    PRE-REGISTRATION WARNING goes to stderr and execution continues.

    ``expected_total`` defaults to the locked 480; overriding it is for
    synthetic fixtures and tests ONLY and must never be exposed as a CLI
    knob on a real-bank tool.

    Returns the parsed manifest on success (and on the warned partial path)
    so callers do not have to re-read it.
    """
    bank_root = Path(bank_root)
    try:
        manifest = load_manifest(bank_root)
    except FileNotFoundError as e:
        raise SystemExit(f"[asset1-interlock] REFUSING to run: {e}") from e

    counts = run_status_counts(manifest)
    n_listed = len(manifest["runs"])
    n_complete = counts.get("COMPLETE", 0)
    if n_listed == expected_total and n_complete == expected_total:
        return manifest

    if not allow_partial:
        raise SystemExit(
            f"[asset1-interlock] REFUSING to run: bank at {bank_root} is not "
            f"complete — {n_complete}/{expected_total} runs COMPLETE "
            f"(manifest lists {n_listed} runs; statuses: {counts}). The "
            f"locked experiment card (2026-07-03) pre-registers every "
            f"D1/D2/D3/D-aux analysis on the FULL bank. Re-run once the "
            f"campaign finishes, or pass --allow-partial-bank for an "
            f"exploratory tooling check that will NEVER be reported.")

    sys.stderr.write(
        "\n"
        "======================================================================\n"
        "***          PRE-REGISTRATION WARNING — PARTIAL BANK              ***\n"
        "======================================================================\n"
        f"Bank root : {bank_root}\n"
        f"Complete  : {n_complete}/{expected_total} runs "
        f"(manifest lists {n_listed}; statuses: {counts})\n"
        "This invocation is running against an INCOMPLETE adapter bank.\n"
        "Every number it produces is EXPLORATORY ONLY. Nothing computed here\n"
        "may be reported, quoted, or compared against the pre-registered\n"
        "hypotheses of the locked experiment card (2026-07-03). Proceeding\n"
        "because --allow-partial-bank was explicitly set.\n"
        "======================================================================\n"
        "\n")
    sys.stderr.flush()
    return manifest


# ── Run enumeration ─────────────────────────────────────────────────


def iter_runs(bank_root: str | Path,
              family: str | None = None,
              task: str | None = None,
              only_complete: bool = True) -> Iterator[dict]:
    """Yield per-run records by walking the MANIFEST (not the filesystem).

    Parameters
    ----------
    family : exact match against either the manifest entry's ``family``
        (full HF id) or ``family_short``; None = all families.
    task : exact match against the entry's ``task``; None = all tasks.
    only_complete : yield only status == COMPLETE entries (default — the
        live campaign is writing into this tree and non-COMPLETE run dirs
        are not safe to read).

    Yields dicts (manifest order — deterministic, run_index ascending):
        {run_dir: Path, config: dict | None, family_short, task,
         run_index, replicate}
    ``config`` is the parsed config.json, or None if the file does not
    exist (possible only for non-COMPLETE entries).
    """
    bank_root = Path(bank_root)
    manifest = load_manifest(bank_root)
    for entry in manifest["runs"]:
        if only_complete and entry["status"] != "COMPLETE":
            continue
        if family is not None and family not in (entry["family"],
                                                 entry["family_short"]):
            continue
        if task is not None and entry["task"] != task:
            continue
        run_dir = (bank_root / entry["family_short"] / entry["task"]
                   / f"run_{entry['run_index']:03d}")
        cfg_path = run_dir / "config.json"
        config = (json.loads(cfg_path.read_text(encoding="utf-8"))
                  if cfg_path.exists() else None)
        yield {
            "run_dir": run_dir,
            "config": config,
            "family_short": entry["family_short"],
            "task": entry["task"],
            "run_index": entry["run_index"],
            "replicate": entry["replicate"],
        }


# ── Adapter loading ─────────────────────────────────────────────────


def load_adapter(run_dir: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load a run's adapter_state.pt as {module: {field: cpu tensor}}.

    Fields per module: lora_A (r, d_in), lora_B (d_out, r), bridge (C, C —
    the EFFECTIVE bridge), scaling (0-dim), n_channels (0-dim), rank (0-dim).
    Delegates key parsing to asset1_canonicalize.load_adapter_modules — the
    single production-proven loader (torch.load map_location='cpu',
    weights_only=True; safe for the bank's plain-tensor dicts under
    torch 2.6).
    """
    path = Path(run_dir) / "adapter_state.pt"
    if not path.exists():
        raise FileNotFoundError(f"no adapter_state.pt under {Path(run_dir)}")
    return load_adapter_modules(path)


# ── Feature flattening (deterministic order) ────────────────────────


def _normalize_include(include: Iterable[str]) -> tuple[str, ...]:
    inc = set(include)
    if not inc:
        raise ValueError("include must name at least one of "
                         f"{sorted(_VALID_INCLUDE)}")
    bad = inc - _VALID_INCLUDE
    if bad:
        raise ValueError(f"unknown include keys {sorted(bad)}; "
                         f"valid: {sorted(_VALID_INCLUDE)}")
    # Canonical order is fixed by _TENSOR_ORDER, NOT by the caller's
    # container order — determinism must not depend on set iteration.
    return tuple(k for k, _ in _TENSOR_ORDER if k in inc)


def _select_modules(adapter: dict, modules) -> list[str]:
    if isinstance(modules, str):
        if modules != "all":
            raise ValueError(f"modules must be 'all' or a list of names, "
                             f"got {modules!r}")
        return sorted(adapter)
    names = list(modules)
    if len(names) != len(set(names)):
        raise ValueError("duplicate module names in modules list")
    missing = [n for n in names if n not in adapter]
    if missing:
        raise ValueError(f"modules not present in adapter: {missing}")
    return sorted(names)   # sorted ALWAYS — order never caller-dependent


def _feature_blocks(adapter: dict, modules, include):
    """Yield (module, field, flat float32 ndarray) in the canonical order:
    sorted module names, then lora_A / lora_B / bridge within each module.
    Shared by flatten_features and feature_layout so they cannot diverge."""
    inc = _normalize_include(include)
    for name in _select_modules(adapter, modules):
        entry = adapter[name]
        for key, field in _TENSOR_ORDER:
            if key not in inc:
                continue
            if field not in entry:
                raise ValueError(
                    f"module {name!r} has no {field!r} tensor (include "
                    f"requested {key!r})")
            t = entry[field]
            flat = np.asarray(t.detach().cpu().numpy(),
                              dtype=np.float32).reshape(-1)
            yield name, field, flat


def flatten_features(adapter: dict,
                     modules="all",
                     include: Iterable[str] = ("A", "B", "bridge"),
                     ) -> np.ndarray:
    """Flatten an adapter into ONE 1-D float32 vector, deterministically.

    Module order: sorted module names. Tensor order within a module:
    lora_A, lora_B, bridge (restricted to ``include``; the include
    container's own order is irrelevant). Values are the RAW stored
    parameters — no scaling, no normalization (D1 raw-parameter headline).
    """
    blocks = [flat for _, _, flat in _feature_blocks(adapter, modules, include)]
    return np.concatenate(blocks)


def feature_layout(adapter: dict,
                   modules="all",
                   include: Iterable[str] = ("A", "B", "bridge"),
                   ) -> list[tuple[str, str, slice]]:
    """Index map for flatten_features: [(module, field, slice), ...].

    Built by the same block generator with identical arguments, so
    ``vec[slc] == adapter[module][field].reshape(-1)`` for every row and
    the slices tile the vector exactly (per-module breakdowns slice the
    same array the classifier saw).
    """
    layout: list[tuple[str, str, slice]] = []
    offset = 0
    for name, field, flat in _feature_blocks(adapter, modules, include):
        layout.append((name, field, slice(offset, offset + flat.size)))
        offset += flat.size
    return layout


# ── Metrics / bridges ───────────────────────────────────────────────


def load_gap_trajectory(run_dir: str | Path
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse metrics.json into (steps, train_loss, val_loss) arrays.

    steps int64; losses float64 with JSON null -> np.nan (step 0 records
    train_loss = null by design). The D-aux gap is val_loss - train_loss.
    """
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.json under {Path(run_dir)}")
    records = json.loads(path.read_text(encoding="utf-8"))["records"]

    def _f(x) -> float:
        return np.nan if x is None else float(x)

    steps = np.array([r["step"] for r in records], dtype=np.int64)
    train = np.array([_f(r["train_loss"]) for r in records], dtype=np.float64)
    val = np.array([_f(r["val_loss"]) for r in records], dtype=np.float64)
    return steps, train, val


def load_bridges(run_dir: str | Path,
                 which: str = "final") -> dict[str, np.ndarray]:
    """Load the per-module bridge .npy files: {module_safe_name: (C, C)}.

    which : 'final' (trained effective bridges) or 'step0' (initial).
    File naming per asset1_bank.py: bridge_{which}_{module_safe}.npy.
    Sorted filename order; np.load with the default allow_pickle=False.
    """
    if which not in ("final", "step0"):
        raise ValueError(f"which must be 'final' or 'step0', got {which!r}")
    run_dir = Path(run_dir)
    prefix = f"bridge_{which}_"
    out: dict[str, np.ndarray] = {}
    for p in sorted(run_dir.glob(prefix + "*.npy")):
        out[p.stem[len(prefix):]] = np.load(p)
    if not out:
        raise FileNotFoundError(
            f"no {prefix}*.npy files under {run_dir}")
    return out
