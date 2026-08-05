"""Acceptance tests for the E-T4 Tinker mini-bank machinery.

Everything here runs on synthetic adapters and injected raw rows, so no
network, no GPU, and no (paid) real bank is touched. Five properties, each
guarding a specific way the mini-bank could be quietly wrong:

1.  FORMATTER FIDELITY — the 6-task emitter builds its dataset objects with
    ``cls.__new__`` to skip eager padded tokenization. That shortcut is only
    legitimate if the text it produces is identical to what the real
    ``Asset1TaskDataset.__init__`` path produces. Checked on the two
    long-document tasks (xsum, squad), whose ``_fit_document`` budgeting is
    the part most likely to depend on skipped state.
2.  EXACT RAW GRAM — the readout accumulates the raw Gram module-major so it
    never materializes 54 x 92M-dim vectors. It must equal the naive Gram of
    the fully concatenated vectors, or every raw distance is wrong.
3.  LABEL PARSING — mini-bank ``task_d<D>_i<I>`` names and the pilot's
    ``task_<seed>`` names must both resolve, since the readout is validated
    by running it on the pilot bank.
4.  LOO k-NN — leave-one-out is structural (a point must never be its own
    neighbour) and a planted per-task signal must be recovered.
5.  VERTEX-DISJOINT PAIRING — every adapter appears in at most one merge_lint
    pair, and no pair is same-task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tinker_minibank_data import make_formatter  # noqa: E402
from tinker_minibank_merge_lint import vertex_disjoint_pairs  # noqa: E402
from tinker_minibank_signal import (  # noqa: E402
    dist_matrix, gram_to_distances, labels_for, loo_knn, raw_gram)

RANK, D_OUT, D_IN = 4, 12, 10


# ── 1. Formatter fidelity ────────────────────────────────────────────


class _StubTokenizer:
    """Whitespace tokenizer with the HF surface ``_fit_document`` uses."""

    def __call__(self, text, add_special_tokens=True, truncation=False,
                 max_length=None, **kw):
        ids = [len(w) for w in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("w" * max(n, 1) for n in ids)


@pytest.mark.parametrize("task, row", [
    ("xsum", {"document": " ".join(f"doc{i}" for i in range(400)),
              "summary": "a short summary"}),
    ("squad", {"context": " ".join(f"ctx{i}" for i in range(400)),
               "question": "who?", "answers": {"text": ["someone"]}}),
])
def test_formatter_matches_full_dataset_path(task, row):
    """__new__-built formatter == the real __init__ path's formatted text."""
    from asset1_datasets import TASK_REGISTRY

    tok = _StubTokenizer()
    fast = make_formatter(task, tok, max_len=128)
    fast_text = fast.format_example(row)

    # The real path, with eager tokenization, over an injected raw dataset.
    cls = TASK_REGISTRY[task]

    class _EagerTok(_StubTokenizer):
        def __call__(self, text, **kw):
            out = super().__call__(text, **kw)
            if kw.get("return_tensors") == "pt":
                ids = torch.tensor(out["input_ids"][:128] or [0])
                ids = torch.nn.functional.pad(ids, (0, 128 - ids.numel()))
                return {"input_ids": ids.unsqueeze(0),
                        "attention_mask": torch.ones_like(ids).unsqueeze(0)}
            return out

    raw = [row] * 501            # > VAL_SIZE so the split is legal
    ds = cls(_EagerTok(), "train", data_seed=0, max_len=128,
             raw=raw, keep_text=True)
    assert ds.formatted_texts[0] == fast_text


# ── 2. Exact raw Gram ────────────────────────────────────────────────


def _write_adapter(run_dir: Path, modules: dict[str, tuple]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, (A, B) in modules.items():
        tensors[f"base_model.model.{name}.lora_A.weight"] = A
        tensors[f"base_model.model.{name}.lora_B.weight"] = B
    save_file(tensors, str(run_dir / "adapter_model.safetensors"))
    (run_dir / "adapter_config.json").write_text(
        '{"r": %d, "lora_alpha": %d}' % (RANK, RANK), encoding="utf-8")


def test_raw_gram_equals_naive_concatenated_gram(tmp_path):
    rng = torch.Generator().manual_seed(0)
    names = ["layers.0.q_proj", "layers.0.mlp.down_proj", "layers.1.v_proj"]
    run_dirs, naive = [], []
    for k in range(5):
        mods = {n: (torch.randn(RANK, D_IN, generator=rng),
                    torch.randn(D_OUT, RANK, generator=rng)) for n in names}
        d = tmp_path / f"task{k % 2}_d0_i{k}"
        _write_adapter(d, mods)
        run_dirs.append(d)
        # Naive: the pilot's definition — A then B per module, sorted names.
        naive.append(np.concatenate(
            [np.concatenate([mods[n][0].reshape(-1).numpy(),
                             mods[n][1].reshape(-1).numpy()])
             for n in sorted(names)]).astype(np.float64))

    G, module_names, raw_dim = raw_gram(run_dirs)
    X = np.stack(naive)
    assert raw_dim == X.shape[1]
    assert len(module_names) == len(names)
    np.testing.assert_allclose(G, X @ X.T, rtol=1e-10, atol=1e-10)

    # ...and the distances derived from it match the direct computation.
    ids = [d.name for d in run_dirs]
    got = gram_to_distances(G, ids)
    norms = np.linalg.norm(X, axis=1)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i < j:
                want = 1.0 - (X[i] @ X[j]) / (norms[i] * norms[j])
                assert got[tuple(sorted((a, b)))] == pytest.approx(want, rel=1e-10)


# ── 3. Label parsing ─────────────────────────────────────────────────


def test_labels_parse_both_naming_schemes(tmp_path):
    mb = tmp_path / "xsum_d2_i1"
    mb.mkdir()
    assert labels_for(mb) == {"task": "xsum", "data_seed": 2, "init_seed": 1}

    pilot = tmp_path / "agnews_1"          # pilot: one seed drove both
    pilot.mkdir()
    assert labels_for(pilot) == {"task": "agnews", "data_seed": 1,
                                 "init_seed": 1}


# ── 4. LOO k-NN ──────────────────────────────────────────────────────


def test_loo_knn_is_leave_one_out_and_recovers_planted_signal():
    # Three tasks x three replicates, tight clusters with a planted offset.
    rng = np.random.default_rng(0)
    centres = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]),
               "c": np.array([-1.0, -1.0])}
    y, X = [], []
    for task, c in centres.items():
        for _ in range(3):
            y.append(task)
            X.append(c + 0.01 * rng.standard_normal(2))
    X = np.stack(X)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)

    acc, preds = loo_knn(D, y, k=1)
    assert acc == 1.0 and preds == y

    # LOO is structural: a zero self-distance must not be votable. If the
    # diagonal were used, k=1 would be trivially perfect even on noise.
    y_shuf = ["a", "b", "c"] * 3
    acc_shuf, _ = loo_knn(D, y_shuf, k=1)
    assert acc_shuf < 1.0


def test_dist_matrix_diagonal_is_infinite():
    dists = {("r0", "r1"): 0.5, ("r0", "r2"): 0.25, ("r1", "r2"): 0.75}
    D = dist_matrix(dists, ["r0", "r1", "r2"])
    assert np.all(np.isinf(np.diag(D)))
    assert D[0, 1] == D[1, 0] == 0.5


# ── 5. Vertex-disjoint pairing ───────────────────────────────────────


def test_pairs_are_vertex_disjoint_and_cross_task():
    tasks = ["alpaca", "code", "math", "xsum", "squad", "agnews"]
    runs, labels = [], {}
    for t in tasks:
        for d in range(3):
            for i in range(3):
                rid = f"{t}_d{d}_i{i}"
                runs.append(rid)
                labels[rid] = {"task": t, "data_seed": d, "init_seed": i}

    pairs = vertex_disjoint_pairs(runs, labels, seed=0)
    flat = [r for p in pairs for r in p]
    assert len(flat) == len(set(flat)), "an adapter appears in two pairs"
    assert all(labels[a]["task"] != labels[b]["task"] for a, b in pairs)
    # 54 adapters, 6 balanced tasks -> a perfect matching is reachable.
    assert len(pairs) == 27
