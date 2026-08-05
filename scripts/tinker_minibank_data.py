"""E-T4 Tinker mini-bank — training-text generation for all SIX Asset-1 tasks.

Extends ``scripts/tinker_pilot_data.py`` from the pilot's three
tokenizer-free tasks (alpaca, math, agnews) to the full Asset-1 set by
adding the three that the pilot could not emit: **code, xsum, squad**.

The locked Asset-1 machinery is REUSED, never forked. Specifically:

    * ``split_ids(n_total, data_seed)`` — the canonical VAL_SEED=777
      shuffle, first VAL_SIZE=500 rows held out as the fixed val split,
      the next POOL_CAP=40,000 forming the train pool whose composition
      is fixed per task and whose ORDER is permuted by
      ``RandomState(data_seed)``.
    * ``TASK_REGISTRY[task].format_example`` — the task's own formatter,
      called on the real class, so the ``### Instruction: / ### Response:``
      templates and the long-document ``_fit_document`` budget truncation
      (xsum, squad) are the bank's code executing, not a copy of it.

Why the classes are instantiated with ``__new__`` instead of ``__init__``
------------------------------------------------------------------------
``Asset1TaskDataset.__init__`` eagerly tokenizes every selected example
to a PADDED fixed-length torch tensor — the bank's training convention.
Tinker needs the opposite: plain text, unpadded (padded positions are
billed as train tokens), tokenized server-side by the SERVED model's
tokenizer. So the dataset object is built without ``__init__`` and given
exactly the two attributes its formatter needs (``tokenizer``,
``max_len``); ``format_example`` / ``_fit_document`` / ``_post_load``
then run verbatim off the registry class. Nothing about the formatting
or the split is reimplemented here.

Tokenizer choice: ``Qwen/Qwen3-8B`` — the served base model. Only xsum
and squad consult it at all (document budget truncation), and budgeting
against the tokenizer that will actually bill the tokens is the point.

Deterministic: same (task, data_seed) always yields byte-identical
output. Regenerable, so the emitted .jsonl streams are gitignored.

RUNS UNDER THE ``falco`` CONDA ENV (needs torch + datasets + transformers):
    C:\\miniconda3\\envs\\falco\\python.exe

Usage
-----
    python scripts/tinker_minibank_data.py                    # 6 tasks x 3 seeds
    python scripts/tinker_minibank_data.py --tasks xsum --seeds 0
    python scripts/tinker_minibank_data.py --verify-against-pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asset1_datasets import (  # noqa: E402  (locked Asset-1 machinery)
    POOL_CAP, TASK_REGISTRY, VAL_SEED, VAL_SIZE,
    load_hf_dataset_with_fallback, split_ids)

BASE_MODEL = "Qwen/Qwen3-8B"
MAX_SEQ_LEN = 512

# Full Asset-1 task set (the pilot ran the first three).
MINIBANK_TASKS = ("alpaca", "code", "math", "xsum", "squad", "agnews")

# Three data seeds; crossed with three LoRA init seeds by the trainer to
# give the 9 runs per task the Director's ruling fixed (PILOT_REPORT §6.1).
MINIBANK_DATA_SEEDS = (0, 1, 2)

# The trainer consumes only what its ~1M-token budget needs. Long-document
# tasks pack ~500 tokens/sequence so they need ~2k examples; agnews packs
# ~80 so it needs ~9k. One cap with margin for all six, and because a
# stream is a strict prefix of the permuted pool, shrinking the cap can
# only truncate — never reorder — what a run sees.
DEFAULT_MAX_EXAMPLES = 15_000


def make_formatter(task: str, tokenizer, max_len: int):
    """A registry dataset object carrying only what its formatter needs.

    ``__init__`` is deliberately bypassed (see module docstring): it would
    eagerly tokenize and pad every example. The returned object exposes the
    class's own ``format_example``.
    """
    cls = TASK_REGISTRY[task]
    obj = cls.__new__(cls)
    obj.tokenizer = tokenizer
    obj.max_len = max_len
    return obj


def emit_stream(task: str, data_seed: int, tokenizer, out_dir: Path,
                max_examples: int = DEFAULT_MAX_EXAMPLES) -> dict:
    """Write <out_dir>/<task>_seed<data_seed>.jsonl and return its metadata."""
    t0 = time.time()
    cls = TASK_REGISTRY[task]
    ds, source = load_hf_dataset_with_fallback(
        cls.dataset_candidates, cls.dataset_config_name, cls.hf_split)
    n_total = len(ds)

    val_ids, train_ids = split_ids(n_total, data_seed)
    take = [int(i) for i in train_ids[:max_examples]]
    selected = cls._select(ds, take)

    fmt = make_formatter(task, tokenizer, MAX_SEQ_LEN)
    fmt._post_load(selected)          # agnews resolves its label names here

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}_seed{data_seed}.jsonl"
    digest = hashlib.sha256()
    n = 0
    total_chars = 0
    # ensure_ascii=True is load-bearing, not cosmetic: raw U+2028/U+2029 in
    # the corpus survive json.dumps unescaped but ARE line breaks to Python's
    # str.splitlines(), which splits a record in half on read. Escaping every
    # non-ASCII codepoint guarantees one record == one line under any reader.
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for raw_id, item in zip(take, selected):
            text = fmt.format_example(item)
            line = json.dumps({"raw_id": raw_id, "text": text},
                              ensure_ascii=True)
            fh.write(line + "\n")
            digest.update(line.encode("utf-8"))
            total_chars += len(text)
            n += 1

    meta = {
        "task": task,
        "data_seed": data_seed,
        "dataset_source": source,
        "n_total_raw": n_total,
        "pool_size": int(len(train_ids)),
        "n_emitted": n,
        "val_seed": VAL_SEED,
        "val_size": VAL_SIZE,
        "pool_cap": POOL_CAP,
        "first_raw_ids": take[:5],
        "sha256": digest.hexdigest(),
        "path": path.as_posix(),
        "mean_chars": total_chars / max(n, 1),
        "emit_seconds": round(time.time() - t0, 1),
    }
    print(f"[data] {task} data_seed={data_seed}: {n} examples from {source} "
          f"(pool {meta['pool_size']}, mean {meta['mean_chars']:.0f} chars, "
          f"{meta['emit_seconds']:.0f}s) -> {path.name}", flush=True)
    return meta


def verify_against_pilot(out_dir: Path) -> int:
    """Byte-compare the three shared streams against the pilot's emitter.

    The pilot inlined its own copies of the three tokenizer-free templates;
    this script delegates to the registry classes instead. Identical bytes
    on (alpaca, math, agnews) x (seed 0, 1) prove the delegation introduced
    no drift. Returns the number of mismatches.
    """
    pilot_dir = REPO_ROOT / "results" / "tinker-pilot" / "data"
    bad = 0
    for task in ("alpaca", "math", "agnews"):
        for seed in (0, 1):
            name = f"{task}_seed{seed}.jsonl"
            a, b = pilot_dir / name, out_dir / name
            if not a.exists() or not b.exists():
                print(f"[verify] SKIP {name} (missing {'pilot' if not a.exists() else 'minibank'})")
                continue
            ha = hashlib.sha256(a.read_bytes()).hexdigest()
            hb = hashlib.sha256(b.read_bytes()).hexdigest()
            ok = ha == hb
            bad += (not ok)
            print(f"[verify] {name:24s} {'MATCH' if ok else 'MISMATCH'}  {ha[:16]}")
    print(f"[verify] {bad} mismatch(es) vs the pilot emitter")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description="Tinker mini-bank text emitter")
    ap.add_argument("--tasks", nargs="+", default=list(MINIBANK_TASKS),
                    choices=list(MINIBANK_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(MINIBANK_DATA_SEEDS))
    ap.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "results" / "tinker-minibank" / "data")
    ap.add_argument("--verify-against-pilot", action="store_true",
                    help="byte-compare the three shared streams and exit")
    args = ap.parse_args()

    if args.verify_against_pilot:
        raise SystemExit(1 if verify_against_pilot(args.out_dir) else 0)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"[data] tokenizer {BASE_MODEL} (document budgeting for xsum/squad)")

    metas = [emit_stream(t, s, tokenizer, args.out_dir, args.max_examples)
             for t in args.tasks for s in args.seeds]

    manifest = args.out_dir / "data_manifest.json"
    manifest.write_text(
        json.dumps({"base_model": BASE_MODEL, "max_seq_len": MAX_SEQ_LEN,
                    "streams": metas}, indent=2) + "\n", encoding="utf-8")
    print(f"\n[data] {len(metas)} streams; manifest -> {manifest}")


if __name__ == "__main__":
    main()
