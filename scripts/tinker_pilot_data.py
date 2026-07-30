"""Tinker signal pilot (E-T4 pre-step) — training-text generation.

Emits the per-(task, seed) training text streams for the Tinker pilot,
reusing the LOCKED Asset-1 pool/template/split logic verbatim from
``asset1_datasets.py``:

    * ``split_ids(n_total, data_seed)`` — the canonical VAL_SEED=777 shuffle,
      first VAL_SIZE=500 canonical rows held out as the fixed val split, the
      next POOL_CAP=40,000 forming the train pool whose composition is fixed
      per task and whose ORDER is permuted by ``RandomState(data_seed)``.
    * ``PROMPT_TEMPLATES`` — the same ``### Instruction: / ### Response:``
      frames used by the 480-run Asset-1 bank.

Why a separate text-emission step: the Asset-1 dataset classes tokenize
eagerly with a local HF tokenizer, but Tinker tokenizes with the tokenizer
of the SERVED base model (Qwen/Qwen3-8B) inside the ``tinker`` conda env,
which has no torch. The three pilot tasks (alpaca, math, agnews) format
WITHOUT a tokenizer — only the long-document tasks (xsum, squad) need one
for ``_fit_document`` budget truncation — so this script runs under the
``falco`` env and hands plain text to the trainer. Padding is deliberately
NOT applied: Tinker has no fixed-shape requirement and padded positions
would be billed as train tokens.

Deterministic: same (task, seed) always yields byte-identical output.
Regenerable, so the emitted .jsonl files are gitignored, not committed.

Usage
-----
    python scripts/tinker_pilot_data.py                     # all 6 streams
    python scripts/tinker_pilot_data.py --tasks agnews --seeds 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asset1_datasets import (  # noqa: E402  (locked Asset-1 machinery)
    PROMPT_TEMPLATES, VAL_SEED, VAL_SIZE, POOL_CAP,
    load_hf_dataset_with_fallback, split_ids)

# Pilot task set — the three Asset-1 tasks that need no tokenizer to format.
PILOT_TASKS = ("alpaca", "math", "agnews")
PILOT_SEEDS = (0, 1)

# Examples emitted per stream. The trainer consumes only as many as its token
# budget needs (~1M tokens ~= 5.6k alpaca / 4.4k math / 9.1k agnews examples);
# this cap leaves margin without emitting the whole 40k pool.
DEFAULT_MAX_EXAMPLES = 15_000

_SOURCES = {
    "alpaca": (["yahma/alpaca-cleaned"], None),
    "math": (["openai/gsm8k"], "main"),
    "agnews": (["fancyzhx/ag_news", "ag_news"], None),
}

_AGNEWS_FALLBACK_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def format_example(task: str, item: dict, label_names: list[str] | None) -> str:
    """Format one raw row exactly as the Asset-1 dataset classes do."""
    if task == "alpaca":
        inp = item.get("input", "")
        if inp:
            return PROMPT_TEMPLATES["alpaca_with_input"].format(
                instruction=item["instruction"], input=inp, output=item["output"])
        return PROMPT_TEMPLATES["alpaca_no_input"].format(
            instruction=item["instruction"], output=item["output"])
    if task == "math":
        return PROMPT_TEMPLATES["math"].format(
            question=item["question"], answer=item["answer"])
    if task == "agnews":
        label = item["label"]
        if isinstance(label, int):
            names = label_names or _AGNEWS_FALLBACK_LABELS
            label = names[label]
        return PROMPT_TEMPLATES["agnews"].format(text=item["text"], label=label)
    raise ValueError(f"unsupported pilot task {task!r}")


def emit_stream(task: str, seed: int, out_dir: Path,
                max_examples: int = DEFAULT_MAX_EXAMPLES) -> dict:
    """Write results/tinker-pilot/data/<task>_seed<seed>.jsonl and return meta."""
    candidates, config_name = _SOURCES[task]
    ds, source = load_hf_dataset_with_fallback(candidates, config_name, "train")
    n_total = len(ds)

    val_ids, train_ids = split_ids(n_total, seed)
    take = [int(i) for i in train_ids[:max_examples]]

    label_names = None
    if task == "agnews":
        try:
            label_names = list(ds.features["label"].names)
        except Exception:
            label_names = list(_AGNEWS_FALLBACK_LABELS)

    selected = ds.select(take) if hasattr(ds, "select") else [ds[i] for i in take]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}_seed{seed}.jsonl"
    digest = hashlib.sha256()
    n = 0
    total_chars = 0
    # ensure_ascii=True is load-bearing, not cosmetic: raw U+2028/U+2029 in the
    # corpus survive json.dumps unescaped but ARE line breaks to Python's
    # str.splitlines(), which splits a record in half on read. Escaping every
    # non-ASCII codepoint guarantees one record == one line under any reader.
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for raw_id, item in zip(take, selected):
            text = format_example(task, item, label_names)
            line = json.dumps({"raw_id": raw_id, "text": text},
                              ensure_ascii=True)
            fh.write(line + "\n")
            digest.update(line.encode("utf-8"))
            total_chars += len(text)
            n += 1

    meta = {
        "task": task,
        "data_seed": seed,
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
    }
    print(f"[data] {task} seed={seed}: {n} examples from {source} "
          f"(pool {meta['pool_size']}, mean {meta['mean_chars']:.0f} chars) "
          f"-> {path.name}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Tinker pilot training-text emitter")
    ap.add_argument("--tasks", nargs="+", default=list(PILOT_TASKS),
                    choices=list(PILOT_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PILOT_SEEDS))
    ap.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "results" / "tinker-pilot" / "data")
    args = ap.parse_args()

    metas = [emit_stream(t, s, args.out_dir, args.max_examples)
             for t in args.tasks for s in args.seeds]

    manifest = args.out_dir / "data_manifest.json"
    manifest.write_text(json.dumps({"streams": metas}, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n[data] manifest -> {manifest}")


if __name__ == "__main__":
    main()
