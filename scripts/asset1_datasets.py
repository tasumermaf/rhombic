"""Asset 1 — Dataset classes for the locked 480-adapter bank campaign.

Six tasks, one instruction format (the pilot's ``### Instruction: /
### Response:`` frame from ``scripts/train_task_fingerprint.py`` and
``scripts/train_exp2_scale.py``):

    alpaca  — instruction-follow   yahma/alpaca-cleaned
    code    — code                 sahil2801/CodeAlpaca-20k
    math    — math                 openai/gsm8k ("main")
    xsum    — summarization        EdinburghNLP/xsum (parquet; fallback "xsum")
    squad   — extractive QA        rajpurkar/squad (SQuAD v1.1; fallback "squad")
    agnews  — classification       fancyzhx/ag_news (fallback "ag_news")

Fixed-val / train-pool design (locked card, CRITICAL):
    For each task the raw HF *train* split is put through a single fixed
    canonical shuffle: ``numpy.random.RandomState(VAL_SEED=777).permutation``.
    The first VAL_SIZE (500) canonical examples are the validation split —
    deterministic and IDENTICAL across every run of that task (data_seed has
    no effect on it). The next POOL_CAP canonical examples form the training
    pool, whose composition is likewise fixed per task; only its ORDER is
    shuffled per run by ``RandomState(data_seed)``. Runs are therefore
    comparable on the same val set while remaining independent in data order.

    POOL_CAP (default 40,000) bounds eager-tokenization cost. A 2,000-step
    run at batch 2 x grad-accum 8 consumes 32,000 examples, so a 40,000
    pool preserves full diversity of what a run can see; tasks smaller than
    the cap (GSM8K, CodeAlpaca) use their whole remainder.

Pilot conventions preserved exactly (regime match):
    - tokenizer(prompt, truncation=True, max_length=max_len,
      padding="max_length", return_tensors="pt")
    - labels = input_ids.clone()  (loss over the full padded sequence —
      the pilot did NOT mask prompt or padding tokens; kept for
      comparability with the pilot bank)

New for the long-document tasks (xsum, squad): the document/context field
is token-budget truncated so the target (summary / answer) always survives
inside the max_len window. Naive right-truncation (the pilot approach on
short tasks) would delete the supervision target for most XSum articles.
This is a documented, auditable deviation confined to the two new
long-document tasks.

Existing pilot classes (AlpacaDataset / CodeAlpacaDataset / GSM8KDataset)
are not importable as-is for this campaign because they lack the fixed-val /
train-pool split design; their prompt templates are replicated verbatim in
PROMPT_TEMPLATES below.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Locked split constants ──────────────────────────────────────────

VAL_SEED = 777      # canonical shuffle seed — NEVER change (locked card)
VAL_SIZE = 500      # fixed validation examples per task
POOL_CAP = 40_000   # training-pool cap (composition fixed per task)


# ── Prompt templates (single auditable dict — locked card requirement) ──
#
# "alpaca"/"code" templates are verbatim from scripts/train_exp2_scale.py
# (AlpacaDataset) and scripts/train_task_fingerprint.py (CodeAlpacaDataset).
# "math" is verbatim from scripts/train_task_fingerprint.py (GSM8KDataset).
# The three new tasks use the same ### frame.

PROMPT_TEMPLATES: dict[str, str] = {
    "alpaca_with_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
        "### Response:\n{output}"
    ),
    "alpaca_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    ),
    "code_with_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
        "### Response:\n{output}"
    ),
    "code_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    ),
    "math": (
        "### Instruction:\nSolve the following math problem step by step."
        "\n\n{question}\n\n### Response:\n{answer}"
    ),
    "xsum": (
        "### Instruction:\nSummarize the following article.\n\n{document}"
        "\n\n### Response:\n{summary}"
    ),
    "squad": (
        "### Instruction:\nAnswer the question using the passage.\n\n"
        "Passage: {context}\n\nQuestion: {question}\n\n### Response:\n{answer}"
    ),
    "agnews": (
        "### Instruction:\nClassify the following news article into one of: "
        "World, Sports, Business, Sci/Tech.\n\n{text}\n\n### Response:\n{label}"
    ),
}

# Map task key -> template keys used (for audit/tests)
TASK_TEMPLATE_KEYS: dict[str, list[str]] = {
    "alpaca": ["alpaca_with_input", "alpaca_no_input"],
    "code": ["code_with_input", "code_no_input"],
    "math": ["math"],
    "xsum": ["xsum"],
    "squad": ["squad"],
    "agnews": ["agnews"],
}


# ── Split logic (pure, testable) ────────────────────────────────────


def split_ids(
    n_total: int,
    data_seed: int,
    val_size: int = VAL_SIZE,
    val_seed: int = VAL_SEED,
    pool_cap: int = POOL_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (val_ids, train_ids) over raw dataset row indices.

    val_ids   : first `val_size` indices of the canonical (val_seed) shuffle.
                Independent of data_seed — identical across all runs.
    train_ids : the next `pool_cap` canonical indices (fixed composition),
                re-ordered by a RandomState(data_seed) permutation.

    numpy's legacy RandomState is version-stable, so these splits are
    reproducible across numpy releases.
    """
    if n_total <= val_size:
        raise ValueError(
            f"dataset too small: n_total={n_total} <= val_size={val_size}"
        )
    canonical = np.random.RandomState(val_seed).permutation(n_total)
    val_ids = canonical[:val_size]
    pool = canonical[val_size: val_size + pool_cap]
    order = np.random.RandomState(data_seed).permutation(len(pool))
    train_ids = pool[order]
    return val_ids, train_ids


def ids_sha256(ids) -> str:
    """Stable hash of an index sequence — recorded in config.json so any
    two runs of a task can be audited for identical val membership."""
    payload = json.dumps([int(i) for i in ids]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_hf_dataset_with_fallback(candidates: list[str], config_name, split: str):
    """Try each HF dataset id in order; return (dataset, source_id_used).

    Which mirror actually loaded is recorded by the caller into config.json.
    """
    from datasets import load_dataset

    errors = []
    for repo in candidates:
        try:
            if config_name is not None:
                ds = load_dataset(repo, config_name, split=split)
            else:
                ds = load_dataset(repo, split=split)
            return ds, repo
        except Exception as e:  # try next mirror
            errors.append(f"{repo}: {type(e).__name__}: {e}")
    raise RuntimeError(
        "All dataset sources failed:\n  " + "\n  ".join(errors)
    )


# ── Base class ──────────────────────────────────────────────────────


class Asset1TaskDataset(Dataset):
    """Base class implementing the fixed-val / train-pool design.

    Parameters
    ----------
    tokenizer : HF tokenizer (Qwen or Llama; pad_token must be set by caller,
        matching the pilot trainer convention).
    split_pool : "train" or "val".
    data_seed : per-run data-shuffle seed. Affects ONLY the train pool order;
        the val split is invariant to it by construction.
    max_len : max sequence length (pilot convention: 512).
    val_size / val_seed / pool_cap : split constants (locked defaults).
    raw : optional injected list[dict] of raw examples (for tests — bypasses
        the HF download).
    keep_text : if True, retain the formatted prompt strings in
        ``self.formatted_texts`` (used by --dry-run for eyeball checks).
    """

    # Subclasses set these:
    task_key: str = ""
    dataset_candidates: list[str] = []
    dataset_config_name = None
    hf_split: str = "train"

    def __init__(
        self,
        tokenizer,
        split_pool: str,
        data_seed: int,
        max_len: int = 512,
        val_size: int = VAL_SIZE,
        val_seed: int = VAL_SEED,
        pool_cap: int = POOL_CAP,
        raw=None,
        keep_text: bool = False,
    ):
        if split_pool not in ("train", "val"):
            raise ValueError(f"split_pool must be 'train' or 'val', got {split_pool!r}")

        self.tokenizer = tokenizer
        self.split_pool = split_pool
        self.data_seed = data_seed
        self.max_len = max_len
        self.val_size = val_size
        self.val_seed = val_seed
        self.pool_cap = pool_cap

        if raw is not None:
            ds, source = raw, "<injected>"
        else:
            ds, source = load_hf_dataset_with_fallback(
                self.dataset_candidates, self.dataset_config_name, self.hf_split
            )
        self.dataset_source = source
        self.n_total_raw = len(ds)

        val_ids, train_ids = split_ids(
            self.n_total_raw, data_seed,
            val_size=val_size, val_seed=val_seed, pool_cap=pool_cap,
        )
        self.val_ids = val_ids
        self.pool_size = len(train_ids)
        ids = val_ids if split_pool == "val" else train_ids
        # Canonical raw-row indices of this split, in serving order:
        self.example_ids = [int(i) for i in ids]

        selected = self._select(ds, self.example_ids)

        self.formatted_texts = [] if keep_text else None
        self.examples = []
        self._post_load(selected)
        for item in selected:
            text = self.format_example(item)
            if self.formatted_texts is not None:
                self.formatted_texts.append(text)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                # Pilot convention: loss over the full padded sequence.
                "labels": encoded["input_ids"].squeeze(0).clone(),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]

    # -- hooks -------------------------------------------------------

    def _post_load(self, selected) -> None:
        """Optional subclass hook run before formatting (e.g. label names)."""

    def format_example(self, item: dict) -> str:
        raise NotImplementedError

    # -- helpers -----------------------------------------------------

    @staticmethod
    def _select(ds, ids: list[int]):
        if hasattr(ds, "select"):
            return ds.select(ids)
        return [ds[i] for i in ids]

    def _fit_document(self, doc: str, fixed_text: str, margin: int = 8) -> str:
        """Token-budget truncate `doc` so the full prompt (fixed scaffold +
        target + doc) fits within max_len, preserving the supervision target.

        fixed_text : the prompt with the document field left EMPTY — its token
        count is the non-document budget.
        """
        tok = self.tokenizer
        fixed_len = len(tok(fixed_text, add_special_tokens=True)["input_ids"])
        budget = self.max_len - fixed_len - margin
        if budget < 32:
            budget = 32  # degenerate safety floor
        doc_ids = tok(
            doc, add_special_tokens=False, truncation=True, max_length=budget,
        )["input_ids"]
        return tok.decode(doc_ids, skip_special_tokens=True)


# ── Task datasets ───────────────────────────────────────────────────


class AlpacaTaskDataset(Asset1TaskDataset):
    """instruction-follow — yahma/alpaca-cleaned (pilot template, verbatim)."""

    task_key = "alpaca"
    dataset_candidates = ["yahma/alpaca-cleaned"]

    def format_example(self, item: dict) -> str:
        instruction = item["instruction"]
        inp = item.get("input", "")
        output = item["output"]
        if inp:
            return PROMPT_TEMPLATES["alpaca_with_input"].format(
                instruction=instruction, input=inp, output=output)
        return PROMPT_TEMPLATES["alpaca_no_input"].format(
            instruction=instruction, output=output)


class CodeAlpacaTaskDataset(Asset1TaskDataset):
    """code — sahil2801/CodeAlpaca-20k (pilot template, verbatim)."""

    task_key = "code"
    dataset_candidates = ["sahil2801/CodeAlpaca-20k"]

    def format_example(self, item: dict) -> str:
        instruction = item["instruction"]
        inp = item.get("input", "")
        output = item["output"]
        if inp:
            return PROMPT_TEMPLATES["code_with_input"].format(
                instruction=instruction, input=inp, output=output)
        return PROMPT_TEMPLATES["code_no_input"].format(
            instruction=instruction, output=output)


class GSM8KTaskDataset(Asset1TaskDataset):
    """math — openai/gsm8k "main" (pilot template, verbatim).

    GSM8K train has 7,473 rows: pool = 6,973 after the fixed val split,
    so a 2,000-step run cycles ~4.6 epochs (same regime as the pilot note).
    """

    task_key = "math"
    dataset_candidates = ["openai/gsm8k"]
    dataset_config_name = "main"

    def format_example(self, item: dict) -> str:
        return PROMPT_TEMPLATES["math"].format(
            question=item["question"], answer=item["answer"])


class XSumTaskDataset(Asset1TaskDataset):
    """summarization — EdinburghNLP/xsum.

    Source note: the canonical "EdinburghNLP/xsum" repo now ships parquet
    data files (verified 2026-07-03), so it loads under datasets>=4 without
    a dataset script. "xsum" is kept as a fallback alias. The source that
    actually loaded is recorded in ``self.dataset_source`` -> config.json.

    Documents are token-budget truncated (see _fit_document) so the summary
    is always present in the loss window.
    """

    task_key = "xsum"
    dataset_candidates = ["EdinburghNLP/xsum", "xsum"]

    def format_example(self, item: dict) -> str:
        summary = item["summary"]
        fixed = PROMPT_TEMPLATES["xsum"].format(document="", summary=summary)
        doc = self._fit_document(item["document"], fixed)
        return PROMPT_TEMPLATES["xsum"].format(document=doc, summary=summary)


class SquadTaskDataset(Asset1TaskDataset):
    """extractive QA — SQuAD v1.1 (rajpurkar/squad).

    The first gold answer is the supervision target. Context is token-budget
    truncated so question + answer always survive in-window.
    """

    task_key = "squad"
    dataset_candidates = ["rajpurkar/squad", "squad"]

    def format_example(self, item: dict) -> str:
        question = item["question"]
        answers = item["answers"]["text"]
        answer = answers[0] if len(answers) > 0 else ""
        fixed = PROMPT_TEMPLATES["squad"].format(
            context="", question=question, answer=answer)
        context = self._fit_document(item["context"], fixed)
        return PROMPT_TEMPLATES["squad"].format(
            context=context, question=question, answer=answer)


class AGNewsTaskDataset(Asset1TaskDataset):
    """classification — AG News (fancyzhx/ag_news), 4 classes."""

    task_key = "agnews"
    dataset_candidates = ["fancyzhx/ag_news", "ag_news"]

    _FALLBACK_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

    def _post_load(self, selected) -> None:
        try:
            self.label_names = list(selected.features["label"].names)
        except Exception:
            self.label_names = list(self._FALLBACK_LABELS)

    def format_example(self, item: dict) -> str:
        label = item["label"]
        if isinstance(label, int):
            label = self.label_names[label]
        return PROMPT_TEMPLATES["agnews"].format(text=item["text"], label=label)


# ── Registry / factory ──────────────────────────────────────────────

TASK_REGISTRY: dict[str, type[Asset1TaskDataset]] = {
    "alpaca": AlpacaTaskDataset,
    "code": CodeAlpacaTaskDataset,
    "math": GSM8KTaskDataset,
    "xsum": XSumTaskDataset,
    "squad": SquadTaskDataset,
    "agnews": AGNewsTaskDataset,
}

TASK_LONG_NAMES: dict[str, str] = {
    "alpaca": "instruction-follow",
    "code": "code",
    "math": "math",
    "xsum": "summarization",
    "squad": "extractive-QA",
    "agnews": "classification",
}


def build_dataset(
    task: str,
    tokenizer,
    split_pool: str,
    data_seed: int,
    max_len: int = 512,
    pool_cap: int = POOL_CAP,
    keep_text: bool = False,
) -> Asset1TaskDataset:
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_REGISTRY)}")
    cls = TASK_REGISTRY[task]
    return cls(
        tokenizer, split_pool, data_seed, max_len,
        pool_cap=pool_cap, keep_text=keep_text,
    )
