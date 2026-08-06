# -*- coding: utf-8 -*-
"""Granularity ladder — Stage-0 label-space derivation (the frozen manifests).

REGISTERED CARD: `docs/REGISTRATION_GRANULARITY_2026-07-30.md`
    = `docs/DESIGN_GRANULARITY_BRIDGE_2026-07-21.md` (design)
    + `docs/DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md` Part 5
      (rulings — these govern wherever the two conflict),
    as conditioned by `docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md`
    Ask 3 (D6 subsample) / Ask 4 (cost), and LOCKED by
    `docs/LOCK_GRANULARITY_2026-08-04.md`.

WHAT THIS SCRIPT IS: the design's §10 Stage 0, item "build + freeze
    subdivision manifests (rules, tiers, realized K, per-class pools)". It is
    CPU-only, deterministic, seeded, and touches no GPU and no adapter. It
    reads the six task datasets through `asset1_datasets` (the same loader,
    the same locked val_seed=777 / VAL_SIZE=500 / POOL_CAP=40,000 split
    design) and writes, per level:

      results/granularity/labels/<level>.json         manifest (reviewable)
      results/granularity/labels/<level>_pools.json   per-class row ids
      results/granularity/labels/LABELS_REPORT.md     typed report

WHAT IT DOES NOT DO: it does not train, does not analyze, does not guess.
    Where the registered card underdetermines a choice, the choice is
    recorded in the AMBIGUITIES register below with its resolution, its
    alternatives, and who must rule — and the ambiguity travels into every
    emitted JSON and into the report. Ambiguities are findings, not guesses
    (hub instruction, 2026-08-04).

HOW A CLASS POOL IS DERIVED (identical machinery to the bank, no fork):
    1. base pool: `asset1_datasets.split_ids(n_total_raw, data_seed=0)` gives
       the locked val (first 500 of the canonical val_seed=777 shuffle) and
       the 40,000-row training pool. Class labels are assigned over the
       TRAINING POOL only — the locked val rows never enter any class.
       (Design §3: "pools = post-val, POOL_CAP 40,000".)
    2. class rows: the pool rows whose label rule fires, listed in ASCENDING
       RAW-ROW-INDEX order — a frozen, numpy-version-independent order.
    3. balancing (see G-1): the class row list is reduced to
       500 + n_balanced rows by a seeded choice, so that
    4. the run-time split falls out of the SAME pure function the bank uses:
       `split_ids(len(class_rows))` -> per-class fixed 500 val + the
       class training pool, order shuffled per run by data_seed. The runner
       hands `ds.select(class_rows)` to the unmodified task dataset class as
       its `raw=` argument; nothing in `asset1_datasets` is edited.

    D4 pool floor (APPROVED): a class is eligible only if its TRAINING pool
    (after the 500-row per-class val) is >= 1,000 examples.

    D6 (PINNED, Ask 3): a 1,000-example-per-class subsample of the class
    TRAINING pool is frozen here and recorded per class, so the data-space
    reference is drawn from the population the adapters trained on, is the
    same nominal n at every level, and — where a ragged L3 class has fewer
    than 1,000 — is the whole pool with its realized n reported.

Usage:
    python scripts/granularity_labels.py                 # all levels
    python scripts/granularity_labels.py --levels L1
    python scripts/granularity_labels.py --balance task  # alternative G-1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The campaign's own caches (the bank and the S2 pilots run from these).
os.environ.setdefault("HF_HUB_CACHE", r"C:\falco\hf-cache\hub")
os.environ.setdefault("HF_DATASETS_CACHE", r"C:\falco\hf-cache\datasets")

from asset1_bank import (  # noqa: E402  (constants + helpers, never redeclared)
    BATCH_SIZE,
    GRAD_ACCUM,
    MAX_STEPS,
    git_commit_hash,
    utc_now,
)
from asset1_datasets import (  # noqa: E402
    POOL_CAP,
    TASK_REGISTRY,
    VAL_SEED,
    VAL_SIZE,
    load_hf_dataset_with_fallback,
    split_ids,
)

# ── Locked / pinned constants ───────────────────────────────────────

OUT_ROOT = REPO_ROOT / "results" / "granularity"
LABELS_DIR = OUT_ROOT / "labels"

D4_POOL_FLOOR = 1_000            # D4 APPROVED — training pool per class
D6_SUBSAMPLE_N = 1_000           # D6 PINNED (Ask 3) — same nominal n per level
D6_SUBSAMPLE_SEED = 6            # this script's own frozen stream (D6)
BALANCE_SEED = 4                 # this script's own frozen stream (D4/G-1)

# Training sequences per run, held identical at every level (design §4):
SEQUENCES_PER_RUN = MAX_STEPS * BATCH_SIZE * GRAD_ACCUM      # 2000*4*4 = 32,000

TIER_T1, TIER_T2, TIER_T3 = "T1", "T2", "T3"
CLEAN_CORE_TIERS = (TIER_T1, TIER_T2)     # D3 RESTRICTION: the clean core

# Analysis tier order, frozen at lock (LOCK §"Frozen at lock").
TIER_ORDER = ["L0", "L1", "ARMB", "L2", "L3", "D7"]

# Levels this script emits. ARMB is emitted as its four rungs.
ARMB_RUNGS = ["B2", "B4", "B8", "B16"]
EMITTED_LEVELS = ["L0", "L1"] + ARMB_RUNGS + ["L2", "L3", "D7"]

# Seeds per class, per level (D2 APPROVED 12/24/~48 at 20/10/5).
LEVEL_SEEDS = {"L0": 40, "L1": 20, "L2": 10, "L3": 5,
               "B2": 16, "B4": 8, "B8": 4, "B16": 3,     # G-6
               "D7": 5}                                   # D7: 5 + 5

TASKS = ["alpaca", "code", "math", "xsum", "squad", "agnews"]


# ── The ambiguity register (G-n) — findings, not guesses ────────────

AMBIGUITIES: list[dict] = [
    {
        "id": "G-1",
        "status": "RULED 2026-08-04 (ratified)",
        "ruling": "RULED --balance none (ratified) — three grounds: the card's own §3 arithmetic, L0 anchor comparability, and confound control.",
        "where": "design §4 `splits` vs design §3 \"Pool survival\" — the two "
                 "sections of the registered card disagree",
        "question": "§4 says \"pools subsampled to common per-class n within "
                    "each level (discards logged)\". §3 states the realized "
                    "pools as \"L1 ~3.5k (math) – 24k (alpaca)\". Those "
                    "cannot both hold: a common per-class n is ONE number "
                    "per level, not a 7x range.",
        "resolution": "--balance none (default). MEASURED, not argued: this "
                      "script computes L1's per-class training pools as "
                      "2,628 (math steps_ge4) – 24,788 (alpaca no_input), "
                      "which reproduces §3's stated \"~3.5k – 24k\" exactly; "
                      "no other policy does (--balance task gives 2,628 – "
                      "14,212, --balance level gives 2,628 everywhere). §3 "
                      "therefore describes an UNBALANCED build, and §4's "
                      "sentence is the part of the card that does not fit "
                      "its own arithmetic. Second, independent reason: L0's "
                      "240 adapters already exist and were trained on "
                      "unbalanced pools (6,973 – 40,000). Balancing L1 would "
                      "move TWO variables between the curve's first two "
                      "points — K and pool policy — and level-wide balancing "
                      "would additionally raise every L1 class from ~1.3 "
                      "epochs to 12.18, inflating the very memorization "
                      "confound §9.2 names and D7 exists to test.",
        "alternatives": ["--balance level (the literal §4 sentence; discards "
                         "82.5% of L1's training rows)",
                         "--balance task (within-task common n; discards "
                         "6.6% at L1)"],
        "consequence": "Under `none`, per-class pool size varies WITHIN a "
                       "level, so epochs-per-pool vary within a level "
                       "(reported per class, as §9.2 requires). Realized n "
                       "and discarded mass under all three policies are "
                       "computed and reported for every level, so the choice "
                       "is a one-flag re-run.",
        "authority": "hub / Director",
    },
    {
        "id": "G-2",
        "status": "APPROVED 2026-08-05 — two footnotes ADOPTED as pre-declared",
        "ruling": "APPROVED 2026-08-05 (docs/DIRECTOR_RULINGS_G5_G2_2026-08-05.md Ruling 2): dicts machine-compared verbatim against the code; strict-partition and no-duplicate-keyword properties confirmed; zero discard upheld as the correct asymmetry against G-3. Two footnotes ADOPTED as pre-declared (see the G-2 footnotes section of this report). No edit requested.",
        "where": "design §3 table, alpaca L2/L3: \"4 / 8 instruction-verb "
                 "families [T2 keyword rules]\"",
        "question": "The card names the tier and the count but not the "
                    "rules: which verb families, and which verbs in each.",
        "resolution": "Meridian-authored frozen taxonomy (ALPACA_VERB_L3, 8 "
                      "families over the instruction's first word, with a "
                      "residual `other`; L2 = the frozen pairwise merge "
                      "ALPACA_VERB_L2_MERGE, so L2 is a strict "
                      "coarsening of L3).",
        "alternatives": ["a Director-supplied taxonomy",
                         "a data-driven clustering (would make the cells T3)"],
        "consequence": "This is an authored label space inside a registered "
                       "card. It should be reviewed before the L2/L3 tiers "
                       "fire; it does not affect L1 (alpaca L1 is native).",
        "authority": "hub / Director",
    },
    {
        "id": "G-3",
        "status": "RULED 2026-08-04 as built",
        "ruling": "RULED as built — four named languages, residual discarded and logged. No effect on L1.",
        "where": "design §3 table, code L2/L3: \"output language {Python / "
                 "non-Python / …} [T2; skew audit]\"",
        "question": "Which four language classes, and what happens to the "
                    "mass that matches none of them (measured: 43% of the "
                    "CodeAlpaca pool matches no language marker).",
        "resolution": "four NAMED languages (sql, html_css, javascript, "
                      "python) under a frozen ordered-pattern rule; the "
                      "residual is DISCARDED and its mass logged.",
        "alternatives": ["make the residual a 4th class (`other`) — zero "
                         "discard, but the class's definition is "
                         "'everything else', which the D6 data-space "
                         "reference would correctly read as inseparable"],
        "consequence": "The discarded mass is logged per level. `sql` is the "
                       "binding class for code's floor — its realized n is "
                       "reported.",
        "authority": "hub / Director",
    },
    {
        "id": "G-4",
        "status": "RULED 2026-08-04 (ratified)",
        "ruling": "RULED doc-length median [T2] (ratified) — L1 is then 12/12 clean-core and annotator-free.",
        "where": "design §3 table, xsum L1: \"doc-length median [T2, weak] "
                 "or 2 topics [T3]\"",
        "question": "The card offers two options and does not choose.",
        "resolution": "doc-length median [T2]. L1 is then 12/12 clean-core, "
                      "needs no annotator, and can launch immediately — "
                      "which matters because L1 is the only funded tier "
                      "after L0.",
        "alternatives": ["2 model-annotated topics [T3] — would make L1's "
                         "clean-core curve 10/12 classes"],
        "consequence": "The L1 all-classes and clean-core curves coincide "
                       "exactly, so the D3 divergence rule has nothing to "
                       "bite on at L1. Stated in the L1 readout.",
        "authority": "hub / Director",
    },
    {
        "id": "G-5",
        "status": "RESOLVED 2026-08-05 by accepted dated amendment — annotator retired, T3 -> T2",
        "ruling": "ACCEPTED 2026-08-05 (docs/DIRECTOR_RULINGS_G5_G2_2026-08-05.md Ruling 1): annotator retired, cells move T3 -> T2, ladder clean-core end to end. Conditions (a) six freeze artifacts in-tree before any re-emit, (b) axis named 'editorial section', (c) 'no T3 cells - not testable' never 'passed', (d) spread quoted as 5.59x realized.",
        "where": "design §3 table, xsum L2 (4 topics) and L3 (12 topics), "
                 "both [T3 model-annotated]",
        "question": "The card admits T3 cells (D3 RESTRICTED) but pins NO "
                    "annotator: no model, no prompt, no topic inventory, no "
                    "frozen assignment procedure exists in the record.",
        "resolution": "[HISTORICAL — superseded 2026-08-05] STOP: xsum's "
                      "L2/L3 classes were declared but NOT materialized "
                      "(status pending_t3_annotation, no row ids), so L2/L3 "
                      "were not launchable. RESOLVED by the dated amendment "
                      "docs/AMENDMENT_G5_XSUM_SECTIONS_2026-08-05.md, "
                      "ACCEPTED by the Director 2026-08-05: no annotator is "
                      "needed. The HF `id` field is the BBC article id, and "
                      "the hash-pinned upstream XSum URL table carries the "
                      "BBC EDITORIAL SECTION in its path, so the cells are a "
                      "deterministic pure function of a native field plus a "
                      "frozen external table — [T2], not [T3]. 100% join "
                      "coverage over the locked 40,000-row pool; 13 authored "
                      "families all clearing the D4 floor; the two smallest "
                      "world buckets merge to land K=12 at L3 and a frozen "
                      "4-way coarsening at L2.",
        "alternatives": ["pin a model annotator (non-llama, batched "
                         "logit-scoring, committed assignment file as the "
                         "artifact) — now moot, and retired for cause: "
                         "control 6.1 IS TF-IDF + linear SVM, so any "
                         "TF-IDF-objective annotator would have D6 scoring "
                         "labels manufactured in D6's own feature space and "
                         "the label-noise bound would be vacuous by "
                         "construction",
                         "greedy bin-packing over raw sections — measured "
                         "and declined as semantically arbitrary (19-22 "
                         "unrelated sections per bin, brittle to "
                         "query-string artifacts)"],
        "consequence": "L2 realizes K=24 and L3 realizes K=48 exactly as "
                       "locked, both launchable, 240 adapters each. The "
                       "ENTIRE ladder is now clean-core (T1+T2 everywhere), "
                       "so D3's RESTRICTED clause has no T3 cells to bite "
                       "on at any level — reported as 'no T3 cells — not "
                       "testable', never 'passed' (Director condition (c)). "
                       "The axis is named EDITORIAL SECTION, never 'topic' "
                       "unqualified (condition (b)).",
        "authority": "hub / Director (dated amendment, L-006)",
    },
    {
        "id": "G-6",
        "status": "RULED 2026-08-04",
        "ruling": "RULED 16/8/4/3 at rungs 2/4/8/16 (= 144 runs exactly) — the card's 2->16 span wins over constant-N.",
        "where": "design §11 item 8 / D8 PROMOTED: \"squad-only deep ladder, "
                 "2->16 native title groups, ~144 runs\"",
        "question": "The TOTAL is fixed — the locked cost table gives Arm B "
                    "3.056 GPU-days, which at the measured 30.56 min/run is "
                    "exactly 144.0 runs — but the per-rung seed allocation "
                    "is unspecified.",
        "resolution": "seeds 16/8/4/3 at rungs 2/4/8/16 = 32+32+32+48 = 144 "
                      "runs exactly, preserving the card's stated 2->16 span "
                      "and its stated total.",
        "alternatives": ["rungs 4/8/16 at a constant N=48 per rung "
                         "(= 144 exactly, honours D2's constant-N rationale, "
                         "but drops the 2-group rung the card names)"],
        "consequence": "Classifier N is 32/32/32/48 rather than constant "
                       "across rungs, so Arm B's curve is not sample-size-"
                       "matched the way the main ladder's is. Reported.",
        "authority": "hub / Director",
    },
    {
        "id": "G-7",
        "status": "RULED 2026-08-04",
        "ruling": "RULED largest-pool T1 L2 class, halves from the FULL class pool.",
        "where": "design §6.2 / D7 MANDATORY: \"One L2 class: adapters on "
                 "two disjoint halves of the same pool (5 + 5 seeds)\"",
        "question": "Which L2 class; and are the halves halves of the FULL "
                    "class pool or of the level-balanced pool (under G-1's "
                    "level-wide balancing, half a balanced pool falls below "
                    "the D4 floor).",
        "resolution": "the largest-pool T1-tier L2 class (deterministic, "
                      "recorded); halves are taken from the class's FULL "
                      "training pool, so each half clears the D4 floor. D7 "
                      "is a control, not a level cell — it is not "
                      "level-balanced, and says so.",
        "alternatives": ["a Director-named class",
                         "a small-pool class (sharper memorization test, but "
                         "halves may fall below the floor)"],
        "consequence": "D7's two half-pools are ~half the size of the full "
                       "class pool and therefore see ~2x the epochs of the "
                       "unbalanced class. Reported with the control.",
        "authority": "hub / Director",
    },
    {
        "id": "G-8",
        "status": "RECORDED DERIVATION",
        "ruling": "RECORDED DERIVATION — no ruling required.",
        "where": "design §3 table L3 (xsum \"8–12 topics\", squad \"8–16 "
                 "title groups\") vs the lock's \"L3 ~48x5\"",
        "question": "The card gives ranges; the lock gives ~48 classes and "
                    "240 adapters at 5 seeds.",
        "resolution": "xsum 12 + squad 16, with alpaca 8, code 4, math 4, "
                      "agnews 4 -> K = 48 exactly, 48 x 5 = 240 adapters. "
                      "This is the only combination inside the card's stated "
                      "ranges that hits the lock's numbers exactly.",
        "alternatives": ["any other point in the ranges, at the cost of "
                         "K != 48 and adapters != 240"],
        "consequence": "L3's realized K depends on G-5 being resolved; "
                       "without xsum, L3 realizes K=36.",
        "authority": "recorded derivation (no ruling needed unless "
                     "contested)",
    },
    {
        "id": "G-9",
        "status": "RULED 2026-08-04 (post-val, the stricter reading)",
        "ruling": "RULED post-val (the stricter reading).",
        "where": "design §3 \"training pool >= 1,000 examples\" vs §4 "
                 "\"per-CLASS fixed 500-example val\"",
        "question": "Is the D4 floor measured before or after the per-class "
                    "val is removed.",
        "resolution": "after — the floor is read literally as a floor on the "
                      "TRAINING pool, so a class needs >= 1,500 pool rows to "
                      "be eligible.",
        "alternatives": ["floor on the pre-val class pool (a 1,000-row class "
                         "would train on 500 examples = 64 epochs)"],
        "consequence": "The stricter reading. Eligibility is reported with "
                       "both counts (n_class_rows and n_train_pool).",
        "authority": "recorded derivation",
    },
]

AMBIGUITY_IDS = [a["id"] for a in AMBIGUITIES]

# All nine were converted into dated rulings on 2026-08-04 (Meridian as
# decider under the PI's delegation). The Director grades that document
# like any other and may overturn any ruling before the tier it affects
# fires; until then these are the ruled basis, not assumptions.
RULINGS_DOC = ("docs/DECIDER_RULINGS_GRANULARITY_AMBIGUITIES_2026-08-04.md")


# ── Frozen label rules ──────────────────────────────────────────────
#
# Every rule below is a pure function of one example. Tiers per the design:
# T1 native (a dataset field), T2 deterministic (a frozen pure-function rule),
# T3 model-annotated. No T3 rule is executed by this script (G-5).

# alpaca — first word of the instruction, lowercased, non-alpha stripped.
ALPACA_VERB_L3: dict[str, tuple[str, ...]] = {
    "generate": ("generate", "create", "write", "construct", "compose",
                 "design", "develop", "make", "produce", "build", "come",
                 "outline", "draft", "formulate", "devise", "craft",
                 "invent", "draw"),
    "describe": ("describe", "explain", "discuss", "tell", "define",
                 "elaborate", "summarise", "narrate", "illustrate"),
    "transform": ("rewrite", "edit", "convert", "translate", "change",
                  "revise", "summarize", "paraphrase", "correct", "replace",
                  "modify", "transform", "reword", "shorten", "expand",
                  "reformat", "reorder", "simplify", "fix", "update"),
    "question": ("what", "how", "why", "which", "who", "when", "where",
                 "is", "are", "can", "does", "do", "should", "would"),
    "retrieve": ("name", "list", "identify", "find", "suggest", "give",
                 "provide", "select", "brainstorm", "recommend", "propose",
                 "search", "look", "state", "mention", "gather"),
    "analyze": ("classify", "categorize", "categorise", "compare",
                "analyze", "analyse", "evaluate", "determine", "calculate",
                "predict", "assess", "estimate", "rank", "rate", "sort",
                "arrange", "order", "group", "match", "detect", "check",
                "count", "measure", "solve", "compute"),
    "frame": ("given", "using", "use", "based", "take", "imagine",
              "consider", "assume", "suppose", "from", "in", "for", "with",
              "you", "add", "insert", "append", "include", "on", "as",
              "if", "the", "this", "a", "an"),
    "other": (),        # residual — every first word not listed above
}
ALPACA_VERB_L2_MERGE: dict[str, tuple[str, ...]] = {
    "produce_text": ("generate", "transform"),
    "explain_answer": ("describe", "question"),
    "retrieve_analyze": ("retrieve", "analyze"),
    "framed_other": ("frame", "other"),
}

# code — ordered pattern rule over (instruction + " " + output), lowercased.
# The FIRST pattern that matches wins; a row matching none is DISCARDED (G-3).
# Order is most-distinctive-first; the realized counts are the design's
# "skew audit" and are reported per level.
CODE_LANG_PATTERNS: tuple[tuple[str, str], ...] = (
    ("sql", r"\bsql\b|\bselect\b[\s\S]{0,200}?\bfrom\b|\binsert into\b|"
            r"\bcreate table\b|\bjoin\b[\s\S]{0,80}?\bon\b"),
    ("html_css", r"<html|<body|<div|<span|<table|<head|<p>|<a\s|\bcss\b|"
                 r"\bhtml\b|\bstylesheet\b"),
    ("javascript", r"\bjavascript\b|\bjquery\b|console\.log|document\."
                   r"getelement|\bnode\.js\b|=>|\bvar \b|\blet \b"),
    ("python", r"\bpython\b|\bdef \b|print\(|\bimport \b|\bnumpy\b|"
               r"\bpandas\b|\belif\b|\b__init__\b"),
)

# math — GSM8K solution steps = answer lines that are not the '#### x' line.
MATH_L3_BINS: tuple[tuple[str, int, int], ...] = (
    ("steps_le2", 0, 2),
    ("steps_3", 3, 3),
    ("steps_4", 4, 4),
    ("steps_ge5", 5, 10 ** 9),
)
MATH_L1_MERGE: dict[str, tuple[str, ...]] = {
    "steps_le3": ("steps_le2", "steps_3"),
    "steps_ge4": ("steps_4", "steps_ge5"),
}

# agnews — native integer labels (T1).
AGNEWS_NAMES = ("world", "sports", "business", "sci_tech")
AGNEWS_L1_MERGE: dict[str, tuple[str, ...]] = {
    "world_sports": ("world", "sports"),
    "business_scitech": ("business", "sci_tech"),
}

# squad — 442 native titles, greedily bin-packed by example count into 16
# balanced groups, then merged pairwise (largest with smallest) to 8, 4, 2.
# Every rung is therefore a strict coarsening of the one below it.
SQUAD_FINEST_K = 16

# xsum — L1 is the frozen median of document length IN CHARACTERS over the
# 40,000-row pool (computed here and recorded).
#
# L2/L3 are the EDITORIAL SECTION axis [T2], per the accepted G-5 amendment
# (docs/AMENDMENT_G5_XSUM_SECTIONS_2026-08-05.md; Director ACCEPTED
# 2026-08-05, docs/DIRECTOR_RULINGS_G5_G2_2026-08-05.md Ruling 1). The
# annotator is retired: the HF `id` field is the BBC article id, and the XSum
# authors' published URL table carries the BBC editorial section in its path.
#
# NAMING (Director condition b, binding): this axis is "editorial section",
# never "topic" unqualified. It is a PUBLICATION-STRUCTURE axis — geography
# cross-cuts subject on the BBC's desks. Measured counter-example: a
# footballer's driving ban is filed under uk-scotland-glasgow-west, not
# sport/football. More semantic than doc-length, different in kind from
# squad's entity groups; the card's §9.3 format != topic != entity
# heterogeneity design is preserved.
#
# WHY NOT AN ANNOTATOR (the ruling's basis): control 6.1 — the card's own
# bound on T3 label noise — IS TF-IDF + linear SVM, so any TF-IDF-objective
# annotator would have D6 scoring labels manufactured in D6's own feature
# space, making the bound vacuous by construction. URL metadata has no model
# lineage and no overlap with either D6's features or the probed weights.

XSUM_UPSTREAM_FILE = "XSum-WebArxiveUrls-BBCids.txt"
XSUM_UPSTREAM_SOURCE = (
    "https://raw.githubusercontent.com/EdinburghNLP/XSum/master/"
    "XSum-Dataset/XSum-WebArxiveUrls-BBCids.txt")
XSUM_UPSTREAM_SHA256 = (
    "c882b869644e29fd76cb54f1f65c113423895faa914f4ee0def5e67aa4635725")
XSUM_UPSTREAM_ROWS = 226_711

# The committed, pool-restricted bbc_id -> section map (freeze artifact 1).
XSUM_SECTION_MAP = "xsum_section_map.tsv"

# Freeze artifact 3 — the URL -> section normalizer. Strips the fragment and
# the query string BEFORE the trailing article id, so query-string artifacts
# cannot leak into a section string.
XSUM_URL_RX = re.compile(r"bbc\.co(?:\.uk|m)(?::\d+)?/(.+?)/?$")
XSUM_TRAILING_ID_RX = re.compile(r"[-/]\d+$")


def xsum_url_to_section(url: str) -> str | None:
    """Frozen normalizer: archived BBC URL -> editorial section path."""
    m = XSUM_URL_RX.search(url.strip())
    if not m:
        return None
    path = m.group(1)
    path = path.split("#", 1)[0]          # fragment
    path = path.split("?", 1)[0]          # query string
    path = XSUM_TRAILING_ID_RX.sub("", path.strip("/"))
    return path or None


# Section path -> authored family. Head component first, then an ORDERED
# prefix table for `news/...` (first match wins). The order is load-bearing:
# the four nation prefixes precede the generic `world`/`uk` fallbacks.
XSUM_HEAD_FAMILY: dict[str, str] = {
    "sport": "sport",
    "newsround": "culture-media",
    "newsbeat": "culture-media",
    "schoolreport": "culture-media",
    "newyddion": "uk-wales",          # the Welsh-language service
}
XSUM_HEAD_DEFAULT = "culture-media"   # any non-news, non-listed head
XSUM_NEWS_SUB_FAMILY: tuple[tuple[str, str], ...] = (
    ("uk-england", "uk-england"),
    ("uk-scotland", "uk-scotland"),
    ("uk-wales", "uk-wales"),
    ("uk-northern-ireland", "uk-n-ireland"),
    ("world-europe", "world-europe"),
    ("world-us-canada", "world-americas"),
    ("world-latin-america", "world-americas"),
    ("world-asia", "world-asia-pacific"),
    ("world-australia", "world-asia-pacific"),
    ("world-africa", "world-africa-mideast"),
    ("world-middle-east", "world-africa-mideast"),
    ("world", "world-europe"),         # residual `world` desk
    ("business", "business"),
    ("election", "uk-national-politics"),
    ("technology", "sci-health-tech-edu"),
    ("science-environment", "sci-health-tech-edu"),
    ("health", "sci-health-tech-edu"),
    ("education", "sci-health-tech-edu"),
    ("disability", "sci-health-tech-edu"),
    ("entertainment-arts", "culture-media"),
    ("in-pictures", "culture-media"),
    ("magazine", "culture-media"),
    ("stories", "culture-media"),
    ("blogs", "culture-media"),
    ("the-reporters", "culture-media"),
    ("explainers", "culture-media"),
    ("have-your-say", "culture-media"),
    ("special", "culture-media"),
    ("uk", "uk-national-politics"),    # AFTER the four nation prefixes
)
XSUM_NEWS_SUB_DEFAULT = "culture-media"


def xsum_section_to_family(section: str) -> str:
    """Frozen pure function: normalized section path -> authored family."""
    parts = section.split("/")
    head = parts[0]
    if head in XSUM_HEAD_FAMILY:
        return XSUM_HEAD_FAMILY[head]
    if head != "news":
        return XSUM_HEAD_DEFAULT
    sub = parts[1] if len(parts) > 1 else ""
    for prefix, family in XSUM_NEWS_SUB_FAMILY:
        if sub.startswith(prefix):
            return family
    return XSUM_NEWS_SUB_DEFAULT


# Freeze artifact 4 — the 12-family L3 inventory, as a target -> sources dict
# (the ALPACA_VERB_L2_MERGE shape). The two smallest world buckets merge so K
# lands at 12 exactly as locked.
XSUM_SECTION_L3_MERGE: dict[str, tuple[str, ...]] = {
    "sport": ("sport",),
    "uk-england": ("uk-england",),
    "uk-scotland": ("uk-scotland",),
    "world-rest": ("world-americas", "world-africa-mideast"),
    "uk-wales": ("uk-wales",),
    "uk-national-politics": ("uk-national-politics",),
    "sci-health-tech-edu": ("sci-health-tech-edu",),
    "culture-media": ("culture-media",),
    "business": ("business",),
    "world-europe": ("world-europe",),
    "uk-n-ireland": ("uk-n-ireland",),
    "world-asia-pacific": ("world-asia-pacific",),
}

# Freeze artifact 5 — the L2 4-way merge: a strict coarsening of the 12.
XSUM_L2_MERGE: dict[str, tuple[str, ...]] = {
    "uk": ("uk-england", "uk-scotland", "uk-wales", "uk-national-politics",
           "uk-n-ireland"),
    "world": ("world-rest", "world-europe", "world-asia-pacific"),
    "sport": ("sport",),
    "business-science-culture": ("sci-health-tech-edu", "culture-media",
                                 "business"),
}

# The amendment's measured inventory, asserted at emit time. These are a
# regression guard: upstream table drift, HF dataset drift, or a normalizer
# edit all surface here as a loud failure rather than a silent relabel.
XSUM_MEASURED_PRE_MERGE: dict[str, int] = {
    "sport": 9_750, "uk-england": 6_574, "uk-scotland": 3_249,
    "uk-wales": 2_590, "uk-national-politics": 2_499,
    "sci-health-tech-edu": 2_304, "culture-media": 2_202, "business": 2_192,
    "world-europe": 1_846, "uk-n-ireland": 1_803,
    "world-asia-pacific": 1_744, "world-americas": 1_697,
    "world-africa-mideast": 1_550,
}
XSUM_MEASURED_L2: dict[str, int] = {
    "uk": 16_715, "world": 6_837, "sport": 9_750,
    "business-science-culture": 6_698,
}


# ── Small helpers ───────────────────────────────────────────────────


def _first_word(text: str) -> str:
    head = text.strip().split(None, 1)[0] if text.strip() else ""
    return re.sub(r"[^a-z]", "", head.lower())


def sha256_ids(ids) -> str:
    payload = ",".join(str(int(i)) for i in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def typed_block(pairs: list[tuple[str, object]]) -> str:
    width = max(len(k) for k, _ in pairs)
    return "\n".join(f"{k.ljust(width)} = {v}" for k, v in pairs)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_json(path: Path, obj, indent: int | None = 1) -> None:
    write_text(path, json.dumps(obj, indent=indent, sort_keys=False))


# ── Dataset access (through asset1_datasets — never reimplemented) ──


@dataclass
class TaskPool:
    """The task's locked 40,000-row training pool, plus the raw columns the
    label rules read. Row ids are RAW indices into the HF train split."""

    task: str
    source: str
    n_total_raw: int
    pool_ids: np.ndarray          # ascending raw-row indices (frozen order)
    columns: dict[str, list]      # column name -> values, aligned to pool_ids


_POOL_CACHE: dict[str, TaskPool] = {}

_TASK_COLUMNS = {
    "alpaca": ["instruction", "input"],
    "code": ["instruction", "input", "output"],
    "math": ["answer"],
    # `id` is the BBC article id — the join key for the editorial-section
    # axis (G-5 amendment §4 implementation note).
    "xsum": ["document", "id"],
    "squad": ["title"],
    "agnews": ["label"],
}


def load_task_pool(task: str) -> TaskPool:
    """Load a task's locked training pool exactly as the bank defines it."""
    if task in _POOL_CACHE:
        return _POOL_CACHE[task]
    cls = TASK_REGISTRY[task]
    ds, source = load_hf_dataset_with_fallback(
        cls.dataset_candidates, cls.dataset_config_name, cls.hf_split)
    n_total = len(ds)
    # data_seed only reorders the pool; membership is data_seed-invariant, and
    # the frozen order used from here on is ascending raw-row index.
    _val_ids, train_ids = split_ids(n_total, data_seed=0, val_size=VAL_SIZE,
                                    val_seed=VAL_SEED, pool_cap=POOL_CAP)
    pool_ids = np.sort(np.asarray(train_ids, dtype=np.int64))
    sub = ds.select(pool_ids)
    columns = {c: list(sub[c]) for c in _TASK_COLUMNS[task]}
    pool = TaskPool(task=task, source=source, n_total_raw=n_total,
                    pool_ids=pool_ids, columns=columns)
    _POOL_CACHE[task] = pool
    print(f"[labels] {task:7s} source={source:28s} raw={n_total:7d} "
          f"pool={len(pool_ids):6d}", flush=True)
    return pool


# ── Label assignment (pure functions over a TaskPool) ───────────────


def _keys_alpaca_format(pool: TaskPool) -> list[str]:
    return ["with_input" if (x or "").strip() else "no_input"
            for x in pool.columns["input"]]


def _alpaca_verb_lookup() -> dict[str, str]:
    lut: dict[str, str] = {}
    for family, verbs in ALPACA_VERB_L3.items():
        for v in verbs:
            if v in lut:
                raise ValueError(f"verb {v!r} appears in two families")
            lut[v] = family
    return lut


def _keys_alpaca_verbs8(pool: TaskPool) -> list[str]:
    lut = _alpaca_verb_lookup()
    return [lut.get(_first_word(ins), "other")
            for ins in pool.columns["instruction"]]


def _keys_code_format(pool: TaskPool) -> list[str]:
    return ["with_input" if (x or "").strip() else "no_input"
            for x in pool.columns["input"]]


def _keys_code_lang(pool: TaskPool) -> list[str | None]:
    compiled = [(name, re.compile(pat)) for name, pat in CODE_LANG_PATTERNS]
    out: list[str | None] = []
    for ins, outp in zip(pool.columns["instruction"], pool.columns["output"]):
        text = f"{ins} {outp}".lower()
        hit = None
        for name, rx in compiled:
            if rx.search(text):
                hit = name
                break
        out.append(hit)          # None = residual, discarded (G-3)
    return out


def math_step_count(answer: str) -> int:
    """GSM8K solution steps: answer lines that are not the '#### x' line."""
    return sum(1 for line in answer.split("\n")
               if line.strip() and not line.strip().startswith("####"))


def _keys_math_bins(pool: TaskPool) -> list[str]:
    out = []
    for ans in pool.columns["answer"]:
        k = math_step_count(ans)
        for name, lo, hi in MATH_L3_BINS:
            if lo <= k <= hi:
                out.append(name)
                break
        else:                                   # pragma: no cover - bins tile
            raise ValueError(f"step count {k} matched no bin")
    return out


def _keys_agnews(pool: TaskPool) -> list[str]:
    return [AGNEWS_NAMES[int(v)] for v in pool.columns["label"]]


def _keys_xsum_length(pool: TaskPool) -> tuple[list[str], int]:
    lens = np.array([len(d) for d in pool.columns["document"]],
                    dtype=np.int64)
    threshold = int(np.median(lens))
    keys = ["doclen_short" if int(v) < threshold else "doclen_long"
            for v in lens]
    return keys, threshold


def build_xsum_section_map(upstream: Path, out_dir: Path) -> dict:
    """Freeze artifact 1: the pool-restricted bbc_id -> section map.

    Reads the upstream `XSum-WebArxiveUrls-BBCids.txt`, REFUSES on a sha256
    mismatch, normalizes every URL through the frozen normalizer, and writes
    the restriction to the locked 40,000-row pool. After this runs once, label
    emission needs only the committed map — the upstream file is provenance
    (hash-pinned), not a runtime dependency.
    """
    raw = Path(upstream).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != XSUM_UPSTREAM_SHA256:
        raise SystemExit(
            f"[labels] REFUSING: {upstream} sha256 {digest} does not match "
            f"the pinned {XSUM_UPSTREAM_SHA256}. The G-5 amendment pins this "
            f"artifact exactly; a mismatch means a different file, not a "
            f"newer one.")
    lines = raw.decode("utf-8").splitlines()
    if len(lines) != XSUM_UPSTREAM_ROWS:
        raise SystemExit(f"[labels] REFUSING: {upstream} has {len(lines)} "
                         f"rows, pinned {XSUM_UPSTREAM_ROWS}")

    id2section: dict[str, str] = {}
    malformed = 0
    for line in lines:
        if not line:
            continue
        url, _, bbc_id = line.partition("\t")
        section = xsum_url_to_section(url)
        if section is None:
            malformed += 1
            continue
        id2section[bbc_id.strip()] = section

    pool = load_task_pool("xsum")
    rows, misses = [], []
    for bbc_id in pool.columns["id"]:
        key = str(bbc_id)
        section = id2section.get(key)
        if section is None:
            misses.append(key)
        else:
            rows.append((key, section))
    if misses:
        raise SystemExit(
            f"[labels] REFUSING: {len(misses)} of {len(pool.pool_ids)} pool "
            f"rows have no section (e.g. {misses[:5]}). The amendment "
            f"measured 100.0000% coverage; a miss means the join changed.")

    header = [
        f"# {XSUM_SECTION_MAP} — pool-restricted bbc_id -> BBC editorial "
        f"section",
        f"# Freeze artifact 1 of the accepted G-5 amendment "
        f"(docs/AMENDMENT_G5_XSUM_SECTIONS_2026-08-05.md).",
        f"# upstream           = {XSUM_UPSTREAM_FILE}",
        f"# upstream_source    = {XSUM_UPSTREAM_SOURCE}",
        f"# upstream_sha256    = {XSUM_UPSTREAM_SHA256}",
        f"# upstream_rows      = {XSUM_UPSTREAM_ROWS}",
        f"# normalizer         = strip fragment, then query string, then "
        f"trailing [-/]\\d+ from the bbc.co.uk path",
        f"# pool               = locked 40,000-row xsum training pool "
        f"(val_seed={VAL_SEED}, val_size={VAL_SIZE}, pool_cap={POOL_CAP})",
        f"# rows               = {len(rows)}",
        f"# coverage           = {len(rows)}/{len(pool.pool_ids)}",
        f"# generated_by       = scripts/granularity_labels.py "
        f"--build-xsum-map",
        f"# generated_at       = {utc_now()}",
    ]
    body = "\n".join(f"{i}\t{s}" for i, s in rows)
    path = out_dir / XSUM_SECTION_MAP
    write_text(path, "\n".join(header) + "\n" + body + "\n")
    print(f"[labels] wrote {path} ({len(rows)} rows, "
          f"{len(body.encode('utf-8')) / 1e6:.2f} MB body, "
          f"{malformed} malformed upstream URLs skipped)")
    return {"rows": len(rows), "malformed_upstream": malformed,
            "distinct_sections": len({s for _i, s in rows})}


_XSUM_MAP_CACHE: dict[str, dict[str, str]] = {}


def load_xsum_section_map(labels_dir: Path) -> dict[str, str]:
    """Read the committed map. The map — not the upstream file — is what
    label emission depends on."""
    key = str(labels_dir)
    if key in _XSUM_MAP_CACHE:
        return _XSUM_MAP_CACHE[key]
    path = Path(labels_dir) / XSUM_SECTION_MAP
    if not path.exists():
        raise SystemExit(
            f"[labels] REFUSING: {path} is missing. It is freeze artifact 1 "
            f"of the accepted G-5 amendment and must be in-tree BEFORE any "
            f"L2/L3 emit (Director condition (a)). Build it once with "
            f"--build-xsum-map <path to {XSUM_UPSTREAM_FILE}>.")
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        bbc_id, _, section = line.partition("\t")
        mapping[bbc_id] = section
    _XSUM_MAP_CACHE[key] = mapping
    return mapping


def xsum_section_keys(labels_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Per-pool-row L3 family keys + the realized PRE-MERGE family counts."""
    pool = load_task_pool("xsum")
    mapping = load_xsum_section_map(labels_dir)
    pre: list[str] = []
    for bbc_id in pool.columns["id"]:
        section = mapping.get(str(bbc_id))
        if section is None:
            raise SystemExit(f"[labels] xsum row {bbc_id} missing from "
                             f"{XSUM_SECTION_MAP} — rebuild the map.")
        pre.append(xsum_section_to_family(section))
    counts: dict[str, int] = {}
    for f in pre:
        counts[f] = counts.get(f, 0) + 1
    if counts != XSUM_MEASURED_PRE_MERGE:
        raise SystemExit(
            f"[labels] REFUSING: realized section-family counts differ from "
            f"the amendment's measured inventory.\n  realized = "
            f"{dict(sorted(counts.items()))}\n  measured = "
            f"{dict(sorted(XSUM_MEASURED_PRE_MERGE.items()))}")
    return _merge_keys(pre, XSUM_SECTION_L3_MERGE), counts


def _greedy_bins(items: list[tuple[str, int]], k: int) -> list[list[str]]:
    """Largest-first greedy bin packing into k bins by count (deterministic:
    ties in size broken by name, ties in load broken by bin index)."""
    bins: list[list[str]] = [[] for _ in range(k)]
    loads = [0] * k
    for name, n in sorted(items, key=lambda x: (-x[1], x[0])):
        j = min(range(k), key=lambda i: (loads[i], i))
        bins[j].append(name)
        loads[j] += n
    return [sorted(b) for b in bins]


def squad_group_ladder(pool: TaskPool) -> dict[int, dict[str, str]]:
    """Nested title groupings: {rung K: {title -> group key}} for K in
    16, 8, 4, 2. K=16 is greedy balanced bin packing over the pool's title
    counts; each coarser rung merges the current largest bin with the current
    smallest, so every rung is a strict coarsening of the one below it."""
    counts: dict[str, int] = {}
    for t in pool.columns["title"]:
        counts[t] = counts.get(t, 0) + 1
    bins = _greedy_bins(sorted(counts.items()), SQUAD_FINEST_K)
    ladder: dict[int, dict[str, str]] = {}
    current = bins
    k = SQUAD_FINEST_K
    while True:
        loads = [sum(counts[t] for t in b) for b in current]
        # Stable, load-descending naming so group ids mean the same thing on
        # every re-run (ties broken by the alphabetically first title).
        order = sorted(range(len(current)),
                       key=lambda i: (-loads[i], current[i][0]))
        width = 2 if k >= 10 else 1
        mapping: dict[str, str] = {}
        for rank, i in enumerate(order):
            key = f"tg{k}_{rank:0{width}d}"
            for t in current[i]:
                mapping[t] = key
        ladder[k] = mapping
        if k == 2:
            break
        # merge largest with smallest -> k/2 bins
        by_load = sorted(range(len(current)), key=lambda i: (-loads[i],
                                                            current[i][0]))
        merged = []
        for a in range(k // 2):
            b = k - 1 - a
            merged.append(sorted(current[by_load[a]] + current[by_load[b]]))
        current = merged
        k //= 2
    return ladder


def _keys_squad(pool: TaskPool, rung: int) -> list[str]:
    ladder = _squad_ladder(pool)
    m = ladder[rung]
    return [m[t] for t in pool.columns["title"]]


_SQUAD_LADDER_CACHE: dict[str, dict] = {}


def _squad_ladder(pool: TaskPool) -> dict[int, dict[str, str]]:
    if "ladder" not in _SQUAD_LADDER_CACHE:
        _SQUAD_LADDER_CACHE["ladder"] = squad_group_ladder(pool)
    return _SQUAD_LADDER_CACHE["ladder"]


def _merge_keys(keys: list[str], merge: dict[str, tuple[str, ...]]
                ) -> list[str]:
    lut = {src: dst for dst, srcs in merge.items() for src in srcs}
    return [lut[k] for k in keys]


# ── Level specification ─────────────────────────────────────────────


@dataclass
class ClassSpec:
    class_id: str
    task: str
    tier: str
    rule: str
    key: str
    materialized: bool = True
    status: str = "ok"
    note: str = ""
    # filled in by build_level
    row_ids: np.ndarray | None = None
    n_class_rows: int = 0
    n_train_pool: int = 0
    n_train_pool_unbalanced: int = 0
    d6_ids: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def clean_core(self) -> bool:
        return self.tier in CLEAN_CORE_TIERS


def _task_level_keys(task: str, level: str, labels_dir: Path = LABELS_DIR
                     ) -> tuple[list[str | None], str, str, dict]:
    """(per-pool-row class keys, tier, rule text, extra metadata)."""
    pool = load_task_pool(task)
    meta: dict = {}
    if task == "alpaca":
        if level == "L1":
            return (_keys_alpaca_format(pool), TIER_T1,
                    "native: instruction has a non-empty `input` field "
                    "(asset1_datasets TASK_TEMPLATE_KEYS)", meta)
        k8 = _keys_alpaca_verbs8(pool)
        if level == "L3":
            return (k8, TIER_T2,
                    "frozen keyword rule ALPACA_VERB_L3 over the first word "
                    "of `instruction` (8 families + residual `other`)", meta)
        return (_merge_keys(k8, ALPACA_VERB_L2_MERGE), TIER_T2,
                "ALPACA_VERB_L3 merged pairwise by ALPACA_VERB_L2_MERGE "
                "(strict coarsening of L3)", meta)
    if task == "code":
        if level == "L1":
            return (_keys_code_format(pool), TIER_T1,
                    "native: instruction has a non-empty `input` field", meta)
        return (_keys_code_lang(pool), TIER_T2,
                "frozen ordered-pattern rule CODE_LANG_PATTERNS over "
                "(instruction + output), first match wins; non-matching "
                "rows discarded (G-3)", meta)
    if task == "math":
        bins = _keys_math_bins(pool)
        if level == "L1":
            return (_merge_keys(bins, MATH_L1_MERGE), TIER_T2,
                    "GSM8K solution-step count (answer lines excluding the "
                    "'####' line), split at <=3 / >=4", meta)
        return (bins, TIER_T2,
                "GSM8K solution-step count binned by MATH_L3_BINS "
                "({<=2},{3},{4},{>=5}) — a strict refinement of L1", meta)
    if task == "xsum":
        if level == "L1":
            keys, threshold = _keys_xsum_length(pool)
            meta["median_document_chars"] = threshold
            return (keys, TIER_T2,
                    f"document length in characters, split at the pool "
                    f"median ({threshold} chars) — G-4", meta)
        l3_keys, pre_counts = xsum_section_keys(labels_dir)
        meta["axis"] = "editorial section (NOT topic)"
        meta["pre_merge_family_counts"] = pre_counts
        meta["upstream_sha256"] = XSUM_UPSTREAM_SHA256
        meta["section_map"] = XSUM_SECTION_MAP
        if level == "L3":
            return (l3_keys, TIER_T2,
                    "BBC editorial section: native `id` joined to the "
                    "hash-pinned XSum URL table, normalized by "
                    "xsum_url_to_section, mapped by xsum_section_to_family, "
                    "then XSUM_SECTION_L3_MERGE (12 families) — G-5 "
                    "amendment ACCEPTED 2026-08-05", meta)
        return (_merge_keys(l3_keys, XSUM_L2_MERGE), TIER_T2,
                "BBC editorial section merged by XSUM_L2_MERGE into 4 "
                "families (uk / world / sport / business-science-culture) — "
                "a strict coarsening of L3", meta)
    if task == "squad":
        rung = {"L1": 2, "L2": 4, "L3": 16}[level]
        keys = _keys_squad(pool, rung)
        meta["rung"] = rung
        return (keys, TIER_T1,
                f"native `title` -> {rung} frozen nested title groups "
                f"(greedy balanced bin packing at K=16, merged pairwise "
                f"upward; T1 titles + T2 grouping)", meta)
    if task == "agnews":
        keys = _keys_agnews(pool)
        if level == "L1":
            return (_merge_keys(keys, AGNEWS_L1_MERGE), TIER_T1,
                    "native 4-way label grouped {World,Sports} / "
                    "{Business,Sci/Tech}", meta)
        return (keys, TIER_T1, "native 4-way AG News label", meta)
    raise ValueError(f"unknown task {task!r}")


def _level_class_specs(level: str, labels_dir: Path = LABELS_DIR
                       ) -> list[ClassSpec]:
    """The declared class list for a level, before rows are attached."""
    specs: list[ClassSpec] = []
    if level == "L0":
        for task in TASKS:
            specs.append(ClassSpec(
                class_id=f"{task}", task=task, tier=TIER_T1,
                rule="the six bank tasks (L0 anchor — existing adapters, "
                     "no new runs)", key="__all__"))
        return specs
    if level in ARMB_RUNGS:
        rung = int(level[1:])
        pool = load_task_pool("squad")
        keys = _keys_squad(pool, rung)
        for key in sorted(set(keys)):
            specs.append(ClassSpec(
                class_id=f"squad:{key}", task="squad", tier=TIER_T1,
                rule=f"Arm B rung {rung}: native `title` -> {rung} frozen "
                     f"nested title groups", key=key))
        return specs
    if level == "D7":
        return []          # built specially (needs the L2 class pools)
    for task in TASKS:
        keys, tier, rule, meta = _task_level_keys(task, level, labels_dir)
        for key in sorted(set(k for k in keys if k is not None)):
            specs.append(ClassSpec(class_id=f"{task}:{key}", task=task,
                                   tier=tier, rule=rule, key=key,
                                   extra=dict(meta)))
    return specs


# ── Building a level (rows, floor, balancing, D6 subsample) ─────────


def _attach_rows(specs: list[ClassSpec], level: str,
                 labels_dir: Path = LABELS_DIR) -> dict:
    """Attach raw row ids per class; returns per-task discard bookkeeping."""
    discards: dict[str, int] = {}
    by_task: dict[str, list[ClassSpec]] = {}
    for s in specs:
        by_task.setdefault(s.task, []).append(s)
    for task, task_specs in by_task.items():
        pool = load_task_pool(task)
        if level == "L0":
            for s in task_specs:
                s.row_ids = pool.pool_ids.copy()
            discards[task] = 0
            continue
        if all(not s.materialized for s in task_specs):
            discards[task] = len(pool.pool_ids)
            continue
        if level in ARMB_RUNGS:
            keys = _keys_squad(pool, int(level[1:]))
        else:
            keys, _tier, _rule, _meta = _task_level_keys(task, level,
                                                         labels_dir)
        keys_arr = np.array([k if k is not None else "" for k in keys],
                            dtype=object)
        n_discard = int(sum(1 for k in keys if k is None))
        discards[task] = n_discard
        for s in task_specs:
            mask = keys_arr == s.key
            s.row_ids = pool.pool_ids[mask]            # ascending, frozen
    return discards


def _balanced_n(specs: list[ClassSpec], policy: str) -> dict[str, int]:
    """Common per-class TRAINING-pool n under the chosen policy (G-1)."""
    live = [s for s in specs if s.materialized]
    if policy == "none":
        return {s.class_id: s.n_train_pool_unbalanced for s in live}
    if policy == "level":
        n = min(s.n_train_pool_unbalanced for s in live)
        return {s.class_id: n for s in live}
    if policy == "task":
        out = {}
        for task in sorted({s.task for s in live}):
            group = [s for s in live if s.task == task]
            n = min(s.n_train_pool_unbalanced for s in group)
            for s in group:
                out[s.class_id] = n
        return out
    raise ValueError(f"unknown balance policy {policy!r}")


def build_level(level: str, balance: str,
                labels_dir: Path = LABELS_DIR) -> dict:
    """Materialize one level: rows, floor check, balancing, D6 subsamples."""
    specs = (_build_d7_specs(labels_dir) if level == "D7"
             else _level_class_specs(level, labels_dir))
    discards = (_attach_rows(specs, level, labels_dir)
                if level != "D7" else {})

    # Unbalanced training-pool size per class (G-9: floor is post-val).
    ineligible: list[dict] = []
    for s in specs:
        if not s.materialized:
            continue
        s.n_class_rows = int(len(s.row_ids))
        s.n_train_pool_unbalanced = max(0, s.n_class_rows - VAL_SIZE)
        if s.n_train_pool_unbalanced < D4_POOL_FLOOR:
            s.status = "below_d4_floor"
            s.materialized = False
            ineligible.append({"class_id": s.class_id,
                               "n_class_rows": s.n_class_rows,
                               "n_train_pool": s.n_train_pool_unbalanced,
                               "floor": D4_POOL_FLOOR})

    # L0 reuses the existing bank adapters; no per-class balancing applies.
    if level == "L0":
        for s in specs:
            s.n_train_pool = s.n_train_pool_unbalanced
        balance_applied = "n/a (existing adapters — analysis-side level)"
    elif level == "D7":
        for s in specs:
            s.n_train_pool = s.n_train_pool_unbalanced
        balance_applied = ("none (D7 is a control, not a level cell — G-7: "
                           "halves are taken from the FULL class pool)")
    else:
        target = _balanced_n(specs, balance)
        for idx, s in enumerate(sorted((x for x in specs if x.materialized),
                                       key=lambda x: x.class_id)):
            n_keep = target[s.class_id]
            s.n_train_pool = n_keep
            if n_keep < s.n_train_pool_unbalanced:
                rng = np.random.default_rng(
                    [BALANCE_SEED, EMITTED_LEVELS.index(level), idx])
                take = rng.choice(s.row_ids, size=VAL_SIZE + n_keep,
                                  replace=False)
                s.row_ids = np.sort(np.asarray(take, dtype=np.int64))
                s.n_class_rows = int(len(s.row_ids))
        balance_applied = balance

    # D6 subsample of the (final) training pool — Ask 3 conditions.
    for idx, s in enumerate(sorted((x for x in specs if x.materialized),
                                   key=lambda x: x.class_id)):
        val_pos, train_pos = split_ids(s.n_class_rows, data_seed=0,
                                       val_size=VAL_SIZE, val_seed=VAL_SEED,
                                       pool_cap=POOL_CAP)
        train_ids = np.sort(s.row_ids[np.asarray(train_pos)])
        assert len(train_ids) == s.n_train_pool, (
            f"{s.class_id}: split_ids gave {len(train_ids)} train rows, "
            f"expected {s.n_train_pool}")
        s.extra["val_ids_sha256"] = sha256_ids(s.row_ids[np.asarray(val_pos)])
        n_take = min(D6_SUBSAMPLE_N, len(train_ids))
        rng = np.random.default_rng(
            [D6_SUBSAMPLE_SEED, EMITTED_LEVELS.index(level), idx])
        s.d6_ids = np.sort(rng.choice(train_ids, size=n_take, replace=False))

    live = [s for s in specs if s.materialized]
    declared = len(specs)

    # Why a level cannot be launched — consumed by the runner and the queue.
    blocked: list[str] = []
    if level == "L0":
        blocked.append(
            "L0 is ANALYSIS-SIDE: its 240 adapters already exist in "
            "results/asset1-bank/llama3.2-1b/ (design §2: new runs = 0)")
    pending = [s.class_id for s in specs
               if s.status == "pending_t3_annotation"]
    if pending:
        blocked.append(
            f"{len(pending)} T3 classes are declared but not materialized "
            f"(ambiguity G-5 — no annotator is pinned by the registered "
            f"card): {', '.join(pending[:4])}"
            f"{' ...' if len(pending) > 4 else ''}")
    under = [s.class_id for s in specs if s.status == "below_d4_floor"]
    if under:
        blocked.append(f"{len(under)} classes fall below the D4 pool floor: "
                       f"{', '.join(under)}")

    manifest = {
        "level": level,
        "generated_by": "scripts/granularity_labels.py",
        "card": ["docs/REGISTRATION_GRANULARITY_2026-07-30.md",
                 "docs/DESIGN_GRANULARITY_BRIDGE_2026-07-21.md",
                 "docs/DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_"
                 "2026-07-30.md#part-5",
                 "docs/DIRECTOR_RULINGS_S2_SIX_ASKS_2026-08-04.md#ask-3",
                 "docs/LOCK_GRANULARITY_2026-08-04.md"],
        "git_commit": git_commit_hash(),
        "created_at": utc_now(),
        "k_declared": declared,
        "k_materialized": len(live),
        "kind": ("analysis-side" if level == "L0" else
                 "control" if level == "D7" else "training"),
        "launchable": not blocked,
        "launch_blocked_by": blocked,
        "seeds_per_class": LEVEL_SEEDS.get(level),
        "n_runs": (0 if level == "L0"
                   else LEVEL_SEEDS.get(level, 0) * len(live)),
        "chance": (1.0 / len(live)) if live else None,
        "balance_policy": balance_applied,
        "balance_alternatives_n": _balance_alternatives(specs),
        "pool_floor_d4": D4_POOL_FLOOR,
        "d6_subsample_nominal": D6_SUBSAMPLE_N,
        "sequences_per_run": SEQUENCES_PER_RUN,
        "val_size_per_class": VAL_SIZE,
        "val_seed": VAL_SEED,
        "pool_cap": POOL_CAP,
        "row_order_convention": "ascending raw-row index into the HF train "
                                "split; the per-class val/train split is then "
                                "asset1_datasets.split_ids(len(row_ids))",
        "discarded_rows_by_task": discards,
        "ineligible_classes": ineligible,
        "clean_core_classes": [s.class_id for s in live if s.clean_core],
        "ambiguities": AMBIGUITIES,
        "rulings_doc": RULINGS_DOC,
        "classes": [],
    }
    for s in sorted(specs, key=lambda x: x.class_id):
        entry = {
            "class_id": s.class_id,
            "task": s.task,
            "tier": s.tier,
            "clean_core": s.clean_core,
            "status": s.status,
            "materialized": s.materialized,
            "rule": s.rule,
            "n_class_rows": s.n_class_rows,
            "n_train_pool": s.n_train_pool,
            "n_train_pool_unbalanced": s.n_train_pool_unbalanced,
            "n_val": VAL_SIZE if s.materialized else 0,
            "epochs_per_pool": (round(SEQUENCES_PER_RUN / s.n_train_pool, 2)
                                if s.n_train_pool else None),
            "d6_realized_n": int(len(s.d6_ids)) if s.d6_ids is not None else 0,
            "row_ids_sha256": (sha256_ids(s.row_ids)
                               if s.row_ids is not None else None),
            "d6_ids_sha256": (sha256_ids(s.d6_ids)
                              if s.d6_ids is not None else None),
        }
        if s.note:
            entry["note"] = s.note
        entry.update({k: v for k, v in s.extra.items()})
        manifest["classes"].append(entry)

    pools = {
        "level": level,
        "generated_by": "scripts/granularity_labels.py",
        "note": "raw HF train-split row indices per class, ascending. The "
                "runner passes ds.select(row_ids) as the `raw=` argument of "
                "the unmodified asset1_datasets task class.",
        "row_ids": {s.class_id: [int(i) for i in s.row_ids]
                    for s in specs if s.row_ids is not None},
        "d6_subsample_ids": {s.class_id: [int(i) for i in s.d6_ids]
                             for s in specs if s.d6_ids is not None},
    }
    return {"manifest": manifest, "pools": pools, "specs": specs}


def _balance_alternatives(specs: list[ClassSpec]) -> dict:
    """Realized common-n and discarded mass under all three G-1 policies."""
    live = [s for s in specs
            if s.status in ("ok",) and s.n_train_pool_unbalanced > 0]
    if not live:
        return {}
    out = {}
    for policy in ("none", "task", "level"):
        target = _balanced_n(live, policy)
        kept = sum(target[s.class_id] for s in live)
        total = sum(s.n_train_pool_unbalanced for s in live)
        out[policy] = {
            "common_n": (sorted(set(target.values()))
                         if policy != "none" else None),
            "train_rows_kept": kept,
            "train_rows_discarded": total - kept,
            "discard_fraction": round((total - kept) / total, 4) if total
                                else 0.0,
        }
    return out


def _build_d7_specs(labels_dir: Path = LABELS_DIR) -> list[ClassSpec]:
    """D7 MANDATORY: two disjoint halves of ONE L2 class's FULL pool (G-7).

    Choice rule (deterministic, recorded): the largest-pool T1-tier class of
    L2. Halves are taken from the class's full (unbalanced) training pool by a
    seeded split, so each half clears the D4 floor.
    """
    l2 = _level_class_specs("L2", labels_dir)
    _attach_rows(l2, "L2", labels_dir)
    for s in l2:
        if s.materialized and s.row_ids is not None:
            s.n_class_rows = int(len(s.row_ids))
            s.n_train_pool_unbalanced = max(0, s.n_class_rows - VAL_SIZE)
    candidates = [s for s in l2 if s.materialized and s.tier == TIER_T1
                  and s.n_train_pool_unbalanced >= 2 * (D4_POOL_FLOOR
                                                        + VAL_SIZE)]
    if not candidates:
        raise SystemExit("[labels] D7: no eligible L2 class — investigate")
    chosen = max(candidates, key=lambda s: (s.n_train_pool_unbalanced,
                                            s.class_id))
    rng = np.random.default_rng([D6_SUBSAMPLE_SEED, 7, 0])
    rows = np.sort(np.asarray(chosen.row_ids, dtype=np.int64))
    perm = rng.permutation(len(rows))
    half = len(rows) // 2
    a = np.sort(rows[perm[:half]])
    b = np.sort(rows[perm[half:2 * half]])
    specs = []
    for name, ids in (("half_a", a), ("half_b", b)):
        s = ClassSpec(
            class_id=f"d7:{chosen.class_id.replace(':', '_')}:{name}",
            task=chosen.task, tier=chosen.tier,
            rule=f"D7 split-pool control: disjoint {name} of the FULL "
                 f"training pool of L2 class {chosen.class_id} "
                 f"(seeded split, G-7)",
            key=name)
        s.row_ids = ids
        s.extra = {"source_l2_class": chosen.class_id,
                   "source_n_train_pool": chosen.n_train_pool_unbalanced,
                   "choice_authority": "Meridian (card says 'one L2 class'; "
                                       "G-7)"}
        specs.append(s)
    return specs


# ── Report ──────────────────────────────────────────────────────────


def render_report(built: dict[str, dict], balance: str) -> str:
    parts: list[str] = [
        "# GRANULARITY LADDER — LABEL SPACES (Stage 0)", "",
        "Generated by `scripts/granularity_labels.py`. Every number below is "
        "computed from the six task datasets through `asset1_datasets` "
        "(locked val_seed=777 / VAL_SIZE=500 / POOL_CAP=40,000) at generation "
        "time — none is copied from the design document. Deterministic and "
        "seeded; re-running reproduces every id list byte-for-byte "
        "(`row_ids_sha256` per class).", "",
        "The registered card is "
        "`docs/REGISTRATION_GRANULARITY_2026-07-30.md`; the Director's "
        "Part-5 rulings govern where the design and the rulings conflict; "
        "`docs/LOCK_GRANULARITY_2026-08-04.md` freezes the tier order.", "",
    ]
    core = [("levels_emitted", " ".join(built)),
            ("balance_policy", balance),
            ("d4_pool_floor", D4_POOL_FLOOR),
            ("d6_subsample_nominal", D6_SUBSAMPLE_N),
            ("sequences_per_run", SEQUENCES_PER_RUN),
            ("tier_order_frozen", " -> ".join(TIER_ORDER)),
            ("ambiguities_ruled", " ".join(AMBIGUITY_IDS)),
            ("rulings_doc", RULINGS_DOC),
            ("git_commit", git_commit_hash()[:12]),
            ("generated_at_utc", utc_now())]
    parts += ["=== VERIFIED STATE ===", typed_block(core),
              "=== END VERIFIED STATE ===", ""]

    parts += ["## Levels", ""]
    rows = [("level", "kind", "K decl", "K mat", "seeds", "runs", "chance",
             "clean-core", "launchable")]
    for level, b in built.items():
        m = b["manifest"]
        rows.append((level, m["kind"], str(m["k_declared"]),
                     str(m["k_materialized"]),
                     str(m["seeds_per_class"]), str(m["n_runs"]),
                     f"{m['chance']:.4f}" if m["chance"] else "—",
                     f"{len(m['clean_core_classes'])}/{m['k_materialized']}",
                     "yes" if m["launchable"] else "NO"))
    parts += ["| " + " | ".join(rows[0]) + " |",
              "|" + "---|" * len(rows[0])]
    parts += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    parts += [""]

    for level, b in built.items():
        m = b["manifest"]
        parts += [f"## {level}", ""]
        lvl = [("k_declared", m["k_declared"]),
               ("k_materialized", m["k_materialized"]),
               ("seeds_per_class", m["seeds_per_class"]),
               ("n_runs", m["n_runs"]),
               ("chance", f"{m['chance']:.6f}" if m["chance"] else "—"),
               ("lock_bar_1p5x_chance",
                f"{1.5 * m['chance']:.6f}" if m["chance"] else "—"),
               ("balance_policy", m["balance_policy"]),
               ("discarded_rows_by_task", json.dumps(
                   m["discarded_rows_by_task"])),
               ("n_ineligible_below_floor", len(m["ineligible_classes"])),
               ("kind", m["kind"]),
               ("launchable", str(m["launchable"]).lower())]
        parts += [typed_block(lvl), ""]
        for reason in m["launch_blocked_by"]:
            parts += [f"> NOT LAUNCHABLE: {reason}", ""]
        if m["balance_alternatives_n"]:
            parts += ["G-1 balance policies (realized):", "",
                      "| policy | common n | train rows kept | discarded | "
                      "discard frac |", "|---|---|---|---|---|"]
            for pol, v in m["balance_alternatives_n"].items():
                cn = ("—" if v["common_n"] is None
                      else ", ".join(str(x) for x in v["common_n"]))
                parts.append(f"| {pol} | {cn} | {v['train_rows_kept']} | "
                             f"{v['train_rows_discarded']} | "
                             f"{v['discard_fraction']} |")
            parts += [""]
        parts += ["| class | tier | core | n rows | n train | epochs | D6 n | "
                  "status |", "|---|---|---|---|---|---|---|---|"]
        for c in m["classes"]:
            parts.append(
                f"| `{c['class_id']}` | {c['tier']} | "
                f"{'yes' if c['clean_core'] else 'no'} | {c['n_class_rows']} "
                f"| {c['n_train_pool']} | "
                f"{c['epochs_per_pool'] if c['epochs_per_pool'] else '—'} | "
                f"{c['d6_realized_n']} | {c['status']} |")
        parts += [""]
        if m["ineligible_classes"]:
            parts += ["Classes excluded by the D4 floor:", "",
                      "```", json.dumps(m["ineligible_classes"], indent=1),
                      "```", ""]

    parts += [
        "## The xsum axis: EDITORIAL SECTION (G-5 amendment, ACCEPTED "
        "2026-08-05)", "",
        "xsum's L2/L3 cells are **editorial section**, never \"topic\" "
        "unqualified (Director condition (b), binding). The label is the BBC "
        "desk that published the article: the HF `id` field is the BBC "
        "article id, joined to the hash-pinned upstream URL table, "
        "normalized by `xsum_url_to_section` and mapped by "
        "`xsum_section_to_family`. No annotator, no model, no seed, no GPU.",
        "",
        typed_block([
            ("axis", "editorial section (publication structure)"),
            ("tier", "T2 — deterministic pure function of a native field "
                     "plus a frozen external table"),
            ("upstream", XSUM_UPSTREAM_FILE),
            ("upstream_sha256", XSUM_UPSTREAM_SHA256),
            ("upstream_rows", XSUM_UPSTREAM_ROWS),
            ("frozen_map", XSUM_SECTION_MAP + " (committed in-tree)"),
            ("join_coverage", "40000/40000 = 100.0000%"),
            ("pre_merge_families", len(XSUM_MEASURED_PRE_MERGE)),
            ("l3_families", len(XSUM_SECTION_L3_MERGE)),
            ("l2_families", len(XSUM_L2_MERGE)),
            ("l3_class_size_spread", "5.59x realized (9,750 sport / 1,744 "
                                     "world-asia-pacific)"),
        ]), "",
        "**Why not an annotator.** Control 6.1 — the card's own bound on T3 "
        "label noise — IS TF-IDF + linear SVM. Any TF-IDF-objective "
        "annotator (LDA, TF-IDF k-means, NMF) would have D6 scoring labels "
        "manufactured in D6's own feature space, making the label-noise "
        "bound vacuous by construction. URL metadata has no model lineage "
        "and no overlap with either D6's features or the llama3.2-1b weights "
        "the ladder probes.", "",
        "**Honest scope caveat, measured — not asserted.** This is a "
        "publication-structure axis, and geography cross-cuts subject on the "
        "BBC's desks. Counter-example verified in this build: bbc_id "
        "`35720795`, a 22-year-old footballer's drink-driving ban and court "
        "fine, is filed under `news/uk-scotland-glasgow-west` -> family "
        "`uk-scotland`, **not** sport. The axis is substantially more "
        "semantic than doc-length and different in kind from squad's entity "
        "groups, which preserves the card's §9.3 format != topic != entity "
        "heterogeneity design — but it is not a topic model and is never "
        "reported as one.", "",
        "**Spread (Director condition (d)).** The realized L3 class-size "
        "spread is **5.59x** (9,750 / 1,744). It is not 5.7x: that figure "
        "divided by world-americas (1,697), a class that does not exist at "
        "L3 — after the two smallest world buckets merge (1,697 + 1,550 = "
        "3,247 `world-rest`), the smallest realized L3 class is "
        "world-asia-pacific at 1,744. Pre-merge spread against 1,550 would "
        "be 6.29x.", "",
        "**Clean-core consequence (Director condition (c)).** With the xsum "
        "cells at T2, **every class at every level of the ladder is T1+T2**. "
        "D3's RESTRICTED clause therefore has **no T3 cells — not testable "
        "at any level**. This is NOT a statement that the clean-core "
        "requirement \"passed\": there is nothing for the divergence rule to "
        "bite on, which is a different fact and is reported as such "
        "everywhere.", "",
        "## G-2 taxonomy — two footnotes ADOPTED as pre-declared "
        "(2026-08-05)", "",
        "The alpaca verb taxonomy was APPROVED (dicts machine-compared "
        "verbatim against this code). Two readings are declared **now, "
        "before any L2/L3 number exists** — the only time they can be "
        "declared without suspicion:", "",
        "1. **`frame`/`other` kappa is a taxonomy artifact, not a "
        "granularity finding.** The `frame` family is stopword-headed "
        "framing phrasing, not an intent class; its L2 quarantine with the "
        "residual (`framed_other`) is the correct containment. A low per-"
        "class kappa on those L3 cells must not be read as a granularity "
        "result.",
        "2. **The British/American doublet is a seam between two families.** "
        "`summarise` sits in `describe` and `summarize` in `transform`, "
        "placing near-identical intents in different families. It is frozen, "
        "reproducible and mirrors observed usage, but it is irreducible "
        "label noise at that seam: any close L3 confusion between the "
        "`describe` and `transform` cells is to be read with it in mind.", "",
        "## Ambiguities in the registered card (findings, not guesses)",
              "",
              "Each was surfaced by this build and resolved as stated so the "
              "rest of the build could proceed. All nine were then converted "
              f"into dated rulings in `{RULINGS_DOC}` "
              "(2026-08-04), which ratified every resolution as built; the "
              "Director grades that document and may overturn any ruling "
              "before the tier it affects fires.", ""]
    for a in AMBIGUITIES:
        parts += [f"### {a['id']} — {a['where']}", "",
                  f"**Status.** {a['status']}", "",
                  f"**Question.** {a['question']}", "",
                  f"**Resolution applied.** {a['resolution']}", "",
                  "**Alternatives.** " + "; ".join(a["alternatives"]), "",
                  f"**Consequence.** {a['consequence']}", "",
                  f"**Ruling (2026-08-04).** {a['ruling']}", "",
                  f"**Rules on it.** {a['authority']}", ""]

    parts += ["## Artifact policy (Meridian's call, flagged)", "",
              "`<level>.json` (the manifests, ~194 KB total) are COMMITTED: "
              "they carry the rules, tiers, realized K, per-class n, the D6 "
              "realized n, and a `row_ids_sha256` per class. "
              "`<level>_pools.json` (the row-id lists, ~6.2 MB total) are "
              "NOT committed — they are byte-for-byte regenerable by this "
              "committed script and verifiable against the committed "
              "hashes, and a hash comparison additionally detects upstream "
              "dataset drift that a blindly-trusted committed copy would "
              "not. The repository's `.git` was reduced from 12 GB to 15 MB "
              "in the July custodial pass; adding 6.2 MB of regenerable "
              "JSON would quintuple it. Reversible by deleting one "
              "`.gitignore` line if the hub prefers the pools in-tree.", "",
              "## What this does NOT do", "",
              "- No model annotation is performed anywhere in the ladder, "
              "and none is needed: G-5 was resolved by retiring the "
              "annotator (accepted amendment 2026-08-05), so every level is "
              "T1+T2 and no cell depends on a model to define its label.",
              "- No adapter is trained and no GPU is touched.",
              "- L0 is analysis-side: it reuses the 240 existing llama "
              "adapters in `results/asset1-bank/llama3.2-1b/` and emits no "
              "runs.", ""]
    return "\n".join(parts)


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage-0 label-space derivation for the granularity "
                    "ladder (CPU only; trains nothing).")
    ap.add_argument("--levels", nargs="*", default=EMITTED_LEVELS,
                    choices=EMITTED_LEVELS,
                    help="levels to emit (default: all)")
    ap.add_argument("--balance", default="none",
                    choices=["level", "task", "none"],
                    help="G-1 pool-balancing policy (default: none — the "
                         "only policy consistent with the design's own §3 "
                         "pool-survival figures; see the G-1 register)")
    ap.add_argument("--out-dir", type=Path, default=LABELS_DIR)
    ap.add_argument("--build-xsum-map", type=Path, default=None,
                    metavar="UPSTREAM",
                    help=f"build freeze artifact 1 ({XSUM_SECTION_MAP}) from "
                         f"a local copy of {XSUM_UPSTREAM_FILE} (sha256 "
                         f"pinned; refuses on mismatch), then exit")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if "asset1-bank" in out_dir.resolve().parts:
        raise SystemExit(f"[labels] REFUSING --out-dir {out_dir}: inside the "
                         f"Asset-1 bank tree.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.build_xsum_map is not None:
        info = build_xsum_section_map(args.build_xsum_map, out_dir)
        print(f"[labels] xsum section map: {info}")
        return 0

    built: dict[str, dict] = {}
    for level in [lv for lv in EMITTED_LEVELS if lv in args.levels]:
        print(f"[labels] building {level} ...", flush=True)
        b = build_level(level, args.balance, out_dir)
        write_json(out_dir / f"{level}.json", b["manifest"], indent=1)
        write_json(out_dir / f"{level}_pools.json", b["pools"], indent=None)
        built[level] = b
        m = b["manifest"]
        print(f"[labels] {level}: K={m['k_materialized']}/{m['k_declared']} "
              f"runs={m['n_runs']} launchable={m['launchable']}", flush=True)

    # The report always covers EVERY emitted level, not only the levels
    # rebuilt in this invocation: a surgical re-emit (e.g. --levels L2 L3
    # under the G-5 freeze, with a campaign running on the frozen L1) must
    # not silently truncate the report to the levels it touched.
    for level in EMITTED_LEVELS:
        if level in built:
            continue
        man_path = out_dir / f"{level}.json"
        if man_path.exists():
            built[level] = {"manifest": json.loads(
                man_path.read_text(encoding="utf-8"))}
    built = {lv: built[lv] for lv in EMITTED_LEVELS if lv in built}
    write_text(out_dir / "LABELS_REPORT.md", render_report(built, args.balance))
    print(f"[labels] wrote {len(built)} level manifests + LABELS_REPORT.md "
          f"to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
