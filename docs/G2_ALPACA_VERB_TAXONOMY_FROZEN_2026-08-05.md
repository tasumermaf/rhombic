# G-2 — The Frozen Alpaca Verb Taxonomy (for Director review before L2 fires)

**Filed 2026-08-05 by Meridian**, per the Director's G-2 grade condition of
2026-08-04 ("send the frozen taxonomy document itself, not a summary").
**Source of truth:** `scripts/granularity_labels.py` (frozen at commit
`8fa3d77`, in force unchanged at HEAD). The dicts below are quoted
**verbatim** from the code — this document adds provenance and rationale,
never a paraphrase of the label space.

## What it is

An authored label space inside a registered card (the one place authorial
degrees of freedom enter after registration — the reason for this review).
The rule: an alpaca instruction is classified by its **first word**,
lowercased, matched against the family lists below. A first word matching no
list falls to the residual `other`. L3 = 8 families (7 named + residual);
L2 = a frozen pairwise merge, so **L2 is a strict coarsening of L3** by
construction. Alpaca L1 is native (no_input / with_input) and does not use
this taxonomy.

## The frozen L3 families (verbatim, `ALPACA_VERB_L3`)

```python
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
```

## The frozen L2 merge (verbatim, `ALPACA_VERB_L2_MERGE`)

```python
ALPACA_VERB_L2_MERGE: dict[str, tuple[str, ...]] = {
    "produce_text": ("generate", "transform"),
    "explain_answer": ("describe", "question"),
    "retrieve_analyze": ("retrieve", "analyze"),
    "framed_other": ("frame", "other"),
}
```

## Realized class masses (from the frozen Stage-0 build, LABELS_REPORT.md)

L3 (5 seeds/class planned): generate 11,003 · retrieve 6,905 · describe
3,806 · other 3,012 · analyze 2,941 · transform 2,941 · frame 2,831 ·
question 2,561 training rows. L2 (10 seeds/class): produce_text 14,444 ·
retrieve_analyze 10,346 · explain_answer 6,867 · framed_other 6,343.
All clear the D4 floor (post-val reading per G-9). Discard under this
taxonomy: zero (the residual is a class, not a discard — unlike code's G-3
rule, and deliberately: alpaca first-words are dense enough that `other`
stays a coherent minority class rather than 43% of the pool).

## What review should probe (stated against ourselves)

1. **`frame` is the weakest family** — a stopword-heavy list ("the", "a",
   "in") capturing instructions that open with context rather than a verb.
   It is semantically "framing-first phrasing," not an intent class. Its
   merge partner is the residual (`framed_other`), which quarantines the
   two least-semantic families together at L2 — but at L3 they stand as
   separate classes, and a low κ on those two cells would be a taxonomy
   artifact, not a granularity finding. The per-class κ reporting (D5)
   exposes this; the Director may prefer a pre-declared footnote.
2. **First-word-only** is maximally reproducible (no model, no seed, no
   drift) at the cost of ignoring instruction bodies. That is the trade we
   chose: T2 keyword rules per the card's own bracket, zero annotator
   dependence.
3. **British/American doublets** (summarise/summarize, categorise) are
   split across families deliberately where usage differs (summarise →
   describe; summarize → transform). This mirrors observed alpaca usage
   but is a judgment call worth the Director's eye.

The taxonomy affects **L2/L3 only** (both gated). No run has trained on it.
A Director edit is a one-constant change + label re-emit + new sha256s —
cheap before L2, expensive after.
