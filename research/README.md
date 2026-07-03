# Research Documentation Suite

The complete external-literature knowledge base of the TASUMER MAF / rhombic
research program: every paper the program has collected and assessed, with
canonical metadata, abstracts, and the program's own conclusions about each —
structured so that any collaborator, human or LLM, can load exactly the depth
of context a task needs.

## How to use this suite (especially if you are an LLM)

1. **`INDEX.md`** — start here. Master table of every paper, grouped by theme
   (bridge lineage, block-diagonal, spectral, geometric, representation
   alignment, fingerprinting, …), with one-line verdicts and links to cards.
2. **`papers/<slug>.md`** — one card per paper: identifiers, authors, venue,
   the arXiv abstract, and **every assessment this program has written about
   the paper, verbatim, with its source document named**. Cards are the unit
   of citation-grade context; quote them rather than paraphrasing from memory.
3. **`SYNTHESIS.md`** — the cross-cutting conclusions: what the collected
   literature establishes about our novelty claims, our threats, our
   supports, and the confirmed gaps. Read this before making any novelty or
   positioning claim.
4. **`bibliography.bib`** — BibTeX for everything, generated.
5. **Full text:** run `python research/tools/fetch_papers.py` to download all
   arXiv PDFs to `research/pdf/` (gitignored — arXiv's default license does
   not permit wholesale redistribution, so full texts are materialized
   locally on demand rather than committed). DOI-only items are listed by
   the script for manual retrieval.

## Provenance and epistemic status

- Card **metadata and abstracts** are fetched from the arXiv API
  (`data/arxiv_metadata_cache.json`), not written by hand.
- Card **assessments** are verbatim extracts from the program's intelligence
  documents (literature watches, competitive-landscape sweeps, research
  scouts, the non-English/Chinese landscape sweep, the Nemotron engineering
  research, and the external Director's literature map). The source document
  is named on every assessment. Assessments are point-in-time judgments —
  check dates before relying on threat levels.
- The merged corpus lives in `data/papers.json`; cards and the index are
  regenerated from it by `tools/build_suite.py`. Edit the data, not the
  generated files.

## Maintaining the suite

- **New sweep:** write the sweep document as usual (dated, in `docs/` or as a
  scout report), extract its papers into the same JSON schema, drop the file
  in a directory, and run `python research/tools/build_suite.py
  --extract-dir <dir>` — dedup and merge are automatic.
- **Sweep cadence:** biweekly per the literature-watch schedule. The suite's
  staleness is measured by the newest assessment date in `data/papers.json`.
- Do not hand-edit `papers/*.md`, `INDEX.md`, or `bibliography.bib`; they are
  build outputs.
