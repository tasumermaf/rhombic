# IP boundary specification — rhombic public repository

Declared 2026-09-05 (PI ruling of the same date). This is the boundary every
public artifact of this repository is certified against: the audit lens
`ip-boundary`, the pre-commit and pre-push guard, and the release checks all
read this document as the spec. The protected set itself is never listed here
or anywhere in the public tree; it lives in a private file the guard reads at
run time.

## Protected (never in the public tree, in any encoding)

1. **The corpus spellings.** The literal Latin-letter spellings of the 24
   symbolic inscriptions that define the corpus distribution, and of the 14
   related names of power. A glyph variant and its ASCII transliteration are
   the same name; so are lowercase, snake_case, split-token, base64 and hex
   forms.
2. **The values bound to them.** The integer assigned to each inscription and
   each name. Protected in every form: bare, thousands-separated, as a
   product or sum that evaluates to it, as a one-per-line restatement, as an
   ordering that ranks the set, or as any published quantity from which a
   value or the set's endpoints can be recovered (raw totals that divide to
   the spread, full corpus-weighted spectra, per-card rank orders).

## Public

The ideas, procedures and results. Explicitly public: the eight tracked
primes, the prime-to-vertex assignment and the mapping procedure, aggregate
statistics that do not invert to a value (Fiedler values, lambda_max,
percentiles, hit counts), organisation and product names (TASUMER MAF,
LIOTHIL, MERIDIAN), and the Greek isopsephy of Greek words. A public item
becomes a violation only when it lets a reader recover a protected name or
value.

**Amendment 2026-09-06 (PI ruling).** Two items previously listed as public
are withheld from the public tree from this date, because each is a
constraint on the protected set even though neither inverts to a value:

3. **The per-card prime-presence matrix** (which of the eight tracked primes
   thread through each named card; formerly `TESSITURA_MATRIX` in
   `scripts/generate_weave.py`, and the weave images rendered from it). Its
   zeros are non-divisibility statements keyed to named cards; with the public
   per-prime census it narrowed one card's value to two candidates
   (verification of 2026-09-05). It lives in
   `rhombic/data/tessitura_private.json` (gitignored); the public script and
   images use a placeholder braid.
4. **The corpus distribution's coefficient of variation and skewness numbers**
   (Paper 2). Shape statistics of the normalized set; not invertible, but a
   filter against candidate reconstructions. The qualitative statement (high
   variance, heavy right tail) stays; the numbers do not.

## Where the private data lives

`rhombic/data/corpus_private.json` and the `*_private.json` sidecars beside
it (gitignored), plus `tests/test_corpus_private.py` (gitignored). Every public
code path runs without them; `rhombic.corpus.corpus_available()` reports
`False` outside the private environment.

## How to certify

Run the guard over the tree with the private set loaded; it prints its
protected-set size on every run and refuses to certify without values. A
fresh-context verification that builds its own detectors and attempts
outside-reader reverse engineering is required after any push that touches
results, papers, audits or figures.
