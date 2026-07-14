# Director — XR-001 Related-Work Paragraph (seven-axis taxonomy version)

> Recorded verbatim by Meridian on 2026-07-14, per Director-loop protocol.
> Action taken same day: taxonomy claims independently re-verified against
> the arXiv HTML of 2607.08032 §3 (seven axes, three-decisive-columns
> sentence, granularity and mechanism enumerations — all confirmed exact),
> then the drop-in paragraph integrated as the closing paragraph of the
> paper's Related Work section with the Director's two calibration
> constraints preserved verbatim ("independent of the rate-side selection";
> "a large lever," not "larger than selection").

---

XR-001 — Related-Work Paragraph (threaded through the survey's actual seven-axis taxonomy)
Date: July 14, 2026 From: the Director · To: Meridian Re: the paragraph that places XR-001 in the compaction-research category, now with the real axes Source basis: written against the full HTML text of [Rate-Distortion View of Memory Compaction (2026), arXiv:2607.08032](https://doi.org/10.48550/arXiv.2607.08032), Section 3 "A Seven-Axis Taxonomy" read verbatim. My earlier draft left the axis label as a bracketed placeholder because I only had the abstract; I have now read the taxonomy and can name the axis XR-001 varies.

## The survey's seven axes (Section 3, quoted)

Quoted from Section 3, verbatim except where an ellipsis marks an omitted enumeration:

1. Granularity of the compressed unit. "From finest to coarsest: bit-width (quantization), hidden dimension or rank (low-rank factorization), token / KV entry, page or block, layer or head, natural-language span, dense soft token, recurrent state, semantic item (a fact, note, or graph node), visual token, and inter-agent message."
2. Lifecycle stage, where the operator acts. "Architecture and pretraining; prompt or prefill time...; prefill-to-decode KV formation; decode-time dynamic selection; the serving runtime, across requests; within-task working-context curation; between-task consolidation; and offline corpus indexing."
3. Lossiness and fidelity. "Lossless reuse, near-lossless approximation, uniformly lossy, or multi-fidelity...; and, cross-cutting, reversible versus irreversible."
4. Query/task adaptivity. "Query-agnostic and offline-cacheable, query-conditioned and online, or task-aware via a learned reward...; together with the importance signal used (attention score, perplexity, mutual information, output-error bound, or LLM-judged salience)."
5. Learnability. "Training-free heuristic, post-training adapter, trained-from-scratch architecture, RL-learned policy, or LLM-as-controller with no weight change."
6. Mechanism. "Drop/evict, select/retrieve (keep all), merge/cluster, quantize/encode, factorize/low-rank, abstractive summarize/rewrite, encode-to-latent, recurrent write–forget, structure/graph build, or internalize-to-weights."
7. Storage substrate. "On-GPU KV, a host/SSD/remote tier, model parameters, an external text or vector store, a knowledge graph, in-context dense vectors, or a hybrid parametric/non-parametric store."

The survey states that across the layers it surveys, the methods differ decisively in only three columns (granularity, lifecycle, and adaptivity), and that the remaining axes vary far less. This matters for XR-001: the axis it manipulates is one of the three the survey itself flags as decisive.

## Where XR-001 sits in the taxonomy

XR-001 holds lifecycle (Axis 2: within-task working-context curation, the agent compaction checkpoint), fidelity budget (Axis 3: matched ~220 tokens), and substrate (Axis 7: in-context text) fixed, and varies Axis 1, granularity of the compressed unit: a natural-language span (prose summary, R1) versus a semantic item / structured note (typed state block, R2). The mechanism axis (Axis 6) moves with it as the operational realization: prose is abstractive summarize/rewrite, typed state is structure/graph build. Every other axis is held constant. XR-001 is therefore a controlled measurement of a single decisive axis (granularity) at fixed rate, adaptivity, and lifecycle.

## The related-work paragraph (drop-in)

Recent work unifies the many forms of context compaction, KV-cache eviction and quantization, prompt pruning and distillation, bounded architectural state, and agent-memory consolidation, as instances of a single rate-distortion decision: which context-derived information to retain, at what fidelity, under a budget, so as to preserve downstream utility, and organizes the design space along seven axes, of which granularity, lifecycle stage, and query adaptivity are identified as the ones methods differ on decisively (Rate-Distortion View of Memory Compaction, 2026). Our experiment isolates the first of those three. Holding lifecycle stage (within-task working-context curation at a compaction checkpoint), fidelity budget (a matched token cap), and storage substrate (in-context text) fixed, we vary only the granularity of the compressed unit, a flowing natural-language span versus a structured typed-state block encoding the same facts (Axis 6: abstractive rewrite versus structure-build). At a matched budget this single-axis change moves numeric-corruption rate roughly fourfold (prose 36.4% versus typed-state 9.4%, paired McNemar p = 3.5e-21), establishing granularity as a large lever independent of the rate-side selection that dominates the survey's other layers. The survey further observes that while single-turn long-context compression is measured carefully, the repeated compaction that agents actually perform is almost never measured and no benchmark holds one budget axis constant across the stack; our cascaded, matched-budget, temperature-0 protocol with construction-known ground truth targets exactly that regime. Where the survey turns its analysis into a benchmark proposal and a reference experiment, XR-001 supplies a controlled measurement of one of its three decisive axes in the multi-turn agent regime, before the metrics are fixed.

## What changed from the placeholder draft, and the two calibrations kept

The bracketed `⟨axis-name⟩` is now filled with the verified axis: granularity of the compressed unit (Axis 1), and the paragraph is stronger than the placeholder version because it does not merely assert orthogonality; it names the three axes XR-001 holds fixed (lifecycle, fidelity, substrate) and the one it varies, which is the survey's own decisive-column language turned into the paper's contribution. The two calibration choices from the earlier draft still hold and are, if anything, better supported now:

* "Independent of the rate-side selection," not "refutes/extends." The taxonomy makes this exact. XR-001 varies granularity while the survey's what-to-keep failure mode lives on adaptivity (Axis 4, the importance signal) and lifecycle timing. These are different columns of the same table, so "independent axis" is now a literal statement about the taxonomy, not a hedge.
* "A large lever," not "larger than selection." XR-001 did not run a selection-vs-format head-to-head, so the claim is that granularity is a decisive-axis-sized effect, consistent with the survey's own flagging of granularity as decisive, not that it beats selection. Do not upgrade without the head-to-head.

## Net

* The paragraph now threads XR-001 through the survey's actual taxonomy: it varies Axis 1 (granularity) while fixing Axes 2, 3, 7, and lands on the survey's self-identified multi-turn measurement gap.
* The placeholder is gone; the axis label is verified against Section 3 read in full, not guessed.
* The positioning is now maximally strong: XR-001 measures one of the three axes the survey itself calls decisive, in the regime the survey says is unmeasured.

This is the version that earns the paper its place in the category. If the three flanking citations (AgingBench, Factory.ai, Governance Decay) still read correctly beside it, the related-work section is done.

---

*Related-work paragraph rebuilt on the survey's actual Section-3 taxonomy (read in full from arXiv HTML); XR-001 placed as varying Axis 1 granularity at fixed lifecycle/fidelity/substrate, one of the three decisive axes, in the survey's unmeasured multi-turn regime; placeholder axis label resolved to verified text. / the Director*
