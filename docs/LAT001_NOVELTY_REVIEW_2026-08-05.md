# LAT-001 Novelty Literature Review

**Date:** 2026-08-05
**Author:** Meridian (literature-review instance), commissioned by main
**Subject:** Novelty of the LAT-001 intersection claim
**Target document:** `rhombic/docs/CARD_LAT001_DRAFT_2026-08-04.md` §6, open item (1)
**Status:** Review complete. Verdict below is binding on abstract language until superseded.
**Committed:** NO (per instruction).

---

## 0. The claim under test

As stated in the commission:

> "the latent-reasoning literature does not ask what TOPOLOGY/CONNECTIVITY the
> latent reasoning substrate should have, and the geometric-deep-learning
> literature does not ask about continuous-CoT step counts — the intersection
> is unclaimed."

This review tests that claim as stated, and reports where it is too strong.

---

## 1. Queries run (verbatim)

WebSearch, in execution order:

1. `continuous chain-of-thought latent reasoning graph topology connectivity`
2. `"Reasoning by Superposition" continuous chain of thought graph reachability arXiv 2505.12514`
3. `Coconut continuous latent reasoning follow-up 2026 graph structure`
4. `graph diameter bound number of reasoning steps transformer inference`
5. `oversquashing graph rewiring expander curvature GNN diameter`
6. `FCC lattice face-centered cubic 12-connectivity neural network architecture topology`
7. `papers citing "Reasoning by Superposition" continuous thought steps graph family diameter 2026`
8. `"continuous thought" steps "diameter" graph reachability empirical validation transformer`
9. `graph structure effect transformer learning reachability random regular vs lattice comparison`
10. `latent reasoning survey 2026 continuous thought open problems architecture design`
11. `"Topology of Reasoning" reasoning graph properties diameter large reasoning models 2506.05744`
12. `small-world lattice connectivity sparse attention topology transformer path length design`
13. `"Transformers can learn connectivity in some graphs but not others" graph families`
14. `latent reasoning substrate topology "what topology" latent space reasoning steps design choice`
15. `graph rewiring oversquashing applied to transformer chain-of-thought reasoning steps LLM`
16. `expander graph diameter design neural architecture reduce reasoning depth sequential steps`
17. `rhombic dodecahedral lattice machine learning topology benchmark connectivity comparison`
18. `"coordination number" OR "degree" lattice topology comparison deep learning substrate cubic alternative`
19. `LLM graph reasoning benchmark varying graph density degree diameter hops accuracy 2026`
20. `"latent reasoning" OR "continuous CoT" topology of the reasoning substrate unexplored open question`
21. `"Transformers need glasses" over-squashing information language tasks Barbero representational collapse`
22. `"Graph Structure of Neural Networks" relational graph clustering coefficient average path length You Leskovec`
23. `continuous chain of thought ProsQA task graph generation structure varying graph family experiment`
24. `degree-preserving rewiring controlled experiment matched node count reasoning steps neural model diameter manipulation`
25. `geometric deep learning latent reasoning intersection "continuous chain-of-thought" graph curvature rewiring 2026`
26. `Coconut continuous latent reasoning advantage "branching factor" graph reasoning tasks discrete CoT`
27. `"number of continuous thoughts" OR "latent steps" scales with "shortest path" OR "distance" empirical law transformer`

**Citation-graph sweep:** the complete citing-paper list for arXiv:2505.12514 was
pulled from the Semantic Scholar graph API
(`api.semanticscholar.org/graph/v1/paper/arXiv:2505.12514/citations`),
returning **44 citing works**, all 2026. Every title was screened against the
claim; the ones that could plausibly bear on it are itemised in §2.B.

---

## 2. The closest works

### A. The anchor and its direct theory line

| # | Work | What it does | Does it claim the intersection? |
|---|---|---|---|
| 1 | **Reasoning by Superposition** (arXiv:2505.12514, NeurIPS 2025; Zhu, Hao, Hu, Jiao, Russell, Tian) | Proves a 2-layer transformer with **D** continuous-thought steps solves directed graph reachability, D = graph diameter; continuous thoughts hold a superposition of BFS frontiers. Discrete CoT needs O(n²) decoding steps. | **Partially — it is the source of the diameter→step-count bound.** But it is a *constructive upper bound* plus a training-dynamics check on one task-graph family (a ProsQA subset). It never varies the graph family, never manipulates diameter at matched N, and never tests whether diameter is *sufficient* — i.e. whether topology has residual effect once distance is conditioned on. That gap is exactly LAT-001. |
| 2 | **Emergence of Superposition: Training Dynamics of Chain of Continuous Thought** (arXiv:2509.23365) | Analyses gradient dynamics by which superposition emerges. Key structural quantity is **node in-degree** (`d_u`, compared against `d_max`); uses the Zhu et al. ProsQA subset. | No. Verified by term search of the full HTML: no "diameter", "lattice", or topology-variation analysis. In-degree enters as a *local* quantity inside the dynamics, not as a manipulated substrate property. **Useful to us:** this paper randomly permutes vertex indices in train and test "to avoid prediction bias" — independent precedent for LAT-001's mandatory node-relabeling control (§3.1 of the card). |
| 3 | **Continuous CoT Enables Parallel Exploration and Reasoning / CoT2** (arXiv:2505.23648) | Continuous-valued CoT tokens; theory + experiments on ProntoQA/ProsQA at fixed 5-hop. | No. Verified: does not vary topology at matched node count, does not relate step count to diameter; one graph family (query-specific DAG subgraphs), not systematically varied families. |
| 4 | **Coconut — Training LLMs to Reason in a Continuous Latent Space** (arXiv:2412.06769) | The origin of continuous thought. Latent reasoning shows emergent BFS-like behaviour; advantage is largest on planning-heavy ProsQA. | No. The literature reports the advantage is larger for graphs with **high branching factor** — a structural property — but this is an observation about *when continuous beats discrete*, not a step-count law and not a controlled topology manipulation. This is the nearest "structure matters" statement in the latent-reasoning literature and should be cited as such. |

### B. Screened from the 44 citing works (2026) — the plausible pre-emptions

| # | Work | What it does | Why it does not claim the intersection |
|---|---|---|---|
| 5 | **Capabilities and Fundamental Limits of Latent Chain-of-Thought** (arXiv:2602.01148) | Theory + experiment on why latent CoT excels at exploration (ProsQA 97.0%) but fails at computation (GSM8K 34.1%). | Splits capability by *task type*, not by substrate topology. No diameter-indexed step-count law; no matched-N topology comparison. (Abstract fetch truncated — see §5.) |
| 6 | **The Illusion of Superposition? A Principled Analysis of Latent Thinking** (arXiv:2604.06374) | Tests whether superposition is actually used, across training-free / fine-tuned / from-scratch regimes. | Manipulates **training regime**, not task-graph structure. No step-count-vs-diameter analysis. |
| 7 | **Do Latent-CoT Models Think Step-by-Step?** (arXiv:2602.00449) | Mechanistic study on strictly sequential polynomial-iteration tasks; varies **hop length** (2-hop, 3-hop, longer). | Varies path *length* on a fixed sequential structure — closest to a "distance" manipulation, but the substrate is a chain, so distance and topology are not separable. No topology contrast. |
| 8 | **Training Continuous CoT Models: A Tale of Two Regimes** (arXiv:2607.16972) | Two regimes of direct vs indirect supervision for compressing traces. | Methodological (supervision), not structural. No graph-family variation. |
| 9 | **Dynamics Within Latent Chain-of-Thought: Empirical Study of Causal Structure** (arXiv:2602.08783) | Finds latent influence graphs are dominated by **skip connections**, unlike the sequential topology of explicit CoT. | **Topology of the model's own influence graph**, discovered post hoc — not the topology of the task substrate, and not a design variable. Different object entirely; see §3.C. |
| 10 | **SuperThoughts: Reasoning Tokens in Superposition** (arXiv:2606.13862); **MUX: Continuous Reasoning via Multiplexed Tokens** (arXiv:2607.18264); **Generative Recursive Reasoning** (arXiv:2605.19376); **Bridging Latent and Explicit Reasoning with Looped Transformers** (arXiv:2606.31779) | Representative of the bulk of the 44: new latent-reasoning *mechanisms, supervision schemes, or architectures*. | None manipulates the task graph's topology as an independent variable. The 44 citing works cluster into mechanism / interpretability / supervision / application; **not one is a substrate-topology study.** |

### C. Transformers × graph structure (no continuous CoT)

| # | Work | What it does | Why it does not claim the intersection |
|---|---|---|---|
| 11 | **Transformers Can Learn Connectivity in Some Graphs but Not Others** (arXiv:2509.22343) | **The single closest work.** Trains transformers on directed-graph connectivity across graph families; finds **grid dimensionality strongly predicts learnability** (higher-dimensional grids are harder), and disconnected-component graphs defeat the model. Full abstract verified verbatim. | Varies substrate topology — but (a) **no continuous CoT**: verified absent, the task is learned end-to-end with discrete decoding; (b) endpoint is **learnability/accuracy**, never a *step count*; (c) grid dimensionality co-varies degree, diameter and node count, so no distance-conditioned inference is possible. **Must be cited; it is why the abstract sentence cannot say "no one varies graph structure."** |
| 12 | **Transformers Provably Learn Algorithmic Solutions for Graph Connectivity / When Do Transformers Learn Heuristics?** (arXiv:2510.19753) | Model capacity is tied to **diameter** (learns the algorithm when instances are within capacity, diameter ≤ 3^L for L layers; otherwise falls back to a degree heuristic). | Diameter appears as an *architectural capacity bound* on a discrete-decoding model, not as a regressor for continuous-thought step count. No topology contrast at matched N. Genuinely adjacent — cite it. |
| 13 | **Understanding Transformer Reasoning Capabilities via Graph Algorithms** (arXiv:2405.18512) | Depth/width requirements for graph tasks under an MPC-style model; parallelisable tasks need log depth. | Expressivity theory over task classes; no substrate-topology manipulation, no continuous thoughts. |
| 14 | **Towards an Understanding of Stepwise Inference in Transformers: A Synthetic Graph Navigation Model** (arXiv:2402.07757) | Autoregressive traversal on synthetic graphs to study stepwise inference. | Discrete stepwise inference; the graph is a controlled testbed, but topology families are not contrasted for step count. |
| 15 | **Lower Bounds for CoT Reasoning in Hard-Attention Transformers** (arXiv:2502.02393) | DAG reachability needs CoT of length Ω(\|E\| log \|V\|) in hard-attention transformers. | Discrete-token lower bound; scales with edge count, not with a topology contrast, and not for continuous thoughts. |

### D. "Topology of reasoning" — the name collision

| # | Work | What it does | Why it does not claim the intersection |
|---|---|---|---|
| 16 | **Topology of Reasoning: Understanding LRMs through Reasoning Graph Properties** (arXiv:2506.05744, NeurIPS 2025) | Builds a *reasoning graph* by clustering hidden states across reasoning steps; measures **cyclicity, diameter, small-world index**; distilled models show larger diameters and ~6× small-worldness, correlating with accuracy. | **Name collision, opposite direction.** The graph is *induced by the model's trajectory* and measured as an outcome; diameter is a dependent variable describing exploration breadth. LAT-001 fixes the substrate a priori and treats distance as the regressor. No continuous thoughts, no substrate manipulation. **Cite it explicitly to disambiguate** — a reviewer who knows this paper will otherwise assume overlap. |

### E. Geometric deep learning / oversquashing / rewiring — the closest adjacent literature

| # | Work | What it does | Why it does not claim the intersection |
|---|---|---|---|
| 17 | **Graph Rewiring to Mitigate Over-Squashing / Over-Smoothing: surveys** (arXiv:2411.17429; arXiv:2210.11790 FoSR; arXiv:2302.06835 effective resistance; arXiv:2208.03471 information contraction) | Diameter, spectral gap, effective resistance and Ricci curvature predict oversquashing in **message-passing GNNs**; rewiring modifies the graph to improve information flow. | Different model class (MPNN, not a transformer doing latent reasoning); the graph is modified *to help the model*; the endpoint is node/graph task accuracy; "depth" is an architectural layer count, not a per-query inference-time step budget. Nothing here touches continuous-CoT step count. See §3.A for the precise differentiation. |
| 18 | **Expander Graph Propagation** (Deac et al.); **Deep Expander Networks** (arXiv:1711.08757); **Schreier Coset Graph Rewiring** (arXiv:2607.27479) | Expanders (low diameter, sparse) used as propagation or wiring substrates for efficient information routing. | Substrate topology *is* the design variable — but for GNN message passing / network sparsification, and the endpoint is accuracy-per-parameter or bottleneck mitigation. Never continuous-thought step count. |
| 19 | **Transformers Need Glasses! Information Over-Squashing in Language Tasks** (arXiv:2406.04267, NeurIPS 2024; Barbero et al.) | **The one genuine bridge**: imports over-squashing into decoder-only transformers, showing representational collapse in the final token via signal-propagation analysis over the causal attention graph. | The graph is the **causal attention graph over the token sequence** (a fixed causal DAG), not a task substrate whose topology is manipulated. No reasoning-step-count law, no continuous thoughts. This is the closest the two literatures have come to touching, and it is still one object removed. |
| 20 | **Graph Structure of Neural Networks** (arXiv:2007.06559, ICML 2020; You, Kaiming He, Leskovec, Xie) | Represents a network as a *relational graph*; finds performance is a smooth function of **clustering coefficient and average path length**, with a "sweet spot." | The strongest precedent anywhere for "wiring topology is a measurable design variable with a path-length-indexed optimum" — but the object is **feedforward network wiring**, evaluated on image classification accuracy. No reasoning, no step count, no task graph. Cite as the intellectual ancestor of the framing, not as a competing claim. |
| 21 | **Sparse-attention topology work** (BigBird and successors; brain-inspired sparse training, arXiv:2501.19107) | Designs attention connectivity to approximate small-world properties, balancing average shortest path against clustering. | Topology of the **attention pattern** chosen for efficiency; endpoint is LM quality/efficiency. Not a task substrate, not a step-count law. |
| 22 | **FCC / rhombic-dodecahedral lattices in computing** (e.g. rhombic dodecahedron topology for banking big data; FCC lattices in protein-structure prediction; RD grid coordinate systems) | Uses 12-coordination FCC/RD lattices for optimisation-network topology, protein folding search, and 3-D digital geometry. | Confirms 12-connectivity is used as a *computational substrate* in other fields, and that FCC/RD carries the known coordination-number-12 and Voronoi-of-FCC facts. **No machine-learning-reasoning use found anywhere.** No contact with transformers or latent reasoning. |

---

## 3. The adjacent literatures, mapped

### A. Oversquashing / graph rewiring (expected closest — confirmed closest on the GDL side)

The differentiation must be stated at four levels, because a reviewer will reach
for this literature first:

| Axis | Oversquashing / rewiring | LAT-001 |
|---|---|---|
| **Model class** | Message-passing GNNs (Barbero et al. extends to the causal attention graph of a decoder-only transformer) | 2-layer transformer performing **continuous-thought latent reasoning**; no message passing |
| **What the graph *is*** | The architecture's computation graph, or input data whose structure the architecture inherits | **Task content** — an edge list presented in context; the architecture is unchanged across arms |
| **Direction of the manipulation** | The graph is **modified to help the model** (add/remove edges, rewire by curvature or spectral gap) | The substrate is **held fixed and un-helped**; topology is an assigned condition, and the one rewire arm is a *control* (degree-preserving, matched N/E/degree sequence) that isolates arrangement from degree |
| **Endpoint** | Task accuracy, information contraction, effective resistance, spectral gap; "depth" = architectural layer count | **Step count**: next-hop-on-shortest-path correctness as a function of continuous-thought budget `c` and directed distance `d`, i.e. the law `k*(d)` at inference time |

The single sentence that carries the distinction: *the rewiring literature asks
how to change a graph so message passing can cope with it; LAT-001 holds the
graph fixed and asks whether its directed-distance structure is a sufficient
statistic for how many latent steps a transformer needs.*

### B. Latent-reasoning literature

Verified structurally, not merely asserted. A term search of **A Survey on
Latent Reasoning** (arXiv:2507.06203) returned **no occurrences** of "topology",
"diameter", "lattice", or "connectivity" as design or open-problem language; the
survey organises the field as vertical recurrence (looped/layer depth),
horizontal recurrence (state evolution), interpretability, and diffusion. The
substrate is discussed as *hidden states, residual stream, layers* — the
representational medium — never as a graph whose connectivity could be chosen.
The 44 citing works of 2505.12514 corroborate: mechanism, supervision,
interpretability, and application; no substrate-topology study. (Caveat: the
survey's §6 conclusion was truncated in the fetch.)

### C. "Topology" as a word used for three different objects

A precision hazard worth recording, because three literatures use "topology of
reasoning" for three non-overlapping objects:

1. **Topology of the induced trajectory graph** — 2506.05744, 2602.08783.
   Measured as an *outcome* of the model's behaviour.
2. **Topology of the architecture** — attention patterns, relational graphs,
   expander wiring (2007.06559, BigBird, EGP). Chosen as a *design variable*.
3. **Topology of the task substrate** — LAT-001. Assigned as an *experimental
   condition*, with the endpoint being inference-time step count.

Only (3) is LAT-001's object, and (3) is the one nobody occupies.

---

## 4. VERDICT

### **UNCLAIMED**, with two boundary conditions that constrain the wording.

No work found asks whether the *topology of a fixed task substrate* affects
*continuous-thought step count* at matched node count. The intersection is
genuinely open. Two facts, however, make the naive phrasing indefensible:

- **Boundary 1 — the diameter→step-count relation is already claimed as theory.**
  arXiv:2505.12514 *is* the D-step bound. LAT-001 may not present "diameter
  governs continuous-thought step count" as a new idea. What is new is testing
  its **sufficiency** in a learned model — whether topology has residual effect
  after conditioning on `d`. Outcome B of the card is precisely a falsification
  of sufficiency, and that framing is safe; "we discover that diameter matters"
  is not.

- **Boundary 2 — graph structure has been varied for transformer connectivity.**
  arXiv:2509.22343 varies grid dimensionality and finds it predicts learnability;
  arXiv:2510.19753 ties model capacity to diameter. The abstract therefore may
  **not** say the literature has never varied graph structure. It must say the
  variation was measured against *learnability under discrete decoding*, not
  *continuous-thought step count*.

### Abstract-safe sentence (use verbatim)

> Diameter-based step-count bounds for continuous chain-of-thought have not been
> tested for sufficiency: existing empirical work holds the task-graph family
> fixed, and the work that does vary graph structure for transformer
> connectivity measures learnability under discrete decoding rather than
> continuous-thought step count.

Longer variant, if the abstract has room for the manipulation:

> The continuous-thought literature derives step-count bounds from graph diameter
> but has not tested whether diameter is sufficient, holding the task-graph
> family fixed; work that does vary graph structure for transformer connectivity
> tasks measures learnability under discrete decoding. We test sufficiency
> directly, comparing fixed substrates at matched node count — including a
> degree-preserving rewire that collapses diameter at fixed degree sequence.

### Phrasings that are NOT safe (would be refuted on sight)

- ~~"No prior work relates graph diameter to reasoning step count."~~ — 2505.12514 does, by construction.
- ~~"No prior work studies how graph structure affects transformer graph reasoning."~~ — 2509.22343, 2510.19753 do.
- ~~"The topology of reasoning is unstudied."~~ — 2506.05744 owns that phrase for a different object.
- ~~"Diameter is unexplored in geometric deep learning."~~ — it is a standard oversquashing predictor.

### Required citations in any LAT-001 write-up

2505.12514 (anchor/bound) · 2509.22343 (closest structural variation) ·
2510.19753 (diameter as capacity bound) · 2506.05744 (disambiguate the name) ·
2406.04267 (oversquashing→transformers bridge) · 2411.17429 or FoSR 2210.11790
(rewiring baseline) · 2007.06559 (topology-as-design-variable ancestor) ·
2509.23365 (independent precedent for node relabeling).

---

## 5. Verified vs. taken on report

**Verified — fetched and read directly:**

- arXiv:2509.22343 — abstract page fetched, **full abstract quoted verbatim**; graph families and absence of continuous CoT confirmed from that text.
- arXiv:2509.23365 — full HTML fetched; in-degree treatment, ProsQA-subset provenance, vertex-permutation control, and **absence of any diameter/lattice/topology analysis** confirmed by term search.
- arXiv:2505.23648v3 — HTML fetched; single-graph-family scope and absence of matched-N topology variation confirmed.
- arXiv:2507.06203 (latent-reasoning survey) — HTML fetched; term search for topology/diameter/lattice/connectivity returned **no hits**. §6 conclusion truncated in the fetch, so "not listed as an open problem" is verified for the body but not for the conclusion.
- Semantic Scholar citation graph for arXiv:2505.12514 — API fetched; **complete 44-item citing list retrieved** and screened title-by-title.

**Partially verified — abstract page fetched, but the fetch returned only the opening fragment of the abstract:**

- arXiv:2510.19753, 2607.16972, 2602.01148, 2604.06374, 2602.00449, 2606.02248.
  For these six, the negative findings ("does not vary topology", "no step-count
  law") rest on the retrieved fragment plus title/metadata plus corroborating
  search snippets — **not on a full-text read**. Confidence: high but not
  complete. Two are worth a full-text pass before submission if the abstract
  leans on them: **2602.01148** (Capabilities and Fundamental Limits) and
  **2510.19753** (diameter-as-capacity), since both make diameter- or
  limit-shaped claims that could sharpen or dent the framing.

**Taken on report — judged from search-result snippets, titles and abstracts without direct fetch:**

- arXiv:2505.12514 (the anchor) — content corroborated across five independent
  snippets, the NeurIPS/ICML listings, the official GitHub, and the senior
  author's own public statement of the D-step result; treated as reliable
  despite no direct fetch.
- arXiv:2412.06769 (Coconut), 2506.05744, 2602.08783, 2406.04267, 2007.06559,
  2402.07757, 2405.18512, 2502.02393, 2501.19107.
- Oversquashing/rewiring corpus: 2411.17429, 2210.11790, 2302.06835, 2208.03471,
  2607.27479, 1711.08757, Expander Graph Propagation.
- The remaining ~35 of the 44 citing works — screened on title only, having been
  classified into mechanism / supervision / interpretability / application
  clusters with no topology signal in the title.
- FCC/RD computing applications (§2.E #22).

**The one claim I could not fully close:** an exhaustive title-only screen of 35
citing works cannot rule out a topology-varying experiment buried in a paper
whose title advertises something else. The risk is low — such a result would
normally surface in the title or abstract — but it is nonzero, and it is the
residual uncertainty behind the UNCLAIMED verdict.

---

*Review conducted 2026-08-05. 27 distinct web queries plus a full citation-graph
sweep. No files committed.*
