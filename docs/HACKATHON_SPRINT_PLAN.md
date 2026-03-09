# Hermes Agent Hackathon Sprint Plan

> **Competition:** Nous Research — Hermes Agent Hackathon
> **Deadline:** EOD Sunday, March 16, 2026
> **Prizes:** $7,500 / $2,000 / $500
> **Judging:** Creativity, usefulness, and presentation
> **Submission:** Tweet @NousResearch with video demo + brief writeup → post tweet link to Discord #hermes-agent-hackathon-submissions
> **Team:** Timothy Paul Bielec (Promptcrafted) + Minta Karlsson (Alvdansen)

---

## What We're Building

**Name:** `rhombic-agent` — a Hermes Agent that thinks in 12 dimensions.

A Hermes Agent equipped with the `rhombic` lattice topology library as
live tools, capable of running experiments, generating visualizations,
and explaining the geometry interactively. The agent embodies the thesis:
it uses the same rhombic architecture it's teaching you about.

**The pitch in one sentence:** "An AI research agent that proves cubic
lattices are leaving 6.1× connectivity on the table, then shows you
exactly how to claim it — keep your cube, add six bridges."

**Target audience for presentation:** Minta. If she gets it, everyone
gets it. She's technical (builds LoRA training pipelines) but hasn't
followed the lattice topology deep-dive. The presentation must make the
geometry tangible, the numbers visceral, and the implications obvious.

---

## Deliverables

### 1. Hermes Agent with Rhombic Tools

A Hermes Agent installation with custom tools and skills that expose
the `rhombic` library through natural conversation:

**Tools (Python, registered in hermes-agent):**
- `lattice_compare` — Build matched SC/FCC lattices, compute stats,
  return comparison table
- `fiedler_ratio` — Compute weighted Fiedler ratio for any weight
  distribution and scale
- `direction_weights` — Run direction-pair weighting experiment,
  return amplification results
- `permutation_control` — Run shuffled vs sorted permutation test,
  return p-value
- `prime_vertex_map` — Run prime-vertex exhaustive search on RD,
  return optimal mapping and p-value
- `spectral_analysis` — Compute weighted Laplacian spectrum of RD
  under four distributions
- `visualize_rd` — Generate rhombic dodecahedron visualization
  (matplotlib → image)
- `visualize_amplification` — Generate the amplification gradient
  bar chart
- `explain_mechanism` — Return structured explanation of bottleneck
  resilience mechanism at requested depth (intuitive / technical / full)

**Skills (Hermes Agent format):**
- `rhombic-tutorial` — Walk a user through the full thesis from
  "what is a lattice" to "6.1× amplification" in conversational steps
- `rhombic-experiment` — Run a custom experiment with user-specified
  parameters and explain results
- `rhombic-lora-concept` — Explain the RhombiLoRA forward vision
  with analogies and diagrams

**Context files:**
- `rhombic-context.md` — the thesis, key numbers, and vocabulary
  so the agent speaks accurately about the research

### 2. Presentation Website

A single-page site in the style of ryanpo.com/multigen — clean,
academic, visually stunning. Our aesthetic layer on top.

**Visual identity:**
- Colors from the 8-Law Weave: Cubic `#3D3D6B` (indaco), FCC `#B34444`
  (mattone). Dark background, light text. The palette tells the story.
- Hero: animated rhombic dodecahedron (Three.js or CSS 3D) with the
  cube skeleton visible inside it
- Typography: sharp serif for headings, clean sans for body. Academic
  but not sterile.

**Sections:**
1. **Hero** — "Keep Your Cube, Add Six Bridges" + animated RD
2. **The Problem** — One paragraph: every computation defaults to cubes.
   What if the default is wrong? Side-by-side cube vs RD visualization.
3. **The Numbers** — Key results as large, bold statistics with minimal
   explanation. 2.3× → 6.1×. p = 0.000025. 208 tests.
4. **The Mechanism** — Bottleneck resilience explained with a diagram.
   "Heterogeneous weights create bottlenecks. Cubes have 6 escape
   routes. The RD has 12." Interactive: slider that adjusts weight
   heterogeneity and shows the Fiedler ratio changing in real-time.
5. **The Agent Demo** — Embedded video of the Hermes Agent in action.
   The agent runs experiments, generates visualizations, explains
   results. Conversational interaction demonstrating the tools.
6. **The Forward Vision** — RhombiLoRA in 3 sentences. "The cube
   isn't the enemy — it's the skeleton. The RD contains it. Your
   existing infrastructure is already Phase 1." One-liner on Karkada
   et al. [2026]: "Recent theory proves data symmetry determines
   representation geometry. RhombiLoRA is the adapter that matches."
7. **Try It** — `pip install rhombic` + link to HF Space + link to
   GitHub
8. **Papers** — Links to Paper 1 and Paper 2 PDFs
9. **Citation** — BibTeX block

**Tech stack:** Static site. HTML/CSS/JS. Three.js for the 3D RD
visualization. Hosted on GitHub Pages or Cloudflare Pages (Timothy
already has the Cloudflare pipeline from portfolio site).

### 3. Video Demo (60-90 seconds)

Screen recording of the Hermes Agent in action:
- User asks "What happens when you add structure to lattice weights?"
- Agent runs `direction_weights` tool, shows 6.1× result
- Agent generates visualization inline
- Agent explains bottleneck resilience in plain language
- Quick cut to the permutation control: "Is this just bin count? No.
  p = 0.001. It's alignment."
- Close on the RhombiLoRA concept: "Keep your cube, add six bridges."

Voiceover by Timothy (optional, adds personality). Background music
if time allows. Screen recording + post in CapCut or similar.

### 4. Tweet + Writeup

Brief writeup (280 chars for tweet, longer in thread):
- Tweet: "Built a Hermes Agent that proves cubic lattices leave 6.1×
  connectivity on the table. It runs the experiments, generates the
  visualizations, and explains the geometry. Keep your cube, add six
  bridges. @NousResearch #HermesAgentHackathon [video] [link]"
- Thread: 3-4 tweets expanding on the findings, linking the website,
  the repo, and the papers.

---

## Sprint Schedule

### Day 1-2 (Mar 7-8): Foundation

**Goal:** Hermes Agent running with rhombic tools on Hermes server.

**Note:** hermes-agent is ALREADY installed on Hermes server (the machine
was named Hermes precisely because we clean-installed the Nous Research
Hermes framework from the ground up). Focus is on rhombic tool integration.

- [x] ~~Install NousResearch/hermes-agent on Hermes server~~ (DONE — pre-sprint)
- [x] Install `rhombic` library in the hermes-agent environment
- [x] Write custom tools wrapping rhombic library functions
- [x] Write context file with thesis, key numbers, and Karkada et al.
  connection (data symmetry → adapter topology)
- [x] Test: agent can run `lattice_compare` and return results
- [x] Test: agent can explain results in natural language

**Owner:** Timothy + Meridian

### Day 3-4 (Mar 9-10): Full Tool Suite + Skills

**Goal:** All tools working, skills tested, visualizations generating.

- [x] Implement all 9 tools listed above
- [x] Write `rhombic-tutorial` skill (conversational walkthrough)
- [x] Write `rhombic-experiment` skill (custom experiments)
- [x] Write `rhombic-lora-concept` skill (forward vision)
- [x] Test visualization generation (matplotlib → image files)
- [x] Test full conversational flow end-to-end
- [ ] Record rough demo footage for iteration

**Owner:** Timothy + Meridian

### Day 5-6 (Mar 11-12): Presentation Website

**Goal:** Website live and beautiful.

- [x] HTML/CSS/JS site structure (based on multigen layout analysis)
- [x] 8-Law Weave color palette implementation
- [x] Three.js animated RD hero visualization (cube skeleton visible)
- [ ] Interactive Fiedler ratio slider (Section 4) — DESCOPED to static bars
- [x] Embed figures from Paper 2 (amplification chart as CSS bars)
- [x] Responsive design (judges may view on mobile)
- [ ] Deploy to GitHub Pages or Cloudflare Pages
- [ ] Minta reviews for clarity — if she doesn't get it, rewrite

**Owner:** Timothy + Meridian (site) + Minta (review)

### Day 7 (Mar 13): Video Production

**Goal:** 60-90 second demo video, polished.

- [ ] Script the demo flow (user questions, agent responses)
- [ ] Screen record the Hermes Agent session
- [ ] Record voiceover (optional)
- [ ] Edit in CapCut: cuts, transitions, text overlays
- [ ] Music selection (if used — subtle, not distracting)
- [ ] Export at 1080p minimum

**Owner:** Timothy (recording + VO) + Minta (edit suggestions)

### Day 8 (Mar 14): Polish + Submission Prep

**Goal:** Everything refined. Website final. Video final.

- [ ] Final pass on all tool outputs for accuracy
- [ ] Website copy review — every number traces to Paper 2
- [ ] Video timing review — does it hold attention?
- [ ] Draft tweet text and thread
- [ ] Test all links (GitHub, PyPI, HF Space, website)
- [ ] Prepare Discord submission message

**Owner:** Both

### Day 9 (Mar 15): Buffer + Early Submit

**Goal:** Submit with a day of buffer.

- [ ] Submit tweet @NousResearch
- [ ] Post tweet link to Discord #hermes-agent-hackathon-submissions
- [ ] Post progress update to #hermes-agent-hackathon channel
- [ ] Verify everything is public and accessible

**Owner:** Timothy

### Day 10 (Mar 16): Deadline (buffer day)

Emergency fixes only. Everything should be submitted by Day 9.

---

## Architecture: How the Agent Works

```
┌─────────────────────────────────────┐
│         Hermes Agent (Nous)         │
│  ┌─────────────┐  ┌──────────────┐ │
│  │   Skills     │  │   Context    │ │
│  │ - tutorial   │  │ - thesis     │ │
│  │ - experiment │  │ - numbers    │ │
│  │ - lora       │  │ - vocabulary │ │
│  └──────┬──────┘  └──────────────┘ │
│         │                           │
│  ┌──────▼──────────────────────┐   │
│  │        Custom Tools          │   │
│  │ lattice_compare              │   │
│  │ fiedler_ratio                │   │
│  │ direction_weights            │   │
│  │ permutation_control          │   │
│  │ prime_vertex_map             │   │
│  │ spectral_analysis            │   │
│  │ visualize_rd                 │   │
│  │ visualize_amplification      │   │
│  │ explain_mechanism            │   │
│  └──────┬──────────────────────┘   │
│         │                           │
│  ┌──────▼──────────────────────┐   │
│  │    rhombic v0.3.0 (PyPI)    │   │
│  │  lattice · benchmark · spatial │ │
│  │  corpus · assignment · spectral│ │
│  │  polyhedron · visualize        │ │
│  └────────────────────────────┘   │
└─────────────────────────────────────┘
            ↕ Hermes Gateway
   Telegram · Discord · CLI · Web
```

---

## Competitive Edge

**Why this wins on creativity:** Nobody else at this hackathon will
submit an agent that runs peer-reviewed lattice topology experiments
in real-time. The geometry is intrinsically beautiful. The 3D
visualization is eye-catching. The cross-domain connection (lattice
theory → neural architecture → world models) is intellectually
ambitious without being hand-wavy — we have the numbers.

**Why this wins on usefulness:** `rhombic` is a real library on PyPI
with 208 tests. The agent makes it conversationally accessible. A
researcher can ask "what happens to the Fiedler ratio at scale 2000
with power-law weights?" and get a computed answer in seconds. The
RhombiLoRA concept is a genuine forward contribution to ML architecture.

**Why this wins on presentation:** We have the aesthetic vocabulary
(8-Law Weave), the website template inspiration (multigen), the
paper credentials (two papers, one published), and the tagline arc
("replace" → "mechanism" → "embrace"). Minta is our clarity test.
If the video makes her excited about lattice topology, it'll make
the judges excited too.

---

## Files and Repos

| Item | Location | Status |
|------|----------|--------|
| rhombic library | `github.com/promptcrafted/rhombic` | Complete (v0.3.0) |
| Paper 1 | `rhombic/paper/rhombic-paper1.tex` | Published |
| Paper 2 | `rhombic/paper/rhombic-paper2.tex` | Remediated, PDF built |
| Paper 3 plan | `rhombic/docs/PAPER3_RESEARCH_PLAN.md` | Plan complete |
| Hermes Agent tools | `rhombic/hermes-tools/` | **TO BUILD** |
| Presentation site | `rhombic/website/` | **TO BUILD** |
| Video demo | `rhombic/assets/demo.mp4` | **TO PRODUCE** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Hermes Agent install issues on Hermes server | Day 1 is dedicated to setup. Fallback: run on Workstation via WSL2 |
| Three.js RD visualization too complex | Fallback: static rotating GIF generated from matplotlib 3D plot |
| Video production takes too long | Prioritize screen recording with text overlays. VO is nice-to-have |
| Tools are slow at large scale | Pre-compute results for demo. Live tools use scale 125-500 for speed |
| Minta doesn't understand the presentation | She reviews Day 6. If unclear, rewrite Day 7 morning before video |

---

## Paper 2 Refinement (Parallel Track)

Paper 2 remediation is COMPLETE as of this session. The PDF is built
and pushed. Any further refinement (e.g., arXiv formatting, response
letter) happens AFTER the hackathon. The hackathon uses Paper 2 as
finished credential, not as active work.

## Paper 3 (After Hackathon)

The full RhombiLoRA research plan (`PAPER3_RESEARCH_PLAN.md`) begins
after the hackathon sprint. The hackathon demo is the proof-of-concept
presentation. Paper 3 is the peer-reviewed follow-through.

---

*Sprint plan locked March 6, 2026. Ten days. One agent. Twelve dimensions.*
