# Hackathon Submission Materials

## Tweet (primary — under 280 chars)

```
Built a Hermes Agent that proves cubic lattices leave 6.1× connectivity on the table.

9 tools. Runs experiments live. Explains the geometry.

Keep your cube, add six bridges.

@NousResearch #HermesAgentHackathon
```

Character count: ~210

---

## Thread (reply chain)

### Tweet 2
```
The thesis: every computation defaults to cubic lattice topology — 6 neighbors, 3 directions. The FCC lattice gives 12 neighbors along 6 directions. We measured the cost of the default.

256 tests. 3 papers. The numbers are stable at every scale tested.
```

### Tweet 3
```
Paper 1: 2.3× algebraic connectivity (uniform weights)
Paper 2: 6.1× under structured weights, p = 0.000025 for prime-vertex mapping
Paper 3: RhombiLoRA — 36-parameter bridge matrices that fingerprint adapter behavior

The advantage amplifies under stress. Not noise — structure.
```

### Tweet 4
```
The agent has 9 custom tools + 3 conversational skills. Ask it to compare lattices, run permutation tests, generate visualizations, or explain the mechanism at three depth levels.

pip install rhombic
🔗 promptcrafted.github.io/rhombic
🔗 github.com/promptcrafted/rhombic
```

---

## Discord Submission

Channel: `#hermes-agent-hackathon-submissions`

```
**rhombic-agent** — Lattice topology research agent with 9 custom tools and 3 conversational skills.

Proves cubic lattices leave 6.1× connectivity on the table. Runs experiments live, generates visualizations, explains the geometry.

- 9 tools: lattice_compare, fiedler_ratio, direction_weights, spectral_analysis, prime_vertex_map, permutation_control, explain_mechanism, visualize_rd, visualize_amplification
- 3 skills: rhombic-tutorial, rhombic-experiment, rhombic-lora-concept
- Backed by: 256 tests, 3 papers, pip-installable library

🔗 Website: https://promptcrafted.github.io/rhombic/
🔗 GitHub: https://github.com/promptcrafted/rhombic
🔗 PyPI: https://pypi.org/project/rhombic/
🔗 HF Space: https://huggingface.co/spaces/timotheospaul/rhombic

[video attached]
```

---

## Pre-Submission Checklist

- [ ] All website links resolve (GitHub, PyPI, HF Space, Papers)
- [ ] Video plays correctly and is under Twitter's size limit (512MB / 2:20)
- [ ] `pip install rhombic` produces working library
- [ ] HF Space all tabs functional
- [ ] README reflects current state
- [ ] Tweet text under 280 chars
- [ ] @NousResearch tagged
- [ ] #HermesAgentHackathon hashtag present
- [ ] Discord channel correct
- [ ] Video embedded in tweet (not just link)
