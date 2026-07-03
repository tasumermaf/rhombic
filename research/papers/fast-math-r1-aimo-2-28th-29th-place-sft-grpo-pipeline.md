# Fast-Math-R1 (AIMO-2 28th/29th place, SFT+GRPO pipeline)

> [link](https://github.com/analokmaus/kaggle-aimo2-fast-math-r1)

**Venue:** GitHub (competition solution)  
**Tags:** nemotron-engineering, grpo, reward-design, competition-solution

## Program assessments

### From `C:\falco\rhombic\competition\RESEARCH_GRPO.md`

**Verdict:** methodology-reference · **Threat:** none · **Confidence:** high

GRPO (Group Relative Policy Optimization) is the
standard approach — it eliminated the critic model from PPO, halving memory requirements,
and was used to train DeepSeek-R1, Nemotron 3 Nano itself, and Fast-Math-R1 (which
placed 28th/29th in AIMO-2 using SFT+GRPO).
---
### Fast-Math-R1 Approach (3-Component Reward)

Fast-Math-R1 placed 28/29 in AIMO-2 using SFT + GRPO with:

1. **Format Reward (binary +1/0):** Does output match `\\boxed{}...` + `</think>` pattern?
2. **Cosine Similarity Reward (0.1-1.0 scaled):** Embedding similarity of reasoning
   trace vs reference. Correct answers get 0.1-1.0 (penalizes verbose correct),
   incorrect get -1.0 to -0.1 (penalizes short incorrect). Max trace: 30K tokens.
3. **Length Penalty:** Proportional to output length, discourages verbosity.
---
The most relevant competition result: **Fast-Math-R1-14B** placed 28th public / 29th
private in AIMO-2 using a two-stage pipeline.

### Stage 1: Extended SFT
- Base: Qwen2.5-14B
- Data: Light-R1-SFT (multi-stage)
- Train until accuracy plateaus

### Stage 2: GRPO
- 8x H200 GPUs, ~10 hours
- num_generations = 8
- Reward: format + cosine similarity + length penalty
- Uses open-r1's faster GRPOTrainer implementation
- Data: Light-R1-SFT second-stage data

**Key insight:** "SFT first pushes the model's accuracy to its limits, after which
GRPO dramatically improves **token efficiency** while preserving peak performance."
GRPO doesn't necessarily improve accuracy — it makes the model faster/shorter at
producing correct answers.
