# Hackathon Demo Video Script — 75 seconds

## Title Card (0-5s)
Visual: Animated RD wireframe (Three.js from website hero) spinning slowly.
Text overlay: **rhombic-agent** / *Keep Your Cube, Add Six Bridges.*

## Act 1 — The Default (5-20s)
Screen recording: Hermes Agent CLI.

**Prompt:** "Compare cubic and FCC lattices at scale 5."
Agent calls `lattice_compare`, returns table.

Text overlay: "Every computation defaults to cubic. 6 neighbors. 3 directions."
Cut to RD visualization (from `visualize_rd` tool output).
Text overlay: "The FCC lattice: 12 neighbors. 6 directions."

## Act 2 — The Amplification (20-45s)
Screen recording: Hermes Agent CLI.

**Prompt:** "What happens with structured weights? Run the direction weighting experiment."
Agent calls `direction_weights` with corpus distribution.
Result shows amplification beyond baseline.

**Prompt:** "Is this significant? Run the permutation control."
Agent calls `permutation_control`.
Result shows p-value.

Cut to amplification chart (from `visualize_amplification` tool output).
Text overlay: "Not noise. Structure. The advantage amplifies under stress."

## Act 3 — The Vision (45-60s)
Screen recording: Hermes Agent CLI.

**Prompt:** "Explain the mechanism at full depth."
Agent calls `explain_mechanism` with depth=full.
Shows Paper 1 → Paper 2 → Paper 3 arc.

Text overlay: "Paper 1 asked what happens when you replace the cube."
Text overlay: "Paper 2 found the mechanism."
Text overlay: "RhombiLoRA asks: what happens when you embrace it?"

## Closing (60-75s)
Visual: Weave pattern banner, fade to website URL.
Text overlays (stacked):
- `pip install rhombic`
- https://promptcrafted.github.io/rhombic/
- 256 tests. 3 papers. 0 competitors.

---

## Production Notes

### Screen Recording Setup
- Terminal: dark theme, large font (18px+)
- Window size: 1920x1080
- Record with OBS or Windows Game Bar (Win+G)
- Hermes Agent running on glm-reap via SSH

### Hermes Session Commands
```bash
ssh hermes
cd ~/hermes-agent
source .venv/bin/activate
python cli.py
```

### Tool Calls to Record
1. `lattice_compare` (scale 5) — ~2 seconds
2. `direction_weights` (corpus, scale 5) — ~5 seconds
3. `permutation_control` (200 trials, scale 5) — ~30 seconds
4. `visualize_amplification` — ~2 seconds
5. `explain_mechanism` (full) — instant

### Assembly (FFmpeg)
```bash
ffmpeg -i screen_recording.mp4 -i title_card.png \
  -filter_complex "[0:v]trim=start=0:end=75[main]; \
  [1:v]fade=in:0:15[title]; \
  [title][main]concat=n=2:v=1[out]" \
  -map "[out]" -c:v h264_nvenc -preset p7 -cq 18 \
  -t 75 output.mp4
```

### Alternative: Text-Only Version (2-hour production)
No voice. Kinetic typography over screen recording.
Title cards between sections. Background: ambient or silence.
This is the guaranteed-shippable fallback.
