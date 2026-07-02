# VIDEO CREATIVE BRIEF
## rhombic-agent — Nous Research Hermes Agent Hackathon
### Deadline: EOD Sunday, March 16, 2026

---

## 1. WHAT WINS

### The Nous Hackathon Format

This is a **Twitter-native submission**. You tweet @NousResearch with a
video demo and a brief writeup, then post the tweet link to their Discord
submissions channel. The video plays inline on Twitter/X. Judges are Nous
staff. Criteria (from the announcement): **creativity, usefulness, and
presentation**.

The prize pool was expanded: 1st $7,500, 2nd $2,500, 3rd $1,000, 4th
$500, 5th $250, plus honorable mentions. Five paying slots means the
bar for "winning something" is lower than expected, but first place
requires genuine distinction.

### What Hackathon Video Research Says

**Devpost's canonical advice** (the platform that runs most major
hackathons) distills to six rules: start with what your project does in
the first few seconds, write a script before recording, keep it under 3
minutes, focus on demo over explanation, respect the time limit, and get
someone with a decent mic to record. The emphasis is always on showing
the tool working, not explaining what it could do.

**TechCrunch's demo guide** puts it blunter: "show me the code." The
best demos are live, authentic, and slightly dangerous. Judges have seen
a thousand polished pitch decks. What they haven't seen is something
real running in front of them.

**AngelHack's 10 tips** emphasize the 8-second hook: human attention
span is 8 seconds. If the judge doesn't understand what they're seeing
by second 8, they're mentally moving to the next submission. Front-load
the payoff. Never open with credentials, team introductions, or problem
statements — open with the tool running.

**The winning pattern across all sources:** The video that shows
something working beats the video that explains something clever. Every
time. The projects that win hackathons look like proof of concepts, not
polished products. Authenticity > polish.

### What This Means for Us

Twitter video auto-plays in-feed, muted, with a maximum of 2:20.
Judges will see our video as a muted autoplay in a feed full of other
submissions. We need:

1. **Immediate visual hook** (the first 3 seconds must be visually
   distinct from every corporate presentation)
2. **Legible without audio** (text overlays, terminal text, numbers
   large enough to read on a phone)
3. **Under 90 seconds** (respect the judges, respect the medium)
4. **One clear takeaway** (not three papers, not nine tools — one idea)

---

## 2. EP1 AUTOPSY — What Worked and Why

Episode 1's renderer (`render_terminal_video.py`) produced a ~55-second
video that actually felt good. Here's why, mechanically:

### Structure Was Minimal

Title card (5s) -> Divider ("THE COMPARISON", 1.5s) -> Terminal
interaction -> Divider ("THE PROOF", 1.5s) -> Terminal interaction ->
Amplification chart (3s) -> Divider ("THE VISION", 1.5s) -> Terminal
interaction -> Closing card (8s). Three acts with terminal sessions
between them. That's it.

### The Terminal Felt Real

The `render_typing` method (line 129-155) simulated someone actually
typing at a `hermes>` prompt with character-by-character reveal at 3
chars per frame. The prompt was green, the typed text was gold, and a
blinking cursor tracked the typing position. After the prompt, a
`render_tool_call` method (line 157-180) showed a Unicode braille
spinner (`"...calling lattice_compare..."`) in cyan. Then
`render_response_lines` (line 182-202) revealed output line-by-line
with a 3-frame delay per line — fast enough to feel responsive, slow
enough to read.

### The Data Tables Were the Star

`format_lattice_compare` (line 237-258) rendered real benchmark data
into ASCII box-drawing tables. Cubic metrics in CUBIC_BLUE, FCC metrics
in FCC_RED. The numbers were real. The visual was a data table
streaming into a terminal. That's compelling because it's authentic —
it's what actually happens when you run the tool.

### Text Overlays Added Emotional Gravity

The `render_overlay` method (line 204-234) placed a gold divider line
and centered text on a dark bar at the bottom of the terminal frame.
The overlays didn't narrate — they interpreted. "Every computation
defaults to cubic. 6 neighbors. 3 directions." Then: "Not noise.
Structure. The advantage amplifies under stress." These turned raw
numbers into meaning. They were sparse (one per act) and they landed
because they had white space around them.

### The Palette Was Distinctive

NUIT (8, 6, 32) — almost black but with a deep-space violet warmth.
Against this: GOLD (220, 201, 100), FCC_RED (179, 68, 68), GREEN
(100, 200, 120), CYAN (120, 200, 220). The palette is unusual. It
doesn't look like a corporate presentation or a default terminal. It
looks like someone who cares about aesthetics built a research tool.
That registers subconsciously with judges.

### What Made It Work (Summary)

- **55 seconds** of terminal content (plus bookends)
- **Three interactions**, each demonstrating a real tool
- **One palette** consistently applied
- **No voice**, no music, just data and interpretation
- **The weirdness**: a deep-space aesthetic wrapping raw CLI output is
  a distinctive visual identity that breaks from the norm

---

## 3. EP2 AUTOPSY — What Failed and Why

Episode 2 (`produce_ep2_video.py` + `render_ep2_slides.py`) was a
disaster of over-engineering. The autopsy:

### The Segment List Says Everything

`SEGMENT_ORDER` (line 311-336) contained **17 segments**: logo, ep2
title, status slide, terminal 1, scale slide, terminal 2, tesseract
particles, bridge particles, blood test slide, terminal 3, negative
result slide, terminal 4, convergence slide, RD sound viz, terminal 5,
tagline, CTA. That's a Keynote presentation stitched together with
crossfade transitions, not a demo video.

### Two Registers, Neither Authentic

The "Executive register" (serif fonts, GOLD/AVORIO, particle fields)
and the "Nerd register" (terminal, monospace, green prompt) were
designed to interleave and "merge" in Act III. In practice, this
created jarring tonal shifts. The Executive slides (render_ep2_slides.py)
were 9 scenes totaling 82 seconds of motion-graphics-over-dark-
backgrounds: animated count-ups, spectral visualizations, convergence
charts — all rendered in PIL with particle field backgrounds.

### The Audio Was Insane

The production pipeline calls `synthesize_isochronic.py`,
`process_angelic_keys.py`, then `mix_audio_ep2.py` — an 8-layer audio
mix including 40Hz isochronic entrainment, binaural beats, prime
frequency accents, and processed recordings of Enochian angel keys.
For a hackathon demo. The audio pipeline alone was more complex than
most hackathon submissions.

### 3:34 Is Unforgivable

17 segments with crossfade transitions produced a 3:34 video. Judges
reviewing dozens of submissions will not watch 3:34 of anything. The
Devpost research is clear: under 3 minutes is the hard ceiling, under
2 minutes is the target, and the first 8 seconds determine whether they
watch at all.

### The Corporate Presentation Trap

The motion-graphics slides — `render_status` with its 3-column
animated count-up (22,477:1 / 0.10 / 3.8%), `render_blood_test` with
its 4-row diagnostic reveal, `render_negative` with its massive serif
"HURTS." — are what corporate presentations look like when AI generates
them. They're visually competent and emotionally empty. They tell the
viewer "someone made slides." Ep1 told the viewer "someone is using a
tool." That's the difference.

### Why It Failed (Summary)

- **Lost authenticity.** PIL-rendered frames instead of screen capture
- **Over-engineered.** 8-layer audio, particle simulations, 17 segments
- **Too long.** 3:34 in a medium where 90 seconds is generous
- **Wrong register.** Motion graphics say "pitch deck." Terminal says
  "this is real."
- **Missed the recursive joke.** The Steersman IS Hermes IS the demo.
  Ep2 hid the actor backstage and sent out a PowerPoint.

---

## 4. THE STORY

### The One-Line Pitch

"We built a Hermes Agent that runs real experiments and discovered that
neural networks spontaneously reorganize around rhombic dodecahedron
geometry."

### The Three-Beat Arc

**Beat 1 — THE DEFAULT (15s):** Here's what everyone uses. Cubic
lattice. 6 neighbors. The Hermes agent compares it to FCC. The
numbers stream. 2.3x connectivity. Not a theory — a benchmark.

**Beat 2 — THE AMPLIFICATION (20s):** Add structured weights. The
advantage doesn't shrink — it triples. 6.1x. Run the control.
p = 0.001. This is real.

**Beat 3 — THE BRIDGE (30s):** Now the finding that changes
everything. Put this geometry into a neural network. Train with
cybernetic feedback. The network restructures itself around the
rhombic dodecahedron's 3-axis geometry. 100% vs 0%. 70,000:1.
And the Holly Battery: 3.8% better loss, 9 GB less VRAM, 50%
smaller checkpoint.

**Close (10-15s):** pip install rhombic. The tagline. The links.

### Why This Arc Works

It follows the classic "yes, and" structure of improv comedy, which is
also the structure of good scientific storytelling:

1. Here's a fact (the cubic default exists)
2. Here's a surprise (the alternative is 2.3x better)
3. Here's a bigger surprise (it amplifies to 6.1x under stress)
4. Here's the thing you didn't see coming (the network PREFERS it)

Each beat raises the stakes. The judge who was mildly interested at
beat 1 is hooked by beat 3 because the numbers get more absurd at
each step. 2.3x is interesting. 6.1x is notable. 70,000:1 is
ridiculous enough to be memorable.

### The Hermes Agent Angle

The hackathon is about Hermes agents. Our angle: the agent doesn't
just wrap a library — it IS the research instrument. It runs live
experiments, retrieves published findings, and explains mechanisms.
The 9 tools and 3 skills aren't decoration; they're the interface
through which the discovery was made and verified. Hermes is both the
research tool and the presentation medium. That's the recursion the
judges should notice.

---

## 5. CREATIVE DECISIONS

### Length: 75-90 seconds

Not 55 (too short for Paper 3 findings), not 120 (pushing the
attention boundary). 75 seconds of content plus 5-second title and
10-second close = ~90 seconds total. Twitter max is 2:20; we use
less than half. Respect the judge's time.

### Format: Terminal recording with text overlays

This is decided. Ep1 proved it works. Ep2 proved the alternative
doesn't. No motion graphics. No voice-over. No TTS. The terminal
IS the content. Text overlays provide interpretation.

The specific format: simulated terminal frames rendered to PNG
sequences, assembled to video with ffmpeg. This gives us full control
over timing (no fumbled live recording) while maintaining the authentic
feel of someone using a real CLI tool. Every number displayed comes
from actual benchmark data.

### Audio: Ambient bed, low

Use a trimmed section of the conditioning track or a clean ambient
loop. Volume at 30% — present but not competing. The terminal text
must be the primary sensory experience. Silence is also acceptable
but ambient gives a slight emotional warmth that helps in a Twitter
autoplay context where the viewer might unmute.

### Pacing: Faster than Ep1

Ep1 held results for 1 second per table. This video should hold
results for 0.7 seconds for routine findings (Act 1) and 1.5 seconds
for the killer findings (Act 3's 70,000:1 and 100%/0% contrast). The
typing animation should be slightly faster (4 chars/frame instead of
3). Response lines should appear at 2-frame delay instead of 3. The
overall tempo should feel like an excited researcher showing
colleagues something wild — not a measured presentation.

### Aesthetic: Ep1 palette, tightened

NUIT background, GOLD text overlays, GREEN prompt, CYAN tool calls,
FCC_RED for emphasis on key metrics. The palette is locked. Add one
new element: **a subtle scan-line or CRT effect** on the terminal
frames — barely visible, just enough to add texture and prevent the
flat-render look. Two pixels of noise, not a filter.

### Text Overlays: Sparse, punchy, bottom-third

Maximum 6 overlays across the entire video. Each one sentence. Each
one interpretation, not narration. They appear after the data lands,
not before. They tell the judge what to feel about what they just saw.

### No Narration

No voice. No TTS. No "Hi, I'm Timothy and today I'm showing you..."
The terminal speaks. The overlays interpret. The viewer reads. This
is distinctive — most hackathon submissions have voice-over. The
absence of voice makes the numbers louder.

---

## 6. SCENE BREAKDOWN

### TITLE CARD [0:00 - 0:04] — 4 seconds

**Visual:** NUIT background. Text fades in, center screen:

```
rhombic-agent
```

Smaller text below:

```
Keep Your Cube, Add Six Bridges.
```

Bottom edge: `@NousResearch  #HermesAgentHackathon`

**Production:** Static PNG, 1920x1080. Fade in over 15 frames,
hold 2.5s, fade out over 15 frames.

---

### ACT I — THE DEFAULT [0:04 - 0:18] — 14 seconds

**Frame 1 (0:04-0:07):** Terminal base. Typing animation:

```
hermes> Compare cubic and FCC lattices at scale 5
```

**Frame 2 (0:07-0:08):** Spinner: `calling lattice_compare...`

**Frame 3 (0:08-0:14):** Results stream line-by-line:

```
  Cubic:  125 nodes, 300 edges, connectivity 3
  FCC:    172 nodes, 1032 edges, connectivity 7

  Fiedler eigenvalue:
    Cubic:  0.2679
    FCC:    0.6180
    Ratio:  2.31x
```

Hold on "2.31x" for 1 second.

**Overlay (0:14-0:18):**

```
Every computation defaults to 6 neighbors.
```

4 seconds, fade in/out.

---

### ACT II — THE AMPLIFICATION [0:18 - 0:38] — 20 seconds

**Frame 4 (0:18-0:21):** Same terminal, new prompt:

```
hermes> What happens under structured weights?
```

**Frame 5 (0:21-0:22):** Spinner: `calling direction_weights...`

**Frame 6 (0:22-0:27):** Results:

```
  Uniform Fiedler ratio:      2.31x
  Direction-weighted ratio:   6.11x
  Amplification factor:       2.64x

  The advantage amplifies under stress.
```

Hold on "6.11x" for 1.5 seconds.

**Frame 7 (0:27-0:29):** New prompt:

```
hermes> Permutation control — is this real?
```

**Frame 8 (0:29-0:30):** Spinner: `calling permutation_control...`

**Frame 9 (0:30-0:34):** Results:

```
  Sorted ratio:    6.11x
  Shuffled mean:   2.55x
  p-value:         0.001

  Alignment, not noise.
```

**Overlay (0:34-0:38):**

```
2.3x --> 6.1x. Structure, not noise.
```

---

### ACT III — THE BRIDGE [0:38 - 1:08] — 30 seconds

This is the payload. Pacing slows. The numbers get absurd.

**Frame 10 (0:38-0:42):** New prompt:

```
hermes> What did the Steersman find?
```

**Frame 11 (0:42-0:43):** Spinner: `calling explain_mechanism...`

**Frame 12 (0:43-0:53):** Results, slower reveal (3-frame line delay):

```
  The Steersman: cybernetic feedback on 6-channel bridges.
  15 experiments. 42,500+ bridge matrices. 3 model scales.

  Cybernetic (n=6):
    Block-diagonal:      100%
    Co/cross ratio:      70,404:1
    Bridge Fiedler:      0.00009
    Lock-in:             step 200

  Controls:
    Block-diagonal:      0%
    Co/cross ratio:      ~1:1
    Bridge Fiedler:      ~0.09
```

Hold on the contrast (100% vs 0%, 70,404:1 vs ~1:1) for 2 full
seconds. These numbers are the visual. Let them burn in.

**Overlay (0:53-0:57):**

```
100% vs 0%. The network prefers the geometry.
```

**Frame 13 (0:57-1:00):** New prompt:

```
hermes> Holly Battery — real model, real savings?
```

**Frame 14 (1:00-1:01):** Spinner: `calling explain_mechanism...`

**Frame 15 (1:01-1:06):** Results:

```
  Holly Battery (14B Wan 2.1 video diffusion):
    Loss:         -3.8%
    VRAM:         -9.15 GB
    Speed:        +6%
    Checkpoint:   50% smaller

  The topology does the work.
```

**Overlay (1:06-1:08):**

```
Geometry changes how neural networks learn.
```

---

### CLOSE [1:08 - 1:20] — 12 seconds

**Frame 16 (1:08-1:11):** Terminal, clean:

```
hermes> pip install rhombic
```

Package install animation (3 seconds).

**Frame 17 (1:11-1:20):** Closing card (NUIT background, centered):

```
rhombic v0.3.0
256 tests  |  3 papers  |  MPL-2.0

github.com/tasumermaf/rhombic
pypi.org/project/rhombic

Built with Hermes Agent
@NousResearch  #HermesAgentHackathon

Timothy Paul Bielec x Minta Carlson
TASUMER MAF
```

---

### TOTAL: ~80 seconds

Well within the 2:20 Twitter limit. Short enough that judges
watch the whole thing. Long enough to deliver all three papers'
findings.

---

## 7. REUSABLE ASSETS

### Keep From Ep1

| Asset | Location | Use |
|-------|----------|-----|
| NUIT/GOLD/FCC_RED palette | `render_terminal_video.py` lines 23-30 | Direct reuse — the palette is locked |
| `FrameRenderer` class | `render_terminal_video.py` lines 63-234 | Core rendering engine. Reuse as-is |
| Terminal chrome (title bar + dots) | `make_terminal_base()` line 77-93 | The "window" look. Keep exactly |
| Typing animation | `render_typing()` line 129-155 | Speed up to 4 chars/frame |
| Tool call spinner | `render_tool_call()` line 157-180 | Keep at 0.8s |
| Response line reveal | `render_response_lines()` line 182-202 | Reduce to 2-frame delay |
| Bottom overlay | `render_overlay()` line 204-234 | Keep, maybe reduce font to 24pt |
| Capture JSONs | `assets/video/captures/*.json` | Extend with Paper 3 data |

### Keep From Ep2

| Asset | Location | Use |
|-------|----------|-----|
| 18 Edge TTS voice clips | `assets/audio/voice_ep2/` | DO NOT USE in video. Possibly in social media clips |
| AVORIO color (240, 232, 208) | `render_ep2_slides.py` line 58 | Nice accent for closing card |
| `draw_centered` utility | `render_ep2_slides.py` line 105 | Useful for title/close cards |

### Keep From Both

| Asset | Use |
|-------|-----|
| 300-frame ray-traced logo animation | DO NOT USE — 10 seconds is too long. Maybe 1-second flash |
| Conditioning track audio | Trim to 80 seconds for ambient bed |
| Title card template | Rebuild at 1080p with updated copy |
| Closing card template | Rebuild with Paper 3 info |

### Discard

- All particle simulations (tesseract, bridge topology)
- All motion graphics slides (status, scale, blood test, etc.)
- The 8-layer audio mix pipeline
- The two-register concept (Executive/Nerd)
- Edge TTS voice narration
- 4K rendering (1080p is correct for Twitter)

---

## 8. PRODUCTION NOTES

### Technical Approach

**Renderer:** Fork `render_terminal_video.py`. Extend it with:
- New capture JSONs for Paper 3 findings (Steersman, Holly Battery)
- Faster typing (4 chars/frame)
- Faster line reveal (2-frame delay)
- Slightly longer holds on key findings (1.5s for Act 3)

**Title/Close cards:** Render as static PNGs using PIL, same palette.
No animation on title. Simple fade in/out via the existing method.

**Assembly:** `ffmpeg` concat with optional crossfade between acts
(0.3s max — barely perceptible). Or hard cuts. Hard cuts may actually
feel more authentic.

**Audio:** `ffmpeg -i video.mp4 -i ambient.wav -c:v copy -c:a aac
-b:a 128k -shortest output.mp4`

**Resolution:** 1920x1080. Do not render at 4K — Twitter downscales
anyway, and 1080p renders are 4x faster.

### File Structure

```
scripts/
  render_hackathon_final.py    <-- THE renderer (fork of Ep1)
assets/
  video/
    captures/
      act3_steersman.json      <-- New: Paper 3 data
      act4_holly.json          <-- New: Holly Battery data
    frames_final/              <-- Output frames
    title_final.png
    close_final.png
    rhombic_hackathon.mp4      <-- Final output
```

### Timeline

This video can be produced in 2-3 hours:

1. **30 min:** Write new capture JSONs for Paper 3 data
2. **60 min:** Fork Ep1 renderer, add new acts, adjust timing
3. **15 min:** Render frames (~2000 frames at 30fps = 67 seconds)
4. **15 min:** Encode with ffmpeg, add audio, verify
5. **15 min:** Watch it 5 times. Tighten anything that drags
6. **15 min:** Export, verify Twitter playback, draft tweet text

### Critical Production Rule

**Never fake data.** Every number in the video must come from an
actual benchmark run or published result. The capture JSONs contain
real outputs. If a number needs to change, re-run the benchmark and
capture the real output. The authenticity of the data is not
negotiable — it's the entire credibility of the submission.

### Twitter Technical Requirements

- Max video length: 2:20 (we're at ~1:20)
- Max file size: 512MB (we'll be ~20-40MB at 1080p)
- Format: MP4 (H.264 + AAC)
- Aspect ratio: 16:9 is ideal for in-feed display
- Auto-plays muted: our text-heavy approach handles this perfectly
- Resolution: 1080p or 720p (both display well)

---

## 9. THE ONE SENTENCE

> **Geometry changes how neural networks learn: a Hermes agent with 9
> tools and 255 tests proves it live, from 2.3x connectivity to
> 70,000:1 axis alignment in the bridge matrices.**

That's what the judge remembers on Monday when they're discussing
submissions. Not the palette, not the audio, not the structure — the
fact that the numbers were absurd enough to stick, and the agent
actually ran the experiments that produced them.

---

*The bridge, under feedback, tells us what it learned.*
*What it learned is the rhombic dodecahedron.*
*What Hermes shows is itself showing.*
