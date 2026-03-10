# RhombiLoRA for Video LoRA Builders

> A bridge document for Minta Carlson and anyone who trains LoRA adapters
> and wants to understand what RhombiLoRA gives them.

---

## 1. Where You Are

You build LoRA adapters. You know the basic shape: every LoRA layer has
two matrices. **A** (the down-projection) compresses the input into a
small number of rank channels. **B** (the up-projection) expands those
channels back out into the model's full dimension. The adapter's output
is `B @ A` applied to the input.

Each rank channel works independently. Channel 1 does its thing. Channel
6 does its thing. They never talk to each other. The information flows
straight through: down into A, up through B, out.

This is standard LoRA. It works. It is also flying blind.

When training finishes, you have weights. You can generate with them and
see if they look right. You can check the loss curve. But if you want
to know *what the adapter actually learned* -- did it pick up motion
style? lighting? composition? -- you have to generate samples and look.
There is no readout. No dashboard. No way to inspect the adapter's
internal structure without running inference.

---

## 2. What RhombiLoRA Adds

One thing. A small learnable matrix -- the **bridge** -- that sits
between A and B.

Standard LoRA:
```
input -> A (down) -> B (up) -> output
```

RhombiLoRA:
```
input -> A (down) -> bridge (6x6) -> B (up) -> output
```

The bridge is a 6x6 matrix. It starts as the identity matrix, which
means it does nothing -- the signal passes straight through, and the
layer behaves exactly like standard LoRA. Literally identical. Same
loss, same gradients, same output.

As training proceeds, the bridge learns which rank channels should
talk to each other. Channel 2's signal might partly flow into channel
5. Channel 4 might suppress channel 1. The bridge encodes the
*coupling pattern* between channels -- which directions of variation
the adapter discovered, and how they relate.

The cost: **36 extra parameters per layer.** A 6x6 matrix. For a 28-layer
model with 4 attention projections each, the total overhead is
28 x 4 x 36 = 4,032 parameters. Your LoRA has millions. This is
rounding error.

Why 6? The six channel pairs come from the rhombic dodecahedron -- a
12-faced geometric solid where faces pair up into 6 opposing directions.
That is where the name comes from. But you do not need to care about
the geometry to use the tool. What matters: 6 channels give you a
rich enough coupling pattern to be diagnostic without being so large
that it is unreadable.

---

## 3. Why You Should Care

Three practical capabilities that standard LoRA does not have.

**Task fingerprinting.** We trained adapters on three different task
types (general instruction, code, math) and then asked: can we tell
which task produced which adapter just by reading its bridge matrices?
No inference. No generation. Just the 36 numbers per layer.

Answer: **84.5% accuracy**, from a pool of 28 Q-projection bridges
(1,008 total parameters). Chance would be 33.3%. The bridge encodes
a fingerprint of what happened during training, and that fingerprint
is readable.

For video LoRA, this means: your Q-projection bridges can tell you
whether the adapter learned motion style, lighting, or composition
-- without generating a single frame.

**Overfitting detection.** Bridge spectral properties (how spread out
the bridge's eigenvalues are) correlate with the generalization gap at
**r = 0.888**. That correlation is not subtle -- it is nearly linear.
When your adapter starts overfitting, the bridge shows it in 36
numbers. You do not need a validation set. You do not need to generate
and eyeball. The bridge is a built-in overfitting detector.

In practice: you watch the bridge deviation metric during training. When
it spikes (we saw an 807x jump at the overfitting transition in our
experiments), your adapter has crossed from learning to memorizing.

**Adapter composition.** When you blend two LoRA adapters (interpolating
weights at different alpha values), the bridge eigenspectrum is preserved
with cosine similarity above 0.999. This means the bridge gives you a
mathematically grounded way to predict how adapters will compose -- before
you spend the GPU time generating with the blend. Non-linear Fiedler
values at specific interpolation points flag where two adapters interfere
with each other.

---

## 4. What This Means for Flimmer

Drop-in replacement. In `flimmer/training/wan/`, where Wan's transformer
blocks get their LoRA layers injected, you replace `LoRALinear` with
`RhombiLoRALinear`. The constructor takes the same `in_features`,
`out_features`, and `rank` arguments. Add `n_channels=6`. Done.

```python
from rhombic.nn import RhombiLoRALinear

# Instead of:
# layer = LoRALinear(in_features=4096, out_features=4096, rank=24)

# Use:
layer = RhombiLoRALinear(
    in_features=4096,
    out_features=4096,
    rank=24,        # must be divisible by n_channels
    n_channels=6,   # the bridge size
)
```

The bridge starts as identity. If you freeze it (`layer.freeze_bridge()`),
you get exactly standard LoRA -- same weights, same math, same output.
Unfreezing it lets it learn during training at negligible cost.

**During training**, two new metrics become available at every step:

- `layer.bridge_deviation()` -- how far the bridge has moved from
  identity. Tracks learning progress. Spikes signal overfitting.
- `layer.bridge_fiedler()` -- algebraic connectivity of the bridge.
  Measures how much cross-channel coupling has developed. Higher =
  more structured internal communication.

**After training**, the bridge matrices are your adapter's dashboard:

- Read Q-projection bridges to fingerprint what the adapter learned.
  Q-projections carry the most task-specific information (40% more
  coupling than other projection types in our experiments).
- Compare bridge eigenspectra across checkpoints to track training
  dynamics without generating video.
- Compare bridges across adapters to predict composition behavior
  before blending.

For Wan video adapters specifically: the model has 28 transformer
blocks, each with self-attention and cross-attention projections (Q, K,
V, O). The Q-projection bridges across those 28 blocks are where the
fingerprint lives. That is 28 x 36 = 1,008 parameters that tell you
what your adapter learned -- readable in milliseconds, no GPU inference
required.

---

## 5. The Three Numbers

**4.6x** -- The cross-channel coupling advantage of 6-channel (RhombiLoRA)
over 3-channel (cubic) bridge topology. More coupling means a richer,
more readable diagnostic signal from fewer parameters.

**84.5%** -- Task fingerprint accuracy from bridge matrices alone. 1,008
parameters classify what happened during training at 2.5x above chance.
No inference needed.

**36** -- Parameters per layer. The total cost of the bridge. An
oscilloscope reading of your adapter's internal structure, for the price
of a rounding error.

The bridge is a 6x6 oscilloscope reading of your adapter's internal
coupling. Standard LoRA is flying blind. RhombiLoRA gives you instruments.
