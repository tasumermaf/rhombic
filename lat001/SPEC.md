# LAT-001 Harness SPEC — Continuous-Thought Next-Hop on Lattice Topologies

**Status: BUILD SPEC (Track B — harness + CPU smoke only). Binding documents:**
`docs/CARD_LAT001_DRAFT_2026-08-04.md` (design), `scripts/lat001_task_graph_sizing.py`
(landed graph constructions, 16605d1), `rhombic/lattice.py` (library),
arXiv:2505.12514 (continuous-CoT mechanism). This SPEC pins everything three
implementers need to build `tasks.py`, `model.py`, and `train.py`/`evaluate.py`
in parallel with zero collisions. Where this SPEC and the card disagree, the
card wins; flag the disagreement, do not silently resolve it.

## 0. Scope and hard rules (restated, non-negotiable)

- All code lives under `C:/falco/rhombic/lat001/` plus tests in
  `C:/falco/rhombic/tests/test_lat001.py`. Touch nothing else.
- **No GPU use in this workflow.** CPU smoke only — the box is running a
  campaign. `device` parameters exist for the later registered GPU phase but
  default to `"cpu"` and this workflow never passes anything else.
- **Seeded determinism everywhere.** No unseeded RNG anywhere in `lat001/`.
  All sub-seeds derive from a single config seed via `common.derive_seed`.
- No COMPLETE-marker or bank-root interaction. Results write only under
  `lat001/results/`.
- No commits, no pushes — the hub reviews and commits.
- Python 3.10; torch available in the falco conda env
  (`C:/miniconda3/envs/falco/python.exe`); numpy/networkx available.
  **scipy is NOT assumed** — statistics use `math.comb` / `math.erfc` only.

## 1. File ownership and dependency graph

| File | Owner | Imports from | Contents |
|---|---|---|---|
| `lat001/common.py` | **FROZEN by this SPEC** (already materialized; edits require a SPEC amendment) | stdlib only | vocab layout, size table, seed derivation, all shared dataclasses, pinned REAL/SMOKE model configs |
| `lat001/tasks.py` | Implementer A | `common`, `rhombic.lattice`, networkx, numpy | graph construction (sizing-script logic verbatim), orientation, relabeling, pair enumeration + ground-truth `d` + next-hop sets, train/eval splits, lazy token encoding, coordinate-exposed control, adjacency permutation for the null |
| `lat001/model.py` | Implementer B | `common`, torch | 2-layer decoder-only transformer, sinusoidal PE, continuous-thought loop, next-hop prediction head |
| `lat001/train.py` | Implementer C | `common`, `tasks`, `model`, torch | training loop, per-batch `c ~ Uniform{c_min..c_max}`, soft-target CE loss, checkpointing, determinism setup |
| `lat001/evaluate.py` | Implementer C | `common`, `tasks`, `model`, torch, numpy | (c,d) accuracy matrix, per-topology curves, permutation null, relabeling-invariance check, smoke entrypoint (`python -m lat001.evaluate --smoke`) |
| `tests/test_lat001.py` | Implementer A (task tests) + B (model tests) + C (train/eval tests), one file, sections marked by comment banners | all of the above | see §9 |

Dependency direction is strictly `common ← {tasks, model} ← {train, evaluate}`.
`tasks.py` and `model.py` never import each other — the only shared surface is
`common.py`. That is what makes parallel implementation safe.

## 2. Task definition

### 2.1 Endpoint

**Next-hop-on-a-shortest-path over reachable ordered pairs** (card §0). For a
directed graph `G` and reachable ordered pair `(s, t)` with directed distance
`d(s,t) = d ≥ 1`, the valid answer set is

```
next_hops(s, t) = { u ∈ out(s) : d(u, t) = d(s, t) − 1 }
```

(always non-empty for reachable pairs). Computed via one reverse-BFS from each
target `t` (`dist_to_t` on the reversed digraph), then filtering `succ(s)`.
Every example carries ground-truth `d` — the primary regressor.

- **Training target:** uniform soft distribution over `next_hops(s,t)`
  (cross-entropy against soft labels). This is relabeling-equivariant; no
  arbitrary canonical-hop choice exists anywhere in the pipeline.
- **Eval metric:** prediction correct iff argmax over the active node-token
  range is ANY member of `next_hops(s,t)` (set-membership accuracy).

### 2.2 Graphs — reuse the landed constructions

`tasks.py` replicates the three construction functions from
`scripts/lat001_task_graph_sizing.py` **with identical bodies** (`rewire`,
`dregular`, `rand_orient`), citing the script and commit 16605d1 in a comment.
`tests/test_lat001.py` enforces parity by loading the script via
`importlib.util.spec_from_file_location` and asserting identical edge sets /
arc sets for each topology at seed 42 (§9, T-1). Do not reinvent the graph
logic.

Five topology arms (strings pinned in `common.TOPOLOGIES`):

| key | construction |
|---|---|
| `"cubic6"` | `CubicLattice(n).to_networkx()` |
| `"fcc12"` | `FCCLattice(m).to_networkx()` |
| `"fcc_rewire"` | `rewire(FCC graph, seed)` — degree-preserving double-edge swap |
| `"rr_fccdeg"` | `dregular(N_fcc, round(mean degree of FCC graph), seed)` |
| `"rr_cubicdeg"` | `dregular(N_cubic, round(mean degree of cubic graph), seed)` |

Matched sizes (`common.SIZE_PAIRS`, index 0/1/2): cubic n=3/6/8 (27/216/512
nodes) vs FCC m=2/4/5 (32/256/500 nodes).

**Orientation:** uniform random orientation of every undirected edge,
arc-keep `p = 1.0`, same for all topologies — i.e. `rand_orient(g, 1.0, seed)`.
No thinning.

### 2.3 Node relabeling (MANDATORY)

After orientation, apply a seeded uniform-random permutation of `range(N)` to
all node IDs. Coordinates make `d` closed-form; a coordinate-reading model
performs no search and the measurement is void (card §2). Relabeling is
controlled by `TaskConfig.relabel_tag`: the same `(topology, size_index, seed)`
with a different `relabel_tag` yields the **same oriented graph** under a
different naming — this is the lever for the invariance check (§6.3).

Pipeline order (pinned): build undirected graph → orient → enumerate reachable
pairs, `d`, next-hop sets, and train/eval split **on original IDs** → apply the
relabel permutation to arcs, queries, and answer sets → encode tokens. Because
pairs/splits/edge-shuffle orders are computed pre-relabel, examples are exactly
paired across relabelings of the same graph (enables McNemar in §6.3).

### 2.4 Splits

- Universe: all reachable ordered pairs `(s, t)`, `s ≠ t`, `d ≥ 1`.
- Seeded shuffle; `eval_fraction = 0.2` held out.
- Eval set capped at `max_eval_pairs = 5000`, sampled proportionally
  stratified by `d` (seeded). Train pool capped at `max_train_pairs = 200000`
  (no practical cap at these sizes).
- One graph instance per cell; the split is over query pairs, not graphs
  (one model per cell, `c` an eval variable — card §2).

### 2.5 Coordinate-exposed control (`expose_coordinates=True`)

Positive control (card §3.2). Valid **only** for `"cubic6"` and `"fcc12"`
(the other arms have no lattice coordinates; `build_task` raises `ValueError`).
Relabeling still applies; the coordinates are attached explicitly to relabeled
IDs via a coordinate block in the prompt (§3.2), making `d` closed-form
recoverable without search. Predicted signature: correctness stops improving
with `c`. Integer coordinates: cubic — `round(pos/spacing)` (range 0..7);
FCC — `round(pos/(0.5a))` (range 0..10). Both fit `COORD_0..COORD_31`.

## 3. Tokenization (pinned exactly)

### 3.1 Vocabulary layout — `common.py` constants, VOCAB_SIZE = 550

| token id | name | meaning |
|---|---|---|
| 0 | `PAD` | padding (unused when batch lengths are equal, which they are within a cell) |
| 1 | `BOS` | `<s>` sequence start |
| 2 | `EDGE_SEP` | `<e>` closes each edge (and each coordinate record) |
| 3 | `QUERY` | `<Q>` opens the query |
| 4 | `ANSWER` | `<A>` answer-decode position |
| 5 | `THOUGHT` | `<T>` reserved placeholder for logging latent positions; its embedding is NEVER consumed — latent positions receive fed-back hidden states |
| 6 … 517 | `NODE_0 … NODE_511` | node IDs; `node_token(i) = 6 + i`; `N_MAX = 512` |
| 518 … 549 | `COORD_0 … COORD_31` | integer coordinate values; `coord_token(v) = 518 + v` |

One fixed vocabulary for all cells and both variants. Prediction is always
masked to the active graph's node range `[NODE_BASE, NODE_BASE + n_nodes)`.

### 3.2 Sequence layouts

Directed arc `u→v` is emitted as `[node(u)] [node(v)] EDGE_SEP` (tail, head).
Arc order is shuffled **per example** with `derive_seed(seed, f"shuf:{uid}")`
where `uid = f"{orig_src}-{orig_tgt}"` — per-example order is deterministic,
differs across examples, and is identical across relabelings of the same graph
(the underlying arc-index permutation is shared; only the token values change).

- **Standard prompt** (length `3m + 4`, constant within a cell):
  `BOS ([u][v] <e>) × m  <Q> [src] [tgt]`
- **Coordinate-control prompt** (length `5N + 3m + 4`): after `BOS`, a
  coordinate block ordered by relabeled node ID ascending —
  `([node] [cx] [cy] [cz] <e>) × N` — then the arc list and query as above.

The prompt ends at `[tgt]`. `Example` objects do NOT contain the `<A>` token
or thought positions — `model.forward` appends the latent steps and the
`<A>` position itself (§4.2). Ground-truth answer tokens never appear in the
input. Context budget: FCC-500 standard ≈ 3·2600+4 ≈ 7.8k tokens; coordinate
control ≈ 10.3k; `max_seq_len = 12288` covers both plus latent slots.

## 4. Model — `lat001/model.py`

### 4.1 Architecture (pinned)

2-layer decoder-only transformer from scratch (matches arXiv:2505.12514):
pre-LN blocks, causal self-attention, GELU MLP, final LayerNorm, **sinusoidal
positional encoding** (paper Definition 1, M = 10⁴; zero learned positional
parameters), token embedding tied with the output projection (the "next-hop
prediction head" = tied unembedding over the vocab; predictions masked to node
tokens). Dropout 0.0. Biases on.

| config | d_model | heads | layers | d_ff | vocab | max_seq_len | params |
|---|---|---|---|---|---|---|---|
| `REAL_MODEL` | 448 | 8 | 2 | 1792 | 550 | 12288 | ≈ 5.07 M |
| `SMOKE_MODEL` | 64 | 4 | 2 | 256 | 550 | 1024 | ≈ 0.14 M |

Param arithmetic (REAL): 2 × 12·448² = 4 824 960 (+biases/LN ≈ 12 k) +
550·448 = 246 400 tied embedding ≈ **5.07 M** — the card's ~5M pin.

### 4.2 Continuous-thought loop (the mechanism under test)

Coconut-style, per arXiv:2505.12514. Given prompt token embeddings
`E_0..E_{T−1}` (token emb + sinusoidal PE) and budget `c ≥ 0`:

```
for j in 1..c:                            # latent steps
    H   = TF(E_0 .. E_{T+j−2})            # causal, full forward
    h   = H[last position]                # final-layer hidden, post final LN
    E_{T+j−1} = h + PE[T+j−1]             # fed back as next input embedding
E_{T+c} = emb(ANSWER) + PE[T+c]           # append <A>
H = TF(E_0 .. E_{T+c})
logits = H[last] @ W_emb.T                # tied head, shape [B, VOCAB_SIZE]
```

- `c = 0` is legal: no latent steps, `<A>` appended directly after the prompt.
- The fed-back hidden state is **NOT detached** — gradients flow through the
  entire unrolled loop (this is what trains the mechanism).
- KV-caching across latent steps is permitted as an optimization iff gradients
  still flow; the naive recompute-per-step form is the reference semantics and
  is acceptable for the smoke.
- Within a cell all sequences have equal length, so batches need no padding;
  `pad_mask` exists for safety and defaults to `None`.

## 5. Training — `lat001/train.py`

- Per optimizer step: sample `batch_size` examples with replacement from the
  train pool (seeded), sample **one `c ~ Uniform{c_min..c_max}` per batch**
  (seeded), encode with `tasks.encode_batch`, forward with that `c`, loss =
  soft-target cross-entropy: `-(target · log_softmax(logits)).sum(-1).mean()`
  with `target` uniform over `next_hops` node tokens (zeros elsewhere).
- `c_min = 0`, `c_max = task.d_max + 2` (pinned defaults; `d_max` = max `d`
  over the task's examples, stored on `TaskData`).
- Optimizer: AdamW, betas (0.9, 0.95), weight_decay 0.01, grad-clip 1.0,
  linear warmup `warmup_steps = 50` then constant LR. The LR value itself is a
  `TrainConfig` field — the per-topology 3-point LR grid (card §2) is run by
  the later registered phase, not hard-coded here.
- Determinism setup at `train()` entry: `torch.manual_seed`, `random.seed`,
  `np.random.seed` (all from `derive_seed(cfg.seed, ...)` tags),
  `torch.use_deterministic_algorithms(True)`, single-threaded determinism not
  required (CPU matmul is deterministic).
- Checkpointing: `torch.save` at every `checkpoint_every` steps and at end, to
  `{checkpoint_dir}/ckpt_step{N:06d}.pt`, containing keys
  `{"model_state", "model_config", "train_config", "task_fingerprint",
  "step", "loss_history"}`. `task_fingerprint` = sha256 over
  `(n_nodes, sorted arcs, config fields)` — computed by `tasks.build_task`
  and stored on `TaskData`; `evaluate.py` refuses a checkpoint whose
  fingerprint mismatches the supplied task.
- CPU/GPU-agnostic (`device` field), CPU-defaulted; this workflow is CPU-only.

## 6. Evaluation — `lat001/evaluate.py`

### 6.1 (c, d) accuracy matrix

`evaluate(model, task, c_values, ...)` scores the eval split at each budget
`c ∈ c_values` (default sweep `range(0, task.d_max + 5)`), bucketing by
ground-truth `d`. Output `EvalResult`:

- `acc_by_c_d: np.ndarray [len(c_values), d_max_eval]` (NaN where count 0)
- `counts_by_c_d: np.ndarray` same shape, int
- `overall_by_c: np.ndarray [len(c_values)]`
- `c_values: tuple[int, ...]`, `d_values: tuple[int, ...]`
- JSON-serializable via `save_results(result, path)` (typed dict, no prose —
  values travel as typed state, XR-001 discipline). Per-topology curves are
  just this matrix per cell; a thin plotting helper may render them but the
  JSON is the artifact.

### 6.2 Permutation null (evaluation-only, card §3.5)

`permutation_null(model, task, c, n_perms, seed)`: for each of `n_perms`
seeded node permutations π (π ≠ identity), rebuild the eval examples with arcs
mapped through π while queries and answer sets keep their original labels
(`tasks.permute_arcs`). Score the trained model. Analytic guess baseline per
example: `p_i = |out_π(src) ∩ next_hops| / |out_π(src)|` (0 if out-degree 0).
Test: normal-approximation z of pooled model successes vs `Σp_i` with variance
`Σ p_i(1−p_i)`; report `z`, pooled accuracies, and `n`. Pass = `|z| < 3`.
No scipy — `math.erfc` for the p-value.

### 6.3 Relabeling-invariance check

`relabeling_invariance(model, base_config, relabel_tags=(1, 2), c, max_pairs)`:
rebuild the SAME oriented graph under two fresh relabelings (same seed,
different `relabel_tag`, both ≠ the training tag), evaluate the same
checkpoint on both. Examples are exactly paired (§2.3), so the test is
**McNemar exact**: `b` = correct-under-A-only, `c_dis` = correct-under-B-only,
two-sided exact binomial `p = P(X ≤ min(b,c_dis) or X ≥ max)` with
`X ~ Binomial(b + c_dis, 0.5)` via `math.comb`. Pass = `p > 0.01`
(statistically indistinguishable). Report both accuracies, `b`, `c_dis`, `p`.

### 6.4 Smoke entrypoint

`python -m lat001.evaluate --smoke` runs the full §8 sequence on CPU and
writes `lat001/results/smoke/SMOKE_REPORT.json` with every criterion's
measured value and PASS/FAIL. Exit code 0 iff all pass.

## 7. Interfaces (frozen — `lat001/common.py`, already materialized)

`common.py` is written and frozen alongside this SPEC. Implementers import
from it and never edit it. Full contents mirrored here for review:

```python
PAD, BOS, EDGE_SEP, QUERY, ANSWER, THOUGHT = 0, 1, 2, 3, 4, 5
NUM_SPECIAL = 6
N_MAX = 512; N_COORD = 32
NODE_BASE = 6; COORD_BASE = 518; VOCAB_SIZE = 550
def node_token(i) / node_id(tok) / coord_token(v)   # range-checked converters
def derive_seed(seed: int, tag: str) -> int         # sha256(f"{seed}:{tag}")[:4]

TOPOLOGIES = ("cubic6", "fcc12", "fcc_rewire", "rr_fccdeg", "rr_cubicdeg")
SIZE_PAIRS = ((3, 2), (6, 4), (8, 5))               # (cubic n, FCC m)

@dataclass(frozen=True) class TaskConfig:
    topology: str; size_index: int; seed: int
    relabel_tag: int = 0
    expose_coordinates: bool = False
    eval_fraction: float = 0.2
    max_eval_pairs: int = 5000; max_train_pairs: int = 200_000

@dataclass(frozen=True) class Example:
    src: int; tgt: int                    # relabeled IDs
    next_hops: tuple[int, ...]            # relabeled IDs, ALL valid hops
    d: int                                # directed distance (ground truth)
    uid: str                              # f"{orig_src}-{orig_tgt}" — pairs across relabelings
    shuffle_seed: int                     # arc-order seed for lazy encoding

@dataclass class TaskData:
    config: TaskConfig; n_nodes: int; m_arcs: int
    arcs: list[tuple[int, int]]           # relabeled directed arcs
    train: list[Example]; eval: list[Example]
    d_max: int; seq_len: int
    relabel: tuple[int, ...]              # relabel[orig_id] = new_id
    coords: tuple[tuple[int, int, int], ...] | None   # by relabeled ID; None unless expose_coordinates
    fingerprint: str                      # sha256, checkpoint interlock

@dataclass(frozen=True) class ModelConfig:
    d_model: int; n_heads: int; n_layers: int; d_ff: int
    vocab_size: int = VOCAB_SIZE; max_seq_len: int = 12288; dropout: float = 0.0

REAL_MODEL  = ModelConfig(448, 8, 2, 1792, max_seq_len=12288)
SMOKE_MODEL = ModelConfig(64, 4, 2, 256,  max_seq_len=1024)

@dataclass(frozen=True) class TrainConfig:
    steps: int; batch_size: int; lr: float
    c_min: int; c_max: int; seed: int
    device: str = "cpu"; checkpoint_dir: str = "lat001/results/ckpt"
    checkpoint_every: int = 100; log_every: int = 25
    warmup_steps: int = 50; weight_decay: float = 0.01; grad_clip: float = 1.0
```

### 7.1 `tasks.py` public signatures

```python
def build_task(config: TaskConfig) -> TaskData
def encode_example(task: TaskData, ex: Example) -> list[int]          # lazy; §3.2 layouts
def encode_batch(task: TaskData, examples: Sequence[Example]) -> torch.LongTensor   # [B, T] — OK to return list[list[int]] + let train.py tensorize; pick ONE and keep it (pinned: returns LongTensor)
def make_targets(task: TaskData, examples: Sequence[Example]) -> torch.FloatTensor  # [B, VOCAB_SIZE] uniform soft labels
def permute_arcs(task: TaskData, perm_seed: int) -> TaskData          # arcs through π; queries/labels untouched (null, §6.2)
# internal (tested but not part of the cross-file contract):
def build_undirected_graph(topology: str, size_index: int, seed: int) -> nx.Graph
def rewire(g, seed) / dregular(n, deg, seed) / rand_orient(g, p, seed)  # verbatim from scripts/lat001_task_graph_sizing.py (16605d1)
```

Seed tags (pinned): graph `"graph"`, orientation `"orient"`, split `"split"`,
relabel `f"relabel:{relabel_tag}"`, per-example shuffle `f"shuf:{uid}"`,
eval-stratification `"stratify"`. All via `derive_seed(config.seed, tag)`.

### 7.2 `model.py` public signatures

```python
class ContinuousThoughtTransformer(torch.nn.Module):
    def __init__(self, config: ModelConfig): ...
    def forward(self, tokens: torch.LongTensor,   # [B, T] prompt only (ends at [tgt])
                c: int,                            # latent budget, c >= 0
                pad_mask: torch.BoolTensor | None = None) -> torch.FloatTensor  # [B, VOCAB_SIZE] logits at <A>

def predict_next_hop(model: ContinuousThoughtTransformer, tokens: torch.LongTensor,
                     c: int, n_nodes: int) -> torch.LongTensor   # [B] node IDs; argmax masked to [NODE_BASE, NODE_BASE+n_nodes)
```

### 7.3 `train.py` / `evaluate.py` public signatures

```python
@dataclass class TrainResult:                       # defined in train.py
    losses: list[float]; final_loss: float; checkpoint_path: str; steps: int

def train(task: TaskData, model_config: ModelConfig, cfg: TrainConfig) -> TrainResult
def load_checkpoint(path: str, expect_fingerprint: str | None = None
                    ) -> tuple[ContinuousThoughtTransformer, dict]

@dataclass class EvalResult: ...                    # §6.1 fields; defined in evaluate.py
@dataclass class NullResult:                        # z, p, acc_model, acc_baseline, n, n_perms
@dataclass class InvarianceResult:                  # acc_a, acc_b, b, c_dis, p, n_pairs

def evaluate(model, task: TaskData, c_values: Sequence[int],
             batch_size: int = 64, device: str = "cpu") -> EvalResult
def permutation_null(model, task: TaskData, c: int,
                     n_perms: int = 20, seed: int = 0) -> NullResult
def relabeling_invariance(model, base_config: TaskConfig,
                          relabel_tags: tuple[int, int], c: int,
                          max_pairs: int = 500) -> InvarianceResult
def save_results(result, path: str) -> None         # typed JSON
```

## 8. Smoke acceptance criteria (CPU, synthetic minima)

Smoke cells: `("cubic6", size_index=0)` (N=27) and `("fcc12", size_index=0)`
(N=32), `SMOKE_MODEL`, `TrainConfig(steps=300, batch_size=32, lr=1e-3,
c_min=0, c_max=d_max+2, seed=42)`. 200–500 steps is the allowed band; 300 is
the default. Target wall-clock: ≤ ~15 min per cell on CPU.

| ID | Criterion | Pass condition |
|---|---|---|
| **SA-1** | Training loss decreases materially | mean loss over final 10% of steps ≤ 0.6 × mean over first 10% of steps, in BOTH cells |
| **SA-2** | (c,d) matrix end-to-end | `evaluate` returns a full matrix for `c ∈ 0..d_max+2`; every `d ≤ d_max` column has count > 0; `save_results` round-trips through JSON |
| **SA-3** | Relabeling invariance | McNemar exact `p > 0.01` on ≥ 200 paired eval items, two fresh relabel tags, per §6.3 |
| **SA-4** | Permutation null at base rate | `|z| < 3` vs the exact guess baseline, `n_perms ≥ 20`, ≥ 1000 pooled scored items, per §6.2 |
| **SA-5** | Seeded determinism | re-running `train` with identical configs reproduces `final_loss` bitwise AND `evaluate` reproduces `overall_by_c` bitwise |

The smoke does NOT require high absolute accuracy — it validates the harness,
not the science. All five criteria + measured values land in
`lat001/results/smoke/SMOKE_REPORT.json` (typed, one fact per key).

## 9. Test file — `tests/test_lat001.py`

Comment-bannered sections, one per implementer; all tests CPU, all seeded,
total runtime target < 90 s (tests use tiny graphs / ≤ 50 train steps):

- **T-1 (A)** Graph parity: load `scripts/lat001_task_graph_sizing.py` via
  `importlib.util.spec_from_file_location`; assert identical undirected edge
  sets and identical oriented arc sets vs `tasks.py` internals for every
  topology at size 0, seed 42.
- **T-2 (A)** Example correctness: for every emitted example on cubic-27,
  verify `d` and `next_hops` against fresh networkx BFS; assert
  `next_hops ⊆ out(src)` and non-empty; token layout matches §3.2 exactly
  (length `3m+4`; coordinate variant `5N+3m+4`).
- **T-3 (A)** Relabel pairing: same seed, tags 1 vs 2 → identical `uid`
  sequences, identical `d` per uid, arc lists related by a permutation.
- **T-4 (B)** Model: forward shape `[B, 550]` for c ∈ {0, 1, 5}; loop
  extends length by exactly `c+1` positions; gradients reach the token
  embedding through a `c=3` forward (grad non-None and non-zero).
- **T-5 (C)** Train smoke: 50 steps on cubic-27 tiny; loss finite, decreasing
  trend; checkpoint written and reloadable; fingerprint interlock rejects a
  mismatched task.
- **T-6 (C)** Eval plumbing: matrix shapes, NaN/count consistency,
  `permutation_null` and `relabeling_invariance` return finite statistics on
  a 50-step model; determinism (two identical eval calls bitwise-equal).

## 10. Provenance

- Design authority: `docs/CARD_LAT001_DRAFT_2026-08-04.md` (DRAFT — no GPU run
  authorized; this harness + smoke is Track B per card §5).
- Graph constructions: `scripts/lat001_task_graph_sizing.py` @ 16605d1.
- Lattices: `rhombic/lattice.py` (`CubicLattice`, `FCCLattice`).
- Mechanism: arXiv:2505.12514 — two-layer transformer, D continuous-thought
  steps solve reachability at diameter D; prompt format `<s> (edges) <Q> … <A>`
  adapted here from reachability-with-candidates to next-hop query; latent
  step = last hidden state fed back as next input embedding (verified against
  the paper's HTML, 2026-08-05).
- This SPEC drafted 2026-08-05 by the harness-spec subagent for hub review.
  Nothing here is registered; the card's Director grade governs what runs.
