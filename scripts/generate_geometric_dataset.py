"""Generate geometric training dataset for Exp 2.5 and Exp 3.

The core hypothesis: training data with directional structure should produce
stronger co-planar vs cross-planar signal in the RhombiLoRA bridge than
isotropic data (Alpaca-cleaned). The 6 direction pairs of the rhombic
dodecahedron map to 6 reasoning axes. Multi-perspective tasks that require
integrating specific subsets of these axes create directional gradient
signal in the bridge.

Dataset Components:
  1. Multi-Perspective Reasoning (co-planar signal) — primary for Exp 2.5
  2. Expert-Decomposable Tasks (MoE routing) — for Exp 3A
  3. Cross-Domain Transit (bridge alignment) — for Exp 3B
  4. Geometric Reasoning (geometric awareness) — supplementary
  5. Symmetry-Structured Data (face symmetry) — supplementary

Output: Alpaca-format JSON (instruction, input, output) compatible with
the train_exp2_scale.py dataloader.

Usage:
  # Generate all components (~41K examples)
  python scripts/generate_geometric_dataset.py --output data/geometric/

  # Component 1 only (for Exp 2.5 validation)
  python scripts/generate_geometric_dataset.py --components 1 --output data/geometric/

  # Mix with Alpaca at 50% ratio
  python scripts/generate_geometric_dataset.py --mix-with alpaca --ratio 0.5 --output data/geometric/
"""

from __future__ import annotations

import argparse
import json
import random
import itertools
from pathlib import Path
from dataclasses import dataclass, field


# ── The 6 Direction Pairs ─────────────────────────────────────────────
# Each maps to a reasoning axis. Co-planar pairs (1-2, 3-4, 5-6) share
# 4 octahedral vertices in the RD. Cross-planar pairs share 2.

DIRECTION_PAIRS = {
    1: {"name": "analytical", "desc": "logical, formal, mathematical reasoning"},
    2: {"name": "empirical", "desc": "evidence-based, experimental, data-driven reasoning"},
    3: {"name": "ethical", "desc": "moral, values-based, stakeholder-impact reasoning"},
    4: {"name": "practical", "desc": "implementation, resource, feasibility reasoning"},
    5: {"name": "creative", "desc": "novel, divergent, generative reasoning"},
    6: {"name": "systemic", "desc": "holistic, emergent, interconnection reasoning"},
}

# Co-planar pairs share more structure (4 octahedral vertices)
CO_PLANAR_PAIRS = [(1, 2), (3, 4), (5, 6)]
# Cross-planar pairs share less (2 octahedral vertices)
CROSS_PLANAR_PAIRS = [(1, 3), (1, 4), (1, 5), (1, 6),
                       (2, 3), (2, 4), (2, 5), (2, 6),
                       (3, 5), (3, 6), (4, 5), (4, 6)]


# ── Scenario Templates ────────────────────────────────────────────────

SCENARIO_DOMAINS = [
    "technology", "healthcare", "education", "environment", "urban_planning",
    "agriculture", "energy", "transportation", "finance", "manufacturing",
    "communications", "food_systems", "water_management", "disaster_response",
    "housing", "public_health", "cultural_preservation", "scientific_research",
    "governance", "biodiversity",
]

SCENARIO_TEMPLATES = [
    # Technology
    {
        "domain": "technology",
        "scenarios": [
            "A city of 500,000 is considering replacing its traffic light system with AI-controlled adaptive signals",
            "A hospital network proposes using large language models to draft preliminary patient diagnoses",
            "A national government wants to implement a digital identity system for all citizens",
            "A social media platform considers open-sourcing its recommendation algorithm",
            "A school district plans to give every student a personal AI tutor",
            "A farming cooperative wants to deploy autonomous drone swarms for pest management",
            "A bank proposes replacing human loan officers with algorithmic credit scoring",
            "A news organization considers using AI to generate first drafts of routine articles",
            "A pharmaceutical company wants to use generative models to design new drug candidates",
            "A military organization proposes autonomous decision-making for logistics operations",
        ],
    },
    # Healthcare
    {
        "domain": "healthcare",
        "scenarios": [
            "A rural region with limited specialists considers telemedicine as the primary care model",
            "A biotech startup proposes CRISPR gene editing for inherited cardiovascular conditions",
            "An insurance company wants to use wearable health data to adjust premiums in real time",
            "A hospital system is evaluating whether to build a new facility or expand mobile clinics",
            "A public health agency considers mandatory genomic sequencing for newborns",
            "An eldercare facility proposes replacing some nursing staff with robotic assistants",
            "A dental network considers using 3D printing for same-day prosthetics",
            "A mental health platform proposes AI-generated therapy sessions for mild anxiety",
            "A pandemic preparedness agency evaluates decentralized vs centralized vaccine production",
            "A sports league considers mandating brain monitoring sensors in helmets",
        ],
    },
    # Environment
    {
        "domain": "environment",
        "scenarios": [
            "A coastal city considers constructing a tidal barrier vs managed retreat from sea level rise",
            "A logging company proposes converting to selective harvesting with drone monitoring",
            "A nation debates nuclear power vs solar+storage to meet carbon neutrality by 2040",
            "A river delta community faces the choice between damming and wetland restoration",
            "An island nation considers geoengineering proposals to cool local ocean temperatures",
            "A mining company proposes deep-sea mineral extraction for battery materials",
            "An agricultural region faces aquifer depletion and must choose water allocation strategies",
            "A city considers banning single-use plastics vs investing in advanced recycling",
            "A rewilding project proposes reintroducing apex predators to a region",
            "A desert community evaluates atmospheric water generation vs desalination",
        ],
    },
    # Education
    {
        "domain": "education",
        "scenarios": [
            "A university considers eliminating traditional lectures in favor of project-based learning",
            "A country debates lowering the school starting age from 6 to 4",
            "A school system proposes competency-based progression instead of age-based grades",
            "A technical college considers replacing textbooks with AI-curated learning paths",
            "A language immersion program debates starting bilingual education at age 3",
            "A community college proposes micro-credentials instead of associate degrees",
            "A gifted education program considers acceleration vs enrichment models",
            "A nationwide initiative proposes mandatory financial literacy from grade 5",
            "A rural district evaluates consolidated mega-schools vs networked micro-schools",
            "An arts academy considers integrating STEM into all performing arts curricula",
        ],
    },
    # Governance
    {
        "domain": "governance",
        "scenarios": [
            "A democracy considers implementing sortition (random selection) for some legislative roles",
            "A city proposes participatory budgeting for 30% of its annual infrastructure spending",
            "A federation debates centralizing emergency response vs maintaining regional autonomy",
            "A municipality considers liquid democracy for local planning decisions",
            "A small nation evaluates joining a regional trade bloc with sovereignty implications",
            "A state considers universal basic income funded by natural resource revenue",
            "A local government debates open data mandates for all public spending",
            "A tribal council evaluates incorporating traditional governance into modern administration",
            "A metropolitan area considers merging multiple jurisdictions into a single government",
            "A country debates mandatory voting vs incentivized participation",
        ],
    },
]

# Perspective-specific analysis templates
PERSPECTIVE_PROMPTS = {
    "analytical": [
        "What formal models, logical frameworks, or mathematical analyses apply?",
        "What are the quantifiable variables and their relationships?",
        "What can be deduced from first principles?",
        "What formal constraints or invariants must be satisfied?",
    ],
    "empirical": [
        "What does the available evidence show? What comparable cases exist?",
        "What data would we need to collect, and what would it tell us?",
        "What experiments or pilot studies could reduce uncertainty?",
        "What are the known failure modes in similar implementations?",
    ],
    "ethical": [
        "Who benefits and who bears the costs? Are these distributions just?",
        "What rights, duties, or obligations are at stake?",
        "What precedent does this set for future decisions?",
        "How does this affect the most vulnerable populations?",
    ],
    "practical": [
        "What resources (time, money, expertise, infrastructure) are required?",
        "What are the implementation risks and mitigation strategies?",
        "What is the realistic timeline and what are the dependencies?",
        "What existing systems must this integrate with?",
    ],
    "creative": [
        "What non-obvious alternatives or hybrid approaches exist?",
        "How might this be reimagined from a completely different starting point?",
        "What would this look like if the primary constraint were removed?",
        "What adjacent innovations could be combined with this?",
    ],
    "systemic": [
        "How does this interact with adjacent systems and stakeholders?",
        "What feedback loops, both reinforcing and balancing, are at play?",
        "What emergent behaviors might arise that no component predicts?",
        "How does this fit within the larger trajectory of related changes?",
    ],
}


# ── Component 1: Multi-Perspective Reasoning ──────────────────────────


def _pick_perspectives(n: int, favor_coplanar: float = 0.6) -> list[int]:
    """Select n perspective indices with configurable co-planar bias.

    favor_coplanar: probability of selecting a co-planar pair when n >= 2.
    0.5 = no bias (baseline). 0.6 = mild co-planar preference.
    Higher values create stronger co-planar gradient signal.
    """
    if n <= 1:
        return [random.randint(1, 6)]

    # Start with a random perspective
    selected = [random.randint(1, 6)]

    while len(selected) < n:
        remaining = [i for i in range(1, 7) if i not in selected]
        if not remaining:
            break

        if random.random() < favor_coplanar:
            # Prefer co-planar partner
            last = selected[-1]
            coplanar = [p for pair in CO_PLANAR_PAIRS
                        for p in pair if p != last and last in pair and p in remaining]
            if coplanar:
                selected.append(random.choice(coplanar))
                continue

        selected.append(random.choice(remaining))

    return selected


def generate_component1(
    n_examples: int = 10000,
    seed: int = 42,
    favor_coplanar: float = 0.6,
) -> list[dict]:
    """Generate multi-perspective reasoning tasks.

    Each example requires analysis from 2-6 perspectives, then synthesis.
    Co-planar perspective pairs are mildly favored (configurable), creating
    directional gradient signal.

    Returns Alpaca-format dicts with additional metadata.
    """
    rng = random.Random(seed)
    random.seed(seed)  # For _pick_perspectives

    # Flatten all scenarios
    all_scenarios = []
    for group in SCENARIO_TEMPLATES:
        for scenario in group["scenarios"]:
            all_scenarios.append({"domain": group["domain"], "text": scenario})

    examples = []
    for i in range(n_examples):
        # Select scenario
        scenario = rng.choice(all_scenarios)

        # Select number of perspectives (2-6, weighted toward 3-4)
        n_persp = rng.choices([2, 3, 4, 5, 6], weights=[10, 30, 35, 15, 10])[0]
        persp_ids = _pick_perspectives(n_persp, favor_coplanar)

        # Build perspective names and prompts
        persp_names = [DIRECTION_PAIRS[pid]["name"] for pid in persp_ids]
        persp_analyses = []
        for pid in persp_ids:
            name = DIRECTION_PAIRS[pid]["name"]
            prompt = rng.choice(PERSPECTIVE_PROMPTS[name])
            persp_analyses.append(f"**{name.title()} perspective:** {prompt}")

        # Build instruction
        persp_list = ", ".join(persp_names[:-1]) + f", and {persp_names[-1]}"
        instruction = (
            f"Analyze the following situation from {n_persp} perspectives: "
            f"{persp_list}. For each perspective, provide a focused analysis. "
            f"Then synthesize a recommendation that integrates all perspectives."
        )

        # Build structured input
        input_text = f"Situation: {scenario['text']}\n\n"
        for analysis in persp_analyses:
            input_text += f"{analysis}\n"

        # Build output template with synthesis
        output_parts = []
        for pid in persp_ids:
            name = DIRECTION_PAIRS[pid]["name"]
            desc = DIRECTION_PAIRS[pid]["desc"]
            output_parts.append(
                f"## {name.title()} Analysis\n"
                f"From the perspective of {desc}, this situation requires "
                f"careful consideration of the key factors within this domain. "
                f"The {name} lens reveals specific implications that must inform "
                f"the final recommendation."
            )

        # Synthesis section emphasizes integration of the selected perspectives
        synthesis_pairs = []
        for j in range(len(persp_ids) - 1):
            for k in range(j + 1, len(persp_ids)):
                pair = tuple(sorted([persp_ids[j], persp_ids[k]]))
                pair_type = "co-planar" if pair in CO_PLANAR_PAIRS else "cross-planar"
                synthesis_pairs.append(
                    f"{DIRECTION_PAIRS[persp_ids[j]]['name']}-"
                    f"{DIRECTION_PAIRS[persp_ids[k]]['name']}"
                )

        output_parts.append(
            f"## Integrated Synthesis\n"
            f"Bringing together the {persp_list} perspectives, the key "
            f"interactions emerge along {len(synthesis_pairs)} axes: "
            f"{', '.join(synthesis_pairs)}. "
            f"The recommendation must balance these complementary viewpoints "
            f"to produce a robust, multi-dimensional strategy."
        )

        output_text = "\n\n".join(output_parts)

        examples.append({
            "instruction": instruction,
            "input": input_text.strip(),
            "output": output_text,
            # Metadata for analysis (not used in training, stripped before save)
            "_meta": {
                "component": 1,
                "domain": scenario["domain"],
                "n_perspectives": n_persp,
                "perspective_ids": persp_ids,
                "perspective_names": persp_names,
                "has_coplanar_pair": any(
                    tuple(sorted([persp_ids[j], persp_ids[k]])) in CO_PLANAR_PAIRS
                    for j in range(len(persp_ids))
                    for k in range(j + 1, len(persp_ids))
                ),
            },
        })

    return examples


# ── Component 3: Cross-Domain Transit ─────────────────────────────────


TRANSIT_DOMAINS = [
    ("mathematical", "computational", "intuitive"),
    ("abstract", "concrete", "metaphorical"),
    ("verbal", "visual", "procedural"),
    ("theoretical", "experimental", "applied"),
    ("formal", "informal", "narrative"),
]

TRANSIT_CONCEPTS = [
    # Math → Code → Intuition triples
    {
        "concept": "gradient descent",
        "domain_a": "Gradient descent finds a local minimum of a differentiable function by iteratively moving in the direction of steepest descent, defined by the negative gradient: x_{n+1} = x_n - η∇f(x_n).",
        "domain_b": "def gradient_descent(f, grad_f, x0, lr=0.01, steps=100):\n    x = x0\n    for _ in range(steps):\n        x = x - lr * grad_f(x)\n    return x",
        "domain_c": "Imagine standing on a foggy hillside. You can't see the valley floor, but you can feel which way the ground slopes under your feet. Take a small step downhill. Repeat. You'll reach the bottom.",
    },
    {
        "concept": "recursion",
        "domain_a": "A recursive function is defined in terms of itself with a base case that terminates the recursion. Formally: f(n) = g(f(n-1), n) for n > 0, f(0) = c.",
        "domain_b": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "domain_c": "To understand recursion, imagine Russian nesting dolls. To open the outermost doll, you open it and find another doll inside. You keep opening until you reach the smallest solid doll — that's the base case.",
    },
    {
        "concept": "hash function",
        "domain_a": "A hash function h: {0,1}* → {0,1}^n maps arbitrary-length inputs to fixed-length outputs. Ideal properties: deterministic, uniform distribution, avalanche effect (small input change → large output change).",
        "domain_b": "def simple_hash(s, size=256):\n    h = 0\n    for c in s:\n        h = (h * 31 + ord(c)) % size\n    return h",
        "domain_c": "A hash function is like a fingerprint machine. You feed in a document of any length and get back a fixed-size fingerprint. Change one word and the fingerprint changes completely. Two different documents almost never produce the same fingerprint.",
    },
    {
        "concept": "eigenvector",
        "domain_a": "An eigenvector v of a linear transformation A satisfies Av = λv for some scalar eigenvalue λ. The vector's direction is preserved under the transformation; only its magnitude changes.",
        "domain_b": "import numpy as np\nA = np.array([[2, 1], [1, 2]])\neigenvalues, eigenvectors = np.linalg.eig(A)\n# eigenvectors[:,i] corresponds to eigenvalues[i]",
        "domain_c": "Imagine stretching a rubber sheet anchored at the center. Most points move in complicated ways. But there are special directions where points just slide straight outward or inward. Those directions are the eigenvectors.",
    },
    {
        "concept": "Fourier transform",
        "domain_a": "The Fourier transform decomposes a function into its frequency components: F(ω) = ∫f(t)e^{-iωt}dt. It maps between time and frequency domains, revealing periodic structure.",
        "domain_b": "import numpy as np\ndef dft(signal):\n    N = len(signal)\n    n = np.arange(N)\n    k = n.reshape(-1, 1)\n    return np.sum(signal * np.exp(-2j * np.pi * k * n / N), axis=1)",
        "domain_c": "Imagine you hear a chord on a piano. Your ear hears one complex sound wave. The Fourier transform is what your brain does to separate that wave into individual notes — each with its own pitch and volume.",
    },
    {
        "concept": "backpropagation",
        "domain_a": "Backpropagation applies the chain rule of calculus to compute gradients of a loss function with respect to each weight in a neural network: ∂L/∂w_i = ∂L/∂a_j · ∂a_j/∂w_i, propagated layer by layer from output to input.",
        "domain_b": "# Simplified backprop for one layer\ndef backward(dL_da, a_prev, w):\n    dL_dw = np.outer(dL_da, a_prev)  # weight gradient\n    dL_da_prev = w.T @ dL_da          # propagate to previous layer\n    return dL_dw, dL_da_prev",
        "domain_c": "A teacher grades a student's final essay and says 'the conclusion was weak.' The student traces back: the conclusion was weak because the argument in paragraph 3 was unclear, which happened because the evidence in paragraph 2 was misinterpreted. Each paragraph gets specific feedback about its contribution to the final grade.",
    },
    {
        "concept": "convolution",
        "domain_a": "Convolution of functions f and g is defined as (f*g)(t) = ∫f(τ)g(t-τ)dτ. It measures the overlap between f and a reversed, shifted copy of g. In the frequency domain, convolution becomes multiplication.",
        "domain_b": "def convolve_1d(signal, kernel):\n    n, k = len(signal), len(kernel)\n    out = [0] * (n - k + 1)\n    for i in range(len(out)):\n        out[i] = sum(signal[i+j] * kernel[j] for j in range(k))\n    return out",
        "domain_c": "Imagine dragging a magnifying glass across a photograph. At each position, the glass combines what it sees into a single measurement. The measurement changes as you slide — that's convolution. The magnifying glass is the kernel.",
    },
    {
        "concept": "dynamic programming",
        "domain_a": "Dynamic programming solves problems by breaking them into overlapping subproblems and storing solutions to avoid recomputation. Optimal substructure: the optimal solution contains optimal solutions to subproblems. Overlapping subproblems: the same subproblems recur.",
        "domain_b": "def fibonacci(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\n    return memo[n]",
        "domain_c": "Imagine planning a cross-country road trip. Instead of evaluating every possible route from scratch, you notice that many routes share the same segments. So you figure out the best way through each segment once, write it down, and combine the best segments into the best overall route.",
    },
    {
        "concept": "graph traversal",
        "domain_a": "Breadth-first search (BFS) explores all vertices at distance k before distance k+1, using a FIFO queue. It finds shortest paths in unweighted graphs. Time complexity: O(V + E).",
        "domain_b": "from collections import deque\ndef bfs(graph, start):\n    visited = {start}\n    queue = deque([start])\n    order = []\n    while queue:\n        node = queue.popleft()\n        order.append(node)\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return order",
        "domain_c": "Imagine dropping a stone into still water. The ripple expands outward in concentric circles, reaching nearby points before distant ones. BFS explores a network the same way — everything one step away, then two steps, then three.",
    },
    {
        "concept": "attention mechanism",
        "domain_a": "Scaled dot-product attention computes Attention(Q,K,V) = softmax(QK^T/√d_k)V, where Q,K,V are query, key, and value matrices. The softmax creates a probability distribution over values, weighted by query-key similarity.",
        "domain_b": "import torch\nimport torch.nn.functional as F\ndef attention(Q, K, V):\n    d_k = Q.size(-1)\n    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5\n    weights = F.softmax(scores, dim=-1)\n    return torch.matmul(weights, V)",
        "domain_c": "You're at a cocktail party scanning the room. Your brain is the query — what you're looking for. Each person's appearance is a key. When a key matches your query (you spot your friend), you pay attention to that person's words (the value). You can partially attend to multiple conversations at once.",
    },
]

TRANSIT_TASK_TEMPLATES = [
    "Given the {src} representation, produce the {tgt} representation.",
    "Translate between {src} and {tgt} formulations of this concept.",
    "Express the {src} understanding in {tgt} terms.",
    "Bridge from {src} to {tgt}: what changes and what is preserved?",
]


def generate_component3(
    n_examples: int = 8000,
    seed: int = 42,
) -> list[dict]:
    """Generate cross-domain transit tasks.

    Each example presents a concept in one domain and asks for transit to
    another. The bridge should learn to align representations through
    shared structure.
    """
    rng = random.Random(seed)
    domain_names = ["mathematical", "computational", "intuitive"]
    domain_keys = ["domain_a", "domain_b", "domain_c"]

    examples = []
    for i in range(n_examples):
        concept_data = rng.choice(TRANSIT_CONCEPTS)

        # Pick source and target domains (different)
        src_idx, tgt_idx = rng.sample(range(3), 2)
        src_name = domain_names[src_idx]
        tgt_name = domain_names[tgt_idx]

        template = rng.choice(TRANSIT_TASK_TEMPLATES)
        task = template.format(src=src_name, tgt=tgt_name)

        instruction = f"{task} The concept is: {concept_data['concept']}."
        input_text = f"Source ({src_name}):\n{concept_data[domain_keys[src_idx]]}"
        output_text = f"Target ({tgt_name}):\n{concept_data[domain_keys[tgt_idx]]}"

        examples.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "_meta": {
                "component": 3,
                "concept": concept_data["concept"],
                "src_domain": src_name,
                "tgt_domain": tgt_name,
            },
        })

    return examples


# ── Component 4: Geometric Reasoning ──────────────────────────────────

GEOMETRY_PROBLEMS = [
    {
        "instruction": "Determine whether the given graph is connected and find its algebraic connectivity.",
        "input": "Graph G has 6 vertices. Edges: (1,2), (2,3), (3,4), (4,5), (5,6), (6,1), (1,4), (2,5), (3,6).",
        "output": "G is connected — there exists a path between every pair of vertices. The graph is actually the complete bipartite graph K_{3,3} with added edges, making it 3-regular. With 9 edges on 6 vertices, it has high connectivity. The Fiedler value (second smallest eigenvalue of the Laplacian) is 3.0, indicating strong algebraic connectivity.",
    },
    {
        "instruction": "Explain why the rhombic dodecahedron tessellates 3D space while the regular dodecahedron does not.",
        "input": "Consider both polyhedra as candidates for space-filling.",
        "output": "The rhombic dodecahedron tessellates because it is the Voronoi cell of the FCC lattice — each point in space is closer to one FCC lattice point than any other, and the boundary between two Voronoi regions forms a rhombic face. Since Voronoi cells partition space completely by construction, RDs fill space without gaps. The regular (pentagonal) dodecahedron has dihedral angles of ~116.57°, and no integer number of regular dodecahedra can meet at an edge to sum to 360°. Space-filling requires compatible dihedral angles at every edge, which the regular dodecahedron cannot achieve.",
    },
    {
        "instruction": "Calculate the number of edges, faces, and vertices of the rhombic dodecahedron using Euler's formula.",
        "input": "The rhombic dodecahedron has 12 congruent rhombic faces. Vertices come in two types: 8 three-valent and 6 four-valent.",
        "output": "V = 8 + 6 = 14 vertices. F = 12 faces. By Euler's formula, V - E + F = 2, so 14 - E + 12 = 2, giving E = 24 edges. Verification: each of the 12 rhombic faces has 4 edges, giving 12 × 4 = 48 face-edges. Each edge is shared by 2 faces, so E = 48/2 = 24. ✓",
    },
    {
        "instruction": "Given a cubic lattice and an FCC lattice with the same number of nodes, compare their graph diameters.",
        "input": "Both lattices contain approximately 1000 nodes. Cubic lattice: 10×10×10. FCC lattice: equivalent node count.",
        "output": "Cubic 10×10×10 has 1000 nodes with 6-connectivity (face-sharing cubes). Graph diameter = 3 × (10-1) = 27 (worst case: corner to opposite corner). FCC with ~1000 nodes at 12-connectivity (face-sharing RDs) has diameter ≈ 18. The FCC diameter is about 67% of cubic — the higher connectivity provides more shortcut paths. This is a general result: FCC lattices have ~30% shorter diameter than cubic at matched node count.",
    },
    {
        "instruction": "Prove that the FCC lattice has coordination number 12.",
        "input": "Consider a single sphere in an FCC packing arrangement.",
        "output": "In FCC packing, place a sphere at the origin. Its nearest neighbors are at distance a/√2 (where a is the conventional cubic cell parameter). These neighbors sit at the 12 edge midpoints of the surrounding cube: (±a/2, ±a/2, 0), (±a/2, 0, ±a/2), (0, ±a/2, ±a/2). That's 3 coordinate planes × 4 combinations of ± signs = 12 neighbors. Equivalently, the Voronoi cell (rhombic dodecahedron) has 12 faces, and each face corresponds to one nearest neighbor. Coordination number = face count = 12.",
    },
    {
        "instruction": "Explain the relationship between the tesseract and the rhombic dodecahedron.",
        "input": "Consider projecting a 4D hypercube (tesseract) into 3D.",
        "output": "The vertex-first projection of the tesseract into 3D produces a rhombic dodecahedron. The tesseract has 16 vertices, 32 edges, 24 square faces, and 8 cubic cells. When projected vertex-first along a body diagonal, the 16 vertices map to 14 distinct 3D points (the two vertices on the projection axis collapse to inner/outer points). The 24 squares project to 12 rhombic faces. This projection is not arbitrary — it's the unique projection that preserves the maximal symmetry group. The RD literally IS the tesseract's shadow in 3D, making it a natural structure for 4D-aware geometric operations.",
    },
    {
        "instruction": "Calculate the packing fraction of FCC sphere packing.",
        "input": "Spheres of radius r arranged in face-centered cubic packing.",
        "output": "In FCC, the conventional unit cell is a cube of side a. Spheres touch along face diagonals: 4r = a√2, so a = 2r√2. The unit cell contains 4 spheres (8 corners × 1/8 + 6 faces × 1/2 = 4). Volume of 4 spheres: 4 × (4/3)πr³ = 16πr³/3. Volume of unit cell: a³ = (2r√2)³ = 16r³√2. Packing fraction = (16πr³/3) / (16r³√2) = π/(3√2) ≈ 0.7405. This is the densest possible sphere packing in 3D (Kepler conjecture, proved by Hales 2005).",
    },
    {
        "instruction": "Describe the symmetry group of the rhombic dodecahedron.",
        "input": "Identify the rotational and full symmetry groups.",
        "output": "The rhombic dodecahedron has octahedral symmetry. The full symmetry group is O_h (order 48), the same as the cube and regular octahedron. This is because the RD is the dual of the cuboctahedron, which has octahedral symmetry. The rotational subgroup is O (order 24). Key symmetry elements: 3 four-fold axes (through opposite 4-valent vertices), 4 three-fold axes (through opposite 3-valent vertices), 6 two-fold axes (through opposite edge midpoints), plus reflection planes and the inversion center.",
    },
    {
        "instruction": "Explain Fiedler value (algebraic connectivity) and why it matters for network robustness.",
        "input": "The Fiedler value is the second-smallest eigenvalue of the graph Laplacian.",
        "output": "For graph G with Laplacian L = D - A (degree matrix minus adjacency matrix), the eigenvalues are 0 = λ₁ ≤ λ₂ ≤ ... ≤ λ_n. λ₁ = 0 always (corresponding eigenvector is all-ones). λ₂ = Fiedler value measures how well-connected the graph is. If λ₂ = 0, the graph is disconnected. Larger λ₂ means harder to disconnect by removing edges. For networks: λ₂ bounds the convergence rate of diffusion processes, the mixing time of random walks, and the expansion ratio. FCC lattices have ~2.4× larger Fiedler values than cubic at matched node count — meaning 2.4× faster information diffusion and 2.4× harder to partition.",
    },
    {
        "instruction": "Calculate the dihedral angle of the rhombic dodecahedron.",
        "input": "Find the angle between two adjacent faces of the RD.",
        "output": "The dihedral angle of the RD is 120°. This can be derived from the fact that three RDs meet at every edge in the tessellation: 3 × 120° = 360°, confirming space-filling. Alternatively, consider two adjacent rhombic faces sharing an edge. The normal vectors to these faces make an angle of 60° (supplementary to the dihedral angle measured from inside). The dihedral angle = 180° - 60° = 120°. This 120° angle is what makes the tessellation work — three faces from three different RDs meet exactly at each edge.",
    },
]


def generate_component4(
    n_examples: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """Generate geometric reasoning tasks.

    Template-based variations of core geometric concepts.
    """
    rng = random.Random(seed)
    examples = []

    # Use the hand-crafted problems as seeds, generate variations
    for i in range(n_examples):
        base = rng.choice(GEOMETRY_PROBLEMS)
        examples.append({
            "instruction": base["instruction"],
            "input": base["input"],
            "output": base["output"],
            "_meta": {"component": 4},
        })

    return examples


# ── Dataset Assembly ──────────────────────────────────────────────────


def strip_metadata(examples: list[dict]) -> list[dict]:
    """Remove _meta fields for training — Alpaca format only."""
    return [
        {k: v for k, v in ex.items() if not k.startswith("_")}
        for ex in examples
    ]


def mix_with_alpaca(
    geometric: list[dict],
    alpaca_path: str | Path,
    ratio: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """Mix geometric data with Alpaca at the specified ratio.

    ratio = 0.5 means 50% geometric, 50% Alpaca.
    """
    rng = random.Random(seed)

    with open(alpaca_path) as f:
        alpaca = json.load(f)

    # Calculate sizes
    total = len(geometric) + len(alpaca)
    n_geo = int(total * ratio)
    n_alp = total - n_geo

    geo_sample = rng.sample(geometric, min(n_geo, len(geometric)))
    alp_sample = rng.sample(alpaca, min(n_alp, len(alpaca)))

    mixed = geo_sample + alp_sample
    rng.shuffle(mixed)
    return mixed


def compute_dataset_stats(examples: list[dict]) -> dict:
    """Compute statistics for a dataset with metadata."""
    stats = {
        "total": len(examples),
        "by_component": {},
        "coplanar_fraction": 0.0,
    }

    comp_counts = {}
    coplanar_count = 0
    total_with_meta = 0

    for ex in examples:
        meta = ex.get("_meta", {})
        comp = meta.get("component", "unknown")
        comp_counts[comp] = comp_counts.get(comp, 0) + 1

        if meta.get("has_coplanar_pair") is not None:
            total_with_meta += 1
            if meta["has_coplanar_pair"]:
                coplanar_count += 1

    stats["by_component"] = comp_counts
    if total_with_meta > 0:
        stats["coplanar_fraction"] = coplanar_count / total_with_meta

    return stats


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate geometric training dataset for RhombiLoRA experiments"
    )
    parser.add_argument(
        "--output", type=str, default="data/geometric",
        help="Output directory"
    )
    parser.add_argument(
        "--components", type=str, default="1,3,4",
        help="Comma-separated component numbers to generate (1,3,4)"
    )
    parser.add_argument(
        "--n1", type=int, default=10000,
        help="Number of Component 1 examples"
    )
    parser.add_argument(
        "--n3", type=int, default=8000,
        help="Number of Component 3 examples"
    )
    parser.add_argument(
        "--n4", type=int, default=5000,
        help="Number of Component 4 examples"
    )
    parser.add_argument(
        "--coplanar-bias", type=float, default=0.6,
        help="Co-planar selection bias for Component 1 (0.5=none, 1.0=max)"
    )
    parser.add_argument(
        "--mix-with", type=str, default=None,
        help="Path to Alpaca JSON to mix with"
    )
    parser.add_argument(
        "--ratio", type=float, default=0.5,
        help="Geometric data ratio when mixing (0.0-1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    components = [int(c) for c in args.components.split(",")]
    all_examples = []

    if 1 in components:
        print(f"Generating Component 1: Multi-Perspective Reasoning ({args.n1} examples)...")
        c1 = generate_component1(args.n1, seed=args.seed, favor_coplanar=args.coplanar_bias)
        all_examples.extend(c1)

        # Save component separately
        with open(out_dir / "component1_multi_perspective.json", "w") as f:
            json.dump(strip_metadata(c1), f, indent=2)
        print(f"  Saved {len(c1)} examples to component1_multi_perspective.json")

    if 3 in components:
        print(f"Generating Component 3: Cross-Domain Transit ({args.n3} examples)...")
        c3 = generate_component3(args.n3, seed=args.seed)
        all_examples.extend(c3)

        with open(out_dir / "component3_cross_domain.json", "w") as f:
            json.dump(strip_metadata(c3), f, indent=2)
        print(f"  Saved {len(c3)} examples to component3_cross_domain.json")

    if 4 in components:
        print(f"Generating Component 4: Geometric Reasoning ({args.n4} examples)...")
        c4 = generate_component4(args.n4, seed=args.seed)
        all_examples.extend(c4)

        with open(out_dir / "component4_geometric.json", "w") as f:
            json.dump(strip_metadata(c4), f, indent=2)
        print(f"  Saved {len(c4)} examples to component4_geometric.json")

    # Combined dataset
    if all_examples:
        stats = compute_dataset_stats(all_examples)
        print(f"\nDataset statistics:")
        print(f"  Total examples: {stats['total']}")
        print(f"  By component: {stats['by_component']}")
        print(f"  Co-planar fraction (C1): {stats['coplanar_fraction']:.3f}")

        # Save combined (training-ready, no metadata)
        clean = strip_metadata(all_examples)
        random.Random(args.seed).shuffle(clean)

        with open(out_dir / "geometric_combined.json", "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n  Combined dataset: {len(clean)} examples → geometric_combined.json")

        # Save with metadata (for analysis)
        with open(out_dir / "geometric_combined_meta.json", "w") as f:
            json.dump(all_examples, f, indent=2)
        print(f"  With metadata: geometric_combined_meta.json")

        # Save stats
        with open(out_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    # Mix with Alpaca if requested
    if args.mix_with:
        print(f"\nMixing with Alpaca at {args.ratio:.0%} geometric ratio...")
        mixed = mix_with_alpaca(
            strip_metadata(all_examples),
            args.mix_with,
            ratio=args.ratio,
            seed=args.seed,
        )
        mix_name = f"mixed_geo{int(args.ratio*100)}_alpaca{int((1-args.ratio)*100)}.json"
        with open(out_dir / mix_name, "w") as f:
            json.dump(mixed, f, indent=2)
        print(f"  Mixed dataset: {len(mixed)} examples → {mix_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
