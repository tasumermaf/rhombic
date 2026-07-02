// ═══════════════════════════════════════════════════════════════════════
// TESSERACT → RHOMBIC DODECAHEDRON — Particle Swarm Simulation
// For https://particles.casberry.in/
//
// A 4D tesseract rotates in hyperspace. Its vertex-first projection
// into 3D IS the rhombic dodecahedron. 20,000 particles orbit the
// 14 vertices (8 cubic + 6 octahedral), forming RD faces as density
// surfaces. Color encodes the 8 prime threads of the Falco corpus.
//
// Promptcrafted LLC — TASUMER MAF
// ═══════════════════════════════════════════════════════════════════════

// ── Controls ──
const scale = addControl("scale", "Lattice Scale", 5, 80, 35);
const rot4d = addControl("rot4d", "4D Rotation Speed", 0, 2, 0.3);
const spread = addControl("spread", "Orbital Spread", 0.5, 8, 2.5);
const pulse = addControl("pulse", "Pulse Intensity", 0, 1, 0.4);
const bridgeVis = addControl("bridge", "Bridge Visibility", 0, 1, 0.6);
const chaos = addControl("chaos", "Chaos Level", 0, 1, 0.15);

// ── 4D Tesseract vertices (16 vertices of {±1}⁴) ──
// Pre-encoded as flat array: [x,y,z,w, x,y,z,w, ...]
// We compute these from index bits to avoid allocation
// Bit pattern: vertex v has coords (bit0?1:-1, bit1?1:-1, bit2?1:-1, bit3?1:-1)

// ── 4D→3D projection with rotation ──
// Rotate in the xw and yw planes, then stereographic project
const angle1 = time * rot4d;
const angle2 = time * rot4d * 0.618; // golden ratio offset
const cos1 = Math.cos(angle1);
const sin1 = Math.sin(angle1);
const cos2 = Math.cos(angle2);
const sin2 = Math.sin(angle2);

// Project all 16 tesseract vertices → 3D
// Under vertex-first projection, these collapse to 14 unique RD vertices:
// 8 cubic (trivalent) + 6 octahedral (tetravalent)
// We store projected coords in a flat scratch array (reuse each frame)
// Since we can't allocate, we compute per-particle from vertex index

// ── Particle assignment ──
// Split 20K particles across 14 RD vertices + 24 edges (bridges)
// Vertices: particles 0–13999 (1000 per vertex × 14)
// Bridges: particles 14000–19999 (250 per edge × 24)
const vertexCount = 14;
const perVertex = Math.floor(count * 0.7 / vertexCount);
const edgeCount = 24;
const perEdge = Math.floor(count * 0.3 / edgeCount);
const vertexParticles = perVertex * vertexCount;

// ── 8 Cubic vertices (trivalent, degree 3) — indices 0-7 ──
// ── 6 Octahedral vertices (tetravalent, degree 4) — indices 8-13 ──
// Encoded as sign patterns for the 4D→3D projection

// RD vertex positions (pre-projection, normalized)
// Cubic: (±1,±1,±1), Octahedral: (±2,0,0),(0,±2,0),(0,0,±2)

// Helper: get RD vertex position with 4D rotation applied
// The tesseract's vertex-first projection produces the RD when
// viewed along the (1,1,1,1) diagonal. We simulate this by
// rotating the 4D coordinates and projecting.

let vx = 0, vy = 0, vz = 0;
let hue = 0, sat = 0, lit = 0;

if (i < vertexParticles) {
  // ── Vertex-orbiting particle ──
  const vertIdx = Math.floor(i / perVertex);
  const localIdx = i % perVertex;

  // Get base RD vertex coordinates
  if (vertIdx < 8) {
    // Cubic vertex: (±1,±1,±1)
    const bx = (vertIdx & 1) ? 1 : -1;
    const by = (vertIdx & 2) ? 1 : -1;
    const bz = (vertIdx & 4) ? 1 : -1;
    // Apply 4D rotation (treat w=bx*by*bz as 4th coord)
    const bw = bx * by * bz;
    // Rotate in xw plane
    const rx = bx * cos1 - bw * sin1;
    const rw = bx * sin1 + bw * cos1;
    // Rotate in yw plane
    const ry = by * cos2 - rw * sin2;
    // Project: divide by (4 - w') for stereographic feel
    const proj = 1.0 / (4.0 - (by * sin2 + rw * cos2) * 0.5);
    vx = rx * proj * scale;
    vy = ry * proj * scale;
    vz = bz * proj * scale;
  } else {
    // Octahedral vertex: axis-aligned at distance 2
    const axIdx = vertIdx - 8;
    const axis = Math.floor(axIdx / 2); // 0=x, 1=y, 2=z
    const sign = (axIdx % 2) ? -1 : 1;
    const mag = 2;
    // Apply 4D rotation with w=0
    let ox = (axis === 0) ? sign * mag : 0;
    let oy = (axis === 1) ? sign * mag : 0;
    let oz = (axis === 2) ? sign * mag : 0;
    // Rotate in xw plane (w starts at 0)
    const rxo = ox * cos1;
    const rwo = ox * sin1;
    // Rotate in yw plane
    const ryo = oy * cos2 - rwo * sin2;
    const proj = 1.0 / (4.0 - (oy * sin2 + rwo * cos2) * 0.5);
    vx = rxo * proj * scale;
    vy = ryo * proj * scale;
    vz = oz * proj * scale;
  }

  // Orbital motion around vertex
  const angle = localIdx * 6.2831853 / perVertex + time * (1.5 + vertIdx * 0.1);
  const radius = spread * (0.5 + 0.5 * Math.sin(localIdx * 0.1 + time * 2));
  const orbZ = spread * 0.3 * Math.sin(localIdx * 0.37 + time * 1.3);

  // Pulse: breathe with the RD
  const breathe = 1.0 + pulse * 0.3 * Math.sin(time * 1.2 + vertIdx * 0.7);

  // Add chaos
  const cx2 = chaos * 3 * Math.sin(localIdx * 1.7 + time * 0.8);
  const cy2 = chaos * 3 * Math.cos(localIdx * 2.3 + time * 0.6);
  const cz2 = chaos * 3 * Math.sin(localIdx * 3.1 + time * 0.9);

  target.set(
    vx + (Math.cos(angle) * radius + cx2) * breathe,
    vy + (Math.sin(angle) * radius + cy2) * breathe,
    vz + (orbZ + cz2) * breathe
  );

  // ── Color: 8 prime threads (ternary chain — zero allocation) ──
  // Cubic vertices carry the 8 tracked primes as hue channels
  hue = vertIdx === 0 ? 0.00 :  // p67 — red
        vertIdx === 1 ? 0.08 :  // p23 — orange
        vertIdx === 2 ? 0.12 :  // p29 — gold
        vertIdx === 3 ? 0.33 :  // p17 — green
        vertIdx === 4 ? 0.50 :  // p19 — cyan
        vertIdx === 5 ? 0.58 :  // p31 — blue
        vertIdx === 6 ? 0.72 :  // p89 — indigo
        vertIdx === 7 ? 0.83 :  // p11 — violet
        vertIdx === 8 ? 0.95 :  // oct +x
        vertIdx === 9 ? 0.05 :  // oct -x
        vertIdx === 10 ? 0.16 : // oct +y
        vertIdx === 11 ? 0.42 : // oct -y
        vertIdx === 12 ? 0.55 : // oct +z
        0.75;                   // oct -z

  // Cubic vertices brighter (trivalent = Laws), octahedral dimmer (tetravalent = axes)
  sat = vertIdx < 8 ? 0.9 : 0.5;
  lit = vertIdx < 8 ? 0.6 : 0.4;
  // Pulse brightness
  lit = lit + 0.15 * Math.sin(time * 2 + vertIdx);

} else {
  // ── Bridge particle (edges between vertices) ──
  const edgeIdx = Math.floor((i - vertexParticles) / perEdge);
  const edgeLocal = (i - vertexParticles) % perEdge;

  // 24 RD edges: each connects one cubic to one octahedral vertex
  // Edge list encoded by index pairs (cubic 0-7, octahedral 8-13)
  // Distance sqrt(3) in coordinate space
  // Encode edge endpoints via edge index
  // RD edges: cubic(±1,±1,±1) to octahedral(±2,0,0) when they differ in 1 coord by ±1
  // We enumerate: for each octahedral vertex (6), connect to its 4 cubic neighbors
  const octIdx = Math.floor(edgeIdx / 4);
  const cubNeighbor = edgeIdx % 4;

  // Get octahedral position
  const oAxis = Math.floor(octIdx / 2);
  const oSign = (octIdx % 2) ? -1 : 1;
  let ox2 = (oAxis === 0) ? oSign * 2 : 0;
  let oy2 = (oAxis === 1) ? oSign * 2 : 0;
  let oz2 = (oAxis === 2) ? oSign * 2 : 0;

  // Get cubic neighbor (4 neighbors per octahedral vertex)
  // For axis X+: neighbors are (1,±1,±1)
  // Encode the 4 cubic neighbors via 2 bits
  let cx3 = 0, cy3 = 0, cz3 = 0;
  if (oAxis === 0) {
    cx3 = oSign;
    cy3 = (cubNeighbor & 1) ? 1 : -1;
    cz3 = (cubNeighbor & 2) ? 1 : -1;
  } else if (oAxis === 1) {
    cx3 = (cubNeighbor & 1) ? 1 : -1;
    cy3 = oSign;
    cz3 = (cubNeighbor & 2) ? 1 : -1;
  } else {
    cx3 = (cubNeighbor & 1) ? 1 : -1;
    cy3 = (cubNeighbor & 2) ? 1 : -1;
    cz3 = oSign;
  }

  // Interpolate along edge with time-varying position
  const t = (edgeLocal / perEdge) + 0.02 * Math.sin(time * 3 + edgeIdx);
  const interpX = cx3 + (ox2 - cx3) * t;
  const interpY = cy3 + (oy2 - cy3) * t;
  const interpZ = cz3 + (oz2 - cz3) * t;

  // Apply same 4D rotation as vertices
  const bw2 = interpX * 0.3; // approximate 4th-coord contribution
  const rx2 = interpX * cos1 - bw2 * sin1;
  const rw2 = interpX * sin1 + bw2 * cos1;
  const ry2 = interpY * cos2 - rw2 * sin2;
  const proj2 = 1.0 / (4.0 - rw2 * cos2 * 0.5);

  target.set(
    rx2 * proj2 * scale + chaos * Math.sin(edgeLocal * 0.5 + time),
    ry2 * proj2 * scale + chaos * Math.cos(edgeLocal * 0.7 + time),
    interpZ * proj2 * scale
  );

  // Bridge color: white-gold flowing along edges, opacity from bridgeVis
  const flow = Math.sin(t * 6.2831853 + time * 4) * 0.5 + 0.5;
  hue = 0.12; // gold
  sat = 0.3 + 0.4 * flow;
  lit = (0.3 + 0.5 * flow) * bridgeVis;
}

color.setHSL(hue, sat, Math.max(0.05, Math.min(0.95, lit)));

// ── HUD + Annotations (i === 0 only) ──
if (i === 0) {
  setInfo(
    "TESSERACT → RHOMBIC DODECAHEDRON",
    "4D hypercube rotating in hyperspace. Vertex-first projection = RD. " +
    "8 cubic vertices (Laws/primes) + 6 octahedral (axes). " +
    "24 edges = bridges. Color = 8 prime threads of the Falco corpus. " +
    "— Promptcrafted / TASUMER MAF"
  );
  annotate("origin", new THREE.Vector3(0, 0, 0), "◆ Origin");
  annotate("p67", new THREE.Vector3(scale * 0.3, scale * 0.3, scale * 0.3), "p67");
  annotate("p23", new THREE.Vector3(scale * 0.3, scale * 0.3, -scale * 0.3), "p23");
  annotate("p17", new THREE.Vector3(-scale * 0.3, -scale * 0.3, scale * 0.3), "p17");
  annotate("p89", new THREE.Vector3(-scale * 0.3, scale * 0.3, -scale * 0.3), "p89");
}
