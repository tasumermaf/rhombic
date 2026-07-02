// ═══════════════════════════════════════════════════════════════════════
// RHOMBI-BRIDGE — 6-Channel FCC Bridge Topology
// For https://particles.casberry.in/
//
// Visualizes the TeLoRA bridge: two LoRA projection clusters
// (A and B) connected by 6 directional channels following FCC geometry.
// Co-planar channels (3) carry massive flow; cross-planar channels (3)
// are nearly empty — the 22,477:1 axis alignment ratio made visible.
//
// Promptcrafted LLC — TASUMER MAF
// ═══════════════════════════════════════════════════════════════════════

// ── Controls ──
const separation = addControl("sep", "A↔B Separation", 15, 80, 40);
const flow = addControl("flow", "Flow Speed", 0.5, 5, 2.0);
const axisRatio = addControl("ratio", "Co/Cross Ratio", 1, 100, 80);
const fiedler = addControl("fiedler", "Fiedler λ₂", 0, 0.15, 0.10);
const turbulence = addControl("turb", "Turbulence", 0, 1, 0.2);
const breathe = addControl("breathe", "Breathe", 0, 1, 0.5);

// ── Architecture ──
// Two clusters (A projection, B projection) at ±separation/2 on x-axis
// 6 bridges between them following FCC direction pairs:
//   Co-planar (high flow):  [1,1,0], [1,-1,0], [1,0,0]  (xy-plane)
//   Cross-planar (low flow): [1,0,1], [1,0,-1], [0,1,1]  (out of plane)

const halfSep = separation * 0.5;
const breathScale = 1.0 + breathe * 0.15 * Math.sin(time * 1.5);
const t = time * flow;

// Particle budget:
// Cluster A: 30% (particles orbiting A projection)
// Cluster B: 30% (particles orbiting B projection)
// Co-planar bridges: 30% (3 bridges, heavy flow)
// Cross-planar bridges: 10% (3 bridges, sparse flow)
const clusterSize = Math.floor(count * 0.3);
const coplanarTotal = Math.floor(count * 0.3);
const crossplanarTotal = count - 2 * clusterSize - coplanarTotal;
const coPerBridge = Math.floor(coplanarTotal / 3);
const crossPerBridge = Math.floor(crossplanarTotal / 3);

let px = 0, py = 0, pz = 0;
let hue = 0, sat = 0, lit = 0;

if (i < clusterSize) {
  // ── Cluster A (left, blue-indigo) ──
  const li = i;
  const phi = li * 2.399 + t * 0.3; // golden angle spiral
  const theta = Math.acos(1 - 2 * (li / clusterSize));
  const r = 8 * breathScale * (0.7 + 0.3 * Math.sin(li * 0.05 + time));
  px = -halfSep + r * Math.sin(theta) * Math.cos(phi);
  py = r * Math.sin(theta) * Math.sin(phi);
  pz = r * Math.cos(theta);
  // Fiedler modulates cluster tightness
  px = px * (1.0 - fiedler * 3);
  hue = 0.65; // blue
  sat = 0.8;
  lit = 0.45 + 0.2 * Math.sin(li * 0.1 + time * 2);

} else if (i < 2 * clusterSize) {
  // ── Cluster B (right, red-orange) ──
  const li = i - clusterSize;
  const phi = li * 2.399 + t * 0.3 + 3.14159;
  const theta = Math.acos(1 - 2 * (li / clusterSize));
  const r = 8 * breathScale * (0.7 + 0.3 * Math.cos(li * 0.07 + time));
  px = halfSep + r * Math.sin(theta) * Math.cos(phi);
  py = r * Math.sin(theta) * Math.sin(phi);
  pz = r * Math.cos(theta);
  px = px * (1.0 - fiedler * 3);
  hue = 0.05; // orange-red
  sat = 0.85;
  lit = 0.45 + 0.2 * Math.cos(li * 0.1 + time * 2);

} else if (i < 2 * clusterSize + coplanarTotal) {
  // ── Co-planar bridges (3 channels, HEAVY flow) ──
  const bridgeI = i - 2 * clusterSize;
  const channelIdx = Math.floor(bridgeI / coPerBridge);
  const localI = bridgeI % coPerBridge;

  // Direction vectors for co-planar channels (xy-plane)
  // Channel 0: (1, 1, 0) — diagonal up
  // Channel 1: (1,-1, 0) — diagonal down
  // Channel 2: (1, 0, 0) — straight across
  const dirY = channelIdx === 0 ? 1 : channelIdx === 1 ? -1 : 0;
  const dirZ = 0;

  // Particle flows from A to B along the channel
  const progress = (localI / coPerBridge + t * 0.15) % 1.0;
  const x = -halfSep + progress * separation;

  // Lateral spread — thick, flowing rivers
  const channelWidth = 3.0 * breathScale;
  const spiralAngle = localI * 0.3 + progress * 12 + time;
  const lateralR = channelWidth * (0.3 + 0.7 * Math.sin(localI * 0.07));
  const y = dirY * 12 * (0.5 - Math.abs(progress - 0.5)) +
            lateralR * Math.cos(spiralAngle);
  const z = dirZ * 12 * (0.5 - Math.abs(progress - 0.5)) +
            lateralR * Math.sin(spiralAngle);

  // Turbulence
  const tx = turbulence * 2 * Math.sin(localI * 1.7 + time * 2);
  const ty = turbulence * 2 * Math.cos(localI * 2.3 + time * 1.5);

  px = x + tx;
  py = y + ty;
  pz = z;

  // Color: transition from A-blue to B-red along the flow
  hue = 0.65 - progress * 0.60; // blue → red
  sat = 0.95;
  lit = 0.55 + 0.2 * Math.sin(progress * 6.28 + time * 3);

} else {
  // ── Cross-planar bridges (3 channels, SPARSE flow) ──
  const bridgeI = i - 2 * clusterSize - coplanarTotal;
  const channelIdx = Math.floor(bridgeI / crossPerBridge);
  const localI = bridgeI % crossPerBridge;

  // Direction vectors for cross-planar channels (out of xy-plane)
  // Channel 3: (1, 0, 1) — up-forward
  // Channel 4: (1, 0,-1) — down-forward
  // Channel 5: (0, 1, 1) — lateral-up
  const dirY2 = channelIdx === 2 ? 1 : 0;
  const dirZ2 = channelIdx === 0 ? 1 : channelIdx === 1 ? -1 : 1;
  const dirX2 = channelIdx === 2 ? 0 : 1;

  // Sparse flow — ratio slider makes these nearly empty
  const effectiveCount = Math.max(1, Math.floor(crossPerBridge / (axisRatio * 0.5 + 1)));
  const visible = localI < effectiveCount;

  const progress = (localI / crossPerBridge + t * 0.08) % 1.0;
  const x = -halfSep * dirX2 + progress * separation * dirX2;
  const channelWidth = 1.5 * breathScale;
  const y = dirY2 * 15 * (0.5 - Math.abs(progress - 0.5)) +
            channelWidth * Math.sin(localI * 0.5 + time);
  const z = dirZ2 * 15 * (0.5 - Math.abs(progress - 0.5)) +
            channelWidth * Math.cos(localI * 0.5 + time);

  if (visible) {
    px = (dirX2 === 0 ? 0 : (-halfSep + progress * separation)) +
         turbulence * Math.sin(localI * 2 + time);
    py = y;
    pz = z;
  } else {
    // Park invisible particles far away
    px = 9999;
    py = 9999;
    pz = 9999;
  }

  // Color: dim, desaturated — the suppressed channels
  hue = 0.35; // green-teal (distinct from A/B)
  sat = 0.4;
  lit = visible ? 0.3 + 0.1 * Math.sin(time + localI) : 0.0;
}

target.set(px, py, pz);
color.setHSL(hue, sat, Math.max(0.02, Math.min(0.95, lit)));

// ── HUD ──
if (i === 0) {
  const ratioText = axisRatio > 50 ? "22,477:1" : Math.floor(axisRatio * 224) + ":1";
  setInfo(
    "RHOMBI-BRIDGE — 6-Channel FCC Topology",
    "LoRA A (blue) ↔ B (red) connected by 6 directional bridges. " +
    "3 co-planar channels (rivers) carry " + ratioText + " more coupling " +
    "than 3 cross-planar channels (trickles). " +
    "Fiedler λ₂ = " + fiedler.toFixed(3) + " (scale-invariant at ~0.10). " +
    "— Promptcrafted / TASUMER MAF"
  );
  annotate("clusterA", new THREE.Vector3(-halfSep, 0, 0), "A Projection");
  annotate("clusterB", new THREE.Vector3(halfSep, 0, 0), "B Projection");
  annotate("coplanar", new THREE.Vector3(0, 14, 0), "Co-planar (high)");
  annotate("crossplanar", new THREE.Vector3(0, 0, 16), "Cross-planar (suppressed)");
}
