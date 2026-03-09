/**
 * Animated Rhombic Dodecahedron — Three.js hero visualization
 *
 * 8 trivalent vertices (cube corners) in FCC color (#B34444)
 * 6 tetravalent vertices (octahedral bridges) in cubic color (#3D3D6B)
 * 24 edges connecting them
 */

(function() {
  const container = document.getElementById('rd-canvas');
  if (!container || typeof THREE === 'undefined') return;

  // ── Colors ──
  const FCC_COLOR = 0xB34444;
  const CUBIC_COLOR = 0x3D3D6B;
  const EDGE_COLOR = 0x444466;

  // ── Scene setup ──
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(3.5, 2.5, 3.5);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  // ── Vertices ──
  // 8 cube corners (trivalent, degree 3)
  const cubeVerts = [
    [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
    [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1]
  ];

  // 6 octahedral bridges (tetravalent, degree 4)
  const octaVerts = [
    [-2,  0,  0], [ 2,  0,  0],
    [ 0, -2,  0], [ 0,  2,  0],
    [ 0,  0, -2], [ 0,  0,  2]
  ];

  // ── Edges (connect each cube vertex to its 3 nearest octahedral vertices) ──
  const edges = [];
  for (let ci = 0; ci < cubeVerts.length; ci++) {
    const c = cubeVerts[ci];
    for (let oi = 0; oi < octaVerts.length; oi++) {
      const o = octaVerts[oi];
      const dx = c[0] - o[0], dy = c[1] - o[1], dz = c[2] - o[2];
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      // Each cube vertex connects to 3 octahedral vertices at distance sqrt(5) ≈ 2.236
      // but scaled: our cube is ±1 and octa is ±2, so distance = sqrt(1+1+4) = sqrt(6) ≈ 2.449
      // Actually, distance from (-1,-1,-1) to (-2,0,0) = sqrt(1+1+1) = sqrt(3) ≈ 1.732
      // from (-1,-1,-1) to (0,-2,0) = sqrt(1+1+1) = sqrt(3) ≈ 1.732
      // from (-1,-1,-1) to (0,0,-2) = sqrt(1+1+1) = sqrt(3) ≈ 1.732
      // from (-1,-1,-1) to (2,0,0) = sqrt(9+1+1) = sqrt(11) ≈ 3.317 → NOT connected
      if (dist < 2.0) {
        edges.push([ci, 8 + oi]);
      }
    }
  }

  // ── Build meshes ──
  const group = new THREE.Group();

  // Cube corner spheres (FCC color)
  const cubeMat = new THREE.MeshPhongMaterial({ color: FCC_COLOR, emissive: FCC_COLOR, emissiveIntensity: 0.3 });
  const sphereGeo = new THREE.SphereGeometry(0.08, 16, 16);
  cubeVerts.forEach(v => {
    const mesh = new THREE.Mesh(sphereGeo, cubeMat);
    mesh.position.set(v[0], v[1], v[2]);
    group.add(mesh);
  });

  // Octahedral bridge diamonds (cubic color)
  const octaMat = new THREE.MeshPhongMaterial({ color: CUBIC_COLOR, emissive: CUBIC_COLOR, emissiveIntensity: 0.3 });
  const diamondGeo = new THREE.OctahedronGeometry(0.12, 0);
  octaVerts.forEach(v => {
    const mesh = new THREE.Mesh(diamondGeo, octaMat);
    mesh.position.set(v[0], v[1], v[2]);
    group.add(mesh);
  });

  // Edges
  const allVerts = [...cubeVerts, ...octaVerts];
  const edgeMat = new THREE.LineBasicMaterial({ color: EDGE_COLOR, transparent: true, opacity: 0.5 });
  edges.forEach(([a, b]) => {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...allVerts[a]),
      new THREE.Vector3(...allVerts[b])
    ]);
    group.add(new THREE.Line(geo, edgeMat));
  });

  // Also draw cube edges (skeleton) very faintly
  const cubeEdgeMat = new THREE.LineBasicMaterial({ color: FCC_COLOR, transparent: true, opacity: 0.12 });
  const cubeEdgePairs = [
    [0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]
  ];
  cubeEdgePairs.forEach(([a,b]) => {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...cubeVerts[a]),
      new THREE.Vector3(...cubeVerts[b])
    ]);
    group.add(new THREE.Line(geo, cubeEdgeMat));
  });

  scene.add(group);

  // ── Lighting ──
  scene.add(new THREE.AmbientLight(0x333344, 0.5));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
  dirLight.position.set(5, 5, 5);
  scene.add(dirLight);

  // ── Animate ──
  function animate() {
    requestAnimationFrame(animate);
    group.rotation.y += 0.003;
    group.rotation.x += 0.001;
    renderer.render(scene, camera);
  }
  animate();

  // ── Resize ──
  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
})();
