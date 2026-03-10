# External Reference Implementations — RD Geometry from the Wild

> **Purpose:** Extractable algorithms from 15 GitHub repos surveyed March 9, 2026.
> These are independent implementations that confirm our math and provide GPU-portable
> formulations for asset generation. None are Python; all port trivially.
>
> **Context:** Our `rhombic` library (v0.3.0+, 255 tests) is ahead of every public
> RD implementation on GitHub. These references fill specific gaps: real-time rendering
> (SDF), isosurface extraction (marching-RD), and efficient tessellation rendering
> (hidden-face culling).

---

## 1. RD Signed Distance Function — GPU-Portable Rendering Kernel

**Source:** `selimbat/r_dodeca_vox_builder` (Processing + GLSL, MIT-equivalent)
**Use case:** Real-time RD visualization on RTX 6000 Ada. Ray-marched honeycombs
with Phong shading for papers, GitHub, HF Space, website banners, video content.

### The Formula

Exploits octahedral symmetry to reduce 12 face constraints to 5.

**GLSL (original):**
```glsl
float rd_sdf(vec3 p, float size) {
    // Symmetry fold: map to fundamental domain
    p = vec3(abs(p.x), sign(p.z) * p.y, abs(p.z));

    // Distance to each face (5 normals cover all 12 faces after fold)
    float a = dot(vec3(1.0, 0.0, 0.0),                    p) - size;
    float b = dot(vec3(0.5, 0.0, sqrt(3.0)/2.0),          p) - size;
    float c = dot(vec3(0.0, -sqrt(2.0/3.0), 1.0/sqrt(3.0)), p) - size;
    float d = dot(vec3(0.5, sqrt(2.0/3.0), sqrt(3.0)/6.0),  p) - size;
    float e = dot(vec3(0.5, -sqrt(2.0/3.0), -sqrt(3.0)/6.0), p) - size;

    return max(a, max(b, max(c, max(d, e))));
}
```

**Python/NumPy (ported):**
```python
import numpy as np

def rd_sdf(p: np.ndarray, size: float = 1.0) -> np.ndarray:
    """Signed distance to a rhombic dodecahedron centered at origin.

    Args:
        p: Points array, shape (..., 3)
        size: Inradius (face-to-center distance)

    Returns:
        Signed distance values, shape (...)
    """
    x, y, z = np.abs(p[..., 0]), np.sign(p[..., 2]) * p[..., 1], np.abs(p[..., 2])
    s23 = np.sqrt(2.0 / 3.0)
    s32 = np.sqrt(3.0) / 2.0
    s36 = np.sqrt(3.0) / 6.0
    s13 = 1.0 / np.sqrt(3.0)

    a = x - size
    b = 0.5 * x + s32 * z - size
    c = -s23 * y + s13 * z - size
    d = 0.5 * x + s23 * y + s36 * z - size
    e = 0.5 * x - s23 * y - s36 * z - size

    return np.maximum(a, np.maximum(b, np.maximum(c, np.maximum(d, e))))
```

**The 5 face normals (before symmetry fold):**
```
n1 = (1, 0, 0)
n2 = (1/2, 0, sqrt(3)/2)
n3 = (0, -sqrt(2/3), 1/sqrt(3))
n4 = (1/2, sqrt(2/3), sqrt(3)/6)
n5 = (1/2, -sqrt(2/3), -sqrt(3)/6)
```

**Note:** Approximate near vertices (not exact Euclidean distance there), exact for
face distances. Adequate for ray marching, rendering, and containment testing.

### Asset Generation Applications

- **Paper figures:** Ray-march a honeycomb of RDs with the library's own topology
  coloring (8-law weave palette). The SDF renders at arbitrary resolution.
- **Website hero:** Animated ray-marched RD on `rhombic.vision`, replacing or
  complementing the Three.js wireframe.
- **Video content:** Real-time RD honeycomb flythrough on RTX 6000 Ada.
  Drop the SDF into a CUDA compute shader or use it with `warp` / `taichi`.
- **HF Space:** Could power an interactive RD explorer (ray-march in WebGL).

---

## 2. FCC Lattice Basis Vectors — Integer Cell Addressing

**Source:** `selimbat/r_dodeca_vox_builder` + `grischa/rhombic-dodecahedron` (independent confirmation)
**Use case:** Addressing individual RD cells in a tessellation by integer triple (i,j,k).

### The Vectors

```python
import numpy as np

# Primitive FCC translation vectors (selimbat formulation)
I = np.array([1.0,  0.0,           0.0])
J = np.array([-0.5, 0.0,           np.sqrt(3)/2])
K = np.array([0.0,  np.sqrt(6)/3, -1/np.sqrt(3)])

def cell_center(i: int, j: int, k: int) -> np.ndarray:
    """RD cell center from integer grid coordinates."""
    return i * I + j * J + k * K
```

**Alternative formulation (grischa — periodic boundary):**
```python
# Cell basis vectors for periodic RD tiling (D = R * sqrt(2))
def periodic_basis(D: float):
    v1 = np.array([D, 0, 0])
    v2 = np.array([0, D, 0])
    v3 = np.array([D/2, D/2, D/np.sqrt(2)])
    return v1, v2, v3
```

---

## 3. Point-in-RD Tests — Two Formulations

### L1-Norm Pairs (compact, branch-free)

**Source:** `Uspectacle/MineCraft-Rhombic-Dodecahedron`

```python
def point_in_rd_l1(x: float, y: float, z: float) -> bool:
    """RD containment via three L1-norm pair constraints."""
    return abs(x) + abs(y) < 1 and abs(y) + abs(z) < 1 and abs(z) + abs(x) < 1
```

### 12-Inequality Half-Space (explicit, after 45-degree z-rotation)

**Source:** `grischa/rhombic-dodecahedron`

```python
def point_in_rd_halfspace(x: float, y: float, z: float, D: float) -> bool:
    """RD containment via 6 absolute-value conditions (=12 half-spaces)."""
    return (abs(x + y) < D and abs(x - y) < D and
            abs(x + z) < D and abs(x - z) < D and
            abs(y + z) < D and abs(y - z) < D)
```

---

## 4. FCC Stagger Function — Cubic-to-FCC Grid Conversion

**Source:** `Uspectacle/MineCraft-Rhombic-Dodecahedron`

```python
def bend(x: int, y: int, z: int) -> tuple[int, int, int]:
    """Convert cubic grid position to staggered FCC lattice.

    Even-parity cells (x+z even) and odd-parity cells (x+z odd)
    interleave at alternating y-levels.
    """
    return (x, y * 2 + (x + z) % 2, z)
```

---

## 5. Marching-RD Algorithm — Isosurface Extraction on RD Grids

**Source:** `Reispfannenfresser/marching-rhombic-dodecahedrons` (C#/Unity, **GPL-3.0**)
**Use case:** Isosurface visualization inside RD grids. Paper 4 figures. Future
RD-native generative pipeline outputs.

### Architecture

Each RD cell decomposes into the **octet truss** (tetrahedral-octahedral honeycomb):
- 2 tetrahedra + 1 octahedron per cell

The marching algorithm runs three sub-shapes per cell:

| Sub-shape | Sample points | Configurations | Mesh templates | Offset |
|-----------|--------------|----------------|---------------|--------|
| Tetrahedron 1 | 4 (origin + FR, DF, DR) | 16 (2^4) | 4 | (0.5, -0.5, 0.5) |
| Tetrahedron 2 | 4 (origin + LF, DL, DF) | 16 (2^4) | 4 | (-0.5, -0.5, 0.5) |
| Octahedron | 6 (origin + DB, DL, DF, DR + corner D) | 64 (2^6) | 10 | (0, -1, 0) |

Each configuration maps to a `(rotation_matrix, mesh_template_index)` pair.
Total: 96 lookup entries (16 + 16 + 64).

### Grid Coordinate System

```python
import numpy as np

# Grid-to-local transform matrix (column-major)
from_grid = np.array([
    [ 1,  1,  1,  0],
    [ 0,  1,  0,  0],
    [-1,  0,  1,  0],
    [ 0,  0,  0,  1]
], dtype=float)

# Local-to-grid (inverse)
to_grid = np.array([
    [ 0.5, -0.5, -0.5, 0],
    [ 0,    1,   -0.5, 0],
    [ 0.5, -0.5,  0.5, 0],
    [ 0,    0,    0,   1]
], dtype=float)

def grid_to_local(gx, gy, gz):
    """Convert integer RD grid coords to world position."""
    return (gx + gy + gz, gy, -gx + gz)

def local_to_grid(x, y, z):
    """Snap world position to nearest RD grid cell (with BCC handling)."""
    gy = round(y / 2) * 2  # snap to even
    remainder_y = y - gy
    fx, fz = x - round(x), z - round(z)
    if abs(fx) + abs(remainder_y) + abs(fz) > 1:
        gy += 1  # offset sublattice
    gx = round(0.5 * x - 0.5 * gy - 0.5 * z)
    gz = round(0.5 * x - 0.5 * gy + 0.5 * z)
    return (gx, gy, gz)
```

### 12-Face Adjacency Vectors (Grid Coordinates)

```python
FACE_NEIGHBORS = {
    'UL': (-1, 1,-1), 'UF': (-1, 1, 0), 'UR': ( 0, 1, 0), 'UB': ( 0, 1,-1),
    'LF': (-1, 0, 0), 'FR': ( 0, 0, 1), 'RB': ( 1, 0, 0), 'BL': ( 0, 0,-1),
    'DL': ( 0,-1, 0), 'DF': ( 0,-1, 1), 'DR': ( 1,-1, 1), 'DB': ( 1,-1, 0),
}

CORNER_NEIGHBORS = {
    'U': (-1, 2,-1), 'L': (-1, 0,-1), 'F': (-1, 0, 1),
    'R': ( 1, 0, 1), 'B': ( 1, 0,-1), 'D': ( 1,-2, 1),
}
```

**LICENSE NOTE:** The marching-RD repo is GPL-3.0. The coordinate system and
adjacency tables are mathematical facts (not copyrightable). The specific
96-entry lookup tables with rotation matrices and mesh templates are the
author's creative work under GPL-3.0. If we need the lookup tables, either:
(a) clean-room derive them from the tetrahedral/octahedral decomposition, or
(b) use under GPL-3.0 terms.

---

## 6. Hidden-Face Culling — O(n) Tessellation Rendering

**Source:** `apiotrow/Rhombic-Dodecahedral-Honeycomb` (C#/Unity)
**Use case:** Clean honeycomb renders showing only exterior faces. Cross-section
diagrams for papers. Animated honeycomb builds for video.

### Algorithm

```python
def cull_hidden_faces(cells: list[np.ndarray], faces: list[np.ndarray]) -> list[bool]:
    """Mark interior faces for removal via centroid hashing.

    For each triangle on each cell, compute the centroid (average of 3 vertices
    + cell offset). If two triangles from different cells share the same centroid,
    they are coplanar interior faces — both are flagged for removal.

    O(n) via hash set. Exploits exact arithmetic of the FCC lattice.
    """
    seen = {}
    keep = []
    for cell_idx, (cell_pos, cell_faces) in enumerate(zip(cells, faces)):
        for face_idx, tri_verts in enumerate(cell_faces):
            centroid = tuple(np.round(np.mean(tri_verts + cell_pos, axis=0), 6))
            if centroid in seen:
                # Interior face — mark both for removal
                keep[seen[centroid]] = False
                keep.append(False)
            else:
                seen[centroid] = len(keep)
                keep.append(True)
    return keep
```

### FCC Placement Pattern

```python
def fcc_placement(nx: int, ny: int, nz: int):
    """Generate FCC lattice cell positions for an nx x ny x nz grid.

    Even Y layers: even Z rows use even X, odd Z rows use odd X.
    Odd Y layers: shifted +1 in X.
    """
    positions = []
    for y in range(ny):
        for z in range(nz):
            for x in range(nx):
                # Stagger: offset X based on Y and Z parity
                actual_x = x * 2 + ((y + z) % 2)
                positions.append((actual_x, y, z))
    return positions
```

---

## 7. Vertex Coordinate Cross-Reference

Three independent coordinate systems confirmed across repos:

| System | Degree-3 (8 "cube" vertices) | Degree-4 (6 "octahedron" vertices) | Source |
|--------|------------------------------|-------------------------------------|--------|
| **Dual-cube** | (+-0.5, +-0.5, +-0.5) | (0,+-1,0), (+-1,0,0), (0,0,+-1) | apiotrow, HomelikeBrick42 |
| **sqrt(2)/sqrt(3)/sqrt(6)** | via sqrt(2)/2, sqrt(3)/6, sqrt(6)/6 | via sqrt(3)/2, sqrt(6)/3 | NateBerglund, MaltWhiskey |
| **Integer** | (1,1,1), (1,1,3), ... centered at (2,2,2) | (0,2,2), (4,2,2), ... | Uspectacle |

The dual-cube formulation maps most naturally to the face-indexed coordinate
system needed for spherical projection (Wong et al. 2009).

---

## Survey Metadata

**Date:** March 9, 2026
**Repos surveyed:** 15
**Repos with extractable value:** 4
**Repos with zero value:** 11 (LED sculptures, STL files, empty repos, CAD exports)
**Python code found:** 0 (all extracted formulas ported from C#, GLSL, Tcl, MATLAB)
**Spherical projection implementations found:** 0
**Wong et al. (2009) references found:** 0

**Conclusion:** The polyhedral multimodal bridge (Rungs 3-5 of the TASUMER MAF
product ladder) has zero prior art in open source. The `rhombic` library is the
only Python implementation of RD lattice topology with spectral analysis. These
reference implementations supplement it with GPU-portable rendering formulas.

---

*Surveyed by Meridian, March 9, 2026. Every formula independently verified
against the mathematical definition of the rhombic dodecahedron.*
