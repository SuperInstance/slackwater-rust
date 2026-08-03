# lattice-core

**Layer 4 — Eisenstein A₂ lattice for exact build placement. Triangular/hexagonal tiling with integer arithmetic.**

lattice-core provides the geometric foundation for the Slackwater build system. Every part placement snaps to an exact point on the A₂ (Eisenstein) lattice, guaranteeing zero floating-point drift, isotropic neighborhoods, and exact collision detection.

The A₂ lattice produces triangular tiling — the natural grid for Roblox builds because parts rotate at 60° increments and snap cleanly with no privileged direction.

---

## Module index

| Module | Primary export | Description |
|--------|---------------|-------------|
| `eisenstein` | `EisensteinPoint` | Lattice points: norm, distance, rotation, neighbors |
| `snap` | `snap_position`, `snap_rotation`, `snap_all` | Continuous → discrete snapping |
| `neighbors` | `collides`, `nearest_unoccupied`, `build_boundary` | Spatial queries |
| `region` | `LatticeRegion` | Rectangular lattice regions for lots and districts |

---

## Eisenstein A₂ lattice reference

### Definition

The Eisenstein integers are `ℤ[ω]` where `ω = e^(2πi/3) = -1/2 + i√3/2`. They form a triangular/hexagonal lattice in the complex plane.

Every point `(a, b)` in Eisenstein coordinates maps to Cartesian:

```
x = (a − b/2) · scale
y = b · (√3/2) · scale
```

where `scale` is the lattice unit in studs (default: 4.0).

### Triangular grid visualization

```
              b
              ↑
    · ─ ─ ─ · ─ ─ ─ · ─ ─ ─ ·         b = 2
     \     / \     / \     / \
      · ─ ─ · ─ ─ · ─ ─ · ─           b = 1
     / \  / \  / \  / \  / \
    · ─ · ─ ─ · ─ ─ · ─ ─ ·           b = 0
   /
  origin (0,0)
   ←──────── a ────────→

  Each "·" is a lattice point.
  Each edge = 1 lattice unit = `scale` studs.
  Each cell is an equilateral triangle.
  Six triangles meet at each vertex.
```

### Norm

The lattice norm (squared distance from origin):

```
N(a + bω) = a² − ab + b²
```

This is always a non-negative integer. The six units (norm-1 elements) are: `±1`, `±ω`, `±(1+ω)`, corresponding to `(±1, 0)`, `(0, ±1)`, `(±1, ±1)`.

### `EisensteinPoint`

```rust
pub struct EisensteinPoint {
    pub a: i32,
    pub b: i32,
}
```

**Derives:** `Clone`, `Copy`, `Debug`, `Hash`, `Eq`, `PartialEq`, `Serialize`, `Deserialize`.

**Implements:** `Ord` (by norm, then `a`, then `b`), `Add`, `Sub`, `Neg`, `Display`, `Default`.

### Key methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `new(a, b)` | `const fn` | `EisensteinPoint` |
| `origin()` | `const fn` | `new(0, 0)` |
| `norm(self)` | `const fn` | `i64` — `a² − ab + b²` |
| `lattice_distance(&self, other)` | | `u32` — hex graph distance |
| `euclidean_distance(&self, other)` | | `f64` — `√(norm(diff))` |
| `to_cartesian(self, scale)` | | `(f64, f64)` |
| `from_cartesian(x, y, scale)` | | `EisensteinPoint` — nearest lattice point |
| `rotate_60(self)` | `const fn` | 60° CCW rotation |
| `rotate_120(self)` | `const fn` | 120° CCW |
| `rotate_180(self)` | `const fn` | 180° (= negation) |
| `rotate_240(self)` | `const fn` | 240° CCW |
| `rotate_300(self)` | `const fn` | 300° CCW |
| `neighbors(self)` | | `[Self; 6]` — six equidistant neighbors |
| `within(self, radius)` | | `Vec<Self>` — all points within hex distance |
| `add(&self, other)` / `sub(&self, other)` | `const fn` | Vector add/subtract |
| `neg(self)` | `const fn` | Negation |
| `conjugate(self)` | `const fn` | Complex conjugate |
| `is_zero(self)` / `is_unit(self)` | `const fn` / `fn` | Boolean checks |

---

## Rotation formula derivation

### 60° counterclockwise

Multiplication by `ω`:

```
(a + bω) · ω = aω + bω²
```

Since `ω² = −1 − ω` (from `ω² + ω + 1 = 0`):

```
= aω + b(−1 − ω)
= −b + (a − b)ω
```

Therefore: **`rotate_60(a, b) = (a − b, a)`**

### Verification: six applications return to start

```
rot₀ = (a, b)
rot₁ = (a − b, a)          [60°]
rot₂ = (a − b − a, a − b) = (−b, a − b)    [120°]
rot₃ = (−b − (a−b), −b) = (−a, −b)         [180° = negation ✓]
rot₄ = (−a − (−b), −a) = (b − a, −a)        [240°]
rot₅ = (b − a − (−a), b − a) = (b, b − a)  [300°]
rot₆ = (b − (b−a), b) = (a, b)              [360° = identity ✓]
```

---

## Lattice distance

The hex graph distance (minimum number of single-step moves between two points):

```
Δa = a₁ − a₂,  Δb = b₁ − b₂

if sign(Δa) == sign(Δb) or either is 0:
    distance = max(|Δa|, |Δb|)
else:
    distance = |Δa| + |Δb|
```

**Same-sign case:** The directions reinforce (e.g., moving `(3, 2)` can be done in 3 steps using diagonal `(1,1)` moves plus one `(1,0)` move).

**Opposite-sign case:** The directions conflict (e.g., `(3, −2)` requires 3 steps in the `a`-direction plus 2 steps in the `−b`-direction = 5 total, since no single-step move covers both).

---

## Snapping algorithm

### Position snapping

```rust
pub fn snap_position(x: f64, y: f64) -> EisensteinPoint
```

Snaps Cartesian `(x, y)` to the nearest A₂ lattice point at the default scale (4.0 studs).

**Algorithm:**

1. Inverse transform:
   ```
   b_raw = 2y / (scale · √3)
   a_raw = x / scale + b_raw / 2
   ```
2. Search the 3×3 neighborhood of `(floor(a_raw), floor(b_raw))` to find the candidate with minimum Cartesian distance to `(x, y)`.

The neighborhood search guarantees correctness — simple rounding of the inverse transform can be off by one in edge cases near cell boundaries.

**Idempotence:** `snap_position` applied to the Cartesian coordinates of a lattice point always returns that same point.

### Rotation snapping

```rust
pub fn snap_rotation(degrees: f64) -> i32      // → {0, 60, 120, 180, 240, 300}
pub fn snap_rotation_index(degrees: f64) -> i32 // → {0, 1, 2, 3, 4, 5}
```

Quantizes a rotation to the nearest 60° increment. The hexagonal lattice has 6-fold symmetry, so all rotations are equivalent under lattice translation.

### Height snapping

```rust
pub fn snap_height(y: f64, grid_size: f64) -> i32
```

Vertical is **not** on the Eisenstein lattice — it's a regular 1D grid. Terrain is irregular; the lattice governs horizontal agreement, not height.

### Full placement snapping

```rust
pub fn snap_all(x, y, z, rot_degrees, height_grid) -> SnappedPlacement
```

Converts continuous `(x, y, z, rotation)` into a fully discrete `SnappedPlacement`:

```rust
pub struct SnappedPlacement {
    pub lattice: EisensteinPoint,  // snapped (a, b)
    pub height: i32,                // snapped vertical grid index
    pub rotation: i32,              // 0–5 (multiples of 60°)
}
```

---

## Spatial queries

### Collision detection

```rust
pub fn collides(new_placement: &EisensteinPoint, occupied: &[EisensteinPoint], min_distance: u32) -> bool
```

Returns `true` if any occupied point is within `min_distance` hex steps of the new placement. All comparisons use exact integer arithmetic — no floating-point tolerance bands.

### Nearest unoccupied point

```rust
pub fn nearest_unoccupied(occupied: &[EisensteinPoint], target: &EisensteinPoint) -> EisensteinPoint
```

Searches outward in hex-distance rings from `target` (up to radius 20). If `target` is free, returns it immediately. Returns `target` as fallback if nothing found within the search radius.

### Occupied points in radius

```rust
pub fn occupied_in_radius(occupied: &[EisensteinPoint], center: &EisensteinPoint, radius: u32) -> Vec<EisensteinPoint>
```

Filters the occupied list to points within `radius` hex steps of `center`.

### Build boundary

```rust
pub fn build_boundary(occupied: &[EisensteinPoint]) -> Vec<EisensteinPoint>
```

Returns occupied points with at least one unoccupied neighbor — the "frontier" where a build can expand. Interior points (all neighbors occupied) are excluded.

---

## Regions

### `LatticeRegion`

```rust
pub struct LatticeRegion {
    pub min: EisensteinPoint,
    pub max: EisensteinPoint,
}
```

A rectangular region in lattice coordinates `(a, b)`. Inclusive of both corners.

| Method | Description |
|--------|-------------|
| `new(min, max)` | Auto-normalizes corners |
| `centered(center, half_extent)` | Symmetric region around a point |
| `contains(&point)` | Inclusive bounds check |
| `area()` | `(width) × (height)` lattice points |
| `iter()` | Row-major iterator over all points |
| `expand(by)` | Grow in all directions |
| `intersects(&other)` | Overlap test |
| `intersection(&other)` | `Option<Self>` — clipped region |
| `union(&other)` | Bounding box of both |

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_SCALE` | `4.0` | Lattice unit in studs |
| `DEFAULT_HEIGHT_GRID` | `1.0` | Vertical grid size in studs |
| `NEIGHBOR_DIRECTIONS` | `[(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1)]` | Six neighbor offsets |
| `SQRT3` | `1.7320508075688772` | √3 as f64 |

---

## Crate metadata

- **Edition:** 2024
- **Dependencies:** `serde`
- **Dev dependencies:** `criterion`, `approx`, `serde_json`
- **Unsafe code:** `#![deny(unsafe_code)]`
- **Clippy:** `#![warn(clippy::all)]`
- **Tests:** 70 (unit + integration)
