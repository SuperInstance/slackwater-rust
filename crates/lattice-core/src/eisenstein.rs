//! Eisenstein A₂ lattice — exact integer arithmetic.
//!
//! The Eisenstein integers are ℤ[ω] where ω = e^(2πi/3) = -1/2 + i√3/2.
//! They form a triangular/hexagonal lattice in the complex plane.
//!
//! Every point (a, b) in Eisenstein coordinates maps to Cartesian:
//!   x = (a - b/2) · scale
//!   y = b · (√3/2) · scale
//!
//! Each lattice point has exactly six equidistant neighbors at unit distance.
//! The lattice norm (squared distance from origin) is:
//!   N(a + bω) = a² - ab + b²
//!
//! This is exact integer arithmetic. No floating point. No drift.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// √3 — the square root of 3 as an f64 constant.
///
/// Used for Cartesian conversion: `y = b · (√3/2) · scale`.
/// We use the `std` sqrt for full f64 precision rather than a hardcoded literal.
const SQRT3: f64 = 1.732_050_807_568_877_2;

/// √3 / 2 — convenience constant derived from [`SQRT3`].
const SQRT3_OVER_2: f64 = SQRT3 / 2.0;

/// The six neighbor directions on the A₂ lattice.
///
/// With ω = e^(2πi/3), the norm is N(a+bω) = a² - ab + b².
/// The six units (norm-1 elements) are: ±1, ±ω, ±(1+ω).
///
/// **Note:** This uses (1,1) / (-1,-1) — not (1,-1) / (-1,1) — because
/// ω² = -1 - ω, so (1+ω) is a unit. The neighbor directions are:
///   (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1)
pub const NEIGHBOR_DIRECTIONS: [(i32, i32); 6] =
    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)];

/// An exact point on the Eisenstein A₂ lattice.
///
/// Represents the value `a + bω` where `ω = e^(2πi/3)`.
/// All arithmetic is exact integer — no floating point, no drift.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct EisensteinPoint {
    /// The real-integer coefficient.
    pub a: i32,
    /// The ω coefficient.
    pub b: i32,
}

impl EisensteinPoint {
    /// Create a new Eisenstein point.
    #[inline]
    #[must_use]
    pub const fn new(a: i32, b: i32) -> Self {
        Self { a, b }
    }

    /// The origin (0, 0).
    #[inline]
    #[must_use]
    pub const fn origin() -> Self {
        Self { a: 0, b: 0 }
    }

    // ── Norm and distance ───────────────────────────────

    /// Squared distance from origin: `N(a + bω) = a² - ab + b²`.
    ///
    /// Always returns a non-negative integer.
    #[inline]
    #[must_use]
    pub const fn norm(self) -> i64 {
        let a = self.a as i64;
        let b = self.b as i64;
        a * a - a * b + b * b
    }

    /// Hexagonal grid distance (number of steps) between two lattice points.
    ///
    /// This is the graph distance on the neighbor graph — the minimum
    /// number of single-step moves to get from `self` to `other`.
    ///
    /// For the A₂ lattice with neighbor directions
    /// `{(±1,0), (0,±1), (±1,±1)}`:
    /// - If `da` and `db` have the same sign (or either is 0): `max(|da|, |db|)`
    /// - If they have opposite signs: `|da| + |db|`
    #[inline]
    #[must_use]
    pub fn lattice_distance(self, other: &Self) -> u32 {
        let da = self.a - other.a;
        let db = self.b - other.b;
        hex_distance_raw(da, db)
    }

    /// Euclidean distance between two lattice points (for spatial queries).
    ///
    /// Uses the exact norm: `sqrt(norm(diff))`.
    #[inline]
    #[must_use]
    pub fn euclidean_distance(self, other: &Self) -> f64 {
        let diff = self.sub(other);
        (diff.norm() as f64).sqrt()
    }

    // ── Coordinate conversion ───────────────────────────

    /// Convert to Cartesian `(x, y)` coordinates at the given scale.
    ///
    /// `x = (a - b/2) · scale`
    /// `y = b · (√3/2) · scale`
    ///
    /// This is for rendering/display only — never for agreement checks.
    #[inline]
    #[must_use]
    pub fn to_cartesian(self, scale: f64) -> (f64, f64) {
        let x = (self.a as f64 - self.b as f64 / 2.0) * scale;
        let y = self.b as f64 * SQRT3_OVER_2 * scale;
        (x, y)
    }

    /// Convert to Cartesian at unit scale (scale = 1.0).
    #[inline]
    #[must_use]
    pub fn to_cartesian_unit(self) -> (f64, f64) {
        self.to_cartesian(1.0)
    }

    /// Snap a Cartesian `(x, y)` point to the nearest Eisenstein lattice point.
    ///
    /// Inverse of [`to_cartesian`](Self::to_cartesian):
    ///   `b = 2y / (scale · √3)`
    ///   `a = x / scale + b/2`
    /// Then round both to the nearest integer.
    #[must_use]
    pub fn from_cartesian(x: f64, y: f64, scale: f64) -> Self {
        let inv_scale = 1.0 / scale;
        let b_raw = 2.0 * y * inv_scale / SQRT3;
        let a_raw = x * inv_scale + b_raw / 2.0;

        // Try the rounded candidate and its neighbors to find the true nearest.
        let b_floor = b_raw.floor() as i32;
        let a_floor = a_raw.floor() as i32;

        let mut best = Self {
            a: a_raw.round() as i32,
            b: b_raw.round() as i32,
        };
        let mut best_dist = {
            let (bx, by) = best.to_cartesian(scale);
            (bx - x) * (bx - x) + (by - y) * (by - y)
        };

        for db in -1..=1 {
            for da in -1..=1 {
                let candidate = Self {
                    a: a_floor + da,
                    b: b_floor + db,
                };
                let (cx, cy) = candidate.to_cartesian(scale);
                let dist = (cx - x) * (cx - x) + (cy - y) * (cy - y);
                if dist < best_dist {
                    best = candidate;
                    best_dist = dist;
                }
            }
        }
        best
    }

    /// Snap at unit scale (scale = 1.0).
    #[inline]
    #[must_use]
    pub fn from_cartesian_unit(x: f64, y: f64) -> Self {
        Self::from_cartesian(x, y, 1.0)
    }

    // ── Rotation ────────────────────────────────────────

    /// Rotate 60° counterclockwise about the origin.
    ///
    /// Multiplication by ω: `(a + bω) · ω = aω + bω² = aω + b(-1-ω) = -b + (a-b)ω`
    #[inline]
    #[must_use]
    pub const fn rotate_60(self) -> Self {
        Self::new(self.a - self.b, self.a)
    }

    /// Rotate 120° counterclockwise (apply rotate_60 twice).
    #[inline]
    #[must_use]
    pub const fn rotate_120(self) -> Self {
        self.rotate_60().rotate_60()
    }

    /// Rotate 180° (negation).
    #[inline]
    #[must_use]
    pub const fn rotate_180(self) -> Self {
        Self::new(-self.a, -self.b)
    }

    /// Rotate 240° counterclockwise (= rotate_60 applied 4 times, or -60° once).
    #[inline]
    #[must_use]
    pub const fn rotate_240(self) -> Self {
        self.rotate_120().rotate_60()
    }

    /// Rotate 300° counterclockwise (= rotate_60 applied 5 times, or -120° once).
    #[inline]
    #[must_use]
    pub const fn rotate_300(self) -> Self {
        self.rotate_240().rotate_60()
    }

    // ── Neighbors ───────────────────────────────────────

    /// Return the six equidistant neighbors of this lattice point.
    ///
    /// On the A₂ lattice, every point has exactly six neighbors,
    /// all at unit distance. There is no privileged direction.
    #[inline]
    #[must_use]
    pub fn neighbors(self) -> [Self; 6] {
        let (a, b) = (self.a, self.b);
        [
            Self::new(a + 1, b),
            Self::new(a - 1, b),
            Self::new(a, b + 1),
            Self::new(a, b - 1),
            Self::new(a + 1, b + 1),
            Self::new(a - 1, b - 1),
        ]
    }

    /// All lattice points within `radius` (hex distance) of this point,
    /// excluding self, ordered by distance then coordinates.
    #[must_use]
    pub fn within(self, radius: u32) -> Vec<Self> {
        if radius == 0 {
            return Vec::new();
        }
        let r = radius as i32;
        let mut result: Vec<Self> = Vec::new();
        for da in -r..=r {
            for db in -r..=r {
                if da == 0 && db == 0 {
                    continue;
                }
                let offset = Self::new(da, db);
                if hex_distance_raw(da, db) <= radius {
                    result.push(self.add(&offset));
                }
            }
        }
        result.sort_by(|a, b| {
            let dist_cmp = a.lattice_distance(&self).cmp(&b.lattice_distance(&self));
            if dist_cmp != Ordering::Equal {
                return dist_cmp;
            }
            match a.a.cmp(&b.a) {
                Ordering::Equal => a.b.cmp(&b.b),
                other => other,
            }
        });
        result
    }

    // ── Arithmetic ──────────────────────────────────────

    /// Lattice addition: `(a+c, b+d)`.
    #[inline]
    #[must_use]
    pub const fn add(self, other: &Self) -> Self {
        Self::new(self.a + other.a, self.b + other.b)
    }

    /// Lattice subtraction: `(a-c, b-d)`.
    #[inline]
    #[must_use]
    pub const fn sub(self, other: &Self) -> Self {
        Self::new(self.a - other.a, self.b - other.b)
    }

    /// Negation (same as rotate_180).
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        Self::new(-self.a, -self.b)
    }

    /// The complex conjugate: `a + bω → (a-b) - bω`.
    #[inline]
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Self::new(self.a - self.b, -self.b)
    }

    /// Is this the origin?
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.a == 0 && self.b == 0
    }

    /// Is this a unit (norm-1 element)? The six units are: ±1, ±ω, ±(1+ω).
    #[inline]
    #[must_use]
    pub fn is_unit(self) -> bool {
        self.norm() == 1
    }
}

// ── Trait implementations ──────────────────────────────

impl Default for EisensteinPoint {
    fn default() -> Self {
        Self::origin()
    }
}

impl std::fmt::Display for EisensteinPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.b == 0 {
            return write!(f, "E({})", self.a);
        }
        if self.a == 0 {
            return write!(f, "E({}ω)", self.b);
        }
        let sign = if self.b > 0 { "+" } else { "-" };
        write!(f, "E({} {} {}ω)", self.a, sign, self.b.unsigned_abs())
    }
}

impl Ord for EisensteinPoint {
    /// Order by norm, then by coordinates for determinism.
    fn cmp(&self, other: &Self) -> Ordering {
        match self.norm().cmp(&other.norm()) {
            Ordering::Equal => {}
            other => return other,
        }
        match self.a.cmp(&other.a) {
            Ordering::Equal => self.b.cmp(&other.b),
            other => other,
        }
    }
}

impl PartialOrd for EisensteinPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── Operator overloads ─────────────────────────────────

impl std::ops::Add for EisensteinPoint {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.a + rhs.a, self.b + rhs.b)
    }
}

impl std::ops::Sub for EisensteinPoint {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.a - rhs.a, self.b - rhs.b)
    }
}

impl std::ops::Neg for EisensteinPoint {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.a, -self.b)
    }
}

// ── Free functions ─────────────────────────────────────

/// Compute hex distance from raw `(da, db)` deltas.
///
/// If `da * db >= 0` (same sign or either is zero): `max(|da|, |db|)`.
/// Otherwise: `|da| + |db|`.
#[inline]
#[must_use]
fn hex_distance_raw(da: i32, db: i32) -> u32 {
    if da.signum() == db.signum() || da == 0 || db == 0 {
        da.unsigned_abs().max(db.unsigned_abs())
    } else {
        da.unsigned_abs() + db.unsigned_abs()
    }
}
