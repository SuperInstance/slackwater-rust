//! Exact arithmetic types for the FLUX constraint engine.
//!
//! Build coordinates are Eisenstein integers (a, b) representing positions
//! on the A₂ lattice. Velocities and confidences are INT8 (0–127,
//! MIDI-compatible). Ticks are uint32 (never negative, never float).
//!
//! No accumulating float drift — ever. Floats appear only in presentation
//! methods ([`EisensteinCoord::to_cartesian`], [`EisensteinCoord::distance_to`]),
//! never in agreement paths between agents.
/// A₂ lattice coordinate: z = a + bω, where ω = e^(2πi/3).
///
/// All build positions live here. Integer (a, b) means exact arithmetic
/// all the way down — two agents never disagree on where something is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EisensteinCoord {
    pub a: i32,
    pub b: i32,
}

/// Lattice scale in studs (from Grand Plan §4: s = 4 studs).
pub const LATTICE_SCALE: f64 = 4.0;

/// √3/2 — used in A₂ lattice coordinate transforms.
pub const FRAC_SQRT_3_2: f64 = 0.866_025_403_784_438_6;

/// ω = e^(2πi/3) — the cube root of unity used for A₂ lattice math.
pub const OMEGA_REAL: f64 = -0.5; // cos(120°)
pub const OMEGA_IMAG: f64 = 0.866_025_403_784_438_6; // sin(120°) = √3/2

// ── Type aliases for bounded quantities ─────────────────────────────

/// MIDI velocity / weight / confidence. Clamped to [0, 127].
pub type Velocity = u8;
/// Confidence value. Clamped to [0, 127].
pub type Confidence = u8;
/// Tick position on the shared BeatClock (96 PPQ). Never negative.
pub type Tick = u32;
/// MIDI channel (0–15).
pub type Channel = u8;
/// MIDI pitch / action code (0–127).
pub type Pitch = u8;

// ── INT8 saturation helpers ─────────────────────────────────────────

/// Clamp a value to the INT8 MIDI range [0, 127].
///
/// Values above 127 saturate to 127 (loudly, via log in the caller).
/// Values below 0 saturate to 0.
#[inline]
pub fn saturate_i8(value: i32) -> u8 {
    value.clamp(0, 127) as u8
}

/// Saturating add: clamps the result to [0, 127].
#[inline]
pub fn saturating_add(a: u8, b: u8) -> u8 {
    saturate_i8(a as i32 + b as i32)
}

/// Saturating sub: clamps the result to [0, 127].
#[inline]
pub fn saturating_sub(a: u8, b: u8) -> u8 {
    saturate_i8(a as i32 - b as i32)
}

// ── EisensteinCoord impl ────────────────────────────────────────────

impl EisensteinCoord {
    /// Create a new lattice coordinate.
    #[inline]
    pub const fn new(a: i32, b: i32) -> Self {
        Self { a, b }
    }

    /// The origin of the lattice.
    pub const ORIGIN: Self = Self::new(0, 0);

    /// Convert to Cartesian coordinates for **presentation only**.
    ///
    /// Returns (x, z) in studs:
    /// ```text
    /// x = (a - b/2) · s
    /// z = b · (√3/2) · s
    /// ```
    ///
    /// Never use this for agreement between agents — use the integer
    /// (a, b) pair directly.
    pub fn to_cartesian(&self) -> (f64, f64) {
        let a = self.a as f64;
        let b = self.b as f64;
        let x = (a - b * 0.5) * LATTICE_SCALE;
        let z = b * FRAC_SQRT_3_2 * LATTICE_SCALE;
        (x, z)
    }

    /// Euclidean distance to another coordinate, in Cartesian space.
    ///
    /// For **spatial queries only** (nearest-neighbor, proximity checks).
    /// Agreement logic uses integer lattice distance.
    pub fn distance_to(&self, other: &Self) -> f64 {
        let (x1, z1) = self.to_cartesian();
        let (x2, z2) = other.to_cartesian();
        let dx = x2 - x1;
        let dz = z2 - z1;
        (dx * dx + dz * dz).sqrt()
    }

    /// Snap floating-point Cartesian coordinates to the nearest A₂ lattice point.
    ///
    /// Algorithm: inverse-transform (x, z) to fractional (a, b), then
    /// round to the nearest integer pair. The result is always a valid
    /// lattice point, and snapping is idempotent.
    pub fn snap_to_lattice(x: f64, z: f64) -> Self {
        // Inverse transform:
        // x = (a - b/2) · s  ⟹  a = x/s + b/2
        // z = b · (√3/2) · s  ⟹  b = z / (s · √3/2)
        let b_frac = z / (LATTICE_SCALE * FRAC_SQRT_3_2);
        let a_frac = x / LATTICE_SCALE + b_frac * 0.5;

        // Round to nearest integers. This is correct for the A₂ lattice
        // because each fundamental cell contains exactly one lattice point.
        let b = b_frac.round() as i32;
        let a = a_frac.round() as i32;
        Self { a, b }
    }

    /// The six equidistant neighbors on the A₂ lattice.
    ///
    /// A₂ lattice neighbors: ±(1,0), ±(0,1), ±(1,1).
    /// All at Cartesian distance = lattice scale.
    pub const fn neighbors(&self) -> [Self; 6] {
        [
            Self::new(self.a + 1, self.b),
            Self::new(self.a - 1, self.b),
            Self::new(self.a, self.b + 1),
            Self::new(self.a, self.b - 1),
            Self::new(self.a + 1, self.b + 1),
            Self::new(self.a - 1, self.b - 1),
        ]
    }

    /// Lattice addition (vector addition in the Eisenstein ring).
    #[inline]
    pub const fn add(&self, other: &Self) -> Self {
        Self::new(self.a + other.a, self.b + other.b)
    }

    /// Lattice subtraction.
    #[inline]
    pub const fn sub(&self, other: &Self) -> Self {
        Self::new(self.a - other.a, self.b - other.b)
    }

    /// Whether this coordinate is the origin.
    #[inline]
    pub const fn is_origin(&self) -> bool {
        self.a == 0 && self.b == 0
    }
}

impl Default for EisensteinCoord {
    fn default() -> Self {
        Self::ORIGIN
    }
}

impl core::fmt::Display for EisensteinCoord {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({},{})", self.a, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_creation() {
        let c = EisensteinCoord::new(3, -2);
        assert_eq!(c.a, 3);
        assert_eq!(c.b, -2);
    }

    #[test]
    fn origin_is_zero() {
        let o = EisensteinCoord::ORIGIN;
        assert!(o.is_origin());
        assert_eq!(o.to_cartesian(), (0.0, 0.0));
    }

    #[test]
    fn cartesian_roundtrip() {
        // (3, -2) from the Grand Plan data flow section
        let coord = EisensteinCoord::new(3, -2);
        let (x, z) = coord.to_cartesian();
        // x = (3 - (-2)/2) · 4 = (3+1)·4 = 16
        // z = -2 · (√3/2) · 4 ≈ -6.928
        assert!((x - 16.0).abs() < 1e-9);
        assert!((z - (-6.928_203_230_275_509)).abs() < 1e-9);
    }

    #[test]
    fn snap_is_idempotent() {
        for &(a, b) in &[(0, 0), (1, 0), (0, 1), (3, -2), (-5, 7), (100, -100)] {
            let coord = EisensteinCoord::new(a, b);
            let (x, z) = coord.to_cartesian();
            let snapped = EisensteinCoord::snap_to_lattice(x, z);
            assert_eq!(snapped, coord, "snap idempotence failed for ({a},{b})");
        }
    }

    #[test]
    fn distance_to_neighbor() {
        let origin = EisensteinCoord::ORIGIN;
        let neighbor = EisensteinCoord::new(1, 0);
        // Distance to (1,0) in Cartesian: x=4, z=0 → distance = 4.0
        let d = origin.distance_to(&neighbor);
        assert!((d - 4.0).abs() < 1e-9);
    }

    #[test]
    fn neighbors_are_six() {
        let coord = EisensteinCoord::new(2, 3);
        let nbrs = coord.neighbors();
        assert_eq!(nbrs.len(), 6);
        // All neighbors are distinct
        for i in 0..6 {
            for j in (i + 1)..6 {
                assert_ne!(nbrs[i], nbrs[j], "duplicate neighbor at {i},{j}");
            }
        }
    }

    #[test]
    fn saturation() {
        assert_eq!(saturate_i8(0), 0);
        assert_eq!(saturate_i8(127), 127);
        assert_eq!(saturate_i8(128), 127);
        assert_eq!(saturate_i8(255), 127);
        assert_eq!(saturate_i8(-1), 0);
        assert_eq!(saturate_i8(-100), 0);
    }

    #[test]
    fn saturating_arithmetic() {
        assert_eq!(saturating_add(100, 50), 127);
        assert_eq!(saturating_add(100, 20), 120);
        assert_eq!(saturating_sub(50, 100), 0);
        assert_eq!(saturating_sub(100, 30), 70);
    }
}
