//! Lattice snapping — continuous space → exact lattice.
//!
//! Every placed part snaps to a lattice point. The lattice guarantees:
//!   - Minimum spacing between parts
//!   - No floating-point misalignment
//!   - Isotropic neighborhoods (no privileged direction)
//!   - Exact collision detection (integer arithmetic)

use crate::eisenstein::EisensteinPoint;

/// Default lattice scale: 4 studs per lattice unit.
pub const DEFAULT_SCALE: f64 = 4.0;

/// Default vertical grid size: 1 stud per height step.
pub const DEFAULT_HEIGHT_GRID: f64 = 1.0;

/// Snap a build position to the nearest A₂ lattice point.
///
/// This replaces floating-point coordinate agreement with exact integer agreement.
/// Two agents that snap the same `(x, y)` will always agree on the lattice point.
#[inline]
#[must_use]
pub fn snap_position(x: f64, y: f64) -> EisensteinPoint {
    EisensteinPoint::from_cartesian(x, y, DEFAULT_SCALE)
}

/// Snap a build position at a custom scale.
#[inline]
#[must_use]
pub fn snap_position_scaled(x: f64, y: f64, scale: f64) -> EisensteinPoint {
    EisensteinPoint::from_cartesian(x, y, scale)
}

/// Snap a rotation to the nearest 60° increment.
///
/// Returns an integer in `{0, 60, 120, 180, 240, 300}`.
/// The hexagonal lattice has 6-fold symmetry, so rotations quantize cleanly.
#[inline]
#[must_use]
pub fn snap_rotation(degrees: f64) -> i32 {
    let snapped = (degrees / 60.0).round() as i32;
    ((snapped % 6) + 6) % 6 * 60
}

/// Snap a rotation to a 60° increment index (0–5).
///
/// Returns `{0, 1, 2, 3, 4, 5}` representing multiples of 60°.
/// More compact than [`snap_rotation`] for storage and comparison.
#[inline]
#[must_use]
pub fn snap_rotation_index(degrees: f64) -> i32 {
    let snapped = (degrees / 60.0).round() as i32;
    ((snapped % 6) + 6) % 6
}

/// Snap a Y-coordinate (height) to the vertical grid.
///
/// Vertical is NOT on the Eisenstein lattice — it's a regular 1D grid.
/// Terrain is irregular; the lattice governs horizontal agreement, not height.
#[inline]
#[must_use]
pub fn snap_height(y: f64, grid_size: f64) -> i32 {
    (y / grid_size).round() as i32
}

/// A fully snapped placement: position + rotation + height.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SnappedPlacement {
    /// Lattice position (exact integer coordinates).
    pub lattice: EisensteinPoint,
    /// Height grid index.
    pub height: i32,
    /// Rotation index 0–5 (multiples of 60°).
    pub rotation: i32,
}

impl SnappedPlacement {
    /// Create a new snapped placement.
    #[inline]
    #[must_use]
    pub const fn new(lattice: EisensteinPoint, height: i32, rotation: i32) -> Self {
        Self {
            lattice,
            height,
            rotation,
        }
    }

    /// The rotation in degrees (0, 60, 120, 180, 240, 300).
    #[inline]
    #[must_use]
    pub const fn rotation_degrees(self) -> i32 {
        self.rotation * 60
    }
}

/// Full snap: position + rotation + height in one call.
///
/// Converts continuous `(x, y, z, rotation)` into a fully discrete placement.
/// Uses default lattice scale (4 studs) and height grid (1 stud).
#[inline]
#[must_use]
pub fn snap_all(x: f64, y: f64, z: f64, rot_degrees: f64, height_grid: f64) -> SnappedPlacement {
    SnappedPlacement {
        lattice: snap_position(x, y),
        height: snap_height(z, height_grid),
        rotation: snap_rotation_index(rot_degrees),
    }
}

/// Full snap with custom scale.
#[inline]
#[must_use]
pub fn snap_all_scaled(
    x: f64,
    y: f64,
    z: f64,
    rot_degrees: f64,
    scale: f64,
    height_grid: f64,
) -> SnappedPlacement {
    SnappedPlacement {
        lattice: snap_position_scaled(x, y, scale),
        height: snap_height(z, height_grid),
        rotation: snap_rotation_index(rot_degrees),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_rotation_basic() {
        assert_eq!(snap_rotation(0.0), 0);
        assert_eq!(snap_rotation(59.0), 60);
        assert_eq!(snap_rotation(30.0), 60);
        assert_eq!(snap_rotation(29.0), 0);
        assert_eq!(snap_rotation(90.0), 120);
        assert_eq!(snap_rotation(180.0), 180);
        assert_eq!(snap_rotation(359.0), 0);
        assert_eq!(snap_rotation(-60.0), 300);
    }

    #[test]
    fn snap_rotation_index_basic() {
        assert_eq!(snap_rotation_index(0.0), 0);
        assert_eq!(snap_rotation_index(59.0), 1);
        assert_eq!(snap_rotation_index(180.0), 3);
        assert_eq!(snap_rotation_index(300.0), 5);
    }

    #[test]
    fn snap_height_basic() {
        assert_eq!(snap_height(0.0, 1.0), 0);
        assert_eq!(snap_height(0.4, 1.0), 0);
        assert_eq!(snap_height(0.6, 1.0), 1);
        assert_eq!(snap_height(2.5, 1.0), 3); // round half away from zero
        assert_eq!(snap_height(10.0, 2.0), 5);
    }

    #[test]
    fn snap_all_roundtrip() {
        let placement = snap_all(8.0, 6.928, 5.5, 45.0, 1.0);
        assert_eq!(placement.rotation, 1); // 45° → 60°
        assert_eq!(placement.height, 6);
    }
}
