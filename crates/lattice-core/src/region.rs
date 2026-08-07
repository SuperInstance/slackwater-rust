//! Region/area operations on the lattice.
//!
//! A rectangular (in lattice coordinates) region for defining lots,
//! districts, build zones, and other bounded areas.

use crate::eisenstein::EisensteinPoint;
use std::cmp::{max, min};

/// A rectangular region on the lattice (for lots, districts, etc.).
///
/// Defined by min and max corners in Eisenstein `(a, b)` coordinates.
/// The region is inclusive of both corners.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LatticeRegion {
    /// The minimum corner (smallest a and b).
    pub min: EisensteinPoint,
    /// The maximum corner (largest a and b).
    pub max: EisensteinPoint,
}

impl LatticeRegion {
    /// Create a new region from two corners.
    ///
    /// The corners are automatically normalized so that `min` has the
    /// smaller coordinates regardless of argument order.
    #[inline]
    #[must_use]
    pub fn new(min: EisensteinPoint, max: EisensteinPoint) -> Self {
        Self {
            min: EisensteinPoint::new(min.a.min(max.a), min.b.min(max.b)),
            max: EisensteinPoint::new(min.a.max(max.a), min.b.max(max.b)),
        }
    }

    /// Create a region centered on a point with the given half-extent.
    #[inline]
    #[must_use]
    pub fn centered(center: EisensteinPoint, half_extent: i32) -> Self {
        Self::new(
            EisensteinPoint::new(center.a - half_extent, center.b - half_extent),
            EisensteinPoint::new(center.a + half_extent, center.b + half_extent),
        )
    }

    /// Check if a point is inside this region (inclusive bounds).
    #[inline]
    #[must_use]
    pub fn contains(&self, point: &EisensteinPoint) -> bool {
        point.a >= self.min.a
            && point.a <= self.max.a
            && point.b >= self.min.b
            && point.b <= self.max.b
    }

    /// Number of lattice points in this region.
    #[inline]
    #[must_use]
    pub fn area(&self) -> usize {
        let w = (self.max.a - self.min.a + 1) as usize;
        let h = (self.max.b - self.min.b + 1) as usize;
        w * h
    }

    /// Width (range of `a` coordinates, inclusive).
    #[inline]
    #[must_use]
    pub fn width(&self) -> i32 {
        self.max.a - self.min.a + 1
    }

    /// Height (range of `b` coordinates, inclusive).
    #[inline]
    #[must_use]
    pub fn height(&self) -> i32 {
        self.max.b - self.min.b + 1
    }

    /// Iterate over all lattice points in this region.
    ///
    /// Points are yielded in row-major order (b varies slowest).
    pub fn iter(&self) -> impl Iterator<Item = EisensteinPoint> + '_ {
        let (min_a, max_a) = (self.min.a, self.max.a);
        let (min_b, max_b) = (self.min.b, self.max.b);
        (min_b..=max_b).flat_map(move |b| (min_a..=max_a).map(move |a| EisensteinPoint::new(a, b)))
    }

    /// Expand the region by `by` units in all directions.
    #[inline]
    #[must_use]
    pub fn expand(&self, by: u32) -> Self {
        let by = by as i32;
        Self {
            min: EisensteinPoint::new(self.min.a - by, self.min.b - by),
            max: EisensteinPoint::new(self.max.a + by, self.max.b + by),
        }
    }

    /// Check if this region overlaps with another.
    #[inline]
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.a <= other.max.a
            && self.max.a >= other.min.a
            && self.min.b <= other.max.b
            && self.max.b >= other.min.b
    }

    /// Compute the intersection of two regions, if any.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if !self.intersects(other) {
            return None;
        }
        Some(Self::new(
            EisensteinPoint::new(max(self.min.a, other.min.a), max(self.min.b, other.min.b)),
            EisensteinPoint::new(min(self.max.a, other.max.a), min(self.max.b, other.max.b)),
        ))
    }

    /// Compute the bounding box of two regions.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self::new(
            EisensteinPoint::new(min(self.min.a, other.min.a), min(self.min.b, other.min.b)),
            EisensteinPoint::new(max(self.max.a, other.max.a), max(self.max.b, other.max.b)),
        )
    }
}

impl Default for LatticeRegion {
    fn default() -> Self {
        Self::new(EisensteinPoint::origin(), EisensteinPoint::origin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use EisensteinPoint as EP;

    #[test]
    fn region_contains() {
        let r = LatticeRegion::new(EP::new(-5, -5), EP::new(5, 5));
        assert!(r.contains(&EP::origin()));
        assert!(r.contains(&EP::new(5, 5)));
        assert!(r.contains(&EP::new(-5, -5)));
        assert!(!r.contains(&EP::new(6, 0)));
        assert!(!r.contains(&EP::new(0, -6)));
    }

    #[test]
    fn region_area() {
        let r = LatticeRegion::new(EP::new(0, 0), EP::new(3, 4));
        assert_eq!(r.area(), 20); // 4 * 5
    }

    #[test]
    fn region_iter() {
        let r = LatticeRegion::new(EP::new(0, 0), EP::new(2, 1));
        let points: Vec<_> = r.iter().collect();
        assert_eq!(points.len(), 6); // 3 * 2
        assert!(points.contains(&EP::new(0, 0)));
        assert!(points.contains(&EP::new(2, 1)));
        assert!(points.contains(&EP::new(1, 0)));
    }

    #[test]
    fn region_expand() {
        let r = LatticeRegion::new(EP::new(0, 0), EP::new(2, 2));
        let expanded = r.expand(1);
        assert_eq!(expanded.min, EP::new(-1, -1));
        assert_eq!(expanded.max, EP::new(3, 3));
    }

    #[test]
    fn region_normalizes_corners() {
        let r = LatticeRegion::new(EP::new(5, 5), EP::new(0, 0));
        assert_eq!(r.min, EP::new(0, 0));
        assert_eq!(r.max, EP::new(5, 5));
    }

    #[test]
    fn region_intersects() {
        let r1 = LatticeRegion::new(EP::new(0, 0), EP::new(3, 3));
        let r2 = LatticeRegion::new(EP::new(2, 2), EP::new(5, 5));
        let r3 = LatticeRegion::new(EP::new(10, 10), EP::new(15, 15));
        assert!(r1.intersects(&r2));
        assert!(!r1.intersects(&r3));
    }

    #[test]
    fn region_intersection() {
        let r1 = LatticeRegion::new(EP::new(0, 0), EP::new(3, 3));
        let r2 = LatticeRegion::new(EP::new(2, 2), EP::new(5, 5));
        let inter = r1.intersection(&r2).unwrap();
        assert_eq!(inter.min, EP::new(2, 2));
        assert_eq!(inter.max, EP::new(3, 3));
    }

    #[test]
    fn region_no_intersection() {
        let r1 = LatticeRegion::new(EP::new(0, 0), EP::new(1, 1));
        let r2 = LatticeRegion::new(EP::new(5, 5), EP::new(6, 6));
        assert!(r1.intersection(&r2).is_none());
    }
}
