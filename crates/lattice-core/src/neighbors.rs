//! Spatial queries on the lattice.
//!
//! Collision detection, nearest-free-point search, and boundary computation.
//! All queries use exact integer arithmetic — no floating-point tolerance bands.

use crate::eisenstein::EisensteinPoint;
use std::collections::HashSet;

/// Given a set of occupied lattice points, find the nearest unoccupied point.
///
/// Searches outward in hex-distance rings from `target`. If `target` itself
/// is free, returns it immediately. Returns `target` as a fallback if nothing
/// is found within the default search radius of 20.
#[must_use]
pub fn nearest_unoccupied(occupied: &[EisensteinPoint], target: &EisensteinPoint) -> EisensteinPoint {
    let occupied_set: HashSet<&EisensteinPoint> = occupied.iter().collect();

    if !occupied_set.contains(target) {
        return *target;
    }

    // Search outward in rings of increasing radius.
    for radius in 1..=20 {
        for candidate in target.within(radius) {
            if !occupied_set.contains(&candidate) {
                return candidate;
            }
        }
    }

    // Fallback: return target even though it's occupied.
    *target
}

/// Find all occupied points within a radius (hex distance) of center.
#[must_use]
pub fn occupied_in_radius(
    occupied: &[EisensteinPoint],
    center: &EisensteinPoint,
    radius: u32,
) -> Vec<EisensteinPoint> {
    occupied
        .iter()
        .copied()
        .filter(|p| p.lattice_distance(center) <= radius)
        .collect()
}

/// Check if a placement collides with existing structures.
///
/// Returns `true` if any occupied point is within `min_distance`
/// (hex steps) of `new_placement`.
#[must_use]
pub fn collides(new_placement: &EisensteinPoint, occupied: &[EisensteinPoint], min_distance: u32) -> bool {
    occupied
        .iter()
        .any(|p| p.lattice_distance(new_placement) <= min_distance)
}

/// Find the boundary of a build: occupied points with at least one free neighbor.
///
/// These are the "frontier" points — useful for determining where a build
/// can expand. Uses the neighbor set of each occupied point to check if
/// any neighbor is unoccupied.
#[must_use]
pub fn build_boundary(occupied: &[EisensteinPoint]) -> Vec<EisensteinPoint> {
    let occupied_set: HashSet<&EisensteinPoint> = occupied.iter().collect();

    occupied
        .iter()
        .copied()
        .filter(|p| {
            p.neighbors()
                .iter()
                .any(|n| !occupied_set.contains(n))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use EisensteinPoint as EP;

    #[test]
    fn nearest_unoccupied_returns_target_if_free() {
        let occupied = vec![EP::new(1, 0), EP::new(0, 1)];
        let target = EP::new(5, 5);
        assert_eq!(nearest_unoccupied(&occupied, &target), target);
    }

    #[test]
    fn nearest_unoccupied_finds_neighbor() {
        let occupied = vec![EP::origin()];
        let target = EP::origin();
        let result = nearest_unoccupied(&occupied, &target);
        // Should be one of the 6 neighbors of origin
        assert!(result.lattice_distance(&target) == 1);
        assert!(!occupied.contains(&result));
    }

    #[test]
    fn occupied_in_radius_basic() {
        let occupied = vec![
            EP::origin(),
            EP::new(1, 0),
            EP::new(5, 5),
        ];
        let result = occupied_in_radius(&occupied, &EP::origin(), 1);
        assert!(result.contains(&EP::origin()));
        assert!(result.contains(&EP::new(1, 0)));
        assert!(!result.contains(&EP::new(5, 5)));
    }

    #[test]
    fn collides_detects_overlap() {
        let occupied = vec![EP::origin(), EP::new(1, 0)];
        assert!(collides(&EP::origin(), &occupied, 0));
        assert!(collides(&EP::new(0, 1), &occupied, 1));
        assert!(!collides(&EP::new(10, 10), &occupied, 1));
    }

    #[test]
    fn build_boundary_finds_frontier() {
        // Three in a row: (0,0), (1,0), (2,0)
        let occupied = vec![EP::origin(), EP::new(1, 0), EP::new(2, 0)];
        let boundary = build_boundary(&occupied);
        // All three have free neighbors — all are on the boundary
        assert_eq!(boundary.len(), 3);
    }

    #[test]
    fn build_boundary_excludes_interior() {
        // A solid hexagonal block: center + 6 neighbors
        let center = EP::origin();
        let mut occupied: Vec<EP> = center.neighbors().to_vec();
        occupied.push(center);

        let boundary = build_boundary(&occupied);
        // Center is fully surrounded — not on boundary
        assert!(!boundary.contains(&center));
        // All 6 neighbors have free neighbors outside the block
        assert_eq!(boundary.len(), 6);
    }
}
