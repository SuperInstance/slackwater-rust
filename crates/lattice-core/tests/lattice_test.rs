//! Comprehensive tests for lattice-core.
//!
//! Covers:
//! - EisensteinPoint: creation, Cartesian roundtrip, lattice distance
//! - Neighbors: all 6 neighbors are distance 1
//! - Rotation: rotate_60 applied 6 times = identity
//! - Snapping: from_cartesian followed by to_cartesian within tolerance
//! - Snapping idempotent: snap(snap(x,y)) = snap(x,y)
//! - Collision detection
//! - Region containment and iteration
//! - nearest_unoccupied finds correct point
//! - Property: any Cartesian point snaps to a lattice point within max distance

#![warn(clippy::all)]
#![deny(unsafe_code)]

use EisensteinPoint as EP;
use approx::assert_relative_eq;
use lattice_core::{
    EisensteinPoint, LatticeRegion, SnappedPlacement, build_boundary, collides, nearest_unoccupied,
    occupied_in_radius, snap_all, snap_height, snap_position, snap_rotation, snap_rotation_index,
};

const SCALE: f64 = 4.0;
const TOLERANCE: f64 = 0.6; // max distance to nearest lattice point is scale/√3 ≈ 0.577

// ── Creation ───────────────────────────────────────────

#[test]
fn test_creation() {
    let p = EP::new(3, -7);
    assert_eq!(p.a, 3);
    assert_eq!(p.b, -7);

    let o = EP::origin();
    assert_eq!(o.a, 0);
    assert_eq!(o.b, 0);
    assert!(o.is_zero());
    assert!(!p.is_zero());
}

#[test]
fn test_default_is_origin() {
    let p: EP = Default::default();
    assert_eq!(p, EP::origin());
}

#[test]
fn test_display() {
    assert_eq!(format!("{}", EP::new(5, 0)), "E(5)");
    assert_eq!(format!("{}", EP::new(0, 3)), "E(3ω)");
    assert_eq!(format!("{}", EP::new(3, 2)), "E(3 + 2ω)");
    assert_eq!(format!("{}", EP::new(3, -2)), "E(3 - 2ω)");
    assert_eq!(format!("{}", EP::origin()), "E(0)");
}

// ── Cartesian roundtrip ────────────────────────────────

#[test]
fn test_cartesian_origin() {
    let (x, y) = EP::origin().to_cartesian(SCALE);
    assert_relative_eq!(x, 0.0, epsilon = 1e-10);
    assert_relative_eq!(y, 0.0, epsilon = 1e-10);
}

#[test]
fn test_cartesian_roundtrip() {
    // For points within a few lattice units of origin, snapping back should be exact.
    for a in -5..=5 {
        for b in -5..=5 {
            let p = EP::new(a, b);
            let (x, y) = p.to_cartesian(SCALE);
            let snapped = EP::from_cartesian(x, y, SCALE);
            assert_eq!(
                snapped, p,
                "roundtrip failed for ({}, {}): got ({}, {})",
                a, b, snapped.a, snapped.b
            );
        }
    }
}

#[test]
fn test_cartesian_known_values() {
    // (1, 0) → x=4, y=0 at scale 4
    let (x, y) = EP::new(1, 0).to_cartesian(SCALE);
    assert_relative_eq!(x, 4.0);
    assert_relative_eq!(y, 0.0);

    // (0, 1) → x=-2, y=2√3 at scale 4
    let (x, y) = EP::new(0, 1).to_cartesian(SCALE);
    assert_relative_eq!(x, -2.0);
    assert_relative_eq!(y, 4.0 * 1.7320508075688772f64 / 2.0);

    // (1, 1) → x=2, y=2√3 at scale 4
    let (x, y) = EP::new(1, 1).to_cartesian(SCALE);
    assert_relative_eq!(x, 2.0);
    assert_relative_eq!(y, 4.0 * 1.7320508075688772f64 / 2.0);
}

// ── Lattice distance ───────────────────────────────────

#[test]
fn test_lattice_distance() {
    assert_eq!(EP::origin().lattice_distance(&EP::origin()), 0);
    assert_eq!(EP::origin().lattice_distance(&EP::new(1, 0)), 1);
    assert_eq!(EP::origin().lattice_distance(&EP::new(0, 1)), 1);
    assert_eq!(EP::origin().lattice_distance(&EP::new(1, 1)), 1);
    assert_eq!(EP::origin().lattice_distance(&EP::new(-1, 0)), 1);
    assert_eq!(EP::origin().lattice_distance(&EP::new(0, -1)), 1);
    assert_eq!(EP::origin().lattice_distance(&EP::new(-1, -1)), 1);

    // Same sign: max(|da|, |db|)
    assert_eq!(EP::origin().lattice_distance(&EP::new(3, 2)), 3);
    assert_eq!(EP::origin().lattice_distance(&EP::new(2, 3)), 3);

    // Opposite signs: |da| + |db|
    assert_eq!(EP::origin().lattice_distance(&EP::new(3, -2)), 5);
    assert_eq!(EP::origin().lattice_distance(&EP::new(-3, 2)), 5);
}

#[test]
fn test_norm() {
    assert_eq!(EP::origin().norm(), 0);
    assert_eq!(EP::new(1, 0).norm(), 1);
    assert_eq!(EP::new(0, 1).norm(), 1);
    assert_eq!(EP::new(1, 1).norm(), 1);
    assert_eq!(EP::new(2, 0).norm(), 4);
    assert_eq!(EP::new(1, -1).norm(), 3); // 1 - (-1) + 1 = 3
}

// ── Neighbors ──────────────────────────────────────────

#[test]
fn test_neighbors_count() {
    let neighbors = EP::origin().neighbors();
    assert_eq!(neighbors.len(), 6);
}

#[test]
fn test_neighbors_are_distance_1() {
    let center = EP::new(5, 5);
    for n in center.neighbors() {
        assert_eq!(
            center.lattice_distance(&n),
            1,
            "neighbor ({}, {}) is not distance 1",
            n.a,
            n.b
        );
    }
}

#[test]
fn test_neighbors_are_unique() {
    let neighbors = EP::origin().neighbors();
    let mut seen = std::collections::HashSet::new();
    for n in &neighbors {
        assert!(seen.insert(*n), "duplicate neighbor ({}, {})", n.a, n.b);
    }
}

#[test]
fn test_neighbors_known_values() {
    let neighbors = EP::origin().neighbors();
    let expected = [
        EP::new(1, 0),
        EP::new(-1, 0),
        EP::new(0, 1),
        EP::new(0, -1),
        EP::new(1, 1),
        EP::new(-1, -1),
    ];
    for e in &expected {
        assert!(neighbors.contains(e), "missing neighbor ({}, {})", e.a, e.b);
    }
}

// ── Rotation ───────────────────────────────────────────

#[test]
fn test_rotate_60_six_times_is_identity() {
    let p = EP::new(3, 7);
    let r1 = p.rotate_60();
    let r2 = r1.rotate_60();
    let r3 = r2.rotate_60();
    let r4 = r3.rotate_60();
    let r5 = r4.rotate_60();
    let r6 = r5.rotate_60();
    assert_eq!(r6, p);
}

#[test]
fn test_rotate_120_three_times_is_identity() {
    let p = EP::new(3, 7);
    let r1 = p.rotate_120();
    let r2 = r1.rotate_120();
    let r3 = r2.rotate_120();
    assert_eq!(r3, p);
}

#[test]
fn test_rotate_180_twice_is_identity() {
    let p = EP::new(3, 7);
    let r1 = p.rotate_180();
    let r2 = r1.rotate_180();
    assert_eq!(r2, p);
}

#[test]
fn test_rotate_60_specific() {
    // 60° CCW rotation: (a, b) → (a-b, a)
    let p = EP::new(2, 3);
    let expected = EP::new(-1, 2); // a-b=2-3=-1, a=2
    assert_eq!(p.rotate_60(), expected);
}

#[test]
fn test_rotate_180_is_negation() {
    let p = EP::new(5, -3);
    assert_eq!(p.rotate_180(), EP::new(-5, 3));
}

// ── Snapping ───────────────────────────────────────────

#[test]
fn test_snap_idempotent() {
    for x in -20..=20 {
        for y in -20..=20 {
            let xf = x as f64 * 0.7;
            let yf = y as f64 * 0.7;
            let s1 = snap_position(xf, yf);
            let (cx, cy) = s1.to_cartesian(SCALE);
            let s2 = snap_position(cx, cy);
            assert_eq!(s1, s2, "snap not idempotent for ({}, {})", xf, yf);
        }
    }
}

#[test]
fn test_snap_within_tolerance() {
    // Any Cartesian point should snap to a lattice point whose Cartesian
    // position is within half a cell (TOLERANCE * scale) of the original.
    let max_err = TOLERANCE * SCALE; // 2.0 studs
    for i in -50..50 {
        for j in -50..50 {
            // Offset to avoid exact lattice points
            let x = i as f64 * 0.37 + 0.13;
            let y = j as f64 * 0.41 + 0.07;
            let snapped = EP::from_cartesian(x, y, SCALE);
            let (sx, sy) = snapped.to_cartesian(SCALE);
            let dist = ((sx - x).powi(2) + (sy - y).powi(2)).sqrt();
            assert!(
                dist < max_err,
                "snap error {} too large for ({}, {}) → ({}, {})",
                dist,
                x,
                y,
                sx,
                sy
            );
        }
    }
}

#[test]
fn test_snap_rotation_all_increments() {
    for deg in 0..360 {
        let snapped = snap_rotation(deg as f64);
        assert!(
            [0, 60, 120, 180, 240, 300].contains(&snapped),
            "rotation {} snapped to {}",
            deg,
            snapped
        );
    }
}

#[test]
fn test_snap_rotation_negative() {
    assert_eq!(snap_rotation(-30.0), 300);
    assert_eq!(snap_rotation(-60.0), 300);
    assert_eq!(snap_rotation(-90.0), 240);
    assert_eq!(snap_rotation(-180.0), 180);
}

#[test]
fn test_snap_height_grid() {
    assert_eq!(snap_height(0.0, 1.0), 0);
    assert_eq!(snap_height(0.49, 1.0), 0);
    assert_eq!(snap_height(0.51, 1.0), 1);
    assert_eq!(snap_height(-0.51, 1.0), -1);
    assert_eq!(snap_height(5.0, 2.0), 3); // 5/2 = 2.5 → rounds to 3
}

#[test]
fn test_snap_all() {
    let placement = snap_all(8.0, 0.0, 3.3, 45.0, 1.0);
    assert_eq!(placement.lattice, EP::from_cartesian(8.0, 0.0, SCALE));
    assert_eq!(placement.height, 3);
    assert_eq!(placement.rotation, 1); // 45° → 60°
    assert_eq!(placement.rotation_degrees(), 60);
}

// ── Collision detection ────────────────────────────────

#[test]
fn test_collides_direct() {
    let occupied = vec![EP::origin(), EP::new(1, 0)];
    assert!(collides(&EP::origin(), &occupied, 0));
    assert!(collides(&EP::new(1, 0), &occupied, 0));
    assert!(!collides(&EP::new(0, 1), &occupied, 0));
}

#[test]
fn test_collides_with_min_distance() {
    let occupied = vec![EP::origin()];
    assert!(collides(&EP::new(1, 0), &occupied, 1));
    assert!(collides(&EP::new(0, 1), &occupied, 1));
    assert!(!collides(&EP::new(2, 0), &occupied, 1));
}

#[test]
fn test_occupied_in_radius() {
    let occupied = vec![EP::origin(), EP::new(1, 0), EP::new(2, 0), EP::new(10, 10)];
    let result = occupied_in_radius(&occupied, &EP::origin(), 2);
    assert_eq!(result.len(), 3);
    assert!(result.contains(&EP::origin()));
    assert!(result.contains(&EP::new(1, 0)));
    assert!(result.contains(&EP::new(2, 0)));
}

// ── Nearest unoccupied ─────────────────────────────────

#[test]
fn test_nearest_unoccupied_target_free() {
    let occupied = vec![EP::new(1, 0)];
    let target = EP::new(5, 5);
    assert_eq!(nearest_unoccupied(&occupied, &target), target);
}

#[test]
fn test_nearest_unoccupied_target_occupied() {
    let occupied = vec![EP::origin()];
    let target = EP::origin();
    let result = nearest_unoccupied(&occupied, &target);
    // Should be a distance-1 neighbor
    assert_eq!(target.lattice_distance(&result), 1);
    assert!(!occupied.contains(&result));
}

#[test]
fn test_nearest_unoccupied_finds_among_many() {
    // Occupy all 6 neighbors of origin but leave origin itself free.
    let occupied: Vec<EP> = EP::origin().neighbors().to_vec();
    let target = EP::origin();

    // Origin is free, so it should be returned directly.
    let result = nearest_unoccupied(&occupied, &target);
    assert_eq!(result, EP::origin());

    // Now occupy origin too — the nearest free point should be at distance 2.
    let mut occupied_with_center = occupied.clone();
    occupied_with_center.push(EP::origin());
    let result2 = nearest_unoccupied(&occupied_with_center, &target);
    let d = target.lattice_distance(&result2);
    assert_eq!(
        d, 2,
        "expected distance 2 when origin and all neighbors occupied"
    );
    assert!(!occupied_with_center.contains(&result2));
}

// ── Build boundary ─────────────────────────────────────

#[test]
fn test_build_boundary_single_point() {
    let occupied = vec![EP::origin()];
    let boundary = build_boundary(&occupied);
    assert_eq!(boundary.len(), 1);
    assert!(boundary.contains(&EP::origin()));
}

#[test]
fn test_build_boundary_hex_cluster() {
    // Center + 6 neighbors = solid hex
    let mut occupied: Vec<EP> = EP::origin().neighbors().to_vec();
    occupied.push(EP::origin());

    let boundary = build_boundary(&occupied);
    // Only the 6 outer points are boundary; center is interior
    assert_eq!(boundary.len(), 6);
    assert!(!boundary.contains(&EP::origin()));
}

// ── Region ─────────────────────────────────────────────

#[test]
fn test_region_contains() {
    let r = LatticeRegion::new(EP::new(-3, -3), EP::new(3, 3));
    assert!(r.contains(&EP::origin()));
    assert!(r.contains(&EP::new(3, 3)));
    assert!(r.contains(&EP::new(-3, -3)));
    assert!(!r.contains(&EP::new(4, 0)));
    assert!(!r.contains(&EP::new(0, 4)));
}

#[test]
fn test_region_area() {
    let r = LatticeRegion::new(EP::new(0, 0), EP::new(4, 4));
    assert_eq!(r.area(), 25);

    let r2 = LatticeRegion::new(EP::new(-1, -1), EP::new(1, 1));
    assert_eq!(r2.area(), 9);
}

#[test]
fn test_region_iter() {
    let r = LatticeRegion::new(EP::new(0, 0), EP::new(2, 1));
    let points: Vec<_> = r.iter().collect();
    assert_eq!(points.len(), 6); // 3 × 2
    assert!(points.contains(&EP::new(0, 0)));
    assert!(points.contains(&EP::new(2, 1)));
    assert!(points.contains(&EP::new(1, 0)));
    assert!(points.contains(&EP::new(2, 0)));
}

#[test]
fn test_region_expand() {
    let r = LatticeRegion::new(EP::new(0, 0), EP::new(2, 2));
    let e = r.expand(1);
    assert_eq!(e.min, EP::new(-1, -1));
    assert_eq!(e.max, EP::new(3, 3));
    assert!(e.contains(&EP::new(-1, -1)));
    assert!(e.contains(&EP::new(3, 3)));
}

#[test]
fn test_region_normalizes_corners() {
    let r = LatticeRegion::new(EP::new(5, 5), EP::new(0, 0));
    assert_eq!(r.min, EP::new(0, 0));
    assert_eq!(r.max, EP::new(5, 5));
}

// ── Arithmetic ─────────────────────────────────────────

#[test]
fn test_add_sub() {
    let p = EP::new(3, 4);
    let q = EP::new(1, 2);
    assert_eq!(p.add(&q), EP::new(4, 6));
    assert_eq!(p.sub(&q), EP::new(2, 2));
    assert_eq!(p + q, EP::new(4, 6));
    assert_eq!(p - q, EP::new(2, 2));
}

#[test]
fn test_neg() {
    let p = EP::new(3, -4);
    assert_eq!(-p, EP::new(-3, 4));
    assert_eq!(p.neg(), EP::new(-3, 4));
}

#[test]
fn test_conjugate() {
    let p = EP::new(3, 4);
    let c = p.conjugate();
    assert_eq!(c, EP::new(-1, -4)); // (a-b, -b) = (3-4, -4)
}

// ── Ordering ───────────────────────────────────────────

#[test]
fn test_ordering_by_norm() {
    let origin = EP::origin();
    let unit = EP::new(1, 0);
    let far = EP::new(5, 5);
    assert!(origin < unit);
    assert!(unit < far);
}

#[test]
fn test_ordering_tiebreak() {
    // Same norm (both 1), tiebreak by a then b
    let p1 = EP::new(0, -1); // norm = 0 - 0 + 1 = 1
    let p2 = EP::new(0, 1); // norm = 0 - 0 + 1 = 1
    assert!(p1 < p2); // same a=0, -1 < 1
}

// ── Within radius ──────────────────────────────────────

#[test]
fn test_within_radius_1() {
    let result = EP::origin().within(1);
    assert_eq!(result.len(), 6); // exactly the 6 neighbors
    for p in &result {
        assert_eq!(EP::origin().lattice_distance(p), 1);
    }
}

#[test]
fn test_within_radius_0() {
    let result = EP::origin().within(0);
    assert!(result.is_empty());
}

#[test]
fn test_within_radius_2_count() {
    let result = EP::origin().within(2);
    // Ring 1: 6 points, Ring 2: 12 points = 18 total
    assert_eq!(result.len(), 18);
}

// ── Property-style tests ───────────────────────────────

#[test]
fn test_lattice_distance_symmetric() {
    let p = EP::new(3, -5);
    let q = EP::new(-2, 7);
    assert_eq!(p.lattice_distance(&q), q.lattice_distance(&p));
}

#[test]
fn test_euclidean_distance_symmetric() {
    let p = EP::new(3, -5);
    let q = EP::new(-2, 7);
    let d1 = p.euclidean_distance(&q);
    let d2 = q.euclidean_distance(&p);
    assert_relative_eq!(d1, d2, epsilon = 1e-10);
}

#[test]
fn test_euclidean_distance_origin() {
    let p = EP::new(1, 0);
    assert_relative_eq!(p.euclidean_distance(&EP::origin()), 1.0);
    let p2 = EP::new(0, 1);
    assert_relative_eq!(p2.euclidean_distance(&EP::origin()), 1.0);
}

#[test]
fn test_is_unit() {
    assert!(EP::new(1, 0).is_unit());
    assert!(EP::new(-1, 0).is_unit());
    assert!(EP::new(0, 1).is_unit());
    assert!(EP::new(0, -1).is_unit());
    assert!(EP::new(1, 1).is_unit());
    assert!(EP::new(-1, -1).is_unit());
    assert!(!EP::new(2, 0).is_unit());
    assert!(!EP::origin().is_unit());
}

#[test]
fn test_all_rotations_cover_hexagon() {
    // Applying rotate_60 six times visits 6 distinct points
    let p = EP::new(2, 3);
    let mut current = p;
    let mut points = vec![current];
    for _ in 0..5 {
        current = current.rotate_60();
        points.push(current);
    }
    // All 6 should be distinct
    let unique: std::collections::HashSet<_> = points.iter().collect();
    assert_eq!(unique.len(), 6, "rotations should visit 6 distinct points");
    // 7th rotation returns to start
    assert_eq!(current.rotate_60(), p);
}

#[test]
fn test_scale_consistency() {
    // Snapping at scale S then converting back at scale S should be exact.
    let scale = 6.0;
    for a in -3..=3 {
        for b in -3..=3 {
            let p = EP::new(a, b);
            let (x, y) = p.to_cartesian(scale);
            let back = EP::from_cartesian(x, y, scale);
            assert_eq!(back, p);
        }
    }
}

#[test]
fn test_snap_rotation_index_range() {
    for deg in -360..=360 {
        let idx = snap_rotation_index(deg as f64);
        assert!(
            idx >= 0 && idx <= 5,
            "rotation index {} out of range for {}",
            idx,
            deg
        );
    }
}

#[test]
fn test_snapped_placement_serde() {
    let p = SnappedPlacement::new(EP::new(3, 4), 5, 2);
    let json = serde_json::to_string(&p).unwrap();
    let back: SnappedPlacement = serde_json::from_str(&json).unwrap();
    assert_eq!(p, back);
}

#[test]
fn test_eisenstein_serde() {
    let p = EP::new(7, -13);
    let json = serde_json::to_string(&p).unwrap();
    let back: EP = serde_json::from_str(&json).unwrap();
    assert_eq!(p, back);
}
