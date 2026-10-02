//! Iff-consistency audit (SuperInstance fleet task 69-c) — the hex_distance ↔
//! neighbors property suite of `slackwater-lattice` (Python, PR #1, merged as
//! 232f49e) translated to the Rust twin's `lattice-core`.
//!
//! The convention trap: this workspace's A₂ neighbor set is
//! `{(±1,0), (0,±1), ±(1,1)}` — the units ±1, ±ω, ±(1+ω) of ℤ[ω]
//! (`NEIGHBOR_DIRECTIONS`, eisenstein.rs). The textbook axial distance
//! `max(|da|, |db|, |da+db|)` assumes the OTHER axial neighbor set
//! `{(±1,0), (0,±1), (1,−1), (−1,1)}`; misapplied here it scores every
//! ±(1+ω) step — which IS a neighbor step — as 2, and scores the
//! (1,−1)/(−1,1) diagonal — which is NOT a neighbor step — as 1.
//! The published PyPI 0.1.0 wheel of the Python twin shipped exactly that
//! defect: 192 iff-violations over the 3,721 ordered pairs of the radius-4
//! ball. `lattice-core`'s `hex_distance_raw` uses the sign-split formula
//! (same-sign → `max(|da|, |db|)`, opposite-sign → `|da| + |db|`), which this
//! suite pins to the neighbor set by exhaustive iff.
//!
//! Properties (mirroring the Python suite):
//!   P1  lattice_distance(a, b) == 1  IFF  b ∈ neighbors(a)
//!       (both directions, every ordered pair of the radius-4 ball:
//!       61 points, 3,721 pairs)
//!   P2  lattice_distance(a, b) == lattice_distance(b, a)   (symmetry)
//!   P5  the published PyPI 0.1.0 formula is vendored and pinned: it must
//!       violate P1 on exactly 192 of the 3,721 pairs, so the defect stays
//!       demonstrable and can never silently return.

#![warn(clippy::all)]
#![deny(unsafe_code)]

use lattice_core::EisensteinPoint;

const RADIUS: u32 = 4;

/// All lattice points within `radius` hex steps of the origin.
fn ball(radius: u32) -> Vec<EisensteinPoint> {
    let origin = EisensteinPoint::origin();
    let r = radius as i32;
    let mut out = Vec::new();
    for da in -r..=r {
        for db in -r..=r {
            let p = EisensteinPoint::new(da, db);
            if origin.lattice_distance(&p) <= radius {
                out.push(p);
            }
        }
    }
    out
}

/// The published PyPI 0.1.0 formula, vendored verbatim from the Python twin's
/// slackwater-lattice commit 5bff9a3 ("📦 Published to PyPI + cleanup") so the
/// defect stays pinned and demonstrable without network access to PyPI.
fn pypi_0_1_0_hex_distance(a: EisensteinPoint, b: EisensteinPoint) -> u32 {
    let da = a.a - b.a;
    let db = a.b - b.b;
    da.unsigned_abs()
        .max(db.unsigned_abs())
        .max((da + db).unsigned_abs())
}

#[test]
fn p1_dist_one_iff_neighbor_all_pairs() {
    let pts = ball(RADIUS);
    assert_eq!(pts.len(), 61, "radius-4 ball must hold 61 points");
    let pairs: usize = pts.len() * pts.len();
    assert_eq!(pairs, 3_721, "ordered-pair count");
    let mut violations: Vec<(EisensteinPoint, EisensteinPoint)> = Vec::new();
    for a in &pts {
        for b in &pts {
            let d1 = a.lattice_distance(b) == 1;
            let nb = a.neighbors().contains(b);
            if d1 != nb {
                violations.push((*a, *b));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "{} ordered pairs violate dist==1 iff neighbor; first: {:?}",
        violations.len(),
        violations.first()
    );
}

#[test]
fn p1a_every_neighbor_is_at_distance_one() {
    for a in ball(RADIUS) {
        for n in a.neighbors() {
            assert_eq!(
                a.lattice_distance(&n),
                1,
                "neighbor {n:?} of {a:?} is not at distance 1"
            );
        }
    }
}

#[test]
fn p1b_every_distance_one_pair_is_a_neighbor() {
    for a in ball(RADIUS) {
        for b in ball(RADIUS) {
            if a.lattice_distance(&b) == 1 {
                assert!(
                    a.neighbors().contains(&b),
                    "{b:?} is at distance 1 from {a:?} but is not a neighbor"
                );
            }
        }
    }
}

#[test]
fn p2_symmetry_all_pairs() {
    let pts = ball(RADIUS);
    let mut asymmetric: Vec<(EisensteinPoint, EisensteinPoint)> = Vec::new();
    for a in &pts {
        for b in &pts {
            if a.lattice_distance(b) != b.lattice_distance(a) {
                asymmetric.push((*a, *b));
            }
        }
    }
    assert!(
        asymmetric.is_empty(),
        "{} asymmetric pairs; first: {:?}",
        asymmetric.len(),
        asymmetric.first()
    );
}

#[test]
fn p5_witnesses_pin_the_convention_on_both_formulas() {
    let o = EisensteinPoint::origin();
    let diag = EisensteinPoint::new(1, 1); // the ±(1+ω) step — a TRUE neighbor
    let anti = EisensteinPoint::new(1, -1); // NOT a neighbor step
    assert!(o.neighbors().contains(&diag), "(1,1) must be a neighbor");
    assert!(
        !o.neighbors().contains(&anti),
        "(1,-1) must not be a neighbor"
    );
    assert_eq!(
        o.lattice_distance(&diag),
        1,
        "current formula: (1,1) at distance 1"
    );
    assert_eq!(
        o.lattice_distance(&anti),
        2,
        "current formula: (1,-1) at distance 2"
    );
    // The published 0.1.0 formula scores exactly the wrong way on both:
    assert_eq!(
        pypi_0_1_0_hex_distance(o, diag),
        2,
        "published-formula witness changed — re-pin against slackwater-lattice 5bff9a3"
    );
    assert_eq!(
        pypi_0_1_0_hex_distance(o, anti),
        1,
        "published-formula witness changed — re-pin against slackwater-lattice 5bff9a3"
    );
}

#[test]
fn p5_published_formula_violates_iff_on_exactly_192_pairs() {
    // The PyPI 0.1.0 defect is real and countable: 192 iff-violations over the
    // 3,721 ordered pairs of the radius-4 ball — the same deterministic count
    // the Python suite pins in tests/test_hex_distance_properties.py.
    let pts = ball(RADIUS);
    let mut bad: Vec<(EisensteinPoint, EisensteinPoint)> = Vec::new();
    for a in &pts {
        for b in &pts {
            let d1 = pypi_0_1_0_hex_distance(*a, *b) == 1;
            if a.neighbors().contains(b) != d1 {
                bad.push((*a, *b));
            }
        }
    }
    assert_eq!(
        bad.len(),
        192,
        "iff-violation count of the vendored 0.1.0 formula changed"
    );
}
