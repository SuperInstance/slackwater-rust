//! Integration tests for perception-core
//!
//! Tests focus on multi-agent convergence scenarios, Eisenstein coordinate
//! packing edge cases, and fleet-scale analysis.

use perception_core::{
    pack_eisenstein, unpack_eisenstein,
    Convergence, ConvergenceReport, ConvergenceStrength,
    MultiTrack, Track, TrackEvent,
};

// ════════════════════════════════════════════════════════════════════
// EISENSTEIN COORDINATE PACKING
// ════════════════════════════════════════════════════════════════════

#[test]
fn eisenstein_origin() {
    let packed = pack_eisenstein(0, 0);
    assert_eq!(packed, 0);
    assert_eq!(unpack_eisenstein(packed), (0, 0));
}

#[test]
fn eisenstein_max_positive() {
    let packed = pack_eisenstein(32767, 32767);
    assert_eq!(unpack_eisenstein(packed), (32767, 32767));
}

#[test]
fn eisenstein_max_negative() {
    let packed = pack_eisenstein(-32768, -32768);
    assert_eq!(unpack_eisenstein(packed), (-32768, -32768));
}

#[test]
fn eisenstein_mixed_signs() {
    let packed = pack_eisenstein(-100, 200);
    assert_eq!(unpack_eisenstein(packed), (-100, 200));
}

#[test]
fn eisenstein_extreme_values() {
    let test_values = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (32767, 32767),
        (-32768, -32768),
        (32767, -32768),
        (-32768, 32767),
    ];
    for &(a, b) in &test_values {
        let packed = pack_eisenstein(a, b);
        let (ra, rb) = unpack_eisenstein(packed);
        assert_eq!((ra, rb), (a, b), "Round-trip failed for ({}, {}): got ({}, {})", a, b, ra, rb);
    }
}

#[test]
fn eisenstein_overflow_wraps_via_i16_cast() {
    // Values beyond i16 range get truncated via `as i16` cast
    let packed = pack_eisenstein(32768, 0);
    let (a, b) = unpack_eisenstein(packed);
    // 32768 as i16 = -32768 (wrap)
    assert_eq!(a, -32768);
    assert_eq!(b, 0);
}

#[test]
fn track_event_from_eisenstein_preserves_tick() {
    for tick in [0, 1, 96, 500, u32::MAX] {
        let event = TrackEvent::from_eisenstein(tick, 0, 0);
        assert_eq!(event.tick, tick);
    }
}

// ════════════════════════════════════════════════════════════════════
// TRACK OPERATIONS
// ════════════════════════════════════════════════════════════════════

#[test]
fn track_new_initializes_correctly() {
    let track = Track::new("alpha", 3);
    assert_eq!(track.agent_id, "alpha");
    assert_eq!(track.channel, 3);
    assert!(track.is_empty());
    assert_eq!(track.len(), 0);
}

#[test]
fn track_channel_masked_to_4_bits() {
    let track = Track::new("test", 255); // 0xFF
    assert_eq!(track.channel, 15); // 0x0F
}

#[test]
fn track_add_multiple_events_sorts_by_tick() {
    let mut track = Track::new("sorted", 0);
    track.add_event(TrackEvent::from_eisenstein(300, 0, 0));
    track.add_event(TrackEvent::from_eisenstein(100, 0, 0));
    track.add_event(TrackEvent::from_eisenstein(200, 0, 0));
    track.add_event(TrackEvent::from_eisenstein(50, 0, 0));

    let ticks: Vec<u32> = track.events.iter().map(|e| e.tick).collect();
    assert_eq!(ticks, vec![50, 100, 200, 300]);
}

#[test]
fn track_events_at_nonexistent_tick() {
    let mut track = Track::new("empty", 0);
    track.add_event(TrackEvent::from_eisenstein(100, 0, 0));
    let results: Vec<_> = track.events_at_tick(999).collect();
    assert!(results.is_empty());
}

#[test]
fn track_events_near_position_with_zero_tolerance() {
    let mut track = Track::new("precise", 0);
    track.add_event(TrackEvent::from_eisenstein(0, 5, 5));
    track.add_event(TrackEvent::from_eisenstein(96, 5, 6));

    let near = track.events_near_position(pack_eisenstein(5, 5), 0);
    assert_eq!(near.len(), 1); // Only exact match
}

#[test]
fn track_events_near_position_empty_track() {
    let track = Track::new("empty", 0);
    let near = track.events_near_position(pack_eisenstein(0, 0), 100);
    assert!(near.is_empty());
}

// ════════════════════════════════════════════════════════════════════
// MULTI-TRACK CONVERGENCE — COMPLEX SCENARIOS
// ════════════════════════════════════════════════════════════════════

#[test]
fn multitrack_empty_analysis() {
    let mt = MultiTrack::new();
    let report = mt.analyze(0);
    assert_eq!(report.exact_count, 0);
    assert_eq!(report.weak_count, 0);
    assert_eq!(report.divergence_count, 0);
    assert_eq!(report.agent_count, 0);
    assert_eq!(report.convergence_ratio(), 0.0);
}

#[test]
fn multitrack_single_track_no_convergence() {
    let mut mt = MultiTrack::new();
    let mut t = Track::new("solo", 0);
    t.add_event(TrackEvent::from_eisenstein(0, 0, 0));
    t.add_event(TrackEvent::from_eisenstein(96, 1, 1));
    mt.add_track(t);

    let report = mt.analyze(0);
    assert_eq!(report.total_convergences(), 0);
    assert_eq!(report.divergence_count, 0);
}

#[test]
fn multitrack_all_agents_converge_every_tick() {
    let mut mt = MultiTrack::new();
    for (i, name) in ["alpha", "beta", "gamma", "delta"].iter().enumerate() {
        let mut t = Track::new(*name, i as u8);
        t.add_event(TrackEvent::from_eisenstein(0, 5, 5));
        t.add_event(TrackEvent::from_eisenstein(96, 6, 6));
        t.add_event(TrackEvent::from_eisenstein(192, 7, 7));
        mt.add_track(t);
    }

    let report = mt.analyze(0);
    assert_eq!(report.exact_count, 3);
    assert_eq!(report.weak_count, 0);
    assert_eq!(report.divergence_count, 0);
    assert_eq!(report.agent_count, 4);
    assert!((report.convergence_ratio() - 1.0).abs() < 0.001);
}

#[test]
fn multitrack_all_diverge() {
    let mut mt = MultiTrack::new();
    for (i, name) in ["alpha", "beta", "gamma"].iter().enumerate() {
        let mut t = Track::new(*name, i as u8);
        t.add_event(TrackEvent::from_eisenstein(0, (i * 100) as i32, (i * 100) as i32));
        mt.add_track(t);
    }

    let report = mt.analyze(5);
    assert_eq!(report.exact_count, 0);
    assert_eq!(report.weak_count, 0);
    assert_eq!(report.divergence_count, 1);
}

#[test]
fn multitrack_mixed_convergence_and_divergence() {
    let mut mt = MultiTrack::new();

    // Alpha and beta converge at tick 0 and 192, diverge at 96
    let mut alpha = Track::new("alpha", 0);
    alpha.add_event(TrackEvent::from_eisenstein(0, 5, 5));
    alpha.add_event(TrackEvent::from_eisenstein(96, 10, 10));
    alpha.add_event(TrackEvent::from_eisenstein(192, 7, 7));

    let mut beta = Track::new("beta", 1);
    beta.add_event(TrackEvent::from_eisenstein(0, 5, 5));
    beta.add_event(TrackEvent::from_eisenstein(96, 50, 50));
    beta.add_event(TrackEvent::from_eisenstein(192, 7, 7));

    mt.add_track(alpha);
    mt.add_track(beta);

    let report = mt.analyze(0);
    assert_eq!(report.exact_count, 2); // tick 0 and 192
    assert_eq!(report.divergence_count, 1); // tick 96
    assert!((report.convergence_ratio() - (2.0 / 3.0)).abs() < 0.01);
}

#[test]
fn multitrack_weak_convergence_boundary_tolerance() {
    let mut mt = MultiTrack::new();

    let mut t1 = Track::new("alpha", 0);
    t1.add_event(TrackEvent::from_eisenstein(0, 0, 0));

    let mut t2 = Track::new("beta", 1);
    // Position (3, 3) — distance 3 from origin
    t2.add_event(TrackEvent::from_eisenstein(0, 3, 3));

    mt.add_track(t1);
    mt.add_track(t2);

    // Tolerance of 2: should diverge (distance 3 > 2)
    let report = mt.analyze(2);
    assert_eq!(report.weak_count, 0);
    assert_eq!(report.divergence_count, 1);

    // Tolerance of 3: should weakly converge
    let report = mt.analyze(3);
    assert_eq!(report.weak_count, 1);
    assert_eq!(report.divergence_count, 0);
}

#[test]
fn multitrack_zero_tolerance_no_weak_convergence() {
    let mut mt = MultiTrack::new();

    let mut t1 = Track::new("alpha", 0);
    t1.add_event(TrackEvent::from_eisenstein(0, 0, 0));

    let mut t2 = Track::new("beta", 1);
    t2.add_event(TrackEvent::from_eisenstein(0, 1, 1));

    mt.add_track(t1);
    mt.add_track(t2);

    let report = mt.analyze(0);
    assert_eq!(report.weak_count, 0);
    assert_eq!(report.exact_count, 0);
    assert_eq!(report.divergence_count, 1);
}

#[test]
fn multitrack_many_agents_partial_convergence() {
    let mut mt = MultiTrack::new();

    // 5 agents: 3 at (5,5), 2 at (10,10) at tick 0
    for (i, &(a, b)) in [(5, 5), (5, 5), (5, 5), (10, 10), (10, 10)].iter().enumerate() {
        let mut t = Track::new(format!("agent_{}", i), i as u8);
        t.add_event(TrackEvent::from_eisenstein(0, a, b));
        mt.add_track(t);
    }

    let report = mt.analyze(0);
    // Not all at same position → divergence (2 distinct groups)
    assert_eq!(report.exact_count, 0);
    assert_eq!(report.divergence_count, 1);
}

#[test]
fn convergence_streNGTH_distinct() {
    assert_ne!(ConvergenceStrength::Exact, ConvergenceStrength::Weak);
}

#[test]
fn convergence_report_total_convergences() {
    let report = ConvergenceReport {
        convergences: vec![
            Convergence {
                tick: 0,
                agents: vec!["a".into(), "b".into()],
                position_packed: 0,
                strength: ConvergenceStrength::Exact,
            },
            Convergence {
                tick: 96,
                agents: vec!["a".into(), "c".into()],
                position_packed: 100,
                strength: ConvergenceStrength::Weak,
            },
        ],
        exact_count: 1,
        weak_count: 1,
        divergence_count: 0,
        agent_count: 3,
    };
    assert_eq!(report.total_convergences(), 2);
}

// ════════════════════════════════════════════════════════════════════
// CONVERGENCE RATIO EDGE CASES
// ════════════════════════════════════════════════════════════════════

#[test]
fn convergence_ratio_all_exact() {
    let report = ConvergenceReport {
        convergences: vec![],
        exact_count: 10,
        weak_count: 0,
        divergence_count: 0,
        agent_count: 2,
    };
    assert!((report.convergence_ratio() - 1.0).abs() < 0.001);
}

#[test]
fn convergence_ratio_all_divergence() {
    let report = ConvergenceReport {
        convergences: vec![],
        exact_count: 0,
        weak_count: 0,
        divergence_count: 10,
        agent_count: 2,
    };
    assert!((report.convergence_ratio() - 0.0).abs() < 0.001);
}

#[test]
fn convergence_ratio_balanced() {
    let report = ConvergenceReport {
        convergences: vec![],
        exact_count: 5,
        weak_count: 5,
        divergence_count: 10,
        agent_count: 3,
    };
    assert!((report.convergence_ratio() - 0.5).abs() < 0.001);
}

// ════════════════════════════════════════════════════════════════════
// MULTI-TRACK MANAGEMENT
// ════════════════════════════════════════════════════════════════════

#[test]
fn multitrack_add_and_count() {
    let mut mt = MultiTrack::new();
    assert_eq!(mt.track_count(), 0);

    mt.add_track(Track::new("a", 0));
    assert_eq!(mt.track_count(), 1);

    mt.add_track(Track::new("b", 1));
    mt.add_track(Track::new("c", 2));
    assert_eq!(mt.track_count(), 3);
}

#[test]
fn multitrack_tracks_returns_in_order() {
    let mut mt = MultiTrack::new();
    mt.add_track(Track::new("first", 0));
    mt.add_track(Track::new("second", 1));
    mt.add_track(Track::new("third", 2));

    let names: Vec<&str> = mt.tracks().iter().map(|t| t.agent_id.as_str()).collect();
    assert_eq!(names, vec!["first", "second", "third"]);
}
