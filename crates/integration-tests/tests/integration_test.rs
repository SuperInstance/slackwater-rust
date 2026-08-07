//! Integration tests across slackwater-rust workspace crates.
//!
//! These tests exercise the interactions between layers:
//! - flux-core (exact types, error mask, SWMIDI events)
//! - tempo-core (beat clock, tempo map)
//! - lattice-core (Eisenstein coordinates, snapping)
//! - harmony-core (flow state detection)
//! - tminus-core (prediction, calibration)
//! - swmidi (wire format codec)
//! - perception-core (multi-track convergence)

use flux_core::error_mask::ErrorMask;
use flux_core::exact::EisensteinCoord;
use lattice_core::snap::snap_position;
use swmidi::{EventType, SwmidiEvent, SwmidiStream};
use tempo_core::{BeatClock, MusicalPosition, PPQ};
use tminus_core::TMinusEngine;

#[test]
fn full_pipeline_build_to_swmidi() {
    // Simulate a build action flowing through the pipeline:
    // 1. Agent decides to place a part at Eisenstein (3, 4)
    // 2. Position is snapped to the A2 lattice
    // 3. Event is encoded as SWMIDI
    // 4. Event flows through the stream without friction

    // 1. The build position
    let coord = EisensteinCoord::new(3, 4);

    // 2. Snap to lattice
    let _snapped = snap_position(coord.a as f64, coord.b as f64);

    // 3. Encode as SWMIDI NoteOn event (flow state — no friction)
    let event = SwmidiEvent::new(
        EventType::NoteOn,
        0,                      // channel 0
        (coord.a as u8) & 0x7F, // pitch = action code
        100,                    // velocity = weight
        ErrorMask::FLOW.bits(), // no friction
        0,                      // tick 0
    );

    // 4. Round-trip through encode/decode
    let encoded = event.encode();
    let decoded = SwmidiEvent::decode(&encoded).unwrap();
    assert_eq!(event, decoded);
    assert!(decoded.is_flow());
}

#[test]
fn tempo_synchronization_across_agents() {
    // Two agents share a BeatClock at 120 BPM.
    // Agent A places at tick 0, Agent B places at tick 96 (one beat later).
    // The BeatClock should show 500_000 µs between them.

    let clock = BeatClock::new();
    let us_at_start = clock.tempo_map().tick_to_us(0);
    let us_at_beat_1 = clock.tempo_map().tick_to_us(PPQ);

    let elapsed_us = us_at_beat_1 - us_at_start;
    assert_eq!(elapsed_us, 500_000); // 0.5 seconds at 120 BPM
}

#[test]
fn tempo_change_affects_calibration() {
    // Agent A is playing at a steady pace. T-Minus engine, trained on
    // this pace, should show calibration drift when the pace changes.

    let mut engine = TMinusEngine::default_engine();

    // Train on steady linear growth
    for i in 0..10i32 {
        engine.observe(i as u32 * 10, i * 100);
    }

    // Predict tick 100 — should be around 900-1100
    let pred = engine.predict(100);
    assert!(pred.expected_value > 800 && pred.expected_value < 1200);

    // Now observe tick 100 with a much lower value (tempo slowed)
    let cal = engine.observe(100, 500);
    assert!(
        cal.magnitude > 20,
        "Expected significant calibration drift, got magnitude={}",
        cal.magnitude
    );
}

#[test]
fn error_mask_flows_through_swmidi() {
    // A build event encounters a spatial collision (bit 0 set).
    // The error mask should survive SWMIDI encode/decode.

    let friction_mask = ErrorMask::SPATIAL.with(ErrorMask::TEMPORAL);
    let event = SwmidiEvent::new(EventType::NoteOn, 2, 60, 80, friction_mask.bits(), 192);

    let encoded = event.encode();
    let decoded = SwmidiEvent::decode(&encoded).unwrap();

    assert_eq!(decoded.error_mask, friction_mask.bits());
    assert!(decoded.has_friction());
    assert!(!decoded.is_flow());

    // Reconstruct the ErrorMask
    let restored = ErrorMask::from_bits(decoded.error_mask);
    assert!(restored.contains(ErrorMask::SPATIAL));
    assert!(restored.contains(ErrorMask::TEMPORAL));
    assert!(!restored.contains(ErrorMask::SAFETY));
}

#[test]
fn musical_position_tracks_beat_clock() {
    // In 4/4 time at 96 PPQ:
    // - Bar 0, Beat 0 = tick 0
    // - Bar 0, Beat 1 = tick 96
    // - Bar 1, Beat 0 = tick 384 (4 * 96)
    // - Bar 2, Beat 3 = tick 1056 (2 * 4 * 96 + 3 * 96)

    let beats_per_bar = 4;

    let pos = MusicalPosition::from_tick(0, beats_per_bar);
    assert_eq!(pos.bar, 0);
    assert_eq!(pos.beat, 0);

    let pos = MusicalPosition::from_tick(96, beats_per_bar);
    assert_eq!(pos.bar, 0);
    assert_eq!(pos.beat, 1);

    let pos = MusicalPosition::from_tick(384, beats_per_bar);
    assert_eq!(pos.bar, 1);
    assert_eq!(pos.beat, 0);

    let pos = MusicalPosition::from_tick(1056, beats_per_bar);
    assert_eq!(pos.bar, 2);
    assert_eq!(pos.beat, 3);

    // Round trip
    assert_eq!(
        MusicalPosition::from_tick(1056, beats_per_bar).to_tick(beats_per_bar),
        1056
    );
}

#[test]
fn swmidi_stream_sorted_by_tick_for_replay() {
    // Events arrive out of order from multiple agents.
    // The stream should be sortable for deterministic replay.

    let mut stream = SwmidiStream::new();
    stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 288));
    stream.push(SwmidiEvent::new(EventType::NoteOn, 1, 64, 100, 0, 0));
    stream.push(SwmidiEvent::new(EventType::NoteOn, 2, 67, 100, 0, 96));
    stream.push(SwmidiEvent::new(EventType::NoteOff, 0, 60, 0, 0, 192));

    stream.sort_by_tick();

    let ticks: Vec<u32> = stream.iter().map(|e| e.tick).collect();
    assert_eq!(ticks, vec![0, 96, 192, 288]);
}

#[test]
fn calibration_history_tracks_improvement() {
    // Agent starts badly calibrated, then improves over time.
    let mut engine = TMinusEngine::default_engine();
    engine.observe(0, 100); // Initialize

    // Bad predictions early (decreasing error over time)
    for i in 1..20i32 {
        let big_error = 100 + (20 - i) * 5;
        engine.observe(i as u32, big_error);
    }

    // Good predictions later (tiny variance)
    for i in 20..40i32 {
        let small_error = 100 + (i % 3);
        engine.observe(i as u32, small_error);
    }

    let history = engine.history();
    assert!(history.is_improving(), "Calibration should be improving");
}

#[test]
fn eisenstein_lattice_round_trip_through_packing() {
    // Eisenstein coordinates should survive packing/unpacking
    // through perception-core's packing format.

    use perception_core::{pack_eisenstein, unpack_eisenstein};

    let points = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, -1), (5, -3), (-2, 7)];

    for (a, b) in points {
        let packed = pack_eisenstein(a, b);
        let (ra, rb) = unpack_eisenstein(packed);
        assert_eq!(a, ra, "Round trip failed for ({}, {})", a, b);
        assert_eq!(b, rb, "Round trip failed for ({}, {})", a, b);
    }
}

#[test]
fn full_fleet_convergence_scenario() {
    // Three agents place parts. Two converge on the same position.
    // Perception-core should detect exact convergence.

    use perception_core::{MultiTrack, Track, TrackEvent};

    let mut mt = MultiTrack::new();

    let mut alpha = Track::new("alpha", 0);
    alpha.add_event(TrackEvent::from_eisenstein(0, 5, 5));
    alpha.add_event(TrackEvent::from_eisenstein(96, 6, 6));

    let mut beta = Track::new("beta", 1);
    beta.add_event(TrackEvent::from_eisenstein(0, 5, 5)); // Converge at tick 0
    beta.add_event(TrackEvent::from_eisenstein(96, 10, 10)); // Diverge at tick 96

    let mut gamma = Track::new("gamma", 2);
    gamma.add_event(TrackEvent::from_eisenstein(0, 5, 5)); // Converge at tick 0
    gamma.add_event(TrackEvent::from_eisenstein(96, 6, 6)); // Converge with alpha at tick 96

    mt.add_track(alpha);
    mt.add_track(beta);
    mt.add_track(gamma);

    let report = mt.analyze(0);

    // Tick 0: all three at (5,5) = exact convergence
    // Tick 96: alpha and gamma at (6,6), beta at (10,10) = divergence
    assert!(
        report.exact_count >= 1,
        "Should have at least one exact convergence"
    );
    assert_eq!(report.agent_count, 3);
}
