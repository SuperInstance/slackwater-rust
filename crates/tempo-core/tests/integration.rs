//! Integration tests for tempo-core
//!
//! Tests focus on multi-tempo scenarios, round-trip conversions,
//! edge cases at bar/beat boundaries, and clock behavior over time.

use tempo_core::{
    BeatClock, MusicalPosition, TempoEvent, TempoMap,
    DEFAULT_US_PER_QUARTER, PPQ,
};

// ════════════════════════════════════════════════════════════════════
// TEMPO EVENT CONVERSIONS
// ════════════════════════════════════════════════════════════════════

#[test]
fn tempo_event_standard_tempos() {
    let test_cases = [
        (60.0, 1_000_000),
        (90.0, 666_667),
        (120.0, 500_000),
        (140.0, 428_571),
        (180.0, 333_333),
        (240.0, 250_000),
    ];
    for (bpm, expected_us) in test_cases {
        let event = TempoEvent::from_bpm(0, bpm);
        assert!(
            (event.bpm() - bpm).abs() < 1.0,
            "BPM round-trip failed: {} → {} → {}",
            bpm,
            event.us_per_quarter,
            event.bpm()
        );
        // us_per_quarter should be close (integer truncation)
        let drift = (event.us_per_quarter as i64 - expected_us as i64).unsigned_abs();
        assert!(drift <= 1, "us_per_quarter drift too large for {} BPM: {}", bpm, drift);
    }
}

#[test]
fn tempo_event_from_bpm_clamps_zero() {
    let event = TempoEvent::from_bpm(0, 0.0);
    assert!(event.us_per_quarter >= 1);
    assert!(event.bpm() > 0.0); // Should be very large (60M / 1 = 60M BPM)
}

#[test]
fn tempo_event_from_bpm_clamps_negative() {
    let event = TempoEvent::from_bpm(0, -10.0);
    assert!(event.us_per_quarter >= 1);
}

#[test]
fn tempo_event_new_is_const_correct() {
    const EVENT: TempoEvent = TempoEvent::new(42, 500_000);
    assert_eq!(EVENT.tick, 42);
    assert_eq!(EVENT.us_per_quarter, 500_000);
}

#[test]
fn tempo_event_ordering_by_tick() {
    let a = TempoEvent::new(100, 500_000);
    let b = TempoEvent::new(200, 400_000);
    let c = TempoEvent::new(100, 600_000);

    assert!(a < b);
    assert!(a <= c);
    // TempoEvent Ord is by tick only. Same tick → Equal (even with different tempo).
    // But Eq requires all fields match. So a != TempoEvent with different us_per_quarter.
    // The Ord impl says cmp == Equal for same tick, but PartialEq checks all fields.
    // This means: a.cmp(&c) == Equal, but a != c.
    assert_eq!(a.cmp(&TempoEvent::new(100, 999_999)), core::cmp::Ordering::Equal);
}

// ════════════════════════════════════════════════════════════════════
// TEMPO MAP — MULTI-TEMPO SCENARIOS
// ════════════════════════════════════════════════════════════════════

#[test]
fn tempo_map_default_is_120_bpm_at_tick_0() {
    let map = TempoMap::new();
    let tempo = map.tempo_at(0);
    assert_eq!(tempo.us_per_quarter, DEFAULT_US_PER_QUARTER);
    assert!((tempo.bpm() - 120.0).abs() < 0.1);
}

#[test]
fn tempo_map_with_bpm_constructor() {
    let map = TempoMap::with_bpm(90.0);
    let tempo = map.tempo_at(0);
    assert!((tempo.bpm() - 90.0).abs() < 1.0);
}

#[test]
fn tempo_map_insert_replaces_at_same_tick() {
    let mut map = TempoMap::new();
    map.insert(TempoEvent::new(0, 400_000)); // Replace default
    let tempo = map.tempo_at(0);
    assert_eq!(tempo.us_per_quarter, 400_000);
    // Should still have exactly one event at tick 0
    assert_eq!(map.len(), 1);
}

#[test]
fn tempo_map_insert_at_nonzero_tick() {
    let mut map = TempoMap::new();
    map.insert(TempoEvent::from_bpm(96, 60.0)); // Tempo change at beat 1
    assert_eq!(map.len(), 2);

    // Before the change: 120 BPM
    assert!((map.tempo_at(0).bpm() - 120.0).abs() < 0.1);
    // After the change: 60 BPM
    assert!((map.tempo_at(96).bpm() - 60.0).abs() < 1.0);
}

#[test]
fn tempo_map_tempo_at_finds_correct_segment() {
    let mut map = TempoMap::new(); // 120 BPM at 0
    map.insert(TempoEvent::from_bpm(192, 90.0));  // 90 BPM at tick 192
    map.insert(TempoEvent::from_bpm(384, 180.0)); // 180 BPM at tick 384

    // Tick 0: 120 BPM
    assert!((map.tempo_at(0).bpm() - 120.0).abs() < 0.1);
    // Tick 95: still 120 BPM
    assert!((map.tempo_at(95).bpm() - 120.0).abs() < 0.1);
    // Tick 192: 90 BPM
    assert!((map.tempo_at(192).bpm() - 90.0).abs() < 1.0);
    // Tick 300: still 90 BPM
    assert!((map.tempo_at(300).bpm() - 90.0).abs() < 1.0);
    // Tick 384: 180 BPM
    assert!((map.tempo_at(384).bpm() - 180.0).abs() < 1.0);
    // Tick 1000: still 180 BPM
    assert!((map.tempo_at(1000).bpm() - 180.0).abs() < 1.0);
}

#[test]
fn tempo_map_tick_to_us_multi_segment() {
    let mut map = TempoMap::new(); // 120 BPM = 500_000 µs/quarter
    map.insert(TempoEvent::new(96, 1_000_000)); // 60 BPM at tick 96

    // Segment 1: ticks 0–95 at 500_000 µs/quarter
    // 96 ticks = 1 quarter = 500_000 µs
    //
    // Segment 2: ticks 96–191 at 1_000_000 µs/quarter
    // 96 ticks = 1 quarter = 1_000_000 µs
    //
    // Total for 192 ticks: 1_500_000 µs
    assert_eq!(map.tick_to_us(192), 1_500_000);
}

#[test]
fn tempo_map_tick_to_us_at_tick_zero() {
    let map = TempoMap::new();
    assert_eq!(map.tick_to_us(0), 0);
}

#[test]
fn tempo_map_us_to_tick_inverse_at_120bpm() {
    let map = TempoMap::new();
    // At 120 BPM, 96 ticks = 500_000 µs. So 1 tick ≈ 5208 µs.
    // Small µs values map to tick 0, so only test values that produce measurable ticks.
    for us in [0, 500_000, 1_000_000, 2_500_000, 5_000_000] {
        let tick = map.us_to_tick(us);
        let us_back = map.tick_to_us(tick);
        // Should be close (integer division may lose precision up to one tick's worth)
        let drift = (us_back as i64 - us as i64).unsigned_abs();
        let max_drift = DEFAULT_US_PER_QUARTER / PPQ as u64; // µs per tick
        assert!(drift <= max_drift, "Round-trip drift too large for {} µs: {} (tick {} → {} µs)", us, drift, tick, us_back);
    }
}

#[test]
fn tempo_map_us_to_tick_multi_tempo() {
    let mut map = TempoMap::new();
    map.insert(TempoEvent::new(96, 1_000_000)); // 60 BPM at tick 96

    // 500_000 µs → tick 96 (end of first segment at 120 BPM)
    assert_eq!(map.us_to_tick(500_000), 96);

    // 1_500_000 µs → tick 192 (end of second segment at 60 BPM)
    let tick = map.us_to_tick(1_500_000);
    assert!(tick >= 190 && tick <= 194, "Expected ~192, got {}", tick);
}

#[test]
fn tempo_map_is_never_empty() {
    let map = TempoMap::new();
    assert!(!map.is_empty());
}

#[test]
fn tempo_map_iter_returns_sorted_events() {
    let mut map = TempoMap::new();
    map.insert(TempoEvent::from_bpm(500, 90.0));
    map.insert(TempoEvent::from_bpm(200, 140.0));
    map.insert(TempoEvent::from_bpm(100, 60.0));

    let ticks: Vec<u32> = map.iter().map(|e| e.tick).collect();
    assert!(ticks.windows(2).all(|w| w[0] <= w[1]), "Events not sorted: {:?}", ticks);
}

// ════════════════════════════════════════════════════════════════════
// BEAT CLOCK — ADVANCED BEHAVIOR
// ════════════════════════════════════════════════════════════════════

#[test]
fn beat_clock_with_custom_bpm() {
    let clock = BeatClock::with_bpm(90.0);
    assert!((clock.bpm() - 90.0).abs() < 1.0);
    assert_eq!(clock.tick(), 0);
}

#[test]
fn beat_clock_advance_accumulates() {
    let mut clock = BeatClock::new();
    clock.advance(10);
    clock.advance(20);
    clock.advance(30);
    assert_eq!(clock.tick(), 60);
}

#[test]
fn beat_clock_advance_saturates_at_max() {
    let mut clock = BeatClock::new();
    clock.advance(u32::MAX);
    let before = clock.tick();
    clock.advance(1);
    assert_eq!(clock.tick(), before); // Saturated, didn't wrap
}

#[test]
fn beat_clock_seek_forward_succeeds() {
    let mut clock = BeatClock::new();
    clock.advance(100);
    let result = clock.seek(200);
    assert!(result.is_ok());
    assert_eq!(clock.tick(), 200);
}

#[test]
fn beat_clock_seek_backward_fails() {
    let mut clock = BeatClock::new();
    clock.advance(200);
    let result = clock.seek(100);
    assert!(result.is_err());
    assert_eq!(clock.tick(), 200); // Unchanged
}

#[test]
fn beat_clock_seek_to_current_succeeds() {
    let mut clock = BeatClock::new();
    clock.advance(100);
    let result = clock.seek(100);
    assert!(result.is_ok());
}

#[test]
fn beat_clock_set_tempo_at_current_tick() {
    let mut clock = BeatClock::new();
    clock.advance(96);
    clock.set_bpm(60.0);

    // Tempo should be 60 BPM at tick 96
    assert!((clock.bpm() - 60.0).abs() < 1.0);

    // The tempo map should have two events: tick 0 (120) and tick 96 (60)
    let map = clock.tempo_map();
    assert_eq!(map.len(), 2);
}

#[test]
fn beat_clock_set_tempo_multiple_times() {
    let mut clock = BeatClock::new();
    clock.set_bpm(100.0);
    clock.advance(96);
    clock.set_bpm(80.0);
    clock.advance(96);
    clock.set_bpm(140.0);

    assert!((clock.bpm() - 140.0).abs() < 1.0);
    assert_eq!(clock.tempo_map().len(), 3); // tick 0, 96, 192
}

#[test]
fn beat_clock_current_us_at_constant_tempo() {
    let mut clock = BeatClock::new(); // 120 BPM
    clock.advance(96); // One quarter note
    assert_eq!(clock.current_us(), 500_000);
    clock.advance(96); // Two quarter notes
    assert_eq!(clock.current_us(), 1_000_000);
}

#[test]
fn beat_clock_current_us_with_tempo_change() {
    let mut clock = BeatClock::new(); // 120 BPM at tick 0
    clock.advance(96);               // Now at tick 96
    assert_eq!(clock.current_us(), 500_000);

    clock.set_bpm(60.0);             // 60 BPM at tick 96
    clock.advance(96);               // Now at tick 192
    // Segment 1: 96 ticks at 500_000 µs/q = 500_000 µs
    // Segment 2: 96 ticks at 1_000_000 µs/q = 1_000_000 µs
    // Total: 1_500_000 µs
    assert_eq!(clock.current_us(), 1_500_000);
}

#[test]
fn beat_clock_reset_clears_tempo_changes() {
    let mut clock = BeatClock::new();
    clock.set_bpm(60.0);
    clock.advance(500);
    assert_ne!(clock.tick(), 0);

    clock.reset();
    assert_eq!(clock.tick(), 0);
    assert!((clock.bpm() - 120.0).abs() < 0.1);
    assert_eq!(clock.tempo_map().len(), 1); // Only the default
}

#[test]
fn beat_clock_us_per_quarter_tracks_tempo() {
    let mut clock = BeatClock::new();
    assert_eq!(clock.us_per_quarter(), DEFAULT_US_PER_QUARTER);

    clock.set_bpm(60.0);
    assert_eq!(clock.us_per_quarter(), 1_000_000);

    clock.set_bpm(180.0);
    assert!((clock.us_per_quarter() as f64 - 333_333.0).abs() < 2.0);
}

// ════════════════════════════════════════════════════════════════════
// MUSICAL POSITION — TIME SIGNATURES
// ════════════════════════════════════════════════════════════════════

#[test]
fn musical_position_4_4_time() {
    // Tick 0 = bar 0, beat 0, sub 0
    let pos = MusicalPosition::from_tick(0, 4);
    assert_eq!((pos.bar, pos.beat, pos.sub_tick), (0, 0, 0));

    // Tick 96 = bar 0, beat 1
    let pos = MusicalPosition::from_tick(96, 4);
    assert_eq!((pos.bar, pos.beat), (0, 1));

    // Tick 384 = bar 1, beat 0 (4 beats × 96 ticks)
    let pos = MusicalPosition::from_tick(384, 4);
    assert_eq!((pos.bar, pos.beat), (1, 0));

    // Tick 480 = bar 1, beat 1
    let pos = MusicalPosition::from_tick(480, 4);
    assert_eq!((pos.bar, pos.beat), (1, 1));
}

#[test]
fn musical_position_3_4_time() {
    // 3/4: 3 beats × 96 = 288 ticks per bar
    let pos = MusicalPosition::from_tick(288, 3);
    assert_eq!((pos.bar, pos.beat), (1, 0));

    // Tick 200 = bar 0, beat 2, sub 8
    let pos = MusicalPosition::from_tick(200, 3);
    assert_eq!((pos.bar, pos.beat), (0, 2));
    assert_eq!(pos.sub_tick, 8);
}

#[test]
fn musical_position_12_8_time() {
    // 12/8: 12 beats × 96 = 1152 ticks per bar
    // But in tensor-midi-core, 12/8 uses 48 ticks per pulse (eighth note)
    // Here in tempo-core, PPQ=96 always. The caller handles the mapping.
    let pos = MusicalPosition::from_tick(1152, 12);
    assert_eq!((pos.bar, pos.beat), (1, 0));
}

#[test]
fn musical_position_round_trip_various_time_signatures() {
    for &beats_per_bar in &[1, 2, 3, 4, 5, 6, 7, 8, 12] {
        for tick in [0, 1, 50, 96, 200, 500, 1000, 5000] {
            let pos = MusicalPosition::from_tick(tick, beats_per_bar);
            let recovered = pos.to_tick(beats_per_bar);
            assert_eq!(
                recovered, tick,
                "Round-trip failed: tick {} in {}/4 → ({}, {}, {}) → {}",
                tick, beats_per_bar, pos.bar, pos.beat, pos.sub_tick, recovered
            );
        }
    }
}

#[test]
fn musical_position_zero_beats_per_bar_clamps_to_one() {
    // Should not panic. With beats_per_bar clamped to 1,
    // tick 96 = bar 1, beat 0, sub 0
    let pos = MusicalPosition::from_tick(96, 0);
    assert_eq!(pos.bar, 1);
    assert_eq!(pos.beat, 0);
    assert_eq!(pos.sub_tick, 0);
}

#[test]
fn musical_position_bar_boundaries_4_4() {
    // Each bar = 384 ticks
    for bar in 0..5 {
        let tick = bar * 384;
        let pos = MusicalPosition::from_tick(tick, 4);
        assert_eq!(pos.bar, bar);
        assert_eq!(pos.beat, 0);
        assert_eq!(pos.sub_tick, 0);
    }
}

#[test]
fn musical_position_last_tick_of_bar() {
    // Last tick of bar 0 in 4/4 = tick 383
    let pos = MusicalPosition::from_tick(383, 4);
    assert_eq!(pos.bar, 0);
    assert_eq!(pos.beat, 3);
    assert_eq!(pos.sub_tick, 95);
}

// ════════════════════════════════════════════════════════════════════
// CONSTANTS
// ════════════════════════════════════════════════════════════════════

#[test]
fn ppq_is_96() {
    assert_eq!(PPQ, 96);
}

#[test]
fn default_us_per_quarter_is_500k() {
    assert_eq!(DEFAULT_US_PER_QUARTER, 500_000);
}

#[test]
fn default_us_per_quarter_is_120_bpm() {
    let bpm = 60_000_000.0 / DEFAULT_US_PER_QUARTER as f64;
    assert!((bpm - 120.0).abs() < 0.001);
}
