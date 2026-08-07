#![warn(clippy::all)]
#![deny(unsafe_code)]

//! # tempo-core
//!
//! Layer 3: BeatClock and TempoMap — the shared temporal spine.
//!
//! Every agent in the Slackwater fleet agrees on one clock. That clock
//! ticks at **96 PPQ** (pulses per quarter note). There are no floating-
//! point tempo calculations in the agreement path — microseconds per
//! quarter note are integer, tempo changes are monotonic in the map.
//!
//! ## Core types
//!
//! - [`BeatClock`] — The shared tick counter. Monotonic, never goes backward.
//! - [`TempoMap`] — A sorted list of tempo changes. Resolves tick → time.
//! - [`TempoEvent`] — A single tempo change at a specific tick.
//!
//! ## Design principles
//!
//! 1. **96 PPQ.** All agents count time in ticks at this resolution.
//! 2. **Integer microseconds.** No float drift in tempo math.
//! 3. **Monotonic clock.** The tick counter only goes forward.
//! 4. **Tempo changes are events, not globals.** They live in the map,
//!    indexed by the tick where they take effect.

use core::cmp::Ordering;
use serde::{Deserialize, Serialize};

/// Pulses per quarter note — the universal tick resolution.
pub const PPQ: u32 = 96;

/// Default tempo in microseconds per quarter note (120 BPM = 500_000 µs/quarter).
pub const DEFAULT_US_PER_QUARTER: u64 = 500_000;

/// A tempo change event, anchored at a specific tick.
///
/// `us_per_quarter` follows the MIDI tempo meta event convention:
/// 500_000 = 120 BPM, 666_667 = 90 BPM, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TempoEvent {
    /// The tick at which this tempo takes effect.
    pub tick: u32,
    /// Microseconds per quarter note at this tempo.
    pub us_per_quarter: u64,
}

impl TempoEvent {
    /// Create a new tempo event at the given tick.
    pub const fn new(tick: u32, us_per_quarter: u64) -> Self {
        Self { tick, us_per_quarter }
    }

    /// Convert microseconds-per-quarter to BPM.
    pub fn bpm(&self) -> f64 {
        60_000_000.0 / self.us_per_quarter as f64
    }

    /// Create a tempo event from a BPM value.
    pub fn from_bpm(tick: u32, bpm: f64) -> Self {
        let us = (60_000_000.0 / bpm.max(1.0)) as u64;
        Self::new(tick, us.max(1))
    }
}

impl PartialOrd for TempoEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TempoEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tick.cmp(&other.tick)
    }
}

// ── TempoMap ────────────────────────────────────────────────────────

/// A sorted map of tempo changes. Resolves any tick to its tempo.
///
/// The map is always sorted by tick. If two events share a tick, the
/// later one in the vec wins (most recent insertion).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TempoMap {
    events: Vec<TempoEvent>,
}

impl Default for TempoMap {
    fn default() -> Self {
        Self::new()
    }
}

impl TempoMap {
    /// Create a new empty map with a default 120 BPM tempo at tick 0.
    pub fn new() -> Self {
        Self {
            events: vec![TempoEvent::new(0, DEFAULT_US_PER_QUARTER)],
        }
    }

    /// Create a map starting at a specific BPM.
    pub fn with_bpm(bpm: f64) -> Self {
        Self {
            events: vec![TempoEvent::from_bpm(0, bpm)],
        }
    }

    /// Insert a tempo change. Keeps the map sorted.
    ///
    /// If a tempo event already exists at the same tick, it is replaced.
    pub fn insert(&mut self, event: TempoEvent) {
        // Find insertion point
        let pos = self.events.binary_search_by(|e| e.tick.cmp(&event.tick));
        match pos {
            Ok(idx) => self.events[idx] = event,
            Err(idx) => self.events.insert(idx, event),
        }
    }

    /// Get the tempo event active at the given tick.
    ///
    /// Returns the last event whose tick is <= `tick`.
    pub fn tempo_at(&self, tick: u32) -> TempoEvent {
        match self.events.binary_search_by(|e| e.tick.cmp(&tick)) {
            Ok(idx) => self.events[idx],
            Err(0) => self.events[0],
            Err(idx) => self.events[idx - 1],
        }
    }

    /// Convert a tick to microseconds since tick 0.
    ///
    /// Walks the tempo map, accumulating integer microseconds per segment.
    /// No float drift — this is an agreement path.
    pub fn tick_to_us(&self, tick: u32) -> u64 {
        let mut total_us: u64 = 0;
        let mut last_tick: u32 = 0;
        let mut current_us_per_quarter: u64 = self.events[0].us_per_quarter;

        for event in &self.events {
            if event.tick > tick {
                break;
            }
            // Accumulate time in the previous tempo segment
            let delta_ticks = (event.tick - last_tick) as u64;
            total_us += delta_ticks * current_us_per_quarter / PPQ as u64;
            last_tick = event.tick;
            current_us_per_quarter = event.us_per_quarter;
        }

        // Final segment from last tempo change to target tick
        let delta_ticks = (tick - last_tick) as u64;
        total_us += delta_ticks * current_us_per_quarter / PPQ as u64;
        total_us
    }

    /// Convert microseconds to a tick (inverse of [`tick_to_us`]).
    ///
    /// May return a tick that is slightly before the exact position due
    /// to integer division. This is acceptable — the clock is monotonic.
    pub fn us_to_tick(&self, us: u64) -> u32 {
        let mut remaining_us = us;
        let mut last_tick: u32 = 0;
        let mut current_us_per_quarter: u64 = self.events[0].us_per_quarter;

        for window in self.events.windows(2) {
            let (prev, next) = (&window[0], &window[1]);
            let segment_ticks = (next.tick - prev.tick) as u64;
            let segment_us = segment_ticks * current_us_per_quarter / PPQ as u64;

            if remaining_us <= segment_us {
                // Target is within this segment
                let ticks_here =
                    (remaining_us * PPQ as u64 / current_us_per_quarter.max(1)) as u32;
                return last_tick + ticks_here;
            }

            remaining_us -= segment_us;
            last_tick = next.tick;
            current_us_per_quarter = next.us_per_quarter;
        }

        // Target is in the final segment (extends to infinity)
        let ticks_here =
            (remaining_us * PPQ as u64 / current_us_per_quarter.max(1)) as u32;
        last_tick + ticks_here
    }

    /// Number of tempo events in the map.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the map is empty (it never is — always has at least tick 0).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Iterate over all tempo events.
    pub fn iter(&self) -> impl Iterator<Item = &TempoEvent> {
        self.events.iter()
    }
}

// ── BeatClock ───────────────────────────────────────────────────────

/// The shared monotonic tick counter.
///
/// Every agent reads from the same BeatClock. The clock advances when
/// the audio thread (or a timer) calls [`advance`]. The tick count
/// never goes backward.
///
/// The clock also tracks the current tempo, which can change via
/// [`set_tempo`]. When the tempo changes, a [`TempoEvent`] is recorded
/// at the current tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatClock {
    /// Current tick position. Monotonic.
    tick: u32,
    /// The tempo map holding all tempo changes.
    map: TempoMap,
}

impl Default for BeatClock {
    fn default() -> Self {
        Self::new()
    }
}

impl BeatClock {
    /// Create a new BeatClock at tick 0, 120 BPM.
    pub fn new() -> Self {
        Self {
            tick: 0,
            map: TempoMap::new(),
        }
    }

    /// Create a BeatClock starting at a specific BPM.
    pub fn with_bpm(bpm: f64) -> Self {
        Self {
            tick: 0,
            map: TempoMap::with_bpm(bpm),
        }
    }

    /// Current tick position.
    pub fn tick(&self) -> u32 {
        self.tick
    }

    /// Current tempo in microseconds per quarter.
    pub fn us_per_quarter(&self) -> u64 {
        self.map.tempo_at(self.tick).us_per_quarter
    }

    /// Current tempo in BPM (float — presentation only).
    pub fn bpm(&self) -> f64 {
        self.map.tempo_at(self.tick).bpm()
    }

    /// Advance the clock by `delta_ticks` ticks.
    ///
    /// Returns the new tick position.
    pub fn advance(&mut self, delta_ticks: u32) -> u32 {
        self.tick = self.tick.saturating_add(delta_ticks);
        self.tick
    }

    /// Set the clock to a specific tick. Must be >= current tick.
    ///
    /// Returns `Err(old_tick)` if the target is in the past (clock is monotonic).
    pub fn seek(&mut self, tick: u32) -> Result<u32, u32> {
        if tick < self.tick {
            return Err(self.tick);
        }
        self.tick = tick;
        Ok(self.tick)
    }

    /// Change the tempo at the current tick position.
    pub fn set_tempo(&mut self, us_per_quarter: u64) {
        let event = TempoEvent::new(self.tick, us_per_quarter);
        self.map.insert(event);
    }

    /// Change the tempo to a specific BPM at the current tick.
    pub fn set_bpm(&mut self, bpm: f64) {
        let event = TempoEvent::from_bpm(self.tick, bpm);
        self.map.insert(event);
    }

    /// Get a reference to the full tempo map.
    pub fn tempo_map(&self) -> &TempoMap {
        &self.map
    }

    /// Convert the current tick to microseconds since tick 0.
    pub fn current_us(&self) -> u64 {
        self.map.tick_to_us(self.tick)
    }

    /// Reset to tick 0 with the default tempo. Use carefully.
    pub fn reset(&mut self) {
        self.tick = 0;
        self.map = TempoMap::new();
    }
}

// ── Bar/Beat helpers ────────────────────────────────────────────────

/// Musical position: bar, beat, and sub-tick within the beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MusicalPosition {
    /// Zero-indexed bar number.
    pub bar: u32,
    /// Zero-indexed beat within the bar (0–3 in 4/4).
    pub beat: u32,
    /// Ticks within the current beat (0–95 at 96 PPQ).
    pub sub_tick: u32,
}

impl MusicalPosition {
    /// Convert a tick and time signature to a musical position.
    ///
    /// `beats_per_bar` is the numerator of the time signature (4 for 4/4).
    pub fn from_tick(tick: u32, beats_per_bar: u32) -> Self {
        let beats_per_bar = beats_per_bar.max(1);
        let ticks_per_bar = PPQ * beats_per_bar;
        let bar = tick / ticks_per_bar;
        let within_bar = tick % ticks_per_bar;
        let beat = within_bar / PPQ;
        let sub_tick = within_bar % PPQ;
        Self { bar, beat, sub_tick }
    }

    /// Convert back to a tick value.
    pub fn to_tick(&self, beats_per_bar: u32) -> u32 {
        let beats_per_bar = beats_per_bar.max(1);
        self.bar * PPQ * beats_per_bar + self.beat * PPQ + self.sub_tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_clock_starts_at_zero() {
        let clock = BeatClock::new();
        assert_eq!(clock.tick(), 0);
        assert_eq!(clock.bpm(), 120.0);
    }

    #[test]
    fn beat_clock_advances() {
        let mut clock = BeatClock::new();
        clock.advance(96);
        assert_eq!(clock.tick(), 96); // One quarter note
    }

    #[test]
    fn beat_clock_is_monotonic() {
        let mut clock = BeatClock::new();
        clock.advance(100);
        let result = clock.seek(50);
        assert!(result.is_err());
    }

    #[test]
    fn tempo_change_takes_effect() {
        let mut clock = BeatClock::new();
        assert!((clock.bpm() - 120.0).abs() < 0.1);

        clock.set_bpm(60.0);
        assert!((clock.bpm() - 60.0).abs() < 0.1);
    }

    #[test]
    fn tempo_map_tick_to_us_at_120bpm() {
        let map = TempoMap::new();
        // At 120 BPM (500_000 µs/quarter), 96 ticks = 500_000 µs = 0.5s
        assert_eq!(map.tick_to_us(96), 500_000);
        assert_eq!(map.tick_to_us(192), 1_000_000);
    }

    #[test]
    fn tempo_map_with_tempo_change() {
        let mut map = TempoMap::new();
        // Change to 60 BPM at tick 96 (after one beat at 120)
        map.insert(TempoEvent::from_bpm(96, 60.0));

        // First beat at 120 BPM = 500_000 µs
        let us_at_96 = map.tick_to_us(96);
        assert_eq!(us_at_96, 500_000);

        // Second beat at 60 BPM (1_000_000 µs/quarter)
        let us_at_192 = map.tick_to_us(192);
        assert_eq!(us_at_192, 500_000 + 1_000_000);
    }

    #[test]
    fn musical_position_round_trip() {
        let pos = MusicalPosition::from_tick(200, 4);
        assert_eq!(pos.bar, 0);
        assert_eq!(pos.beat, 2);
        assert_eq!(pos.sub_tick, 8);

        assert_eq!(pos.to_tick(4), 200);
    }

    #[test]
    fn musical_position_bar_2_beat_3() {
        // Bar 2, beat 3 = tick 2*4*96 + 3*96 = 768 + 288 = 1056
        let pos = MusicalPosition::from_tick(1056, 4);
        assert_eq!(pos.bar, 2);
        assert_eq!(pos.beat, 3);
    }

    #[test]
    fn tempo_event_bpm_conversion() {
        let event = TempoEvent::from_bpm(0, 120.0);
        assert!((event.bpm() - 120.0).abs() < 0.1);

        let event = TempoEvent::from_bpm(0, 90.0);
        assert!((event.bpm() - 90.0).abs() < 0.1);
    }

    #[test]
    fn us_to_tick_inverse() {
        let map = TempoMap::new();
        // 500_000 µs = 96 ticks at 120 BPM
        let tick = map.us_to_tick(500_000);
        assert_eq!(tick, 96);
    }

    #[test]
    fn clock_reset_returns_to_zero() {
        let mut clock = BeatClock::new();
        clock.advance(500);
        clock.reset();
        assert_eq!(clock.tick(), 0);
        assert!((clock.bpm() - 120.0).abs() < 0.1);
    }
}
