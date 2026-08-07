#![warn(clippy::all)]
#![deny(unsafe_code)]

//! # perception-core
//!
//! Layer 7: Multi-track MIDI encoding and convergence detection.
//!
//! When multiple agents (tracks) are building simultaneously, Perception
//! watches for **convergence** — moments where independent agents arrive
//! at the same (or compatible) build positions at the same time. Convergence
//! is the signal that the fleet is self-organizing. Divergence is the signal
//! that someone needs to recalibrate.
//!
//! ## Core types
//!
//! - [`Track`] — A single agent's stream of events on a channel.
//! - [`MultiTrack`] — A collection of tracks with convergence analysis.
//! - [`Convergence`] — A detected convergence point between tracks.
//! - [`ConvergenceReport`] — Summary of convergence across the fleet.
//!
//! ## Design
//!
//! - Convergence is spatial + temporal: same position AND same tick.
//! - Near-misses (within a tolerance) count as weak convergence.
//! - The report tracks both convergence count and divergence count.

use serde::{Deserialize, Serialize};

/// A single track (agent channel) with its events.
///
/// Each track belongs to one agent. The `agent_id` identifies which agent
/// (e.g., "kimi", "claude", "glm-5"). Events are SWMIDI-style: tick + position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    /// The agent that owns this track.
    pub agent_id: String,
    /// The channel number (0–15).
    pub channel: u8,
    /// Events: (tick, position) pairs, sorted by tick.
    pub events: Vec<TrackEvent>,
}

/// A single event in a track: a positioned action at a tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrackEvent {
    /// Tick position on the shared BeatClock.
    pub tick: u32,
    /// Spatial position encoded as a single integer hash.
    /// This is the Eisenstein (a, b) packed as (a << 16) | (b & 0xFFFF).
    pub position_packed: u32,
}

impl TrackEvent {
    /// Create a new track event from Eisenstein coordinates.
    pub fn from_eisenstein(tick: u32, a: i32, b: i32) -> Self {
        let position_packed = pack_eisenstein(a, b);
        Self {
            tick,
            position_packed,
        }
    }

    /// Unpack to Eisenstein (a, b) coordinates.
    pub fn to_eisenstein(&self) -> (i32, i32) {
        unpack_eisenstein(self.position_packed)
    }
}

/// Pack Eisenstein (a, b) into a single u32.
pub fn pack_eisenstein(a: i32, b: i32) -> u32 {
    let a_u = (a as i16) as u16;
    let b_u = (b as i16) as u16;
    ((a_u as u32) << 16) | (b_u as u32)
}

/// Unpack Eisenstein (a, b) from a u32.
pub fn unpack_eisenstein(packed: u32) -> (i32, i32) {
    let a_u = (packed >> 16) as u16;
    let b_u = (packed & 0xFFFF) as u16;
    (a_u as i16 as i32, b_u as i16 as i32)
}

impl Track {
    /// Create a new track for an agent.
    pub fn new(agent_id: impl Into<String>, channel: u8) -> Self {
        Self {
            agent_id: agent_id.into(),
            channel: channel & 0x0F,
            events: Vec::new(),
        }
    }

    /// Add an event to this track.
    pub fn add_event(&mut self, event: TrackEvent) {
        self.events.push(event);
        self.events.sort_by_key(|e| e.tick);
    }

    /// Number of events in the track.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the track has any events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get events at a specific tick.
    pub fn events_at_tick(&self, tick: u32) -> impl Iterator<Item = &TrackEvent> {
        self.events.iter().filter(move |e| e.tick == tick)
    }

    /// Get events near a position (within tolerance).
    pub fn events_near_position(&self, position_packed: u32, tolerance: u32) -> Vec<&TrackEvent> {
        let (ta, tb) = unpack_eisenstein(position_packed);
        self.events
            .iter()
            .filter(|e| {
                let (ea, eb) = e.to_eisenstein();
                let da = (ea - ta).abs();
                let db = (eb - tb).abs();
                da <= tolerance as i32 && db <= tolerance as i32
            })
            .collect()
    }
}

/// A detected convergence point between tracks.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Convergence {
    /// The tick where convergence was detected.
    pub tick: u32,
    /// The agents that converged.
    pub agents: Vec<String>,
    /// The position they converged on (packed Eisenstein).
    pub position_packed: u32,
    /// Whether this is exact (same position) or weak (within tolerance).
    pub strength: ConvergenceStrength,
}

/// How strong a convergence is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConvergenceStrength {
    /// Exact position match — all agents placed at the same (a, b).
    Exact,
    /// Within tolerance — agents are close but not identical.
    Weak,
}

/// A report on fleet convergence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceReport {
    /// All convergence points found, sorted by tick.
    pub convergences: Vec<Convergence>,
    /// Total number of exact convergences.
    pub exact_count: usize,
    /// Total number of weak convergences.
    pub weak_count: usize,
    /// Total number of divergences (agents at different positions at the same tick).
    pub divergence_count: usize,
    /// Number of agents analyzed.
    pub agent_count: usize,
}

impl ConvergenceReport {
    /// Total convergence points.
    pub fn total_convergences(&self) -> usize {
        self.convergences.len()
    }

    /// Convergence ratio: convergences / (convergences + divergences).
    ///
    /// Returns 0.0 if no events at all.
    pub fn convergence_ratio(&self) -> f64 {
        let total = self.exact_count + self.weak_count + self.divergence_count;
        if total == 0 {
            return 0.0;
        }
        (self.exact_count + self.weak_count) as f64 / total as f64
    }
}

/// A collection of tracks with convergence analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiTrack {
    tracks: Vec<Track>,
}

impl MultiTrack {
    /// Create an empty multi-track.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a track.
    pub fn add_track(&mut self, track: Track) {
        self.tracks.push(track);
    }

    /// Get all tracks.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Number of tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Get all unique ticks across all tracks, sorted.
    fn all_ticks(&self) -> Vec<u32> {
        let mut ticks: Vec<u32> = self
            .tracks
            .iter()
            .flat_map(|t| t.events.iter().map(|e| e.tick))
            .collect();
        ticks.sort();
        ticks.dedup();
        ticks
    }

    /// Detect convergence across all tracks.
    ///
    /// `tolerance` is the maximum Manhattan distance on the Eisenstein lattice
    /// for weak convergence.
    pub fn analyze(&self, tolerance: u32) -> ConvergenceReport {
        let mut convergences = Vec::new();
        let mut exact_count = 0;
        let mut weak_count = 0;
        let mut divergence_count = 0;

        for tick in self.all_ticks() {
            // Collect all (agent_id, position) pairs at this tick
            let mut placements: Vec<(&str, u32)> = Vec::new();

            for track in &self.tracks {
                for event in track.events_at_tick(tick) {
                    placements.push((&track.agent_id, event.position_packed));
                }
            }

            if placements.len() < 2 {
                continue; // Need at least 2 agents to converge/diverge
            }

            // Group by exact position first
            let mut exact_groups: std::collections::HashMap<u32, Vec<&str>> =
                std::collections::HashMap::new();
            for (agent, pos) in &placements {
                exact_groups.entry(*pos).or_default().push(agent);
            }

            if exact_groups.len() == 1 {
                // All agents at the exact same position
                let (pos, agents) = exact_groups.into_iter().next().unwrap();
                convergences.push(Convergence {
                    tick,
                    agents: agents.iter().map(|s| s.to_string()).collect(),
                    position_packed: pos,
                    strength: ConvergenceStrength::Exact,
                });
                exact_count += 1;
            } else if exact_groups.len() <= 4 && tolerance > 0 {
                // Check if all positions are within tolerance of each other
                let positions: Vec<u32> = exact_groups.keys().copied().collect();
                let all_close = positions.iter().enumerate().all(|(i, &p)| {
                    let (pa, pb) = unpack_eisenstein(p);
                    positions[i + 1..].iter().all(|&q| {
                        let (qa, qb) = unpack_eisenstein(q);
                        (pa - qa).abs() <= tolerance as i32 && (pb - qb).abs() <= tolerance as i32
                    })
                });

                if all_close {
                    let all_agents: Vec<String> = exact_groups
                        .values()
                        .flat_map(|agents| agents.iter().map(|s| s.to_string()))
                        .collect();
                    convergences.push(Convergence {
                        tick,
                        agents: all_agents,
                        position_packed: positions[0],
                        strength: ConvergenceStrength::Weak,
                    });
                    weak_count += 1;
                } else {
                    divergence_count += 1;
                }
            } else {
                divergence_count += 1;
            }
        }

        convergences.sort_by_key(|c| c.tick);

        ConvergenceReport {
            convergences,
            exact_count,
            weak_count,
            divergence_count,
            agent_count: self.tracks.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        for &(a, b) in &[(0, 0), (1, 2), (-1, -2), (100, 200), (-30000, 30000)] {
            let packed = pack_eisenstein(a, b);
            let (ra, rb) = unpack_eisenstein(packed);
            assert_eq!(a, ra, "a mismatch for ({}, {})", a, b);
            assert_eq!(b, rb, "b mismatch for ({}, {})", a, b);
        }
    }

    #[test]
    fn track_add_and_sort() {
        let mut track = Track::new("alpha", 0);
        track.add_event(TrackEvent::from_eisenstein(192, 1, 0));
        track.add_event(TrackEvent::from_eisenstein(0, 0, 0));
        track.add_event(TrackEvent::from_eisenstein(96, 0, 1));

        assert_eq!(track.events[0].tick, 0);
        assert_eq!(track.events[1].tick, 96);
        assert_eq!(track.events[2].tick, 192);
    }

    #[test]
    fn track_events_at_tick() {
        let mut track = Track::new("beta", 1);
        track.add_event(TrackEvent::from_eisenstein(0, 1, 1));
        track.add_event(TrackEvent::from_eisenstein(96, 2, 2));
        track.add_event(TrackEvent::from_eisenstein(96, 3, 3)); // Same tick, different position

        let at_96: Vec<_> = track.events_at_tick(96).collect();
        assert_eq!(at_96.len(), 2);
    }

    #[test]
    fn multitrack_exact_convergence() {
        let mut mt = MultiTrack::new();

        let mut t1 = Track::new("alpha", 0);
        t1.add_event(TrackEvent::from_eisenstein(0, 5, 5));
        t1.add_event(TrackEvent::from_eisenstein(96, 6, 6));

        let mut t2 = Track::new("beta", 1);
        t2.add_event(TrackEvent::from_eisenstein(0, 5, 5)); // Same position!
        t2.add_event(TrackEvent::from_eisenstein(96, 10, 10)); // Different position

        mt.add_track(t1);
        mt.add_track(t2);

        let report = mt.analyze(0);

        assert_eq!(report.exact_count, 1); // tick 0: both at (5,5)
        assert_eq!(report.weak_count, 0);
        assert_eq!(report.divergence_count, 1); // tick 96: different positions
    }

    #[test]
    fn multitrack_weak_convergence() {
        let mut mt = MultiTrack::new();

        let mut t1 = Track::new("alpha", 0);
        t1.add_event(TrackEvent::from_eisenstein(0, 5, 5));

        let mut t2 = Track::new("beta", 1);
        t2.add_event(TrackEvent::from_eisenstein(0, 6, 6)); // Close but not exact

        mt.add_track(t1);
        mt.add_track(t2);

        let report = mt.analyze(2); // tolerance of 2

        assert_eq!(report.exact_count, 0);
        assert_eq!(report.weak_count, 1);
        assert_eq!(report.divergence_count, 0);
    }

    #[test]
    fn multitrack_no_convergence_single_agent() {
        let mut mt = MultiTrack::new();
        let mut t1 = Track::new("solo", 0);
        t1.add_event(TrackEvent::from_eisenstein(0, 0, 0));
        mt.add_track(t1);

        let report = mt.analyze(0);
        assert_eq!(report.total_convergences(), 0); // Only one agent
    }

    #[test]
    fn convergence_ratio() {
        let report = ConvergenceReport {
            convergences: vec![],
            exact_count: 3,
            weak_count: 2,
            divergence_count: 5,
            agent_count: 3,
        };
        // 5 convergences / 10 total = 0.5
        assert!((report.convergence_ratio() - 0.5).abs() < 0.001);
    }

    #[test]
    fn convergence_ratio_empty() {
        let report = ConvergenceReport {
            convergences: vec![],
            exact_count: 0,
            weak_count: 0,
            divergence_count: 0,
            agent_count: 0,
        };
        assert_eq!(report.convergence_ratio(), 0.0);
    }

    #[test]
    fn events_near_position() {
        let mut track = Track::new("gamma", 2);
        track.add_event(TrackEvent::from_eisenstein(0, 10, 10));
        track.add_event(TrackEvent::from_eisenstein(96, 12, 12));
        track.add_event(TrackEvent::from_eisenstein(192, 50, 50));

        let near = track.events_near_position(pack_eisenstein(11, 11), 2);
        assert_eq!(near.len(), 2); // (10,10) and (12,12) both within tolerance 2
    }

    #[test]
    fn three_agent_convergence() {
        let mut mt = MultiTrack::new();

        for (i, name) in ["alpha", "beta", "gamma"].iter().enumerate() {
            let mut t = Track::new(*name, i as u8);
            t.add_event(TrackEvent::from_eisenstein(0, 3, 3));
            mt.add_track(t);
        }

        let report = mt.analyze(0);
        assert_eq!(report.exact_count, 1);
        assert_eq!(report.convergences[0].agents.len(), 3);
    }

    #[test]
    fn track_event_eisenstein_round_trip() {
        let event = TrackEvent::from_eisenstein(42, 7, -3);
        let (a, b) = event.to_eisenstein();
        assert_eq!(a, 7);
        assert_eq!(b, -3);
        assert_eq!(event.tick, 42);
    }
}
