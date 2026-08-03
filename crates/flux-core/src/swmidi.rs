//! SWMIDI — Tensor-MIDI event packing.
//!
//! The one wire format for everything in the Slackwater stack: build
//! commands, model outputs, player actions, tide changes, flow measurements.
//!
//! ## Binary format (SWMIDI-8)
//!
//! Each event packs into exactly **8 bytes**, little-endian:
//!
//! ```text
//! byte 0     status:     type(4 bits) | channel(4 bits)
//! byte 1     pitch:      action type, 0–127
//! byte 2     velocity:   weight / confidence, 0–127
//! byte 3     error_mask  (Layer 1)
//! bytes 4–7  tick:       uint32, 96 PPQ on the shared BeatClock
//! ```
//!
//! CC (control change) pairs carry spatial payload on the same channel
//! at the same tick — they are stored in the stream, not in the 8-byte
//! packed event.
//!
//! ## Migration
//!
//! During Phase 2→3, JSON envelopes carry the same fields. The binary
//! packing is a serialization flip, not a redesign — the shape is fixed.

use crate::error_mask::ErrorMask;
use crate::exact::{Channel, Pitch, Tick, Velocity};
use core::fmt;
use serde::{Deserialize, Serialize};

// ── Event type nibble (4 bits, 0–15) ────────────────────────────────

/// Type of SWMIDI event. Fits in 4 bits (the high nibble of byte 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EventType {
    /// Note on — build action, placement, activation.
    NoteOn = 0,
    /// Note off — release, deactivate, end.
    NoteOff = 1,
    /// Control change — parameters, spatial payload.
    ControlChange = 2,
    /// Program change — pipeline stage transition.
    ProgramChange = 3,
    /// Meta — tempo change, prediction, convergence, end-of-track.
    Meta = 4,
}

impl EventType {
    /// Convert to the 4-bit type code.
    #[inline]
    pub const fn to_nibble(self) -> u8 {
        self as u8 & 0x0F
    }

    /// Convert from a 4-bit type code.
    ///
    /// Returns `None` for invalid codes (5–15).
    pub fn from_nibble(nibble: u8) -> Option<Self> {
        match nibble & 0x0F {
            0 => Some(Self::NoteOn),
            1 => Some(Self::NoteOff),
            2 => Some(Self::ControlChange),
            3 => Some(Self::ProgramChange),
            4 => Some(Self::Meta),
            _ => None,
        }
    }
}

/// Meta event subtype (carried in the pitch field for META events).
///
/// Pitch values from the Grand Plan §2 pitch map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetaType {
    /// Pitch 81 — tempo change or T-Minus prediction event.
    TempoChange = 81,
    /// Pitch 83 — Φ (phi) reading from the Governor.
    PhiReading = 83,
    /// Pitch 84 — convergence event (tracks aligned).
    Convergence = 84,
    /// Pitch 0 — end of track / stream terminator.
    EndOfTrack = 0,
}

impl MetaType {
    /// Convert to a pitch value.
    #[inline]
    pub const fn to_pitch(self) -> Pitch {
        self as Pitch
    }
}

// ── Decode error ────────────────────────────────────────────────────

/// Errors that can occur during SWMIDI decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Input buffer is too short.
    BufferTooShort { need: usize, got: usize },
    /// Invalid event type nibble.
    InvalidEventType(u8),
    /// Invalid JSON structure.
    Json(String),
    /// CC pairs data has odd length.
    OddCcData(usize),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooShort { need, got } => {
                write!(f, "buffer too short: need {need} bytes, got {got}")
            }
            Self::InvalidEventType(n) => write!(f, "invalid event type nibble: {n}"),
            Self::Json(msg) => write!(f, "JSON decode error: {msg}"),
            Self::OddCcData(len) => write!(f, "CC pairs data has odd length: {len}"),
        }
    }
}

impl std::error::Error for DecodeError {}

// ── SwmidiEvent ─────────────────────────────────────────────────────

/// A single SWMIDI event — a 4-dimensional point in Tensor-MIDI space.
///
/// Packed binary representation is exactly 8 bytes (see module docs).
/// CC pairs are carried separately in the stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwmidiEvent {
    pub event_type: EventType,
    pub channel: Channel,
    pub pitch: Pitch,
    pub velocity: Velocity,
    pub tick: Tick,
    pub error_mask: ErrorMask,
    /// Control change pairs (controller, value). Only present for CC events
    /// or events carrying spatial payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cc: Option<Vec<(u8, u8)>>,
}

impl SwmidiEvent {
    /// Create a new minimal event.
    pub fn new(
        event_type: EventType,
        channel: Channel,
        pitch: Pitch,
        velocity: Velocity,
        tick: Tick,
    ) -> Self {
        Self {
            event_type,
            channel,
            pitch,
            velocity,
            tick,
            error_mask: ErrorMask::FLOW,
            cc: None,
        }
    }

    /// Create a note-on event.
    pub fn note_on(channel: Channel, pitch: Pitch, velocity: Velocity, tick: Tick) -> Self {
        Self::new(EventType::NoteOn, channel, pitch, velocity, tick)
    }

    /// Create a note-off event.
    pub fn note_off(channel: Channel, pitch: Pitch, tick: Tick) -> Self {
        Self::new(EventType::NoteOff, channel, pitch, 0, tick)
    }

    /// Attach CC pairs to this event.
    pub fn with_cc(mut self, cc: Vec<(u8, u8)>) -> Self {
        self.cc = if cc.is_empty() { None } else { Some(cc) };
        self
    }

    /// Attach an error mask to this event.
    pub fn with_mask(mut self, mask: ErrorMask) -> Self {
        self.error_mask = mask;
        self
    }

    /// Pack this event into exactly 8 bytes (little-endian).
    ///
    /// CC pairs are NOT included in the packed representation — they
    /// travel as separate events in the stream.
    pub fn pack(&self) -> [u8; 8] {
        let status = (self.event_type.to_nibble() << 4) | (self.channel & 0x0F);
        let tick_bytes = self.tick.to_le_bytes();
        [
            status,
            self.pitch,
            self.velocity,
            self.error_mask.bits(),
            tick_bytes[0],
            tick_bytes[1],
            tick_bytes[2],
            tick_bytes[3],
        ]
    }

    /// Unpack an 8-byte binary representation into an event.
    ///
    /// Returns `Err` if the type nibble is invalid.
    pub fn unpack(data: &[u8; 8]) -> Result<Self, DecodeError> {
        let status = data[0];
        let type_nibble = (status >> 4) & 0x0F;
        let channel = status & 0x0F;

        let event_type = EventType::from_nibble(type_nibble)
            .ok_or(DecodeError::InvalidEventType(type_nibble))?;

        let pitch = data[1];
        let velocity = data[2];
        let error_mask = ErrorMask::from_bits(data[3]);
        let tick = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        Ok(Self {
            event_type,
            channel,
            pitch,
            velocity,
            tick,
            error_mask,
            cc: None,
        })
    }
}

// ── SwmidiStream ────────────────────────────────────────────────────

/// A stream of SWMIDI events — the wire format for builds, actions, and scores.
///
/// Supports binary packing (8 bytes/event for compact transport) and
/// JSON serialization (for the Phase 2 migration period).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwmidiStream {
    events: Vec<SwmidiEvent>,
}

impl SwmidiStream {
    /// Create an empty stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a stream with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
        }
    }

    /// Push an event onto the stream.
    pub fn push(&mut self, event: SwmidiEvent) {
        self.events.push(event);
    }

    /// Number of events in the stream.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the stream is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get all events at a specific tick.
    pub fn at_tick(&self, tick: Tick) -> Vec<&SwmidiEvent> {
        self.events.iter().filter(|e| e.tick == tick).collect()
    }

    /// Get all events in a tick range [start, end).
    pub fn in_range(&self, start: Tick, end: Tick) -> Vec<&SwmidiEvent> {
        self.events
            .iter()
            .filter(|e| e.tick >= start && e.tick < end)
            .collect()
    }

    /// Get a reference to all events.
    pub fn events(&self) -> &[SwmidiEvent] {
        &self.events
    }

    /// Sort events by tick (stable sort, preserves insertion order for same-tick events).
    pub fn sort_by_tick(&mut self) {
        self.events.sort_by_key(|e| e.tick);
    }

    /// Pack the entire stream as compact binary.
    ///
    /// Format: 4-byte event count (LE u32) + N × 8-byte events.
    /// For events with CC pairs, CC data follows in a separate section:
    /// after the event block, a 4-byte CC record count (LE u32),
    /// then CC records: [u32 event_index, u8 cc_count, cc_count × (u8, u8)].
    ///
    /// A 100-event build with no CC payload = 4 + 800 = 804 bytes.
    pub fn pack_binary(&self) -> Vec<u8> {
        // Event block
        let mut buf = Vec::with_capacity(4 + self.events.len() * 8);
        buf.extend_from_slice(&(self.events.len() as u32).to_le_bytes());
        for event in &self.events {
            buf.extend_from_slice(&event.pack());
        }

        // CC block — only for events that have CC pairs
        let cc_events: Vec<(usize, &[(u8, u8)])> = self
            .events
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.cc.as_deref().map(|cc| (i, cc)))
            .collect();

        buf.extend_from_slice(&(cc_events.len() as u32).to_le_bytes());
        for (event_idx, cc_pairs) in cc_events {
            buf.extend_from_slice(&(event_idx as u32).to_le_bytes());
            buf.push(cc_pairs.len() as u8);
            for &(controller, value) in cc_pairs {
                buf.push(controller);
                buf.push(value);
            }
        }

        buf
    }

    /// Unpack a binary stream produced by [`pack_binary`](Self::pack_binary).
    pub fn unpack_binary(data: &[u8]) -> Result<Self, DecodeError> {
        if data.len() < 4 {
            return Err(DecodeError::BufferTooShort {
                need: 4,
                got: data.len(),
            });
        }

        let event_count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let events_end = 4 + event_count * 8;

        if data.len() < events_end {
            return Err(DecodeError::BufferTooShort {
                need: events_end,
                got: data.len(),
            });
        }

        let mut events: Vec<SwmidiEvent> = Vec::with_capacity(event_count);
        for i in 0..event_count {
            let offset = 4 + i * 8;
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&data[offset..offset + 8]);
            events.push(SwmidiEvent::unpack(&chunk)?);
        }

        // CC block
        let mut pos = events_end;
        if data.len() >= pos + 4 {
            let cc_count = u32::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
            ]) as usize;
            pos += 4;

            for _ in 0..cc_count {
                if data.len() < pos + 5 {
                    return Err(DecodeError::BufferTooShort {
                        need: pos + 5,
                        got: data.len(),
                    });
                }
                let event_idx =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                        as usize;
                let pair_count = data[pos + 4] as usize;
                pos += 5;

                if event_idx >= events.len() {
                    return Err(DecodeError::InvalidEventType(0xFF)); // misuse, but distinct
                }

                if data.len() < pos + pair_count * 2 {
                    return Err(DecodeError::BufferTooShort {
                        need: pos + pair_count * 2,
                        got: data.len(),
                    });
                }

                let mut pairs = Vec::with_capacity(pair_count);
                for j in 0..pair_count {
                    let controller = data[pos + j * 2];
                    let value = data[pos + j * 2 + 1];
                    pairs.push((controller, value));
                }
                events[event_idx].cc = Some(pairs);
                pos += pair_count * 2;
            }
        }

        Ok(Self { events })
    }

    /// Serialize to a JSON envelope string.
    ///
    /// Used during the Phase 2 migration period. Same field shape as
    /// the binary format — the binary flip in Phase 3 changes
    /// serialization, not semantics.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Deserialize from a JSON envelope string.
    pub fn from_json(json: &str) -> Result<Self, DecodeError> {
        serde_json::from_str(json).map_err(|e| DecodeError::Json(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_creation() {
        let event = SwmidiEvent::note_on(0, 60, 96, 43200);
        assert_eq!(event.event_type, EventType::NoteOn);
        assert_eq!(event.channel, 0);
        assert_eq!(event.pitch, 60);
        assert_eq!(event.velocity, 96);
        assert_eq!(event.tick, 43200);
        assert!(event.error_mask.is_flow());
        assert!(event.cc.is_none());
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let event = SwmidiEvent::new(
            EventType::NoteOn,
            3,
            60,
            96,
            43200,
        )
        .with_mask(ErrorMask::SPATIAL | ErrorMask::SAFETY)
        .with_cc(vec![(16, 64), (17, 32)]);

        let packed = event.pack();
        let unpacked = SwmidiEvent::unpack(&packed).unwrap();

        assert_eq!(unpacked.event_type, event.event_type);
        assert_eq!(unpacked.channel, event.channel);
        assert_eq!(unpacked.pitch, event.pitch);
        assert_eq!(unpacked.velocity, event.velocity);
        assert_eq!(unpacked.tick, event.tick);
        assert_eq!(unpacked.error_mask, event.error_mask);
        // CC pairs are not stored in the 8-byte packed form
        assert!(unpacked.cc.is_none());
    }

    #[test]
    fn pack_format_is_8_bytes() {
        let event = SwmidiEvent::note_on(0, 60, 127, 0xFFFFFFFF);
        let packed = event.pack();
        assert_eq!(packed.len(), 8);
        // Verify the format: type|channel in byte 0, pitch in byte 1, etc.
        assert_eq!(packed[0], 0x00); // NoteOn(0) << 4 | channel(0)
        assert_eq!(packed[1], 60);
        assert_eq!(packed[2], 127);
        assert_eq!(packed[3], 0x00); // FLOW mask
        // Tick 0xFFFFFFFF in LE
        assert_eq!(&packed[4..8], &[0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn all_event_types_pack_unpack() {
        for event_type in [
            EventType::NoteOn,
            EventType::NoteOff,
            EventType::ControlChange,
            EventType::ProgramChange,
            EventType::Meta,
        ] {
            let event = SwmidiEvent::new(event_type, 15, 127, 127, 99999);
            let packed = event.pack();
            let unpacked = SwmidiEvent::unpack(&packed).unwrap();
            assert_eq!(unpacked.event_type, event_type);
        }
    }

    #[test]
    fn invalid_type_nibble_fails() {
        // Manually craft an invalid type nibble
        let data = [0xF0, 0, 0, 0, 0, 0, 0, 0]; // type nibble = 0xF (invalid)
        let result = SwmidiEvent::unpack(&data);
        assert!(result.is_err());
    }

    #[test]
    fn stream_pack_unpack_binary_roundtrip() {
        let mut stream = SwmidiStream::new();
        for i in 0..50 {
            stream.push(
                SwmidiEvent::note_on(0, 48 + i as u8 / 10, 80 + i as u8 / 5, i * 96)
                    .with_cc(vec![(16, 64 + i as u8), (17, 32)]),
            );
        }

        let packed = stream.pack_binary();
        let unpacked = SwmidiStream::unpack_binary(&packed).unwrap();

        assert_eq!(unpacked.len(), stream.len());
        for (orig, unpacked_evt) in stream.events().iter().zip(unpacked.events().iter()) {
            assert_eq!(orig.event_type, unpacked_evt.event_type);
            assert_eq!(orig.channel, unpacked_evt.channel);
            assert_eq!(orig.pitch, unpacked_evt.pitch);
            assert_eq!(orig.velocity, unpacked_evt.velocity);
            assert_eq!(orig.tick, unpacked_evt.tick);
            assert_eq!(orig.error_mask, unpacked_evt.error_mask);
            assert_eq!(orig.cc, unpacked_evt.cc);
        }
    }

    #[test]
    fn stream_json_roundtrip() {
        let mut stream = SwmidiStream::new();
        stream.push(SwmidiEvent::note_on(0, 60, 96, 43200));
        stream.push(
            SwmidiEvent::note_on(9, 36, 120, 43296)
                .with_mask(ErrorMask::TEMPORAL),
        );

        let json = stream.to_json();
        let unpacked = SwmidiStream::from_json(&json).unwrap();

        assert_eq!(unpacked.len(), 2);
        assert_eq!(unpacked.events()[0].pitch, 60);
        assert_eq!(unpacked.events()[1].channel, 9);
    }

    #[test]
    fn tick_queries() {
        let mut stream = SwmidiStream::new();
        stream.push(SwmidiEvent::note_on(0, 60, 96, 100));
        stream.push(SwmidiEvent::note_on(1, 64, 80, 100));
        stream.push(SwmidiEvent::note_on(0, 67, 88, 200));
        stream.push(SwmidiEvent::note_off(0, 60, 300));

        let at_100 = stream.at_tick(100);
        assert_eq!(at_100.len(), 2);

        let range = stream.in_range(100, 200);
        assert_eq!(range.len(), 2); // 100, 100 (not 200 — exclusive end)

        let range2 = stream.in_range(100, 301);
        assert_eq!(range2.len(), 4); // all events
    }

    #[test]
    fn hundred_part_build_size() {
        // From the Grand Plan: a 100-part build is ~800 bytes of events
        let mut stream = SwmidiStream::new();
        for i in 0..100 {
            stream.push(SwmidiEvent::note_on(
                0,
                48 + (i as u8 % 12),
                64 + (i as u8 % 64),
                i as u32 * 12,
            ));
        }

        let packed = stream.pack_binary();
        // 4 (count) + 100 × 8 (events) + 4 (cc count = 0) = 808 bytes
        assert_eq!(packed.len(), 4 + 800 + 4);
        assert!(packed.len() < 900); // Well under 1KB
    }
}
