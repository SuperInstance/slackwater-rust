#![warn(clippy::all)]
#![deny(unsafe_code)]

//! # swmidi
//!
//! Standalone SWMIDI wire format codec — encode and decode the 8-byte
//! Tensor-MIDI event format without pulling in the full flux-core crate.
//!
//! This crate is the lightweight serialization layer for agents that
//! only need to speak the protocol (read/write packets) without the
//! full constraint engine.
//!
//! ## Binary format (SWMIDI-8)
//!
//! Each event packs into exactly **8 bytes**, little-endian:
//!
//! ```text
//! byte 0     status:     type(4 bits) | channel(4 bits)
//! byte 1     pitch:      action type, 0–127
//! byte 2     velocity:   weight / confidence, 0–127
//! byte 3     error_mask  (friction bitfield)
//! bytes 4–7  tick:       uint32, 96 PPQ on the shared BeatClock
//! ```
//!
//! ## Design
//!
//! - Fixed-size: every event is exactly 8 bytes. No variable-length encoding.
//! - Little-endian: matches x86/ARM native byte order.
//! - No allocation needed for encode/decode of single events.

use serde::{Deserialize, Serialize};

/// SWMIDI event type (4 bits, fits in the high nibble of byte 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EventType {
    NoteOn = 0,
    NoteOff = 1,
    ControlChange = 2,
    ProgramChange = 3,
    Meta = 4,
}

impl EventType {
    /// Convert to 4-bit nibble.
    #[inline]
    pub const fn to_nibble(self) -> u8 {
        self as u8 & 0x0F
    }

    /// Convert from nibble. Returns `None` for invalid codes.
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

/// A decoded SWMIDI event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SwmidiEvent {
    pub event_type: EventType,
    pub channel: u8,
    pub pitch: u8,
    pub velocity: u8,
    pub error_mask: u8,
    pub tick: u32,
}

/// Number of bytes in a packed SWMIDI event.
pub const PACKED_SIZE: usize = 8;

impl SwmidiEvent {
    /// Create a new event with all fields specified.
    pub const fn new(
        event_type: EventType,
        channel: u8,
        pitch: u8,
        velocity: u8,
        error_mask: u8,
        tick: u32,
    ) -> Self {
        Self { event_type, channel: channel & 0x0F, pitch: pitch & 0x7F, velocity: velocity & 0x7F, error_mask, tick }
    }

    /// Encode this event into 8 bytes.
    pub fn encode(&self) -> [u8; PACKED_SIZE] {
        let mut buf = [0u8; PACKED_SIZE];
        buf[0] = (self.event_type.to_nibble() << 4) | (self.channel & 0x0F);
        buf[1] = self.pitch & 0x7F;
        buf[2] = self.velocity & 0x7F;
        buf[3] = self.error_mask;
        buf[4..8].copy_from_slice(&self.tick.to_le_bytes());
        buf
    }

    /// Decode 8 bytes into an event.
    ///
    /// Returns `None` if the event type nibble is invalid.
    pub fn decode(buf: &[u8; PACKED_SIZE]) -> Option<Self> {
        let event_type = EventType::from_nibble(buf[0] >> 4)?;
        let channel = buf[0] & 0x0F;
        let pitch = buf[1] & 0x7F;
        let velocity = buf[2] & 0x7F;
        let error_mask = buf[3];
        let tick = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        Some(Self { event_type, channel, pitch, velocity, error_mask, tick })
    }

    /// Whether this event has zero error mask (flow state).
    pub fn is_flow(&self) -> bool {
        self.error_mask == 0
    }

    /// Whether this event carries friction (any error bit set).
    pub fn has_friction(&self) -> bool {
        self.error_mask != 0
    }
}

/// Error encountered while decoding an SWMIDI stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeError {
    /// Buffer too short to contain a full event.
    Truncated,
    /// Invalid event type nibble.
    InvalidEventType,
    /// Trailing bytes that don't form a complete event.
    TrailingBytes,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => write!(f, "buffer too short for a complete SWMIDI event"),
            Self::InvalidEventType => write!(f, "invalid event type nibble"),
            Self::TrailingBytes => write!(f, "trailing bytes after complete events"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// A stream of SWMIDI events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwmidiStream {
    events: Vec<SwmidiEvent>,
}

impl SwmidiStream {
    /// Create an empty stream.
    pub fn new() -> Self {
        Self::default()
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

    /// Get an event by index.
    pub fn get(&self, index: usize) -> Option<&SwmidiEvent> {
        self.events.get(index)
    }

    /// Iterate over events.
    pub fn iter(&self) -> impl Iterator<Item = &SwmidiEvent> {
        self.events.iter()
    }

    /// Sort events by tick (stable sort preserves insertion order for ties).
    pub fn sort_by_tick(&mut self) {
        self.events.sort_by_key(|e| e.tick);
    }

    /// Encode the entire stream to bytes.
    pub fn encode_all(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.events.len() * PACKED_SIZE);
        for event in &self.events {
            buf.extend_from_slice(&event.encode());
        }
        buf
    }

    /// Decode a byte buffer into a stream.
    ///
    /// Returns an error if the buffer length is not a multiple of 8,
    /// or if any event has an invalid type nibble.
    pub fn decode_all(buf: &[u8]) -> Result<Self, DecodeError> {
        if buf.len() % PACKED_SIZE != 0 {
            return if buf.is_empty() {
                Ok(Self::new())
            } else {
                Err(DecodeError::Truncated)
            };
        }

        let mut stream = Self::new();
        for chunk in buf.chunks_exact(PACKED_SIZE) {
            let arr: [u8; PACKED_SIZE] = chunk.try_into().unwrap();
            let event = SwmidiEvent::decode(&arr).ok_or(DecodeError::InvalidEventType)?;
            stream.push(event);
        }
        Ok(stream)
    }

    /// Filter events to those within a tick range [start, end).
    pub fn in_tick_range(&self, start: u32, end: u32) -> impl Iterator<Item = &SwmidiEvent> {
        self.events.iter().filter(move |e| e.tick >= start && e.tick < end)
    }

    /// Count events with friction (error_mask != 0).
    pub fn friction_count(&self) -> usize {
        self.events.iter().filter(|e| e.has_friction()).count()
    }

    /// Count events in flow state (error_mask == 0).
    pub fn flow_count(&self) -> usize {
        self.events.iter().filter(|e| e.is_flow()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip() {
        let event = SwmidiEvent::new(EventType::NoteOn, 3, 60, 100, 0, 192);
        let encoded = event.encode();
        let decoded = SwmidiEvent::decode(&encoded).unwrap();
        assert_eq!(event, decoded);
    }

    #[test]
    fn encode_decode_all_types() {
        let types = [
            EventType::NoteOn,
            EventType::NoteOff,
            EventType::ControlChange,
            EventType::ProgramChange,
            EventType::Meta,
        ];
        for &ty in &types {
            let event = SwmidiEvent::new(ty, 0, 60, 100, 0, 0);
            let encoded = event.encode();
            let decoded = SwmidiEvent::decode(&encoded).unwrap();
            assert_eq!(event.event_type, decoded.event_type);
        }
    }

    #[test]
    fn invalid_event_type_returns_none() {
        // Manually craft an invalid event type nibble (5)
        let mut buf = [0u8; PACKED_SIZE];
        buf[0] = 0x50; // type = 5, channel = 0
        assert!(SwmidiEvent::decode(&buf).is_none());
    }

    #[test]
    fn channel_is_masked() {
        let event = SwmidiEvent::new(EventType::NoteOn, 20, 0, 0, 0, 0);
        // channel should be masked to 4 bits
        assert_eq!(event.channel, 20 & 0x0F);
    }

    #[test]
    fn pitch_is_masked() {
        let event = SwmidiEvent::new(EventType::NoteOn, 0, 200, 0, 0, 0);
        assert_eq!(event.pitch, 200 & 0x7F);
    }

    #[test]
    fn flow_and_friction() {
        let flow_event = SwmidiEvent::new(EventType::NoteOn, 0, 0, 0, 0, 0);
        let friction_event = SwmidiEvent::new(EventType::NoteOn, 0, 0, 0, 0b0000_0001, 0);

        assert!(flow_event.is_flow());
        assert!(!flow_event.has_friction());
        assert!(!friction_event.is_flow());
        assert!(friction_event.has_friction());
    }

    #[test]
    fn stream_encode_decode_round_trip() {
        let mut stream = SwmidiStream::new();
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
        stream.push(SwmidiEvent::new(EventType::NoteOff, 0, 60, 0, 0, 96));
        stream.push(SwmidiEvent::new(EventType::ControlChange, 1, 7, 80, 0, 48));

        let encoded = stream.encode_all();
        let decoded = SwmidiStream::decode_all(&encoded).unwrap();

        assert_eq!(stream.len(), decoded.len());
        for (a, b) in stream.iter().zip(decoded.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn stream_decode_truncated() {
        let buf = [0u8; 7]; // Not a multiple of 8
        let result = SwmidiStream::decode_all(&buf);
        assert!(matches!(result, Err(DecodeError::Truncated)));
    }

    #[test]
    fn stream_decode_empty() {
        let stream = SwmidiStream::decode_all(&[]).unwrap();
        assert!(stream.is_empty());
    }

    #[test]
    fn stream_sort_by_tick() {
        let mut stream = SwmidiStream::new();
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 192));
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 64, 100, 0, 0));
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 67, 100, 0, 96));

        stream.sort_by_tick();
        assert_eq!(stream.get(0).unwrap().tick, 0);
        assert_eq!(stream.get(1).unwrap().tick, 96);
        assert_eq!(stream.get(2).unwrap().tick, 192);
    }

    #[test]
    fn stream_tick_range_filter() {
        let mut stream = SwmidiStream::new();
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 64, 100, 0, 96));
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 67, 100, 0, 192));
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 72, 100, 0, 288));

        let in_range: Vec<_> = stream.in_tick_range(96, 288).collect();
        assert_eq!(in_range.len(), 2);
    }

    #[test]
    fn stream_flow_friction_counts() {
        let mut stream = SwmidiStream::new();
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 64, 100, 0b0000_0001, 96));
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 67, 100, 0, 192));
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 72, 100, 0b0000_0010, 288));

        assert_eq!(stream.flow_count(), 2);
        assert_eq!(stream.friction_count(), 2);
    }

    #[test]
    fn tick_encodes_little_endian() {
        let event = SwmidiEvent::new(EventType::Meta, 0, 0, 0, 0, 0x01020304);
        let encoded = event.encode();
        // Little-endian: 04 03 02 01
        assert_eq!(&encoded[4..8], &[0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn decode_error_display() {
        assert!(format!("{}", DecodeError::Truncated).contains("too short"));
        assert!(format!("{}", DecodeError::InvalidEventType).contains("invalid"));
        assert!(format!("{}", DecodeError::TrailingBytes).contains("trailing"));
    }
}
