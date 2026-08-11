//! Integration tests for swmidi
//!
//! Tests binary format edge cases, stream operations, and wire compatibility.

use swmidi::{DecodeError, EventType, SwmidiEvent, SwmidiStream, PACKED_SIZE};

// ════════════════════════════════════════════════════════════════════
// BINARY FORMAT EDGE CASES
// ════════════════════════════════════════════════════════════════════

#[test]
fn encode_all_event_types_round_trip() {
    let types = [
        EventType::NoteOn,
        EventType::NoteOff,
        EventType::ControlChange,
        EventType::ProgramChange,
        EventType::Meta,
    ];
    for &ty in &types {
        let event = SwmidiEvent::new(ty, 7, 65, 110, 0x42, 999_999);
        let encoded = event.encode();
        assert_eq!(encoded.len(), PACKED_SIZE);
        let decoded = SwmidiEvent::decode(&encoded).unwrap();
        assert_eq!(event, decoded, "round-trip failed for {:?}", ty);
    }
}

#[test]
fn encode_max_values() {
    let event = SwmidiEvent::new(
        EventType::Meta,
        15,     // max channel
        127,    // max pitch
        127,    // max velocity
        0xFF,   // max error_mask
        u32::MAX,
    );
    let encoded = event.encode();
    let decoded = SwmidiEvent::decode(&encoded).unwrap();
    assert_eq!(decoded.channel, 15);
    assert_eq!(decoded.pitch, 127);
    assert_eq!(decoded.velocity, 127);
    assert_eq!(decoded.error_mask, 0xFF);
    assert_eq!(decoded.tick, u32::MAX);
}

#[test]
fn encode_min_values() {
    let event = SwmidiEvent::new(EventType::NoteOn, 0, 0, 0, 0, 0);
    let encoded = event.encode();
    let decoded = SwmidiEvent::decode(&encoded).unwrap();
    assert_eq!(event, decoded);
    // All bytes except the type nibble should be 0
    assert_eq!(encoded[0], 0x00); // NoteOn=0, channel=0
    assert_eq!(encoded[1..], [0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn channel_above_15_is_masked() {
    for ch in [16, 32, 128, 255] {
        let event = SwmidiEvent::new(EventType::NoteOn, ch, 0, 0, 0, 0);
        assert!(event.channel <= 15, "channel {} should be masked, got {}", ch, event.channel);
    }
}

#[test]
fn pitch_above_127_is_masked() {
    for pitch in [128, 200, 255] {
        let event = SwmidiEvent::new(EventType::NoteOn, 0, pitch, 0, 0, 0);
        assert!(event.pitch <= 127, "pitch {} should be masked, got {}", pitch, event.pitch);
    }
}

#[test]
fn velocity_above_127_is_masked() {
    for vel in [128, 200, 255] {
        let event = SwmidiEvent::new(EventType::NoteOn, 0, 0, vel, 0, 0);
        assert!(event.velocity <= 127, "velocity {} should be masked", vel);
    }
}

#[test]
fn all_invalid_event_types_return_none() {
    for nibble in 5..16 {
        let mut buf = [0u8; PACKED_SIZE];
        buf[0] = nibble << 4;
        assert!(SwmidiEvent::decode(&buf).is_none(), "nibble {} should be invalid", nibble);
    }
}

#[test]
fn tick_little_endian_round_trip() {
    let test_ticks = [0, 1, 96, 255, 65536, 1_000_000, u32::MAX];
    for &tick in &test_ticks {
        let event = SwmidiEvent::new(EventType::Meta, 0, 0, 0, 0, tick);
        let encoded = event.encode();
        let decoded = SwmidiEvent::decode(&encoded).unwrap();
        assert_eq!(decoded.tick, tick, "tick {} round-trip failed", tick);
    }
}

#[test]
fn status_byte_encoding() {
    // Verify status byte layout: type(4) | channel(4)
    let event = SwmidiEvent::new(EventType::ControlChange, 5, 0, 0, 0, 0);
    let encoded = event.encode();
    // ControlChange = 2, channel = 5 → 0x25
    assert_eq!(encoded[0], 0x25);
}

// ════════════════════════════════════════════════════════════════════
// EVENT TYPE CONVERSIONS
// ════════════════════════════════════════════════════════════════════

#[test]
fn event_type_nibble_round_trip() {
    for ty in [
        EventType::NoteOn,
        EventType::NoteOff,
        EventType::ControlChange,
        EventType::ProgramChange,
        EventType::Meta,
    ] {
        let nibble = ty.to_nibble();
        let recovered = EventType::from_nibble(nibble).unwrap();
        assert_eq!(ty, recovered);
    }
}

#[test]
fn event_type_from_nibble_masks_high_bits() {
    // High bits should be masked off
    assert_eq!(EventType::from_nibble(0x10), Some(EventType::NoteOn));   // 0x10 & 0x0F = 0
    assert_eq!(EventType::from_nibble(0x21), Some(EventType::NoteOff));  // 0x21 & 0x0F = 1
}

#[test]
fn event_type_values_are_sequential() {
    assert_eq!(EventType::NoteOn as u8, 0);
    assert_eq!(EventType::NoteOff as u8, 1);
    assert_eq!(EventType::ControlChange as u8, 2);
    assert_eq!(EventType::ProgramChange as u8, 3);
    assert_eq!(EventType::Meta as u8, 4);
}

// ════════════════════════════════════════════════════════════════════
// STREAM OPERATIONS
// ════════════════════════════════════════════════════════════════════

#[test]
fn stream_large_encode_decode() {
    let mut stream = SwmidiStream::new();
    for i in 0..1000 {
        stream.push(SwmidiEvent::new(
            EventType::NoteOn,
            (i % 16) as u8,
            (i % 128) as u8,
            ((i * 7) % 128) as u8,
            if i % 3 == 0 { 0x01 } else { 0 },
            i as u32 * 96,
        ));
    }

    let encoded = stream.encode_all();
    assert_eq!(encoded.len(), 1000 * PACKED_SIZE);

    let decoded = SwmidiStream::decode_all(&encoded).unwrap();
    assert_eq!(decoded.len(), 1000);

    for (i, (orig, dec)) in stream.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(orig, dec, "mismatch at event {}", i);
    }
}

#[test]
fn stream_decode_sixteen_byte_boundary() {
    // Exactly 2 events
    let mut stream = SwmidiStream::new();
    stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
    stream.push(SwmidiEvent::new(EventType::NoteOff, 0, 60, 0, 0, 96));
    let encoded = stream.encode_all();
    assert_eq!(encoded.len(), 16);

    let decoded = SwmidiStream::decode_all(&encoded).unwrap();
    assert_eq!(decoded.len(), 2);
}

#[test]
fn stream_decode_13_bytes_truncated() {
    let buf = [0u8; 13]; // 1 full event + 5 trailing bytes
    let result = SwmidiStream::decode_all(&buf);
    assert!(matches!(result, Err(DecodeError::Truncated)));
}

#[test]
fn stream_decode_5_bytes_truncated() {
    let buf = [0u8; 5];
    assert!(matches!(SwmidiStream::decode_all(&buf), Err(DecodeError::Truncated)));
}

#[test]
fn stream_sort_preserves_insertion_order_for_ties() {
    let mut stream = SwmidiStream::new();
    stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 96));
    stream.push(SwmidiEvent::new(EventType::NoteOff, 1, 60, 100, 0, 96));
    stream.push(SwmidiEvent::new(EventType::NoteOn, 2, 64, 100, 0, 0));

    stream.sort_by_tick();
    // tick 0 event should be first
    assert_eq!(stream.get(0).unwrap().channel, 2);
    // tick 96 events: NoteOn (channel 0) should come before NoteOff (channel 1)
    assert_eq!(stream.get(1).unwrap().channel, 0);
    assert_eq!(stream.get(2).unwrap().channel, 1);
}

#[test]
fn stream_in_tick_range_boundaries() {
    let mut stream = SwmidiStream::new();
    for i in 0..10 {
        stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, i * 96));
    }

    // Range [192, 576) → ticks 192, 288, 384, 480
    let in_range: Vec<_> = stream.in_tick_range(192, 576).collect();
    assert_eq!(in_range.len(), 4);
    assert_eq!(in_range[0].tick, 192);
    assert_eq!(in_range[3].tick, 480);
}

#[test]
fn stream_in_tick_range_empty() {
    let mut stream = SwmidiStream::new();
    stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 100));

    let in_range: Vec<_> = stream.in_tick_range(200, 300).collect();
    assert!(in_range.is_empty());
}

#[test]
fn stream_friction_flow_count_accuracy() {
    let mut stream = SwmidiStream::new();
    // 3 flow events
    stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
    stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 96));
    stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 192));
    // 2 friction events
    stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0x01, 288));
    stream.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0xFF, 384));

    assert_eq!(stream.flow_count(), 3);
    assert_eq!(stream.friction_count(), 2);
    assert_eq!(stream.len(), 5);
}

#[test]
fn stream_get_out_of_bounds() {
    let stream = SwmidiStream::new();
    assert!(stream.get(0).is_none());
}

// ════════════════════════════════════════════════════════════════════
// DECODE ERROR HANDLING
// ════════════════════════════════════════════════════════════════════

#[test]
fn decode_error_implements_std_error() {
    let err = DecodeError::Truncated;
    assert!(std::error::Error::source(&err).is_none());
}

#[test]
fn decode_error_equality() {
    assert_eq!(DecodeError::Truncated, DecodeError::Truncated);
    assert_ne!(DecodeError::Truncated, DecodeError::InvalidEventType);
    assert_ne!(DecodeError::InvalidEventType, DecodeError::TrailingBytes);
}

#[test]
fn stream_decode_all_with_invalid_type_in_second_event() {
    let mut encoded = Vec::new();
    // First event: valid
    encoded.extend_from_slice(&SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0).encode());
    // Second event: invalid type
    let mut bad_event = [0u8; PACKED_SIZE];
    bad_event[0] = 0xF0; // type nibble = 15 (invalid)
    encoded.extend_from_slice(&bad_event);

    let result = SwmidiStream::decode_all(&encoded);
    assert!(matches!(result, Err(DecodeError::InvalidEventType)));
}

// ════════════════════════════════════════════════════════════════════
// FLOW / FRICTION PREDICATES
// ════════════════════════════════════════════════════════════════════

#[test]
fn is_flow_and_has_friction_are_complementary() {
    let flow = SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0);
    let friction = SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 1, 0);

    assert!(flow.is_flow());
    assert!(!flow.has_friction());
    assert!(!friction.is_flow());
    assert!(friction.has_friction());
}

#[test]
fn error_mask_preserves_all_bits() {
    let masks = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0xFF];
    for &mask in &masks {
        let event = SwmidiEvent::new(EventType::Meta, 0, 0, 0, mask, 0);
        let encoded = event.encode();
        let decoded = SwmidiEvent::decode(&encoded).unwrap();
        assert_eq!(decoded.error_mask, mask, "error_mask {} not preserved", mask);
    }
}
