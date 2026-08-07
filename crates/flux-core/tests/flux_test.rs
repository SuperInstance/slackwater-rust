//! Integration tests for flux-core.
//!
//! Verifies:
//! - EisensteinCoord: creation, cartesian conversion, lattice snapping idempotence
//! - ErrorMask: all flags, combinations, flow detection, friction count
//! - SwmidiEvent: pack/unpack roundtrip is lossless
//! - SwmidiStream: pack/unpack binary roundtrip, JSON roundtrip, tick queries
//! - 100-part build packing: verify ~800 bytes for 100 events

#![warn(clippy::all)]

use flux_core::EisensteinCoord;
use flux_core::{ErrorMask, EventType, SwmidiEvent, SwmidiStream};

// ═══════════════════════════════════════════════════════════════════════
//  EisensteinCoord tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn coord_basic_creation() {
    let c = EisensteinCoord::new(3, -2);
    assert_eq!(c.a, 3);
    assert_eq!(c.b, -2);
}

#[test]
fn coord_origin() {
    let o = EisensteinCoord::ORIGIN;
    assert!(o.is_origin());
    let (x, z) = o.to_cartesian();
    assert_eq!(x, 0.0);
    assert_eq!(z, 0.0);
}

#[test]
fn coord_cartesian_from_grand_plan() {
    // From Grand Plan §2: Eisenstein (3, -2) → Cartesian (16, ≈-6.93)
    let coord = EisensteinCoord::new(3, -2);
    let (x, z) = coord.to_cartesian();
    assert!((x - 16.0).abs() < 1e-9, "x should be 16.0, got {x}");
    assert!(
        (z - (-6.928_203_230_275_509)).abs() < 1e-9,
        "z should be ≈-6.928, got {z}"
    );
}

#[test]
fn coord_snap_idempotent() {
    let test_cases = [
        (0i32, 0i32),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (3, -2),
        (-5, 7),
        (10, -10),
        (100, 100),
        (-100, -100),
        (42, -17),
        (7, 3),
    ];

    for (a, b) in test_cases {
        let coord = EisensteinCoord::new(a, b);
        let (x, z) = coord.to_cartesian();
        let snapped = EisensteinCoord::snap_to_lattice(x, z);
        assert_eq!(
            snapped, coord,
            "snap idempotence failed for ({a},{b}): got ({},{})",
            snapped.a, snapped.b
        );
    }
}

#[test]
fn coord_snap_arbitrary_floats() {
    // Snap some arbitrary float coordinates and verify they land on lattice points
    let test_cases: &[(f64, f64, i32, i32)] = &[
        (0.0, 0.0, 0, 0),
        (4.0, 0.0, 1, 0),                   // x=4, z=0 → (1, 0)
        (0.0, 6.928_203_230_275_509, 1, 2), // b=2, a=0/4+2/2=1
        (8.0, 0.0, 2, 0),
        (-2.0, 3.464_101_615_137_755, 0, 1), // b=1, a=-2/4+0.5=0
    ];

    for &(x, z, expected_a, expected_b) in test_cases {
        let snapped = EisensteinCoord::snap_to_lattice(x, z);
        assert_eq!(
            (snapped.a, snapped.b),
            (expected_a, expected_b),
            "snap({x},{z}) expected ({expected_a},{expected_b}), got ({},{})",
            snapped.a,
            snapped.b
        );
    }
}

#[test]
fn coord_distance() {
    let origin = EisensteinCoord::ORIGIN;
    let neighbor = EisensteinCoord::new(1, 0);
    // Distance between adjacent lattice points = lattice scale = 4.0
    let d = origin.distance_to(&neighbor);
    assert!(
        (d - 4.0).abs() < 1e-9,
        "neighbor distance should be 4.0, got {d}"
    );
}

#[test]
fn coord_neighbors_are_six_equidistant() {
    let center = EisensteinCoord::new(2, 3);
    let neighbors = center.neighbors();
    assert_eq!(neighbors.len(), 6);

    // All neighbors should be at distance 4.0 (one lattice cell)
    for n in &neighbors {
        let d = center.distance_to(n);
        assert!(
            (d - 4.0).abs() < 1e-9,
            "neighbor {n} at distance {d}, expected 4.0"
        );
    }

    // All neighbors distinct
    for i in 0..6 {
        for j in (i + 1)..6 {
            assert_ne!(neighbors[i], neighbors[j]);
        }
    }
}

#[test]
fn coord_add_sub() {
    let a = EisensteinCoord::new(3, -2);
    let b = EisensteinCoord::new(1, 4);
    let sum = a.add(&b);
    assert_eq!(sum.a, 4);
    assert_eq!(sum.b, 2);

    let diff = a.sub(&b);
    assert_eq!(diff.a, 2);
    assert_eq!(diff.b, -6);
}

// ═══════════════════════════════════════════════════════════════════════
//  ErrorMask tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn mask_all_eight_flags() {
    assert_eq!(ErrorMask::SPATIAL.bits(), 0x01);
    assert_eq!(ErrorMask::TEMPORAL.bits(), 0x02);
    assert_eq!(ErrorMask::SEMANTIC.bits(), 0x04);
    assert_eq!(ErrorMask::SAFETY.bits(), 0x08);
    assert_eq!(ErrorMask::RESOURCE.bits(), 0x10);
    assert_eq!(ErrorMask::TOPOLOGY.bits(), 0x20);
    assert_eq!(ErrorMask::AUTHORITY.bits(), 0x40);
    assert_eq!(ErrorMask::CONSISTENCY.bits(), 0x80);
}

#[test]
fn mask_flow_state() {
    let flow = ErrorMask::FLOW;
    assert!(flow.is_flow());
    assert_eq!(flow.friction_count(), 0);
    assert!(!flow.is_blocked());
    assert_eq!(flow.set_flags(), Vec::<&str>::new());
}

#[test]
fn mask_single_flag() {
    let mask = ErrorMask::SPATIAL;
    assert!(!mask.is_flow());
    assert_eq!(mask.friction_count(), 1);
    assert!(!mask.is_blocked());
    assert!(mask.contains(ErrorMask::SPATIAL));
    assert_eq!(mask.set_flags(), vec!["SPATIAL"]);
}

#[test]
fn mask_combinations() {
    let mask = ErrorMask::SPATIAL | ErrorMask::TEMPORAL | ErrorMask::SAFETY;
    assert_eq!(mask.friction_count(), 3);
    assert!(mask.is_blocked()); // 3 ≥ 3
    assert!(mask.contains(ErrorMask::SPATIAL));
    assert!(mask.contains(ErrorMask::TEMPORAL));
    assert!(mask.contains(ErrorMask::SAFETY));
    assert!(!mask.contains(ErrorMask::RESOURCE));
    assert_eq!(mask.bits(), 0x0B);
}

#[test]
fn mask_blocked_threshold() {
    // 2 flags: not blocked
    let two = ErrorMask::SPATIAL | ErrorMask::TEMPORAL;
    assert_eq!(two.friction_count(), 2);
    assert!(!two.is_blocked());

    // 3 flags: blocked
    let three = ErrorMask::SPATIAL | ErrorMask::TEMPORAL | ErrorMask::SEMANTIC;
    assert_eq!(three.friction_count(), 3);
    assert!(three.is_blocked());

    // 8 flags: definitely blocked
    let all = ErrorMask::BLOCKED_ALL;
    assert_eq!(all.friction_count(), 8);
    assert!(all.is_blocked());
}

#[test]
fn mask_with_without() {
    let base = ErrorMask::SPATIAL | ErrorMask::TEMPORAL;

    let with_safety = base.with(ErrorMask::SAFETY);
    assert_eq!(with_safety.friction_count(), 3);

    let without_spatial = with_safety.without(ErrorMask::SPATIAL);
    assert!(!without_spatial.contains(ErrorMask::SPATIAL));
    assert!(without_spatial.contains(ErrorMask::TEMPORAL));
    assert!(without_spatial.contains(ErrorMask::SAFETY));
}

#[test]
fn mask_u8_roundtrip() {
    for bits in 0u8..=255 {
        let mask = ErrorMask::from(bits);
        assert_eq!(mask.bits(), bits);
        let back: u8 = mask.into();
        assert_eq!(back, bits);
    }
}

#[test]
fn mask_display() {
    assert_eq!(format!("{}", ErrorMask::FLOW), "FLOW(0x00)");

    let mask = ErrorMask::SPATIAL;
    let s = format!("{mask}");
    assert!(s.contains("SPATIAL"));
    assert!(s.contains("0x01"));
}

// ═══════════════════════════════════════════════════════════════════════
//  SwmidiEvent tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn event_creation_and_constructors() {
    let note_on = SwmidiEvent::note_on(0, 60, 96, 43200);
    assert_eq!(note_on.event_type, EventType::NoteOn);
    assert_eq!(note_on.channel, 0);
    assert_eq!(note_on.pitch, 60);
    assert_eq!(note_on.velocity, 96);
    assert_eq!(note_on.tick, 43200);
    assert!(note_on.error_mask.is_flow());

    let note_off = SwmidiEvent::note_off(3, 72, 44100);
    assert_eq!(note_off.event_type, EventType::NoteOff);
    assert_eq!(note_off.velocity, 0);
}

#[test]
fn event_pack_unpack_roundtrip_lossless() {
    let test_events = [
        SwmidiEvent::note_on(0, 60, 96, 43200),
        SwmidiEvent::note_off(15, 0, 0),
        SwmidiEvent::new(EventType::ControlChange, 7, 16, 64, 192000),
        SwmidiEvent::new(EventType::ProgramChange, 10, 0, 0, 96000),
        SwmidiEvent::new(EventType::Meta, 15, 81, 120, 0),
        SwmidiEvent::note_on(9, 36, 127, 0xFFFFFFFF)
            .with_mask(ErrorMask::SPATIAL | ErrorMask::TEMPORAL | ErrorMask::SAFETY),
        SwmidiEvent::new(EventType::NoteOn, 0, 48, 0, 1),
        SwmidiEvent::new(EventType::NoteOn, 15, 127, 127, 123456789),
    ];

    for event in &test_events {
        let packed = event.pack();
        assert_eq!(packed.len(), 8, "packed must be exactly 8 bytes");

        let unpacked = SwmidiEvent::unpack(&packed).expect("unpack should succeed");

        assert_eq!(unpacked.event_type, event.event_type, "event_type mismatch");
        assert_eq!(unpacked.channel, event.channel, "channel mismatch");
        assert_eq!(unpacked.pitch, event.pitch, "pitch mismatch");
        assert_eq!(unpacked.velocity, event.velocity, "velocity mismatch");
        assert_eq!(unpacked.tick, event.tick, "tick mismatch");
        assert_eq!(unpacked.error_mask, event.error_mask, "error_mask mismatch");
    }
}

#[test]
fn event_pack_byte_layout() {
    // Verify exact byte layout matches Grand Plan §2 spec
    let event = SwmidiEvent::new(EventType::NoteOn, 3, 60, 96, 0x12345678);

    let packed = event.pack();

    // byte 0: type(4) | channel(4) = 0x03 (NoteOn=0, channel=3)
    assert_eq!(packed[0], 0x03);
    // byte 1: pitch = 60
    assert_eq!(packed[1], 60);
    // byte 2: velocity = 96
    assert_eq!(packed[2], 96);
    // byte 3: error_mask = 0x00 (FLOW)
    assert_eq!(packed[3], 0x00);
    // bytes 4-7: tick = 0x12345678 in LE
    assert_eq!(packed[4..8], [0x78, 0x56, 0x34, 0x12]);
}

#[test]
fn event_invalid_type_nibble() {
    // Type nibble 0xF is invalid (only 0-4 defined)
    let bad_data = [0xF0, 0, 0, 0, 0, 0, 0, 0];
    assert!(SwmidiEvent::unpack(&bad_data).is_err());
}

#[test]
fn event_all_type_nibbles() {
    // Valid type nibbles: 0-4
    for type_nibble in 0u8..=4 {
        let status = type_nibble << 4; // channel 0
        let data = [status, 60, 96, 0, 0, 0, 0, 0];
        let result = SwmidiEvent::unpack(&data);
        assert!(result.is_ok(), "type nibble {type_nibble} should be valid");
    }

    // Invalid: 5-15
    for type_nibble in 5u8..=15 {
        let status = type_nibble << 4;
        let data = [status, 0, 0, 0, 0, 0, 0, 0];
        assert!(
            SwmidiEvent::unpack(&data).is_err(),
            "type nibble {type_nibble} should be invalid"
        );
    }
}

#[test]
fn event_with_cc_builder() {
    let event = SwmidiEvent::note_on(0, 60, 96, 0)
        .with_cc(vec![(16, 64), (17, 32), (20, 10)])
        .with_mask(ErrorMask::SPATIAL);

    assert_eq!(event.cc.as_ref().unwrap().len(), 3);
    assert_eq!(event.cc.as_ref().unwrap()[0], (16, 64));
    assert!(event.error_mask.contains(ErrorMask::SPATIAL));
}

#[test]
fn event_empty_cc_is_none() {
    let event = SwmidiEvent::note_on(0, 60, 96, 0).with_cc(vec![]);
    assert!(event.cc.is_none());
}

// ═══════════════════════════════════════════════════════════════════════
//  SwmidiStream tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stream_basic_operations() {
    let mut stream = SwmidiStream::new();
    assert!(stream.is_empty());

    stream.push(SwmidiEvent::note_on(0, 60, 96, 100));
    assert!(!stream.is_empty());
    assert_eq!(stream.len(), 1);

    stream.push(SwmidiEvent::note_off(0, 60, 200));
    assert_eq!(stream.len(), 2);
}

#[test]
fn stream_binary_pack_unpack_roundtrip() {
    let mut stream = SwmidiStream::new();
    for i in 0..50 {
        let event = SwmidiEvent::new(
            EventType::NoteOn,
            (i % 16) as u8,
            48 + (i % 12) as u8,
            64 + (i % 64) as u8,
            (i as u32) * 96,
        )
        .with_mask(if i % 5 == 0 {
            ErrorMask::SPATIAL
        } else if i % 7 == 0 {
            ErrorMask::TEMPORAL | ErrorMask::SAFETY
        } else {
            ErrorMask::FLOW
        });
        stream.push(event);
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
    }
}

#[test]
fn stream_binary_pack_unpack_with_cc() {
    let mut stream = SwmidiStream::new();
    for i in 0..30 {
        let event = SwmidiEvent::note_on(0, 60, 96, i * 96).with_cc(vec![
            (16, (i % 128) as u8),
            (17, (i % 128) as u8),
            (20, 10),
            (21, 5),
            (22, (i % 8) as u8),
        ]);
        stream.push(event);
    }
    // Add some events without CC
    for i in 30..50 {
        stream.push(SwmidiEvent::note_on(0, 60, 96, i * 96));
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
        assert_eq!(orig.cc, unpacked_evt.cc, "CC mismatch at event");
    }
}

#[test]
fn stream_json_roundtrip() {
    let mut stream = SwmidiStream::new();
    stream.push(SwmidiEvent::note_on(0, 60, 96, 43200));
    stream.push(
        SwmidiEvent::note_on(9, 36, 120, 43296)
            .with_mask(ErrorMask::TEMPORAL)
            .with_cc(vec![(16, 64)]),
    );
    stream.push(SwmidiEvent::note_off(0, 60, 44000));

    let json = stream.to_json();
    let unpacked = SwmidiStream::from_json(&json).unwrap();

    assert_eq!(unpacked.len(), 3);
    assert_eq!(unpacked.events()[0].pitch, 60);
    assert_eq!(unpacked.events()[1].channel, 9);
    assert_eq!(unpacked.events()[1].error_mask, ErrorMask::TEMPORAL);
    assert_eq!(unpacked.events()[1].cc.as_ref().unwrap(), &[(16, 64)]);
    assert_eq!(unpacked.events()[2].event_type, EventType::NoteOff);
}

#[test]
fn stream_json_invalid() {
    let result = SwmidiStream::from_json("not valid json");
    assert!(result.is_err());
}

#[test]
fn stream_at_tick() {
    let mut stream = SwmidiStream::new();
    stream.push(SwmidiEvent::note_on(0, 60, 96, 100));
    stream.push(SwmidiEvent::note_on(1, 64, 80, 100));
    stream.push(SwmidiEvent::note_on(0, 67, 88, 200));
    stream.push(SwmidiEvent::note_on(0, 72, 64, 100));

    let at_100 = stream.at_tick(100);
    assert_eq!(at_100.len(), 3);

    let at_200 = stream.at_tick(200);
    assert_eq!(at_200.len(), 1);
    assert_eq!(at_200[0].pitch, 67);

    let at_999 = stream.at_tick(999);
    assert!(at_999.is_empty());
}

#[test]
fn stream_in_range() {
    let mut stream = SwmidiStream::new();
    stream.push(SwmidiEvent::note_on(0, 60, 96, 0));
    stream.push(SwmidiEvent::note_on(0, 62, 80, 96));
    stream.push(SwmidiEvent::note_on(0, 64, 88, 192));
    stream.push(SwmidiEvent::note_on(0, 65, 72, 288));
    stream.push(SwmidiEvent::note_on(0, 67, 96, 384));

    // Range [0, 192) → ticks 0, 96
    let early = stream.in_range(0, 192);
    assert_eq!(early.len(), 2);

    // Range [96, 288) → ticks 96, 192
    let mid = stream.in_range(96, 288);
    assert_eq!(mid.len(), 2);

    // Range [0, 1000) → all
    let all = stream.in_range(0, 1000);
    assert_eq!(all.len(), 5);

    // Empty range
    let none = stream.in_range(500, 600);
    assert!(none.is_empty());
}

#[test]
fn stream_sort_by_tick() {
    let mut stream = SwmidiStream::new();
    stream.push(SwmidiEvent::note_on(0, 72, 96, 300));
    stream.push(SwmidiEvent::note_on(0, 60, 96, 100));
    stream.push(SwmidiEvent::note_on(0, 67, 96, 200));

    stream.sort_by_tick();
    assert_eq!(stream.events()[0].tick, 100);
    assert_eq!(stream.events()[1].tick, 200);
    assert_eq!(stream.events()[2].tick, 300);
}

// ═══════════════════════════════════════════════════════════════════════
//  100-part build benchmark / size verification
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn hundred_part_build_packed_size() {
    // From the Grand Plan: "A 100-part build is ~800 bytes of events"
    let mut stream = SwmidiStream::new();
    for i in 0..100 {
        stream.push(SwmidiEvent::note_on(
            0,
            48 + (i % 12) as u8,
            64 + (i % 64) as u8,
            (i as u32) * 12,
        ));
    }

    let packed = stream.pack_binary();

    // 4 (event count) + 100 × 8 (events) + 4 (CC count = 0) = 808 bytes
    assert_eq!(
        packed.len(),
        808,
        "100 events with no CC should be 808 bytes (4 + 800 + 4)"
    );
    assert!(
        packed.len() < 850,
        "100-event build should be well under 850 bytes"
    );
}

#[test]
fn hundred_part_build_with_cc_size() {
    // With CC spatial payload: ~800 bytes events + ~2KB CC payload
    // Grand Plan says "~800 bytes of events plus ~2 KB of CC payload"
    let mut stream = SwmidiStream::new();
    for i in 0..100 {
        let event = SwmidiEvent::note_on(0, 60, 96, i * 12).with_cc(vec![
            (16, (i % 128) as u8), // Eisenstein a (low byte)
            (17, (i % 128) as u8), // Eisenstein b (low byte)
            (20, (i % 128) as u8), // Y-register
            (21, (i % 16) as u8),  // material
            (22, (i % 8) as u8),   // size class
        ]);
        stream.push(event);
    }

    let packed = stream.pack_binary();

    // 4 + 800 + 4 + 100 × (5 + 10) = 4 + 800 + 4 + 1500 = 2308 bytes
    // Grand Plan says ~800 + ~2KB ≈ 2.8KB. Our tighter packing is even better.
    let total = packed.len();
    assert!(
        total < 3000,
        "100-event build with CC should be under 3KB, got {total}"
    );
    assert!(
        total > 2000,
        "100-event build with 5 CC pairs each should be over 2KB, got {total}"
    );
}

#[test]
fn json_vs_binary_size_ratio() {
    // The Grand Plan thesis: ~30× payload reduction from binary
    let mut stream = SwmidiStream::new();
    for i in 0..100 {
        stream.push(SwmidiEvent::note_on(
            0,
            48 + (i % 12) as u8,
            64 + (i % 64) as u8,
            (i as u32) * 12,
        ));
    }

    let binary = stream.pack_binary();
    let json = stream.to_json();

    // Binary should be much smaller than JSON
    let ratio = json.len() as f64 / binary.len() as f64;
    assert!(
        ratio > 5.0,
        "JSON should be at least 5× larger than binary (got {ratio:.1}×: JSON {} vs binary {})",
        json.len(),
        binary.len()
    );
}

#[test]
fn empty_stream_operations() {
    let stream = SwmidiStream::new();
    assert!(stream.is_empty());
    assert_eq!(stream.len(), 0);
    assert!(stream.at_tick(0).is_empty());
    assert!(stream.in_range(0, 1000).is_empty());

    let packed = stream.pack_binary();
    assert_eq!(packed.len(), 8); // 4 (count=0) + 4 (cc_count=0)

    let unpacked = SwmidiStream::unpack_binary(&packed).unwrap();
    assert!(unpacked.is_empty());
}

#[test]
fn channel_bitfield_isolation() {
    // Grand Plan rule: channels are authorization.
    // Channel 13 (voice) cannot emit build notes.
    // Verify that we can distinguish events by channel.

    let voice_event = SwmidiEvent::note_on(13, 60, 96, 0); // Hermes trying to build
    let build_event = SwmidiEvent::note_on(0, 60, 96, 0); // Lucineer building

    assert_ne!(voice_event.channel, build_event.channel);

    // The packed form preserves channel in the low nibble of byte 0
    let voice_packed = voice_event.pack();
    let build_packed = build_event.pack();

    assert_eq!(voice_packed[0] & 0x0F, 13);
    assert_eq!(build_packed[0] & 0x0F, 0);
}

#[test]
fn error_mask_travels_with_event() {
    // Grand Plan rule: the error mask travels with the event.
    // No layer throws across a boundary; it sets bits and forwards.

    let mut event = SwmidiEvent::note_on(0, 60, 96, 43200);
    assert!(event.error_mask.is_flow());

    // Layer detects a collision
    event.error_mask |= ErrorMask::SPATIAL;
    assert!(!event.error_mask.is_flow());
    assert!(event.error_mask.contains(ErrorMask::SPATIAL));

    // Another layer detects a safety issue
    event.error_mask |= ErrorMask::SAFETY;
    assert_eq!(event.error_mask.friction_count(), 2);

    // The packed form preserves the mask
    let packed = event.pack();
    let unpacked = SwmidiEvent::unpack(&packed).unwrap();
    assert_eq!(unpacked.error_mask, event.error_mask);
    assert!(unpacked.error_mask.contains(ErrorMask::SPATIAL));
    assert!(unpacked.error_mask.contains(ErrorMask::SAFETY));
}

#[test]
fn grand_plan_channel_map_coverage() {
    // Verify all 16 channels from the Grand Plan channel map are representable
    for ch in 0u8..=15 {
        let event = SwmidiEvent::note_on(ch, 60, 96, 0);
        let packed = event.pack();
        let unpacked = SwmidiEvent::unpack(&packed).unwrap();
        assert_eq!(
            unpacked.channel, ch,
            "channel {ch} should survive pack/unpack"
        );
    }
}
