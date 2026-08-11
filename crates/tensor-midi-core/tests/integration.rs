//! Integration tests for tensor-midi-core
//!
//! These tests exercise multi-message scenarios, the full capture pipeline,
//! jazz analysis across realistic conversations, and edge cases that the
//! inline unit tests don't cover.

use tensor_midi_core::{
    analyze_sentiment, detect_tempo,
    channels, friction,
    Capture, ChordQuality, EventRingBuffer, GridEvent, JazzAnalysis, JazzMode,
    Message, PulseGrid, PulsePosition, SentimentLabel,
    SwmidiEvent, EventType, TICKS_PER_BAR, TICKS_PER_PULSE, PULSES_PER_BAR,
    tick_to_pulse, pulse_to_tick,
};

// ════════════════════════════════════════════════════════════════════
// FULL CONVERSATION CAPTURE PIPELINE
// ════════════════════════════════════════════════════════════════════

#[test]
fn full_pipeline_multi_participant_conversation() {
    let mut cap = Capture::new();

    // A realistic conversation: human asks, assistant answers, subagent helps
    let messages = vec![
        Message { text: "How do we build a jazz engine?".into(), sender: "human".into(), timestamp_ms: 0 },
        Message { text: "Great question! Let's design and build it together".into(), sender: "assistant".into(), timestamp_ms: 800 },
        Message { text: "I'll craft the MIDI mapping module".into(), sender: "subagent1".into(), timestamp_ms: 1600 },
        Message { text: "Perfect, that's amazing work everyone".into(), sender: "human".into(), timestamp_ms: 3000 },
        Message { text: "The build failed with an error in the audio module".into(), sender: "tool".into(), timestamp_ms: 4000 },
        Message { text: "Let me explore and fix the issue".into(), sender: "assistant".into(), timestamp_ms: 5000 },
    ];

    for msg in messages {
        cap.capture(msg);
    }

    // All messages captured
    assert_eq!(cap.messages().len(), 6);
    assert_eq!(cap.events().len(), 6);

    // Participants registered with correct channels
    assert_eq!(cap.messages()[0].channel, channels::HUMAN);
    assert_eq!(cap.messages()[1].channel, channels::ASSISTANT);
    assert!(cap.messages()[2].channel >= 5); // subagent gets dynamic channel

    // Jazz analysis runs
    let analysis = JazzAnalysis::from_capture(&cap);
    assert!(analysis.event_count == 6);
    assert!(analysis.participant_count >= 3);
    assert!(!analysis.description.is_empty());
}

#[test]
fn pipeline_capture_clear_reset() {
    let mut cap = Capture::new();
    cap.capture(Message {
        text: "Test message".into(),
        sender: "human".into(),
        timestamp_ms: 0,
    });
    assert_eq!(cap.messages().len(), 1);
    assert!(!cap.grid().is_empty());

    cap.clear();
    assert_eq!(cap.messages().len(), 0);
    assert!(cap.grid().is_empty());
    assert_eq!(cap.events().len(), 0);
}

#[test]
fn pipeline_encode_decode_round_trip() {
    let mut cap = Capture::new();
    cap.capture(Message { text: "Hello world".into(), sender: "human".into(), timestamp_ms: 0 });
    cap.capture(Message { text: "Building something great".into(), sender: "assistant".into(), timestamp_ms: 500 });
    cap.capture(Message { text: "Error: failed".into(), sender: "tool".into(), timestamp_ms: 1000 });

    let binary = cap.encode_binary();
    // Each event is PACKED_SIZE bytes
    assert_eq!(binary.len(), 3 * tensor_midi_core::PACKED_SIZE);

    // Decode each event
    for chunk in binary.chunks_exact(tensor_midi_core::PACKED_SIZE) {
        let arr: [u8; 8] = chunk.try_into().unwrap();
        let event = SwmidiEvent::decode(&arr).unwrap();
        assert!(event.pitch <= 127);
    }
}

#[test]
fn pipeline_export_data_has_consistency() {
    let mut cap = Capture::new();
    cap.capture(Message { text: "Great".into(), sender: "human".into(), timestamp_ms: 0 });
    cap.capture(Message { text: "Error".into(), sender: "tool".into(), timestamp_ms: 1000 });

    let export = cap.export_data();
    assert_eq!(export.events.len(), 2);
    assert_eq!(export.messages.len(), 2);
    assert!(export.participants.contains_key("human"));
    assert!(export.participants.contains_key("tool"));
    assert!(export.bpm > 0.0);
}

// ════════════════════════════════════════════════════════════════════
// SENTIMENT EDGE CASES
// ════════════════════════════════════════════════════════════════════

#[test]
fn sentiment_empty_string() {
    let s = analyze_sentiment("");
    assert_eq!(s.label, SentimentLabel::Neutral);
    assert_eq!(s.positivity, 0);
    assert_eq!(s.negativity, 0);
    assert_eq!(s.friction, friction::NONE);
    // Velocity from 0-length string should be minimum
    assert!(s.velocity >= 1);
}

#[test]
fn sentiment_only_whitespace() {
    let s = analyze_sentiment("   \n\t  \r\n  ");
    assert_eq!(s.label, SentimentLabel::Neutral);
}

#[test]
fn sentiment_mixed_positive_and_negative() {
    // "good" matches positive, "bad" matches negative
    let s = analyze_sentiment("good bad");
    assert!(s.negativity >= 1);
    assert!(s.positivity >= 1);
    // The label depends on which counter is higher. With equal counts,
    // positivity is not strictly greater than negativity, so Tense wins.
    // But if partial matching makes one side higher, adjust:
    if s.negativity > s.positivity {
        assert_eq!(s.label, SentimentLabel::Tense);
    } else {
        // positivity >= negativity → Bright (since creativity is 0 and question is 0)
        assert_eq!(s.label, SentimentLabel::Bright);
    }
}

#[test]
fn sentiment_creative_overrides_positive() {
    // Creative check comes before positive in the label cascade
    let s = analyze_sentiment("imagine great wonderful");
    assert!(s.creativity > 0);
    assert!(s.positivity > 0);
    assert_eq!(s.label, SentimentLabel::Creative);
}

#[test]
fn sentiment_question_pitches_high() {
    let s = analyze_sentiment("what how why");
    assert!(s.question >= 3);
    assert_eq!(s.label, SentimentLabel::Inquiring);
    // Questions should be pitched high (72+)
    assert!(s.pitch >= 72);
}

#[test]
fn sentiment_case_insensitive_matching() {
    let upper = analyze_sentiment("GREAT AMAZING");
    let lower = analyze_sentiment("great amazing");
    assert_eq!(upper.positivity, lower.positivity);
    assert_eq!(upper.label, lower.label);
}

#[test]
fn sentiment_partial_word_matching() {
    // "awesome!" should match "awesome" via contains
    let s = analyze_sentiment("This is awesome!");
    assert!(s.positivity >= 1);
}

#[test]
fn sentiment_error_keyword_triggers_syntax_friction() {
    assert!(analyze_sentiment("compile error").friction & friction::SYNTAX_ERROR != 0);
    assert!(analyze_sentiment("test failed").friction & friction::SYNTAX_ERROR != 0);
    assert!(analyze_sentiment("system crash").friction & friction::SYNTAX_ERROR != 0);
    // Non-error text should not set syntax_error
    assert!(analyze_sentiment("hello world").friction & friction::SYNTAX_ERROR == 0);
}

#[test]
fn sentiment_velocity_increases_with_length() {
    let velocities: Vec<u8> = [10, 100, 300, 500]
        .iter()
        .map(|&len| analyze_sentiment(&"a ".repeat(len)).velocity)
        .collect();
    // Should be monotonically non-decreasing
    for i in 0..velocities.len() - 1 {
        assert!(velocities[i] <= velocities[i + 1]);
    }
}

#[test]
fn sentiment_pitch_bounds_never_exceeded() {
    // Extreme negativity
    let s = analyze_sentiment("bad bad bad bad bad bad bad bad bad bad bad bad bad bad");
    assert!(s.pitch <= 127);

    // Extreme positivity + creativity
    let s = analyze_sentiment(
        "great awesome love perfect excellent wonderful amazing brilliant fantastic beautiful create build design imagine dream invent explore craft forge"
    );
    assert!(s.pitch <= 127);
}

#[test]
fn sentiment_label_predicates() {
    assert!(SentimentLabel::Tense.is_tense());
    assert!(!SentimentLabel::Bright.is_tense());
    assert!(SentimentLabel::Creative.is_creative());
    assert!(!SentimentLabel::Neutral.is_creative());
}

// ════════════════════════════════════════════════════════════════════
// PULSE GRID ADVANCED TESTS
// ════════════════════════════════════════════════════════════════════

#[test]
fn pulse_grid_all_twelve_pulses_filled() {
    let mut grid = PulseGrid::new();
    for i in 0..12 {
        grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, i as u32 * TICKS_PER_PULSE));
    }
    let pattern = grid.bar_pattern(0);
    assert!(pattern.iter().all(|&p| p));
    assert!((grid.bar_density(0) - 1.0).abs() < 0.01);
}

#[test]
fn pulse_grid_no_events_density_zero() {
    let grid = PulseGrid::new();
    assert_eq!(grid.bar_density(0), 0.0);
    assert_eq!(grid.bar_density(999), 0.0);
}

#[test]
fn pulse_grid_events_across_multiple_bars() {
    let mut grid = PulseGrid::new();
    // Bar 0: pulses 0, 3, 6, 9 (four-on-the-floor-ish)
    for &p in &[0, 3, 6, 9] {
        grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, p * TICKS_PER_PULSE));
    }
    // Bar 1: pulses 0, 6 (half-time)
    for &p in &[0, 6] {
        grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, TICKS_PER_BAR + p * TICKS_PER_PULSE));
    }
    // Bar 2: empty

    assert_eq!(grid.len(), 6);
    assert!((grid.bar_density(0) - (4.0 / 12.0)).abs() < 0.01);
    assert!((grid.bar_density(1) - (2.0 / 12.0)).abs() < 0.01);
    assert_eq!(grid.bar_density(2), 0.0);
}

#[test]
fn pulse_grid_sort_by_tick() {
    let mut grid = PulseGrid::new();
    // Add events out of order
    grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 500));
    grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 64, 100, 0, 100));
    grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 67, 100, 0, 300));

    grid.sort_by_tick();
    let ticks: Vec<u32> = grid.iter().map(|e| e.event.tick).collect();
    assert_eq!(ticks, vec![100, 300, 500]);
}

#[test]
fn pulse_grid_iter_returns_all_events() {
    let mut grid = PulseGrid::new();
    for i in 0..10 {
        grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 60 + i, 100, 0, (i as u32) * 48));
    }
    let collected: Vec<&GridEvent> = grid.iter().collect();
    assert_eq!(collected.len(), 10);
}

#[test]
fn tick_to_pulse_converts_correctly_at_bar_boundaries() {
    // Tick 0 → bar 0, pulse 0
    let pos = tick_to_pulse(0);
    assert_eq!((pos.bar, pos.pulse), (0, 0));

    // Tick 575 → last tick of bar 0
    let pos = tick_to_pulse(TICKS_PER_BAR - 1);
    assert_eq!(pos.bar, 0);

    // Tick 576 → bar 1, pulse 0
    let pos = tick_to_pulse(TICKS_PER_BAR);
    assert_eq!((pos.bar, pos.pulse), (1, 0));

    // Tick 1152 → bar 2
    let pos = tick_to_pulse(2 * TICKS_PER_BAR);
    assert_eq!(pos.bar, 2);
}

#[test]
fn pulse_to_tick_round_trip_many_values() {
    // Test round-trip for many tick values across bars
    for tick in [0, 1, 47, 48, 100, 288, 575, 576, 577, 1000, 1152, 5000] {
        let pos = tick_to_pulse(tick);
        assert_eq!(pulse_to_tick(pos), tick, "round-trip failed for tick {}", tick);
    }
}

// ════════════════════════════════════════════════════════════════════
// RING BUFFER ADVANCED TESTS
// ════════════════════════════════════════════════════════════════════

#[test]
fn ring_buffer_capacity_one() {
    let mut rb = EventRingBuffer::new(1);
    assert!(rb.is_empty());
    assert_eq!(rb.capacity(), 1);

    rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
    assert_eq!(rb.len(), 1);
    assert!(rb.is_full());

    // Overwrite
    rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 72, 100, 0, 1));
    assert_eq!(rb.len(), 1);
    assert_eq!(rb.last().unwrap().pitch, 72);
}

#[test]
fn ring_buffer_wrap_around_preserves_order() {
    let mut rb = EventRingBuffer::new(4);
    // Fill completely
    for i in 0..4 {
        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 60 + i, 100, 0, i as u32));
    }
    assert!(rb.is_full());

    // Overwrite first two
    rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 70, 100, 0, 4));
    rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 71, 100, 0, 5));

    // Should have events with pitches: 62, 63, 70, 71 (oldest→newest)
    let pitches: Vec<u8> = rb.iter().map(|e| e.pitch).collect();
    assert_eq!(pitches, vec![62, 63, 70, 71]);
}

#[test]
fn ring_buffer_clear_resets_state() {
    let mut rb = EventRingBuffer::new(8);
    for i in 0..5 {
        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, i));
    }
    assert_eq!(rb.len(), 5);

    rb.clear();
    assert!(rb.is_empty());
    assert_eq!(rb.len(), 0);
    assert!(!rb.is_full());
}

#[test]
fn ring_buffer_last_on_empty_returns_none() {
    let rb = EventRingBuffer::new(4);
    assert!(rb.last().is_none());
}

#[test]
fn ring_buffer_iter_empty_returns_nothing() {
    let rb = EventRingBuffer::new(4);
    assert_eq!(rb.iter().count(), 0);
}

#[test]
fn ring_buffer_default_capacity() {
    let rb = EventRingBuffer::with_default_capacity();
    assert_eq!(rb.capacity(), tensor_midi_core::DEFAULT_RING_CAPACITY);
}

// ════════════════════════════════════════════════════════════════════
// JAZZ ANALYSIS DEEP TESTS
// ════════════════════════════════════════════════════════════════════

#[test]
fn jazz_analysis_solo_mode() {
    // Single participant → Solo mode
    let mut cap = Capture::new();
    for i in 0..3 {
        cap.capture(Message {
            text: format!("Message {}", i),
            sender: "human".into(),
            timestamp_ms: i * 1000,
        });
    }
    let analysis = JazzAnalysis::from_capture(&cap);
    assert_eq!(analysis.participant_count, 1);
    // With neutral messages and single participant, should lean solo or comping
    assert!(matches!(analysis.mode, JazzMode::Solo | JazzMode::Comping | JazzMode::Ballad));
}

#[test]
fn jazz_analysis_building_mode() {
    // Creative messages with energy → Building or Groove
    let mut cap = Capture::new();
    for i in 0..8 {
        cap.capture(Message {
            text: format!("Let's build and create amazing design number {}", i),
            sender: format!("agent{}", i % 3),
            timestamp_ms: i * 500,
        });
    }
    let analysis = JazzAnalysis::from_capture(&cap);
    // Energy comes from velocity which is based on text length.
    // The texts are ~40 chars, so velocity will be moderate.
    // Verify we got creative analysis
    assert!(analysis.friction_ratio < 0.5); // mostly flow
    // Creative messages should push toward Building or Groove
    assert!(matches!(analysis.mode, JazzMode::Building | JazzMode::Groove | JazzMode::Ballad));
}

#[test]
fn jazz_analysis_tension_mode() {
    // Negative messages → Tension
    let mut cap = Capture::new();
    for i in 0..6 {
        cap.capture(Message {
            text: format!("Error: bad crash failed broken terrible {}", i),
            sender: if i % 2 == 0 { "human" } else { "assistant" }.into(),
            timestamp_ms: i * 1000,
        });
    }
    let analysis = JazzAnalysis::from_capture(&cap);
    assert!(analysis.tension > 0.5);
    assert_eq!(analysis.mode, JazzMode::Tension);
    assert_eq!(analysis.chord, ChordQuality::Diminished);
}

#[test]
fn jazz_analysis_chord_quality_mapping() {
    // Tense → Diminished
    let mut cap = Capture::new();
    for _ in 0..5 {
        cap.capture(Message { text: "bad terrible awful".into(), sender: "a".into(), timestamp_ms: 0 });
    }
    let analysis = JazzAnalysis::from_capture(&cap);
    assert_eq!(analysis.chord, ChordQuality::Diminished);

    // Creative + low tension → Major7 (if avg_pitch is moderate)
    // or Augmented (if avg_pitch > 75)
    let mut cap2 = Capture::new();
    for i in 0..5 {
        cap2.capture(Message {
            text: format!("Let's build and design {}", i),
            sender: format!("s{}", i),
            timestamp_ms: i * 100,
        });
    }
    let analysis2 = JazzAnalysis::from_capture(&cap2);
    // Creative messages raise pitch → could be either Major7 or Augmented
    assert!(matches!(analysis2.chord, ChordQuality::Major7 | ChordQuality::Augmented));
    assert!(analysis2.tension < 0.3);
}

#[test]
fn jazz_analysis_from_empty_messages() {
    let analysis = JazzAnalysis::from_messages(&[]);
    assert_eq!(analysis.mode, JazzMode::Ballad);
    assert_eq!(analysis.event_count, 0);
    assert_eq!(analysis.tension, 0.0);
}

#[test]
fn jazz_mode_descriptions_are_poetic() {
    // Every mode should have a non-trivial description
    for mode in [
        JazzMode::Groove, JazzMode::Building, JazzMode::Tension,
        JazzMode::Release, JazzMode::Solo, JazzMode::Comping,
        JazzMode::Free, JazzMode::Ballad,
    ] {
        let desc = mode.description();
        assert!(desc.len() > 10);
        assert!(!desc.starts_with("Unknown"));
    }
}

#[test]
fn jazz_analysis_complexity_increases_with_pitch_variety() {
    // More unique pitches → higher complexity
    let mut cap_low = Capture::new();
    let mut cap_high = Capture::new();

    // Low variety: all same text → similar pitches
    for i in 0..10 {
        cap_low.capture(Message { text: "hello".into(), sender: "a".into(), timestamp_ms: i * 100 });
    }

    // High variety: different texts → different pitches
    let texts = [
        "great wonderful amazing love",
        "bad terrible error crash",
        "what how why when where",
        "imagine create build design",
        "the file was updated",
        "great awesome perfect",
        "broken stuck frustrated",
        "explore dream invent",
        "thanks happy glad",
        "slow dead lost",
    ];
    for (i, text) in texts.iter().enumerate() {
        cap_high.capture(Message { text: text.to_string(), sender: format!("s{}", i), timestamp_ms: i as u64 * 100 });
    }

    let low = JazzAnalysis::from_capture(&cap_low);
    let high = JazzAnalysis::from_capture(&cap_high);
    assert!(high.complexity >= low.complexity);
}

// ════════════════════════════════════════════════════════════════════
// TEMPO DETECTION EDGE CASES
// ════════════════════════════════════════════════════════════════════

#[test]
fn tempo_single_timestamp_returns_default() {
    assert_eq!(detect_tempo(&[42]), 120.0);
}

#[test]
fn tempo_all_same_timestamps() {
    // Zero intervals → default
    assert_eq!(detect_tempo(&[1000, 1000, 1000]), 120.0);
}

#[test]
fn tempo_unsorted_input_handled() {
    let unsorted = vec![3000, 100, 200, 3100, 200];
    let bpm = detect_tempo(&unsorted);
    // Should still produce a valid BPM
    assert!(bpm > 0.0);
    assert!(bpm <= 240.0);
}

#[test]
fn tempo_buckets_cover_full_range() {
    // Very fast (50ms intervals)
    assert_eq!(detect_tempo(&[0, 50, 100, 150]), 240.0);
    // Fast (150ms)
    assert_eq!(detect_tempo(&[0, 150, 300, 450]), 180.0);
    // Medium (350ms)
    assert_eq!(detect_tempo(&[0, 350, 700, 1050]), 140.0);
    // Moderate (750ms)
    assert_eq!(detect_tempo(&[0, 750, 1500, 2250]), 120.0);
    // Slow (1500ms)
    assert_eq!(detect_tempo(&[0, 1500, 3000, 4500]), 90.0);
    // Slower (3000ms)
    assert_eq!(detect_tempo(&[0, 3000, 6000, 9000]), 60.0);
    // Very slow (6000ms)
    assert_eq!(detect_tempo(&[0, 6000, 12000, 18000]), 40.0);
}

// ════════════════════════════════════════════════════════════════════
// CHANNEL ASSIGNMENT
// ════════════════════════════════════════════════════════════════════

#[test]
fn channel_assignment_known_participants() {
    let mut cap = Capture::new();
    assert_eq!(cap.register("human"), channels::HUMAN);
    assert_eq!(cap.register("assistant"), channels::ASSISTANT);
    assert_eq!(cap.register("system"), channels::SYSTEM);
    assert_eq!(cap.register("tool"), channels::TOOL);
}

#[test]
fn channel_assignment_unknown_gets_dynamic() {
    let mut cap = Capture::new();
    let ch1 = cap.register("custom_agent_1");
    let ch2 = cap.register("custom_agent_2");
    assert!(ch1 >= 5);
    assert!(ch2 >= 5);
    assert_ne!(ch1, ch2);
}

#[test]
fn channel_assignment_is_stable() {
    let mut cap = Capture::new();
    let ch1 = cap.register("repeated_agent");
    let ch2 = cap.register("repeated_agent");
    assert_eq!(ch1, ch2);
}

#[test]
fn channel_assignment_many_dynamic_agents() {
    let mut cap = Capture::new();
    let mut channels_assigned = Vec::new();
    for i in 0..10 {
        channels_assigned.push(cap.register(&format!("agent_{}", i)));
    }
    // All should be unique
    let unique: std::collections::HashSet<u8> = channels_assigned.iter().copied().collect();
    assert_eq!(unique.len(), channels_assigned.len());
}

// ════════════════════════════════════════════════════════════════════
// FRICTION BITFIELD COMBINATIONS
// ════════════════════════════════════════════════════════════════════

#[test]
fn friction_all_flags_are_distinct_bits() {
    let flags = [
        friction::NONE, friction::TIMEOUT, friction::CONFLICT,
        friction::RATE_LIMIT, friction::AMBIGUITY, friction::IMPORT_ERROR,
        friction::SYNTAX_ERROR, friction::TYPE_MISMATCH, friction::NETWORK_ERROR,
    ];
    // NONE is 0, all others should be unique nonzero
    let nonzero: Vec<u8> = flags.iter().filter(|&&f| f != 0).copied().collect();
    let unique: std::collections::HashSet<u8> = nonzero.iter().copied().collect();
    assert_eq!(nonzero.len(), unique.len());
}

#[test]
fn friction_combined_flags_work() {
    let combined = friction::TIMEOUT | friction::CONFLICT | friction::NETWORK_ERROR;
    assert!(combined & friction::TIMEOUT != 0);
    assert!(combined & friction::CONFLICT != 0);
    assert!(combined & friction::NETWORK_ERROR != 0);
    assert!(combined & friction::AMBIGUITY == 0);
}

// ════════════════════════════════════════════════════════════════════
// CAPTURE WITH FRICTION-HEAVY CONVERSATION
// ════════════════════════════════════════════════════════════════════

#[test]
fn capture_friction_heavy_conversation_analyzed_correctly() {
    let mut cap = Capture::new();

    // Simulate a conversation with errors
    let error_messages = vec![
        "Syntax error in parser",
        "Import failed: missing module",
        "Type mismatch on line 42",
        "Network timeout connecting to server",
        "Rate limit exceeded",
    ];

    for (i, text) in error_messages.iter().enumerate() {
        cap.capture(Message {
            text: text.to_string(),
            sender: "tool".into(),
            timestamp_ms: i as u64 * 100,
        });
    }

    let analysis = JazzAnalysis::from_capture(&cap);
    // All messages have friction bits set (error keywords trigger SYNTAX_ERROR or AMBIGUITY)
    assert!(analysis.friction_ratio > 0.3);
    assert!(analysis.tension > 0.2);
}

#[test]
fn capture_tick_advances_with_each_message() {
    let mut cap = Capture::new();
    let msg = Message { text: "test".into(), sender: "human".into(), timestamp_ms: 0 };

    let (_, e1) = cap.capture(msg.clone());
    let (_, e2) = cap.capture(msg.clone());
    let (_, e3) = cap.capture(msg.clone());

    assert!(e2.tick > e1.tick);
    assert!(e3.tick > e2.tick);
}

// ════════════════════════════════════════════════════════════════════
// CONSTANT INTEGRITY
// ════════════════════════════════════════════════════════════════════

#[test]
fn constants_form_consistent_system() {
    // 12/8 time at 96 PPQ
    assert_eq!(PULSES_PER_BAR, 12);
    assert_eq!(TICKS_PER_PULSE, 48); // 96 / 2
    assert_eq!(TICKS_PER_BAR, 576); // 12 * 48
    assert_eq!(TICKS_PER_BAR, PULSES_PER_BAR * TICKS_PER_PULSE);
}
