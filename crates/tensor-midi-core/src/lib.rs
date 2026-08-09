#![warn(clippy::all)]
#![deny(unsafe_code)]

//! # tensor-midi-core
//!
//! The 12-pulse conversation-as-jazz engine. Maps conversation messages
//! to SWMIDI events on a 12/8 grid at 96 PPQ, analyzes sentiment and
//! jazz dynamics (tension, energy, complexity), and provides the core
//! data structures that the C, Zig, Python, and CUDA implementations
//! all derive from.
//!
//! ## What Rust teaches the polyformalism
//!
//! The borrow checker forces us to confront lifetimes: the sentiment
//! analyzer works on **borrowed** string slices (`&str`), never allocating
//! copies. The event stream uses a fixed-capacity ring buffer — no
//! heap allocation on the hot path. The conversation capture owns its
//! data, but analysis functions take references. This is the ownership
//! discipline the other implementations must match.

use serde::{Deserialize, Serialize};

// ── Re-exports ──────────────────────────────────────────────────────

pub use swmidi::{EventType, SwmidiEvent, SwmidiStream, PACKED_SIZE};
pub use tempo_core::{BeatClock, MusicalPosition, PPQ, TempoMap};

// ── Constants ───────────────────────────────────────────────────────

/// 12/8 time: 12 eighth-note pulses per bar.
pub const PULSES_PER_BAR: u32 = 12;

/// Ticks per pulse (96 PPQ / 2 = 48 ticks per eighth note).
pub const TICKS_PER_PULSE: u32 = PPQ / 2;

/// Ticks per bar in 12/8 (12 × 48 = 576).
pub const TICKS_PER_BAR: u32 = PULSES_PER_BAR * TICKS_PER_PULSE;

/// Friction bitfield — maps to the SWMIDI error_mask byte.
pub mod friction {
    pub const NONE: u8 = 0x00;
    pub const TIMEOUT: u8 = 0x01;
    pub const CONFLICT: u8 = 0x02;
    pub const RATE_LIMIT: u8 = 0x04;
    pub const AMBIGUITY: u8 = 0x08;
    pub const IMPORT_ERROR: u8 = 0x10;
    pub const SYNTAX_ERROR: u8 = 0x20;
    pub const TYPE_MISMATCH: u8 = 0x40;
    pub const NETWORK_ERROR: u8 = 0x80;
}

// ════════════════════════════════════════════════════════════════════
// SENTIMENT ANALYSIS
// ════════════════════════════════════════════════════════════════════

/// Sentiment categories mapped to musical qualities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SentimentLabel {
    Bright = 0,      // positive, joyful
    Creative = 1,    // imaginative, building
    Inquiring = 2,   // questioning, curious
    Neutral = 3,     // no strong signal
    Tense = 4,       // negative, frustrated
    Resolved = 5,    // conciliatory, grateful
}

impl SentimentLabel {
    /// Whether this sentiment carries tension.
    pub fn is_tense(&self) -> bool {
        matches!(self, Self::Tense)
    }

    /// Whether this sentiment signals creative energy.
    pub fn is_creative(&self) -> bool {
        matches!(self, Self::Creative)
    }
}

/// Result of analyzing a single message's sentiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sentiment {
    /// MIDI pitch (0–127). Higher = brighter/creative, lower = tense.
    pub pitch: u8,
    /// Friction bitfield (error_mask in SWMIDI).
    pub friction: u8,
    /// Weight/confidence (0–127 for velocity).
    pub velocity: u8,
    /// Categorical label.
    pub label: SentimentLabel,
    /// Raw positivity count.
    pub positivity: u8,
    /// Raw negativity count.
    pub negativity: u8,
    /// Raw question score.
    pub question: u8,
    /// Raw creativity score.
    pub creativity: u8,
}

/// Word lists for sentiment analysis. Static, const, no allocation.
///
/// The borrow checker teaches us: these live in static memory, shared
/// by all callers, never copied. The analyzer iterates over borrowed
/// slices of &'static str.
pub const POSITIVE_WORDS: &[&str] = &[
    "great", "awesome", "love", "perfect", "excellent", "wonderful",
    "yes", "good", "amazing", "fantastic", "beautiful", "brilliant",
    "nice", "cool", "happy", "glad", "thanks", "thank", "sweet",
    "perfect", "win", "success", "proud",
];

pub const NEGATIVE_WORDS: &[&str] = &[
    "bad", "error", "fail", "broken", "hate", "wrong", "no", "terrible",
    "awful", "crash", "bug", "issue", "stuck", "frustrated", "annoying",
    "slow", "dead", "lost", "miss", "angry", "sad",
];

pub const QUESTION_WORDS: &[&str] = &[
    "what", "how", "why", "where", "when", "who", "which", "?",
];

pub const CREATIVE_WORDS: &[&str] = &[
    "imagine", "create", "build", "design", "compose", "paint", "draw",
    "write", "dream", "invent", "explore", "craft", "forge", "shape",
    "mold", "weave", "spark",
];

/// Count how many words in `text` appear in `word_list`.
///
/// This is the hot-path function. It takes a borrowed string slice,
/// splits on whitespace, and counts matches. No allocation beyond
/// the split iterator's state.
#[inline]
fn count_matches(text: &str, word_list: &[&str]) -> u8 {
    let mut count: u8 = 0;
    for word in text.split_whitespace() {
        // Simple contains check — handles partial matches like "awesome!" matching "awesome"
        let lower = word.to_ascii_lowercase();
        for &target in word_list {
            if lower.contains(target) {
                count = count.saturating_add(1);
                break; // one match per word max
            }
        }
    }
    count
}

/// Analyze sentiment of a text slice.
///
/// **No allocation on the hot path.** Works entirely on borrowed data.
/// The `to_ascii_lowercase` on each word is the only allocation, and
/// in a future version we could eliminate even that with a case-insensitive
/// comparison.
pub fn analyze_sentiment(text: &str) -> Sentiment {
    let positivity = count_matches(text, POSITIVE_WORDS);
    let negativity = count_matches(text, NEGATIVE_WORDS);
    let question = count_matches(text, QUESTION_WORDS);
    let creativity = count_matches(text, CREATIVE_WORDS);

    // Map to pitch (0–127), centering at 60 (middle C)
    let mut pitch: i32 = 60;
    pitch += creativity as i32 * 8;
    pitch += positivity as i32 * 5;
    pitch -= negativity as i32 * 10;
    if question > 0 {
        pitch = 72 + question as i32 * 3;
    }
    let pitch = pitch.clamp(0, 127) as u8;

    // Friction bitfield
    let mut fr = friction::NONE;
    if negativity > 0 {
        fr |= friction::AMBIGUITY;
    }
    if text.contains("error") || text.contains("fail") || text.contains("crash") {
        fr |= friction::SYNTAX_ERROR;
    }

    // Weight from message length (capped)
    let len = text.len().min(500);
    let velocity = ((len as f32 / 500.0) * 127.0).round() as u8;
    let velocity = velocity.clamp(1, 127);

    // Label
    let label = if negativity > positivity {
        SentimentLabel::Tense
    } else if creativity > 0 {
        SentimentLabel::Creative
    } else if question > 0 {
        SentimentLabel::Inquiring
    } else if positivity > 0 {
        SentimentLabel::Bright
    } else {
        SentimentLabel::Neutral
    };

    Sentiment {
        pitch,
        friction: fr,
        velocity,
        label,
        positivity,
        negativity,
        question,
        creativity,
    }
}

// ════════════════════════════════════════════════════════════════════
// 12-PULSE GRID
// ════════════════════════════════════════════════════════════════════

/// A message event mapped onto the 12-pulse grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridEvent {
    /// The SWMIDI event.
    pub event: SwmidiEvent,
    /// Bar number.
    pub bar: u32,
    /// Pulse within the bar (0–11).
    pub pulse: u8,
    /// Sub-tick within the pulse (0–47).
    pub sub_tick: u8,
}

/// Position within a 12/8 bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PulsePosition {
    pub bar: u32,
    pub pulse: u8,      // 0–11
    pub sub_tick: u8,   // 0–47
}

/// Convert a tick to a pulse position in 12/8 time.
pub const fn tick_to_pulse(tick: u32) -> PulsePosition {
    let bar = tick / TICKS_PER_BAR;
    let within_bar = tick % TICKS_PER_BAR;
    let pulse = (within_bar / TICKS_PER_PULSE) as u8;
    let sub_tick = (within_bar % TICKS_PER_PULSE) as u8;
    PulsePosition { bar, pulse, sub_tick }
}

/// Convert a pulse position back to a tick.
pub const fn pulse_to_tick(pos: PulsePosition) -> u32 {
    pos.bar * TICKS_PER_BAR + pos.pulse as u32 * TICKS_PER_PULSE + pos.sub_tick as u32
}

/// The 12-pulse grid — maps events to bars and pulses.
///
/// Uses a Vec internally, but the ring buffer version (see below) is
/// for real-time paths. This struct is for analysis and rendering.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PulseGrid {
    events: Vec<GridEvent>,
}

impl PulseGrid {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an event to the grid.
    pub fn add(&mut self, event: SwmidiEvent) {
        let pos = tick_to_pulse(event.tick);
        self.events.push(GridEvent {
            event,
            bar: pos.bar,
            pulse: pos.pulse,
            sub_tick: pos.sub_tick,
        });
    }

    /// Get all events in a specific bar.
    pub fn events_in_bar(&self, bar: u32) -> impl Iterator<Item = &GridEvent> {
        self.events.iter().filter(move |e| e.bar == bar)
    }

    /// Get the pulse pattern for a bar (which of the 12 pulses are filled).
    pub fn bar_pattern(&self, bar: u32) -> [bool; 12] {
        let mut pattern = [false; 12];
        for e in self.events_in_bar(bar) {
            if e.pulse < 12 {
                pattern[e.pulse as usize] = true;
            }
        }
        pattern
    }

    /// Density of a bar (fraction of pulses filled, 0.0–1.0).
    pub fn bar_density(&self, bar: u32) -> f32 {
        let pattern = self.bar_pattern(bar);
        let filled = pattern.iter().filter(|&&p| p).count();
        filled as f32 / 12.0
    }

    /// Total event count.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Iterate over all grid events.
    pub fn iter(&self) -> impl Iterator<Item = &GridEvent> {
        self.events.iter()
    }

    /// Sort events by tick.
    pub fn sort_by_tick(&mut self) {
        self.events.sort_by_key(|e| e.event.tick);
    }
}

// ════════════════════════════════════════════════════════════════════
// RING BUFFER — No allocation on the hot path
// ════════════════════════════════════════════════════════════════════

/// A fixed-capacity ring buffer for SWMIDI events.
///
/// The borrow checker's gift: since the buffer owns the data, and we
/// hand out references, we can never have dangling pointers to old
/// events. When the buffer wraps, old events are overwritten in place.
pub const DEFAULT_RING_CAPACITY: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRingBuffer {
    buffer: Vec<SwmidiEvent>,
    capacity: usize,
    head: usize, // next write position
    len: usize,  // current count
}

impl EventRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            len: 0,
        }
    }

    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_RING_CAPACITY)
    }

    /// Push an event. Overwrites oldest if full.
    pub fn push(&mut self, event: SwmidiEvent) {
        if self.len < self.capacity {
            self.buffer.push(event);
            self.head = (self.head + 1) % self.capacity;
            self.len += 1;
        } else {
            self.buffer[self.head] = event;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Current number of events.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Get the most recent event.
    pub fn last(&self) -> Option<&SwmidiEvent> {
        if self.len == 0 {
            None
        } else {
            let idx = if self.head == 0 {
                self.capacity - 1
            } else {
                self.head - 1
            };
            self.buffer.get(idx)
        }
    }

    /// Iterate over events in insertion order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &SwmidiEvent> {
        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };
        self.buffer
            .iter()
            .cycle()
            .skip(start)
            .take(self.len)
    }

    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.head = 0;
        self.len = 0;
    }
}

impl Default for EventRingBuffer {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

// ════════════════════════════════════════════════════════════════════
// CONVERSATION CAPTURE
// ════════════════════════════════════════════════════════════════════

/// A raw conversation message before encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub text: String,
    pub sender: String,
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
}

/// Channel assignment for known participant types.
pub mod channels {
    pub const HUMAN: u8 = 0;
    pub const ASSISTANT: u8 = 1;
    pub const SUBAGENT_1: u8 = 2;
    pub const SUBAGENT_2: u8 = 3;
    pub const SUBAGENT_3: u8 = 4;
    pub const SYSTEM: u8 = 8;
    pub const TOOL: u8 = 9;
    pub const ERROR: u8 = 15;
}

/// The capture system — listens to conversation and produces SWMIDI events.
///
/// Owns: the event ring buffer, the pulse grid, participant map, and beat clock.
/// The ownership model: Capture owns everything, analysis functions borrow.
#[derive(Debug, Clone)]
pub struct Capture {
    ring: EventRingBuffer,
    grid: PulseGrid,
    clock: BeatClock,
    participants: std::collections::HashMap<String, u8>,
    next_channel: u8,
    /// All captured messages with sentiment data.
    messages: Vec<CapturedMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedMessage {
    pub text: String,
    pub sender: String,
    pub timestamp_ms: u64,
    pub sentiment: Sentiment,
    pub channel: u8,
    pub tick: u32,
}

impl Capture {
    pub fn new() -> Self {
        Self {
            ring: EventRingBuffer::with_default_capacity(),
            grid: PulseGrid::new(),
            clock: BeatClock::new(),
            participants: std::collections::HashMap::new(),
            next_channel: 5, // 0–4 reserved
            messages: Vec::new(),
        }
    }

    /// Register a participant and assign them a channel.
    pub fn register(&mut self, name: &str) -> u8 {
        if let Some(&ch) = self.participants.get(name) {
            return ch;
        }
        let channel = match name {
            "human" => channels::HUMAN,
            "assistant" => channels::ASSISTANT,
            "system" => channels::SYSTEM,
            "tool" => channels::TOOL,
            _ => {
                let ch = self.next_channel.min(14);
                self.next_channel = self.next_channel.saturating_add(1);
                ch
            }
        };
        self.participants.insert(name.to_string(), channel);
        channel
    }

    /// Capture a single message.
    pub fn capture(&mut self, msg: Message) -> (Sentiment, SwmidiEvent) {
        let channel = self.register(&msg.sender);
        let sentiment = analyze_sentiment(&msg.text);

        // Map timestamp to tick (simplified: use message count as proxy)
        // In production, this would use the beat clock's tempo map
        let tick = self.clock.tick();
        self.clock.advance(TICKS_PER_PULSE);

        let event = SwmidiEvent::new(
            EventType::NoteOn,
            channel,
            sentiment.pitch,
            sentiment.velocity,
            sentiment.friction,
            tick,
        );

        self.ring.push(event);
        self.grid.add(event);
        self.messages.push(CapturedMessage {
            text: msg.text,
            sender: msg.sender,
            timestamp_ms: msg.timestamp_ms,
            sentiment,
            channel,
            tick,
        });

        (sentiment, event)
    }

    /// Get a reference to the event ring buffer.
    pub fn events(&self) -> &EventRingBuffer {
        &self.ring
    }

    /// Get a reference to the pulse grid.
    pub fn grid(&self) -> &PulseGrid {
        &self.grid
    }

    /// Get a reference to the beat clock.
    pub fn clock(&self) -> &BeatClock {
        &self.clock
    }

    /// Get all captured messages.
    pub fn messages(&self) -> &[CapturedMessage] {
        &self.messages
    }

    /// Encode all events to SWMIDI binary.
    pub fn encode_binary(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.ring.len() * PACKED_SIZE);
        for event in self.ring.iter() {
            buf.extend_from_slice(&event.encode());
        }
        buf
    }

    /// Export as JSON-serializable data.
    pub fn export_data(&self) -> CaptureExport {
        CaptureExport {
            bpm: self.clock.bpm(),
            events: self.ring.iter().copied().collect(),
            messages: self.messages.clone(),
            participants: self.participants.iter()
                .map(|(k, &v)| (k.clone(), v))
                .collect(),
        }
    }

    /// Clear all captured data.
    pub fn clear(&mut self) {
        self.ring.clear();
        self.grid = PulseGrid::new();
        self.messages.clear();
        self.clock.reset();
    }
}

impl Default for Capture {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable export of all capture data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureExport {
    pub bpm: f64,
    pub events: Vec<SwmidiEvent>,
    pub messages: Vec<CapturedMessage>,
    pub participants: std::collections::HashMap<String, u8>,
}

// ════════════════════════════════════════════════════════════════════
// JAZZ ANALYSIS
// ════════════════════════════════════════════════════════════════════

/// Jazz mode — the overall feel of a conversation segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JazzMode {
    Groove,     // flowing, low tension, moderate energy
    Building,   // increasing energy, creative
    Tension,    // high friction, dissonant
    Release,    // resolving, tension dropping
    Solo,       // one dominant voice
    Comping,    // short supportive exchanges
    Free,       // chaotic, unpredictable
    Ballad,     // slow, sparse, emotional
}

impl JazzMode {
    pub fn description(&self) -> &'static str {
        match self {
            Self::Groove => "The ensemble is in the pocket. The groove lives.",
            Self::Building => "Energy is rising. Something is being constructed.",
            Self::Tension => "Friction in the room. Dissonance building.",
            Self::Release => "Tension resolving. The harmony breathes again.",
            Self::Solo => "One voice carries the melody. Others listen.",
            Self::Comping => "Short exchanges. The rhythm section holds it down.",
            Self::Free => "Free improvisation. Anything can happen.",
            Self::Ballad => "Slow, sparse, emotional. Every note matters.",
        }
    }
}

/// Chord quality detected from sentiment distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChordQuality {
    Major7,
    Minor7,
    Dominant7,
    Diminished,
    Augmented,
    Sus4,
}

/// Result of jazz analysis on a capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JazzAnalysis {
    /// Detected mode.
    pub mode: JazzMode,
    /// Tension level (0.0–1.0).
    pub tension: f32,
    /// Energy level (0.0–1.0).
    pub energy: f32,
    /// Complexity — rhythmic density × pitch variety (0.0–1.0).
    pub complexity: f32,
    /// Average pitch across all events.
    pub avg_pitch: f32,
    /// Average velocity across all events.
    pub velocity: f32,
    /// Flow ratio (flow events / total events).
    pub flow_ratio: f32,
    /// Friction ratio (friction events / total events).
    pub friction_ratio: f32,
    /// Dominant chord quality.
    pub chord: ChordQuality,
    /// Number of events analyzed.
    pub event_count: usize,
    /// Number of unique participants.
    pub participant_count: usize,
    /// Human-readable description.
    pub description: String,
}

impl JazzAnalysis {
    /// Analyze a slice of captured messages.
    pub fn from_messages(messages: &[CapturedMessage]) -> Self {
        if messages.is_empty() {
            return Self::default();
        }

        let n = messages.len();
        let mut total_pitch: f32 = 0.0;
        let mut total_velocity: f32 = 0.0;
        let mut tense_count = 0usize;
        let mut creative_count = 0usize;
        let mut friction_events = 0usize;
        let mut flow_events = 0usize;
        let mut unique_pitches = std::collections::HashSet::new();
        let mut unique_participants = std::collections::HashSet::new();

        for msg in messages {
            total_pitch += msg.sentiment.pitch as f32;
            total_velocity += msg.sentiment.velocity as f32;
            if msg.sentiment.label.is_tense() {
                tense_count += 1;
            }
            if msg.sentiment.label.is_creative() {
                creative_count += 1;
            }
            if msg.sentiment.friction != 0 {
                friction_events += 1;
            } else {
                flow_events += 1;
            }
            unique_pitches.insert(msg.sentiment.pitch);
            unique_participants.insert(&msg.sender);
        }

        let avg_pitch = total_pitch / n as f32;
        let tension = tense_count as f32 / n as f32;
        let energy = total_velocity / (n as f32 * 127.0);
        let complexity = (unique_pitches.len() as f32 / 12.0) * (n as f32 / 20.0).min(1.0);
        let flow_ratio = flow_events as f32 / n as f32;
        let friction_ratio = friction_events as f32 / n as f32;

        // Mode detection
        let mode = if tension > 0.5 {
            JazzMode::Tension
        } else if creative_count > 0 && energy > 0.5 {
            JazzMode::Building
        } else if friction_ratio > 0.3 {
            JazzMode::Free
        } else if tension > 0.2 {
            JazzMode::Release
        } else if flow_ratio > 0.7 && (n >= 5 || unique_participants.len() > 1) {
            JazzMode::Groove
        } else if unique_participants.len() <= 1 {
            JazzMode::Solo
        } else if energy < 0.4 && n < 10 {
            JazzMode::Comping
        } else if flow_ratio > 0.8 {
            JazzMode::Ballad
        } else {
            JazzMode::Groove
        };

        // Chord quality
        let chord = if tension > 0.5 {
            ChordQuality::Diminished
        } else if avg_pitch > 75.0 {
            ChordQuality::Augmented
        } else if friction_ratio > 0.3 {
            ChordQuality::Dominant7
        } else if creative_count > 0 && tension < 0.3 {
            ChordQuality::Major7
        } else if tension > 0.2 {
            ChordQuality::Minor7
        } else {
            ChordQuality::Sus4
        };

        let description = format!(
            "The ensemble is {}. The harmony lives in {}. Tension: {:.0}%. Energy: {:.0}%.",
            match mode {
                JazzMode::Groove => "in the pocket",
                JazzMode::Building => "building energy",
                JazzMode::Tension => "in the tension",
                JazzMode::Release => "finding release",
                JazzMode::Solo => "in solo flight",
                JazzMode::Comping => "comping softly",
                JazzMode::Free => "in free improvisation",
                JazzMode::Ballad => "in ballad territory",
            },
            format!("{:?}", chord),
            tension * 100.0,
            energy * 100.0,
        );

        Self {
            mode,
            tension,
            energy,
            complexity,
            avg_pitch,
            velocity: total_velocity / n as f32,
            flow_ratio,
            friction_ratio,
            chord,
            event_count: n,
            participant_count: unique_participants.len(),
            description,
        }
    }

    /// Analyze directly from a capture.
    pub fn from_capture(capture: &Capture) -> Self {
        Self::from_messages(capture.messages())
    }
}

impl Default for JazzAnalysis {
    fn default() -> Self {
        Self {
            mode: JazzMode::Ballad,
            tension: 0.0,
            energy: 0.0,
            complexity: 0.0,
            avg_pitch: 60.0,
            velocity: 0.0,
            flow_ratio: 1.0,
            friction_ratio: 0.0,
            chord: ChordQuality::Major7,
            event_count: 0,
            participant_count: 0,
            description: "Silence. No messages to analyze.".to_string(),
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// TEMPO DETECTION
// ════════════════════════════════════════════════════════════════════

/// Detect tempo (BPM) from message timestamps.
///
/// Uses median inter-message interval to infer conversational pacing.
pub fn detect_tempo(timestamps_ms: &[u64]) -> f64 {
    if timestamps_ms.len() < 2 {
        return 120.0;
    }

    let mut intervals: Vec<u64> = Vec::with_capacity(timestamps_ms.len() - 1);
    let mut sorted = timestamps_ms.to_vec();
    sorted.sort_unstable();
    for i in 1..sorted.len() {
        if sorted[i] > sorted[i - 1] {
            intervals.push(sorted[i] - sorted[i - 1]);
        }
    }

    if intervals.is_empty() {
        return 120.0;
    }

    intervals.sort_unstable();
    let median = intervals[intervals.len() / 2];

    match median {
        0..=99 => 240.0,
        100..=249 => 180.0,
        250..=499 => 140.0,
        500..=999 => 120.0,
        1000..=1999 => 90.0,
        2000..=4999 => 60.0,
        _ => 40.0,
    }
}

// ════════════════════════════════════════════════════════════════════
// TESTS
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sentiment tests ──

    #[test]
    fn test_positive_sentiment() {
        let s = analyze_sentiment("This is great and wonderful!");
        assert!(s.positivity >= 2);
        assert_eq!(s.label, SentimentLabel::Bright);
        assert!(s.pitch > 60);
    }

    #[test]
    fn test_negative_sentiment() {
        let s = analyze_sentiment("This is terrible and broken");
        assert!(s.negativity >= 2);
        assert_eq!(s.label, SentimentLabel::Tense);
        assert!(s.pitch < 60);
        assert!(s.friction & friction::AMBIGUITY != 0);
    }

    #[test]
    fn test_creative_sentiment() {
        let s = analyze_sentiment("Let's build and design something amazing");
        assert!(s.creativity >= 2);
        assert_eq!(s.label, SentimentLabel::Creative);
    }

    #[test]
    fn test_question_sentiment() {
        let s = analyze_sentiment("What is this? How does it work?");
        assert!(s.question >= 2);
        assert_eq!(s.label, SentimentLabel::Inquiring);
        assert!(s.pitch >= 72);
    }

    #[test]
    fn test_neutral_sentiment() {
        let s = analyze_sentiment("The file was updated");
        assert_eq!(s.label, SentimentLabel::Neutral);
        assert!((55..=65).contains(&s.pitch));
    }

    #[test]
    fn test_error_triggers_syntax_friction() {
        let s = analyze_sentiment("The build failed with an error");
        assert!(s.friction & friction::SYNTAX_ERROR != 0);
    }

    #[test]
    fn test_pitch_clamped() {
        let s = analyze_sentiment("bad bad bad bad bad bad bad bad bad bad");
        assert_eq!(s.pitch, 0); // should clamp at 0

        let s = analyze_sentiment("great great great great great great great great great great");
        // With positivity 10: 60 + 50 = 110, clamped
        assert!(s.pitch <= 127);
    }

    #[test]
    fn test_velocity_from_length() {
        let short = analyze_sentiment("hi");
        let long = analyze_sentiment(&"word ".repeat(200));
        assert!(long.velocity > short.velocity);
    }

    // ── Pulse grid tests ──

    #[test]
    fn test_tick_to_pulse_round_trip() {
        let tick = 1234;
        let pos = tick_to_pulse(tick);
        assert_eq!(pulse_to_tick(pos), tick);
    }

    #[test]
    fn test_pulse_bar_zero() {
        let pos = tick_to_pulse(0);
        assert_eq!(pos.bar, 0);
        assert_eq!(pos.pulse, 0);
        assert_eq!(pos.sub_tick, 0);
    }

    #[test]
    fn test_pulse_bar_one() {
        // One bar = 576 ticks
        let pos = tick_to_pulse(TICKS_PER_BAR);
        assert_eq!(pos.bar, 1);
        assert_eq!(pos.pulse, 0);
    }

    #[test]
    fn test_pulse_six() {
        // Pulse 6 = tick 288 (middle of bar)
        let pos = tick_to_pulse(6 * TICKS_PER_PULSE);
        assert_eq!(pos.pulse, 6);
        assert_eq!(pos.bar, 0);
    }

    #[test]
    fn test_pulse_grid_add_and_query() {
        let mut grid = PulseGrid::new();
        grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
        grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 64, 100, 0, 48));
        grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 67, 100, 0, 576));

        assert_eq!(grid.len(), 3);
        let pattern = grid.bar_pattern(0);
        assert!(pattern[0]); // pulse 0
        assert!(pattern[1]); // pulse 1
        assert!(!pattern[2]); // pulse 2 empty
    }

    #[test]
    fn test_bar_density() {
        let mut grid = PulseGrid::new();
        for i in 0..6 {
            grid.add(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, i * TICKS_PER_PULSE));
        }
        // 6 of 12 pulses filled = 0.5 density
        assert!((grid.bar_density(0) - 0.5).abs() < 0.01);
    }

    // ── Ring buffer tests ──

    #[test]
    fn test_ring_buffer_basic() {
        let mut rb = EventRingBuffer::new(4);
        assert!(rb.is_empty());

        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 64, 100, 0, 48));
        assert_eq!(rb.len(), 2);
        assert!(!rb.is_full());
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut rb = EventRingBuffer::new(3);
        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 61, 100, 0, 1));
        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 62, 100, 0, 2));
        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 63, 100, 0, 3)); // overwrites oldest

        assert_eq!(rb.len(), 3);
        assert!(rb.is_full());
        // First event (pitch 60) should be overwritten
        let pitches: Vec<u8> = rb.iter().map(|e| e.pitch).collect();
        assert!(!pitches.contains(&60));
        assert!(pitches.contains(&63));
    }

    #[test]
    fn test_ring_buffer_last() {
        let mut rb = EventRingBuffer::new(4);
        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 60, 100, 0, 0));
        rb.push(SwmidiEvent::new(EventType::NoteOn, 0, 64, 100, 0, 48));
        assert_eq!(rb.last().unwrap().pitch, 64);
    }

    // ── Capture tests ──

    #[test]
    fn test_capture_basic() {
        let mut cap = Capture::new();
        let (sentiment, event) = cap.capture(Message {
            text: "Hello world".to_string(),
            sender: "human".to_string(),
            timestamp_ms: 1000,
        });

        assert_eq!(event.channel, channels::HUMAN);
        assert!(sentiment.pitch <= 127);
        assert_eq!(cap.messages().len(), 1);
    }

    #[test]
    fn test_capture_multiple_messages() {
        let mut cap = Capture::new();
        cap.capture(Message {
            text: "Great work everyone!".to_string(),
            sender: "human".to_string(),
            timestamp_ms: 0,
        });
        cap.capture(Message {
            text: "Thanks! Let's build something amazing".to_string(),
            sender: "assistant".to_string(),
            timestamp_ms: 500,
        });
        cap.capture(Message {
            text: "I'll design the architecture".to_string(),
            sender: "subagent1".to_string(),
            timestamp_ms: 1000,
        });

        assert_eq!(cap.messages().len(), 3);
        assert_eq!(cap.events().len(), 3);
        assert!(!cap.grid().is_empty());
    }

    #[test]
    fn test_capture_encode_binary() {
        let mut cap = Capture::new();
        cap.capture(Message {
            text: "Test".to_string(),
            sender: "human".to_string(),
            timestamp_ms: 0,
        });

        let binary = cap.encode_binary();
        assert_eq!(binary.len(), PACKED_SIZE);
        // Decode and verify
        let arr: [u8; PACKED_SIZE] = binary[..].try_into().unwrap();
        let decoded = SwmidiEvent::decode(&arr).unwrap();
        assert_eq!(decoded.channel, channels::HUMAN);
    }

    // ── Jazz analysis tests ──

    #[test]
    fn test_jazz_empty() {
        let analysis = JazzAnalysis::default();
        assert_eq!(analysis.mode, JazzMode::Ballad);
        assert_eq!(analysis.event_count, 0);
    }

    #[test]
    fn test_jazz_positive_messages() {
        let mut cap = Capture::new();
        for i in 0..5 {
            cap.capture(Message {
                text: format!("Great job on build {}!", i),
                sender: "human".to_string(),
                timestamp_ms: i * 1000,
            });
        }
        let analysis = JazzAnalysis::from_capture(&cap);
        assert!(analysis.tension < 0.3);
        assert!(analysis.flow_ratio > 0.5);
        assert_eq!(analysis.mode, JazzMode::Groove);
    }

    #[test]
    fn test_jazz_tense_messages() {
        let mut cap = Capture::new();
        for i in 0..5 {
            cap.capture(Message {
                text: format!("Error: build {} failed badly", i),
                sender: "tool".to_string(),
                timestamp_ms: i * 1000,
            });
        }
        let analysis = JazzAnalysis::from_capture(&cap);
        assert!(analysis.tension > 0.5);
        assert!(analysis.friction_ratio > 0.3);
    }

    #[test]
    fn test_jazz_description_nonempty() {
        let mut cap = Capture::new();
        cap.capture(Message {
            text: "Hello".to_string(),
            sender: "human".to_string(),
            timestamp_ms: 0,
        });
        let analysis = JazzAnalysis::from_capture(&cap);
        assert!(!analysis.description.is_empty());
    }

    // ── Tempo detection tests ──

    #[test]
    fn test_detect_tempo_default() {
        assert_eq!(detect_tempo(&[]), 120.0);
        assert_eq!(detect_tempo(&[1000]), 120.0);
    }

    #[test]
    fn test_detect_tempo_fast() {
        let ts = vec![0, 100, 200, 300, 400];
        let bpm = detect_tempo(&ts);
        assert!(bpm > 120.0);
    }

    #[test]
    fn test_detect_tempo_slow() {
        let ts = vec![0, 5000, 10000, 15000];
        let bpm = detect_tempo(&ts);
        assert!(bpm < 100.0);
    }

    // ── Constants tests ──

    #[test]
    fn test_constants_12_8() {
        assert_eq!(PULSES_PER_BAR, 12);
        assert_eq!(TICKS_PER_PULSE, 48);
        assert_eq!(TICKS_PER_BAR, 576);
        assert_eq!(PPQ, 96);
    }

    #[test]
    fn test_friction_flags() {
        assert_eq!(friction::NONE, 0);
        assert!(friction::TIMEOUT != friction::CONFLICT);
        let combined = friction::TIMEOUT | friction::NETWORK_ERROR;
        assert!(combined & friction::TIMEOUT != 0);
        assert!(combined & friction::NETWORK_ERROR != 0);
    }
}
