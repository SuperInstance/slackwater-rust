#![warn(clippy::all)]
#![deny(unsafe_code)]

//! # tminus-core
//!
//! Layer 5: T-Minus prediction and calibration math.
//!
//! Named after the countdown — T-Minus predicts where an agent *should* be
//! by a given tick, compares it to where the agent *actually is*, and
//! produces a calibration signal. The calibration is INT8 (0–127): 0 means
//! "perfectly on time," 127 means "catastrophically out of sync."
//!
//! This is the engine that lets the fleet self-synchronize without a
//! central conductor. Each agent runs T-Minus locally, compares its
//! predicted position to its peers' actual positions, and adjusts.
//!
//! ## Core types
//!
//! - [`Prediction`] — Where an agent should be at a future tick.
//! - [`Calibration`] — The gap between prediction and reality, as INT8.
//! - [`CalibrationHistory`] — Rolling window of calibration readings.
//! - [`TMinusEngine`] — Stateful predictor with exponential smoothing.
//!
//! ## Design principles
//!
//! 1. **Calibration is INT8.** No float drift in agreement paths.
//! 2. **Predictions decay.** A prediction far in the future has higher
//!    uncertainty than one for the next tick.
//! 3. **History is bounded.** The rolling window prevents memory growth
//!    and gives recent readings more weight.

use serde::{Deserialize, Serialize};

/// Maximum calibration window size.
const MAX_WINDOW: usize = 64;

/// A prediction of where an agent should be at a future tick.
///
/// `expected_tick` is where the model thinks the agent will be.
/// `confidence` is INT8 (0–127): how sure the model is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Prediction {
    /// The tick this prediction targets.
    pub target_tick: u32,
    /// The expected position/value at that tick.
    pub expected_value: i32,
    /// Confidence in this prediction (0–127).
    pub confidence: u8,
}

impl Prediction {
    /// Create a new prediction.
    pub const fn new(target_tick: u32, expected_value: i32, confidence: u8) -> Self {
        Self {
            target_tick,
            expected_value,
            confidence,
        }
    }

    /// Decay confidence based on temporal distance.
    ///
    /// The further into the future, the less confident we are.
    /// Uses a simple linear decay: each tick of distance reduces
    /// confidence by 1, floored at 0.
    pub fn decayed(&self, current_tick: u32) -> Self {
        let distance = self.target_tick.saturating_sub(current_tick);
        let decay = (distance / 8).min(127) as u8; // decay 1 per 8 ticks
        let new_confidence = self.confidence.saturating_sub(decay);
        Self {
            target_tick: self.target_tick,
            expected_value: self.expected_value,
            confidence: new_confidence,
        }
    }
}

/// Calibration reading: the gap between prediction and reality.
///
/// `delta` is signed — positive means the agent is ahead of prediction,
/// negative means behind. `magnitude` is the INT8 calibration signal
/// (0 = perfect, 127 = worst).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Calibration {
    /// The tick this calibration was measured at.
    pub tick: u32,
    /// Signed delta: actual - expected. Positive = ahead.
    pub delta: i32,
    /// Unsigned magnitude (0–127). 0 = perfectly calibrated.
    pub magnitude: u8,
}

impl Calibration {
    /// Compute a calibration from expected and actual values.
    pub fn measure(tick: u32, expected: i32, actual: i32) -> Self {
        let delta = actual - expected;
        let magnitude = delta.unsigned_abs().min(127) as u8;
        Self {
            tick,
            delta,
            magnitude,
        }
    }

    /// Is this calibration within an acceptable threshold?
    pub fn within(&self, threshold: u8) -> bool {
        self.magnitude <= threshold
    }

    /// Is this a perfect calibration (zero delta)?
    pub fn is_perfect(&self) -> bool {
        self.magnitude == 0
    }

    /// Classify the calibration severity.
    pub fn severity(&self) -> CalibrationSeverity {
        match self.magnitude {
            0..=5 => CalibrationSeverity::Flow,
            6..=20 => CalibrationSeverity::Minor,
            21..=50 => CalibrationSeverity::Moderate,
            51..=100 => CalibrationSeverity::Significant,
            _ => CalibrationSeverity::Critical,
        }
    }
}

/// Severity classification of a calibration reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CalibrationSeverity {
    /// 0–5: In flow. No action needed.
    Flow,
    /// 6–20: Slight drift. Monitor.
    Minor,
    /// 21–50: Noticeable drift. Consider adjustment.
    Moderate,
    /// 51–100: Significant drift. Adjustment recommended.
    Significant,
    /// 101–127: Critical. Immediate correction required.
    Critical,
}

impl CalibrationSeverity {
    /// Whether this severity requires immediate action.
    pub fn requires_action(&self) -> bool {
        matches!(self, Self::Significant | Self::Critical)
    }
}

/// Rolling window of calibration readings.
///
/// Keeps the last N readings (up to [`MAX_WINDOW`]). Used for
/// computing trends and smoothed calibration values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationHistory {
    readings: Vec<Calibration>,
}

impl Default for CalibrationHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self {
            readings: Vec::with_capacity(MAX_WINDOW),
        }
    }

    /// Push a new reading. Automatically trims to MAX_WINDOW.
    pub fn push(&mut self, reading: Calibration) {
        self.readings.push(reading);
        if self.readings.len() > MAX_WINDOW {
            self.readings.remove(0);
        }
    }

    /// Number of readings in the window.
    pub fn len(&self) -> usize {
        self.readings.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.readings.is_empty()
    }

    /// Average magnitude of recent readings (0–127).
    ///
    /// Returns 0 if empty.
    pub fn average_magnitude(&self) -> u8 {
        if self.readings.is_empty() {
            return 0;
        }
        let sum: u64 = self.readings.iter().map(|r| r.magnitude as u64).sum();
        (sum / self.readings.len() as u64).min(127) as u8
    }

    /// The most recent reading, if any.
    pub fn latest(&self) -> Option<&Calibration> {
        self.readings.last()
    }

    /// Trend: positive = drifting ahead, negative = falling behind.
    ///
    /// Compares the average delta of the first half of the window
    /// to the second half. Returns 0 if insufficient data.
    pub fn trend(&self) -> i32 {
        if self.readings.len() < 4 {
            return 0;
        }
        let mid = self.readings.len() / 2;
        let first_half_avg: f64 = self.readings[..mid]
            .iter()
            .map(|r| r.delta as f64)
            .sum::<f64>()
            / mid as f64;
        let second_half_avg: f64 = self.readings[mid..]
            .iter()
            .map(|r| r.delta as f64)
            .sum::<f64>()
            / (self.readings.len() - mid) as f64;
        (second_half_avg - first_half_avg) as i32
    }

    /// Whether calibration is improving (trend magnitude decreasing).
    pub fn is_improving(&self) -> bool {
        if self.readings.len() < 4 {
            return false;
        }
        let mid = self.readings.len() / 2;
        let first_avg: f64 = self.readings[..mid]
            .iter()
            .map(|r| r.magnitude as f64)
            .sum::<f64>()
            / mid as f64;
        let second_avg: f64 = self.readings[mid..]
            .iter()
            .map(|r| r.magnitude as f64)
            .sum::<f64>()
            / (self.readings.len() - mid) as f64;
        second_avg < first_avg
    }

    /// Iterate over all readings.
    pub fn iter(&self) -> impl Iterator<Item = &Calibration> {
        self.readings.iter()
    }
}

/// The T-Minus prediction engine.
///
/// Uses exponential smoothing to predict future values based on
/// observed history. The smoothing factor (α) controls how quickly
/// the engine adapts to changes: higher α = more reactive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TMinusEngine {
    /// Exponential smoothing factor (0.0–1.0).
    alpha: f64,
    /// Last smoothed value.
    smoothed_value: f64,
    /// Last smoothed rate of change.
    smoothed_rate: f64,
    /// Last tick we observed.
    last_tick: u32,
    /// Calibration history.
    history: CalibrationHistory,
    /// Whether the engine has been initialized.
    initialized: bool,
}

impl TMinusEngine {
    /// Create a new engine with smoothing factor α.
    ///
    /// Panics if alpha is not in [0.0, 1.0].
    pub fn new(alpha: f64) -> Self {
        assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0.0, 1.0]");
        Self {
            alpha,
            smoothed_value: 0.0,
            smoothed_rate: 0.0,
            last_tick: 0,
            history: CalibrationHistory::new(),
            initialized: false,
        }
    }

    /// Create an engine with default α = 0.3 (moderate reactivity).
    pub fn default_engine() -> Self {
        Self::new(0.3)
    }

    /// Observe a new actual value at a tick.
    ///
    /// Updates internal smoothing and pushes a calibration reading.
    pub fn observe(&mut self, tick: u32, actual_value: i32) -> Calibration {
        let calibration = if !self.initialized {
            // First observation — no prediction possible
            self.smoothed_value = actual_value as f64;
            self.smoothed_rate = 0.0;
            self.initialized = true;
            Calibration::measure(tick, actual_value, actual_value)
        } else {
            let predicted = self.predict(tick);
            let calibration = Calibration::measure(tick, predicted.expected_value, actual_value);

            // Update smoothed value (exponential smoothing)
            let delta_ticks = (tick - self.last_tick).max(1) as f64;
            let observed_rate = (actual_value as f64 - self.smoothed_value) / delta_ticks;
            self.smoothed_value =
                self.alpha * actual_value as f64 + (1.0 - self.alpha) * self.smoothed_value;
            self.smoothed_rate =
                self.alpha * observed_rate + (1.0 - self.alpha) * self.smoothed_rate;

            calibration
        };

        self.last_tick = tick;
        self.history.push(calibration);
        calibration
    }

    /// Predict the value at a future tick.
    ///
    /// Uses the smoothed rate of change extrapolated from the last observation.
    pub fn predict(&self, target_tick: u32) -> Prediction {
        if !self.initialized {
            return Prediction::new(target_tick, 0, 0);
        }

        let delta_ticks = (target_tick - self.last_tick) as f64;
        let predicted = self.smoothed_value + self.smoothed_rate * delta_ticks;

        // Confidence decays with distance
        let distance = target_tick.saturating_sub(self.last_tick);
        let confidence = (127u8).saturating_sub((distance / 16).min(127) as u8);

        Prediction::new(target_tick, predicted.round() as i32, confidence)
    }

    /// Get a reference to the calibration history.
    pub fn history(&self) -> &CalibrationHistory {
        &self.history
    }

    /// Current smoothed value.
    pub fn smoothed_value(&self) -> f64 {
        self.smoothed_value
    }

    /// Current smoothed rate of change.
    pub fn smoothed_rate(&self) -> f64 {
        self.smoothed_rate
    }

    /// Whether the engine has been initialized with at least one observation.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_perfect() {
        let cal = Calibration::measure(0, 100, 100);
        assert!(cal.is_perfect());
        assert_eq!(cal.severity(), CalibrationSeverity::Flow);
    }

    #[test]
    fn calibration_ahead() {
        let cal = Calibration::measure(0, 100, 110);
        assert_eq!(cal.delta, 10);
        assert_eq!(cal.magnitude, 10);
        assert_eq!(cal.severity(), CalibrationSeverity::Minor);
    }

    #[test]
    fn calibration_behind() {
        let cal = Calibration::measure(0, 100, 50);
        assert_eq!(cal.delta, -50);
        assert_eq!(cal.magnitude, 50);
        assert_eq!(cal.severity(), CalibrationSeverity::Moderate);
    }

    #[test]
    fn calibration_clamped() {
        let cal = Calibration::measure(0, 0, 1000);
        assert_eq!(cal.magnitude, 127);
        assert_eq!(cal.severity(), CalibrationSeverity::Critical);
        assert!(cal.severity().requires_action());
    }

    #[test]
    fn calibration_within_threshold() {
        let cal = Calibration::measure(0, 100, 103);
        assert!(cal.within(5));
        assert!(!cal.within(2));
    }

    #[test]
    fn prediction_decay() {
        let pred = Prediction::new(100, 50, 127);
        let decayed = pred.decayed(0);
        // distance = 100, decay = 100/8 = 12
        assert_eq!(decayed.confidence, 127 - 12);
    }

    #[test]
    fn prediction_decay_at_target() {
        let pred = Prediction::new(50, 50, 127);
        let decayed = pred.decayed(50);
        // distance = 0, no decay
        assert_eq!(decayed.confidence, 127);
    }

    #[test]
    fn history_average() {
        let mut hist = CalibrationHistory::new();
        hist.push(Calibration::measure(0, 0, 10));
        hist.push(Calibration::measure(1, 0, 20));
        hist.push(Calibration::measure(2, 0, 30));
        assert_eq!(hist.average_magnitude(), 20);
    }

    #[test]
    fn history_trend_positive() {
        let mut hist = CalibrationHistory::new();
        // Delta increasing — agent is drifting ahead over time
        for i in 0..10i32 {
            hist.push(Calibration::measure(i as u32, 0, i * 5));
        }
        assert!(hist.trend() > 0);
    }

    #[test]
    fn history_trend_negative() {
        let mut hist = CalibrationHistory::new();
        // Delta decreasing — agent is falling behind
        for i in 0..10i32 {
            hist.push(Calibration::measure(i as u32, 0, -i * 5));
        }
        assert!(hist.trend() < 0);
    }

    #[test]
    fn history_is_improving() {
        let mut hist = CalibrationHistory::new();
        // Start bad, get better
        for i in 0..5 {
            hist.push(Calibration::measure(i, 0, 100));
        }
        for i in 5..10 {
            hist.push(Calibration::measure(i, 0, 10));
        }
        assert!(hist.is_improving());
    }

    #[test]
    fn engine_predicts_linear_growth() {
        let mut engine = TMinusEngine::new(0.5);
        // Observe a linear sequence: 0, 10, 20, 30 at ticks 0, 1, 2, 3
        engine.observe(0, 0);
        engine.observe(1, 10);
        engine.observe(2, 20);
        engine.observe(3, 30);

        // Predict tick 4 — should be around 40
        let pred = engine.predict(4);
        assert!(pred.expected_value > 25 && pred.expected_value < 55);
    }

    #[test]
    fn engine_first_observation_is_perfect() {
        let mut engine = TMinusEngine::default_engine();
        let cal = engine.observe(0, 42);
        assert!(cal.is_perfect());
        assert!(engine.is_initialized());
    }

    #[test]
    fn engine_uninitialized_predicts_zero() {
        let engine = TMinusEngine::default_engine();
        let pred = engine.predict(100);
        assert_eq!(pred.expected_value, 0);
        assert_eq!(pred.confidence, 0);
    }

    #[test]
    fn history_window_trims() {
        let mut hist = CalibrationHistory::new();
        for i in 0..(MAX_WINDOW + 20) {
            hist.push(Calibration::measure(i as u32, 0, 5));
        }
        assert_eq!(hist.len(), MAX_WINDOW);
    }
}
