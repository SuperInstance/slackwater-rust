//! Flow state detection state machine.
//!
//! The [`FlowStateDetector`] consumes Φ (flow friction) readings and tracks
//! the player's progression through the flow lifecycle:
//!
//! ```text
//! OutOfFlow → ApproachingFlow → InFlow → DeepFlow
//!                 ↑__________________|
//! ```
//!
//! Flow is not declared on a single reading. It must sustain for
//! `min_window` consecutive observations. And once detected, transitions
//! back are gradual — not instantaneous.
//!
//! Flow is a soap bubble. You don't grab it. You hold still and let the
//! air do the work.

use serde::{Deserialize, Serialize};

/// The four states of the flow lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlowState {
    /// Player is not in flow. Φ is high or readings are inconsistent.
    OutOfFlow,
    /// Φ is dropping, approaching the flow threshold. The system should
    /// be quiet and observant.
    ApproachingFlow,
    /// Flow has been detected and sustained. The player is in the zone.
    /// Tempo locks, chatter minimizes.
    InFlow,
    /// Flow has persisted well past the threshold. The player may be
    /// losing track of time. The system becomes nearly invisible.
    DeepFlow,
}

impl FlowState {
    /// Returns true if this state represents active flow.
    pub fn is_flow(self) -> bool {
        matches!(self, FlowState::InFlow | FlowState::DeepFlow)
    }
}

impl std::fmt::Display for FlowState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlowState::OutOfFlow => write!(f, "OutOfFlow"),
            FlowState::ApproachingFlow => write!(f, "ApproachingFlow"),
            FlowState::InFlow => write!(f, "InFlow"),
            FlowState::DeepFlow => write!(f, "DeepFlow"),
        }
    }
}

/// Whether Φ (friction) is improving, stable, or declining.
///
/// "Improving" means friction is decreasing (approaching flow).
/// "Declining" means friction is increasing (leaving flow).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowTrend {
    /// Φ is decreasing — flow is improving.
    Improving,
    /// Φ is stable — no clear trend.
    Stable,
    /// Φ is increasing — flow is degrading.
    Declining,
}

/// Flow state detector — consumes Φ readings and tracks the flow lifecycle.
///
/// The detector implements a state machine with hysteresis: it requires
/// `min_window` consecutive readings below the approach threshold before
/// entering flow, and sustained low Φ before declaring deep flow.
///
/// # Examples
///
/// ```
/// use harmony_core::flow_state::{FlowState, FlowStateDetector};
///
/// let mut detector = FlowStateDetector::new();
///
/// // Feed low Φ readings to enter flow.
/// for _ in 0..15 {
///     detector.observe(0.02);
/// }
/// assert!(detector.in_flow());
/// ```
pub struct FlowStateDetector {
    /// Φ below this = flow territory.
    phi_threshold: f64,
    /// Φ below this / 3 = deep flow territory.
    deep_flow_threshold: f64,
    /// Minimum actions/observations before a transition is allowed.
    min_window: usize,
    /// Current flow state.
    state: FlowState,
    /// History of Φ readings for trend analysis.
    phi_history: Vec<f64>,
    /// Counter for sustained readings at the current level.
    sustained_count: usize,
    /// Maximum history to retain.
    max_history: usize,
}

impl FlowStateDetector {
    /// Create a new detector with default parameters.
    ///
    /// Defaults:
    /// - `phi_threshold`: 0.05 (below this = flow)
    /// - `min_window`: 10 (observations needed to confirm a transition)
    /// - `max_history`: 500 readings retained for trend analysis
    pub fn new() -> Self {
        Self {
            phi_threshold: 0.05,
            deep_flow_threshold: 0.05 / 3.0,
            min_window: 10,
            state: FlowState::OutOfFlow,
            phi_history: Vec::with_capacity(500),
            sustained_count: 0,
            max_history: 500,
        }
    }

    /// Create a detector with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `phi_threshold` — Φ below this means flow (default 0.05).
    /// * `min_window` — Minimum observations to confirm a transition (default 10).
    pub fn with_params(phi_threshold: f64, min_window: usize) -> Self {
        Self {
            phi_threshold,
            deep_flow_threshold: phi_threshold / 3.0,
            min_window,
            state: FlowState::OutOfFlow,
            phi_history: Vec::with_capacity(500),
            sustained_count: 0,
            max_history: 500,
        }
    }

    /// Observe a new Φ reading and advance the state machine.
    ///
    /// Returns the current [`FlowState`] after processing this reading.
    pub fn observe(&mut self, phi: f64) -> FlowState {
        self.phi_history.push(phi);
        if self.phi_history.len() > self.max_history {
            self.phi_history.remove(0);
        }

        let prev_state = self.state;

        match self.state {
            FlowState::OutOfFlow => {
                if phi < self.phi_threshold {
                    self.sustained_count += 1;
                    if self.sustained_count >= self.min_window {
                        self.state = FlowState::InFlow;
                        self.sustained_count = 0;
                    } else if self.sustained_count >= self.min_window / 3 {
                        // Start showing approach state.
                        self.state = FlowState::ApproachingFlow;
                    }
                } else if phi < self.phi_threshold * 2.0 {
                    // Getting closer.
                    self.sustained_count += 1;
                    if self.sustained_count >= self.min_window / 2 {
                        self.state = FlowState::ApproachingFlow;
                    }
                } else {
                    self.sustained_count = 0;
                }
            }
            FlowState::ApproachingFlow => {
                if phi < self.phi_threshold {
                    self.sustained_count += 1;
                    if self.sustained_count >= self.min_window {
                        self.state = FlowState::InFlow;
                        self.sustained_count = 0;
                    }
                } else if phi < self.phi_threshold * 2.0 {
                    // Still approaching, not there yet.
                    // Keep the counter but don't reset.
                } else {
                    // Φ went back up.
                    self.sustained_count = 0;
                    self.state = FlowState::OutOfFlow;
                }
            }
            FlowState::InFlow => {
                if phi < self.deep_flow_threshold {
                    self.sustained_count += 1;
                    if self.sustained_count >= self.min_window {
                        self.state = FlowState::DeepFlow;
                        self.sustained_count = 0;
                    }
                } else if phi < self.phi_threshold {
                    // Still in flow, not deep.
                    self.sustained_count = 0;
                } else {
                    // Φ rose above threshold — flow is breaking.
                    self.state = FlowState::ApproachingFlow;
                    self.sustained_count = 0;
                }
            }
            FlowState::DeepFlow => {
                if phi < self.deep_flow_threshold {
                    // Still in deep flow.
                    self.sustained_count = 0;
                } else if phi < self.phi_threshold {
                    // Dropped out of deep but still in flow.
                    self.state = FlowState::InFlow;
                    self.sustained_count = 0;
                } else {
                    // Deep flow broken.
                    self.state = FlowState::ApproachingFlow;
                    self.sustained_count = 0;
                }
            }
        }

        let _ = prev_state; // Available for logging/transitions if needed.
        self.state
    }

    /// Returns true if the player is currently in flow (InFlow or DeepFlow).
    pub fn in_flow(&self) -> bool {
        self.state.is_flow()
    }

    /// Returns the current flow state.
    pub fn state(&self) -> FlowState {
        self.state
    }

    /// Returns the current Φ trend based on recent history.
    ///
    /// Compares the average of the last few readings against the average
    /// of the preceding few. If Φ is decreasing, flow is improving.
    pub fn phi_trend(&self) -> FlowTrend {
        let n = self.phi_history.len();
        if n < 6 {
            return FlowTrend::Stable;
        }

        let recent_len = n.min(3);
        let prev_len = n.min(6) - recent_len;

        let recent_start = n - recent_len;
        let prev_start = recent_start.saturating_sub(prev_len);

        let recent_avg: f64 =
            self.phi_history[recent_start..].iter().sum::<f64>() / recent_len as f64;
        let prev_avg: f64 = if prev_len > 0 {
            self.phi_history[prev_start..recent_start]
                .iter()
                .sum::<f64>()
                / prev_len as f64
        } else {
            recent_avg
        };

        let delta = recent_avg - prev_avg;

        // Threshold: 5% of the current Φ level.
        let threshold = (recent_avg * 0.05).max(0.005);

        if delta < -threshold {
            FlowTrend::Improving
        } else if delta > threshold {
            FlowTrend::Declining
        } else {
            FlowTrend::Stable
        }
    }

    /// Returns the most recent Φ reading, or 1.0 if no observations yet.
    pub fn last_phi(&self) -> f64 {
        self.phi_history.last().copied().unwrap_or(1.0)
    }

    /// Returns the number of observations recorded.
    pub fn observation_count(&self) -> usize {
        self.phi_history.len()
    }

    /// Reset the detector to its initial state.
    pub fn reset(&mut self) {
        self.state = FlowState::OutOfFlow;
        self.phi_history.clear();
        self.sustained_count = 0;
    }
}

impl Default for FlowStateDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starts_out_of_flow() {
        let d = FlowStateDetector::new();
        assert_eq!(d.state(), FlowState::OutOfFlow);
        assert!(!d.in_flow());
    }

    #[test]
    fn test_high_phi_stays_out() {
        let mut d = FlowStateDetector::new();
        for _ in 0..20 {
            d.observe(0.8);
        }
        assert_eq!(d.state(), FlowState::OutOfFlow);
    }

    #[test]
    fn test_progression_to_deep_flow() {
        let mut d = FlowStateDetector::with_params(0.05, 5);

        // Phase 1: Approach — moderate Φ.
        for _ in 0..3 {
            d.observe(0.08);
        }
        // Should be approaching or out of flow (0.08 < 0.05*2 = 0.10).
        assert!(d.state() == FlowState::ApproachingFlow || d.state() == FlowState::OutOfFlow);

        // Phase 2: Enter flow — low Φ sustained.
        for _ in 0..6 {
            d.observe(0.03);
        }
        assert_eq!(d.state(), FlowState::InFlow);
        assert!(d.in_flow());

        // Phase 3: Deep flow — very low Φ sustained.
        for _ in 0..6 {
            d.observe(0.01);
        }
        assert_eq!(d.state(), FlowState::DeepFlow);
    }

    #[test]
    fn test_flow_breaks_on_rising_phi() {
        let mut d = FlowStateDetector::with_params(0.05, 3);

        // Get into flow.
        for _ in 0..5 {
            d.observe(0.02);
        }
        assert!(d.in_flow());

        // Φ rises — flow breaks.
        d.observe(0.5);
        d.observe(0.5);
        assert!(!d.in_flow() || d.state() == FlowState::ApproachingFlow);
    }

    #[test]
    fn test_trend_improving() {
        let mut d = FlowStateDetector::new();
        // High values first, then decreasing.
        for v in [0.8, 0.7, 0.6, 0.3, 0.2, 0.1] {
            d.observe(v);
        }
        assert_eq!(d.phi_trend(), FlowTrend::Improving);
    }

    #[test]
    fn test_trend_declining() {
        let mut d = FlowStateDetector::new();
        // Low values first, then increasing.
        for v in [0.1, 0.1, 0.1, 0.3, 0.5, 0.7] {
            d.observe(v);
        }
        assert_eq!(d.phi_trend(), FlowTrend::Declining);
    }

    #[test]
    fn test_trend_stable() {
        let mut d = FlowStateDetector::new();
        for _ in 0..10 {
            d.observe(0.3);
        }
        assert_eq!(d.phi_trend(), FlowTrend::Stable);
    }

    #[test]
    fn test_trend_insufficient_data() {
        let mut d = FlowStateDetector::new();
        d.observe(0.1);
        d.observe(0.2);
        assert_eq!(d.phi_trend(), FlowTrend::Stable);
    }

    #[test]
    fn test_reset() {
        let mut d = FlowStateDetector::with_params(0.05, 3);
        for _ in 0..10 {
            d.observe(0.01);
        }
        assert!(d.in_flow());

        d.reset();
        assert_eq!(d.state(), FlowState::OutOfFlow);
        assert_eq!(d.observation_count(), 0);
    }

    #[test]
    fn test_last_phi() {
        let mut d = FlowStateDetector::new();
        assert_eq!(d.last_phi(), 1.0); // Default when empty.
        d.observe(0.3);
        d.observe(0.42);
        assert!((d.last_phi() - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_flow_state_display() {
        assert_eq!(FlowState::OutOfFlow.to_string(), "OutOfFlow");
        assert_eq!(FlowState::InFlow.to_string(), "InFlow");
    }
}
