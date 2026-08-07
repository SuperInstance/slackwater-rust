//! Φ (phi) — flow friction computation.
//!
//! Φ measures how much friction exists in the player's experience.
//! When Φ → 0, the player is in flow state — the system should suppress,
//! not augment. When Φ is high, the player is struggling — the system
//! should help.
//!
//! ## Formula
//!
//! ```text
//! Φ = w1 * persistence_friction     // 1 - max(0, hurst - 0.5) / 0.5
//!   + w2 * entropy_friction          // normalized_entropy(intervals)
//!   + w3 * cadence_friction          // 1 - cadence_regularity(intervals)
//!   + w4 * idle_penalty              // idle_ratio
//! ```
//!
//! Each component is in [0, 1] where 0 = flow-conducive and 1 = friction.
//! Lower Φ = more flow. Higher Φ = more friction.

use crate::cadence::cadence_regularity;
use crate::entropy::normalized_entropy;
use crate::hurst::hurst_exponent;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Weights for the Φ (flow friction) computation.
///
/// All weights should sum to 1.0. Each weight controls how much its
/// respective signal contributes to the total friction.
///
/// # Default Weights
///
/// | Component    | Weight | Rationale |
/// |--------------|--------|-----------|
/// | Persistence  | 0.35   | Hurst exponent > 0.5 is the strongest flow signal |
/// | Entropy      | 0.25   | Action regularity matters but less than trending |
/// | Cadence      | 0.25   | Timing regularity complements entropy |
/// | Idle         | 0.15   | Idle time breaks flow but is common |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiWeights {
    /// Weight for the persistence (Hurst) component.
    pub persistence: f64,
    /// Weight for the entropy component.
    pub entropy: f64,
    /// Weight for the cadence component.
    pub cadence: f64,
    /// Weight for the idle penalty.
    pub idle: f64,
}

impl Default for PhiWeights {
    fn default() -> Self {
        Self {
            persistence: 0.35,
            entropy: 0.25,
            cadence: 0.25,
            idle: 0.15,
        }
    }
}

impl PhiWeights {
    /// Verify that weights sum to approximately 1.0.
    pub fn is_normalized(&self) -> bool {
        let sum = self.persistence + self.entropy + self.cadence + self.idle;
        (sum - 1.0).abs() < 1e-6
    }
}

/// Compute Φ (phi) — flow friction — for a window of player actions.
///
/// Φ represents the total cognitive friction the player is experiencing.
/// When Φ → 0, the player is in flow. When Φ → 1, the player is struggling.
///
/// # Arguments
///
/// * `action_intervals` — Inter-action time intervals (seconds between actions).
///   Used for Hurst, entropy, and cadence computation.
/// * `idle_ratio` — Fraction of time spent idle (0.0 = constantly active,
///   1.0 = completely idle).
/// * `weights` — Component weights (use [`PhiWeights::default`] for standard weights).
///
/// # Returns
///
/// Φ in [0, 1]. Lower is better (less friction = more flow).
///
/// # Examples
///
/// ```
/// use harmony_core::phi::{compute_phi, PhiWeights};
///
/// // Regular, fast actions → low Φ (flow).
/// let regular: Vec<f64> = (0..100).map(|_| 0.5).collect();
/// let phi = compute_phi(&regular, 0.0, &PhiWeights::default());
/// assert!(phi < 0.3, "regular actions should have low Φ, got {phi}");
///
/// // Irregular actions → high Φ.
/// let irregular: Vec<f64> = vec![0.1, 5.0, 0.2, 8.0, 0.3, 10.0, 0.1, 7.0];
/// let phi = compute_phi(&irregular, 0.5, &PhiWeights::default());
/// assert!(phi > 0.3, "irregular actions should have high Φ, got {phi}");
/// ```
pub fn compute_phi(action_intervals: &[f64], idle_ratio: f64, weights: &PhiWeights) -> f64 {
    if action_intervals.is_empty() {
        // No actions = maximum friction (player is stuck or absent).
        return 1.0;
    }

    // Zero-variance check: perfectly regular intervals = perfect cadence = flow.
    let mean = action_intervals.iter().sum::<f64>() / action_intervals.len() as f64;
    let variance = action_intervals
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / action_intervals.len() as f64;

    let (persistence_friction, entropy_friction, cadence_friction);

    if variance < 1e-12 {
        // Perfectly metronomic — maximum flow indicators.
        persistence_friction = 0.0;
        entropy_friction = 0.0;
        cadence_friction = 0.0;
    } else {
        // Persistence component: hurst > 0.5 means trending (flow).
        let hurst = hurst_exponent(action_intervals);
        persistence_friction = 1.0 - (hurst - 0.5).max(0.0) / 0.5;

        // Entropy component: high entropy = high friction.
        entropy_friction = normalized_entropy(action_intervals);

        // Cadence component: irregular cadence = high friction.
        cadence_friction = 1.0 - cadence_regularity(action_intervals);
    }

    // Idle penalty: directly proportional to idle ratio.
    let idle_penalty = idle_ratio.clamp(0.0, 1.0);

    let phi = weights.persistence * persistence_friction
        + weights.entropy * entropy_friction
        + weights.cadence * cadence_friction
        + weights.idle * idle_penalty;

    phi.clamp(0.0, 1.0)
}

/// Batch compute Φ over a sliding window (parallelized with rayon).
///
/// Takes raw action timestamps and computes Φ for each window position,
/// using all available CPU cores.
///
/// # Arguments
///
/// * `actions` — Timestamps of player actions (monotonically increasing).
/// * `window_size` — Number of actions per window.
/// * `weights` — Component weights.
///
/// # Returns
///
/// Vector of Φ values, one per window position. Length = max(0, actions.len() - window_size + 1).
///
/// # Examples
///
/// ```
/// use harmony_core::phi::{compute_phi_windowed, PhiWeights};
///
/// let timestamps: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
/// let phis = compute_phi_windowed(&timestamps, 20, &PhiWeights::default());
/// assert_eq!(phis.len(), 81);
/// ```
pub fn compute_phi_windowed(actions: &[f64], window_size: usize, weights: &PhiWeights) -> Vec<f64> {
    let n = actions.len();
    if window_size < 2 || n < window_size {
        return Vec::new();
    }

    let num_windows = n - window_size + 1;

    // Precompute intervals for each window, then compute Φ in parallel.
    (0..num_windows)
        .into_par_iter()
        .map(|i| {
            let window = &actions[i..i + window_size];

            // Compute inter-action intervals for this window.
            let intervals: Vec<f64> = window.windows(2).map(|pair| pair[1] - pair[0]).collect();

            // Estimate idle ratio within the window.
            // Idle = intervals significantly longer than the median.
            let idle_ratio = if intervals.is_empty() {
                0.0
            } else {
                let mut sorted = intervals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = sorted[sorted.len() / 2];
                let threshold = median * 3.0; // 3× median = "idle"
                let idle_count = intervals.iter().filter(|&&v| v > threshold).count();
                idle_count as f64 / intervals.len() as f64
            };

            compute_phi(&intervals, idle_ratio, weights)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_intervals_max_friction() {
        let phi = compute_phi(&[], 0.0, &PhiWeights::default());
        assert_eq!(phi, 1.0);
    }

    #[test]
    fn test_regular_actions_low_phi() {
        // Perfectly regular, fast actions → low Φ.
        let regular: Vec<f64> = (0..200).map(|_| 0.5).collect();
        let phi = compute_phi(&regular, 0.0, &PhiWeights::default());
        assert!(phi < 0.3, "regular actions should have low Φ, got {phi}");
    }

    #[test]
    fn test_irregular_actions_high_phi() {
        let irregular: Vec<f64> = vec![
            0.1, 5.0, 0.2, 8.0, 0.3, 10.0, 0.1, 7.0, 0.2, 6.0, 0.1, 9.0, 0.3, 4.0, 0.1, 8.0,
        ];
        let phi = compute_phi(&irregular, 0.3, &PhiWeights::default());
        assert!(phi > 0.3, "irregular actions should have high Φ, got {phi}");
    }

    #[test]
    fn test_idle_increases_phi() {
        let intervals: Vec<f64> = (0..100).map(|_| 0.5).collect();
        let phi_active = compute_phi(&intervals, 0.0, &PhiWeights::default());
        let phi_idle = compute_phi(&intervals, 0.8, &PhiWeights::default());
        assert!(phi_idle > phi_active, "idle should increase Φ");
    }

    #[test]
    fn test_windowed_correct_length() {
        let timestamps: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
        let phis = compute_phi_windowed(&timestamps, 20, &PhiWeights::default());
        assert_eq!(phis.len(), 81);
    }

    #[test]
    fn test_windowed_all_in_range() {
        let timestamps: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();
        let phis = compute_phi_windowed(&timestamps, 20, &PhiWeights::default());
        for (i, &phi) in phis.iter().enumerate() {
            assert!(
                phi >= 0.0 && phi <= 1.0,
                "Φ at window {i} out of range: {phi}"
            );
        }
    }

    #[test]
    fn test_windowed_too_few_actions() {
        let timestamps = vec![1.0, 2.0, 3.0];
        let phis = compute_phi_windowed(&timestamps, 10, &PhiWeights::default());
        assert!(phis.is_empty());
    }

    #[test]
    fn test_default_weights_normalized() {
        let w = PhiWeights::default();
        assert!(w.is_normalized(), "default weights should sum to 1.0");
    }
}
