//! Cadence regularity and stability.
//!
//! Cadence measures the rhythmic consistency of player actions. A player in
//! flow has steady, metronomic timing — their hands have internalized the
//! beat. A player searching or disrupted has erratic timing.
//!
//! - **Low coefficient of variation (CV)** = regular cadence = flow.
//! - **High CV** = irregular timing = disrupted.
//!
//! We report regularity as `1.0 - CV` (clamped to [0, 1]), so high = regular.

/// Coefficient of variation of inter-action intervals.
///
/// Returns `1.0 - CV` (clamped to [0, 1]), where CV = σ / μ.
///
/// A return value of 1.0 means perfectly metronomic timing.
/// A return value of 0.0 means highly irregular timing.
///
/// For fewer than 2 intervals, returns 0.0 (no data).
///
/// # Examples
///
/// ```
/// use harmony_core::cadence::cadence_regularity;
///
/// // Metronomic timing → 1.0.
/// let regular = vec![1.0, 1.0, 1.0, 1.0, 1.0];
/// let r = cadence_regularity(&regular);
/// assert!((r - 1.0).abs() < 1e-10);
/// ```
pub fn cadence_regularity(intervals: &[f64]) -> f64 {
    let n = intervals.len();
    if n < 2 {
        return 0.0;
    }

    // Mean.
    let mean: f64 = intervals.iter().sum::<f64>() / n as f64;
    if mean.abs() < f64::EPSILON {
        return 0.0;
    }

    // Population standard deviation.
    let mut sq_sum = 0.0f64;
    for &v in intervals {
        let diff = v - mean;
        sq_sum += diff * diff;
    }
    let std = (sq_sum / n as f64).sqrt();

    let cv = std / mean;

    // Convert CV to regularity: CV=0 → 1.0, CV≥1 → 0.0.
    (1.0 - cv).clamp(0.0, 1.0)
}

/// Rolling cadence stability over a window.
///
/// Computes cadence regularity within a sliding window of `window` intervals,
/// producing a time series of stability values. This reveals how the player's
/// rhythm steadies or falters over time.
///
/// Returns an empty vector if there aren't enough intervals.
///
/// # Examples
///
/// ```
/// use harmony_core::cadence::cadence_stability;
///
/// let intervals = vec![1.0, 1.0, 1.1, 0.9, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0];
/// let stability = cadence_stability(&intervals, 4);
/// assert!(!stability.is_empty());
/// ```
pub fn cadence_stability(intervals: &[f64], window: usize) -> Vec<f64> {
    let n = intervals.len();
    if window < 2 || n < window {
        return Vec::new();
    }

    let num_windows = n - window + 1;
    let mut result = Vec::with_capacity(num_windows);

    for i in 0..num_windows {
        let w = &intervals[i..i + window];
        result.push(cadence_regularity(w));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_returns_zero() {
        assert_eq!(cadence_regularity(&[]), 0.0);
        assert_eq!(cadence_regularity(&[1.0]), 0.0);
    }

    #[test]
    fn test_metronomic_is_one() {
        let regular = vec![1.0; 50];
        let r = cadence_regularity(&regular);
        assert!((r - 1.0).abs() < 1e-10, "metronomic should be 1.0, got {r}");
    }

    #[test]
    fn test_irregular_is_low() {
        let irregular = vec![0.1, 5.0, 0.2, 8.0, 0.3, 10.0, 0.1, 7.0];
        let r = cadence_regularity(&irregular);
        assert!(r < 0.4, "irregular cadence should be low, got {r}");
    }

    #[test]
    fn test_stability_window_count() {
        let intervals = vec![1.0; 20];
        let stability = cadence_stability(&intervals, 5);
        assert_eq!(stability.len(), 16);
        for &s in &stability {
            assert!((s - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stability_too_few() {
        let intervals = vec![1.0, 2.0];
        assert!(cadence_stability(&intervals, 5).is_empty());
    }

    #[test]
    fn test_zero_mean_returns_zero() {
        let zeros = vec![0.0, 0.0, 0.0, 0.0];
        assert_eq!(cadence_regularity(&zeros), 0.0);
    }
}
