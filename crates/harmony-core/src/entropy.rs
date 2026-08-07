//! Action entropy computation.
//!
//! Shannon entropy measures the unpredictability of inter-action intervals.
//!
//! - **Low entropy** = regular cadence = the player has converged on a task.
//!   They are focused, not scattered.
//! - **High entropy** = irregular timing = the player is searching,
//!   context-switching, disrupted.
//!
//! Flow is characterized by low entropy — the player has found their rhythm.

use std::collections::HashMap;

/// Compute the Shannon entropy of inter-action intervals.
///
/// Uses histogram-based entropy: intervals are bucketed into bins, and the
/// Shannon entropy of the resulting probability distribution is computed.
///
/// Higher entropy means more irregular intervals. For a completely constant
/// sequence (all intervals identical), entropy is 0.
///
/// # Examples
///
/// ```
/// use harmony_core::entropy::action_entropy;
///
/// // Constant intervals → zero entropy.
/// let regular = vec![1.0, 1.0, 1.0, 1.0, 1.0];
/// assert_eq!(action_entropy(&regular), 0.0);
/// ```
pub fn action_entropy(intervals: &[f64]) -> f64 {
    let n = intervals.len();
    if n < 2 {
        return 0.0;
    }

    // Find range for bucketing.
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in intervals {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    // If all values are the same, entropy is zero.
    if (max_val - min_val).abs() < f64::EPSILON {
        return 0.0;
    }

    // Bucket into bins (sqrt-rule for bin count).
    let num_bins = (n as f64).sqrt().ceil() as usize;
    let num_bins = num_bins.max(2);
    let bin_width = (max_val - min_val) / num_bins as f64;

    // Build histogram.
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &v in intervals {
        let bin = (((v - min_val) / bin_width) as usize).min(num_bins - 1);
        *counts.entry(bin).or_insert(0) += 1;
    }

    // Shannon entropy: H = -Σ p(x) * log2(p(x)).
    let n_f = n as f64;
    let mut entropy = 0.0f64;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / n_f;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Normalize entropy to [0, 1] range.
///
/// Divides the raw entropy by log2(num_bins) to produce a normalized score
/// where 0 = completely regular and 1 = maximally irregular.
///
/// # Examples
///
/// ```
/// use harmony_core::entropy::normalized_entropy;
///
/// let regular = vec![1.0, 1.0, 1.0, 1.0, 1.0];
/// assert_eq!(normalized_entropy(&regular), 0.0);
/// ```
pub fn normalized_entropy(intervals: &[f64]) -> f64 {
    let n = intervals.len();
    if n < 2 {
        return 0.0;
    }

    let raw = action_entropy(intervals);

    // Max entropy for the bin count we used.
    let num_bins = ((n as f64).sqrt().ceil() as usize).max(2);
    let max_entropy = (num_bins as f64).log2();

    if max_entropy == 0.0 {
        return 0.0;
    }

    (raw / max_entropy).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_returns_zero() {
        assert_eq!(action_entropy(&[]), 0.0);
        assert_eq!(normalized_entropy(&[]), 0.0);
    }

    #[test]
    fn test_constant_intervals_zero_entropy() {
        let constant = vec![1.0; 50];
        assert_eq!(action_entropy(&constant), 0.0);
        assert_eq!(normalized_entropy(&constant), 0.0);
    }

    #[test]
    fn test_uniform_distribution_high_entropy() {
        // Evenly spaced distinct values → high (but not necessarily 1.0 due to binning).
        let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let ne = normalized_entropy(&values);
        assert!(
            ne > 0.8,
            "uniform distribution should have high entropy, got {ne}"
        );
    }

    #[test]
    fn test_normalized_in_range() {
        let values = vec![0.5, 1.0, 1.5, 2.0, 0.3, 0.7, 1.2, 3.0, 0.1, 2.5];
        let ne = normalized_entropy(&values);
        assert!(
            ne >= 0.0 && ne <= 1.0,
            "normalized entropy should be in [0,1], got {ne}"
        );
    }
}
