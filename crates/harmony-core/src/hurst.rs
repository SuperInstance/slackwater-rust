//! Hurst exponent via rescaled range (R/S) analysis.
//!
//! The Hurst exponent characterizes long-term memory in a time series:
//!
//! - **H < 0.5** — mean-reverting (anti-persistent): the action stream is
//!   irregular, oscillating without sustaining direction.
//! - **H ≈ 0.5** — random walk: no clear trend.
//! - **H > 0.5** — trending (persistent): the action stream has momentum,
//!   each action building on the last. **This is the signature of flow.**
//!
//! The R/S method divides the series into windows of increasing size,
//! computes the rescaled range (R/S) for each, and fits a line in log-log
//! space. The slope of that line is the Hurst exponent.
//!
//! This is the tight inner loop: O(n log n) with sub-loops that benefit
//! enormously from Rust's zero-cost abstractions and SIMD auto-vectorization.

/// Compute the Hurst exponent using the rescaled range (R/S) method.
///
/// Divides `data` into windows at multiple scales, computes R/S for each,
/// and returns the slope of log(R/S) vs log(window_size).
///
/// Returns 0.5 (neutral) for insufficient data (< 8 points).
///
/// # Examples
///
/// ```
/// use harmony_core::hurst::hurst_exponent;
///
/// // Trending data should have H > 0.5
/// let trending: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
/// let h = hurst_exponent(&trending);
/// assert!(h > 0.5, "trending data should be persistent");
///
/// // Insufficient data returns neutral
/// assert_eq!(hurst_exponent(&[1.0, 2.0, 3.0]), 0.5);
/// ```
pub fn hurst_exponent(data: &[f64]) -> f64 {
    let n = data.len();

    // Need enough data for meaningful R/S analysis at multiple scales.
    if n < 8 {
        return 0.5;
    }

    // Collect (log(n), log(R/S)) pairs at multiple window sizes.
    // We use window sizes that are powers of 2 for clean partitioning,
    // falling back to divisor-based windows for non-power-of-2 lengths.
    let mut log_sizes: Vec<f64> = Vec::new();
    let mut log_rs: Vec<f64> = Vec::new();

    // Start from small windows (min 4) up to n/2.
    let mut window = 4usize;
    while window <= n / 2 {
        let rs = rescaled_range(data, window);
        if rs.is_finite() && rs > 0.0 {
            log_sizes.push((window as f64).ln());
            log_rs.push(rs.ln());
        }
        window *= 2;
    }

    // If we didn't get enough points for a regression, use the single-window
    // estimate as a fallback.
    if log_sizes.len() < 2 {
        // Fallback: single-window R/S over the full series.
        return single_window_hurst(data);
    }

    let slope = regression_slope(&log_sizes, &log_rs);
    // Clamp to plausible range. Theoretical range is [0, 1].
    slope.clamp(0.0, 1.0)
}

/// Rescaled range for a given window size.
///
/// Partitions `data` into non-overlapping windows of `window_size`, computes
/// the mean rescaled range (R/S) across all windows, and returns it.
///
/// R/S for a single window:
/// 1. Compute the mean of the window.
/// 2. Compute the cumulative deviation from the mean (random walk with drift removed).
/// 3. R = max(cumdev) - min(cumdev) (the range).
/// 4. S = standard deviation of the window.
/// 5. R/S = R / S.
fn rescaled_range(data: &[f64], window_size: usize) -> f64 {
    let n = data.len();
    if window_size < 2 || n < window_size {
        return 0.0;
    }

    let num_windows = n / window_size;
    if num_windows == 0 {
        return 0.0;
    }

    let mut rs_sum = 0.0f64;
    let mut valid_windows = 0usize;

    for w in 0..num_windows {
        let start = w * window_size;
        let end = start + window_size;
        let window = &data[start..end];

        // Mean of the window.
        let mean: f64 = window.iter().sum::<f64>() / window_size as f64;

        // Cumulative deviation from the mean.
        let mut cumdev = Vec::with_capacity(window_size);
        let mut running = 0.0f64;
        for &val in window {
            running += val - mean;
            cumdev.push(running);
        }

        // Range R = max(cumdev) - min(cumdev).
        let mut min_cd = f64::INFINITY;
        let mut max_cd = f64::NEG_INFINITY;
        for &cd in &cumdev {
            if cd < min_cd {
                min_cd = cd;
            }
            if cd > max_cd {
                max_cd = cd;
            }
        }
        let r = max_cd - min_cd;

        // Standard deviation S (population).
        let mut sq_sum = 0.0f64;
        for &val in window {
            let diff = val - mean;
            sq_sum += diff * diff;
        }
        let s = (sq_sum / window_size as f64).sqrt();

        if s > 0.0 && r.is_finite() {
            rs_sum += r / s;
            valid_windows += 1;
        }
    }

    if valid_windows == 0 {
        return 0.0;
    }

    rs_sum / valid_windows as f64
}

/// Single-window Hurst estimate (fallback for short series).
///
/// Uses the full series as one window: H ≈ log(R/S) / log(n).
fn single_window_hurst(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 4 {
        return 0.5;
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;

    // Cumulative deviation.
    let mut running = 0.0f64;
    let mut min_cd = f64::INFINITY;
    let mut max_cd = f64::NEG_INFINITY;
    for &val in data {
        running += val - mean;
        if running < min_cd {
            min_cd = running;
        }
        if running > max_cd {
            max_cd = running;
        }
    }

    let r = max_cd - min_cd;
    if r <= 0.0 {
        return 0.5; // No variation — neutral.
    }

    let mut sq_sum = 0.0f64;
    for &val in data {
        let diff = val - mean;
        sq_sum += diff * diff;
    }
    let s = (sq_sum / n as f64).sqrt();
    if s <= 0.0 {
        return 0.5;
    }

    let rs = r / s;
    if rs <= 0.0 || n <= 1 {
        return 0.5;
    }

    let h = rs.ln() / (n as f64).ln();
    h.clamp(0.0, 1.0)
}

/// Linear regression slope (for log-log fit of R/S vs window size).
///
/// Computes the least-squares slope: β = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²).
///
/// Returns 0.5 (neutral) if the denominator is zero or arrays are empty.
pub fn regression_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n != ys.len() || n < 2 {
        return 0.5;
    }

    let n_f = n as f64;
    let x_mean: f64 = xs.iter().sum::<f64>() / n_f;
    let y_mean: f64 = ys.iter().sum::<f64>() / n_f;

    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;

    for i in 0..n {
        let dx = xs[i] - x_mean;
        numerator += dx * (ys[i] - y_mean);
        denominator += dx * dx;
    }

    if denominator == 0.0 {
        return 0.5;
    }

    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_data_returns_neutral() {
        assert_eq!(hurst_exponent(&[]), 0.5);
        assert_eq!(hurst_exponent(&[1.0]), 0.5);
        assert_eq!(hurst_exponent(&[1.0, 2.0, 3.0]), 0.5);
    }

    #[test]
    fn test_constant_series_returns_neutral() {
        // No variation → neutral.
        let constant = vec![5.0; 100];
        assert_eq!(hurst_exponent(&constant), 0.5);
    }

    #[test]
    fn test_trending_data_is_persistent() {
        // Monotonically increasing data should have H > 0.5.
        let trending: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let h = hurst_exponent(&trending);
        assert!(h > 0.5, "trending data should have H > 0.5, got {h}");
    }

    #[test]
    fn test_regression_slope_basic() {
        // Perfect line: y = 2x.
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![2.0, 4.0, 6.0, 8.0];
        let slope = regression_slope(&xs, &ys);
        assert!((slope - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_regression_slope_empty() {
        assert_eq!(regression_slope(&[], &[]), 0.5);
        assert_eq!(regression_slope(&[1.0], &[1.0]), 0.5);
    }
}
