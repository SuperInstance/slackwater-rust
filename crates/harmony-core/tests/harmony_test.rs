//! Integration tests for harmony-core.
//!
//! These tests verify the full pipeline: Hurst exponent → entropy → cadence
//! → Φ computation → flow state detection → protection actions.

#![warn(clippy::all)]

use approx::assert_relative_eq;
use harmony_core::{
    cadence::{cadence_regularity, cadence_stability},
    entropy::{action_entropy, normalized_entropy},
    flow_state::{FlowState, FlowStateDetector, FlowTrend},
    hurst::hurst_exponent,
    phi::{compute_phi, compute_phi_windowed, PhiWeights},
    protector::{FlowStateProtector, ProtectionAction},
};

// ── Hurst Exponent ────────────────────────────────────────

#[test]
fn hurst_trending_data_is_persistent() {
    // Monotonically increasing data should have H > 0.5.
    let trending: Vec<f64> = (0..2000).map(|i| i as f64 * 0.1).collect();
    let h = hurst_exponent(&trending);
    assert!(
        h > 0.5,
        "trending data should have H > 0.5 (persistent), got {h}"
    );
}

#[test]
fn hurst_short_data_returns_neutral() {
    assert_eq!(hurst_exponent(&[]), 0.5);
    assert_eq!(hurst_exponent(&[1.0]), 0.5);
    assert_eq!(hurst_exponent(&[1.0, 2.0, 3.0]), 0.5);
}

#[test]
fn hurst_constant_returns_neutral() {
    let constant = vec![5.0; 100];
    assert_eq!(hurst_exponent(&constant), 0.5);
}

#[test]
fn hurst_random_walk_near_half() {
    // Independent random data should have H near 0.5.
    // We use a deterministic pseudo-random sequence for reproducibility.
    let mut data = Vec::with_capacity(5000);
    let mut state: u64 = 42;
    for _ in 0..5000 {
        // xorshift for deterministic pseudo-randomness.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = ((state as f64) / (u64::MAX as f64) - 0.5) * 2.0;
        data.push(noise);
    }
    let h = hurst_exponent(&data);
    // Independent data should be around 0.5. Allow a generous band.
    assert!(
        h > 0.2 && h < 0.9,
        "random data Hurst should be near 0.5, got {h}"
    );
}

// ── Entropy ───────────────────────────────────────────────

#[test]
fn entropy_uniform_distribution_is_high() {
    // Evenly spaced distinct values.
    let values: Vec<f64> = (0..200).map(|i| i as f64 * 0.1).collect();
    let ne = normalized_entropy(&values);
    assert!(
        ne > 0.8,
        "uniform distribution should have high normalized entropy, got {ne}"
    );
}

#[test]
fn entropy_constant_intervals_is_zero() {
    let constant = vec![1.0; 100];
    assert_eq!(action_entropy(&constant), 0.0);
    assert_eq!(normalized_entropy(&constant), 0.0);
}

#[test]
fn entropy_empty_returns_zero() {
    assert_eq!(action_entropy(&[]), 0.0);
    assert_eq!(normalized_entropy(&[]), 0.0);
}

#[test]
fn entropy_normalized_in_range() {
    let test_cases: &[&[f64]] = &[
        &[0.5, 1.0, 1.5, 2.0, 0.3, 0.7],
        &[1.0; 50],
        &(0..100).map(|i| i as f64 * 0.1).collect::<Vec<_>>(),
        &[0.1, 5.0, 0.2, 8.0, 0.3, 10.0],
    ];
    for intervals in test_cases {
        let ne = normalized_entropy(intervals);
        assert!(
            (0.0..=1.0).contains(&ne),
            "normalized entropy out of [0,1]: {ne} for {intervals:?}"
        );
    }
}

// ── Cadence ───────────────────────────────────────────────

#[test]
fn cadence_metronomic_is_one() {
    let regular = vec![1.0; 100];
    let r = cadence_regularity(&regular);
    assert_relative_eq!(r, 1.0, epsilon = 1e-10);
}

#[test]
fn cadence_random_is_low() {
    let irregular = vec![0.1, 5.0, 0.2, 8.0, 0.3, 10.0, 0.1, 7.0];
    let r = cadence_regularity(&irregular);
    assert!(r < 0.4, "irregular cadence should be low, got {r}");
}

#[test]
fn cadence_stability_window_count() {
    let intervals: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let stability = cadence_stability(&intervals, 10);
    assert_eq!(stability.len(), 41);
    for &s in &stability {
        assert!((0.0..=1.0).contains(&s));
    }
}

#[test]
fn cadence_empty_returns_zero() {
    assert_eq!(cadence_regularity(&[]), 0.0);
    assert_eq!(cadence_regularity(&[1.0]), 0.0);
}

// ── Φ (Phi) Computation ───────────────────────────────────

#[test]
fn phi_regular_fast_actions_produce_low_phi() {
    // Regular, fast intervals = flow.
    let regular: Vec<f64> = (0..200).map(|_| 0.5).collect();
    let phi = compute_phi(&regular, 0.0, &PhiWeights::default());
    assert!(
        phi < 0.3,
        "regular fast actions should have low Φ (flow), got {phi}"
    );
}

#[test]
fn phi_irregular_actions_produce_high_phi() {
    let irregular: Vec<f64> = vec![
        0.1, 5.0, 0.2, 8.0, 0.3, 10.0, 0.1, 7.0,
        0.2, 6.0, 0.1, 9.0, 0.3, 4.0, 0.1, 8.0,
    ];
    let phi = compute_phi(&irregular, 0.3, &PhiWeights::default());
    assert!(
        phi > 0.3,
        "irregular actions should have high Φ, got {phi}"
    );
}

#[test]
fn phi_empty_returns_max() {
    let phi = compute_phi(&[], 0.0, &PhiWeights::default());
    assert_eq!(phi, 1.0);
}

#[test]
fn phi_in_range() {
    let test_intervals: &[&[f64]] = &[
        &[0.5; 100],
        &[0.1, 5.0, 0.2, 8.0],
        &(0..50).map(|i| (i as f64).sin().abs() + 0.5).collect::<Vec<_>>(),
    ];
    for intervals in test_intervals {
        let phi = compute_phi(intervals, 0.2, &PhiWeights::default());
        assert!(
            (0.0..=1.0).contains(&phi),
            "Φ out of [0,1]: {phi}"
        );
    }
}

#[test]
fn phi_idle_ratio_increases_phi() {
    let intervals: Vec<f64> = (0..100).map(|_| 0.5).collect();
    let phi_active = compute_phi(&intervals, 0.0, &PhiWeights::default());
    let phi_idle = compute_phi(&intervals, 0.9, &PhiWeights::default());
    assert!(phi_idle > phi_active, "idle should increase Φ");
}

#[test]
fn phi_default_weights_sum_to_one() {
    let w = PhiWeights::default();
    let sum = w.persistence + w.entropy + w.cadence + w.idle;
    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
}

// ── Windowed Φ ────────────────────────────────────────────

#[test]
fn phi_windowed_correct_length() {
    let timestamps: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();
    let phis = compute_phi_windowed(&timestamps, 20, &PhiWeights::default());
    assert_eq!(phis.len(), 181);
}

#[test]
fn phi_windowed_all_in_range() {
    let timestamps: Vec<f64> = (0..500).map(|i| i as f64 * 0.3).collect();
    let phis = compute_phi_windowed(&timestamps, 30, &PhiWeights::default());
    for (i, &phi) in phis.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&phi),
            "windowed Φ at position {i} out of range: {phi}"
        );
    }
}

#[test]
fn phi_windowed_too_few_returns_empty() {
    let timestamps = vec![1.0, 2.0, 3.0];
    assert!(compute_phi_windowed(&timestamps, 10, &PhiWeights::default()).is_empty());
}

#[test]
fn phi_windowed_parallel_consistent() {
    // The parallelized computation should produce the same result regardless
    // of thread count. Run twice and compare.
    let timestamps: Vec<f64> = (0..300).map(|i| i as f64 * 0.4).collect();
    let phis1 = compute_phi_windowed(&timestamps, 25, &PhiWeights::default());
    let phis2 = compute_phi_windowed(&timestamps, 25, &PhiWeights::default());
    assert_eq!(phis1.len(), phis2.len());
    for (a, b) in phis1.iter().zip(phis2.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-10);
    }
}

// ── FlowStateDetector ─────────────────────────────────────

#[test]
fn flow_state_starts_out_of_flow() {
    let d = FlowStateDetector::new();
    assert_eq!(d.state(), FlowState::OutOfFlow);
    assert!(!d.in_flow());
}

#[test]
fn flow_state_high_phi_stays_out() {
    let mut d = FlowStateDetector::new();
    for _ in 0..20 {
        d.observe(0.8);
    }
    assert_eq!(d.state(), FlowState::OutOfFlow);
}

#[test]
fn flow_state_transitions_out_to_approaching_to_flow() {
    let mut d = FlowStateDetector::with_params(0.05, 5);

    // Start out of flow.
    assert_eq!(d.state(), FlowState::OutOfFlow);

    // Feed moderate Φ to approach flow.
    for _ in 0..4 {
        d.observe(0.08);
    }
    // Should be approaching (0.08 < 0.05 * 2.0 = 0.10).
    assert_eq!(d.state(), FlowState::ApproachingFlow);

    // Feed low Φ to enter flow.
    for _ in 0..6 {
        d.observe(0.02);
    }
    assert_eq!(d.state(), FlowState::InFlow);
    assert!(d.in_flow());
}

#[test]
fn flow_state_enters_deep_flow() {
    let mut d = FlowStateDetector::with_params(0.05, 5);

    // Get into flow first.
    for _ in 0..8 {
        d.observe(0.03);
    }
    assert_eq!(d.state(), FlowState::InFlow);

    // Very low Φ for deep flow.
    for _ in 0..8 {
        d.observe(0.005);
    }
    assert_eq!(d.state(), FlowState::DeepFlow);
}

#[test]
fn flow_state_breaks_on_rising_phi() {
    let mut d = FlowStateDetector::with_params(0.05, 3);

    // Enter flow.
    for _ in 0..5 {
        d.observe(0.02);
    }
    assert!(d.in_flow());

    // Φ spikes up.
    for _ in 0..3 {
        d.observe(0.6);
    }
    assert_eq!(d.state(), FlowState::OutOfFlow);
}

#[test]
fn flow_state_trend_improving() {
    let mut d = FlowStateDetector::new();
    for v in [0.9, 0.8, 0.7, 0.4, 0.3, 0.2] {
        d.observe(v);
    }
    assert_eq!(d.phi_trend(), FlowTrend::Improving);
}

#[test]
fn flow_state_trend_declining() {
    let mut d = FlowStateDetector::new();
    for v in [0.1, 0.1, 0.2, 0.4, 0.6, 0.8] {
        d.observe(v);
    }
    assert_eq!(d.phi_trend(), FlowTrend::Declining);
}

#[test]
fn flow_state_reset() {
    let mut d = FlowStateDetector::with_params(0.05, 3);
    for _ in 0..10 {
        d.observe(0.01);
    }
    assert!(d.in_flow());

    d.reset();
    assert_eq!(d.state(), FlowState::OutOfFlow);
    assert_eq!(d.observation_count(), 0);
}

// ── FlowStateProtector ────────────────────────────────────

#[test]
fn protector_starts_inactive() {
    let p = FlowStateProtector::new();
    assert!(!p.is_protecting());
}

#[test]
fn protector_engages_on_low_phi() {
    let mut p = FlowStateProtector::new();
    let action = p.on_phi_update(0.02);
    assert_eq!(action, Some(ProtectionAction::LockTempo));
    assert!(p.is_protecting());
}

#[test]
fn protector_returns_none_when_uncertain() {
    let mut p = FlowStateProtector::new();
    // Φ in the uncertain zone — not low enough to engage.
    assert_eq!(p.on_phi_update(0.10), None);
    assert!(!p.is_protecting());
}

#[test]
fn protector_releases_on_high_phi() {
    let mut p = FlowStateProtector::new();
    p.on_phi_update(0.02);
    assert!(p.is_protecting());

    let action = p.on_phi_update(0.20);
    assert_eq!(action, Some(ProtectionAction::Release));
    assert!(!p.is_protecting());
}

#[test]
fn protector_holds_during_hysteresis() {
    let mut p = FlowStateProtector::new();

    // Engage at low Φ.
    p.on_phi_update(0.03);
    assert!(p.is_protecting());

    // Φ rises but below ceiling — should hold (None = doing nothing well).
    let action = p.on_phi_update(0.10);
    assert_eq!(action, None);
    assert!(p.is_protecting());
}

#[test]
fn protector_escalation_on_deep_flow() {
    let mut p = FlowStateProtector::new();

    // Engage. phi=0.045 is between 0.8*floor and floor, so level 2.
    p.on_phi_update(0.045);
    assert_eq!(p.escalation_level(), 2);

    // Very low Φ — escalate.
    let action = p.on_phi_update(0.01);
    assert!(action.is_some());
    assert!(p.escalation_level() > 1);
}

#[test]
fn protector_engage_release_cycle() {
    let mut p = FlowStateProtector::with_thresholds(0.05, 0.15);

    // Engage.
    assert_eq!(p.on_phi_update(0.02), Some(ProtectionAction::LockTempo));
    assert!(p.is_protecting());

    // Release.
    assert_eq!(p.on_phi_update(0.20), Some(ProtectionAction::Release));
    assert!(!p.is_protecting());

    // Re-engage.
    assert_eq!(p.on_phi_update(0.01), Some(ProtectionAction::LockTempo));
    assert!(p.is_protecting());
}

#[test]
fn protector_force_release() {
    let mut p = FlowStateProtector::new();
    p.on_phi_update(0.01);
    assert!(p.is_protecting());

    p.force_release();
    assert!(!p.is_protecting());
    assert!(!p.is_tempo_locked());
}
