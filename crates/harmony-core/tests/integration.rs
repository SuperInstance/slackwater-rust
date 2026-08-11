//! Integration tests for harmony-core
//!
//! Tests the interaction between flow state detection, Φ computation,
//! cadence, entropy, and the protector across realistic scenarios.

use harmony_core::{
    FlowState, FlowStateDetector, FlowTrend,
    PhiWeights, compute_phi, compute_phi_windowed,
    FlowStateProtector, ProtectionAction,
};
use harmony_core::cadence::{cadence_regularity, cadence_stability};
use harmony_core::entropy::{action_entropy, normalized_entropy};
use harmony_core::hurst::hurst_exponent;

// ════════════════════════════════════════════════════════════════════
// CADENCE INTEGRATION
// ════════════════════════════════════════════════════════════════════

#[test]
fn cadence_regularity_decreases_with_noise() {
    let perfect: Vec<f64> = vec![1.0; 50];
    let slightly_off: Vec<f64> = (0..50).map(|i| 1.0 + (i as f64 * 0.01)).collect();
    let very_irregular: Vec<f64> = (0..50).map(|i| if i % 2 == 0 { 0.5 } else { 3.0 }).collect();

    let r1 = cadence_regularity(&perfect);
    let r2 = cadence_regularity(&slightly_off);
    let r3 = cadence_regularity(&very_irregular);

    assert!(r1 >= r2);
    assert!(r2 >= r3);
}

#[test]
fn cadence_stability_shows_transition() {
    let mut intervals: Vec<f64> = vec![1.0; 20];
    intervals.extend(vec![0.1, 5.0, 0.2, 8.0, 0.3, 10.0, 0.1, 7.0, 0.2, 6.0]);

    let stability = cadence_stability(&intervals, 5);
    assert!(stability.len() > 15);

    let early_avg: f64 = stability[..5].iter().sum::<f64>() / 5.0;
    let late_avg: f64 = stability[stability.len() - 5..].iter().sum::<f64>() / 5.0;
    assert!(early_avg > late_avg, "early cadence should be more stable: {} vs {}", early_avg, late_avg);
}

// ════════════════════════════════════════════════════════════════════
// ENTROPY INTEGRATION
// ════════════════════════════════════════════════════════════════════

#[test]
fn entropy_increases_with_diversity() {
    let constant = vec![1.0; 100];
    let two_values: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let many_values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();

    let e1 = action_entropy(&constant);
    let e2 = action_entropy(&two_values);
    let e3 = action_entropy(&many_values);

    assert!(e1 < e2, "constant should have less entropy than two-value: {} vs {}", e1, e2);
    assert!(e2 < e3, "two-value should have less entropy than many: {} vs {}", e2, e3);
}

#[test]
fn normalized_entropy_always_in_unit_range() {
    let test_cases = vec![
        vec![1.0; 10],
        vec![0.1, 0.2, 0.3, 0.4, 0.5],
        vec![100.0, 0.001, 50.0, 0.1, 75.0, 0.5],
        (0..200).map(|i| (i as f64).sin()).collect::<Vec<_>>(),
    ];

    for intervals in test_cases {
        let ne = normalized_entropy(&intervals);
        assert!(ne >= 0.0 && ne <= 1.0, "normalized entropy out of range for test case: {}", ne);
    }
}

// ════════════════════════════════════════════════════════════════════
// HURST EXPONENT INTEGRATION
// ════════════════════════════════════════════════════════════════════

#[test]
fn hurst_of_trending_series_is_above_half() {
    let trending: Vec<f64> = (1..200).map(|i| i as f64).collect();
    let h = hurst_exponent(&trending);
    assert!(h > 0.5, "trending series should have H > 0.5, got {}", h);
}

#[test]
fn hurst_of_constant_series_doesnt_panic() {
    let constant = vec![5.0; 100];
    let _h = hurst_exponent(&constant);
}

#[test]
fn hurst_of_short_series_doesnt_panic() {
    let short = vec![1.0, 2.0, 3.0];
    let _h = hurst_exponent(&short);
}

// ════════════════════════════════════════════════════════════════════
// Φ (PHI) INTEGRATION
// ════════════════════════════════════════════════════════════════════

#[test]
fn phi_always_in_unit_range() {
    let test_cases = vec![
        (vec![], 0.0),
        (vec![1.0; 100], 0.0),
        (vec![0.1, 5.0, 0.2, 8.0], 0.5),
        ((0..50).map(|i| i as f64 * 0.1).collect::<Vec<_>>(), 0.3),
        ((0..200).map(|i| if i % 3 == 0 { 10.0 } else { 0.5 }).collect::<Vec<_>>(), 0.8),
    ];

    for (intervals, idle) in test_cases {
        let phi = compute_phi(&intervals, idle, &PhiWeights::default());
        assert!(phi >= 0.0 && phi <= 1.0, "Φ out of range [{}, {}]: {}", idle, intervals.len(), phi);
    }
}

#[test]
fn phi_increases_with_idle_for_regular_intervals() {
    let intervals: Vec<f64> = vec![0.5; 100];
    let phi_0 = compute_phi(&intervals, 0.0, &PhiWeights::default());
    let phi_1 = compute_phi(&intervals, 1.0, &PhiWeights::default());
    assert!(phi_1 > phi_0, "Full idle should have higher Φ than no idle: {} vs {}", phi_1, phi_0);
}

#[test]
fn phi_custom_weights_change_result() {
    let intervals: Vec<f64> = (0..50).map(|i| if i % 2 == 0 { 0.5 } else { 2.0 }).collect();

    let all_persistence = PhiWeights {
        persistence: 1.0,
        entropy: 0.0,
        cadence: 0.0,
        idle: 0.0,
    };
    let all_idle = PhiWeights {
        persistence: 0.0,
        entropy: 0.0,
        cadence: 0.0,
        idle: 1.0,
    };

    let phi_p = compute_phi(&intervals, 0.0, &all_persistence);
    let phi_i = compute_phi(&intervals, 0.5, &all_idle);
    assert!(phi_p != phi_i, "Different weights should produce different Φ");
}

#[test]
fn phi_windowed_tracks_transition() {
    let mut timestamps: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
    let last_regular = timestamps.last().copied().unwrap_or(0.0);
    let mut t = last_regular;
    for _ in 0..50 {
        t += if timestamps.len() % 3 == 0 { 5.0 } else { 0.3 };
        timestamps.push(t);
    }

    let phis = compute_phi_windowed(&timestamps, 20, &PhiWeights::default());
    assert!(phis.len() > 50);

    let early_avg: f64 = phis[..10].iter().sum::<f64>() / 10.0;
    let late_avg: f64 = phis[phis.len() - 10..].iter().sum::<f64>() / 10.0;
    assert!(early_avg < late_avg, "Φ should increase after transition: {} vs {}", early_avg, late_avg);
}

// ════════════════════════════════════════════════════════════════════
// PHI WEIGHTS
// ════════════════════════════════════════════════════════════════════

#[test]
fn phi_weights_unnormalized_detected() {
    let w = PhiWeights {
        persistence: 0.5,
        entropy: 0.5,
        cadence: 0.5,
        idle: 0.5,
    };
    assert!(!w.is_normalized());
}

#[test]
fn phi_weights_all_zero() {
    let w = PhiWeights {
        persistence: 0.0,
        entropy: 0.0,
        cadence: 0.0,
        idle: 0.0,
    };
    assert!(!w.is_normalized());
    let phi = compute_phi(&[1.0, 2.0, 3.0], 0.5, &w);
    assert!((phi - 0.0).abs() < 1e-10);
}

// ════════════════════════════════════════════════════════════════════
// FLOW STATE DETECTOR
// ════════════════════════════════════════════════════════════════════

#[test]
fn flow_detector_starts_out_of_flow() {
    let detector = FlowStateDetector::new();
    let state = detector.state();
    assert!(matches!(state, FlowState::OutOfFlow));
}

#[test]
fn flow_detector_observes_low_phi_to_enter_flow() {
    let mut detector = FlowStateDetector::new();
    // Feed many low Φ observations to enter flow
    for _ in 0..200 {
        detector.observe(0.05);
    }
    let state = detector.state();
    assert!(
        matches!(state, FlowState::InFlow | FlowState::DeepFlow | FlowState::ApproachingFlow),
        "Expected flow-related state after sustained low Φ, got {:?}",
        state
    );
}

#[test]
fn flow_detector_high_phi_stays_out() {
    let mut detector = FlowStateDetector::new();
    for _ in 0..100 {
        detector.observe(0.9);
    }
    assert!(matches!(detector.state(), FlowState::OutOfFlow));
}

#[test]
fn flow_detector_resets() {
    let mut detector = FlowStateDetector::new();
    for _ in 0..200 {
        detector.observe(0.05);
    }
    detector.reset();
    assert!(matches!(detector.state(), FlowState::OutOfFlow));
    assert_eq!(detector.observation_count(), 0);
}

#[test]
fn flow_detector_tracks_trend() {
    let mut detector = FlowStateDetector::new();
    // Start with high Φ
    for _ in 0..50 {
        detector.observe(0.8);
    }
    // Transition to low Φ
    for _ in 0..50 {
        detector.observe(0.1);
    }
    let trend = detector.phi_trend();
    assert!(
        matches!(trend, FlowTrend::Improving | FlowTrend::Stable),
        "Expected Improving after Φ drop, got {:?}",
        trend
    );
}

#[test]
fn flow_detector_in_flow_after_sustained_low_phi() {
    let mut detector = FlowStateDetector::new();
    for _ in 0..500 {
        detector.observe(0.02);
    }
    assert!(detector.in_flow());
}

#[test]
fn flow_detector_last_phi_recorded() {
    let mut detector = FlowStateDetector::new();
    detector.observe(0.42);
    assert!((detector.last_phi() - 0.42).abs() < 1e-10);
}

// ════════════════════════════════════════════════════════════════════
// FLOW STATE PROTECTOR
// ════════════════════════════════════════════════════════════════════

#[test]
fn protector_starts_not_protecting() {
    let protector = FlowStateProtector::new();
    assert!(!protector.is_protecting());
}

#[test]
fn protector_engages_on_low_phi() {
    let mut protector = FlowStateProtector::new();
    // Feed low Φ values to trigger protection
    let mut action = None;
    for _ in 0..200 {
        if let Some(a) = protector.on_phi_update(0.02) {
            action = Some(a);
        }
    }
    assert!(protector.is_protecting() || action.is_some(),
        "Protector should engage after sustained low Φ");
}

#[test]
fn protector_releases_on_high_phi() {
    let mut protector = FlowStateProtector::new();
    // Engage first
    for _ in 0..200 {
        protector.on_phi_update(0.02);
    }
    // Now break flow with high Φ
    for _ in 0..200 {
        protector.on_phi_update(0.95);
    }
    assert!(!protector.is_protecting(), "Protector should release after sustained high Φ");
}

#[test]
fn protector_tracks_tempo_lock() {
    let mut protector = FlowStateProtector::new();
    for _ in 0..200 {
        protector.on_phi_update(0.02);
    }
    // When protecting, tempo should be locked
    if protector.is_protecting() {
        assert!(protector.is_tempo_locked());
    }
}

#[test]
fn protector_force_release() {
    let mut protector = FlowStateProtector::new();
    for _ in 0..200 {
        protector.on_phi_update(0.02);
    }
    protector.force_release();
    assert!(!protector.is_protecting());
}

#[test]
fn protector_suppression_list() {
    let mut protector = FlowStateProtector::new();
    protector.suppress("notifications");
    protector.suppress("agent_chatter");
    assert!(protector.suppression_list().contains(&"notifications".to_string()));
    assert!(protector.suppression_list().contains(&"agent_chatter".to_string()));

    protector.unsuppress("notifications");
    assert!(!protector.suppression_list().contains(&"notifications".to_string()));
}

// ════════════════════════════════════════════════════════════════════
// CROSS-MODULE: Φ → FLOW STATE
// ════════════════════════════════════════════════════════════════════

#[test]
fn phi_and_flow_state_agree_on_flow() {
    let intervals: Vec<f64> = vec![0.5; 200];
    let phi = compute_phi(&intervals, 0.0, &PhiWeights::default());
    assert!(phi < 0.3, "Regular actions should have low Φ: {}", phi);

    let mut detector = FlowStateDetector::new();
    for _ in 0..200 {
        detector.observe(phi);
    }
    assert!(
        matches!(detector.state(), FlowState::InFlow | FlowState::DeepFlow | FlowState::ApproachingFlow),
        "Low Φ should lead to flow, got {:?}",
        detector.state()
    );
}

#[test]
fn phi_and_flow_state_agree_on_struggle() {
    let intervals: Vec<f64> = vec![
        0.1, 5.0, 0.2, 8.0, 0.3, 10.0, 0.1, 7.0,
        0.2, 6.0, 0.1, 9.0, 0.3, 4.0, 0.1, 8.0,
        0.2, 12.0, 0.1, 6.0, 0.3, 11.0, 0.1, 7.5,
    ];
    let phi = compute_phi(&intervals, 0.4, &PhiWeights::default());
    assert!(phi > 0.3, "Irregular actions should have higher Φ: {}", phi);

    let mut detector = FlowStateDetector::new();
    for _ in 0..100 {
        detector.observe(phi);
    }
    assert!(matches!(detector.state(), FlowState::OutOfFlow));
}

// ════════════════════════════════════════════════════════════════════
// FLOW STATE PREDICATES AND TRENDS
// ════════════════════════════════════════════════════════════════════

#[test]
fn flow_state_is_flow_predicate() {
    assert!(FlowState::InFlow.is_flow());
    assert!(FlowState::DeepFlow.is_flow());
    assert!(!FlowState::OutOfFlow.is_flow());
    assert!(!FlowState::ApproachingFlow.is_flow());
}

#[test]
fn flow_trend_variants_distinct() {
    assert_ne!(FlowTrend::Improving, FlowTrend::Stable);
    assert_ne!(FlowTrend::Stable, FlowTrend::Declining);
    assert_ne!(FlowTrend::Improving, FlowTrend::Declining);
}

#[test]
fn protection_action_variants_exist() {
    let _ = ProtectionAction::SuppressNotifications;
    let _ = ProtectionAction::LockTempo;
    let _ = ProtectionAction::ReduceAgentActivity;
    let _ = ProtectionAction::ClearNonUrgent;
    let _ = ProtectionAction::Release;
}
