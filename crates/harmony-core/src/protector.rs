//! Flow State Protector — imperceptible adjustments when flow is detected.
//!
//! The protector makes tiny, invisible adjustments to protect the player's
//! flow state. When uncertain, it returns `None` (does nothing).
//!
//! > "Doing nothing well is safer than doing something clever."
//!
//! When flow is detected (Φ < `phi_floor`):
//! - Notifications are suppressed.
//! - Tempo is locked (no adjustments).
//! - Agent activity is reduced (less chatter).
//! - Non-urgent tasks are deferred.
//!
//! When flow ends (Φ > `phi_ceiling`), protections are released and normal
//! operations resume.

use serde::{Deserialize, Serialize};

/// Actions the protector can take to shield the player's flow.
///
/// These are suppressions, not augmentations. The protector makes the
/// environment quieter — it never adds stimulation during flow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtectionAction {
    /// Suppress incoming notifications during flow.
    SuppressNotifications,
    /// Lock tempo — no BPM adjustments (the groove is the groove).
    LockTempo,
    /// Reduce agent dialogue and activity level.
    ReduceAgentActivity,
    /// Clear non-urgent queued items (deferrable tasks).
    ClearNonUrgent,
    /// Flow has ended — release all protections and resume normal operations.
    Release,
}

/// Flow State Protector — makes imperceptible adjustments when flow is detected.
///
/// The protector has hysteresis: once engaged (Φ < `phi_floor`), it stays
/// engaged until Φ rises above `phi_ceiling`. This prevents rapid
/// engage/disengage cycling near the threshold.
///
/// # Examples
///
/// ```
/// use harmony_core::protector::{FlowStateProtector, ProtectionAction};
///
/// let mut p = FlowStateProtector::new();
///
/// // Φ drops into flow territory → protector engages.
/// let action = p.on_phi_update(0.02);
/// assert_eq!(action, Some(ProtectionAction::LockTempo));
/// assert!(p.is_protecting());
///
/// // Φ stays low → deep flow, suppress everything.
/// let action = p.on_phi_update(0.01);
/// assert_eq!(action, Some(ProtectionAction::SuppressNotifications));
///
/// // Φ rises above ceiling → release.
/// let action = p.on_phi_update(0.20);
/// assert_eq!(action, Some(ProtectionAction::Release));
/// assert!(!p.is_protecting());
/// ```
pub struct FlowStateProtector {
    /// Φ below this = deep flow, lock everything.
    phi_floor: f64,
    /// Φ above this = flow broken, resume normal operations.
    phi_ceiling: f64,
    /// Items to suppress during flow.
    suppression_list: Vec<String>,
    /// Whether tempo is currently locked.
    tempo_locked: bool,
    /// Whether protection is currently active.
    protecting: bool,
    /// Escalation level: how many protections are active.
    escalation: u8,
}

impl FlowStateProtector {
    /// Create a new protector with default settings.
    ///
    /// Defaults:
    /// - `phi_floor`: 0.05
    /// - `phi_ceiling`: 0.15
    /// - Suppressions: notifications, agent chatter, non-urgent tasks
    pub fn new() -> Self {
        Self {
            phi_floor: 0.05,
            phi_ceiling: 0.15,
            suppression_list: vec![
                "notifications".to_string(),
                "agent_chatter".to_string(),
                "non_urgent_tasks".to_string(),
            ],
            tempo_locked: false,
            protecting: false,
            escalation: 0,
        }
    }

    /// Create a protector with custom thresholds.
    pub fn with_thresholds(phi_floor: f64, phi_ceiling: f64) -> Self {
        Self {
            phi_floor,
            phi_ceiling,
            suppression_list: vec![
                "notifications".to_string(),
                "agent_chatter".to_string(),
                "non_urgent_tasks".to_string(),
            ],
            tempo_locked: false,
            protecting: false,
            escalation: 0,
        }
    }

    /// Process a Φ update and return an action if the protector state changes.
    ///
    /// Returns `None` when uncertain (the default — doing nothing well).
    ///
    /// # State Transitions
    ///
    /// | Current State | Φ Update | Action |
    /// |---------------|----------|--------|
    /// | Not protecting | Φ < floor | `LockTempo` (engage) |
    /// | Protecting | Φ drops further | Escalate (`SuppressNotifications`, etc.) |
    /// | Protecting | Φ < ceiling | `None` (hold steady) |
    /// | Protecting | Φ ≥ ceiling | `Release` (disengage) |
    /// | Not protecting | Φ ≥ floor | `None` (nothing to do) |
    pub fn on_phi_update(&mut self, phi: f64) -> Option<ProtectionAction> {
        if !self.protecting {
            // Not currently protecting.
            if phi < self.phi_floor {
                // Flow detected — engage.
                self.protecting = true;
                self.tempo_locked = true;
                self.escalation = 1;
                return Some(ProtectionAction::LockTempo);
            }
            // Not in flow — nothing to do.
            None
        } else {
            // Currently protecting.
            if phi >= self.phi_ceiling {
                // Flow broken — release all protections.
                self.protecting = false;
                self.tempo_locked = false;
                self.escalation = 0;
                return Some(ProtectionAction::Release);
            }

            // Still in flow. Check if we should escalate.
            let target_escalation = if phi < self.phi_floor / 2.0 {
                4 // Deep flow — everything suppressed.
            } else if phi < self.phi_floor * 0.8 {
                3
            } else if phi < self.phi_floor {
                2
            } else {
                1
            };

            if target_escalation > self.escalation {
                self.escalation = target_escalation;
                // Return the next escalation action.
                match target_escalation {
                    2 => {
                        return Some(ProtectionAction::ReduceAgentActivity);
                    }
                    3 => {
                        return Some(ProtectionAction::ClearNonUrgent);
                    }
                    4 => {
                        return Some(ProtectionAction::SuppressNotifications);
                    }
                    _ => None
                }
            }

            // Holding steady — no action needed. "Doing nothing well."
            None
        }
    }

    /// Returns true if the protector is currently active.
    pub fn is_protecting(&self) -> bool {
        self.protecting
    }

    /// Returns true if tempo is currently locked.
    pub fn is_tempo_locked(&self) -> bool {
        self.tempo_locked
    }

    /// Returns the current escalation level (0 = not protecting, 1-4 = increasing).
    pub fn escalation_level(&self) -> u8 {
        self.escalation
    }

    /// Returns the list of items being suppressed.
    pub fn suppression_list(&self) -> &[String] {
        &self.suppression_list
    }

    /// Add an item to the suppression list.
    pub fn suppress(&mut self, item: impl Into<String>) {
        let item = item.into();
        if !self.suppression_list.contains(&item) {
            self.suppression_list.push(item);
        }
    }

    /// Remove an item from the suppression list.
    pub fn unsuppress(&mut self, item: &str) {
        self.suppression_list.retain(|s| s != item);
    }

    /// Force-release all protections (e.g., on user override).
    pub fn force_release(&mut self) {
        self.protecting = false;
        self.tempo_locked = false;
        self.escalation = 0;
    }
}

impl Default for FlowStateProtector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starts_inactive() {
        let p = FlowStateProtector::new();
        assert!(!p.is_protecting());
        assert!(!p.is_tempo_locked());
    }

    #[test]
    fn test_engages_on_low_phi() {
        let mut p = FlowStateProtector::new();
        let action = p.on_phi_update(0.02);
        assert_eq!(action, Some(ProtectionAction::LockTempo));
        assert!(p.is_protecting());
        assert!(p.is_tempo_locked());
    }

    #[test]
    fn test_no_action_when_uncertain() {
        let mut p = FlowStateProtector::new();
        // Φ in the "uncertain zone" — between floor and ceiling, not protecting.
        let action = p.on_phi_update(0.10);
        assert_eq!(action, None);
        assert!(!p.is_protecting());
    }

    #[test]
    fn test_no_action_when_high_phi() {
        let mut p = FlowStateProtector::new();
        let action = p.on_phi_update(0.5);
        assert_eq!(action, None);
        assert!(!p.is_protecting());
    }

    #[test]
    fn test_releases_on_rising_phi() {
        let mut p = FlowStateProtector::new();

        // Engage.
        p.on_phi_update(0.02);
        assert!(p.is_protecting());

        // Φ rises above ceiling.
        let action = p.on_phi_update(0.20);
        assert_eq!(action, Some(ProtectionAction::Release));
        assert!(!p.is_protecting());
        assert!(!p.is_tempo_locked());
    }

    #[test]
    fn test_hysteresis_holds_during_flow() {
        let mut p = FlowStateProtector::new();

        // Engage at low Φ.
        p.on_phi_update(0.02);
        assert!(p.is_protecting());

        // Φ rises but stays below ceiling — should hold.
        let action = p.on_phi_update(0.08);
        assert_eq!(action, None); // Doing nothing well.
        assert!(p.is_protecting());
    }

    #[test]
    fn test_escalation_on_deep_flow() {
        let mut p = FlowStateProtector::new();

        // Engage.
        p.on_phi_update(0.04);
        assert_eq!(p.escalation_level(), 1);

        // Deep flow — escalate.
        let action = p.on_phi_update(0.01);
        assert!(action.is_some());
        assert!(p.escalation_level() > 1);
    }

    #[test]
    fn test_returns_none_on_steady_low_phi() {
        let mut p = FlowStateProtector::new();

        // Engage.
        p.on_phi_update(0.03);

        // Same Φ level — no new action.
        let action = p.on_phi_update(0.03);
        assert_eq!(action, None);
    }

    #[test]
    fn test_force_release() {
        let mut p = FlowStateProtector::new();
        p.on_phi_update(0.01);
        assert!(p.is_protecting());

        p.force_release();
        assert!(!p.is_protecting());
        assert!(!p.is_tempo_locked());
        assert_eq!(p.escalation_level(), 0);
    }

    #[test]
    fn test_suppress_unsuppress() {
        let mut p = FlowStateProtector::new();
        p.suppress("custom_thing");
        assert!(p.suppression_list().contains(&"custom_thing".to_string()));

        p.unsuppress("custom_thing");
        assert!(!p.suppression_list().contains(&"custom_thing".to_string()));
    }

    #[test]
    fn test_engage_release_cycle() {
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
}
