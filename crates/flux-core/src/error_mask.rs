//! The 8-bit error mask — FLUX's core contribution.
//!
//! Every event in the system carries one byte of honesty: a bitfield of
//! constraint violations attached at whatever layer detected them.
//!
//! `errmask == 0x00` means **ALL CLEAR** — flow state. The event executes.
//! Any bit set = friction in that dimension. The event routes to the
//! Harmony executive instead of the world.
//!
//! ## The eight friction dimensions
//!
//! | Bit | Name         | Meaning                          |
//! |-----|--------------|----------------------------------|
//! | 0   | SPATIAL      | position collision               |
//! | 1   | TEMPORAL     | timing violation                 |
//! | 2   | SEMANTIC     | nonsensical output               |
//! | 3   | SAFETY       | content safety flag              |
//! | 4   | RESOURCE     | resource unavailable             |
//! | 5   | TOPOLOGY     | connectivity issue               |
//! | 6   | AUTHORITY    | permission denied                |
//! | 7   | CONSISTENCY  | state inconsistency              |
//!
//! ## Philosophy
//!
//! Errors are data on the same bus as everything else, not exceptions
//! thrown across layer boundaries. The mask is the whole error-handling
//! philosophy in one byte.

use serde::{Deserialize, Serialize};

/// 8-bit error mask: each bit is a friction dimension.
///
/// `errmask == 0x00` means ALL CLEAR = flow state.
/// Any bit set = friction in that dimension.
/// Three or more bits set = blocked (route to executive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ErrorMask(u8);

// ── Constants for each friction dimension ───────────────────────────

impl ErrorMask {
    /// Bit 0: position collision
    pub const SPATIAL: Self = Self(0b0000_0001);
    /// Bit 1: timing violation
    pub const TEMPORAL: Self = Self(0b0000_0010);
    /// Bit 2: nonsensical output
    pub const SEMANTIC: Self = Self(0b0000_0100);
    /// Bit 3: content safety flag
    pub const SAFETY: Self = Self(0b0000_1000);
    /// Bit 4: resource unavailable
    pub const RESOURCE: Self = Self(0b0001_0000);
    /// Bit 5: connectivity issue
    pub const TOPOLOGY: Self = Self(0b0010_0000);
    /// Bit 6: permission denied
    pub const AUTHORITY: Self = Self(0b0100_0000);
    /// Bit 7: state inconsistency
    pub const CONSISTENCY: Self = Self(0b1000_0000);

    /// All clear — flow state. No friction in any dimension.
    pub const FLOW: Self = Self(0b0000_0000);

    /// All bits set — maximum friction.
    pub const BLOCKED_ALL: Self = Self(0b1111_1111);

    /// Create from raw bits.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Get the raw bits.
    #[inline]
    pub const fn bits(&self) -> u8 {
        self.0
    }

    /// Whether this mask indicates pure flow (zero friction).
    #[inline]
    pub const fn is_flow(&self) -> bool {
        self.0 == 0
    }

    /// Count of friction dimensions currently set.
    #[inline]
    pub const fn friction_count(&self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Whether the event is blocked (3+ friction dimensions).
    ///
    /// Per the Grand Plan: events with heavy friction route to the
    /// Harmony executive for repair rather than executing.
    #[inline]
    pub const fn is_blocked(&self) -> bool {
        self.friction_count() >= 3
    }

    /// Check if a specific friction dimension is set.
    #[inline]
    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Set a friction dimension (returns new mask).
    #[inline]
    pub const fn with(&self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Clear a friction dimension (returns new mask).
    #[inline]
    pub const fn without(&self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Intersection of two masks.
    #[inline]
    pub const fn intersection(&self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Union of two masks.
    #[inline]
    pub const fn union(&self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Toggle a friction dimension (returns new mask).
    #[inline]
    pub const fn toggle(&self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// Return a list of set friction dimension names.
    pub fn set_flags(&self) -> Vec<&'static str> {
        let mut flags = Vec::new();
        if self.contains(Self::SPATIAL) {
            flags.push("SPATIAL");
        }
        if self.contains(Self::TEMPORAL) {
            flags.push("TEMPORAL");
        }
        if self.contains(Self::SEMANTIC) {
            flags.push("SEMANTIC");
        }
        if self.contains(Self::SAFETY) {
            flags.push("SAFETY");
        }
        if self.contains(Self::RESOURCE) {
            flags.push("RESOURCE");
        }
        if self.contains(Self::TOPOLOGY) {
            flags.push("TOPOLOGY");
        }
        if self.contains(Self::AUTHORITY) {
            flags.push("AUTHORITY");
        }
        if self.contains(Self::CONSISTENCY) {
            flags.push("CONSISTENCY");
        }
        flags
    }
}

impl Default for ErrorMask {
    fn default() -> Self {
        Self::FLOW
    }
}

impl core::fmt::Display for ErrorMask {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_flow() {
            return write!(f, "FLOW(0x00)");
        }
        let flags = self.set_flags().join("|");
        write!(f, "{flags}(0x{:02X})", self.0)
    }
}

impl core::ops::BitOr for ErrorMask {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl core::ops::BitAnd for ErrorMask {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self.intersection(rhs)
    }
}

impl core::ops::BitOrAssign for ErrorMask {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::ops::BitAndAssign for ErrorMask {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl From<u8> for ErrorMask {
    #[inline]
    fn from(bits: u8) -> Self {
        Self::from_bits(bits)
    }
}

impl From<ErrorMask> for u8 {
    #[inline]
    fn from(mask: ErrorMask) -> u8 {
        mask.bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_state() {
        let mask = ErrorMask::FLOW;
        assert!(mask.is_flow());
        assert_eq!(mask.friction_count(), 0);
        assert!(!mask.is_blocked());
        assert_eq!(mask.bits(), 0x00);
    }

    #[test]
    fn single_flags() {
        assert_eq!(ErrorMask::SPATIAL.bits(), 0x01);
        assert_eq!(ErrorMask::TEMPORAL.bits(), 0x02);
        assert_eq!(ErrorMask::SEMANTIC.bits(), 0x04);
        assert_eq!(ErrorMask::SAFETY.bits(), 0x08);
        assert_eq!(ErrorMask::RESOURCE.bits(), 0x10);
        assert_eq!(ErrorMask::TOPOLOGY.bits(), 0x20);
        assert_eq!(ErrorMask::AUTHORITY.bits(), 0x40);
        assert_eq!(ErrorMask::CONSISTENCY.bits(), 0x80);
    }

    #[test]
    fn combinations() {
        let mask = ErrorMask::SPATIAL | ErrorMask::TEMPORAL;
        assert!(!mask.is_flow());
        assert_eq!(mask.friction_count(), 2);
        assert!(!mask.is_blocked()); // 2 < 3
        assert_eq!(mask.bits(), 0x03);
        assert!(mask.contains(ErrorMask::SPATIAL));
        assert!(mask.contains(ErrorMask::TEMPORAL));
        assert!(!mask.contains(ErrorMask::SAFETY));
    }

    #[test]
    fn blocked_at_three() {
        let mask = ErrorMask::SPATIAL | ErrorMask::SAFETY | ErrorMask::RESOURCE;
        assert_eq!(mask.friction_count(), 3);
        assert!(mask.is_blocked());
    }

    #[test]
    fn all_blocked() {
        let mask = ErrorMask::BLOCKED_ALL;
        assert_eq!(mask.friction_count(), 8);
        assert!(mask.is_blocked());
    }

    #[test]
    fn with_and_without() {
        let base = ErrorMask::SPATIAL | ErrorMask::TEMPORAL;
        let with_safety = base.with(ErrorMask::SAFETY);
        assert!(with_safety.contains(ErrorMask::SAFETY));
        let without_spatial = with_safety.without(ErrorMask::SPATIAL);
        assert!(!without_spatial.contains(ErrorMask::SPATIAL));
        assert!(without_spatial.contains(ErrorMask::TEMPORAL));
        assert!(without_spatial.contains(ErrorMask::SAFETY));
    }

    #[test]
    fn set_flags_names() {
        let mask = ErrorMask::SPATIAL | ErrorMask::SAFETY | ErrorMask::CONSISTENCY;
        let flags = mask.set_flags();
        assert_eq!(flags, vec!["SPATIAL", "SAFETY", "CONSISTENCY"]);
    }

    #[test]
    fn display_flow() {
        assert_eq!(format!("{}", ErrorMask::FLOW), "FLOW(0x00)");
    }

    #[test]
    fn display_friction() {
        let mask = ErrorMask::SPATIAL | ErrorMask::TEMPORAL;
        let s = format!("{mask}");
        assert!(s.contains("SPATIAL|TEMPORAL"));
        assert!(s.contains("0x03"));
    }

    #[test]
    fn u8_roundtrip() {
        let bits: u8 = 0b1010_0101;
        let mask = ErrorMask::from(bits);
        assert_eq!(mask.bits(), bits);
        let back: u8 = mask.into();
        assert_eq!(back, bits);
    }
}
