#![warn(clippy::all)]
#![deny(unsafe_code)]

//! # harmony-core
//!
//! High-performance flow state detection — the Rust performance twin of
//! `slackwater-harmony`'s Python flow state module.
//!
//! This is where Rust genuinely outperforms Python: the Hurst exponent
//! computation via rescaled range analysis is O(n log n) with tight inner
//! loops that benefit enormously from zero-cost abstractions and SIMD
//! auto-vectorization.
//!
//! ## Modules
//!
//! - [`hurst`] — Hurst exponent via rescaled range (R/S) analysis
//! - [`entropy`] — Shannon entropy of inter-action intervals
//! - [`cadence`] — Cadence regularity and stability
//! - [`phi`] — Φ (flow friction) computation
//! - [`flow_state`] — Flow state detection state machine
//! - [`protector`] — Flow State Protector (imperceptible adjustments)
//!
//! ## Philosophy
//!
//! Flow is a soap bubble. You don't grab it. You hold still and let the
//! air do the work. When Φ → 0, the player is in flow state, and the
//! system's job inverts — suppress, don't augment.

pub mod cadence;
pub mod entropy;
pub mod flow_state;
pub mod hurst;
pub mod phi;
pub mod protector;

// Re-export the primary types at the crate root for convenience.
pub use flow_state::{FlowState, FlowStateDetector, FlowTrend};
pub use phi::{compute_phi, compute_phi_windowed, PhiWeights};
pub use protector::{FlowStateProtector, ProtectionAction};
