#![warn(clippy::all)]
#![deny(unsafe_code)]

//! # flux-core
//!
//! Layer 1 of the Slackwater Rust workspace: the FLUX constraint engine.
//!
//! Provides exact arithmetic types, the 8-bit error mask, and SWMIDI event
//! packing. No floats in agreement paths — only in presentation.
//!
//! ## Core types
//!
//! - [`exact::EisensteinCoord`] — A₂ lattice coordinates (exact integer arithmetic)
//! - [`error_mask::ErrorMask`] — 8-bit friction bitfield (0x00 = flow state)
//! - [`swmidi::SwmidiEvent`] / [`swmidi::SwmidiStream`] — Tensor-MIDI 4D events
//!
//! ## Design principles
//!
//! 1. **Exact arithmetic.** Coordinates are Eisenstein integers, velocities
//!    are INT8 (0–127), ticks are uint32. No accumulating float drift.
//! 2. **INT8 saturation.** Bounded quantities clamp at [0, 127].
//! 3. **Errors are data.** Every event carries an error mask, not an exception.

pub mod error_mask;
pub mod exact;
pub mod swmidi;

pub use error_mask::ErrorMask;
pub use exact::{Channel, Confidence, EisensteinCoord, Pitch, Tick, Velocity};
pub use swmidi::{DecodeError, EventType, MetaType, SwmidiEvent, SwmidiStream};
