//! Integration tests crate — exercises cross-crate interactions.
//!
//! These tests verify that the layers compose correctly:
//! flux-core types flow through SWMIDI encoding,
//! BeatClock drives T-Minus predictions,
//! Eisenstein coordinates survive lattice packing, etc.

#![warn(clippy::all)]
