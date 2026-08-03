#![warn(clippy::all)]
#![deny(unsafe_code)]

//! lattice-core — Eisenstein A2 lattice for exact build placement.
//!
//! The A2 lattice gives triangular/hexagonal tiling — the natural grid for Roblox builds
//! because parts rotate at 60° increments and snap cleanly. Coordinates are exact integers,
//! never floats — "no accumulating float drift, ever."

pub mod eisenstein;
pub mod neighbors;
pub mod region;
pub mod snap;

pub use eisenstein::EisensteinPoint;
pub use neighbors::{build_boundary, collides, nearest_unoccupied, occupied_in_radius};
pub use region::LatticeRegion;
pub use snap::{snap_all, snap_height, snap_position, snap_rotation, snap_rotation_index, SnappedPlacement};
