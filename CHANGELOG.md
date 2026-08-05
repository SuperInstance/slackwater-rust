# Changelog

All notable changes to **slackwater-rust** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-08-03

### Added
- Rust workspace: 7 crate slots across a 12-layer architecture
- **flux-core** (Layer 1): exact arithmetic, 8-bit error mask, SWMIDI event packing
- **lattice-core** (Layer 4): Eisenstein A₂ lattice math for build placement
  - Region system, snapping, collision detection, neighbor queries
- **harmony-core** (Layer 6): Φ computation, Hurst exponent, flow state detection
  - Entropy analysis, cadence tracking, protector logic
- 228 tests — all passing, zero `unsafe`
- Benches for packing (flux-core), snapping (lattice-core), Hurst (harmony-core)
- Engineering-manual READMEs for workspace + all three implemented crates
- crates.io metadata: description, license, repository, keywords
- 4 stub crates: swmidi (Layer 2), tempo-core (Layer 3), tminus-core (Layer 5),
  perception-core (Layer 7)
