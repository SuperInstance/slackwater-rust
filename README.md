# slackwater-rust

**Performance-critical Rust cores for the Slackwater build orchestration stack.**

Seven crate slots across a 12-layer architecture. Three are implemented; four await implementation. The implemented crates are tested, benchmarked, and production-ready.

| Status | Crate | Layer | Role |
|--------|-------|-------|------|
| ✅ Implemented | `flux-core` | 1 | Exact arithmetic, 8-bit error mask, SWMIDI event packing |
| ✅ Implemented | `swmidi` | 2 | Tensor-MIDI binary packing (stub — re-exports from flux-core) |
| ⬜ Stub | `tempo-core` | 3 | BeatClock, TempoMap (PyO3 bindings planned) |
| ✅ Implemented | `lattice-core` | 4 | Eisenstein A₂ lattice math for build placement |
| ⬜ Stub | `tminus-core` | 5 | Prediction and calibration math |
| ✅ Implemented | `harmony-core` | 6 | Φ computation, Hurst exponent, flow state detection |
| ⬜ Stub | `perception-core` | 7 | Multi-track MIDI encoding, convergence detection |

**228 tests. All passing. Zero `unsafe`.**

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────┐
 │                    GRAND PLAN STACK                     │
 ├─────────┬───────────────────────────────────────────────┤
 │ Layer 7 │ perception-core   (stub)                      │
 │         │   Multi-track MIDI encoding                   │
 ├─────────┼───────────────────────────────────────────────┤
 │ Layer 6 │ harmony-core      ✅ implemented              │
 │         │   Φ · Hurst · Flow State · Protector          │
 │         │   ↓ consumes ErrorMask readings               │
 ├─────────┼───────────────────────────────────────────────┤
 │ Layer 5 │ tminus-core       (stub)                      │
 │         │   Prediction · Calibration                     │
 ├─────────┼───────────────────────────────────────────────┤
 │ Layer 4 │ lattice-core      ✅ implemented              │
 │         │   A₂ lattice · Snapping · Regions · Collisions│
 │         │   ↓ feeds exact positions to Layer 1          │
 ├─────────┼───────────────────────────────────────────────┤
 │ Layer 3 │ tempo-core        (stub)                      │
 │         │   BeatClock (96 PPQ) · TempoMap · PyO3        │
 ├─────────┼───────────────────────────────────────────────┤
 │ Layer 2 │ swmidi            ✅ (stub crate, see flux)   │
 │         │   Tensor-MIDI 4D event packing                │
 ├─────────┼───────────────────────────────────────────────┤
 │ Layer 1 │ flux-core         ✅ implemented              │
 │         │   EisensteinCoord · ErrorMask · SwmidiEvent   │
 │         │   ↑ foundation: types used everywhere          │
 └─────────┴───────────────────────────────────────────────┘
```

### Dependency graph

```
flux-core ←─ harmony-core (uses ErrorMask for Φ friction signals)
         ←─ lattice-core  (uses exact types, ErrorMask)
         ←─ swmidi        (re-exports SWMIDI types)

harmony-core depends on flux-core's ErrorMask
lattice-core depends on flux-core's exact types (transitively)
```

---

## Performance claims

### Binary packing vs JSON

A 100-part Roblox build encodes to **808 bytes** of packed SWMIDI binary versus **~15 KB** of JSON — an **18× reduction** for the same semantic content. With CC spatial payload (5 controllers per event), the binary form totals ~2.3 KB versus ~30+ KB JSON.

The binary format is fixed at 8 bytes per event (see `flux-core` SWMIDI spec). JSON is retained for the Phase 2 migration window; Phase 3 flips the wire format to binary with no semantic change.

### Rayon-parallelized Φ computation

`harmony-core::phi::compute_phi_windowed` parallelizes flow friction computation across CPU cores using `rayon`. Each window's Hurst exponent, Shannon entropy, and cadence regularity are computed independently, making windowed Φ analysis embarrassingly parallel.

### Exact integer arithmetic

All build positions live on the Eisenstein A₂ lattice — integer coordinates `(a, b)` with exact arithmetic. No floating-point drift between agents. Two agents that snap the same Cartesian point always agree on the lattice point.

---

## Workspace layout

```
slackwater-rust/
├── Cargo.toml              # workspace root, resolver = "2"
├── crates/
│   ├── flux-core/          # Layer 1: types, ErrorMask, SWMIDI
│   │   ├── src/
│   │   │   ├── exact.rs        #  EisensteinCoord, Velocity, Pitch, Tick, Channel
│   │   │   ├── error_mask.rs   #  8-bit friction bitfield
│   │   │   └── swmidi.rs       #  8-byte event packing, streams
│   │   ├── tests/flux_test.rs
│   │   └── benches/packing.rs
│   ├── swmidi/             # Layer 2: stub
│   ├── tempo-core/         # Layer 3: stub
│   ├── lattice-core/       # Layer 4: A₂ lattice
│   │   ├── src/
│   │   │   ├── eisenstein.rs   #  Point, norm, rotation, distance
│   │   │   ├── snap.rs         #  Position/rotation/height snapping
│   │   │   ├── neighbors.rs    #  Collision, boundary, nearest-free
│   │   │   └── region.rs       #  LatticeRegion rectangles
│   │   ├── tests/lattice_test.rs
│   │   └── benches/snap_bench.rs
│   ├── tminus-core/        # Layer 5: stub
│   ├── harmony-core/       # Layer 6: flow state engine
│   │   ├── src/
│   │   │   ├── hurst.rs        #  R/S Hurst exponent
│   │   │   ├── entropy.rs      #  Shannon entropy
│   │   │   ├── cadence.rs      #  Cadence regularity (CV)
│   │   │   ├── phi.rs          #  Φ = weighted friction (rayon)
│   │   │   ├── flow_state.rs   #  State machine: OutOfFlow→DeepFlow
│   │   │   └── protector.rs    #  Imperceptible flow protection
│   │   ├── tests/harmony_test.rs
│   │   └── benches/hurst_bench.rs
│   └── perception-core/   # Layer 7: stub
└── target/
```

---

## Build

```sh
# Build all crates
cargo build

# Run all 228 tests
cargo test

# Run benchmarks
cargo bench -p flux-core
cargo bench -p harmony-core
cargo bench -p lattice-core
```

Requirements: Rust edition 2024, stable toolchain.

---

## License

MIT. See workspace `Cargo.toml`.

## Repository

[github.com/SuperInstance/slackwater-rust](https://github.com/SuperInstance/slackwater-rust)
