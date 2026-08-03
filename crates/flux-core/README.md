# flux-core

**Layer 1 — the FLUX constraint engine. Exact arithmetic, 8-bit error mask, and SWMIDI binary event packing.**

flux-core is the foundation of the Slackwater stack. Every other crate depends on its types: `EisensteinCoord` for positions, `ErrorMask` for error propagation, and `SwmidiEvent` / `SwmidiStream` for the wire format.

---

## Module index

| Module | Exports | Purpose |
|--------|---------|---------|
| `exact` | `EisensteinCoord`, `Velocity`, `Pitch`, `Tick`, `Channel`, `Confidence` | Exact integer coordinates, INT8 saturation |
| `error_mask` | `ErrorMask` | 8-bit friction bitfield — the whole error philosophy in one byte |
| `swmidi` | `SwmidiEvent`, `SwmidiStream`, `EventType`, `MetaType`, `DecodeError` | Tensor-MIDI 4D events, 8-byte binary packing |

---

## Type reference

### `EisensteinCoord`

```rust
pub struct EisensteinCoord {
    pub a: i32,
    pub b: i32,
}
```

A point on the A₂ (Eisenstein) lattice: `z = a + bω` where `ω = e^(2πi/3) = -½ + i√3/2`.

**Constants:**

| Constant | Value | Meaning |
|----------|-------|---------|
| `LATTICE_SCALE` | `4.0` | Studs per lattice unit (from Grand Plan §4) |
| `FRAC_SQRT_3_2` | `0.8660254037844386` | √3/2, used in coordinate transforms |
| `OMEGA_REAL` | `-0.5` | cos(120°) |
| `OMEGA_IMAG` | `0.8660254037844386` | sin(120°) = √3/2 |
| `ORIGIN` | `new(0, 0)` | Lattice origin |

**Methods:**

| Method | Signature | Returns |
|--------|-----------|---------|
| `new` | `(a: i32, b: i32)` | `EisensteinCoord` |
| `to_cartesian` | `(&self)` | `(f64, f64)` — (x, z) in studs |
| `distance_to` | `(&self, other: &Self)` | `f64` — Euclidean distance |
| `snap_to_lattice` | `(x: f64, z: f64)` | `EisensteinCoord` — nearest lattice point |
| `neighbors` | `(&self)` | `[EisensteinCoord; 6]` — six equidistant neighbors |
| `add` / `sub` | `(&self, &Self)` | `EisensteinCoord` — vector arithmetic |
| `is_origin` | `(&self)` | `bool` |

**Cartesian transform:**

```
x = (a − b/2) · s
z = b · (√3/2) · s
```

where `s = LATTICE_SCALE = 4.0` studs. Inverse transform (snapping) rounds fractional (a, b) to the nearest integers.

### Bounded type aliases

| Alias | Underlying | Range | Clamp behavior |
|-------|------------|-------|----------------|
| `Velocity` | `u8` | [0, 127] | `saturate_i8()` |
| `Confidence` | `u8` | [0, 127] | `saturate_i8()` |
| `Tick` | `u32` | [0, 2³²−1] | N/A (96 PPQ) |
| `Channel` | `u8` | [0, 15] | Packed in status nibble |
| `Pitch` | `u8` | [0, 127] | INT8 saturation |

### Saturation helpers

```rust
pub fn saturate_i8(value: i32) -> u8       // clamp to [0, 127]
pub fn saturating_add(a: u8, b: u8) -> u8  // saturating addition
pub fn saturating_sub(a: u8, b: u8) -> u8  // saturating subtraction
```

---

## ErrorMask — the 8-bit friction bitfield

```rust
pub struct ErrorMask(u8);
```

Every event carries one byte of honesty. `0x00` = flow state (all clear). Any bit set = friction in that dimension. Three or more bits = blocked (route to executive).

### Bit table

| Bit | Mask | Hex | Constant | Meaning |
|-----|------|-----|----------|---------|
| 0 | `0000_0001` | `0x01` | `SPATIAL` | Position collision |
| 1 | `0000_0010` | `0x02` | `TEMPORAL` | Timing violation |
| 2 | `0000_0100` | `0x04` | `SEMANTIC` | Nonsensical output |
| 3 | `0000_1000` | `0x08` | `SAFETY` | Content safety flag |
| 4 | `0001_0000` | `0x10` | `RESOURCE` | Resource unavailable |
| 5 | `0010_0000` | `0x20` | `TOPOLOGY` | Connectivity issue |
| 6 | `0100_0000` | `0x40` | `AUTHORITY` | Permission denied |
| 7 | `1000_0000` | `0x80` | `CONSISTENCY` | State inconsistency |

### Semantic constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `FLOW` | `0x00` | All clear — execute |
| `BLOCKED_ALL` | `0xFF` | Maximum friction |

### Key methods

| Method | Returns | Notes |
|--------|---------|-------|
| `is_flow()` | `bool` | `bits == 0` |
| `friction_count()` | `u8` | `popcount(bits)` |
| `is_blocked()` | `bool` | `friction_count >= 3` |
| `contains(other)` | `bool` | Bitwise AND test |
| `with(other)` / `without(other)` | `Self` | Set/clear dimensions |
| `union(other)` / `intersection(other)` | `Self` | Bitwise OR/AND |

**Operator overloads:** `|`, `&`, `|=`, `&=` for ergonomic composition.

**Design principle:** Errors are data on the same bus as everything else — not exceptions thrown across layer boundaries. A layer that detects a collision sets `SPATIAL` and forwards the event. The mask travels with the event through the entire pipeline.

---

## SWMIDI — Tensor-MIDI binary format

### Binary format (SWMIDI-8)

Each event packs into exactly **8 bytes**, little-endian:

```
 ┌───────────┬───────────┬───────────┬───────────┬───────────────────┐
 │  Byte 0   │  Byte 1   │  Byte 2   │  Byte 3   │  Bytes 4–7        │
 │  status   │  pitch    │  velocity │  errmask  │  tick (u32 LE)    │
 └───────────┴───────────┴───────────┴───────────┴───────────────────┘
```

**Status byte layout:**

```
  bit 7  bit 6  bit 5  bit 4  │  bit 3  bit 2  bit 1  bit 0
  ┌───── type (4 bits)─────┐  │  └──── channel (4 bits)────┘
```

### EventType enumeration

| Variant | Nibble | Meaning |
|---------|--------|---------|
| `NoteOn` | `0` | Build action, placement, activation |
| `NoteOff` | `1` | Release, deactivate, end |
| `ControlChange` | `2` | Parameters, spatial payload (CC pairs) |
| `ProgramChange` | `3` | Pipeline stage transition |
| `Meta` | `4` | Tempo change, Φ reading, convergence, EOT |

Nibbles 5–15 are invalid; `unpack()` returns `DecodeError::InvalidEventType`.

### MetaType (pitch field for META events)

| Variant | Pitch | Meaning |
|---------|-------|---------|
| `TempoChange` | 81 | Tempo change or T-Minus prediction |
| `PhiReading` | 83 | Φ reading from the Governor |
| `Convergence` | 84 | Tracks aligned |
| `EndOfTrack` | 0 | Stream terminator |

### `SwmidiEvent`

```rust
pub struct SwmidiEvent {
    pub event_type: EventType,
    pub channel: Channel,
    pub pitch: Pitch,
    pub velocity: Velocity,
    pub tick: Tick,
    pub error_mask: ErrorMask,
    pub cc: Option<Vec<(u8, u8)>>,  // CC pairs — not in 8-byte packed form
}
```

### `SwmidiStream`

A sequence of events supporting two serializations:

**Binary packing:**

```
 ┌───────────────────┬─────────────────────────────┬───────────────────┐
 │  u32 LE           │  N × 8 bytes                │  CC section       │
 │  event_count      │  (packed events)            │  (see below)      │
 └───────────────────┴─────────────────────────────┴───────────────────┘
```

CC section (only for events with CC pairs):

```
 u32 LE cc_record_count
   ┌─────────────────┬───────────┬──────────────────────────┐
   │  u32 event_idx  │  u8 count │  count × (u8, u8) pairs  │
   └─────────────────┴───────────┴──────────────────────────┘
   ... repeated per event with CC data
```

**JSON:** `serde` serialization with identical field shape. `cc` is `skip_serializing_if = "None"`.

### Stream methods

| Method | Returns | Description |
|--------|---------|-------------|
| `push(event)` | `()` | Append event |
| `at_tick(tick)` | `Vec<&SwmidiEvent>` | Events at exact tick |
| `in_range(start, end)` | `Vec<&SwmidiEvent>` | Events in `[start, end)` |
| `sort_by_tick()` | `()` | Stable sort by tick |
| `pack_binary()` | `Vec<u8>` | Compact binary serialization |
| `unpack_binary(data)` | `Result<Self, DecodeError>` | Binary deserialization |
| `to_json()` | `String` | JSON serialization |
| `from_json(s)` | `Result<Self, DecodeError>` | JSON deserialization |

---

## Benchmark: 100-part build

| Metric | Binary | JSON | Ratio |
|--------|--------|------|-------|
| 100 events (no CC) | **808 bytes** | ~15 KB | ~18× |
| 100 events (5 CC pairs each) | ~2,308 bytes | ~30+ KB | ~13× |
| Per-event overhead | 8 bytes | ~150 bytes | — |

Binary format: `4 (count) + 100 × 8 (events) + 4 (CC count) = 808` bytes for a 100-event build with no CC payload.

---

## Crate metadata

- **Edition:** 2024
- **Dependencies:** `serde`, `serde_json`
- **Dev dependencies:** `criterion`
- **Unsafe code:** `#![deny(unsafe_code)]`
- **Clippy:** `#![warn(clippy::all)]`
- **Tests:** 88 (unit + integration)
