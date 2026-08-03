# harmony-core

**Layer 6 — flow state detection and protection. The computational heart of the Slackwater Harmony executive.**

harmony-core measures cognitive friction (Φ) and detects when a player enters or leaves flow state. It provides the Hurst exponent, Shannon entropy, cadence regularity, and a hysteresis-driven state machine that transitions through four flow states. A Flow State Protector makes imperceptible adjustments — suppressing notifications, locking tempo, reducing agent chatter — when flow is detected.

---

## Module index

| Module | Primary export | Description |
|--------|---------------|-------------|
| `hurst` | `hurst_exponent(data: &[f64]) -> f64` | Rescaled range (R/S) analysis for long-term memory detection |
| `entropy` | `action_entropy(intervals: &[f64]) -> f64` | Shannon entropy of inter-action intervals |
| `cadence` | `cadence_regularity(intervals: &[f64]) -> f64` | Coefficient of variation → regularity score |
| `phi` | `compute_phi(...) -> f64` | Weighted flow friction (rayon-parallelized) |
| `flow_state` | `FlowStateDetector` | Hysteresis state machine: OutOfFlow → DeepFlow |
| `protector` | `FlowStateProtector` | Imperceptible flow protection with escalation levels |

---

## Φ — flow friction computation

### Formula

```
Φ = w₁ · persistence_friction      // 1 − max(0, H − 0.5) / 0.5
  + w₂ · entropy_friction          // normalized_entropy(intervals)
  + w₃ · cadence_friction          // 1 − cadence_regularity(intervals)
  + w₄ · idle_penalty              // idle_ratio
```

Each component ∈ [0, 1] where 0 = flow-conducive, 1 = friction. Lower Φ = more flow.

### Default weights (`PhiWeights`)

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Persistence (Hurst) | 0.35 | H > 0.5 is the strongest flow signal |
| Entropy | 0.25 | Action regularity matters, less than trending |
| Cadence | 0.25 | Timing regularity complements entropy |
| Idle | 0.15 | Idle time breaks flow but is common |
| **Total** | **1.00** | Weights must sum to 1.0 |

### Component derivations

**Persistence friction:**

```
H = hurst_exponent(action_intervals)
persistence_friction = 1 − max(0, H − 0.5) / 0.5
```

H > 0.5 (trending/persistent) → low friction. H < 0.5 (mean-reverting) → high friction. H = 0.5 (random walk) → 0.5 friction.

**Entropy friction:** `normalized_entropy(intervals)` — Shannon entropy of histogram-bucketed intervals, normalized to [0, 1] by dividing by `log₂(num_bins)`.

**Cadence friction:** `1 − cadence_regularity(intervals)` — inverse of the coefficient of variation regularity score.

**Idle penalty:** Directly proportional to the fraction of time spent idle. Idle intervals are those exceeding 3× the median interval within the window.

### Special case: zero variance

When all intervals are identical (variance < 10⁻¹²), all three signal components are set to 0.0 — perfectly metronomic timing is the maximum flow indicator.

### Parallel windowed computation

`compute_phi_windowed(actions, window_size, weights)` computes Φ across a sliding window of action timestamps, parallelized with `rayon`. Each window position is independent, making this embarrassingly parallel.

```
actions: [t₀, t₁, t₂, ..., tₙ₋₁]
windows: [t₀..t_w], [t₁..t_{w+1}], ..., [t_{n-w}..t_{n-1}]
         ←─ rayon parallel iterator ─→
```

Returns `Vec<f64>` of length `max(0, n − window_size + 1)`.

---

## Hurst exponent (R/S method)

### Algorithm

The Hurst exponent characterizes long-term memory in a time series:

| H range | Behavior | Player state |
|---------|----------|--------------|
| H < 0.5 | Mean-reverting (anti-persistent) | Irregular, oscillating |
| H ≈ 0.5 | Random walk | No clear trend |
| H > 0.5 | Trending (persistent) | **Flow** — each action builds on the last |

**Method:** Divide the series into non-overlapping windows of size `w` (powers of 2 from 4 to n/2). For each window:

1. Compute mean `μ`.
2. Compute cumulative deviation from mean: `Y_t = Σᵢ₌₁ᵗ (xᵢ − μ)`.
3. Range `R = max(Y) − min(Y)`.
4. Standard deviation `S = √(Σ(xᵢ − μ)² / w)`.
5. Rescaled range `R/S = R / S`.

Average R/S across all windows of size `w`. Fit `log(R/S)` vs `log(w)` — the slope is H.

**Complexity:** O(n log n). Window sizes double: 4, 8, 16, ..., n/2.

**Fallback:** For series shorter than 8 points, returns 0.5 (neutral). For series with insufficient log-log points, uses single-window estimate: `H ≈ log(R/S) / log(n)`.

**Result clamped to [0, 1].**

---

## Shannon entropy

```rust
pub fn action_entropy(intervals: &[f64]) -> f64
pub fn normalized_entropy(intervals: &[f64]) -> f64
```

Histogram-based Shannon entropy of inter-action intervals.

**Method:**

1. Find `[min, max]` of intervals.
2. If all identical, return 0.
3. Bucket into `⌈√n⌉` bins (sqrt-rule).
4. Compute `H = −Σ p(x) · log₂(p(x))`.

**Normalization:** `normalized_entropy = H / log₂(num_bins)`, producing [0, 1].

Low entropy = regular cadence = focused player. High entropy = scattered timing = disrupted player.

---

## Cadence regularity

```rust
pub fn cadence_regularity(intervals: &[f64]) -> f64  // → [0, 1]
pub fn cadence_stability(intervals: &[f64], window: usize) -> Vec<f64>
```

Measures rhythmic consistency via the coefficient of variation:

```
μ  = mean(intervals)
σ  = population_stddev(intervals)
CV = σ / μ
regularity = clamp(1.0 − CV, 0, 1)
```

Returns 1.0 for metronomic timing, 0.0 for highly irregular. Requires ≥ 2 intervals.

`cadence_stability` applies a sliding window of `cadence_regularity`, producing a time series showing how rhythm steadies or falters.

---

## Flow state machine

```
 ┌──────────────┐  Φ < threshold    ┌──────────────────┐
 │  OutOfFlow   │ ────────────────→ │  ApproachingFlow │
 │  (Φ high)    │                   │  (Φ dropping)    │
 │              │ ←──────────────── │                  │
 └──────────────┘  Φ ≥ threshold×2  └──────────────────┘
                                          │
                           Φ < threshold  │ sustained min_window
                           ┌──────────────▼──────────┐
                           │       InFlow            │
                           │  (Φ sustained low)      │
                           └──────────────┬──────────┘
                                  │       │
                  Φ < threshold/3 │       │ Φ ≥ threshold
                  sustained       │       │
           ┌──────────────────────▼─┐    │
           │      DeepFlow          │    │
           │  (Φ deeply low)        │────┘
           └────────────────────────┘
```

### States

| State | Entry condition | Exit condition |
|-------|----------------|----------------|
| `OutOfFlow` | Φ ≥ threshold × 2, or fallback after breaking | Φ < threshold × 2 for sustained period |
| `ApproachingFlow` | Φ < threshold × 2 sustained, or Φ < threshold for ≥ min_window/3 | Φ < threshold sustained → InFlow; Φ rises → OutOfFlow |
| `InFlow` | Φ < threshold for `min_window` consecutive observations | Φ < threshold/3 → DeepFlow; Φ ≥ threshold → ApproachingFlow |
| `DeepFlow` | Φ < threshold/3 for `min_window` consecutive observations | Φ ≥ threshold/3 → InFlow; Φ ≥ threshold → ApproachingFlow |

### Defaults

| Parameter | Value | Description |
|-----------|-------|-------------|
| `phi_threshold` | 0.05 | Below this = flow territory |
| `deep_flow_threshold` | threshold / 3 ≈ 0.0167 | Below this = deep flow |
| `min_window` | 10 | Observations to confirm a transition |
| `max_history` | 500 | Readings retained for trend analysis |

### Trend detection

`phi_trend()` compares the average of the last 3 readings against the preceding 3. If ΔΦ < −5% × current level → `Improving`. If ΔΦ > +5% → `Declining`. Otherwise `Stable`.

---

## Flow State Protector

> *"Doing nothing well is safer than doing something clever."*

The protector makes imperceptible adjustments when flow is detected. It **suppresses** stimulation — it never adds. When uncertain, it returns `None` (does nothing).

### Hysteresis

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phi_floor` | 0.05 | Engage protection below this |
| `phi_ceiling` | 0.15 | Release protection above this |

The gap between floor and ceiling prevents rapid engage/disengage cycling.

### Escalation levels

| Level | Trigger | Action |
|-------|---------|--------|
| 1 | Φ < floor (mild flow) | `LockTempo` — freeze BPM adjustments |
| 2 | Φ < floor × 0.8 | `ReduceAgentActivity` — less chatter |
| 3 | Φ < floor × 0.6 | `ClearNonUrgent` — defer non-urgent tasks |
| 4 | Φ < floor / 2 (deep flow) | `SuppressNotifications` — everything quiet |

### State transition table

| Current | Φ Update | Action |
|---------|----------|--------|
| Not protecting | Φ < floor | `LockTempo` (engage at appropriate level) |
| Protecting | Φ drops further | Escalate (`ReduceAgentActivity`, etc.) |
| Protecting | Φ stable within band | `None` (hold — doing nothing well) |
| Protecting | Φ ≥ ceiling | `Release` (disengage) |
| Not protecting | Φ ≥ floor | `None` (nothing to do) |

### Suppression list

Default suppressions: `notifications`, `agent_chatter`, `non_urgent_tasks`. Custom items can be added via `suppress(item)` and removed via `unsuppress(item)`.

---

## Ethics

The protector's core design principle is **suppression over augmentation**. When a player is in flow, the system becomes quieter — it never adds stimulation. Notifications are suppressed, tempo locks, agent chatter reduces, non-urgent tasks defer.

This principle — *"doing nothing well is safer than doing something clever"* — reflects the understanding that interventions during flow state are more likely to break it than to improve it. The protector's `None` return value is not a failure case; it is the correct response to sustained flow.

`force_release()` exists for user override — when the human explicitly wants to break flow protection, the protector yields immediately.

---

## Crate metadata

- **Edition:** 2024
- **Dependencies:** `serde`, `rayon`
- **Dev dependencies:** `criterion`, `approx`
- **Unsafe code:** `#![deny(unsafe_code)]`
- **Clippy:** `#![warn(clippy::all)]`
- **Tests:** 70 (unit + integration + doctests)
