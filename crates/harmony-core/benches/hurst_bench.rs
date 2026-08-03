//! Benchmarks for Hurst exponent computation.
//!
//! Measures performance on datasets of increasing size to demonstrate
//! the O(n log n) scaling characteristics of the R/S method.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use harmony_core::hurst::hurst_exponent;

fn generate_trending_data(n: usize) -> Vec<f64> {
    // Generate trending data with some noise — realistic action stream.
    (0..n)
        .map(|i| {
            let trend = i as f64 * 0.01;
            let noise = ((i as f64).sin() * 0.1) + (i as f64 % 7.0 - 3.5) * 0.05;
            trend + noise
        })
        .collect()
}

fn bench_hurst_1000(c: &mut Criterion) {
    let data = generate_trending_data(1_000);
    c.bench_function("hurst_exponent/1000", |b| {
        b.iter(|| {
            black_box(hurst_exponent(black_box(&data)));
        })
    });
}

fn bench_hurst_10000(c: &mut Criterion) {
    let data = generate_trending_data(10_000);
    c.bench_function("hurst_exponent/10000", |b| {
        b.iter(|| {
            black_box(hurst_exponent(black_box(&data)));
        })
    });
}

fn bench_hurst_100000(c: &mut Criterion) {
    let data = generate_trending_data(100_000);
    c.bench_function("hurst_exponent/100000", |b| {
        b.iter(|| {
            black_box(hurst_exponent(black_box(&data)));
        })
    });
}

criterion_group!(benches, bench_hurst_1000, bench_hurst_10000, bench_hurst_100000);
criterion_main!(benches);
