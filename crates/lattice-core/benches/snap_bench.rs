//! Benchmark lattice snapping on random coordinates.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lattice_core::{snap_all, snap_position, EisensteinPoint};

fn bench_snap_position(c: &mut Criterion) {
    // Pre-generate 10,000 random-ish coordinates.
    // We use a simple LCG to avoid pulling in a rand dependency.
    let coords: Vec<(f64, f64)> = (0..10_000)
        .map(|i| {
            let state = i.wrapping_mul(1103515245).wrapping_add(12345);
            let x = ((state >> 16) as f64) * 0.001;
            let y = ((state.wrapping_mul(2654435761) >> 16) as f64) * 0.001;
            (x, y)
        })
        .collect();

    c.bench_function("snap_position_10k", |b| {
        b.iter(|| {
            for &(x, y) in black_box(&coords) {
                let _ = snap_position(x, y);
            }
        })
    });
}

fn bench_from_cartesian(c: &mut Criterion) {
    c.bench_function("from_cartesian_single", |b| {
        b.iter(|| EisensteinPoint::from_cartesian(black_box(3.7), black_box(2.1), 4.0))
    });
}

fn bench_neighbors(c: &mut Criterion) {
    let p = EisensteinPoint::new(100, 200);
    c.bench_function("neighbors", |b| {
        b.iter(|| black_box(p.neighbors()))
    });
}

fn bench_within(c: &mut Criterion) {
    let p = EisensteinPoint::origin();
    c.bench_function("within_radius_5", |b| {
        b.iter(|| black_box(p.within(5)))
    });
}

fn bench_snap_all(c: &mut Criterion) {
    c.bench_function("snap_all_single", |b| {
        b.iter(|| snap_all(black_box(7.3), black_box(5.2), black_box(3.1), black_box(45.0), 1.0))
    });
}

criterion_group!(
    benches,
    bench_snap_position,
    bench_from_cartesian,
    bench_neighbors,
    bench_within,
    bench_snap_all,
);
criterion_main!(benches);
