//! Benchmark: binary packing vs JSON serialization for SWMIDI streams.
//!
//! Measures the core thesis: 8-byte binary events vs JSON envelopes.
//! Run with: `cargo bench --bench packing`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use flux_core::{ErrorMask, EventType, SwmidiEvent, SwmidiStream};

fn make_stream(n: usize) -> SwmidiStream {
    let mut stream = SwmidiStream::with_capacity(n);
    for i in 0..n {
        let event = SwmidiEvent::new(
            EventType::NoteOn,
            (i % 16) as u8,
            48 + (i % 12) as u8,
            64 + (i % 64) as u8,
            (i as u32) * 96,
        )
        .with_mask(if i % 7 == 0 {
            ErrorMask::SPATIAL | ErrorMask::SAFETY
        } else {
            ErrorMask::FLOW
        });
        stream.push(event);
    }
    stream
}

fn make_stream_with_cc(n: usize) -> SwmidiStream {
    let mut stream = SwmidiStream::with_capacity(n);
    for i in 0..n {
        let event = SwmidiEvent::new(
            EventType::NoteOn,
            (i % 16) as u8,
            60,
            96,
            (i as u32) * 96,
        )
        .with_cc(vec![
            (16, 64 + (i % 64) as u8),
            (17, 32 + (i % 64) as u8),
            (20, (i % 128) as u8),
            (21, (i % 16) as u8),
        ]);
        stream.push(event);
    }
    stream
}

fn bench_packing(c: &mut Criterion) {
    let mut group = c.benchmark_group("swmidi_packing");

    for size in [10, 50, 100, 500, 1000] {
        let stream = make_stream(size);

        group.bench_with_input(BenchmarkId::new("binary_pack", size), &stream, |b, s| {
            b.iter(|| black_box(s.pack_binary()));
        });

        group.bench_with_input(BenchmarkId::new("json_serialize", size), &stream, |b, s| {
            b.iter(|| black_box(s.to_json()));
        });

        // Also benchmark unpacking
        let packed = stream.pack_binary();
        group.bench_with_input(BenchmarkId::new("binary_unpack", size), &packed, |b, data| {
            b.iter(|| black_box(SwmidiStream::unpack_binary(data).unwrap()));
        });

        let json = stream.to_json();
        group.bench_with_input(BenchmarkId::new("json_deserialize", size), &json, |b, data| {
            b.iter(|| black_box(SwmidiStream::from_json(data).unwrap()));
        });
    }

    group.finish();

    // Benchmark with CC payload
    let mut cc_group = c.benchmark_group("swmidi_with_cc");

    for size in [50, 100, 500] {
        let stream = make_stream_with_cc(size);

        cc_group.bench_with_input(BenchmarkId::new("binary_pack_cc", size), &stream, |b, s| {
            b.iter(|| black_box(s.pack_binary()));
        });

        cc_group.bench_with_input(BenchmarkId::new("json_pack_cc", size), &stream, |b, s| {
            b.iter(|| black_box(s.to_json()));
        });
    }

    cc_group.finish();
}

fn bench_single_event(c: &mut Criterion) {
    let event = SwmidiEvent::new(
        EventType::NoteOn,
        3,
        60,
        96,
        43200,
    )
    .with_mask(ErrorMask::SPATIAL);

    c.bench_function("single_event_pack", |b| {
        b.iter(|| black_box(event.pack()));
    });

    let packed = event.pack();
    c.bench_function("single_event_unpack", |b| {
        b.iter(|| black_box(SwmidiEvent::unpack(black_box(&packed)).unwrap()));
    });
}

criterion_group!(benches, bench_packing, bench_single_event);
criterion_main!(benches);
