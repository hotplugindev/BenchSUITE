// src/ram_bench.rs
// Contains benchmarks for testing RAM performance.
// Focuses on sequential and random read/write speeds.
// Uses Criterion for benchmarking and summarizes results from JSON output.

use criterion::{black_box, Criterion, BenchmarkId, Throughput, criterion_group};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const BUFFER_SIZE_BYTES: usize = 1024 * 1024 * 64; // 64 MB
const ITERATIONS_SMALL: usize = 1000; // For random access, many small operations

// Helper to create a buffer with random data
fn create_random_buffer(size: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut buffer = Vec::with_capacity(size);
    for _ in 0..size {
        buffer.push(rng.gen());
    }
    buffer
}

// Benchmark sequential write speed
fn bench_sequential_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("RAM_SequentialWrite");
    let data_to_write = create_random_buffer(BUFFER_SIZE_BYTES, 42);

    group.throughput(Throughput::Bytes(BUFFER_SIZE_BYTES as u64));
    group.bench_with_input(BenchmarkId::from_parameter(BUFFER_SIZE_BYTES), &data_to_write, |b, data| {
        b.iter(|| {
            let mut target_buffer = Vec::with_capacity(data.len());
            target_buffer.extend_from_slice(black_box(data));
            black_box(target_buffer);
        });
    });
    group.finish();
}

// Benchmark sequential read speed
fn bench_sequential_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("RAM_SequentialRead");
    let source_buffer = create_random_buffer(BUFFER_SIZE_BYTES, 43);

    group.throughput(Throughput::Bytes(BUFFER_SIZE_BYTES as u64));
    group.bench_with_input(BenchmarkId::from_parameter(BUFFER_SIZE_BYTES), &source_buffer, |b, data| {
        b.iter(|| {
            let mut sum: u8 = 0; // Ensure data is actually read
            for byte in data.iter() {
                sum = sum.wrapping_add(*byte);
            }
            black_box(sum);
        });
    });
    group.finish();
}

// Benchmark random read speed
// This is more complex to measure accurately for RAM vs cache effects.
// We'll do many small reads at random locations within a larger buffer.
fn bench_random_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("RAM_RandomRead");
    let data_buffer = create_random_buffer(BUFFER_SIZE_BYTES, 44);
    let mut rng = StdRng::seed_from_u64(45);

    // Generate random indices beforehand to reduce overhead during iteration
    let mut indices = Vec::with_capacity(ITERATIONS_SMALL);
    for _ in 0..ITERATIONS_SMALL {
        indices.push(rng.gen_range(0..data_buffer.len()));
    }

    // Each iteration reads ITERATIONS_SMALL bytes, one byte at a time from random locations
    group.throughput(Throughput::Bytes(ITERATIONS_SMALL as u64));
    group.bench_with_input(BenchmarkId::from_parameter(ITERATIONS_SMALL), &indices, |b, idxs| {
        b.iter(|| {
            let mut sum: u8 = 0;
            for &index in idxs.iter() {
                sum = sum.wrapping_add(black_box(data_buffer[index]));
            }
            black_box(sum);
        });
    });
    group.finish();
}


use tempfile::tempdir;
use std::fs;
use std::path::PathBuf;
use serde_json::Value;

// Helper function to parse Criterion JSON - assuming it's similar to CPU's.
// This could be moved to a shared utility module if it becomes common.
fn parse_criterion_json_ram(json_path: PathBuf, id: &str, throughput_bytes: Option<u64>) -> Result<String, String> {
    let content = fs::read_to_string(&json_path)
        .map_err(|e| format!("Failed to read Criterion JSON for {} at {:?}: {}", id, json_path, e))?;

    let v: Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse Criterion JSON for {} from {:?}: {}", id, json_path, e))?;

    if let Some(mean_estimate_ns) = v.get("mean").and_then(|m| m.get("point_estimate")).and_then(|pe| pe.as_f64()) {
        let time_formatted = if mean_estimate_ns < 1_000.0 {
            format!("{:.2} ns", mean_estimate_ns)
        } else if mean_estimate_ns < 1_000_000.0 {
            format!("{:.2} µs", mean_estimate_ns / 1_000.0)
        } else if mean_estimate_ns < 1_000_000_000.0 {
            format!("{:.2} ms", mean_estimate_ns / 1_000_000.0)
        } else {
            format!("{:.2} s", mean_estimate_ns / 1_000_000_000.0)
        };

        if let Some(bytes) = throughput_bytes {
            if mean_estimate_ns > 0.0 {
                let seconds = mean_estimate_ns / 1_000_000_000.0;
                let mbps = (bytes as f64 / (1024.0 * 1024.0)) / seconds;
                return Ok(format!("{} ({:.2} MB/s)", time_formatted, mbps));
            }
        }
        return Ok(time_formatted);
    }
    Err(format!("Could not parse mean point_estimate for {} from JSON: {:?}", id, json_path))
}


pub fn run_ram_benchmarks_and_summarize(config: Option<String>) -> Result<String, String> {
    let mut summary = String::from("RAM Benchmark Summary (from Criterion JSON):\n");
    let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir for RAM bench: {}", e))?;
    let output_dir = temp_dir.path().to_path_buf();

    let mut crit_config = Criterion::default()
        .output_directory(output_dir.clone())
        .sample_size(10); // Keep sample size small for UI responsiveness

    // --- Sequential Write ---
    // The benchmark function itself defines the group "RAM_SequentialWrite" and bench name from BenchmarkId
    bench_sequential_write(&mut crit_config.clone());
    let seq_write_json_path = output_dir.join("RAM_SequentialWrite")
                                       .join(&format!("{}", BUFFER_SIZE_BYTES)) // BenchmarkId was from_parameter(BUFFER_SIZE_BYTES)
                                       .join("estimates.json");
    match parse_criterion_json_ram(seq_write_json_path, "Sequential Write", Some(BUFFER_SIZE_BYTES as u64)) {
        Ok(res) => summary.push_str(&format!("- Sequential Write ({}MB): {}\n", BUFFER_SIZE_BYTES / (1024*1024), res)),
        Err(e) => summary.push_str(&format!("- Sequential Write ({}MB): Error - {}\n", BUFFER_SIZE_BYTES / (1024*1024), e)),
    }

    // --- Sequential Read ---
    bench_sequential_read(&mut crit_config.clone());
    let seq_read_json_path = output_dir.join("RAM_SequentialRead")
                                      .join(&format!("{}", BUFFER_SIZE_BYTES))
                                      .join("estimates.json");
    match parse_criterion_json_ram(seq_read_json_path, "Sequential Read", Some(BUFFER_SIZE_BYTES as u64)) {
        Ok(res) => summary.push_str(&format!("- Sequential Read ({}MB): {}\n", BUFFER_SIZE_BYTES / (1024*1024), res)),
        Err(e) => summary.push_str(&format!("- Sequential Read ({}MB): Error - {}\n", BUFFER_SIZE_BYTES / (1024*1024), e)),
    }

    // --- Random Read ---
    bench_random_read(&mut crit_config.clone());
    let rand_read_json_path = output_dir.join("RAM_RandomRead")
                                       .join(&format!("{}", ITERATIONS_SMALL)) // BenchmarkId was from_parameter(ITERATIONS_SMALL)
                                       .join("estimates.json");
    // For random read, throughput is often measured in ops/sec rather than MB/s.
    // The parse function currently calculates MB/s if throughput_bytes is Some.
    // We can adapt or just show time. Let's show time, and ops/sec can be derived if needed.
    match parse_criterion_json_ram(rand_read_json_path, "Random Read", Some((ITERATIONS_SMALL * std::mem::size_of::<u8>()) as u64 ) ) {
        Ok(res) => summary.push_str(&format!("- Random Read ({} accesses): {}\n", ITERATIONS_SMALL, res)),
        Err(e) => summary.push_str(&format!("- Random Read ({} accesses): Error - {}\n", ITERATIONS_SMALL, e)),
    }

    Ok(summary)
}

pub fn run_ram_benchmarks_cli(config: Option<String>) {
    println!("Running RAM benchmarks via CLI...");

    match run_ram_benchmarks_and_summarize(config.clone()) {
        Ok(summary) => println!("Summary:\n{}", summary),
        Err(e) => eprintln!("RAM Benchmark Error during summarization: {}", e),
    }

    println!("\nDetailed Criterion output for RAM (CLI):");
    let mut c_detailed = Criterion::default().with_plots();
    bench_sequential_write(&mut c_detailed.clone());
    bench_sequential_read(&mut c_detailed.clone());
    bench_random_read(&mut c_detailed);
    println!("RAM benchmarks finished for CLI.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_ram_benchmarks_and_summarize_runs() {
        // This test mainly checks that the summarization logic executes without panicking
        // and returns a String containing expected substrings.
        // It doesn't validate the performance numbers themselves.
        let result = run_ram_benchmarks_and_summarize(None);
        assert!(result.is_ok(), "RAM summarizer failed: {:?}", result.err());
        let summary = result.unwrap();

        assert!(summary.contains("RAM Benchmark Summary"), "Summary title missing");
        assert!(summary.contains("Sequential Write"), "Seq Write section missing");
        assert!(summary.contains("Sequential Read"), "Seq Read section missing");
        assert!(summary.contains("Random Read"), "Rand Read section missing");
        assert!(summary.contains("MB/s") || summary.contains("accesses"), "Units missing or unexpected");
    }

    #[test]
    fn test_create_random_buffer_works() {
        let size = 1024;
        let seed = 123;
        let buffer = create_random_buffer(size, seed);
        assert_eq!(buffer.len(), size);
        // Check if data seems somewhat random (e.g., not all zeros)
        // This is a weak check for randomness.
        let mut is_all_same = true;
        if size > 1 {
            for i in 1..size {
                if buffer[i] != buffer[0] {
                    is_all_same = false;
                    break;
                }
            }
        } else if size == 0 {
            is_all_same = false; // Empty buffer is not "all same" in this context
        }
        assert!(!is_all_same || size <= 1, "Buffer data appears to be all the same, which is unlikely for random data of sufficient size.");
    }
}
