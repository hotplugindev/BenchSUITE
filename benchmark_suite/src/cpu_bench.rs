// src/cpu_bench.rs
// Contains benchmarks for testing CPU performance.
// Includes tests like Fibonacci sequence calculation and prime number counting.
// Uses Criterion for benchmarking and provides functions to summarize results from JSON output.

use criterion::{black_box, Criterion};

// A simple Fibonacci function to benchmark
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

// A simple prime calculation function (naive)
fn is_prime(num: u64) -> bool {
    if num <= 1 {
        return false;
    }
    for i in 2..=(num as f64).sqrt() as u64 {
        if num % i == 0 {
            return false;
        }
    }
    true
}

fn count_primes_up_to(limit: u64) -> u64 {
    let mut count = 0;
    for i in 2..=limit {
        if is_prime(i) {
            count += 1;
        }
    }
    count
}

// Criterion benchmark functions
pub fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("CPU_Fibonacci");
    group.bench_function("fibonacci_20", |b| b.iter(|| fibonacci(black_box(20))));
    group.bench_function("fibonacci_30", |b| b.iter(|| fibonacci(black_box(30))));
    // Add more parameters or larger numbers if desired, but fibonacci gets slow fast
}

pub fn bench_primes(c: &mut Criterion) {
    let mut group = c.benchmark_group("CPU_Primes");
    group.bench_function("count_primes_up_to_1000", |b| b.iter(|| count_primes_up_to(black_box(1000))));
    group.bench_function("count_primes_up_to_10000", |b| b.iter(|| count_primes_up_to(black_box(10000))));
}

// This setup is for running benchmarks using `cargo bench`.
// We will need to adapt this or call these functions differently for CLI/UI integration.
// For now, this provides the core benchmark logic.

// To make these runnable with `cargo bench`, we'd typically do:
// criterion_group!(benches, bench_fibonacci, bench_primes);
// criterion_main!(benches);
// However, we want to integrate these into our main application,
use tempfile::tempdir;
use std::fs;
use std::path::PathBuf;
use serde_json::Value;

// Function to parse Criterion's estimates.json
fn parse_criterion_json(json_path: PathBuf) -> Result<String, String> {
    let content = fs::read_to_string(&json_path)
        .map_err(|e| format!("Failed to read Criterion JSON output at {:?}: {}", json_path, e))?;

    let v: Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse Criterion JSON from {:?}: {}", json_path, e))?;

    // Assuming the structure of estimates.json:
    // { "mean": { "point_estimate": ..., "confidence_interval": {...}, ...}, ... }
    // We are interested in point_estimate of the mean.
    if let Some(mean_estimate) = v.get("mean").and_then(|m| m.get("point_estimate")).and_then(|pe| pe.as_f64()) {
        // Format the time. Criterion times are in nanoseconds.
        let time_ns = mean_estimate;
        if time_ns < 1_000.0 { // Less than 1 microsecond
            return Ok(format!("{:.2} ns", time_ns));
        } else if time_ns < 1_000_000.0 { // Less than 1 millisecond
            return Ok(format!("{:.2} µs", time_ns / 1_000.0));
        } else if time_ns < 1_000_000_000.0 { // Less than 1 second
            return Ok(format!("{:.2} ms", time_ns / 1_000_000.0));
        } else {
            return Ok(format!("{:.2} s", time_ns / 1_000_000_000.0));
        }
    }
    Err(format!("Could not parse mean point_estimate from JSON: {:?}", json_path))
}


// Function to run benchmarks and capture a summary by parsing Criterion JSON.
pub fn run_cpu_benchmarks_and_summarize(_config: Option<String>) -> Result<String, String> {
    let mut summary = String::from("CPU Benchmark Summary (from Criterion JSON):\n");

    // Create a temporary directory for Criterion's output
    let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;
    let output_dir = temp_dir.path().to_path_buf();

    // Configure Criterion to output JSON to the temp directory
    // We need to run benchmarks one by one or group by group to get distinct JSON files if needed,
    // or parse a more complex JSON if all are in one.
    // Criterion typically creates a structure like: <output_dir>/<group_name>/<benchmark_name>/estimates.json

    // --- Fibonacci Benchmarks ---
    let mut crit_fib = Criterion::default()
        .output_directory(&output_dir)
        .with_plots() // Keep plots for CLI if desired, but we mainly need JSON
        .sample_size(10); // Small sample size for faster UI feedback

    bench_fibonacci(&mut crit_fib); // This will run both fibonacci_20 and fibonacci_30

    // Parse JSON for fibonacci_20
    let fib20_json_path = output_dir.join("CPU_Fibonacci").join("fibonacci_20").join("estimates.json");
    match parse_criterion_json(fib20_json_path) {
        Ok(time_str) => summary.push_str(&format!("- Fibonacci(20): {}\n", time_str)),
        Err(e) => summary.push_str(&format!("- Fibonacci(20): Error parsing results - {}\n", e)),
    }
    // Parse JSON for fibonacci_30
    let fib30_json_path = output_dir.join("CPU_Fibonacci").join("fibonacci_30").join("estimates.json");
    match parse_criterion_json(fib30_json_path) {
        Ok(time_str) => summary.push_str(&format!("- Fibonacci(30): {}\n", time_str)),
        Err(e) => summary.push_str(&format!("- Fibonacci(30): Error parsing results - {}\n", e)),
    }

    // --- Primes Benchmarks ---
    let mut crit_primes = Criterion::default()
        .output_directory(&output_dir)
        .with_plots()
        .sample_size(10);

    bench_primes(&mut crit_primes); // Runs both count_primes_up_to_1000 and count_primes_up_to_10000

    let primes1k_json_path = output_dir.join("CPU_Primes").join("count_primes_up_to_1000").join("estimates.json");
     match parse_criterion_json(primes1k_json_path) {
        Ok(time_str) => summary.push_str(&format!("- Count Primes up to 1000: {}\n", time_str)),
        Err(e) => summary.push_str(&format!("- Count Primes up to 1000: Error parsing results - {}\n", e)),
    }
    let primes10k_json_path = output_dir.join("CPU_Primes").join("count_primes_up_to_10000").join("estimates.json");
    match parse_criterion_json(primes10k_json_path) {
        Ok(time_str) => summary.push_str(&format!("- Count Primes up to 10000: {}\n", time_str)),
        Err(e) => summary.push_str(&format!("- Count Primes up to 10000: Error parsing results - {}\n", e)),
    }

    // The temp_dir will be cleaned up when it goes out of scope.
    Ok(summary)
}

// CLI runner function
pub fn run_cpu_benchmarks_cli(config: Option<String>) {
    println!("Running CPU benchmarks via CLI (Criterion default output will follow if not suppressed)...");

    // Option 1: Use the JSON parsing summary for CLI as well
    match run_cpu_benchmarks_and_summarize(config.clone()) {
        Ok(summary) => println!("Summary:\n{}", summary),
        Err(e) => eprintln!("CPU Benchmark Error during summarization: {}", e),
    }

    // Option 2: Or, for CLI, just run Criterion with its normal output (more detailed)
    // This might be preferred for CLI users.
    println!("\nDetailed Criterion output for CLI (CPU):");
    let mut c_detailed = Criterion::default().with_plots(); // Enable plots for CLI
    bench_fibonacci(&mut c_detailed);
    bench_primes(&mut c_detailed);

    println!("CPU benchmarks finished for CLI.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(2), 1);
        assert_eq!(fibonacci(3), 2);
        assert_eq!(fibonacci(4), 3);
        assert_eq!(fibonacci(10), 55);
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(!is_prime(10));
        assert!(is_prime(13));
        assert!(is_prime(997)); // A larger prime
    }

    #[test]
    fn test_count_primes_up_to() {
        assert_eq!(count_primes_up_to(1), 0); // No primes <= 1
        assert_eq!(count_primes_up_to(2), 1); // Prime: 2
        assert_eq!(count_primes_up_to(10), 4); // Primes: 2, 3, 5, 7
        assert_eq!(count_primes_up_to(30), 10); // Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
    }

    // Basic test for the summarizer - checks if it runs without panic and produces some output.
    // Does not check specific timings as they are variable.
    #[test]
    fn test_run_cpu_benchmarks_and_summarize_runs() {
        let result = run_cpu_benchmarks_and_summarize(None);
        assert!(result.is_ok());
        let summary = result.unwrap();
        assert!(summary.contains("CPU Benchmark Summary"));
        assert!(summary.contains("Fibonacci(20)"));
        assert!(summary.contains("Count Primes up to 1000"));
    }
}
