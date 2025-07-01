// src/disk_bench.rs
// Contains benchmarks for testing disk I/O performance.
// Includes sequential and random read/write tests to a temporary file.
// Uses Criterion for benchmarking and summarizes results from JSON output.

use criterion::{black_box, Criterion, BenchmarkId, Throughput};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;

const DISK_BENCH_FILE_NAME: &str = "disk_benchmark_temp_file.dat";
const FILE_SIZE_BYTES: usize = 1024 * 1024 * 128; // 128 MB
const BLOCK_SIZE_BYTES: usize = 1024 * 4; // 4 KB blocks for random I/O
const NUM_RANDOM_ACCESSES: usize = 1000;

// Helper to create a buffer with random data
fn create_random_data(size: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut buffer = Vec::with_capacity(size);
    for _ in 0..size {
        buffer.push(rng.gen());
    }
    buffer
}

// Prepare and cleanup benchmark file
fn setup_benchmark_file(file_path: &Path, size: usize, fill_data: Option<&[u8]>) -> File {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .truncate(true) // Overwrite if exists
        .open(file_path)
        .expect("Failed to create/open benchmark file");

    if let Some(data) = fill_data {
        file.write_all(data).expect("Failed to write initial data to benchmark file");
    } else {
        // Fill with zeros if no specific data provided, to ensure space is allocated
        let buffer = vec![0u8; BLOCK_SIZE_BYTES.min(size)];
        let mut written = 0;
        while written < size {
            let to_write = buffer.len().min(size - written);
            file.write_all(&buffer[..to_write]).expect("Failed to fill benchmark file");
            written += to_write;
        }
    }
    file.seek(SeekFrom::Start(0)).expect("Failed to seek to start of file");
    file
}

fn cleanup_benchmark_file(file_path: &Path) {
    if file_path.exists() {
        std::fs::remove_file(file_path).expect("Failed to remove benchmark file");
    }
}

// Benchmark sequential write speed to disk
fn bench_sequential_disk_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("Disk_SequentialWrite");
    let data_to_write = create_random_data(FILE_SIZE_BYTES, 50);
    let file_path = Path::new(DISK_BENCH_FILE_NAME);

    group.throughput(Throughput::Bytes(FILE_SIZE_BYTES as u64));
    group.bench_function(BenchmarkId::from_parameter(FILE_SIZE_BYTES), |b| {
        b.iter_with_setup(
            || {
                // Ensure file is fresh for each iteration if measuring file creation + write
                // Or, pre-create if only measuring write to existing (but truncated) file
                cleanup_benchmark_file(&file_path); // Clean before setup
                let file = setup_benchmark_file(&file_path, 0, None); // Create empty
                (file, data_to_write.clone())
            },
            |(mut file, data)| {
                file.write_all(black_box(&data)).expect("Failed to write to disk");
                file.sync_all().expect("Failed to sync disk"); // Ensure data is flushed
            },
        )
    });
    group.finish();
    cleanup_benchmark_file(&file_path); // Final cleanup
}

// Benchmark sequential read speed from disk
fn bench_sequential_disk_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("Disk_SequentialRead");
    let file_data = create_random_data(FILE_SIZE_BYTES, 51);
    let file_path = Path::new(DISK_BENCH_FILE_NAME);

    // Setup file once for the whole benchmark group for reads
    setup_benchmark_file(&file_path, FILE_SIZE_BYTES, Some(&file_data));

    let mut read_buffer = vec![0u8; FILE_SIZE_BYTES];

    group.throughput(Throughput::Bytes(FILE_SIZE_BYTES as u64));
    group.bench_function(BenchmarkId::from_parameter(FILE_SIZE_BYTES), |b| {
        b.iter_with_setup(
            || {
                let mut file = File::open(&file_path).expect("Failed to open file for read");
                file.seek(SeekFrom::Start(0)).expect("Failed to seek for read");
                file
            },
            |mut file| {
                file.read_exact(black_box(&mut read_buffer)).expect("Failed to read from disk");
            },
        )
    });
    group.finish();
    cleanup_benchmark_file(&file_path);
}

// Benchmark random read speed from disk
fn bench_random_disk_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("Disk_RandomRead");
    let file_data = create_random_data(FILE_SIZE_BYTES, 52);
    let file_path = Path::new(DISK_BENCH_FILE_NAME);
    setup_benchmark_file(&file_path, FILE_SIZE_BYTES, Some(&file_data));

    let mut rng = StdRng::seed_from_u64(53);
    let mut block_buffer = vec![0u8; BLOCK_SIZE_BYTES];
    let num_blocks_in_file = FILE_SIZE_BYTES / BLOCK_SIZE_BYTES;

    // Generate random block offsets
    let mut offsets = Vec::with_capacity(NUM_RANDOM_ACCESSES);
    for _ in 0..NUM_RANDOM_ACCESSES {
        offsets.push(rng.gen_range(0..num_blocks_in_file) as u64 * BLOCK_SIZE_BYTES as u64);
    }

    group.throughput(Throughput::Bytes((NUM_RANDOM_ACCESSES * BLOCK_SIZE_BYTES) as u64));
    group.bench_with_input(BenchmarkId::from_parameter(NUM_RANDOM_ACCESSES), &offsets, |b, offs| {
        b.iter_with_setup(
            || File::open(&file_path).expect("Failed to open file for random read"),
            |mut file| {
                for &offset in offs.iter() {
                    file.seek(SeekFrom::Start(black_box(offset))).expect("Seek failed");
                    file.read_exact(black_box(&mut block_buffer)).expect("Random read failed");
                }
            },
        )
    });
    group.finish();
    cleanup_benchmark_file(&file_path);
}


use tempfile::tempdir;
use std::fs;
use std::path::PathBuf;
use serde_json::Value;

// Reusing a similar JSON parsing helper. This could be centralized.
fn parse_criterion_json_disk(json_path: PathBuf, id: &str, total_bytes_processed: Option<u64>) -> Result<String, String> {
    let content = fs::read_to_string(&json_path)
        .map_err(|e| format!("Failed to read Criterion JSON for {} at {:?}: {}", id, json_path, e))?;

    let v: Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse Criterion JSON for {} from {:?}: {}", id, json_path, e))?;

    if let Some(mean_estimate_ns) = v.get("mean").and_then(|m| m.get("point_estimate")).and_then(|pe| pe.as_f64()) {
        let time_formatted = if mean_estimate_ns < 1_000.0 { // ns
            format!("{:.2} ns", mean_estimate_ns)
        } else if mean_estimate_ns < 1_000_000.0 { // µs
            format!("{:.2} µs", mean_estimate_ns / 1_000.0)
        } else if mean_estimate_ns < 1_000_000_000.0 { // ms
            format!("{:.2} ms", mean_estimate_ns / 1_000_000.0)
        } else { // s
            format!("{:.2} s", mean_estimate_ns / 1_000_000_000.0)
        };

        if let Some(bytes) = total_bytes_processed {
            if mean_estimate_ns > 0.0 {
                let seconds = mean_estimate_ns / 1_000_000_000.0;
                if bytes / (1024*1024) > 0 { // If MB make sense
                    let mbps = (bytes as f64 / (1024.0 * 1024.0)) / seconds;
                    return Ok(format!("{} ({:.2} MB/s)", time_formatted, mbps));
                } else { // For smaller ops like random block reads, ops/sec might be better
                    let ops = bytes / BLOCK_SIZE_BYTES as u64; // Assuming total_bytes is multiple of block_size here
                    if ops > 0 && seconds > 0.0 {
                         let iops = ops as f64 / seconds;
                         return Ok(format!("{} ({:.0} IOPS for {}KB blocks)", time_formatted, iops, BLOCK_SIZE_BYTES/1024));
                    }
                }
            }
        }
        return Ok(time_formatted);
    }
    Err(format!("Could not parse mean point_estimate for {} from JSON: {:?}", id, json_path))
}


pub fn run_disk_benchmarks_and_summarize(_config: Option<String>) -> Result<String, String> {
    let mut summary = String::from("Disk Benchmark Summary (from Criterion JSON):\n");
    let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir for Disk bench: {}", e))?;
    let output_dir = temp_dir.path().to_path_buf();
    let file_path_obj = Path::new(DISK_BENCH_FILE_NAME); // Used for cleanup, not directly by Criterion's output pathing

    let _crit_config = Criterion::default()
        .output_directory(&output_dir)
        .sample_size(10); // Small sample size for UI

    // --- Sequential Write ---
    // Note: `bench_sequential_disk_write` handles its own file cleanup per iteration due to nature of test
    let mut crit_seq_write = Criterion::default()
        .output_directory(&output_dir)
        .sample_size(10);
    bench_sequential_disk_write(&mut crit_seq_write);
    let seq_write_json_path = output_dir.join("Disk_SequentialWrite")
                                       .join(&format!("{}", FILE_SIZE_BYTES))
                                       .join("estimates.json");
    match parse_criterion_json_disk(seq_write_json_path, "Sequential Disk Write", Some(FILE_SIZE_BYTES as u64)) {
        Ok(res) => summary.push_str(&format!("- Sequential Write ({}MB): {}\n", FILE_SIZE_BYTES / (1024*1024), res)),
        Err(e) => summary.push_str(&format!("- Sequential Write ({}MB): Error - {}\n", FILE_SIZE_BYTES / (1024*1024), e)),
    }
    cleanup_benchmark_file(&file_path_obj); // Ensure cleanup after this group

    // --- Sequential Read ---
    let mut crit_seq_read = Criterion::default()
        .output_directory(&output_dir)
        .sample_size(10);
    bench_sequential_disk_read(&mut crit_seq_read);
    let seq_read_json_path = output_dir.join("Disk_SequentialRead")
                                      .join(&format!("{}", FILE_SIZE_BYTES))
                                      .join("estimates.json");
    match parse_criterion_json_disk(seq_read_json_path, "Sequential Disk Read", Some(FILE_SIZE_BYTES as u64)) {
        Ok(res) => summary.push_str(&format!("- Sequential Read ({}MB): {}\n", FILE_SIZE_BYTES / (1024*1024), res)),
        Err(e) => summary.push_str(&format!("- Sequential Read ({}MB): Error - {}\n", FILE_SIZE_BYTES / (1024*1024), e)),
    }
    cleanup_benchmark_file(&file_path_obj); // Ensure cleanup

    // --- Random Read ---
    let mut crit_rand_read = Criterion::default()
        .output_directory(&output_dir)
        .sample_size(10);
    bench_random_disk_read(&mut crit_rand_read);
    let rand_read_json_path = output_dir.join("Disk_RandomRead")
                                       .join(&format!("{}", NUM_RANDOM_ACCESSES))
                                       .join("estimates.json");
    let total_bytes_for_random_read = (NUM_RANDOM_ACCESSES * BLOCK_SIZE_BYTES) as u64;
    match parse_criterion_json_disk(rand_read_json_path, "Random Disk Read", Some(total_bytes_for_random_read)) {
        Ok(res) => summary.push_str(&format!("- Random Read ({} accesses of {}KB): {}\n", NUM_RANDOM_ACCESSES, BLOCK_SIZE_BYTES/1024, res)),
        Err(e) => summary.push_str(&format!("- Random Read ({} accesses of {}KB): Error - {}\n", NUM_RANDOM_ACCESSES, BLOCK_SIZE_BYTES/1024, e)),
    }
    cleanup_benchmark_file(&file_path_obj); // Final cleanup

    Ok(summary)
}

pub fn run_disk_benchmarks_cli(config: Option<String>) {
    println!("Running Disk benchmarks via CLI...");
    cleanup_benchmark_file(Path::new(DISK_BENCH_FILE_NAME)); // Initial safety

    match run_disk_benchmarks_and_summarize(config.clone()) {
        Ok(summary) => println!("Summary:\n{}", summary),
        Err(e) => eprintln!("Disk Benchmark Error during summarization: {}", e),
    }

    println!("\nDetailed Criterion output for Disk (CLI):");
    // These bench functions handle their own setup/cleanup of the benchmark file
    let mut c_seq_write_detailed = Criterion::default().with_plots();
    bench_sequential_disk_write(&mut c_seq_write_detailed);
    let mut c_seq_read_detailed = Criterion::default().with_plots();
    bench_sequential_disk_read(&mut c_seq_read_detailed);
    let mut c_rand_read_detailed = Criterion::default().with_plots();
    bench_random_disk_read(&mut c_rand_read_detailed);

    println!("Disk benchmarks finished for CLI.");
    cleanup_benchmark_file(Path::new(DISK_BENCH_FILE_NAME)); // Final safety cleanup
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_run_disk_benchmarks_and_summarize_runs() {
        // Ensures the summarizer runs, creates & cleans up its temp file, and produces output.
        let result = run_disk_benchmarks_and_summarize(None);
        assert!(result.is_ok(), "Disk summarizer failed: {:?}", result.err());
        let summary = result.unwrap();

        assert!(summary.contains("Disk Benchmark Summary"), "Summary title missing");
        assert!(summary.contains("Sequential Write"), "Seq Write section missing");
        assert!(summary.contains("Sequential Read"), "Seq Read section missing");
        assert!(summary.contains("Random Read"), "Rand Read section missing");
        assert!(summary.contains("MB/s") || summary.contains("IOPS"), "Units missing or unexpected");

        // Check that the benchmark file was cleaned up
        assert!(!Path::new(DISK_BENCH_FILE_NAME).exists(), "Benchmark file was not cleaned up");
    }

    #[test]
    fn test_setup_and_cleanup_benchmark_file() {
        let test_file_path = Path::new("test_temp_bench_file.dat");
        let test_file_size = 1024; // 1KB

        // Cleanup before test, in case of previous failed run
        if test_file_path.exists() {
            cleanup_benchmark_file(&test_file_path);
        }

        // Test setup
        let data = create_random_data(test_file_size, 123);
        let mut file = setup_benchmark_file(&test_file_path, test_file_size, Some(&data));
        assert!(test_file_path.exists(), "Benchmark file was not created by setup");

        let metadata = fs::metadata(&test_file_path).expect("Failed to get metadata");
        assert_eq!(metadata.len(), test_file_size as u64, "File size after setup is incorrect");

        // Test read back (simple check)
        let mut read_data = Vec::new();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.read_to_end(&mut read_data).unwrap();
        assert_eq!(read_data, data, "Data read back does not match data written");

        drop(file); // Close file before cleanup

        // Test cleanup
        cleanup_benchmark_file(&test_file_path);
        assert!(!test_file_path.exists(), "Benchmark file was not cleaned up after test");
    }
}
