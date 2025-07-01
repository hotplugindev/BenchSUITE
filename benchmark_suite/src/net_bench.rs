// src/net_bench.rs
// Contains benchmarks for testing network performance.
// Currently implements TCP connection time and echo throughput against a public server.
// Uses Tokio async runtime and Criterion for benchmarking.

use criterion::{black_box, Criterion, BenchmarkId, Throughput};
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::time::Duration;

// Configuration for network benchmarks
const ECHO_SERVER_ADDRESS: &str = "tcpbin.com:4242"; // A public TCP echo server
const DATA_PAYLOAD_SIZE_BYTES: usize = 1024 * 1024; // 1 MB
const CONNECTION_TIMEOUT_SECONDS: u64 = 5;

async fn connect_to_server(address: &str) -> Result<TcpStream, Box<dyn std::error::Error>> {
    match tokio::time::timeout(Duration::from_secs(CONNECTION_TIMEOUT_SECONDS), TcpStream::connect(address)).await {
        Ok(Ok(stream)) => Ok(stream),
        Ok(Err(e)) => Err(Box::new(e)),
        Err(_) => Err("Connection attempt timed out".into()),
    }
}

// Benchmark connection time
fn bench_tcp_connection_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("Net_TCP_ConnectionTime");
    let runtime = tokio::runtime::Runtime::new().unwrap();

    // This benchmark measures the time to establish a TCP connection.
    // It's tricky with Criterion's iteration model for async setup.
    // We'll perform a single connection attempt per "measurement" for simplicity here.
    // A more robust approach might involve multiple connections within the iter_custom loop.
    group.bench_function(BenchmarkId::from_parameter(ECHO_SERVER_ADDRESS), |b| {
        b.to_async(&runtime).iter_batched(
            || (), // No setup needed for each batch iteration
            async |_| {
                match connect_to_server(ECHO_SERVER_ADDRESS).await {
                    Ok(stream) => {
                        // Ensure the stream is properly closed.
                        drop(stream);
                    },
                    Err(e) => {
                        // Log error or handle, but don't panic to allow benchmark to complete if possible
                        eprintln!("Failed to connect for benchmark: {}", e);
                        // We might need to return a specific error or skip if connection fails repeatedly
                    }
                }
            },
            criterion::BatchSize::SmallInput, // Each iteration is one connection attempt
        );
    });

    group.finish();
}

// Benchmark TCP throughput (echo test)
async fn perform_echo_test(stream: &mut TcpStream, payload: &[u8]) -> Result<usize, Box<dyn std::error::Error>> {
    stream.write_all(payload).await?;

    let mut buffer = vec![0u8; payload.len()];
    let mut bytes_read = 0;
    // The echo server should send back exactly what we sent.
    // Need a timeout here as well.
    let timeout_duration = Duration::from_secs(10); // Timeout for the read operation

    match tokio::time::timeout(timeout_duration, async {
        while bytes_read < payload.len() {
            let n = stream.read(&mut buffer[bytes_read..]).await?;
            if n == 0 { // Connection closed prematurely
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Connection closed by peer during echo read"));
            }
            bytes_read += n;
        }
        Ok(bytes_read)
    }).await {
        Ok(Ok(n)) => Ok(n), // Successfully read
        Ok(Err(e)) => Err(Box::new(e)), // Tokio read error
        Err(_) => Err("Echo read timed out".into()), // Timeout error
    }
}


fn bench_tcp_echo_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("Net_TCP_EchoThroughput");
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let payload = vec![0u8; DATA_PAYLOAD_SIZE_BYTES]; // Create a 1MB payload of zeros

    group.throughput(Throughput::Bytes(DATA_PAYLOAD_SIZE_BYTES as u64 * 2)); // Sent + Received

    // This benchmark is more complex due to async nature and connection persistence.
    // For simplicity, we establish a new connection for each iteration.
    // In a real-world scenario, you might want to test throughput over an established connection.
    group.bench_function(BenchmarkId::from_parameter(DATA_PAYLOAD_SIZE_BYTES), |b| {
        b.to_async(&runtime).iter_with_setup(
            // Setup: Connect to the server for each iteration
            || {
                // This setup is async
                let rt = tokio::runtime::Runtime::new().unwrap(); // Mini-runtime for setup
                rt.block_on(async {
                    match connect_to_server(ECHO_SERVER_ADDRESS).await {
                        Ok(stream) => Some(stream),
                        Err(e) => {
                            eprintln!("Failed to connect for throughput benchmark setup: {}", e);
                            None // Skip this iteration if connection fails
                        }
                    }
                })
            },
            // Routine: Perform the echo test
            |stream_option| async {
                if let Some(mut stream) = stream_option {
                    if let Err(e) = perform_echo_test(&mut stream, black_box(&payload)).await {
                        eprintln!("Echo test failed: {}", e);
                    }
                    // stream is dropped here, closing the connection.
                }
                // If stream_option was None, this iteration is effectively skipped.
            },
        );
    });

    group.finish();
}

use tempfile::tempdir;
use std::fs;
use std::path::PathBuf;
use serde_json::Value;

// JSON parsing helper for network benchmarks
fn parse_criterion_json_net(json_path: PathBuf, id: &str, total_bytes_processed: Option<u64>) -> Result<String, String> {
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
            // For network, total_bytes_processed is payload sent + payload received for echo
            if mean_estimate_ns > 0.0 {
                let seconds = mean_estimate_ns / 1_000_000_000.0;
                let mbps = (bytes as f64 / (1024.0 * 1024.0)) / seconds;
                return Ok(format!("{} ({:.2} MB/s)", time_formatted, mbps));
            }
        }
        return Ok(time_formatted); // For connection time, no MB/s
    }
    Err(format!("Could not parse mean point_estimate for {} from JSON: {:?}", id, json_path))
}


pub fn run_network_benchmarks_and_summarize(config: Option<String>) -> Result<String, String> {
    let mut summary = String::from("Network Benchmark Summary (from Criterion JSON):\n");
    summary.push_str(&format!("Target Server: {}\n", ECHO_SERVER_ADDRESS));

    let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir for Net bench: {}", e))?;
    let output_dir = temp_dir.path().to_path_buf();

    // Check connectivity first
    let runtime_check = tokio::runtime::Runtime::new().map_err(|e| format!("Failed to create Tokio runtime for connectivity check: {}", e))?;
    if runtime_check.block_on(TcpStream::connect(ECHO_SERVER_ADDRESS)).is_err() {
        let err_msg = format!("Cannot connect to {}. Network benchmarks skipped.", ECHO_SERVER_ADDRESS);
        eprintln!("{}", err_msg);
        return Ok(format!("{}\n{}", summary, err_msg));
    }    drop(runtime_check);

    // --- TCP Connection Time ---
    let mut crit_conn_time = Criterion::default()
        .output_directory(&output_dir)
        .measurement_time(Duration::from_secs(15))
        .sample_size(10);
    bench_tcp_connection_time(&mut crit_conn_time);
    // BenchmarkId was from_parameter(ECHO_SERVER_ADDRESS)
    let conn_time_json_path = output_dir.join("Net_TCP_ConnectionTime")
                                       .join(ECHO_SERVER_ADDRESS.replace(":", "_")) // Sanitize filename from address
                                       .join("estimates.json");
    match parse_criterion_json_net(conn_time_json_path, "TCP Connection Time", None) {
        Ok(res) => summary.push_str(&format!("- TCP Connection Time: {}\n", res)),
        Err(e) => summary.push_str(&format!("- TCP Connection Time: Error - {}\n", e)),
    }

    // --- TCP Echo Throughput ---
    let mut crit_echo_tp = Criterion::default()
        .output_directory(&output_dir)
        .measurement_time(Duration::from_secs(15))
        .sample_size(10);
    bench_tcp_echo_throughput(&mut crit_echo_tp);
    // BenchmarkId was from_parameter(DATA_PAYLOAD_SIZE_BYTES)
    let echo_tp_json_path = output_dir.join("Net_TCP_EchoThroughput")
                                     .join(&format!("{}", DATA_PAYLOAD_SIZE_BYTES))
                                     .join("estimates.json");
    // Throughput is for payload sent + received
    match parse_criterion_json_net(echo_tp_json_path, "TCP Echo Throughput", Some(DATA_PAYLOAD_SIZE_BYTES as u64 * 2)) {
        Ok(res) => summary.push_str(&format!("- TCP Echo Throughput ({}KB payload): {}\n", DATA_PAYLOAD_SIZE_BYTES / 1024, res)),
        Err(e) => summary.push_str(&format!("- TCP Echo Throughput ({}KB payload): Error - {}\n", DATA_PAYLOAD_SIZE_BYTES / 1024, e)),
    }

    Ok(summary)
}

pub fn run_network_benchmarks_cli(config: Option<String>) {
    println!("Running Network benchmarks via CLI...");
    println!("Using echo server: {}", ECHO_SERVER_ADDRESS);
    println!("Ensure you have an active internet connection.");

    // Perform connectivity check for CLI as well
    let runtime_check = tokio::runtime::Runtime::new().unwrap();
    if runtime_check.block_on(TcpStream::connect(ECHO_SERVER_ADDRESS)).is_err() {
        eprintln!("Cannot connect to {}. Skipping network benchmarks for CLI.", ECHO_SERVER_ADDRESS);
        return;
    }
    drop(runtime_check);

    match run_network_benchmarks_and_summarize(config.clone()) {
        Ok(summary) => println!("Summary:\n{}", summary),
        Err(e) => eprintln!("Network Benchmark Error during summarization: {}", e),
    }

    println!("\nDetailed Criterion output for Network (CLI):");
    let mut c_conn_detailed = Criterion::default()
        .with_plots()
        .measurement_time(Duration::from_secs(15))
        .sample_size(10);
    bench_tcp_connection_time(&mut c_conn_detailed);
    
    let mut c_echo_detailed = Criterion::default()
        .with_plots()
        .measurement_time(Duration::from_secs(15))
        .sample_size(10);
    bench_tcp_echo_throughput(&mut c_echo_detailed);

    println!("Network benchmarks finished for CLI.");
}

#[cfg(test)]
mod tests {
    use super::*;

    // This test requires an internet connection and the ECHO_SERVER_ADDRESS to be available.
    // It might be flaky if the network or server is temporarily down.
    // Consider adding features to skip network-dependent tests or using a local mock server.
    #[tokio::test]
    async fn test_run_network_benchmarks_and_summarize_runs() {
        // Check if we can even attempt a connection; if not, the test result might be trivial (just a skip message).
        let can_connect = tokio::net::TcpStream::connect(ECHO_SERVER_ADDRESS).await.is_ok();

        let result = run_network_benchmarks_and_summarize(None);
        assert!(result.is_ok(), "Network summarizer failed: {:?}", result.err());
        let summary = result.unwrap();

        assert!(summary.contains("Network Benchmark Summary"), "Summary title missing");
        assert!(summary.contains(ECHO_SERVER_ADDRESS), "Echo server address missing in summary");

        if can_connect {
            // If connection was possible, expect more detailed results
            assert!(summary.contains("TCP Connection Time"), "TCP Connection Time section missing");
            assert!(summary.contains("TCP Echo Throughput"), "TCP Echo Throughput section missing");
            // We don't assert specific numbers, just that the sections are there.
            // Could also check for "Error" in case the benchmarks ran but failed for other reasons.
        } else {
            // If connection was not possible, expect a message indicating benchmarks were skipped.
            assert!(summary.contains("Network benchmarks skipped") || summary.contains("Cannot connect"),
                    "Expected skip message not found when connection failed. Summary: {}", summary);
        }
    }

    #[tokio::test]
    async fn test_perform_echo_test_basic() {
        let payload = b"hello echo server";
        match connect_to_server(ECHO_SERVER_ADDRESS).await {
            Ok(mut stream) => {
                let result = perform_echo_test(&mut stream, payload).await;
                assert!(result.is_ok(), "Echo test failed: {:?}", result.err());
                let bytes_read = result.unwrap();
                assert_eq!(bytes_read, payload.len(), "Echo test did not return expected number of bytes.");
            }
            Err(e) => {
                // If connection fails, this test part is skipped.
                // This is acceptable as the main summarizer test covers the "cannot connect" path.
                println!("Skipping test_perform_echo_test_basic due to connection failure: {}", e);
            }
        }
    }
}
