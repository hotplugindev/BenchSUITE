use std::process::Command;
use std::str;

// Helper to get the path to the benchmark executable
fn get_executable_path() -> String {
    // Assumes `cargo build` has been run and the executable is in target/debug/
    // This path might need adjustment based on the environment or build profiles.
    // For tests run via `cargo test`, CARGO_BIN_EXE_<name> is set.
    let exe_name = "benchmark_suite"; // Match your package name in Cargo.toml
    env!("CARGO_BIN_EXE_benchmark_suite").to_string()
}

#[test]
fn cli_runs_help() {
    let output = Command::new(get_executable_path())
        .arg("--help")
        .output()
        .expect("Failed to execute benchmark suite --help");

    assert!(output.status.success(), "CLI --help exited with error: {:?}", output);
    let stdout = str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.contains("A comprehensive benchmark suite"), "Help message seems incorrect.");
    assert!(stdout.contains("-ui"), "Help message missing -ui flag.");
}

#[test]
fn cli_runs_cpu_bench() {
    let output = Command::new(get_executable_path())
        .arg("--cpu-bench")
        .arg("some_config") // The config string is not strictly used yet by cpu bench
        .output()
        .expect("Failed to execute CPU benchmark via CLI");

    assert!(output.status.success(), "CLI CPU bench exited with error: {:?}", output);
    let stdout = str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.contains("CPU benchmark selected"), "CPU selection message missing.");
    assert!(stdout.contains("CPU Benchmark Summary"), "CPU summary missing.");
    assert!(stdout.contains("Fibonacci(20)"), "Fibonacci result missing in CPU summary.");
    // Also check for detailed Criterion output part
    assert!(stdout.contains("Detailed Criterion output for CLI (CPU)"), "Detailed CPU output missing.");
}

#[test]
fn cli_runs_ram_bench() {
    let output = Command::new(get_executable_path())
        .arg("--ram-bench")
        .arg("some_config")
        .output()
        .expect("Failed to execute RAM benchmark via CLI");

    assert!(output.status.success(), "CLI RAM bench exited with error: {:?}", output);
    let stdout = str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.contains("RAM benchmark selected"), "RAM selection message missing.");
    assert!(stdout.contains("RAM Benchmark Summary"), "RAM summary missing.");
    assert!(stdout.contains("Sequential Write"), "Sequential Write result missing in RAM summary.");
    assert!(stdout.contains("Detailed Criterion output for RAM (CLI)"), "Detailed RAM output missing.");
}

#[test]
fn cli_runs_disk_bench() {
    let output = Command::new(get_executable_path())
        .arg("--disk-bench")
        .arg("some_config")
        .output()
        .expect("Failed to execute Disk benchmark via CLI");

    assert!(output.status.success(), "CLI Disk bench exited with error: {:?}", output);
    let stdout = str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.contains("Disk benchmark selected"), "Disk selection message missing.");
    assert!(stdout.contains("Disk Benchmark Summary"), "Disk summary missing.");
    assert!(stdout.contains("Sequential Write"), "Sequential Write result missing in Disk summary.");
    assert!(stdout.contains("Detailed Criterion output for Disk (CLI)"), "Detailed Disk output missing.");
}

// Network test might be flaky due to external dependency.
#[test]
fn cli_runs_net_bench() {
    let output = Command::new(get_executable_path())
        .arg("--ethernet-bench") // Ensure this matches the flag in main.rs
        .arg("some_config")
        .output()
        .expect("Failed to execute Network benchmark via CLI");

    assert!(output.status.success(), "CLI Net bench exited with error: {:?}", output);
    let stdout = str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.contains("Ethernet benchmark selected"), "Net selection message missing.");
    assert!(stdout.contains("Network Benchmark Summary"), "Net summary missing.");
    // If connection fails, it should print a skip message, which is also a valid outcome.
    assert!(stdout.contains("TCP Connection Time") || stdout.contains("Network benchmarks skipped"), "Net results or skip message missing.");
    if !stdout.contains("Network benchmarks skipped") { // Only check for detailed if not skipped
        assert!(stdout.contains("Detailed Criterion output for Network (CLI)"), "Detailed Net output missing.");
    }
}

// GPU test can be very system-dependent.
// It might fail if no suitable GPU/driver or if CUDA/LibTorch is not set up.
// The test should be tolerant of this.
#[test]
fn cli_runs_gpu_bench() {
    let output = Command::new(get_executable_path())
        .arg("--gpu-bench")
        .arg("some_config")
        .output()
        .expect("Failed to execute GPU benchmark via CLI");

    assert!(output.status.success(), "CLI GPU bench exited with error: {:?}", output);
    let stdout = str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.contains("GPU benchmark selected"), "GPU selection message missing.");
    assert!(stdout.contains("GPU & Neural Network Benchmark Summary"), "GPU/NN summary missing.");

    // Check for WGPU part (or skip message)
    assert!(stdout.contains("WGPU Vector Add") || stdout.contains("WGPU context initialization failed"), "WGPU results or skip message missing.");

    // Check for NN part
    assert!(stdout.contains("Neural Network (tch-rs) Benchmark Summary"), "NN summary part missing.");
    assert!(stdout.contains("Device:"), "NN device info missing.");

    // Detailed output might be conditional on context/device setup.
    // Only assert if the main summary parts indicate success.
    if !stdout.contains("WGPU context initialization failed") {
         assert!(stdout.contains("Detailed WGPU"), "Detailed WGPU output missing when context presumably succeeded.");
    }
    assert!(stdout.contains("Detailed NN (tch-rs on"), "Detailed NN output missing.");
}

#[test]
fn cli_runs_without_specific_bench_flag() {
    // This tests the case where no specific benchmark is selected,
    // which should print a message about running all benchmarks (though not yet implemented).
    let output = Command::new(get_executable_path())
        .output()
        .expect("Failed to execute benchmark suite without specific flags");

    assert!(output.status.success(), "CLI without specific flags exited with error: {:?}", output);
    let stdout = str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.contains("No specific benchmark selected. Running all benchmarks (not yet implemented)."), "Default message missing.");
}
