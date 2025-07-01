# Rust Benchmark Suite

A comprehensive benchmark suite for CPU, GPU (General Purpose & Neural Network), RAM, Disk, and Ethernet, written in Rust.
It features both a Command-Line Interface (CLI) and a Graphical User Interface (GUI).

## Features

-   **CPU Benchmarks**: Measures CPU performance using algorithms like Fibonacci calculation and prime number counting.
-   **RAM Benchmarks**: Tests RAM speed with sequential and random read/write operations.
-   **Disk Benchmarks**: Evaluates disk I/O performance for sequential and random file access.
-   **Ethernet Benchmarks**: Assesses network performance (TCP connection time, echo throughput) against a public echo server.
-   **GPU (General Purpose) Benchmarks**: Uses `wgpu` for basic compute shader performance tests (e.g., vector addition).
-   **GPU (Neural Network) Benchmarks**: Employs `tch-rs` (LibTorch bindings) to benchmark MLP inference speed on the GPU (CUDA if available, otherwise CPU).
-   **Dual Interface**:
    -   **CLI**: Run benchmarks directly from the terminal. View summaries and detailed Criterion output.
    -   **GUI**: An `iced`-based graphical interface for selecting, running, and viewing benchmark results.

## Prerequisites

-   **Rust Toolchain**: Install from [rustup.rs](https://rustup.rs/).
-   **Build Tools**: A C compiler (like GCC or Clang) is needed for some dependencies.
-   **For GPU (Neural Network with `tch-rs` and CUDA)**:
    -   **LibTorch**: Download from the [PyTorch website](https://pytorch.org/get-started/locally/). Ensure the version matches what `tch-rs` expects (check `tch` crate documentation). Set the `LIBTORCH` environment variable to its location (e.g., `export LIBTORCH=/path/to/libtorch`).
    -   **CUDA Toolkit**: If you have an NVIDIA GPU and want to run NN benchmarks on it, install the CUDA Toolkit compatible with your driver and LibTorch version.
-   **For GPU (General Purpose with `wgpu`)**:
    -   Drivers supporting Vulkan (Linux/Windows), Metal (macOS), or DirectX 12 (Windows).

## Building

1.  Clone the repository (or ensure you are in the project root directory).
2.  Build the project:
    ```bash
    cargo build
    ```
    For a release build (recommended for benchmarking and smaller executable):
    ```bash
    cargo build --release
    ```

## Running

The executable will be located at `target/debug/benchmark_suite` or `target/release/benchmark_suite`.

### GUI Mode

To run with the Graphical User Interface:

```bash
# For debug build
./target/debug/benchmark_suite -ui

# For release build
./target/release/benchmark_suite -ui
```

In the GUI:
-   Select the benchmarks you want to run from the checkboxes on the "Home" screen.
-   Click "Run Selected Benchmarks".
-   View results on the "Results" screen.

### CLI Mode

To run specific benchmarks from the command line, use the respective flags. Each flag takes an optional (currently unused) configuration string.

-   **CPU**: `--cpu-bench <config_string>`
    ```bash
    ./target/release/benchmark_suite --cpu-bench run_all
    ```
-   **RAM**: `--ram-bench <config_string>`
    ```bash
    ./target/release/benchmark_suite --ram-bench run_all
    ```
-   **Disk**: `--disk-bench <config_string>`
    ```bash
    ./target/release/benchmark_suite --disk-bench run_all
    ```
-   **Ethernet**: `--ethernet-bench <config_string>` (Requires internet connection)
    ```bash
    ./target/release/benchmark_suite --ethernet-bench run_all
    ```
-   **GPU (General + NN)**: `--gpu-bench <config_string>`
    ```bash
    ./target/release/benchmark_suite --gpu-bench run_all
    ```

If no specific benchmark flag is provided, a message will indicate that running all benchmarks via CLI is not yet fully implemented with individual control (it will currently run the default help). The UI is recommended for running multiple selected benchmarks.

The CLI output for each benchmark includes:
1.  A summary derived from Criterion's JSON output.
2.  Detailed, standard Criterion console output.

## Code Structure (Key Modules)

-   `src/main.rs`: Handles CLI argument parsing and launches either CLI or UI mode.
-   `src/ui/mod.rs`: Implements the Iced-based GUI.
-   `src/cpu_bench.rs`: CPU benchmark logic.
-   `src/ram_bench.rs`: RAM benchmark logic.
-   `src/disk_bench.rs`: Disk benchmark logic.
-   `src/net_bench.rs`: Network benchmark logic.
-   `src/gpu_bench.rs`: GPU (wgpu general compute and tch-rs Neural Network) benchmark logic.
-   `tests/`: Contains unit and integration tests.

## Notes on Benchmarking

-   **Release Mode**: Always use `--release` builds for accurate performance measurements. Debug builds are significantly slower.
-   **System State**: For consistent results, ensure your system is relatively idle while benchmarking.
-   **Disk Benchmarks**: These create a temporary file (`disk_benchmark_temp_file.dat`) in the current working directory, which is cleaned up afterwards.
-   **Network Benchmarks**: Depend on the availability of the public echo server (`tcpbin.com:4242`) and your internet connection.
-   **GPU Benchmarks**:
    -   `wgpu` benchmarks will attempt to use the primary high-performance adapter.
    -   `tch-rs` (NN) benchmarks will use CUDA if available and `LIBTORCH` is correctly set up; otherwise, they fall back to CPU. Check console output for device information.
```
