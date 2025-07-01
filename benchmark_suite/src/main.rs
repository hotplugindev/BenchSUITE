// src/main.rs
// Entry point for the Rust Benchmark Suite.
// Handles command-line argument parsing and dispatches to either the CLI or UI handlers.

use clap::Parser;

/// A comprehensive benchmark suite for CPU, GPU, RAM, Disk, and Ethernet.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// Run the benchmark suite with a graphical user interface
    #[clap(short, long)]
    ui: bool,

    /// Specify which CPU benchmark to run
    #[clap(long)]
    cpu_bench: Option<String>, // Later, this could be an enum or specific subcommands

    /// Specify which GPU benchmark to run
    #[clap(long)]
    gpu_bench: Option<String>,

    /// Specify which RAM benchmark to run
    #[clap(long)]
    ram_bench: Option<String>,

    /// Specify which Disk benchmark to run
    #[clap(long)]
    disk_bench: Option<String>,

    /// Specify which Ethernet benchmark to run
    #[clap(long)]
    ethernet_bench: Option<String>,
    // TODO: Add parameters for each benchmark type
}

mod cpu_bench; // Declare the cpu_bench module
mod ram_bench; // Declare the ram_bench module
mod disk_bench; // Declare the disk_bench module
mod net_bench; // Declare the net_bench module
mod gpu_bench; // Declare the gpu_bench module
mod ui; // Declare the ui module

fn main() {
    env_logger::init(); // Initialize logger

    let cli = Cli::parse();

    if cli.ui {
        println!("UI mode requested. Initializing UI...");
        match ui::run_ui() {
            Ok(_) => println!("UI finished."),
            Err(e) => eprintln!("UI error: {:?}", e),
        }
    } else {
        println!("CLI mode requested.");
        if cli.cpu_bench.is_some() {
            println!("CPU benchmark selected: {}", cli.cpu_bench.as_ref().unwrap());
            // Call CPU benchmark function
            // The config string could be used to select specific CPU benchmarks or parameters
            cpu_bench::run_cpu_benchmarks_cli(cli.cpu_bench.clone());
        }
        if cli.gpu_bench.is_some() {
            println!("GPU benchmark selected: {}", cli.gpu_bench.as_ref().unwrap());
            gpu_bench::run_gpu_benchmarks_cli(cli.gpu_bench.clone());
        }
        if cli.ram_bench.is_some() {
            println!("RAM benchmark selected: {}", cli.ram_bench.as_ref().unwrap());
            ram_bench::run_ram_benchmarks_cli(cli.ram_bench.clone());
        }
        if cli.disk_bench.is_some() {
            println!("Disk benchmark selected: {}", cli.disk_bench.as_ref().unwrap());
            disk_bench::run_disk_benchmarks_cli(cli.disk_bench.clone());
        }
        if cli.ethernet_bench.is_some() {
            println!("Ethernet benchmark selected: {}", cli.ethernet_bench.as_ref().unwrap());
            net_bench::run_network_benchmarks_cli(cli.ethernet_bench.clone());
        }

        if cli.cpu_bench.is_none() && cli.gpu_bench.is_none() && cli.ram_bench.is_none() && cli.disk_bench.is_none() && cli.ethernet_bench.is_none() {
            println!("No specific benchmark selected. Running all benchmarks (not yet implemented).");
            // TODO: Implement logic to run all benchmarks or a default set.
        }
    }
}
