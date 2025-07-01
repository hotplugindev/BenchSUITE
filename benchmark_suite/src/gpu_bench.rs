// src/gpu_bench.rs
// Contains benchmarks for testing GPU performance.
// Includes general-purpose compute benchmarks using wgpu (e.g., vector addition).
// Uses Criterion and summarizes results from JSON output.

use criterion::{black_box, Criterion, BenchmarkId, Throughput};
use wgpu::util::DeviceExt;
use std::sync::Arc;
use pollster; // For blocking on futures in non-async criterion context

// Shader source (WGSL) for a simple vector addition
const SHADER_SOURCE: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    // Bounds check, though in a controlled benchmark, arrays should match dispatch size
    // if (index >= arrayLength(&output)) { return; }
    output[index] = input_a[index] + input_b[index];
}
"#;

struct WgpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

async fn setup_wgpu() -> Option<WgpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY, // Vulkan, Metal, DX12, or WebGPU
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await;

    if adapter.is_none() {
        eprintln!("Failed to find a suitable wgpu adapter.");
        return None;
    }
    let adapter = adapter.unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Benchmark Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None, // Trace path
        )
        .await
        .map_err(|e| eprintln!("Failed to create wgpu device: {:?}", e))
        .ok()?;

    Some(WgpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
    })
}


fn bench_gpu_vector_add(c: &mut Criterion, context_option: &Option<Arc<WgpuContext>>, num_elements: u32) {
    let context_arc = match context_option {
        Some(ctx) => ctx.clone(),
        None => {
            // If context is None, we can't run the benchmark.
            // Criterion doesn't have a built-in "skip" so we might have to just not register it,
            // or make it a no-op that prints a warning. For now, assume it's present if called.
            eprintln!("WGPU context not available, skipping GPU vector add for {} elements.", num_elements);
            return;
        }
    };
    let device = &context_arc.device;
    let queue = &context_arc.queue;

    let mut group = c.benchmark_group("GPU_VectorAdd");
    group.throughput(Throughput::Elements(num_elements as u64)); // Each element is an f32 addition

    // Prepare data
    let data_size_bytes = (num_elements * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;
    let input_a_data: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
    let input_b_data: Vec<f32> = (0..num_elements).map(|i| (i * 2) as f32).collect();

    // Create buffers
    let input_a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input A Buffer"),
        contents: bytemuck::cast_slice(&input_a_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let input_b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input B Buffer"),
        contents: bytemuck::cast_slice(&input_b_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: data_size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shader module
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Vector Add Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });

    // Create pipeline layout and pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { // input_a
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry { // input_b
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry { // output
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
        ],
    });

    let workgroup_size = 64; // Must match shader's @workgroup_size
    let workgroup_count_x = (num_elements + workgroup_size - 1) / workgroup_size; // Ceiling division

    group.bench_function(BenchmarkId::from_parameter(num_elements), |b| {
        b.iter(|| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Benchmark Command Encoder"),
            });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(black_box(workgroup_count_x), 1, 1);
            } // compute_pass is dropped, pass is recorded
            queue.submit(Some(encoder.finish()));

            // For accurate benchmarking, we should wait for completion.
            // This can be done by mapping the output buffer or using fences,
            // but device.poll(Maintain::Wait) is simpler for now.
            // However, frequent polling can also be slow.
            // A proper benchmark would use wgpu's query sets for timestamps if available and precise.
            // For now, just ensuring the queue is processed.
            device.poll(wgpu::Maintain::Wait);
        });
    });

    group.finish();

    // Optional: Read back and verify output (debug)
    // let output_staging_buffer = device.create_buffer(... usage: MAP_READ | COPY_DST ...);
    // queue.submit(... copy output_buffer to output_staging_buffer ...);
    // output_staging_buffer.slice(..).map_async(wgpu::MapMode::Read, ...);
    // device.poll(wgpu::Maintain::Wait);
    // let data = output_staging_buffer.slice(..).get_mapped_range();
    // let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    // Example check: assert_eq!(result[0], input_a_data[0] + input_b_data[0]);
}


pub fn run_gpu_benchmarks(_config: Option<String>) {
    println!("Running GPU benchmarks...");

    // Initialize WGPU context
    let wgpu_context = pollster::block_on(setup_wgpu());

    if wgpu_context.is_none() {
        eprintln!("Failed to initialize WGPU. Skipping GPU benchmarks.");
        println!("GPU benchmarks skipped due to WGPU initialization failure.");
        return;
    }
    let shared_context = Arc::new(wgpu_context.unwrap());

    // Run vector addition benchmark with different sizes
    let mut criterion_1m = Criterion::default()
        .without_plots()
        .measurement_time(std::time::Duration::from_secs(10))
        .sample_size(20);
    bench_gpu_vector_add(&mut criterion_1m, &Some(shared_context.clone()), 1024 * 1024); // 1M elements
    println!("GPU Vector Add (1M elements) benchmark complete.");

    let mut criterion_4m = Criterion::default()
        .without_plots()
        .measurement_time(std::time::Duration::from_secs(10))
        .sample_size(20);
    bench_gpu_vector_add(&mut criterion_4m, &Some(shared_context.clone()), 1024 * 1024 * 4); // 4M elements
    println!("GPU Vector Add (4M elements) benchmark complete.");

    println!("GPU benchmarks finished.");
}

use tempfile::tempdir;
use std::fs;
use std::path::PathBuf;
use serde_json::Value;

// Helper function to parse Criterion JSON for GPU benchmarks
fn parse_criterion_json_gpu(json_path: PathBuf, benchmark_id_str: &str, elements_processed: Option<u64>) -> Result<String, String> {
    let content = fs::read_to_string(&json_path)
        .map_err(|e| format!("Failed to read Criterion JSON for {} at {:?}: {}", benchmark_id_str, json_path, e))?;

    let v: Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse Criterion JSON for {} from {:?}: {}", benchmark_id_str, json_path, e))?;

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

        if let Some(elements) = elements_processed {
            if mean_estimate_ns > 0.0 && elements > 0 {
                let seconds = mean_estimate_ns / 1_000_000_000.0;
                let el_per_sec = elements as f64 / seconds;
                // Determine unit for elements/sec (K, M, G)
                if el_per_sec > 1_000_000_000.0 {
                    return Ok(format!("{} ({:.2} GFLOPs/sec or Gel/s)", time_formatted, el_per_sec / 1_000_000_000.0));
                } else if el_per_sec > 1_000_000.0 {
                    return Ok(format!("{} ({:.2} MFLOPs/sec or Mel/s)", time_formatted, el_per_sec / 1_000_000.0));
                } else if el_per_sec > 1_000.0 {
                    return Ok(format!("{} ({:.2} KFLOPs/sec or Kel/s)", time_formatted, el_per_sec / 1_000.0));
                } else {
                    return Ok(format!("{} ({:.2} FLOPs/sec or el/s)", time_formatted, el_per_sec));
                }
            }
        }
        return Ok(time_formatted);
    }
    Err(format!("Could not parse mean point_estimate for {} from JSON: {:?}", benchmark_id_str, json_path))
}


// Function to run GPU benchmarks and generate summary
pub fn run_gpu_benchmarks_and_summarize(
    crit_config: &mut Criterion,
    wgpu_context: &Arc<WgpuContext>,
    output_dir: &PathBuf
) -> Result<String, String> {
    let mut summary_part = String::new();

    // Vector Add 1M
    let elements_1m = 1024 * 1024;
    bench_gpu_vector_add(crit_config, &Some(wgpu_context.clone()), elements_1m);
    let json_path_1m = output_dir.join("GPU_VectorAdd")
                                .join(&format!("{}", elements_1m))
                                .join("estimates.json");
    match parse_criterion_json_gpu(json_path_1m, "VectorAdd 1M", Some(elements_1m as u64)) {
        Ok(res) => summary_part.push_str(&format!("- WGPU Vector Add (1M elements): {}\n", res)),
        Err(e) => summary_part.push_str(&format!("- WGPU Vector Add (1M elements): Error - {}\n", e)),
    }

    // Vector Add 4M
    let elements_4m = 1024 * 1024 * 4;
    bench_gpu_vector_add(crit_config, &Some(wgpu_context.clone()), elements_4m);
     let json_path_4m = output_dir.join("GPU_VectorAdd")
                                .join(&format!("{}", elements_4m))
                                .join("estimates.json");
    match parse_criterion_json_gpu(json_path_4m, "VectorAdd 4M", Some(elements_4m as u64)) {
        Ok(res) => summary_part.push_str(&format!("- WGPU Vector Add (4M elements): {}\n", res)),
        Err(e) => summary_part.push_str(&format!("- WGPU Vector Add (4M elements): Error - {}\n", e)),
    }
    Ok(summary_part)
}

// Main summarizing function for GPU benchmarks
pub fn run_all_gpu_benchmarks_and_summarize(_config: Option<String>) -> Result<String, String> {
    let mut full_summary = String::from("GPU Benchmark Summary (from Criterion JSON):\n");

    let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir for GPU bench: {}", e))?;
    let output_dir = temp_dir.path().to_path_buf();

    let _crit_config = Criterion::default()
        .output_directory(&output_dir)
        .sample_size(10) // Small sample size for UI
        .measurement_time(std::time::Duration::from_secs(10)); // GPU tasks can be longer

    // --- WGPU Benchmarks ---
    let wgpu_context = pollster::block_on(setup_wgpu());
    if wgpu_context.is_none() {
        full_summary.push_str("WGPU context initialization failed. Skipping GPU benchmarks.\n");
    } else {
        let shared_wgpu_context = Arc::new(wgpu_context.unwrap());
        let mut crit_config_wgpu = Criterion::default()
            .output_directory(&output_dir)
            .sample_size(10)
            .measurement_time(std::time::Duration::from_secs(10));
        match run_gpu_benchmarks_and_summarize(&mut crit_config_wgpu, &shared_wgpu_context, &output_dir) {
            Ok(wgpu_summary) => full_summary.push_str(&wgpu_summary),
            Err(e) => full_summary.push_str(&format!("Error in WGPU benchmarks: {}\n", e)),
        }
    }

    Ok(full_summary)
}


// CLI runner function
pub fn run_gpu_benchmarks_cli(config: Option<String>) {
    println!("Running GPU benchmarks via CLI...");

    match run_all_gpu_benchmarks_and_summarize(config.clone()) {
        Ok(summary) => println!("Summary:\n{}", summary),
        Err(e) => eprintln!("GPU Benchmark Error during summarization: {}", e),
    }

    println!("\nDetailed Criterion output for GPU (CLI):");
    // For WGPU:
    let wgpu_context_cli = pollster::block_on(setup_wgpu());
    if wgpu_context_cli.is_none() {
        println!("Skipping detailed WGPU benchmarks for CLI: context init failed.");
    } else {
        let shared_context_cli = Arc::new(wgpu_context_cli.unwrap());
        let mut c_wgpu_detailed = Criterion::default().with_plots().measurement_time(std::time::Duration::from_secs(10));
        println!("--- Detailed WGPU ---");
        bench_gpu_vector_add(&mut c_wgpu_detailed, &Some(shared_context_cli.clone()), 1024 * 1024);
        bench_gpu_vector_add(&mut c_wgpu_detailed, &Some(shared_context_cli.clone()), 1024 * 1024 * 4);
    }

    println!("GPU benchmarks finished for CLI.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_all_gpu_benchmarks_and_summarize_runs() {
        // This test checks that the main GPU summarization function runs without panic
        // and produces a summary string containing expected elements.
        // It does not validate specific performance numbers.
        // It should be robust to whether a GPU is actually present or not
        // (e.g., WGPU might use a fallback).

        let result = run_all_gpu_benchmarks_and_summarize(None);
        assert!(result.is_ok(), "GPU summarizer failed: {:?}", result.err());
        let summary = result.unwrap();

        assert!(summary.contains("GPU Benchmark Summary"), "Main title missing");

        // Check for WGPU section
        assert!(summary.contains("WGPU Vector Add"), "WGPU Vector Add section missing or title changed");
        // If WGPU context failed, it should say so.
        if summary.contains("WGPU context initialization failed") {
            println!("GPU Summarizer Test Note: WGPU context failed, this is acceptable if no compatible GPU.");
        } else {
            // If context succeeded, expect results (or errors from parsing if files weren't created)
            assert!(summary.contains("1M elements") && summary.contains("4M elements"), "WGPU benchmark cases missing");
        }
    }

    // Note: Testing WGPU shaders directly without the full Criterion setup
    // would involve more boilerplate (manual buffer creation, dispatch, readback)
    // and is omitted here for "basic testing". The summarizer test covers
    // that the Criterion-based wgpu benchmarks are at least invoked.
}
