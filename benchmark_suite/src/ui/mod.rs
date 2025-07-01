// src/ui/mod.rs
// Implements the Iced-based Graphical User Interface for the benchmark suite.
// Defines the application state, messages, update logic, and view rendering.

use iced::{Application, Command, Element, Settings, Theme, executor};
use iced::widget::{button, column, container, text, scrollable, Column, Row, Space, checkbox}; // Added checkbox
use iced::Length;

// --- Application State ---
#[derive(Debug, Default)]
pub struct BenchmarkApp {
    // UI state
    current_page: Page,
    // Benchmark selection state
    run_cpu_bench: bool,
    run_gpu_bench: bool,
    run_ram_bench: bool,
    run_disk_bench: bool,
    run_net_bench: bool,
    run_nn_bench: bool, // Separate toggle for NN benchmarks

    // Benchmark results
    cpu_results: Option<Result<String, String>>, // Store Result directly
    gpu_results: Option<Result<String, String>>, // For general GPU part
    ram_results: Option<Result<String, String>>,
    disk_results: Option<Result<String, String>>,
    net_results: Option<Result<String, String>>,
    nn_results: Option<Result<String, String>>,  // For NN part

    is_benchmarking: bool,
    active_benchmarks: usize, // Counter for active benchmarks
    current_benchmark_status: String,
    run_button_text: String,
}

impl Default for BenchmarkApp {
    fn default() -> Self {
        BenchmarkApp {
            current_page: Page::Home,
            run_cpu_bench: true, // Default some to true for convenience
            run_gpu_bench: true,
            run_ram_bench: true,
            run_disk_bench: true,
            run_net_bench: true,
            run_nn_bench: true,
            cpu_results: None,
            gpu_results: None,
            ram_results: None,
            disk_results: None,
            net_results: None,
            nn_results: None,
            is_benchmarking: false,
            active_benchmarks: 0,
            current_benchmark_status: "Select benchmarks and click 'Run'.".to_string(),
            run_button_text: "Run Selected Benchmarks".to_string(),
        }
    }
}


// --- Messages ---
#[derive(Debug, Clone)]
pub enum Message {
    // Navigation
    NavigateTo(Page),

    // Benchmark selection
    ToggleCpuBench(bool),
    ToggleGpuBench(bool),
    ToggleRamBench(bool),
    ToggleDiskBench(bool),
    ToggleNetBench(bool),
    ToggleNnBench(bool),

    // Actions
    RunSelectedBenchmarks,
    // BenchmarkStarted(String) // Replaced by active_benchmarks counter and status updates
    BenchmarkStepComplete(BenchmarkType, Result<String, String>), // Type, and Ok(results) or Err(error_msg)
    // AllBenchmarksComplete, // This will be implicit when active_benchmarks reaches 0

    // Generic event for Iced (e.g., window events, though not explicitly used yet)
    IcedEvent(iced::Event),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Home,
    Settings, // Placeholder for potential future settings page
    Results,
}

impl Default for Page {
    fn default() -> Self {
        Page::Home
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkType {
    Cpu,
    Gpu,
    Ram,
    Disk,
    Network,
    NeuralNetwork,
}


// --- Application Logic ---
impl Application for BenchmarkApp {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = (); // No flags for now

    fn new(_flags: ()) -> (Self, Command<Message>) {
        (BenchmarkApp::default(), Command::none())
    }

    fn title(&self) -> String {
        String::from("Rust Benchmark Suite")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::NavigateTo(page) => {
                self.current_page = page;
                Command::none()
            }
            Message::ToggleCpuBench(val) => { self.run_cpu_bench = val; Command::none() }
            Message::ToggleGpuBench(val) => { self.run_gpu_bench = val; Command::none() }
            Message::ToggleRamBench(val) => { self.run_ram_bench = val; Command::none() }
            Message::ToggleDiskBench(val) => { self.run_disk_bench = val; Command::none() }
            Message::ToggleNetBench(val) => { self.run_net_bench = val; Command::none() }
            Message::ToggleNnBench(val) => { self.run_nn_bench = val; Command::none() }

            Message::RunSelectedBenchmarks => {
                if self.is_benchmarking {
                    return Command::none();
                }
                self.is_benchmarking = true;
                self.run_button_text = "Running...".to_string();
                self.active_benchmarks = 0;

                // Clear previous results
                self.cpu_results = None;
                self.gpu_results = None;
                self.ram_results = None;
                self.disk_results = None;
                self.net_results = None;
                self.nn_results = None;

                let mut commands = Vec::new();

                if self.run_cpu_bench {
                    self.active_benchmarks += 1;
                    commands.push(Command::perform(perform_cpu_benchmarks_async(None), |res| Message::BenchmarkStepComplete(BenchmarkType::Cpu, res)));
                }
                if self.run_ram_bench {
                    self.active_benchmarks += 1;
                    commands.push(Command::perform(perform_ram_benchmarks_async(None), |res| Message::BenchmarkStepComplete(BenchmarkType::Ram, res)));
                }
                if self.run_disk_bench {
                    self.active_benchmarks += 1;
                    commands.push(Command::perform(perform_disk_benchmarks_async(None), |res| Message::BenchmarkStepComplete(BenchmarkType::Disk, res)));
                }
                if self.run_net_bench {
                    self.active_benchmarks += 1;
                    commands.push(Command::perform(perform_net_benchmarks_async(None), |res| Message::BenchmarkStepComplete(BenchmarkType::Network, res)));
                }
                if self.run_gpu_bench || self.run_nn_bench {
                    self.active_benchmarks += 1; // This one command accounts for both GPU and NN if selected
                    commands.push(Command::perform(perform_gpu_benchmarks_async(None), |res| Message::BenchmarkStepComplete(BenchmarkType::Gpu, res)));
                }

                if commands.is_empty() {
                    self.is_benchmarking = false;
                    self.current_benchmark_status = "No benchmarks selected.".to_string();
                    self.run_button_text = "Run Selected Benchmarks".to_string();
                    return Command::none();
                } else {
                    self.current_benchmark_status = format!("Starting {} benchmark(s)...", self.active_benchmarks);
                }

                self.current_page = Page::Results;
                Command::batch(commands)
            }
            Message::BenchmarkStepComplete(bench_type, result) => {
                match bench_type {
                    BenchmarkType::Cpu => self.cpu_results = Some(result),
                    BenchmarkType::Gpu => { // This handles combined GPU and NN results
                        let result_clone = result.clone(); // Clone for separate error storing if needed
                        if self.run_gpu_bench || self.run_nn_bench { // Only process if one was intended
                            if let Ok(ref result_string) = result {
                                let nn_marker = "Neural Network (tch-rs) Benchmark Summary:";
                                if let Some(nn_start_index) = result_string.find(nn_marker) {
                                    if self.run_gpu_bench {
                                        self.gpu_results = Some(Ok(result_string[..nn_start_index].trim_end().to_string()));
                                    }
                                    if self.run_nn_bench {
                                        self.nn_results = Some(Ok(result_string[nn_start_index..].to_string()));
                                    }
                                } else { // No NN marker
                                    if self.run_gpu_bench {
                                        self.gpu_results = Some(result); // Assign full result to GPU
                                    }
                                    if self.run_nn_bench { // If NN was selected but no marker
                                        self.nn_results = Some(Err("NN benchmark data not found in GPU results.".to_string()));
                                    }
                                }
                            } else { // Original result was an Err
                                if self.run_gpu_bench { self.gpu_results = Some(result_clone.clone()); }
                                if self.run_nn_bench { self.nn_results = Some(result_clone); }
                            }
                        }
                    }
                    BenchmarkType::Ram => self.ram_results = Some(result),
                    BenchmarkType::Disk => self.disk_results = Some(result),
                    BenchmarkType::Network => self.net_results = Some(result),
                    BenchmarkType::NeuralNetwork => {
                        // This case should ideally not be hit if Gpu type triggers combined summary and is parsed.
                        // If it is, it means perform_gpu_benchmarks_async was called with is_nn_only=true,
                        // which is not the current logic. For safety, handle it.
                        if self.run_nn_bench { self.nn_results = Some(result); }
                    }
                }

                if self.active_benchmarks > 0 {
                    self.active_benchmarks -= 1;
                }

                if self.active_benchmarks == 0 {
                    self.is_benchmarking = false;
                    self.current_benchmark_status = "All selected benchmarks complete.".to_string();
                    self.run_button_text = "Run Again".to_string();
                } else {
                    self.current_benchmark_status = format!("{} benchmark(s) remaining...", self.active_benchmarks);
                }
                Command::none()
            }
            Message::IcedEvent(_) => {
                Command::none()
            }
        }
    }

    fn view(&self) -> Element<Message> {
        let header = column![
            text("Rust Benchmark Suite").size(30),
            Row::new()
                .spacing(10)
                .push(button("Home").on_press(Message::NavigateTo(Page::Home)))
                .push(button("Results").on_press(Message::NavigateTo(Page::Results)))
                // .push(button("Settings").on_press(Message::NavigateTo(Page::Settings)))
        ]
        .spacing(10)
        .padding(10);

        let content: Element<Message> = match self.current_page {
            Page::Home => self.view_home(),
            Page::Settings => self.view_settings(),
            Page::Results => self.view_results(),
        };

        container(column![header, content].spacing(20))
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x()
            .into()
    }

    fn theme(&self) -> Self::Theme {
        Theme::Dark // Or Theme::Light, or custom
    }

    fn subscription(&self) -> iced::Subscription<Self::Message> {
        iced::event::listen().map(Message::IcedEvent)
    }
}

// --- UI Views ---
impl BenchmarkApp {
    fn view_home(&self) -> Element<Message> {
        let mut selection_column = Column::new()
            .spacing(10)
            .push(text("Select Benchmarks to Run:").size(20));

        selection_column = selection_column
            .push(iced::widget::checkbox("CPU Benchmarks", self.run_cpu_bench).on_toggle(Message::ToggleCpuBench))
            .push(iced::widget::checkbox("RAM Benchmarks", self.run_ram_bench).on_toggle(Message::ToggleRamBench))
            .push(iced::widget::checkbox("Disk Benchmarks", self.run_disk_bench).on_toggle(Message::ToggleDiskBench))
            .push(iced::widget::checkbox("Network Benchmarks", self.run_net_bench).on_toggle(Message::ToggleNetBench))
            .push(iced::widget::checkbox("GPU (General) Benchmarks", self.run_gpu_bench).on_toggle(Message::ToggleGpuBench))
            .push(iced::widget::checkbox("GPU (Neural Network) Benchmarks", self.run_nn_bench).on_toggle(Message::ToggleNnBench));

        let run_button_widget = if self.is_benchmarking {
            button(text(&self.run_button_text).horizontal_alignment(iced::alignment::Horizontal::Center)).padding(10)
        } else {
            button(text(&self.run_button_text).horizontal_alignment(iced::alignment::Horizontal::Center))
                .padding(10)
                .on_press(Message::RunSelectedBenchmarks)
        };

        selection_column = selection_column
            .push(Space::with_height(Length::Fixed(20.0)))
            .push(run_button_widget.width(Length::Fixed(250.0))) // Give button a fixed width
            .push(Space::with_height(Length::Fixed(10.0)))
            .push(text(&self.current_benchmark_status).size(16));

        container(selection_column)
            .width(Length::Fill)
            .center_x()
            .padding(20)
            .into()
    }

    fn view_settings(&self) -> Element<Message> {
        container(text("Settings Page - TBD"))
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x()
            .center_y()
            .into()
    }

    fn view_results(&self) -> Element<Message> {
        let results_display = column![
            text("Benchmark Results:").size(24),
            Space::with_height(Length::Fixed(10.0)),
            text(&self.current_benchmark_status).size(16), // Shows overall status like "All complete" or "X remaining"
            Space::with_height(Length::Fixed(20.0)),
            result_entry("CPU:", &self.cpu_results, self.run_cpu_bench),
            result_entry("RAM:", &self.ram_results, self.run_ram_bench),
            result_entry("Disk:", &self.disk_results, self.run_disk_bench),
            result_entry("Network:", &self.net_results, self.run_net_bench),
            result_entry("GPU (General):", &self.gpu_results, self.run_gpu_bench),
            result_entry("GPU (Neural Network):", &self.nn_results, self.run_nn_bench),
        ]
        .spacing(15) // Increased spacing a bit
        .padding(20);

        scrollable(container(results_display).width(Length::Fill).center_x()).into()
    }
}

fn result_entry<'a>(label: &str, result_option: &Option<Result<String, String>>, was_selected: bool) -> Element<'a, Message> {
    let content_str = match result_option {
        Some(Ok(summary)) => summary.clone(),
        Some(Err(e)) => format!("Error: {}", e),
        None => if was_selected { "Pending..." } else { "Not selected" }.to_string(),
    };

    column![
        text(label).size(18), // Slightly larger label
        container(text(content_str))
            .style(iced::theme::Container::Box) // Add a subtle box for each result
            .padding(5)
    ]
    .spacing(5)
    .into()
}


// --- Placeholder async functions for benchmarks ---
// These would wrap the actual benchmark calls from other modules.
// They need to be `async` to be used with `Command::perform`.
// The actual benchmark functions (e.g. `cpu_bench::run_cpu_benchmarks`) are synchronous
// and use Criterion, which has its own way of running and reporting.
// Bridging Criterion's output into this async UI model will be the challenging part.
// For now, these placeholders just simulate work and return a string.

// --- Async Wrappers for Benchmark Execution ---

// CPU Benchmark Async Wrapper
async fn perform_cpu_benchmarks_async(config: Option<String>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        // This is where you call your actual synchronous benchmark function
        // Ensure this function returns Result<String, String> or adapt as needed.
        crate::cpu_bench::run_cpu_benchmarks_and_summarize(config)
    })
    .await
    .map_err(|e| format!("CPU benchmark task failed: {}", e)) // Handle JoinError
    .and_then(|res| res) // Flatten Result<Result<_,_>,_> if run_... itself returns Result
}

// RAM Benchmark Async Wrapper
async fn perform_ram_benchmarks_async(config: Option<String>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        crate::ram_bench::run_ram_benchmarks_and_summarize(config)
    })
    .await
    .map_err(|e| format!("RAM benchmark task failed: {}", e))
    .and_then(|res| res)
}

// Disk Benchmark Async Wrapper
async fn perform_disk_benchmarks_async(config: Option<String>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        crate::disk_bench::run_disk_benchmarks_and_summarize(config)
    })
    .await
    .map_err(|e| format!("Disk benchmark task failed: {}", e))
    .and_then(|res| res)
}

// Network Benchmark Async Wrapper
async fn perform_net_benchmarks_async(config: Option<String>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        crate::net_bench::run_network_benchmarks_and_summarize(config)
    })
    .await
    .map_err(|e| format!("Network benchmark task failed: {}", e))
    .and_then(|res| res)
}

// GPU Benchmarks (General + NN) Async Wrapper
async fn perform_gpu_benchmarks_async(config: Option<String>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        crate::gpu_bench::run_all_gpu_benchmarks_and_summarize(config)
    })
    .await
    .map_err(|e| format!("GPU/NN benchmark task failed: {}", e))
    .and_then(|res| res)
}

// --- Main UI launch function ---
pub fn run_ui() -> iced::Result {
    BenchmarkApp::run(Settings {
        window: iced::window::Settings {
            size: (800, 600),
            ..iced::window::Settings::default()
        },
        ..Settings::default()
    })
}
