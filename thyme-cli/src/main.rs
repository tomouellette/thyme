// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use clap::{Parser, Subcommand};
use thyme_cli::{download, measure, neural, process, profile, utils};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    name: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Download(download::DownloadArgs),
    Measure(measure::MeasureArgs),
    Neural(neural::NeuralArgs),
    Process(process::ProcessArgs),
    Profile(profile::ProfileArgs),
    Utils(utils::UtilsArgs),
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Download(download_args)) => download::download(download_args),
        Some(Commands::Measure(measure_args)) => measure::measure(measure_args),
        Some(Commands::Neural(neural_args)) => neural::neural(neural_args),
        Some(Commands::Process(process_args)) => process::process(process_args),
        Some(Commands::Profile(profile_args)) => profile::profile(profile_args),
        Some(Commands::Utils(utils_args)) => utils::utils(utils_args),
        None => {}
    }
}
