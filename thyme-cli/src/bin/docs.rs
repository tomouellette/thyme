#![allow(clippy::all)]
use clap::{Parser, Subcommand};
use clap_markdown;

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
    let args = Cli::parse();
    clap_markdown::print_help_markdown::<Cli>();
}
