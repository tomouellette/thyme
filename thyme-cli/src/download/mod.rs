// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use clap::{Args, Subcommand};

mod benchmark;
mod segmentation;
mod weights;

use benchmark::{DownloadBenchmarkArgs, download_benchmark};
use segmentation::{DownloadSegmentationArgs, download_segmentation};
use weights::{DownloadWeightsArgs, download_weights};

#[derive(Debug, Args)]
#[command(about = "Download standardized bioimaging datasets or pre-trained neural networks.")]
#[command(args_conflicts_with_subcommands = true)]
#[command(arg_required_else_help = true)]
#[command(flatten_help = true)]
pub struct DownloadArgs {
    #[command(subcommand)]
    command: Option<DownloadCommands>,
}

#[derive(Debug, Subcommand)]
enum DownloadCommands {
    Segmentation(DownloadSegmentationArgs),
    Benchmark(DownloadBenchmarkArgs),
    Weights(DownloadWeightsArgs),
}

pub fn download(args: &DownloadArgs) {
    match args.command.as_ref().unwrap() {
        DownloadCommands::Segmentation(segmentation) => download_segmentation(segmentation),
        DownloadCommands::Benchmark(benchmark) => download_benchmark(benchmark),
        DownloadCommands::Weights(weights) => download_weights(weights),
    }
}
