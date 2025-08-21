// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use clap::Args;
use colored::Colorize;

use thyme_core::ut::track::progress_log;
use thyme_data::data::Weights;

#[derive(Debug, Args)]
#[command(about = "Download pre-trained neural network weights.")]
pub struct DownloadWeightsArgs {
    #[arg(short, long, help = "Weights name.")]
    pub name: Option<String>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(long, help = "List all available neural net weights.")]
    pub list: bool,

    #[arg(long, help = "Download all available neural net weights.")]
    pub all: bool,
}

pub fn download_weights(args: &DownloadWeightsArgs) {
    if args.list {
        print_weights();
    }

    if args.all {
        progress_log("Downloading all neural net weights to cache", args.verbose);

        for weights in Weights::iter() {
            weights.download(args.verbose);
        }

        std::process::exit(1);
    }

    if args.name.is_none() {
        eprintln!(
            "[thyme::download::weights] The weights --name/-n must be specified. Run `thyme download weights --list` to see all available weights."
        );
        std::process::exit(1);
    }

    let weights = Weights::select(args.name.as_ref().unwrap());
    weights.download(args.verbose);
}

fn print_weights() {
    println!("{:^69}", "\n");
    println!("| {:-^74} |", "");
    println!("| {:^74} |", "thyme".truecolor(103, 194, 69).bold());
    println!("| {:^74} |", "Pre-trained neural network weights");
    println!("| {:-^18} | {:-^19} | {:-^10} | {:-^18} |", "", "", "", "");
    println!(
        "| {:^18} | {:^19} | {:^10} | {:^18} |",
        "model".bold(),
        "author".bold(),
        "size (GB)".bold(),
        "license".bold()
    );
    println!("| {:-^18} | {:-^19} | {:-^10} | {:-^18} |", "", "", "", "");

    for weights in Weights::iter() {
        println!(
            "| {:^18} | {:^19} | {:^10} | {:^18} |",
            weights.model_name().replace(".safetensors", ""),
            weights.data_authors(),
            weights.data_size(),
            weights.license(),
        );
    }

    println!("| {:-^18} | {:-^19} | {:-^10} | {:-^18} |", "", "", "", "");
    println!("{:^69}", "\n");

    std::process::exit(1);
}
