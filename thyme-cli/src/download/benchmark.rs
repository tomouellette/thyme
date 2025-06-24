// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use clap::Args;
use colored::Colorize;

use thyme_core::ut::track::progress_log;
use thyme_data::data::BenchmarkDatasets;

#[derive(Debug, Args)]
#[command(about = "Download standardized single-object benchmark datasets.")]
pub struct DownloadBenchmarkArgs {
    #[arg(
        short,
        long,
        help = "Dataset name. Run thyme download benchmark --list` to see all available benchmarks. Run `thyme download benchmark --all` to download all benchmark datasets"
    )]
    pub name: Option<String>,

    #[arg(short, long, help = "Output path to save downloaded dataset.")]
    pub output: Option<String>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(long, help = "List all available benchmark datasets.")]
    pub list: bool,

    #[arg(long, help = "Download all available benchmark datasets.")]
    pub all: bool,
}

pub fn download_benchmark(args: &DownloadBenchmarkArgs) {
    if args.list {
        print_benchmark();
    }

    if args.all {
        if args.output.is_none() {
            eprintln!(
                "[thyme::download::benchmark] Please specify the output path (--output/-o) to save all downloaded benchmark datasets."
            );
            std::process::exit(1);
        }

        let output = std::path::Path::new(args.output.as_ref().unwrap());
        std::fs::create_dir_all(output).unwrap();

        progress_log("Downloading all benchmark datasets", args.verbose);

        for dataset in BenchmarkDatasets::iter() {
            dataset.download(output, args.verbose);
        }

        std::process::exit(1);
    }

    if !(args.name.is_some() && args.output.is_some()) {
        eprintln!("[thyme::download::benchmark] Both --name/-n and --output/-o must be specified.");
        std::process::exit(1);
    }

    let output = std::path::Path::new(args.output.as_ref().unwrap());
    let benchmark = BenchmarkDatasets::select(args.name.as_ref().unwrap());
    benchmark.download(output, args.verbose);
}

fn print_benchmark() {
    println!("{:^71}", "\n");
    println!("| {:-^71} |", "");
    println!("| {:^71} |", "thyme".truecolor(103, 194, 69).bold());
    println!("| {:^71} |", "Single-object image benchmark datasets");
    println!("| {:-^71} |", "");
    // println!("| {:-^16} | {:-^19} | {:-^10} | {:-^17} |", "", "", "", "");
    println!(
        "| {:^16} | {:^19} | {:^10} | {:^17} |",
        "dataset".bold(),
        "author".bold(),
        "size (GB)".bold(),
        "license".bold()
    );
    println!("| {:-^16} | {:-^19} | {:-^10} | {:-^17} |", "", "", "", "");

    for dataset in BenchmarkDatasets::iter() {
        println!(
            "| {:^16} | {:^19} | {:^10} | {:^17} |",
            dataset.name(),
            dataset.data_authors(),
            dataset.data_size(),
            dataset.license(),
        );
    }

    println!("| {:-^16} | {:-^19} | {:-^10} | {:-^17} |", "", "", "", "");
    println!("{:^71}", "\n");

    std::process::exit(1);
}
