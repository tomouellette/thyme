// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use clap::Args;
use colored::Colorize;

use thyme_core::ut::track::progress_log;
use thyme_data::data::SegmentationDatasets;

#[derive(Debug, Args)]
#[command(about = "Download standardized bioimaging segmentation datasets.")]
pub struct DownloadSegmentationArgs {
    #[arg(
        short,
        long,
        help = "Dataset name. Run thyme download segmentation --list` to see all available datasets. Run `thyme download segementation --all` to download all segmentation datasets"
    )]
    pub name: Option<String>,

    #[arg(short, long, help = "Output path to save downloaded dataset.")]
    pub output: Option<String>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(long, help = "List all available segmentation datasets.")]
    pub list: bool,

    #[arg(long, help = "Download all available segmentation datasets.")]
    pub all: bool,
}

pub fn download_segmentation(args: &DownloadSegmentationArgs) {
    if args.list {
        print_segmentation();
    }

    if args.all {
        if args.output.is_none() {
            eprintln!(
                "[thyme::download::segmentation] Please specify the output path (--output/-o) to save all downloaded segmentation datasets."
            );
            std::process::exit(1);
        }

        let output = std::path::Path::new(args.output.as_ref().unwrap());
        std::fs::create_dir_all(output).unwrap();

        progress_log("Downloading all segmentation datasets", args.verbose);

        for dataset in SegmentationDatasets::iter() {
            dataset.download(output, args.verbose);
        }

        std::process::exit(1);
    }

    if !(args.name.is_some() && args.output.is_some()) {
        eprintln!(
            "[thyme::download::segmentation] Both --name/-n and --output/-o must be specified."
        );
        std::process::exit(1);
    }

    let output = std::path::Path::new(args.output.as_ref().unwrap());
    let segmentation = SegmentationDatasets::select(args.name.as_ref().unwrap());
    segmentation.download(output, args.verbose);
}

fn print_segmentation() {
    println!("{:^72}", "\n");
    println!("| {:-^72} |", "");
    println!("| {:^72} |", "thyme".truecolor(103, 194, 69).bold());
    println!("| {:^72} |", "Segmentation datasets");
    println!("| {:-^72} |", "");
    // println!("| {:-^14} | {:-^21} | {:-^10} | {:-^18} |", "", "", "", "");
    println!(
        "| {:^14} | {:^21} | {:^10} | {:^18} |",
        "segmentation".bold(),
        "author".bold(),
        "size (GB)".bold(),
        "license".bold()
    );
    println!("| {:-^14} | {:-^21} | {:-^10} | {:-^18} |", "", "", "", "");

    for segmentation in SegmentationDatasets::iter() {
        println!(
            "| {:^14} | {:^21} | {:^10} | {:^18} |",
            segmentation.name(),
            segmentation.data_authors(),
            segmentation.data_size(),
            segmentation.license(),
        );
    }

    println!("| {:-^14} | {:-^21} | {:-^10} | {:-^18} |", "", "", "", "");
    println!("{:^72}", "\n");

    std::process::exit(1);
}
