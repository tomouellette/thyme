// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use clap::Args;
use kdam::TqdmParallelIterator;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use thyme_core::constant;
use thyme_core::error::ThymeError;
use thyme_core::im;
use thyme_core::ut;

#[derive(Debug, Args)]
pub struct Mask2polygonsArgs {
    #[arg(short = 'i', long, help = "Mask or mask directory.", required = true)]
    pub mask: Option<String>,

    #[arg(short = 'o', long, help = "Output polygons file.", required = true)]
    pub output: Option<String>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(long, help = "Substring specifying masks (e.g. _mask).")]
    pub mask_substring: Option<String>,

    #[arg(short = 't', long, help = "Number of threads.")]
    pub threads: Option<usize>,
}

pub fn utils_mask2polygons(args: &Mask2polygonsArgs) {
    if let Some(threads) = args.threads.to_owned() {
        if threads < 1 {
            println!(
                "[thyme::utils::mask2polygons] ERROR: Threads must be set to a positive integer if provided."
            );
            std::process::exit(1);
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let mask_path = args.mask.to_owned().unwrap();

    let mut output = PathBuf::from(args.output.to_owned().unwrap());

    let mask_extension = Path::new(&mask_path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    let output_extension = output
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    let is_mask_dir = if let Some(ext) = mask_extension {
        if !constant::SUPPORTED_IMAGE_FORMATS.contains(&ext.as_str()) {
            eprintln!(
                "[thyme::utils::mask2polygons] ERROR: Invalid mask extension {}. Must be one of: {:?}.",
                ext,
                constant::SUPPORTED_IMAGE_FORMATS
            );
            std::process::exit(1);
        }
        false
    } else {
        true
    };

    if let Some(ext) = output_extension {
        if is_mask_dir {
            eprintln!(
                "[thyme::utils::mask2polygons] ERROR: If mask input is a directory then output must be a directory."
            );
            std::process::exit(1);
        }

        if !["json"].iter().any(|e| e == &ext) {
            eprintln!(
                "[thyme::utils::mask2polygons] ERROR: Invalid file extension. Must end with .json."
            );
            std::process::exit(1);
        }

        if let Some(parent) = output.parent() {
            if !parent.is_dir() && parent.to_str().unwrap() != "" {
                eprintln!(
                    "[thyme::utils::mask2polygons] ERROR: Invalid file path. Parent directory of output file path does not exist."
                );
                std::process::exit(1);
            }
        }

        mask2polygons(Path::new(&mask_path), &output, false).unwrap_or_else(|_| {
            eprintln!("[thyme::utils::mask2polygons] ERROR: Failed to convert mask to polygons.");
            std::process::exit(1);
        });
    } else {
        if !is_mask_dir {
            eprintln!(
                "[thyme::utils::mask2polygons] ERROR: If output is a directory then mask input must be a directory."
            );
            std::process::exit(1);
        }

        let mask_files = ut::path::collect_file_paths(
            &mask_path,
            constant::SUPPORTED_IMAGE_FORMATS.as_slice(),
            args.mask_substring.to_owned(),
        )
        .unwrap_or_else(|err| {
            eprintln!("{}", err);
            std::process::exit(1);
        });

        if mask_files.is_empty() {
            eprintln!(
                "[thyme::utils::mask2polygons] ERROR: No mask files were detected. Please check your path and/or substring identifier."
            );
            std::process::exit(1);
        }

        ut::track::progress_log(
            &format!(
                "Detected {} masks.",
                ut::track::thousands_format(mask_files.len())
            ),
            args.verbose,
        );

        output = ut::path::create_directory(&output).unwrap_or_else(|_| {
            eprintln!("[thyme::utils::mask2polygons] ERROR: Could not create directory.");
            std::process::exit(1);
        });

        let pb = ut::track::progress_bar(
            mask_files.len(),
            "Converting masks to polygons",
            args.verbose,
        );

        let error: Mutex<Vec<usize>> = Mutex::new(Vec::with_capacity(mask_files.len()));

        (0..mask_files.len())
            .into_par_iter()
            .tqdm_with_bar(pb)
            .for_each(|idx| {
                mask2polygons(&mask_files[idx], &output, true).unwrap_or_else(|_| {
                    error.lock().unwrap().push(idx);
                });
            });

        let error = error.into_inner().unwrap();

        if args.verbose {
            println!()
        }

        let message = if !error.is_empty() {
            &format!(
                "Complete. {} images succesfully converted to polygons. {} images failed.",
                ut::track::thousands_format(mask_files.len() - error.len()),
                ut::track::thousands_format(error.len())
            )
        } else {
            &format!(
                "Complete. {} images converted to polygons.",
                ut::track::thousands_format(mask_files.len() - error.len()),
            )
        };

        ut::track::progress_log(message, args.verbose);
    }
}

/// Convert an input mask to polygons
fn mask2polygons(mask_path: &Path, output_path: &Path, is_dir: bool) -> Result<(), ThymeError> {
    let mut mask = im::ThymeMask::open(mask_path)?;

    let (_, polygons) = mask.polygons()?;

    if is_dir {
        polygons.save(
            output_path
                .join(mask_path.file_stem().unwrap())
                .with_extension("json"),
        )?;
    } else {
        polygons.save(output_path)?;
    }

    Ok(())
}
