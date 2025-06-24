// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use clap::Args;
use kdam::TqdmParallelIterator;
use polars::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use thyme_core::constant;
use thyme_core::error::ThymeError;
use thyme_core::im;
use thyme_core::io;
use thyme_core::ut;

#[derive(Debug, Args)]
pub struct ZernikeArgs {
    #[arg(short = 'i', long, help = "Image or image directory.", required = true)]
    pub images: Option<String>,

    #[arg(
        short = 'o',
        long,
        help = "Output directory or file (.csv, .txt, .tsv, .pq)."
    )]
    pub output: Option<String>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(long, help = "Substring specifying images (e.g. _image).")]
    pub image_substring: Option<String>,

    #[arg(short = 't', long, help = "Number of threads.")]
    pub threads: Option<usize>,
}

pub fn measure_zernike(args: &ZernikeArgs) {
    if let Some(threads) = args.threads.to_owned() {
        if threads < 1 {
            println!(
                "[thyme::measure::zernike] ERROR: Threads must be set to a positive integer if provided."
            );
            std::process::exit(1);
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let image_path = args.images.to_owned().unwrap();

    let image_extension = Path::new(&image_path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    let is_image_dir = if let Some(ext) = &image_extension {
        if !constant::SUPPORTED_IMAGE_FORMATS.contains(&ext.as_str()) {
            eprintln!(
                "[thyme::measure::zernike] ERROR: Invalid image extension {}. Must be one of: {:?}.",
                ext,
                constant::SUPPORTED_IMAGE_FORMATS
            );
            std::process::exit(1);
        }
        false
    } else {
        true
    };

    if let Some(output) = args.output.to_owned() {
        if !is_image_dir {
            eprintln!(
                "[thyme::measure::zernike] ERROR: If output is provided, then input image path must specify an image directory."
            );
            std::process::exit(1);
        }

        let output = PathBuf::from(output);

        let extension = output
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = &extension {
            if !["csv", "txt", "tsv", "pq"].iter().any(|e| e == ext) {
                eprintln!(
                    "[thyme::measure::zernike] ERROR: Invalid file extension. Must end with one of .csv, .txt, .tsv, .pq."
                );
                std::process::exit(1);
            }
        } else {
            eprintln!(
                "[thyme::measure::zernike] ERROR: Invalid output path. Output file must be a file with a valid extension."
            );
            std::process::exit(1);
        }

        if let Some(parent) = output.parent() {
            if !parent.is_dir() && parent.to_str().unwrap() != "" {
                eprintln!(
                    "[thyme::measure::zernike] ERROR: Invalid file path. Parent directory of output file path does not exist."
                );
                std::process::exit(1);
            }
        }

        let image_files = ut::path::collect_file_paths(
            &image_path,
            constant::SUPPORTED_IMAGE_FORMATS.as_slice(),
            args.image_substring.to_owned(),
        )
        .unwrap_or_else(|err| {
            eprintln!("{}", err);
            std::process::exit(1);
        });

        if image_files.is_empty() {
            eprintln!(
                "[thyme::measure::zernike] ERROR: No image files were detected. Please check your path and/or substring identifier."
            );
            std::process::exit(1);
        }

        ut::track::progress_log(
            &format!(
                "Detected {} images.",
                ut::track::thousands_format(image_files.len())
            ),
            args.verbose,
        );

        let pb = ut::track::progress_bar(image_files.len(), "Measuring zernike", args.verbose);

        let failure: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(image_files.len()));
        let name: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(image_files.len()));
        let data: Mutex<Vec<[f32; 30]>> = Mutex::new(Vec::with_capacity(30 * image_files.len()));

        (0..image_files.len())
            .into_par_iter()
            .tqdm_with_bar(pb)
            .for_each(|idx| {
                let result = zernike(&image_files[idx]);

                let image_name = image_files[idx]
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();

                if let Ok(descriptors) = result {
                    name.lock().unwrap().push(image_name);
                    data.lock().unwrap().push(descriptors);
                } else {
                    failure.lock().unwrap().push(format!(
                        "{}\t{}",
                        image_name,
                        result.unwrap_err()
                    ));
                }
            });

        let failure = failure.into_inner().unwrap();
        let name = name.into_inner().unwrap();
        let data = data.into_inner().unwrap();

        if args.verbose {
            println!()
        }

        if !data.is_empty() {
            write_zernike(&data, &name, &output);
        }

        let message = if !failure.is_empty() {
            &format!(
                "Complete. {} images measured succesfully. {} images failed.",
                ut::track::thousands_format(image_files.len() - failure.len()),
                ut::track::thousands_format(failure.len())
            )
        } else {
            &format!(
                "Complete. {} images measured successfully.",
                ut::track::thousands_format(image_files.len() - failure.len()),
            )
        };

        ut::track::progress_log(message, args.verbose);
    } else {
        if is_image_dir {
            eprintln!(
                "[thyme::measure::zernike] ERROR: If output is not provided, then input image path should specify a single file."
            );
            std::process::exit(1);
        }

        let image_path = Path::new(&image_path);

        if !image_path.is_file() {
            eprintln!(
                "[thyme::measure::zernike] ERROR: The provided image file path does not exist."
            );
            std::process::exit(1);
        }

        let data = zernike(Path::new(&image_path)).unwrap_or_else(|_| {
            eprintln!("[thyme::measure::zernike] ERROR: Failed to measure zernike descriptors.");
            std::process::exit(1);
        });

        let output: Vec<String> = constant::ZERNIKE_DESCRIPTOR_NAMES
            .iter()
            .copied()
            .zip(data.iter().map(|x| x.to_string()).collect::<Vec<String>>())
            .map(|(c, d)| format!("{}\t{}\n", c, d))
            .collect();

        let mut stdout = std::io::stdout();

        for row in output.iter() {
            stdout.write_all(row.as_bytes()).unwrap();
        }
    }
}

/// Measure zernike descriptors across an image
fn zernike(image_path: &Path) -> Result<[f32; 30], ThymeError> {
    let image = im::ThymeImage::open(image_path)?;

    Ok(image
        .crop_view(0, 0, image.width(), image.height())
        .zernike())
}

/// Write zernike descriptors to data table
fn write_zernike(data: &[[f32; 30]], name: &Vec<String>, output: &Path) {
    let columns = constant::ZERNIKE_DESCRIPTOR_NAMES;

    let mut df = DataFrame::new(vec![Column::new("image".into(), &name)]).unwrap();

    // Note that this requires generating two copies of the computed descriptors
    // which is definitely not ideal. We probaby want to redesign the computation
    // so that column-major data is generated directly or we just use a flat buffer
    // and then just handle the saving with indexing. Also look into the polars API.
    let mut column_data: Vec<Vec<f32>> = vec![Vec::with_capacity(data.len()); data[0].len()];

    for row in data {
        for (idx, &descriptor) in row.iter().enumerate() {
            column_data[idx].push(descriptor);
        }
    }

    for (column, descriptor) in columns.iter().zip(column_data) {
        df.with_column(Column::new(column.to_string().into(), descriptor))
            .unwrap();
    }

    let descriptors_path = if output.is_dir() {
        output.join("descriptors.csv")
    } else {
        output.to_path_buf()
    };

    io::write_table(&mut df, descriptors_path).unwrap_or_else(|_| {
        eprintln!("[thyme::measure::zernike] ERROR: Failed to write descriptors table.");
        std::process::exit(1);
    });
}
