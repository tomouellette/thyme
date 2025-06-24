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
pub struct FormArgs {
    #[arg(
        short = 'i',
        long,
        help = "Polygons or polygons directory.",
        required = true
    )]
    pub polygons: Option<String>,

    #[arg(
        short = 'o',
        long,
        help = "Output directory or file (.csv, .txt, .tsv, .pq)."
    )]
    pub output: Option<String>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(long, help = "Substring specifying polygons (e.g. _polygons).")]
    pub polygon_substring: Option<String>,

    #[arg(short = 't', long, help = "Number of threads.")]
    pub threads: Option<usize>,
}

pub fn measure_form(args: &FormArgs) {
    if let Some(threads) = args.threads.to_owned() {
        if threads < 1 {
            println!(
                "[thyme::measure::form] ERROR: Threads must be set to a positive integer if provided."
            );
            std::process::exit(1);
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let polygons_path = args.polygons.to_owned().unwrap();

    let polygon_extension = Path::new(&polygons_path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    let is_polygon_dir = if let Some(ext) = &polygon_extension {
        if !constant::SUPPORTED_ARRAY_FORMATS.contains(&ext.as_str()) {
            eprintln!(
                "[thyme::measure::form] ERROR: Invalid polygon extension {}. Must be one of: {:?}.",
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
        if !is_polygon_dir {
            eprintln!(
                "[thyme::measure::form] ERROR: If output is provided, then input polygons path must specify a polygons directory."
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
                    "[thyme::measure::form] ERROR: Invalid file extension. Must end with one of .csv, .txt, .tsv, .pq."
                );
                std::process::exit(1);
            }
        } else {
            eprintln!(
                "[thyme::measure::form] ERROR: Invalid output path. Output file must be a file with a valid extension."
            );
            std::process::exit(1);
        }

        if let Some(parent) = output.parent() {
            if !parent.is_dir() && parent.to_str().unwrap() != "" {
                eprintln!(
                    "[thyme::measure::form] ERROR: Invalid file path. Parent directory of output file path does not exist."
                );
                std::process::exit(1);
            }
        }

        let polygon_files = ut::path::collect_file_paths(
            &polygons_path,
            constant::SUPPORTED_ARRAY_FORMATS.as_slice(),
            args.polygon_substring.to_owned(),
        )
        .unwrap_or_else(|err| {
            eprintln!("{}", err);
            std::process::exit(1);
        });

        if polygon_files.is_empty() {
            eprintln!(
                "[thyme::measure::form] ERROR: No polygon files were detected. Please check your path and/or substring identifier."
            );
            std::process::exit(1);
        }

        ut::track::progress_log(
            &format!(
                "Detected {} polygons.",
                ut::track::thousands_format(polygon_files.len())
            ),
            args.verbose,
        );

        let pb = ut::track::progress_bar(polygon_files.len(), "Measuring form", args.verbose);

        let failure: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(polygon_files.len()));
        let name: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(polygon_files.len()));
        let item: Mutex<Vec<u32>> = Mutex::new(Vec::with_capacity(polygon_files.len()));
        let data: Mutex<Vec<[f32; 23]>> = Mutex::new(Vec::with_capacity(23 * polygon_files.len()));

        (0..polygon_files.len())
            .into_par_iter()
            .tqdm_with_bar(pb)
            .for_each(|idx| {
                let result = form(&polygon_files[idx]);

                let polygon_name = polygon_files[idx]
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();

                if let Ok(descriptors) = result {
                    let n = descriptors.len();

                    name.lock()
                        .unwrap()
                        .extend((0..n).map(|_| polygon_name.clone()));

                    item.lock()
                        .unwrap()
                        .extend((0..n as u32).collect::<Vec<u32>>());

                    data.lock().unwrap().extend(descriptors);
                } else {
                    failure.lock().unwrap().push(format!(
                        "{}\t{}",
                        polygon_name,
                        result.unwrap_err()
                    ));
                }
            });

        let failure = failure.into_inner().unwrap();
        let name = name.into_inner().unwrap();
        let item = item.into_inner().unwrap();
        let data = data.into_inner().unwrap();

        if args.verbose {
            println!()
        }

        if !data.is_empty() {
            write_form(&data, &name, &item, &output);
        }

        let message = if !failure.is_empty() {
            &format!(
                "Complete. {} images measured succesfully. {} images failed.",
                ut::track::thousands_format(polygon_files.len() - failure.len()),
                ut::track::thousands_format(failure.len())
            )
        } else {
            &format!(
                "Complete. {} images measured successfully.",
                ut::track::thousands_format(polygon_files.len() - failure.len()),
            )
        };

        ut::track::progress_log(message, args.verbose);
    } else {
        if is_polygon_dir {
            eprintln!(
                "[thyme::measure::form] ERROR: If output is not provided, then input polygon path should specify a single file."
            );
            std::process::exit(1);
        }

        let polygons_path = Path::new(&polygons_path);

        if !polygons_path.is_file() {
            eprintln!(
                "[thyme::measure::form] ERROR: The provided polygon file path does not exist."
            );
            std::process::exit(1);
        }

        let data = form(Path::new(&polygons_path)).unwrap_or_else(|_| {
            eprintln!("[thyme::measure::form] ERROR: Failed to measure form descriptors.");
            std::process::exit(1);
        });

        let mut stdout = std::io::stdout();

        for (i, d) in data.iter().enumerate() {
            let output: Vec<String> = constant::FORM_DESCRIPTOR_NAMES
                .iter()
                .copied()
                .zip(d.iter().map(|x| x.to_string()).collect::<Vec<String>>())
                .map(|(c, d)| format!("object_{}\t{}\t{}\n", i, c, d))
                .collect();

            for row in output.iter() {
                stdout.write_all(row.as_bytes()).unwrap();
            }
        }
    }
}

/// Measure form descriptors across a set of polygons
fn form(polygons_path: &Path) -> Result<Vec<[f32; 23]>, ThymeError> {
    let mut polygons = im::Polygons::open(polygons_path)?;
    Ok(polygons.descriptors())
}

/// Write form descriptors to data table
fn write_form(data: &[[f32; 23]], name: &Vec<String>, item: &Vec<u32>, output: &Path) {
    let columns = constant::FORM_DESCRIPTOR_NAMES;

    let mut df = DataFrame::new(vec![
        Column::new("image".into(), &name),
        Column::new("object".into(), &item),
    ])
    .unwrap();

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
        eprintln!("[thyme::measure::form] ERROR: Failed to write descriptors table.");
        std::process::exit(1);
    });
}
