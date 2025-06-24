// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::path::Path;
use std::path::PathBuf;
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
pub struct ProfileBoxesArgs {
    #[arg(short = 'i', long, help = "Image directory.", required = true)]
    pub images: Option<String>,

    #[arg(short = 's', long, help = "Bounding boxes directory.")]
    pub boxes: Option<String>,

    #[arg(
        short = 'o',
        long,
        help = "Output directory or file (.csv, .txt, .tsv, .pq).",
        required = true
    )]
    pub output: Option<String>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(short = 'd', long, help = "Exclude objects touching edge of image.")]
    pub drop_borders: bool,

    #[arg(
        long,
        short = 'm',
        help = "Mode. Compute descriptors across one or more features including c (complete pixels) and/or x (bounding boxes).",
        default_value = "cx"
    )]
    pub mode: Option<String>,

    #[arg(
        short = 'p',
        long,
        help = "Add padding around extracted objects before computing profiles.",
        default_value = "1"
    )]
    pub pad: Option<u32>,

    #[arg(long, help = "Substring specifying images (e.g. _image).")]
    pub image_substring: Option<String>,

    #[arg(long, help = "Substring specifying bounding boxes (e.g. _boxes).")]
    pub box_substring: Option<String>,

    #[arg(
        long,
        help = "Exclude objects smaller than a minimum size.",
        default_value = "1"
    )]
    pub min_size: Option<u32>,

    #[arg(short = 't', long, help = "Number of threads.")]
    pub threads: Option<usize>,
}

pub fn profile_image_boxes(args: &ProfileBoxesArgs) {
    if let Some(threads) = args.threads.to_owned() {
        if threads < 1 {
            println!(
                "[thyme::profile::boxes] Threads must be set to a positive integer if provided."
            );
            std::process::exit(1);
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let mode = args.mode.to_owned().unwrap_or("cmbfpx".to_string());
    let pad = args.pad.unwrap_or(1);
    let min_size = args.min_size.unwrap_or(1);

    if mode.chars().any(|c| !matches!(c, 'c' | 'x')) {
        eprintln!(
            "[thyme::profile::boxes] Invalid mode. Argument mode must only contain one or more of: c, x."
        );
        std::process::exit(1);
    }

    if min_size < 1 {
        eprintln!("[thyme::profile::boxes] ERROR: min_size cannot be less than 1.0.");
        std::process::exit(1);
    }

    let image_path = args.images.to_owned().unwrap();
    let boxes_path = args.boxes.to_owned().unwrap_or(image_path.clone());

    if image_path == boxes_path && args.image_substring == args.box_substring {
        eprintln!(
            "[thyme::profile::boxes] ERROR: If images and boxes are located in same path, different image and bounding box substrings must be provided."
        );
        std::process::exit(1);
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

    let boxes_files = ut::path::collect_file_paths(
        &boxes_path,
        constant::SUPPORTED_ARRAY_FORMATS.as_slice(),
        args.box_substring.to_owned(),
    )
    .unwrap_or_else(|err| {
        eprintln!("{}", err);
        std::process::exit(1);
    });

    if image_files.is_empty() {
        eprintln!(
            "[thyme::profile::boxes] ERROR: No image files were detected. Please check your path and/or substring identifier."
        );
        std::process::exit(1);
    }

    if boxes_files.is_empty() {
        eprintln!(
            "[thyme::profile::boxes] ERROR: No bounding box files were detected. Please check your path and/or substring identifier."
        );
        std::process::exit(1);
    }

    let mut pairs = ut::path::collect_file_pairs(
        &image_files,
        &boxes_files,
        args.image_substring.to_owned(),
        args.box_substring.to_owned(),
    );

    pairs.sort_unstable();

    ut::track::progress_log(
        &format!(
            "Detected {} image and bounding box pairs.",
            ut::track::thousands_format(pairs.len())
        ),
        args.verbose,
    );

    let mut output = PathBuf::from(args.output.to_owned().unwrap());

    let extension = output
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    if let Some(ext) = extension {
        if !["csv", "txt", "tsv", "pq"].iter().any(|e| e == &ext) {
            eprintln!(
                "[thyme::profile::boxes] ERROR: Invalid file extension. Must end with one of .csv, .txt, .tsv, .pq."
            );
            std::process::exit(1);
        }

        if let Some(parent) = output.parent() {
            if !parent.is_dir() && parent.to_str().unwrap() != "" {
                eprintln!(
                    "[thyme::profile::boxes] ERROR: Invalid file path. Parent directory of output file path does not exist."
                );
                std::process::exit(1);
            }
        }
    } else {
        output = ut::path::create_directory(&output).unwrap_or_else(|_| {
            eprintln!("[thyme::profile::boxes] ERROR: Could not create directory.");
            std::process::exit(1);
        });
    }

    let pb = ut::track::progress_bar(pairs.len(), "Profiling", args.verbose);

    let objects: Mutex<usize> = Mutex::new(0);
    let success: Mutex<Vec<String>> = Mutex::new(vec![]);
    let failure: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(pairs.len()));

    let name: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(pairs.len()));
    let item: Mutex<Vec<u32>> = Mutex::new(Vec::with_capacity(pairs.len()));
    let data: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::with_capacity(300 * pairs.len()));

    (0..pairs.len())
        .into_par_iter()
        .tqdm_with_bar(pb)
        .for_each(|idx| {
            let (id, image, boxes) = &pairs[idx];
            let run = profile(image, boxes, pad, args.drop_borders, min_size, &mode);

            if let Ok((ids, descriptors)) = run {
                let n = ids.len();

                success.lock().unwrap().push(format!("{}\t{}", id, n));

                let image = image.file_stem().unwrap().to_string_lossy().to_string();

                name.lock().unwrap().extend((0..n).map(|_| image.clone()));
                item.lock().unwrap().extend(ids);
                data.lock().unwrap().extend(descriptors);
                *objects.lock().unwrap() += n;
            } else {
                failure
                    .lock()
                    .unwrap()
                    .push(format!("{}\t{}", id, run.unwrap_err()));
            }
        });

    let objects = objects.into_inner().unwrap();
    let success = success.into_inner().unwrap();
    let failure = failure.into_inner().unwrap();

    let name = name.into_inner().unwrap();
    let item = item.into_inner().unwrap();
    let data = data.into_inner().unwrap();

    if args.verbose {
        println!();
    }

    ut::track::progress_log(
        &format!(
            "Complete. {} profiles computed across {} images.",
            ut::track::thousands_format(objects),
            ut::track::thousands_format(success.len())
        ),
        args.verbose,
    );

    if !success.is_empty() {
        let columns = descriptor_columns(&mode);

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

        for row in &data {
            for (idx, &descriptor) in row.iter().enumerate() {
                column_data[idx].push(descriptor);
            }
        }

        for (column, descriptor) in columns.iter().zip(column_data) {
            df.with_column(Column::new(column.into(), descriptor))
                .unwrap();
        }

        let descriptors_path = if output.is_dir() {
            output.join("descriptors.csv")
        } else {
            output.clone()
        };

        io::write_table(&mut df, descriptors_path).unwrap_or_else(|_| {
            eprintln!("[thyme::profile::boxes] ERROR: Failed to write descriptors table.");
            std::process::exit(1);
        });
    }

    if output.is_dir() {
        if !success.is_empty() {
            std::fs::write(output.join("object_counts.tsv"), success.join("\n")).unwrap();
        }

        if !failure.is_empty() {
            std::fs::write(output.join("object_errors.tsv"), failure.join("\n")).unwrap();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn profile(
    image_path: &Path,
    boxes_path: &Path,
    pad: u32,
    drop_borders: bool,
    min_size: u32,
    mode: &str,
) -> Result<(Vec<u32>, Vec<Vec<f32>>), ThymeError> {
    let image = im::ThymeImage::open(image_path)?;

    let bounding_boxes = im::BoundingBoxes::open(boxes_path)?;

    let width = image.width();
    let height = image.height();

    let pad_f32 = pad as f32;

    let mut ids: Vec<u32> = Vec::with_capacity(bounding_boxes.len());
    let mut results: Vec<Vec<f32>> = Vec::with_capacity(50 * bounding_boxes.len());

    for (idx, [min_x, min_y, max_x, max_y]) in bounding_boxes.as_xyxy().iter().enumerate() {
        let min_x = min_x - pad_f32;
        let min_y = min_y - pad_f32;
        let max_x = max_x + pad_f32;
        let max_y = max_y + pad_f32;

        if drop_borders
            && (min_x <= 0.0 || min_y <= 0.0 || max_x >= width as f32 || max_y >= height as f32)
        {
            continue;
        }

        let min_x = min_x.max(0.0) as u32;
        let min_y = min_y.max(0.0) as u32;
        let max_x = max_x.min(width as f32) as u32;
        let max_y = max_y.min(height as f32) as u32;

        let w = max_x - min_x;
        let h = max_y - min_y;

        if w < min_size || h < min_size {
            continue;
        }

        let mut result: Vec<f32> = Vec::with_capacity(100);

        if mode.contains("x") {
            result.extend([w as f32, h as f32, (w * h) as f32]);
        }

        if mode.contains("c") {
            result.extend(image.crop_view(min_x, min_y, w, h).descriptors());
        }

        ids.push(idx as u32);
        results.push(result)
    }

    Ok((ids, results))
}

/// Generate the column names for the output descriptor table
///
/// # Arguments
///
/// * `mode` - Profiling mode
fn descriptor_columns(mode: &str) -> Vec<String> {
    let mut names: Vec<String> = vec![];

    if mode.contains("x") {
        names.extend([
            "bbox_width".to_string(),
            "bbox_height".to_string(),
            "bbox_area".to_string(),
        ]);
    }

    let suffixes: Vec<&str> = constant::INTENSITY_DESCRIPTOR_NAMES
        .into_iter()
        .chain(constant::MOMENTS_DESCRIPTOR_NAMES)
        .chain(constant::TEXTURE_DESCRIPTOR_NAMES)
        .chain(constant::ZERNIKE_DESCRIPTOR_NAMES)
        .collect();

    if mode.contains("c") {
        names.extend(suffixes.iter().map(|s| "complete_".to_string() + s));
    }

    names
}
