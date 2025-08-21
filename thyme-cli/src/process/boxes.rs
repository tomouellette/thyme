// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::Args;
use futures::stream::{self, StreamExt};
use kdam::BarExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use thyme_core::constant;
use thyme_core::error::ThymeError;
use thyme_core::im;
use thyme_core::ut;

#[derive(Debug, Args)]
pub struct ProcessBoxesArgs {
    #[arg(short = 'i', long, help = "Image directory.", required = true)]
    pub images: Option<String>,

    #[arg(short = 's', long, help = "Bounding boxes directory.")]
    pub boxes: Option<String>,

    #[arg(short = 'o', long, help = "Output directory.", required = true)]
    pub output: Option<String>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(short = 'd', long, help = "Exclude objects touching edge of image.")]
    pub drop_borders: bool,

    #[arg(
        long,
        short = 'm',
        help = "Mode. One or more of c (complete pixels) and/or x (bounding boxes).",
        default_value = "cx"
    )]
    pub mode: Option<String>,

    #[arg(
        short = 'p',
        long,
        help = "Add padding around extracted objects",
        default_value = "1"
    )]
    pub pad: Option<u32>,

    #[arg(long, help = "Substring specifying images (e.g. _image).")]
    pub image_substring: Option<String>,

    #[arg(long, help = "Substring specifying boxes (e.g. _boxes).")]
    pub box_substring: Option<String>,

    #[arg(
        long,
        help = "Exclude objects smaller than a minimum size.",
        default_value = "1"
    )]
    pub min_size: Option<u32>,

    #[arg(
        short = 'e',
        long,
        help = "Format to save extracted object images (e.g. png, jpeg, npy).",
        default_value = "png"
    )]
    pub image_format: Option<String>,

    #[arg(
        short = 'a',
        long,
        help = "Format to save extracted polygons and/or bounding boxes (e.g. json).",
        default_value = "json"
    )]
    pub array_format: Option<String>,

    #[arg(short = 't', long, help = "Number of threads.")]
    pub threads: Option<usize>,
}

pub fn process_image_boxes(args: &ProcessBoxesArgs) {
    let mode = args.mode.to_owned().unwrap_or("cmbfpx".to_string());
    let pad = args.pad.unwrap_or(1);
    let min_size = args.min_size.unwrap_or(1);
    let image_format = args.image_format.to_owned().unwrap_or("png".to_string());
    let array_format = args.array_format.to_owned().unwrap_or("json".to_string());

    let threads = if let Some(t) = args.threads {
        t
    } else {
        std::thread::available_parallelism().unwrap_or_else(|_| {
            eprintln!("[thyme::process::boxes] Could not automatically assign number of tasks. Please manually set the --threads (-t) argument.");
            std::process::exit(1);
        }).get()
    };

    if mode.chars().any(|c| !matches!(c, 'c' | 'x')) {
        eprintln!(
            "[thyme::process::boxes] Invalid mode. Argument mode must only contain one or more of: c, x."
        );
        std::process::exit(1);
    }

    if min_size < 1 {
        eprintln!("[thyme::process::boxes] ERROR: min_size cannot be less than 1.0.");
        std::process::exit(1);
    }

    if !constant::SUPPORTED_IMAGE_FORMATS.contains(&image_format.as_str()) {
        eprintln!(
            "[thyme::process::boxes] ERROR: Invalid image_format {}. Must be one of: {:?}.",
            image_format,
            constant::SUPPORTED_IMAGE_FORMATS
        );
        std::process::exit(1);
    }

    if !constant::SUPPORTED_ARRAY_FORMATS.contains(&array_format.as_str()) {
        eprintln!(
            "[thyme::process::boxes] ERROR: Invalid array_format {}. Must be one of: {:?}.",
            array_format,
            constant::SUPPORTED_ARRAY_FORMATS
        );
        std::process::exit(1);
    }

    let image_path = args.images.to_owned().unwrap();
    let boxes_path = args.boxes.to_owned().unwrap_or(image_path.clone());

    if image_path == boxes_path && args.image_substring == args.box_substring {
        eprintln!(
            "[thyme::process::boxes] ERROR: If images and boxes are located in same path, different image and bounding box substrings must be provided."
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
            "[thyme::process::boxes] ERROR: No image files were detected. Please check your path and/or substring identifier."
        );
        std::process::exit(1);
    }

    if boxes_files.is_empty() {
        eprintln!(
            "[thyme::process::boxes] ERROR: No bounding boxes files were detected. Please check your path and/or substring identifier."
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

    let output = PathBuf::from(args.output.to_owned().unwrap());

    let output = ut::path::create_directory(&output).unwrap_or_else(|_| {
        eprintln!("[thyme::process::boxes] ERROR: Could not create directory.");
        std::process::exit(1);
    });

    if mode.contains("c") {
        std::fs::create_dir(output.join("complete")).unwrap();
    }

    if mode.contains("x") {
        std::fs::create_dir(output.join("bounding_boxes")).unwrap();
    }

    let rt = tokio::runtime::Runtime::new().unwrap();

    let results = rt.block_on(run_all(
        pairs,
        pad,
        args.drop_borders,
        min_size,
        &mode,
        &output,
        &image_format,
        &array_format,
        threads,
        args.verbose,
    ));

    let objects: Mutex<usize> = Mutex::new(0);
    let success: Mutex<Vec<String>> = Mutex::new(vec![]);
    let failure: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(results.len()));

    results.into_par_iter().for_each(|(id, run)| {
        if let Ok(n_objects) = run {
            *objects.lock().unwrap() += n_objects as usize;
            success
                .lock()
                .unwrap()
                .push(format!("{}\t{}", id, n_objects));
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

    if args.verbose {
        println!();
    }

    ut::track::progress_log(
        &format!(
            "Complete. {} objects detected across {} images.",
            ut::track::thousands_format(objects),
            ut::track::thousands_format(success.len())
        ),
        args.verbose,
    );

    if !success.is_empty() {
        std::fs::write(output.join("object_counts.tsv"), success.join("\n")).unwrap();
    }

    if !failure.is_empty() {
        std::fs::write(output.join("object_errors.tsv"), failure.join("\n")).unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
fn extract(
    id: &str,
    image_path: &Path,
    boxes_path: &Path,
    pad: u32,
    drop_borders: bool,
    min_size: u32,
    mode: &str,
    output: &Path,
    image_format: &str,
    array_format: &str,
) -> Result<u32, ThymeError> {
    let image = im::ThymeImage::open(image_path)?;

    let mut bounding_boxes = im::BoundingBoxes::open(boxes_path)?;

    let width = image.width();
    let height = image.height();

    let mut n_objects = 0;
    let pad_f32 = pad as f32;

    let mut remove_indices: Vec<usize> = Vec::with_capacity(bounding_boxes.len());

    for (idx, [min_x, min_y, max_x, max_y]) in bounding_boxes.as_xyxy().iter().enumerate() {
        let min_x = min_x - pad_f32;
        let min_y = min_y - pad_f32;
        let max_x = max_x + pad_f32;
        let max_y = max_y + pad_f32;

        if drop_borders
            && (min_x <= 0.0 || min_y <= 0.0 || max_x >= width as f32 || max_y >= height as f32)
        {
            remove_indices.push(idx);
            continue;
        }

        let min_x = min_x.max(0.0) as u32;
        let min_y = min_y.max(0.0) as u32;
        let max_x = max_x.min(width as f32) as u32;
        let max_y = max_y.min(height as f32) as u32;

        let w = max_x - min_x;
        let h = max_y - min_y;

        if w < min_size || h < min_size {
            remove_indices.push(idx);
            continue;
        }

        let object_name = format!("{}_{}.{}", id, idx, image_format);

        if mode.contains("c") {
            image
                .crop(min_x, min_y, w, h)?
                .save(output.join("complete").join(&object_name))?;
        }

        n_objects += 1;
    }

    let object_name = format!("{}.{}", id, array_format);

    if mode.contains("x") {
        bounding_boxes.remove(&remove_indices);
        bounding_boxes.save(output.join("bounding_boxes").join(&object_name))?;
    }

    Ok(n_objects)
}

#[allow(clippy::too_many_arguments)]
pub async fn run_all(
    pairs: Vec<(String, PathBuf, PathBuf)>,
    pad: u32,
    drop_borders: bool,
    min_size: u32,
    mode: &str,
    output: &Path,
    image_format: &str,
    array_format: &str,
    threads: usize,
    verbose: bool,
) -> Vec<(String, Result<u32, ThymeError>)> {
    let pb = Arc::new(Mutex::new(ut::track::progress_bar(
        pairs.len(),
        "Processing",
        verbose,
    )));

    stream::iter(pairs)
        .map(|(id, image, bounding_boxes)| {
            let mode = mode.to_string();
            let output = output.to_path_buf();
            let image_format = image_format.to_string();
            let array_format = array_format.to_string();
            let pb_clone = pb.clone();

            async move {
                let id_clone = id.clone();
                let result = tokio::task::spawn_blocking(move || {
                    extract(
                        &id,
                        &image,
                        &bounding_boxes,
                        pad,
                        drop_borders,
                        min_size,
                        &mode,
                        &output,
                        &image_format,
                        &array_format,
                    )
                })
                .await
                .unwrap_or_else(|_| {
                    Err(ThymeError::OtherError(
                        "Failed to extract objects.".to_string(),
                    ))
                });

                if verbose {
                    pb_clone.lock().unwrap().update(1).unwrap();
                }

                (id_clone, result)
            }
        })
        .buffer_unordered(threads)
        .collect::<Vec<_>>()
        .await
}
