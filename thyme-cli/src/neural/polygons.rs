// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;

use candle_core::{Device, utils::cuda_is_available, utils::metal_is_available};
use clap::Args;
use kdam::TqdmParallelIterator;
use polars::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use thyme_core::constant;
use thyme_core::error::ThymeError;
use thyme_core::im;
use thyme_core::io;
use thyme_core::ut;
use thyme_data::data::Weights;
use thyme_neural::nn::Models;

#[derive(Debug, Args)]
pub struct NeuralPolygonsArgs {
    #[arg(short = 'i', long, help = "Image directory.", required = true)]
    pub images: Option<String>,

    #[arg(short = 's', long, help = "Polygons directory.")]
    pub polygons: Option<String>,

    #[arg(long, help = "Device (cpu, cuda, metal).", default_value = "cpu")]
    pub device: Option<String>,

    #[arg(
        short = 'o',
        long,
        help = "Output directory or file (.csv, .txt, .tsv, .pq, .npy, .npz).",
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
        help = "Model name.",
        default_value = "dino_vit_small"
    )]
    pub model: Option<String>,

    #[arg(
        short = 'p',
        long,
        help = "Add padding around objects before computing self-supervised features.",
        default_value = "1"
    )]
    pub pad: Option<u32>,

    #[arg(long, help = "Substring specifying images (e.g. _image).")]
    pub image_substring: Option<String>,

    #[arg(long, help = "Substring specifying polygons (e.g. _polygons).")]
    pub polygon_substring: Option<String>,

    #[arg(
        long,
        help = "Exclude objects smaller than a minimum size.",
        default_value = "1"
    )]
    pub min_size: Option<u32>,

    #[arg(short = 't', long, help = "Number of threads.")]
    pub threads: Option<usize>,
}

pub fn neural_image_polygons(args: &NeuralPolygonsArgs) {
    let device = args.device.to_owned().unwrap_or("cpu".to_string());

    if !["cpu", "metal", "cuda"].iter().any(|d| d == &device) {
        eprintln!(
            "[thyme::neural::polygons] ERROR: Invalid device. Must be one of: cpu, metal, cuda."
        );
        std::process::exit(1);
    }

    if device == "cuda" && !cuda_is_available() {
        println!(
            "[thyme::neural::polygons] Device 'cuda' specified but no cuda device was detected."
        );
        std::process::exit(1);
    }

    if device == "metal" && !metal_is_available() {
        println!(
            "[thyme::neural::polygons] Device 'metal' specified but no metal device was detected."
        );
        std::process::exit(1);
    }

    let (threads, device) = if device == "cuda" && cuda_is_available() {
        ut::track::progress_log("Cuda device detected.", args.verbose);
        (1, Device::new_cuda(0).unwrap())
    } else if device == "metal" && metal_is_available() {
        ut::track::progress_log("Metal device detected.", args.verbose);
        (1, Device::new_metal(0).unwrap())
    } else {
        (args.threads.to_owned().unwrap(), Device::Cpu)
    };

    if threads < 1 {
        println!(
            "[thyme::neural::polygons] Threads must be set to a positive integer if provided."
        );
        std::process::exit(1);
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let model_name = args
        .model
        .to_owned()
        .unwrap_or("dino_vit_small".to_string());

    let pad = args.pad.unwrap_or(1);
    let min_size = args.min_size.unwrap_or(1);

    if !Weights::iter().any(|m| m.model_name() == model_name) {
        // If model name is invalid, select will terminate and show error with list of available models
        Weights::select(&model_name);
    }

    if min_size < 1 {
        eprintln!("[thyme::neural::polygons] ERROR: min_size cannot be less than 1.0.");
        std::process::exit(1);
    }

    let image_path = args.images.to_owned().unwrap();
    let polygons_path = args.polygons.to_owned().unwrap_or(image_path.clone());

    if image_path == polygons_path && args.image_substring == args.polygon_substring {
        eprintln!(
            "[thyme::neural::polygons] ERROR: If images and polygons are located in same path, different image and polygon substrings must be provided."
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

    let polygon_files = ut::path::collect_file_paths(
        &polygons_path,
        constant::SUPPORTED_ARRAY_FORMATS.as_slice(),
        args.polygon_substring.to_owned(),
    )
    .unwrap_or_else(|err| {
        eprintln!("{}", err);
        std::process::exit(1);
    });

    if image_files.is_empty() {
        eprintln!(
            "[thyme::neural::polygons] ERROR: No image files were detected. Please check your path and/or substring identifier."
        );
        std::process::exit(1);
    }

    if polygon_files.is_empty() {
        eprintln!(
            "[thyme::neural::polygons] ERROR: No polygon files were detected. Please check your path and/or substring identifier."
        );
        std::process::exit(1);
    }

    let mut pairs = ut::path::collect_file_pairs(
        &image_files,
        &polygon_files,
        args.image_substring.to_owned(),
        args.polygon_substring.to_owned(),
    );

    pairs.sort_unstable();

    ut::track::progress_log(
        &format!(
            "Detected {} image and polygon pairs.",
            ut::track::thousands_format(pairs.len())
        ),
        args.verbose,
    );

    let mut output = PathBuf::from(args.output.to_owned().unwrap());

    let extension = output
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    if let Some(ext) = &extension {
        if !["npy", "npz", "csv", "txt", "tsv", "pq"]
            .iter()
            .any(|e| e == ext)
        {
            eprintln!(
                "[thyme::neural::polygons] ERROR: Invalid file extension. Must end with one of .npy, .npz, .csv, .txt, .tsv, .pq."
            );
            std::process::exit(1);
        }

        if let Some(parent) = output.parent() {
            if !parent.is_dir() && parent.to_str().unwrap() != "" {
                eprintln!(
                    "[thyme::neural::polygons] ERROR: Invalid file path. Parent directory of output file path does not exist."
                );
                std::process::exit(1);
            }
        }
    } else {
        output = ut::path::create_directory(&output).unwrap_or_else(|_| {
            eprintln!("[thyme::neural::polygons] ERROR: Could not create directory.");
            std::process::exit(1);
        });
    }

    let pb = ut::track::progress_bar(pairs.len(), "Embedding", args.verbose);

    let objects: Mutex<usize> = Mutex::new(0);
    let success: Mutex<Vec<String>> = Mutex::new(vec![]);
    let failure: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(pairs.len()));

    let name: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(pairs.len()));
    let item: Mutex<Vec<u32>> = Mutex::new(Vec::with_capacity(pairs.len()));
    let spot: Mutex<Vec<[f32; 2]>> = Mutex::new(Vec::with_capacity(2 * pairs.len()));
    let data: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::with_capacity(768 * pairs.len()));

    let model = Arc::new(Models::load(&model_name, &device, args.verbose));

    (0..pairs.len())
        .into_par_iter()
        .tqdm_with_bar(pb)
        .for_each(|idx| {
            let (id, image, polygons) = &pairs[idx];
            let run = neural(
                image,
                polygons,
                pad,
                args.drop_borders,
                min_size,
                &model.clone(),
                &device,
            );

            if let Ok((ids, centroids, embeddings)) = run {
                let n = ids.len();

                success.lock().unwrap().push(format!("{}\t{}", id, n));

                let image = image.file_stem().unwrap().to_string_lossy().to_string();

                name.lock().unwrap().extend((0..n).map(|_| image.clone()));
                item.lock().unwrap().extend(ids);
                spot.lock().unwrap().extend(centroids);
                data.lock().unwrap().extend(embeddings);

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
    let spot = spot.into_inner().unwrap();
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
        let n_row = data.len();
        let n_col = data[0].len();

        if let Some(ext) = &extension {
            if ["csv", "txt", "tsv", "pq"].iter().any(|e| e == ext) {
                let mut df = DataFrame::new(vec![
                    Column::new("image".into(), &name),
                    Column::new("object".into(), &item),
                    Column::new(
                        "centroid_x".into(),
                        &spot.iter().map(|x| x[0]).collect::<Vec<f32>>(),
                    ),
                    Column::new(
                        "centroid_y".into(),
                        &spot.iter().map(|x| x[0]).collect::<Vec<f32>>(),
                    ),
                ])
                .unwrap();

                let mut column_data: Vec<Vec<f32>> = vec![Vec::with_capacity(n_row); n_col];

                for row in &data {
                    for (idx, &descriptor) in row.iter().enumerate() {
                        column_data[idx].push(descriptor);
                    }
                }

                for (idx, column) in column_data.iter().enumerate() {
                    df.with_column(Column::new(idx.to_string().into(), column))
                        .unwrap();
                }

                io::write_table(&mut df, &output).unwrap_or_else(|_| {
                    eprintln!(
                        "[thyme::neural::polygons] ERROR: Failed to write embeddings to a table."
                    );
                    std::process::exit(1);
                });
            } else if ext == "npy" {
                io::write_numpy(
                    &output,
                    data.iter().flatten().collect(),
                    vec![n_row as u64, n_col as u64],
                )
                .unwrap_or_else(|_| {
                    eprintln!(
                        "[thyme::neural::polygons] ERROR: Failed to write embeddings to a npy array."
                    );
                    std::process::exit(1);
                });
            } else if ext == "npz" {
                io::write_embeddings_npz(name, item, spot, data, &output).unwrap_or_else(|_| {
                    eprintln!(
                        "[thyme::neural::polygons] ERROR: Failed to write embeddings to an npz array."
                    );
                    std::process::exit(1);
                });
            }
        } else {
            io::write_embeddings_npz(name, item, spot, data, &output.join("embeddings.npz"))
                .unwrap_or_else(|_| {
                    eprintln!(
                        "[thyme::neural::polygons] ERROR: Failed to write embeddings to an npz array."
                    );
                    std::process::exit(1);
                });
        }
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

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn neural(
    image_path: &Path,
    polygons_path: &Path,
    pad: u32,
    drop_borders: bool,
    min_size: u32,
    model: &Models,
    device: &Device,
) -> Result<(Vec<u32>, Vec<[f32; 2]>, Vec<Vec<f32>>), ThymeError> {
    let image = im::ThymeImage::open(image_path)?;

    let polygons = im::Polygons::open(polygons_path)?;
    let bounding_boxes = polygons.to_bounding_boxes()?;

    let width = image.width();
    let height = image.height();

    let pad_f32 = pad as f32;

    let mut ids: Vec<u32> = Vec::with_capacity(bounding_boxes.len());
    let mut centroids: Vec<[f32; 2]> = Vec::with_capacity(2 * bounding_boxes.len());
    let mut results: Vec<Vec<f32>> = Vec::with_capacity(300 * bounding_boxes.len());

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

        let min_x_u32 = min_x.max(0.0) as u32;
        let min_y_u32 = min_y.max(0.0) as u32;
        let max_x_u32 = max_x.min(width as f32) as u32;
        let max_y_u32 = max_y.min(height as f32) as u32;

        let w = max_x_u32 - min_x_u32;
        let h = max_y_u32 - min_y_u32;

        if w < min_size || h < min_size {
            continue;
        }

        ids.push(idx as u32);
        centroids.push([(max_x + min_x) / 2.0, (max_y + min_y) / 2.0]);

        results.push(
            model
                .forward(
                    &model
                        .preprocess(&image.crop(min_x_u32, min_y_u32, w, h)?, device)
                        .unwrap(),
                )
                .unwrap()
                .get(0)
                .unwrap()
                .to_vec1()
                .unwrap(),
        );
    }

    Ok((ids, centroids, results))
}
