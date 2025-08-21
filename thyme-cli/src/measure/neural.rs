// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::io::Write;
use std::path::{Path, PathBuf};
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
pub struct NeuralArgs {
    #[arg(short = 'i', long, help = "Image or image directory.", required = true)]
    pub images: Option<String>,

    #[arg(
        short = 'o',
        long,
        help = "Output directory or file (.csv, .txt, .tsv, .pq, .npy, .npz)."
    )]
    pub output: Option<String>,

    #[arg(
        long,
        short = 'm',
        help = "Model name.",
        default_value = "dino_vit_small"
    )]
    pub model: Option<String>,

    #[arg(
        short = 'd',
        long,
        help = "Device (cpu, cuda, metal).",
        default_value = "cpu"
    )]
    pub device: Option<String>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(long, help = "Substring specifying images (e.g. _image).")]
    pub image_substring: Option<String>,

    #[arg(short = 't', long, help = "Number of threads.")]
    pub threads: Option<usize>,
}

pub fn measure_neural(args: &NeuralArgs) {
    let device = args.device.to_owned().unwrap_or("cpu".to_string());

    if !["cpu", "metal", "cuda"].iter().any(|d| d == &device) {
        eprintln!(
            "[thyme::measure::neural] ERROR: Invalid device. Must be one of: cpu, metal, cuda."
        );
        std::process::exit(1);
    }

    if device == "cuda" && !cuda_is_available() {
        println!(
            "[thyme::measure::neural] Device 'cuda' specified but no cuda device was detected."
        );
        std::process::exit(1);
    }

    if device == "metal" && !metal_is_available() {
        println!(
            "[thyme::measure::neural] Device 'metal' specified but no metal device was detected."
        );
        std::process::exit(1);
    }

    let device = if device == "cuda" && cuda_is_available() {
        Device::new_cuda(0).unwrap()
    } else if device == "metal" && metal_is_available() {
        Device::new_metal(0).unwrap()
    } else {
        Device::Cpu
    };

    let model_name = args
        .model
        .to_owned()
        .unwrap_or("dino_vit_small".to_string());

    if !Weights::iter().any(|m| m.model_name() == model_name) {
        // If model name is invalid, select will terminate and show error with list of available models
        Weights::select(&model_name);
    }

    let image_path = args.images.to_owned().unwrap();

    let image_extension = Path::new(&image_path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    let is_image_dir = if let Some(ext) = &image_extension {
        if !constant::SUPPORTED_IMAGE_FORMATS.contains(&ext.as_str()) {
            eprintln!(
                "[thyme::measure::neural] ERROR: Invalid image extension {}. Must be one of: {:?}.",
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
        if let Some(threads) = args.threads.to_owned() {
            if threads < 1 {
                println!(
                    "[thyme::measure::neural] Threads must be set to a positive integer if provided."
                );
                std::process::exit(1);
            }

            if matches!(device, Device::Cpu) {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build_global()
                    .unwrap();
            }
        }

        if !is_image_dir {
            eprintln!(
                "[thyme::measure::neural] ERROR: If output is provided, then input image path must specify an image directory."
            );
            std::process::exit(1);
        }

        let output = PathBuf::from(output);

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
                    "[thyme::measure::neural] ERROR: Invalid file extension. Must end with one of .npy, .npz, .csv, .txt, .tsv, .pq."
                );
                std::process::exit(1);
            }
        } else {
            eprintln!(
                "[thyme::measure::neural] ERROR: Invalid output path. Output file must be a file with a valid extension."
            );
            std::process::exit(1);
        }

        if let Some(parent) = output.parent() {
            if !parent.is_dir() && parent.to_str().unwrap() != "" {
                eprintln!(
                    "[thyme::measure::neural] ERROR: Invalid file path. Parent directory of output file path does not exist."
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
                "[thyme::measure::neural] ERROR: No image files were detected. Please check your path and/or substring identifier."
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

        let pb = ut::track::progress_bar(image_files.len(), "Measuring neural", args.verbose);

        let failure: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(image_files.len()));
        let name: Mutex<Vec<String>> = Mutex::new(Vec::with_capacity(image_files.len()));
        let data: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::with_capacity(768 * image_files.len()));

        let model = Arc::new(Models::load(&model_name, &device, args.verbose));

        (0..image_files.len())
            .into_par_iter()
            .tqdm_with_bar(pb)
            .for_each(|idx| {
                let result = neural(&image_files[idx], &model, &device);

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

        if !data.is_empty() {
            write_neural(&data, &name, &output, extension.unwrap().as_str());
        }
    } else {
        if is_image_dir {
            eprintln!(
                "[thyme::measure::neural] ERROR: If output is not provided, then input image path should specify a single file."
            );
            std::process::exit(1);
        }

        let image_path = Path::new(&image_path);

        if !image_path.is_file() {
            eprintln!(
                "[thyme::measure::neural] ERROR: The provided image file path does not exist."
            );
            std::process::exit(1);
        }

        let model = Models::load(&model_name, &device, args.verbose);

        let data = neural(Path::new(&image_path), &model, &device).unwrap_or_else(|_| {
            eprintln!("[thyme::measure::neural] ERROR: Failed to measure neural descriptors.");
            std::process::exit(1);
        });

        let output: String = data
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("\t");

        let mut stdout = std::io::stdout();

        stdout.write_all(output.as_bytes()).unwrap();
    }
}

/// Measure neural descriptors across an image
fn neural(image_path: &Path, model: &Models, device: &Device) -> Result<Vec<f32>, ThymeError> {
    let image = im::ThymeImage::open(image_path)?;

    Ok(model
        .forward(&model.preprocess(&image, device).unwrap())
        .unwrap()
        .get(0)
        .unwrap()
        .to_vec1()
        .unwrap())
}

/// Write neural descriptors to data table
fn write_neural(data: &[Vec<f32>], name: &Vec<String>, output: &PathBuf, extension: &str) {
    let n_row = data.len();
    let n_col = data[0].len();

    if ["csv", "txt", "tsv", "pq"].iter().any(|e| e == &extension) {
        let mut df = DataFrame::new(vec![Column::new("image".into(), &name)]).unwrap();

        let mut column_data: Vec<Vec<f32>> = vec![Vec::with_capacity(n_row); n_col];

        for row in data {
            for (idx, &descriptor) in row.iter().enumerate() {
                column_data[idx].push(descriptor);
            }
        }

        for (idx, column) in column_data.iter().enumerate() {
            df.with_column(Column::new(idx.to_string().into(), column))
                .unwrap();
        }

        io::write_table(&mut df, output).unwrap_or_else(|_| {
            eprintln!("[thyme::measure::neural] ERROR: Failed to write embeddings to a table.");
            std::process::exit(1);
        });
    } else if extension == "npy" {
        io::write_numpy(
            output,
            data.iter().flatten().collect(),
            vec![n_row as u64, n_col as u64],
        )
        .unwrap_or_else(|_| {
            eprintln!("[thyme::measure::neural] ERROR: Failed to write embeddings to a npy array.");
            std::process::exit(1);
        });
    } else if extension == "npz" {
        io::write_embeddings_npz(name.to_vec(), vec![], vec![], data.to_vec(), &output)
            .unwrap_or_else(|_| {
                eprintln!(
                    "[thyme::measure::neural] ERROR: Failed to write embeddings to an npz array."
                );
                std::process::exit(1);
            });
    }
}
