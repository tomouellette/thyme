// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use zarrs::array::codec::GzipCodec;
use zarrs::array::{ArrayBuilder, DataType, FillValue, ZARR_NAN_F32, ZARR_NAN_F64};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::storage::ReadableWritableListableStorage;

use clap::Args;
use kdam::TqdmParallelIterator;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use thyme_core::constant;
use thyme_core::im;
use thyme_core::ut;

#[derive(Debug, Args)]
pub struct Images2zarrArgs {
    #[arg(short = 'i', long, help = "Image directory.", required = true)]
    pub images: Option<String>,

    #[arg(short = 'o', long, help = "Output zarr file.", required = true)]
    pub output: Option<String>,

    #[arg(long, help = "Resize each image to specified width.", required = true)]
    pub resize_width: Option<u32>,

    #[arg(long, help = "Resize each image to specified height.", required = true)]
    pub resize_height: Option<u32>,

    #[arg(long, help = "Number of image channels.", required = true)]
    pub channels: Option<u32>,

    #[arg(
        long,
        help = "Cast subpixels to specified data type (u8, u16, u32, f32, or f64).",
        default_value = "f32"
    )]
    pub dtype: Option<String>,

    #[arg(long, help = "Gzip compression level (0 - 9)", default_value = "5")]
    pub gzip_compression: Option<u32>,

    #[arg(short = 'v', long, help = "Verbose output.")]
    pub verbose: bool,

    #[arg(long, help = "Substring specifying images (e.g. _image).")]
    pub image_substring: Option<String>,

    #[arg(short = 't', long, help = "Number of threads.")]
    pub threads: Option<usize>,
}

pub fn utils_images2zarr(args: &Images2zarrArgs) {
    if let Some(threads) = args.threads.to_owned() {
        if threads < 1 {
            println!(
                "[thyme::utils::images2zarr] ERROR: Threads must be set to a positive integer if provided."
            );
            std::process::exit(1);
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let image_path = args.images.to_owned().unwrap();
    let resize_width = args.resize_width.unwrap();
    let resize_height = args.resize_width.unwrap();
    let channels = args.channels.unwrap();
    let dtype = args.dtype.to_owned().unwrap();
    let gzip_compression = args.gzip_compression.to_owned().unwrap();

    let (zarr_dtype, zarr_fill) = match dtype.as_str() {
        "u8" => (DataType::UInt8, FillValue::from(0_u8)),
        "u16" => (DataType::UInt16, FillValue::from(0_u16)),
        "u32" => (DataType::UInt32, FillValue::from(0_u32)),
        "f32" => (DataType::Float32, FillValue::from(ZARR_NAN_F32)),
        "f64" => (DataType::Float64, FillValue::from(ZARR_NAN_F64)),
        _ => {
            eprintln!(
                "[thyme::utils::images2zarr] ERROR: Invalid dtype. Only u8, u16, f32, f64 data types are supported."
            );
            std::process::exit(1);
        }
    };

    if !(0..=9).contains(&gzip_compression) {
        eprintln!(
            "[thyme::utils::images2zarr] ERROR: Invalid gzip_compression. Must be 0 to 9 inclusive."
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

    if image_files.is_empty() {
        eprintln!(
            "[thyme::utils::images2zarr] ERROR: No image files were detected. Please check your path and/or substring identifier."
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

    let output = PathBuf::from(args.output.to_owned().unwrap());

    let extension = output
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    if let Some(ext) = extension {
        if "zarr" != ext {
            eprintln!(
                "[thyme::utils::images2zarr] ERROR: Invalid output extension. Must end with one of zarr."
            );
            std::process::exit(1);
        }

        if let Some(parent) = output.parent() {
            if !parent.is_dir() && parent.to_str().unwrap() != "" {
                eprintln!(
                    "[thyme::utils::images2zarr] ERROR: Invalid output path. Parent directory of output file path does not exist."
                );
                std::process::exit(1);
            }
        }
    } else {
        eprintln!(
            "[thyme::utils::images2zarr] ERROR: Invalid output extension. Must end with one of zarr."
        );
        std::process::exit(1);
    }

    let pb = ut::track::progress_bar(image_files.len(), "Convert images to zarr", args.verbose);

    let shape: [u64; 4] = [
        image_files.len() as u64,
        resize_height as u64,
        resize_width as u64,
        channels as u64,
    ];

    let chunk_shape: [u64; 4] = [
        1,
        resize_height as u64,
        resize_width as u64,
        channels as u64,
    ];

    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(output).unwrap_or_else(|_| {
            eprintln!("[thyme::utils::images2zarr] Failed to create zarr filesystem store.");
            std::process::exit(1);
        }));

    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    let mut images_builder = ArrayBuilder::new(
        shape.to_vec(),
        zarr_dtype.clone(),
        chunk_shape.to_vec().try_into().unwrap(),
        zarr_fill,
    );

    let images_builder = if gzip_compression == 0 {
        images_builder.dimension_names(["n", "y", "x", "c"].into())
    } else if (1..=9).contains(&gzip_compression) {
        images_builder
            .bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(gzip_compression).unwrap())])
            .dimension_names(["n", "y", "x", "c"].into())
    } else {
        eprintln!("[thyme::utils::images2zarr] Gzip compression level must be in [0, 9].");
        std::process::exit(1);
    };

    let images_array = images_builder
        .attributes(
            serde_json::json!({
                "resize_width": resize_width,
                "resize_height": resize_height,
                "channels": channels
            })
            .as_object()
            .unwrap()
            .clone(),
        )
        .build_arc(store.clone(), "/images")
        .unwrap();

    images_array.store_metadata().unwrap();

    let max_name_length = 100;

    let names_array = zarrs::array::ArrayBuilder::new(
        vec![image_files.len() as u64, max_name_length],
        DataType::UInt8,
        vec![1u64, max_name_length].try_into().unwrap(),
        FillValue::from(0u8),
    )
    .dimension_names(["y", "s"].into())
    .build_arc(store.clone(), "/names")
    .unwrap();

    names_array.store_metadata().unwrap();

    let erase: Mutex<Vec<u64>> = Mutex::new(Vec::with_capacity(image_files.len()));

    (0..image_files.len())
        .into_par_iter()
        .tqdm_with_bar(pb)
        .for_each(|idx| {
            let image = im::ThymeImage::open(&image_files[idx]);

            let image_name = image_files[idx]
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();

            if let Ok(img) = image {
                if img.channels() != channels {
                    erase.lock().unwrap().push(idx as u64);
                    return;
                }

                if let Ok(img) = img.resize(resize_width, resize_height) {
                    let result = match zarr_dtype {
                        DataType::UInt8 => images_array
                            .store_chunk_elements(&[idx as u64, 0, 0, 0], img.to_u8().as_slice()),
                        DataType::UInt16 => images_array
                            .store_chunk_elements(&[idx as u64, 0, 0, 0], img.to_u16().as_slice()),
                        DataType::UInt32 => images_array
                            .store_chunk_elements(&[idx as u64, 0, 0, 0], img.to_u32().as_slice()),
                        DataType::Float32 => images_array
                            .store_chunk_elements(&[idx as u64, 0, 0, 0], img.to_f32().as_slice()),
                        DataType::Float64 => images_array
                            .store_chunk_elements(&[idx as u64, 0, 0, 0], img.to_f64().as_slice()),
                        _ => unreachable!(),
                    };

                    if result.is_ok() {
                        result.unwrap();

                        let mut padded = vec![0u8; max_name_length as usize];
                        let bytes = image_name.as_bytes();
                        let len = bytes.len().min(max_name_length as usize);
                        padded[..len].copy_from_slice(&bytes[..len]);

                        names_array
                            .store_chunk_elements(&[idx as u64, 0], &padded)
                            .unwrap();
                    } else {
                        erase.lock().unwrap().push(idx as u64);
                    }
                } else {
                    erase.lock().unwrap().push(idx as u64);
                };
            } else {
                erase.lock().unwrap().push(idx as u64);
            }
        });

    let erase = erase.into_inner().unwrap();

    for idx in erase.iter() {
        images_array.erase_chunk(&[*idx, 0, 0, 0]).unwrap();
        names_array.erase_chunk(&[*idx]).unwrap();
    }

    if args.verbose {
        println!()
    }

    let message = if !erase.is_empty() {
        &format!(
            "Complete. {} images transferred to zarr array. {} images failed transfer.",
            ut::track::thousands_format(image_files.len() - erase.len()),
            ut::track::thousands_format(erase.len())
        )
    } else {
        &format!(
            "Complete. {} images transferred to zarr array.",
            ut::track::thousands_format(image_files.len() - erase.len()),
        )
    };

    ut::track::progress_log(message, args.verbose);
}
