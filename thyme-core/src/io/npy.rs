// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::fs::File;
use std::io;
use std::path::Path;

use npyz::{self, WriterBuilder};
use npyz::{TypeStr, npz};
use zip::write::ExtendedFileOptions;

use crate::error::ThymeError;

/// Write a numpy file from a vector of specified shape
///
/// # Arguments
///
/// * `path` - Path to output numpy file
/// * `data` - Vector of numeric type
/// * `shape` - Shape of the vector (shape product must equal length of data)
pub fn write_numpy<T, P: AsRef<Path>>(
    path: P,
    data: Vec<T>,
    shape: Vec<u64>,
) -> Result<(), ThymeError>
where
    T: npyz::Serialize + npyz::AutoSerialize,
{
    let mut buffer = vec![];
    let mut writer = npyz::WriteOptions::<T>::new()
        .default_dtype()
        .shape(&shape)
        .writer(&mut buffer)
        .begin_nd()
        .map_err(|_| ThymeError::ImageWriteError)?;

    for d in data {
        let _ = writer.push(&d);
    }

    writer.finish().map_err(|_| ThymeError::ImageWriteError)?;
    std::fs::write(path, buffer).map_err(|_| ThymeError::ImageWriteError)?;
    Ok(())
}

/// Write neural network single object embeddings to a .npz file
///
/// # Arguments
///
/// * `images` - Image names for each object
/// * `ids` - Object identifiers
/// * `centroids` - Object centroids
/// * `embeddings` - Object self-supervised features/embeddings
/// * `output` - Path to output .npz file
///
/// # Examples
///
/// ```no_run
/// ```
/// use thyme_core::io::write_embeddings_npz;
pub fn write_embeddings_npz<P: AsRef<Path>>(
    images: Vec<String>,
    ids: Vec<u32>,
    centroids: Vec<[f32; 2]>,
    embeddings: Vec<Vec<f32>>,
    output: &P,
) -> Result<(), ThymeError> {
    let file = io::BufWriter::new(
        File::create(output)
            .map_err(|_| ThymeError::OtherError("Failed to create .npz file".to_string()))?,
    );

    let mut zip = zip::ZipWriter::new(file);

    if images.len() != embeddings.len() {
        return Err(ThymeError::OtherError(
            "Image names and embeddings must have same length when saving .npz.".to_string(),
        ));
    }

    if !ids.is_empty() && ids.len() != embeddings.len() {
        return Err(ThymeError::OtherError(
            "Object identifiers and embeddings must have same length when saving .npz.".to_string(),
        ));
    }

    if !centroids.is_empty() && centroids.len() != embeddings.len() {
        return Err(ThymeError::OtherError(
            "Object centroids and embeddings must have same length when saving .npz.".to_string(),
        ));
    }

    let n = embeddings.len() as u64;
    let m = embeddings[0].len() as u64;

    // IMAGE NAMES

    zip.start_file::<_, ExtendedFileOptions>(
        npz::file_name_from_array_name("image"),
        Default::default(),
    )
    .map_err(|_| {
        ThymeError::OtherError(
            "Failed to initiailize zip file for image names in .npz file".to_string(),
        )
    })?;

    let mut writer = npyz::WriteOptions::new()
        .dtype(npyz::DType::Plain("<U53".parse::<TypeStr>().unwrap()))
        .shape(&[n])
        .writer(&mut zip)
        .begin_nd()
        .map_err(|_| {
            ThymeError::OtherError(
                "Failed to initiailize writer for image names in .npz file".to_string(),
            )
        })?;

    writer
        .extend(images.iter().map(|image| image.as_str()))
        .map_err(|_| {
            ThymeError::OtherError("Failed to add image names to .npz file".to_string())
        })?;

    writer.finish().map_err(|_| {
        ThymeError::OtherError("Failed to write image names to .npz file".to_string())
    })?;

    // IDENTIFIERS

    if !ids.is_empty() {
        zip.start_file::<_, ExtendedFileOptions>(
            npz::file_name_from_array_name("id"),
            Default::default(),
        )
        .map_err(|_| {
            ThymeError::OtherError(
                "Failed to initiailize zip file for identifiers in .npz file".to_string(),
            )
        })?;

        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[n])
            .writer(&mut zip)
            .begin_nd()
            .map_err(|_| {
                ThymeError::OtherError(
                    "Failed to initialize writer for identifiers in .npz file".to_string(),
                )
            })?;

        writer.extend(ids).map_err(|_| {
            ThymeError::OtherError("Failed to add identifiers to .npz file".to_string())
        })?;

        writer.finish().map_err(|_| {
            ThymeError::OtherError("Failed to write identifiers to .npz file".to_string())
        })?;
    }

    // CENTROIDS

    if !centroids.is_empty() {
        zip.start_file::<_, ExtendedFileOptions>(
            npz::file_name_from_array_name("centroid"),
            Default::default(),
        )
        .map_err(|_| {
            ThymeError::OtherError(
                "Failed to initiailize zip file for centroids in .npz file".to_string(),
            )
        })?;

        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[n, 2])
            .writer(&mut zip)
            .begin_nd()
            .map_err(|_| {
                ThymeError::OtherError(
                    "Failed to initialize writer for centroids in .npz file".to_string(),
                )
            })?;

        writer
            .extend(centroids.iter().flat_map(|r| r.iter().cloned()))
            .map_err(|_| {
                ThymeError::OtherError("Failed to add centroids to .npz file".to_string())
            })?;

        writer.finish().map_err(|_| {
            ThymeError::OtherError("Failed to write centroids to .npz file".to_string())
        })?;
    }

    // EMBEDDINGS

    zip.start_file::<_, ExtendedFileOptions>(
        npz::file_name_from_array_name("embedding"),
        Default::default(),
    )
    .map_err(|_| {
        ThymeError::OtherError(
            "Failed to initiailize zip file for embeddings in .npz file".to_string(),
        )
    })?;

    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[n, m])
        .writer(&mut zip)
        .begin_nd()
        .map_err(|_| {
            ThymeError::OtherError(
                "Failed to initiailize writer for embeddings in .npz file".to_string(),
            )
        })?;

    writer
        .extend(embeddings.iter().flat_map(|r| r.iter().cloned()))
        .map_err(|_| ThymeError::OtherError("Failed to add embeddings to .npz file".to_string()))?;

    writer.finish().map_err(|_| {
        ThymeError::OtherError("Failed to write image names to .npz file".to_string())
    })?;

    zip.finish()
        .map_err(|_| ThymeError::OtherError("Failed to zip .npz file".to_string()))?;

    Ok(())
}
