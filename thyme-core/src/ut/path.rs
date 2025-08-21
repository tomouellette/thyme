// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::ThymeError;

/// Ensures a new directory is created with an incrementing suffix if necessary.
///
/// # Arguments
///
/// * `directory` - Path to new directory - no overwrites allowed
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use thyme_core::ut::path::create_directory;
///
/// let base = Path::new("TEST_CREATE_DIRECTORY");
///
/// std::fs::create_dir(base).unwrap();
/// assert!(base.exists());
///
/// let increment_0 = Path::new("TEST_CREATE_DIRECTORY_0/");
/// let increment_1 = Path::new("TEST_CREATE_DIRECTORY_1/");
/// let increment_2 = Path::new("TEST_CREATE_DIRECTORY_2/");
///
/// create_directory(base);
/// create_directory(base);
/// create_directory(base);
///
/// assert!(increment_0.exists());
/// assert!(increment_1.exists());
/// assert!(increment_2.exists());
///
/// std::fs::remove_dir(base);
/// std::fs::remove_dir(increment_0);
/// std::fs::remove_dir(increment_1);
/// std::fs::remove_dir(increment_2);
/// ```
pub fn create_directory<P: AsRef<Path>>(directory: P) -> Result<PathBuf, ThymeError> {
    let directory = directory.as_ref();

    if !directory.exists() {
        std::fs::create_dir(directory).map_err(|err| ThymeError::DirError(err.to_string()))?;
        return Ok(directory.to_path_buf());
    }

    let parent = directory.parent().unwrap_or_else(|| Path::new("."));
    let base_name = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ThymeError::DirError("Invalid directory name".to_string()))?;

    for index in 0..30 {
        let new_dir = parent.join(format!("{}_{}", base_name, index));

        if !new_dir.exists() {
            std::fs::create_dir(&new_dir).map_err(|err| ThymeError::DirError(err.to_string()))?;
            return Ok(new_dir);
        }
    }

    Err(ThymeError::DirError(format!(
        "Could not create a directory in alotted increments. Check the directory path: {}",
        directory.display()
    )))
}

/// Collect file paths from a directory with an optional substring filter
///
/// # Arguments
///
/// * `directory` - Path to directory containing files
/// * `substring` - Only include files containing this substring
///
/// # Examples
///
/// ```no_run
/// use thyme_core::ut::path::collect_file_paths;
/// use thyme_core::constant::SUPPORTED_IMAGE_FORMATS;
/// let files = collect_file_paths("directory/", SUPPORTED_IMAGE_FORMATS.as_slice(), None);
/// ```
pub fn collect_file_paths<P>(
    directory: P,
    valid_ext: &[&str],
    substring: Option<String>,
) -> Result<Vec<PathBuf>, ThymeError>
where
    P: AsRef<Path> + ToString,
{
    let message = directory.to_string();

    let mut files: Vec<PathBuf> = std::fs::read_dir(directory)
        .map_err(|_| ThymeError::DirError(message))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| valid_ext.contains(&ext))
        })
        .collect();

    if let Some(substring) = substring {
        files.retain(|f| {
            f.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains(&substring)
        });
    }

    Ok(files)
}

/// Collect file pairs that share matching prefix
///
/// # Arguments
///
/// * `files_a` - List of file paths
/// * `files_b` - List of file paths
/// * `suffix` - Optionally remove a suffix from the first set of file paths
///
/// # Examples
///
/// ```
/// use std::path::{Path, PathBuf};
/// use thyme_core::ut::path::collect_file_pairs;
///
/// let files_a: [PathBuf; 4] = [
///     PathBuf::from("directory/id_1.png"),
///     PathBuf::from("directory/id_2.png"),
///     PathBuf::from("directory/id_3.png"),
///     PathBuf::from("directory/id_4_image.png"),
/// ];
///
/// let files_b: [PathBuf; 4] = [
///     PathBuf::from("directory/id_1.png"),
///     PathBuf::from("directory/id_2.png"),
///     PathBuf::from("directory/id_3.png"),
///     PathBuf::from("directory/id_4.png"),
/// ];
///
/// let pairs = collect_file_pairs(&files_a, &files_b, None, None);
/// assert_eq!(pairs.len(), 3);
///
/// let pairs = collect_file_pairs(&files_a, &files_b, Some("_image".to_string()), None);
/// assert_eq!(pairs.len(), 4);
/// ```
pub fn collect_file_pairs(
    files_a: &[PathBuf],
    files_b: &[PathBuf],
    substring_a: Option<String>,
    substring_b: Option<String>,
) -> Vec<(String, PathBuf, PathBuf)> {
    let substring_a = substring_a.unwrap_or_default();
    let substring_b = substring_b.unwrap_or_default();

    let file_map: HashMap<String, &PathBuf> = files_a
        .iter()
        .filter_map(|file| {
            file.file_stem().map(|stem| {
                let name = stem.to_string_lossy().replace(&substring_a, "");
                (name, file)
            })
        })
        .collect();

    files_b
        .par_iter()
        .filter_map(|file_b| {
            file_b.file_stem().and_then(|stem| {
                let name = stem.to_string_lossy().replace(&substring_b, "");
                file_map
                    .get(&name)
                    .map(|file_a| (name, (*file_a).clone(), file_b.clone()))
            })
        })
        .collect()
}
