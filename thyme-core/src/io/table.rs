// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::fs::File;
use std::path::Path;

use polars::prelude::*;

use crate::error::ThymeError;

/// Write a table to a CSV file
///
/// # Arguments
///
/// * `df` - A DataFrame
/// * `output` - A string containing the name of the output file
/// * `header` - A boolean indicating whether the output file should contain a header
///
/// # Examples
///
/// ```no_run
/// use polars::prelude::*;
/// use thyme_core::io::write_table_csv;
///
/// let column = vec![Column::new("area".into(), [2.5, 3.1, 3.4])];
/// let mut df: DataFrame = DataFrame::new(column).unwrap();
///
/// write_table_csv(&mut df, "output.csv", true).unwrap()
/// ```
pub fn write_table_csv<P: AsRef<Path>>(
    df: &mut DataFrame,
    path: P,
    header: bool,
) -> Result<(), ThymeError> {
    let mut output: File = File::create(&path).map_err(|_| {
        ThymeError::OtherError(format!(
            "Failed to create CSV file: {}",
            path.as_ref().to_str().unwrap()
        ))
    })?;

    CsvWriter::new(&mut output)
        .include_header(header)
        .finish(df)
        .map_err(|_| ThymeError::OtherError("Failed to write CSV file.".to_string()))
}

/// Write a table to a TSV file
///
/// # Arguments
///
/// * `df` - A DataFrame
/// * `output` - A string containing the name of the output file
/// * `header` - A boolean indicating whether the output file should contain a header
///
/// # Examples
///
/// ```no_run
/// use polars::prelude::*;
/// use thyme_core::io::write_table_tsv;
///
/// let column = vec![Column::new("area".into(), [2.5, 3.1, 3.4])];
/// let mut df: DataFrame = DataFrame::new(column).unwrap();
///
/// write_table_tsv(&mut df, "output.tsv", true).unwrap()
/// ```
pub fn write_table_tsv<P: AsRef<Path>>(
    df: &mut DataFrame,
    path: P,
    header: bool,
) -> Result<(), ThymeError> {
    let mut output: File = File::create(&path).map_err(|_| {
        ThymeError::OtherError(format!(
            "Failed to create TSV file: {}",
            path.as_ref().to_str().unwrap()
        ))
    })?;

    CsvWriter::new(&mut output)
        .include_header(header)
        .with_separator("\t".as_bytes()[0])
        .finish(df)
        .map_err(|_| ThymeError::OtherError("Failed to write TSV file.".to_string()))
}

/// Write a table to a parquet file
///
/// # Arguments
///
/// * `df` - A DataFrame
/// * `output` - A string containing the name of the output file
///
/// # Examples
///
/// ```no_run
/// use polars::prelude::*;
/// use thyme_core::io::write_table_pq;
///
/// let column = vec![Column::new("area".into(), [2.5, 3.1, 3.4])];
/// let mut df: DataFrame = DataFrame::new(column).unwrap();
///
/// write_table_pq(&mut df, "output.pq").unwrap()
/// ```
pub fn write_table_pq<P: AsRef<Path>>(df: &mut DataFrame, path: P) -> Result<(), ThymeError> {
    let mut output: File = File::create(&path).map_err(|_| {
        ThymeError::OtherError(format!(
            "Failed to create TSV file: {}",
            path.as_ref().to_str().unwrap()
        ))
    })?;

    ParquetWriter::new(&mut output)
        .finish(df)
        .map(|_| ())
        .map_err(|_| ThymeError::OtherError("Failed to write parquet file.".to_string()))
}

/// Write a DataFrame to disk
///
/// # Arguments
///
/// * `df` - A DataFrame
/// * `output` - A string containing the name of the output file
/// * `id` - An optional string specifying a single id to prepend as first column
///
/// # Examples
///
/// ```no_run
/// use polars::prelude::*;
/// use thyme_core::io::write_table;
///
/// let column = vec![Column::new("area".into(), [2.5, 3.1, 3.4])];
/// let mut df: DataFrame = DataFrame::new(column).unwrap();
///
/// write_table(&mut df, "output.csv").unwrap()
/// ```
pub fn write_table<P: AsRef<Path>>(df: &mut DataFrame, path: P) -> Result<(), ThymeError> {
    let extension = path
        .as_ref()
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    if let Some(ext) = extension {
        match ext.as_str() {
            "csv" => write_table_csv(df, path, true),
            "tsv" => write_table_tsv(df, path, true),
            "txt" => write_table_tsv(df, path, true),
            "parquet" => write_table_pq(df, path),
            "pq" => write_table_pq(df, path),
            _ => Err(ThymeError::OtherError("Failed to write table.".to_string())),
        }
    } else {
        Err(ThymeError::OtherError(
            "Provided table path has an invalid extension. Must be one of: csv, tsv, txt, parquet, or pq.".to_string()
        ))
    }
}
