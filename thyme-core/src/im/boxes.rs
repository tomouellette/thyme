// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::constant::BOUNDING_BOX_JSON_VALID_KEYS;
use crate::error::ThymeError;

/// A bounding box container for storing locations of detected objects
///
/// The bounding boxes are stored in xyxy format. Any input set of
/// bounding boxes that has a box with a non-positive area will
/// return an error.
///
/// # Examples
///
/// ```
/// use thyme_core::im::BoundingBoxes;
///
/// let data: Vec<[f32; 4]> = vec![[0., 0., 1., 1.], [3., 4., 5., 7.]];
/// let boxes = BoundingBoxes::new(data);
/// assert!(boxes.is_ok());
///
/// let data: Vec<[f32; 4]> = vec![[2., 2., 1., 1.], [3., 4., 5., 7.]];
/// let boxes = BoundingBoxes::new(data);
/// assert!(boxes.is_err());
/// ```
#[derive(Debug, Clone)]
pub struct BoundingBoxes {
    data: Vec<[f32; 4]>,
}

impl BoundingBoxes {
    /// Initialize a new bounding boxes container
    ///
    /// # Arguments
    ///
    /// * `data` - Bounding boxes in xyxy format
    ///
    /// # Examples
    ///
    /// ```
    /// use thyme_core::im::BoundingBoxes;
    ///
    /// let data: Vec<[f32; 4]> = vec![[0., 0., 1., 1.], [3., 4., 5., 7.]];
    /// let boxes = BoundingBoxes::new(data);
    /// ```
    pub fn new(data: Vec<[f32; 4]>) -> Result<Self, ThymeError> {
        let n = data.len();

        let data: Vec<[f32; 4]> = data
            .into_iter()
            .filter(|[min_x, min_y, max_x, max_y]| max_x >= min_x && max_y >= min_y)
            .collect();

        if data.len() != n {
            return Err(ThymeError::BoxesSizeError);
        }

        Ok(Self { data })
    }
}

// >>> I/O METHODS

impl BoundingBoxes {
    /// Open bounding boxes from the provided path
    ///
    /// # Arguments
    ///
    /// * `path` - A path to bounding boxes with a valid extension
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use thyme_core::im::BoundingBoxes;
    /// let bounding_boxes = BoundingBoxes::open("boxes.json");
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<BoundingBoxes, ThymeError> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = extension {
            if ext == "json" {
                return read_boxes_json(path);
            }
        }

        Err(ThymeError::BoxesReadError)
    }

    /// Save bounding boxes at the provided path
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save bounding boxes
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use thyme_core::im::BoundingBoxes;
    /// let boxes = BoundingBoxes::open("boxes.json").unwrap();
    /// boxes.save("boxes.json").unwrap();
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), ThymeError> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = extension {
            if ext == "json" {
                return write_boxes_json(path, &self.data);
            }
        }

        Err(ThymeError::BoxesWriteError)
    }
}

// <<< I/O METHODS

// >>> PROPERTY METHODS

impl BoundingBoxes {
    /// Number of bounding boxes
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if bounding boxes are empty
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }
}

// <<< PROPERTY METHODS

// >>> CONVERSION METHODS

impl BoundingBoxes {
    /// Return a reference to underlying bounding boxes data
    pub fn as_xyxy(&self) -> &Vec<[f32; 4]> {
        &self.data
    }

    /// Return the underlying bounding boxes data
    pub fn to_xyxy(self) -> Vec<[f32; 4]> {
        self.data
    }

    /// Return the bounding box data in xywh format
    pub fn to_xywh(self) -> Vec<[f32; 4]> {
        self.data
            .into_iter()
            .map(|[min_x, min_y, max_x, max_y]| [min_x, min_y, max_x - min_x, max_y - min_y])
            .collect()
    }
}

// <<< CONVERSION METHODS

// >>> TRANSFORM METHODS

impl BoundingBoxes {
    /// Remove bounding boxes based on an array of pre-sorted (ascending) indices
    pub fn remove(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }

        let mut data: Vec<[f32; 4]> = Vec::with_capacity(self.len() - indices.len());
        let mut indices_iter = indices.iter().peekable();
        let mut next_remove = indices_iter.next().copied();

        for (idx, bounding_box) in self.data.iter().enumerate() {
            if Some(idx) == next_remove {
                next_remove = indices_iter.next().copied();
            } else {
                data.push(*bounding_box)
            }
        }

        self.data = data;
    }
}

// <<< TRANSFORM METHODS

/// Read bounding boxes stored as json format
pub fn read_boxes_json<P: AsRef<Path>>(path: P) -> Result<BoundingBoxes, ThymeError> {
    let mut contents = String::new();

    File::open(path)
        .map_err(|err| ThymeError::NoFileError(err.to_string()))?
        .read_to_string(&mut contents)
        .map_err(|err| ThymeError::NoFileError(err.to_string()))?;

    let data: Value = serde_json::from_str(&contents).map_err(|_| ThymeError::BoxesReadError)?;

    fn to_f32(value: &Value) -> Result<f32, ThymeError> {
        if let Some(n) = value.as_f64() {
            Ok(n as f32)
        } else if let Some(n) = value.as_u64() {
            Ok(n as f32)
        } else if let Some(n) = value.as_i64() {
            Ok(n as f32)
        } else {
            Err(ThymeError::BoxesReadError)
        }
    }

    for key in &BOUNDING_BOX_JSON_VALID_KEYS {
        if let Some(boxes) = data.get(key).and_then(|v| v.as_array()) {
            let boxes: Result<Vec<[f32; 4]>, _> = boxes
                .iter()
                .map(|item| {
                    item.as_array()
                        .ok_or(ThymeError::BoxesReadError)
                        .and_then(|b| {
                            if b.len() == 4 {
                                let min_x = to_f32(&b[0])?;
                                let min_y = to_f32(&b[1])?;
                                let max_x = to_f32(&b[2])?;
                                let max_y = to_f32(&b[3])?;
                                Ok([min_x, min_y, max_x, max_y])
                            } else {
                                Err(ThymeError::BoxesReadError)
                            }
                        })
                })
                .collect();

            if let Ok(boxes) = boxes {
                return BoundingBoxes::new(boxes);
            }
        };
    }

    Err(ThymeError::BoxesReadError)
}

/// Write bounding boxes to a json file
pub fn write_boxes_json<P, T>(path: P, boxes: &Vec<[T; 4]>) -> Result<(), ThymeError>
where
    P: AsRef<Path>,
    T: Serialize,
{
    let file = File::create(path).map_err(|_| ThymeError::BoxesWriteError)?;
    let writer = BufWriter::new(file);

    serde_json::to_writer(writer, &serde_json::json!({ "bounding_boxes": boxes }))
        .map_err(|_| ThymeError::BoxesWriteError)?;

    Ok(())
}

#[cfg(test)]
mod test {

    use super::*;

    const TEST_DATA_JSON: &str = "../data/tests/test_boxes.json";

    #[test]
    pub fn test_open_json_success() {
        let bounding_boxes = BoundingBoxes::open(TEST_DATA_JSON);
        assert!(bounding_boxes.is_ok());
    }

    #[test]
    pub fn test_open_json_failure() {
        let bounding_boxes = BoundingBoxes::open("does_not_exist/");
        assert!(bounding_boxes.is_err())
    }

    #[test]
    pub fn test_open_json_format() {
        let bounding_boxes = BoundingBoxes::open(TEST_DATA_JSON).unwrap();

        for (i, bounding_box) in bounding_boxes.as_xyxy().iter().enumerate() {
            assert_eq!(*bounding_box, [0_f32, 0_f32, i as f32, i as f32]);
        }
    }

    #[test]
    pub fn test_write_json() {
        const OUTPUT: &str = "TEST_BOX_WRITE.json";

        let bounding_boxes = BoundingBoxes::open(TEST_DATA_JSON).unwrap();

        bounding_boxes.save(OUTPUT).unwrap();

        let reloaded_boxes = BoundingBoxes::open(OUTPUT).unwrap();

        assert_eq!(bounding_boxes.as_xyxy(), reloaded_boxes.as_xyxy());

        std::fs::remove_file(OUTPUT).unwrap();
    }
}
