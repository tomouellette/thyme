// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::constant::POLYGON_JSON_VALID_KEYS;
use crate::cv::points::{dedup_points, order_points, resample_points};
use crate::error::ThymeError;
use crate::im::boxes::BoundingBoxes;
use crate::mp::form;

/// A polygon container for storing object outlines
///
/// The polygons are stored in (N, 2, K) format where N is the
/// number of polygons for a given mask/image, 2 is xy, and K
/// specifies the number of points in each polygon. Note that
/// the polygons can be ragged so K can vary for each polygon.
///
/// # Examples
///
/// ```
/// use thyme_core::im::Polygons;
///
/// let data: Vec<Vec<[f32; 2]>> = vec![
///     vec![[0., 1.], [1., 1.], [1., 2.], [0., 2.]],
///     vec![[1., 1.], [2., 1.], [2., 2.], [1., 2.]],
/// ];
///
/// let polygons = Polygons::new(data);
/// assert!(polygons.is_ok());
///
/// let data: Vec<Vec<[f32; 2]>> = vec![
///     vec![[0., 1.], [1., 1.]],
///     vec![[1., 1.], [2., 1.]],
/// ];
///
/// let polygons = Polygons::new(data);
/// assert!(polygons.is_err());
/// ```
#[derive(Debug, Clone)]
pub struct Polygons {
    data: Vec<Vec<[f32; 2]>>,
    deduped: bool,
    ordered: bool,
}

impl Polygons {
    /// Initialize a new polygons container
    ///
    /// # Arguments
    ///
    /// * `data` - Polygons in (N, 2, K) format
    ///
    /// # Examples
    ///
    /// ```
    /// use thyme_core::im::Polygons;
    ///
    /// let data: Vec<Vec<[f32; 2]>> = vec![
    ///     vec![[0., 1.], [1., 1.], [1., 2.]],
    ///     vec![[1., 1.], [2., 1.], [2., 2.]],
    /// ];
    ///
    /// let polygons = Polygons::new(data);
    /// ```
    pub fn new(data: Vec<Vec<[f32; 2]>>) -> Result<Self, ThymeError> {
        let n = data.len();

        let data: Vec<Vec<[f32; 2]>> = data
            .into_iter()
            .filter(|polygon| polygon.len() > 2)
            .collect();

        if data.len() != n {
            return Err(ThymeError::PolygonsSizeError);
        }

        Ok(Self {
            data,
            deduped: false,
            ordered: false,
        })
    }
}

// >>> I/O METHODS

impl Polygons {
    /// Open polygons from the provided path
    ///
    /// # Arguments
    ///
    /// * `path` - A path to polygons with a valid extension
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use thyme_core::im::Polygons;
    /// let polygons = Polygons::open("polygons.json");
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Polygons, ThymeError> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = extension {
            if ext == "json" {
                return read_polygons_json(path);
            }
        }

        Err(ThymeError::PolygonsReadError)
    }

    /// Save a polygons at the provided paath
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save polygons
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use thyme_core::im::Polygons;
    /// let polygons = Polygons::open("polygons.json").unwrap();
    /// polygons.save("polygons.json").unwrap();
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), ThymeError> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = extension {
            if ext == "json" {
                return write_polygons_json(path, &self.data);
            }
        }

        Err(ThymeError::PolygonsWriteError)
    }
}

// <<< I/O METHODS

// >>> PROPERTY METHODS
impl Polygons {
    /// Return the number of stored polygons
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if polygon has no data
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }
}

// <<< PROPERTY METHODS

// >>> CONVERSION METHODS

impl Polygons {
    /// Return a reference to the underlying polygon point
    pub fn as_points(&self) -> &Vec<Vec<[f32; 2]>> {
        &self.data
    }

    /// Return the underlying polygon point
    pub fn to_points(self) -> Vec<Vec<[f32; 2]>> {
        self.data
    }

    /// Convert the polygons to bounding boxes
    pub fn to_bounding_boxes(&self) -> Result<BoundingBoxes, ThymeError> {
        BoundingBoxes::new(
            self.data
                .iter()
                .map(|polygon| {
                    let &[fx, fy] = &polygon[0];

                    let mut min_x = fx;
                    let mut min_y = fy;
                    let mut max_x = fx;
                    let mut max_y = fy;

                    for &[x, y] in polygon {
                        min_x = min_x.min(x);
                        min_y = min_y.min(y);
                        max_x = max_x.max(x);
                        max_y = max_y.max(y);
                    }

                    [min_x, min_y, max_x, max_y]
                })
                .collect::<Vec<[f32; 4]>>(),
        )
    }
}

// <<< CONVERSION METHODS

// >>> TRANSFORM METHODS

impl Polygons {
    /// Deduplicate redundant points in each polygon
    pub fn dedup_points(&mut self) {
        if !self.deduped {
            self.data.iter_mut().for_each(dedup_points);
            self.deduped = true;
        }
    }

    /// Order the points in each polygon
    pub fn order_points(&mut self) {
        if !self.ordered {
            self.data
                .iter_mut()
                .for_each(|polygon| order_points(polygon));

            self.ordered = true;
        }
    }

    /// Resample each polygon to an equal number of equidistant points
    pub fn resample_points(&mut self, n: usize) {
        self.dedup_points();
        self.order_points();
        self.data
            .iter_mut()
            .for_each(|polygon| resample_points(polygon, n));
    }

    /// Remove polygons based on an array of pre-sorted (ascending) indices
    pub fn remove(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }

        let mut data: Vec<Vec<[f32; 2]>> = Vec::with_capacity(self.len() - indices.len());
        let mut indices_iter = indices.iter().peekable();
        let mut next_remove = indices_iter.next().copied();

        for (idx, polygon) in self.data.iter().enumerate() {
            if Some(idx) == next_remove {
                next_remove = indices_iter.next().copied();
            } else {
                data.push(polygon.to_vec());
            }
        }

        self.data = data;
    }

    /// Compute morphological measurements from polygons
    pub fn descriptors(&mut self) -> Vec<[f32; 23]> {
        if !self.deduped {
            self.dedup_points();
            self.deduped = true;
        }

        if !self.ordered {
            self.order_points();
            self.ordered = true;
        }

        self.data
            .iter()
            .map(|points| form::descriptors(points))
            .collect()
    }
}

// <<< TRANSFORM METHODS

/// Read polygons stored as json format
pub fn read_polygons_json<P: AsRef<Path>>(path: P) -> Result<Polygons, ThymeError> {
    let mut contents = String::new();

    File::open(path)
        .map_err(|err| ThymeError::NoFileError(err.to_string()))?
        .read_to_string(&mut contents)
        .map_err(|err| ThymeError::NoFileError(err.to_string()))?;

    let data: Value = serde_json::from_str(&contents).map_err(|_| ThymeError::PolygonsReadError)?;

    fn to_f32(value: &Value) -> Result<f32, ThymeError> {
        if let Some(n) = value.as_f64() {
            Ok(n as f32)
        } else if let Some(n) = value.as_u64() {
            Ok(n as f32)
        } else if let Some(n) = value.as_i64() {
            Ok(n as f32)
        } else {
            Err(ThymeError::PolygonsReadError)
        }
    }

    for key in &POLYGON_JSON_VALID_KEYS {
        if let Some(polygons) = data.get(key).and_then(|v| v.as_array()) {
            let polygons: Result<Vec<Vec<[f32; 2]>>, _> = polygons
                .iter()
                .filter_map(Value::as_array)
                .map(|polygon| {
                    polygon
                        .iter()
                        .filter_map(Value::as_array)
                        .map(|p| {
                            if p.len() == 2 {
                                let x = to_f32(&p[0])?;
                                let y = to_f32(&p[1])?;
                                Ok([x, y])
                            } else {
                                Err(ThymeError::PolygonsReadError)
                            }
                        })
                        .collect::<Result<Vec<[f32; 2]>, _>>()
                })
                .collect();

            if let Ok(polygons) = polygons {
                return Polygons::new(polygons);
            }
        }
    }

    Err(ThymeError::PolygonsReadError)
}

/// Write polygons to a json file
pub fn write_polygons_json<P, T>(path: P, polygons: &[Vec<[T; 2]>]) -> Result<(), ThymeError>
where
    P: AsRef<Path>,
    T: Serialize,
{
    let file = File::create(path).map_err(|_| ThymeError::PolygonsWriteError)?;
    let writer = BufWriter::new(file);

    serde_json::to_writer(writer, &serde_json::json!({ "polygons": polygons }))
        .map_err(|_| ThymeError::PolygonsWriteError)?;

    Ok(())
}

#[cfg(test)]
mod test {

    use super::*;

    const TEST_DATA_JSON: &str = "../data/tests/test_polygons.json";

    #[test]
    pub fn test_open_json_success() {
        let polygons = Polygons::open(TEST_DATA_JSON);
        polygons.clone().unwrap();
        assert!(polygons.is_ok());
    }

    #[test]
    pub fn test_open_json_failure() {
        let polygons = Polygons::open("does_not_exist/");
        assert!(polygons.is_err())
    }

    #[test]
    pub fn test_open_json_count() {
        let polygons = Polygons::open(TEST_DATA_JSON).unwrap();
        assert_eq!(polygons.len(), 2);
        assert!(polygons.as_points()[0].len() > 2);
        assert_eq!(polygons.as_points()[0][0].len(), 2);
    }

    #[test]
    pub fn test_write_json() {
        const OUTPUT: &str = "TEST_POLYGONS_WRITE.json";

        let polygons = Polygons::open(TEST_DATA_JSON).unwrap();

        polygons.save(OUTPUT).unwrap();

        let polygons = Polygons::open(OUTPUT).unwrap();

        assert_eq!(polygons.as_points(), polygons.as_points());

        std::fs::remove_file(OUTPUT).unwrap();
    }
}
