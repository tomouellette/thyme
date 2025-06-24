// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::collections::BTreeSet;
use std::path::Path;

use image::{DynamicImage, ImageBuffer, Luma, open as open_dynamic};
use npyz::{self, DType, NpyFile, TypeChar, WriterBuilder};

use crate::constant;
use crate::cv::{connected_components, find_labeled_contours};
use crate::error::ThymeError;
use crate::im::{Polygons, ThymeBuffer, ThymeViewBuffer};

/// A row-major container storing mask pixels
///
/// Masks must have pixels in either u8 or u32 format. By default, we cast
/// u8 masks to u32 type for consistency. The length of the container must
/// be equal to the product of `w` * `h`.
///
/// # Examples
///
/// ```
/// use thyme_core::im::ThymeMask;
///
/// let width = 10;
/// let height = 10;
/// let buffer = vec![0u32; (width * height) as usize];
/// let buffer = ThymeMask::new(width, height, 1, buffer);
///
/// assert_eq!(buffer.unwrap().len(), (width * height) as usize);
/// ```
///
/// ```
/// use thyme_core::im::ThymeMask;
///
/// let width = 10;
/// let height = 10;
/// let buffer = vec![0u32; (width * height * 10) as usize];
/// let buffer = ThymeMask::new(width, height, 1, buffer);
///
/// assert!(buffer.is_err()); // Buffer size does not match dimensions
/// ```
pub type ThymeMask = ThymeBuffer<u32, Vec<u32>>;

// >>> I/O METHODS

impl ThymeMask {
    /// Open a new mask from a provided path
    ///
    /// # Arguments
    ///
    /// * `path` - A path to an image with a valid extension
    ///
    /// ```no_run
    /// use thyme_core::im::ThymeMask;
    /// let image = ThymeMask::open("mask.png");
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<ThymeMask, ThymeError> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = extension {
            if ext == "npy" {
                if let Ok(bytes) = std::fs::read(&path) {
                    if let Ok(npy) = NpyFile::new(&bytes[..]) {
                        return Self::new_from_numpy(npy);
                    }
                }

                return Err(ThymeError::ImageReadError);
            }

            if constant::IMAGE_DYNAMIC_FORMATS.iter().any(|e| e == &ext) {
                if let Ok(image) = open_dynamic(&path) {
                    return Self::new_from_dynamic(image);
                }

                return Err(ThymeError::ImageReadError);
            }
        }

        Err(ThymeError::ImageExtensionError)
    }

    /// Initialize a new mask from a DynamicImage
    ///
    /// # Arguments
    ///
    /// * `image` - An 8 or 16-bit grayscale DynamicImage
    ///
    /// # Examples
    ///
    /// ```
    /// use image::{GrayImage, DynamicImage};
    /// use thyme_core::im::ThymeMask;
    ///
    /// let gray = GrayImage::new(10, 10);
    /// let dynamic = DynamicImage::ImageLuma8(gray);
    /// let image = ThymeMask::new_from_dynamic(dynamic);
    /// ```
    pub fn new_from_dynamic(mask: DynamicImage) -> Result<ThymeMask, ThymeError> {
        let width = mask.width();
        let height = mask.height();

        match mask {
            DynamicImage::ImageLuma8(buffer) => Ok(ThymeMask::new(
                width,
                height,
                1,
                buffer
                    .into_raw()
                    .into_iter()
                    .map(|pixel| pixel as u32)
                    .collect(),
            )?),
            DynamicImage::ImageLumaA8(buffer) => Ok(ThymeMask::new(
                width,
                height,
                1,
                buffer
                    .into_raw()
                    .chunks_exact(2)
                    .map(|pixel| pixel[0] as u32)
                    .collect(),
            )?),
            DynamicImage::ImageLuma16(buffer) => Ok(ThymeMask::new(
                width,
                height,
                1,
                buffer
                    .into_raw()
                    .into_iter()
                    .map(|pixel| pixel as u32)
                    .collect(),
            )?),
            DynamicImage::ImageLumaA16(buffer) => Ok(ThymeMask::new(
                width,
                height,
                1,
                buffer
                    .into_raw()
                    .chunks_exact(2)
                    .map(|pixel| pixel[0] as u32)
                    .collect(),
            )?),
            _ => Err(ThymeError::MaskError(
                "A dynamic image mask with a valid data type was not detected.",
            )),
        }
    }

    /// Initialize a new image from a numpy array buffer
    ///
    /// # Arguments
    ///
    /// * `npy` - A (height, width, channel) shaped numpy array buffer
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use npyz::NpyFile;
    /// use thyme_core::im::ThymeMask;
    ///
    /// let bytes = std::fs::read("mask.npy").unwrap();
    /// let npy = NpyFile::new(&bytes[..]).unwrap();
    /// let image = ThymeMask::new_from_numpy(npy);
    /// ```
    pub fn new_from_numpy(npy: NpyFile<&[u8]>) -> Result<ThymeMask, ThymeError> {
        let shape = npy.shape().to_vec();

        let (h, w, c) = match shape.len() {
            2 => (shape[0] as u32, shape[1] as u32, 1u32),
            3 => (shape[0] as u32, shape[1] as u32, shape[2] as u32),
            _ => {
                return Err(ThymeError::MaskError(
                    "Numpy array masks must have an (H, W) shape.",
                ));
            }
        };

        if c != 1 {
            return Err(ThymeError::MaskFormatError);
        }

        match npy.dtype() {
            DType::Plain(x) => match (x.type_char(), x.size_field()) {
                (TypeChar::Uint, 1) => Ok(ThymeMask::new(
                    w,
                    h,
                    1,
                    npy.into_vec()
                        .unwrap()
                        .into_iter()
                        .map(|pixel: u8| pixel as u32)
                        .collect(),
                )?),
                (TypeChar::Uint, 2) => Ok(ThymeMask::new(
                    w,
                    h,
                    1,
                    npy.into_vec()
                        .unwrap()
                        .into_iter()
                        .map(|pixel: u16| pixel as u32)
                        .collect(),
                )?),
                (TypeChar::Uint, 4) => Ok(ThymeMask::new(w, h, 1, npy.into_vec().unwrap())?),
                _ => Err(ThymeError::MaskError(
                    "A numpy mask array with a valid data type was not detected.",
                )),
            },
            _ => Err(ThymeError::MaskError(
                "Only plain numpy mask arrays are currentled supported.",
            )),
        }
    }
}

// <<< I/O METHODS

// >>> TRANSFORM METHODS

impl ThymeMask {
    /// Re-label the mask using connected components and return unique labels
    ///
    /// # Notes
    ///
    /// Re-labelling is guaranteed to assign the correct number of labels when
    /// assuming 8-connectivity. However, the labels are not guaranteed to be
    /// incremental (e.g. 1, 2, 3, ..). This should be taken into account when
    /// iterating over objects.
    pub fn label(&mut self) -> Vec<u32> {
        let mut labels: Vec<u32> = self
            .as_raw()
            .iter()
            .filter(|&&x| x != 0)
            .cloned()
            .collect::<BTreeSet<u32>>()
            .into_iter()
            .collect();

        // Currently, we only re-label binary masks and assume any mask
        // with more than one unique label is an integer-labeled mask.
        if labels.len() == 1 {
            self.buffer = connected_components(self.width(), self.height(), &self.buffer);
            labels = self
                .as_raw()
                .iter()
                .filter(|&&x| x != 0)
                .cloned()
                .collect::<BTreeSet<u32>>()
                .into_iter()
                .collect();
        }

        labels
    }

    /// Extract polygons from a segmentation mask
    pub fn polygons(&mut self) -> Result<(Vec<u32>, Polygons), ThymeError> {
        let labels = self.label();
        let (labels, contours) =
            find_labeled_contours(self.width(), self.height(), &self.buffer, &labels);

        Ok((labels, Polygons::new(contours)?))
    }

    /// Crops image while only including pixels with a specified label
    ///
    /// # Arguments
    ///
    /// * `x` - Minimum x-coordinate (left)
    /// * `y` - Minimum y-coordinate (bottom)
    /// * `w` - Width of crop
    /// * `h` - Height of crop
    /// * `label` - Only include mask pixels equal to this label
    pub fn crop_binary(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        label: u32,
    ) -> Result<ThymeMask, ThymeError> {
        if x + w > self.width() || y + h > self.height() {
            return Err(ThymeError::MaskError("Cropping coordinates out of bounds"));
        }

        let c = self.channels() as usize;
        let orig_w = self.width() as usize;
        let orig_buffer: &[u32] = self.buffer.as_ref();

        let mut new_buffer = Vec::with_capacity((w * h * self.channels()) as usize);

        for row in y..y + h {
            let start = ((row as usize) * orig_w + (x as usize)) * c;
            let end = start + (w as usize) * c;

            new_buffer.extend(
                orig_buffer[start..end]
                    .iter()
                    .map(|&v| if v == label { 1 } else { 0 }),
            );
        }

        ThymeMask::new(w, h, self.channels(), new_buffer)
    }
}

// <<< TRANSFORM METHODS

/// A type for mask object buffer
pub type ThymeMaskView<'a> = ThymeViewBuffer<'a, u32, Vec<u32>>;

// I/O METHODS

impl<'a> ThymeMaskView<'a> {
    /// Save an object
    ///
    /// # Arguments
    ///
    /// * `path` - A path to an image with a valid extension
    ///
    /// ```no_run
    /// use thyme_core::im::ThymeImage;
    /// let image = ThymeImage::open("image.png");
    /// ```
    pub fn save<P: AsRef<Path>>(&'a self, path: P, label: &u32) -> Result<(), ThymeError> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = extension {
            if ext == "npy" {
                let mut buffer = vec![];
                let mut writer = npyz::WriteOptions::<u8>::new()
                    .default_dtype()
                    .shape(&[self.height() as u64, self.width() as u64])
                    .writer(&mut buffer)
                    .begin_nd()
                    .map_err(|_| ThymeError::ImageWriteError)?;

                for d in self.iter() {
                    if d == label {
                        writer.push(&255u8).unwrap();
                    } else {
                        writer.push(&0u8).unwrap();
                    };
                }

                writer.finish().map_err(|_| ThymeError::ImageWriteError)?;
                std::fs::write(&path, buffer).map_err(|_| ThymeError::ImageWriteError)?;

                return Ok(());
            }

            if constant::IMAGE_DYNAMIC_FORMATS.iter().any(|e| e == &ext) {
                ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                    self.width() as u32,
                    self.height() as u32,
                    self.iter()
                        .map(|p| if p == label { 255u8 } else { 0u8 })
                        .collect(),
                )
                .unwrap()
                .save(path)
                .map_err(|_| ThymeError::ImageWriteError)?;

                return Ok(());
            }
        }

        Err(ThymeError::ImageExtensionError)
    }
}

// <<< I/O METHODS

/// Type of masking style to use
pub enum MaskingStyle {
    Foreground,
    Background,
}

#[cfg(test)]
mod test {

    use super::*;

    const TEST_MASK: &str = "../data/tests/test_mask";
    const TEST_BLOB: &str = "../data/tests/test_mask_binary_blobs.png";

    #[test]
    fn test_mask_open() {
        let extensions = [
            "_binary.png",
            "_binary_1.npy",
            "_binary_1_u16.npy",
            "_binary_255.npy",
            "_binary_255_u16.npy",
            "_integer.png",
            "_integer.npy",
            "_integer_u16.npy",
        ];

        for ext in extensions.into_iter() {
            let img = ThymeMask::open(format!("{}{}", TEST_MASK, ext));
            assert!(img.is_ok(), "{}", ext);

            let img = img.unwrap();
            assert_eq!(img.width(), 621);
            assert_eq!(img.height(), 621);
            assert_eq!(img.channels(), 1, "{}", ext);
        }
    }

    #[test]
    fn test_mask_save() {
        const TEST_DEFAULT: &str = "TEST_SAVE_DEFAULT_MASK.png";
        const TEST_NUMPY: &str = "TEST_SAVE_NUMPY_MASK.npy";

        let mask = ThymeMask::new(2, 2, 1, vec![0, 255, 0, 0]).unwrap();

        mask.crop_view(0, 0, 2, 2).save(TEST_DEFAULT, &255).unwrap();
        mask.crop_view(0, 0, 2, 2).save(TEST_NUMPY, &255).unwrap();

        let mask_default = ThymeMask::open(TEST_DEFAULT).unwrap();
        let mask_numpy = ThymeMask::open(TEST_NUMPY).unwrap();

        assert_eq!(mask.as_raw(), mask_default.as_raw());
        assert_eq!(mask.as_raw(), mask_numpy.as_raw());

        std::fs::remove_file(TEST_DEFAULT).unwrap();
        std::fs::remove_file(TEST_NUMPY).unwrap();
    }

    #[test]
    fn test_label_blob() {
        let mut mask = ThymeMask::open(TEST_BLOB).unwrap();
        let labels = mask.label();
        assert_eq!(labels.len(), 11);
    }

    #[test]
    fn test_mask_crop() {
        let width = 10;
        let height = 10;
        let data: Vec<u32> = (0..width * height).collect();

        let buffer = ThymeMask::new(width, height, 1, data).unwrap();
        let crop = buffer.crop_view(0, 0, 10, 1);

        for (i, col) in crop.iter().enumerate() {
            assert_eq!(col, &(i as u32));
        }
    }

    #[test]
    fn test_mask_label() {
        let width = 10;
        let height = 10;
        let mut data: Vec<u32> = vec![0u32; 100];

        data[5] = 1u32;
        data[25] = 1u32;
        data[45] = 1u32;
        data[65] = 1u32;
        data[85] = 1u32;

        let mut buffer = ThymeMask::new(width, height as u32, 1, data).unwrap();

        let labels = buffer.label();
        assert_eq!(labels.len(), 5);

        assert_eq!(labels[0], 1);
        assert_eq!(labels[1], 2);
        assert_eq!(labels[2], 3);
        assert_eq!(labels[3], 4);
        assert_eq!(labels[4], 5);
    }

    #[test]
    fn test_mask_crop_binary() {
        let width = 2;
        let height = 2;
        let data: Vec<u32> = vec![0, 1, 2, 3];

        let buffer = ThymeMask::new(width, height, 1, data).unwrap();

        let binary = buffer.crop_binary(0, 0, 2, 2, 1).unwrap();

        assert_eq!(binary.as_raw(), &[0, 1, 0, 0]);
    }
}
