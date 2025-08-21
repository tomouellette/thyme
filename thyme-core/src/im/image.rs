// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::path::Path;

use fast_image_resize::PixelType;
use image::{DynamicImage, ImageBuffer, Luma, Rgb, open as open_dynamic};
use npyz::{self, DType, NpyFile, TypeChar};

use crate::constant;
use crate::cv::transform;
use crate::error::ThymeError;
use crate::im::{MaskingStyle, ThymeBuffer, ThymeMaskView, ThymeView};
use crate::impl_enum_dispatch;
use crate::io::write_numpy;

/// A wrapper for representing and storing array-shaped pixels
///
/// The enum holds all valid, or potentially valid, image formats in terms
/// of their subpixel data types. All external image types (e.g `DynamicImage`)
/// should be converted to a ThymeImage via a method on this enum.
///
/// # Examples
///
/// ```
/// use image::{RgbImage, DynamicImage};
/// use thyme_core::im::ThymeImage;
///
/// let rgb = RgbImage::new(10, 10);
/// let dynamic = DynamicImage::ImageRgb8(rgb);
/// let image = ThymeImage::new_from_default(dynamic);
/// ```
///
/// ```no_run
/// use image::ImageReader;
/// use thyme_core::im::ThymeImage;
///
/// let dynamic = ImageReader::open("image.png")
///     .expect("Failed to read image")
///     .with_guessed_format()
///     .unwrap()
///     .decode()
///     .unwrap();
///
/// let image = ThymeImage::new_from_default(dynamic);
/// ```
#[derive(Debug, Clone)]
pub enum ThymeImage {
    U8(ThymeBuffer<u8, Vec<u8>>),
    U16(ThymeBuffer<u16, Vec<u16>>),
    U32(ThymeBuffer<u32, Vec<u32>>),
    U64(ThymeBuffer<u64, Vec<u64>>),
    I32(ThymeBuffer<i32, Vec<i32>>),
    I64(ThymeBuffer<i64, Vec<i64>>),
    F32(ThymeBuffer<f32, Vec<f32>>),
    F64(ThymeBuffer<f64, Vec<f64>>),
}

// >>> I/O METHODS

impl ThymeImage {
    /// Open a new image from a provided path
    ///
    /// # Arguments
    ///
    /// * `path` - A path to an image with a valid extension
    ///
    /// ```no_run
    /// use thyme_core::im::ThymeImage;
    /// let image = ThymeImage::open("image.png");
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<ThymeImage, ThymeError> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = extension {
            if ext == "npy" {
                if let Ok(bytes) = std::fs::read(&path) {
                    if let Ok(npy) = NpyFile::new(&bytes[..]) {
                        Self::new_from_numpy(npy.clone()).unwrap();

                        return Self::new_from_numpy(npy);
                    }
                }

                return Err(ThymeError::ImageReadError);
            }

            if constant::IMAGE_DYNAMIC_FORMATS.iter().any(|e| e == &ext) {
                if let Ok(image) = open_dynamic(&path) {
                    return Self::new_from_default(image);
                }

                return Err(ThymeError::ImageReadError);
            }
        }

        Err(ThymeError::ImageExtensionError)
    }

    /// Initialize a new image from a DynamicImage
    ///
    /// # Arguments
    ///
    /// * `image` - An 8 or 16-bit grayscale or rgb DynamicImage
    ///
    /// # Examples
    ///
    /// ```
    /// use image::{GrayImage, DynamicImage};
    /// use thyme_core::im::ThymeImage;
    ///
    /// let gray = GrayImage::new(10, 10);
    /// let dynamic = DynamicImage::ImageLuma8(gray);
    /// let image = ThymeImage::new_from_default(dynamic);
    /// ```
    pub fn new_from_default(image: DynamicImage) -> Result<ThymeImage, ThymeError> {
        let width = image.width();
        let height = image.height();

        match image {
            DynamicImage::ImageLuma8(buffer) => Ok(ThymeImage::U8(ThymeBuffer::new(
                width,
                height,
                1,
                buffer.into_raw(),
            )?)),
            DynamicImage::ImageLumaA8(buffer) => Ok(ThymeImage::U8(ThymeBuffer::new(
                width,
                height,
                1,
                buffer
                    .into_raw()
                    .chunks_exact(2)
                    .map(|pixel| pixel[0])
                    .collect(),
            )?)),
            DynamicImage::ImageLuma16(buffer) => Ok(ThymeImage::U16(ThymeBuffer::new(
                width,
                height,
                1,
                buffer.into_raw(),
            )?)),
            DynamicImage::ImageLumaA16(buffer) => Ok(ThymeImage::U16(ThymeBuffer::new(
                width,
                height,
                1,
                buffer
                    .into_raw()
                    .chunks_exact(2)
                    .map(|pixel| pixel[0])
                    .collect(),
            )?)),
            DynamicImage::ImageRgb8(buffer) => Ok(ThymeImage::U8(ThymeBuffer::new(
                width,
                height,
                3,
                buffer.into_raw(),
            )?)),
            DynamicImage::ImageRgba8(buffer) => Ok(ThymeImage::U8(ThymeBuffer::new(
                width,
                height,
                3,
                buffer
                    .into_raw()
                    .chunks_exact(4)
                    .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                    .collect(),
            )?)),
            DynamicImage::ImageRgb16(buffer) => Ok(ThymeImage::U16(ThymeBuffer::new(
                width,
                height,
                3,
                buffer.into_raw(),
            )?)),
            DynamicImage::ImageRgba16(buffer) => Ok(ThymeImage::U16(ThymeBuffer::new(
                width,
                height,
                3,
                buffer
                    .into_raw()
                    .chunks_exact(4)
                    .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                    .collect(),
            )?)),
            DynamicImage::ImageRgb32F(buffer) => Ok(ThymeImage::F32(ThymeBuffer::new(
                width,
                height,
                3,
                buffer.into_raw(),
            )?)),
            DynamicImage::ImageRgba32F(buffer) => Ok(ThymeImage::F32(ThymeBuffer::new(
                width,
                height,
                3,
                buffer
                    .into_raw()
                    .chunks_exact(4)
                    .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                    .collect(),
            )?)),
            _ => Err(ThymeError::ImageError(
                "A dynamic image with a valid data type was not detected.",
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
    /// use thyme_core::im::ThymeImage;
    ///
    /// let bytes = std::fs::read("image.npy").unwrap();
    /// let npy = NpyFile::new(&bytes[..]).unwrap();
    /// let image = ThymeImage::new_from_numpy(npy);
    /// ```
    pub fn new_from_numpy(npy: NpyFile<&[u8]>) -> Result<ThymeImage, ThymeError> {
        let shape = npy.shape().to_vec();

        let (h, w, c) = match shape.len() {
            2 => (shape[0] as u32, shape[1] as u32, 1u32),
            3 => (shape[0] as u32, shape[1] as u32, shape[2] as u32),
            _ => {
                return Err(ThymeError::ImageError(
                    "Numpy array inputs must have an (H, W) or (H, W, C) shape.",
                ));
            }
        };

        match npy.dtype() {
            DType::Plain(x) => match (x.type_char(), x.size_field()) {
                (TypeChar::Uint, 1) => Ok(ThymeImage::U8(ThymeBuffer::new(
                    w,
                    h,
                    c,
                    npy.into_vec().unwrap(),
                )?)),
                (TypeChar::Uint, 2) => Ok(ThymeImage::U16(ThymeBuffer::new(
                    w,
                    h,
                    c,
                    npy.into_vec().unwrap(),
                )?)),
                (TypeChar::Int, 4) => Ok(ThymeImage::I32(ThymeBuffer::new(
                    w,
                    h,
                    c,
                    npy.into_vec().unwrap(),
                )?)),
                (TypeChar::Int, 8) => Ok(ThymeImage::I64(ThymeBuffer::new(
                    w,
                    h,
                    c,
                    npy.into_vec().unwrap(),
                )?)),
                (TypeChar::Float, 4) => Ok(ThymeImage::F32(ThymeBuffer::new(
                    w,
                    h,
                    c,
                    npy.into_vec().unwrap(),
                )?)),
                (TypeChar::Float, 8) => Ok(ThymeImage::F64(ThymeBuffer::new(
                    w,
                    h,
                    c,
                    npy.into_vec().unwrap(),
                )?)),
                _ => Err(ThymeError::ImageError(
                    "A numpy array with a valid data type was not detected.",
                )),
            },
            _ => Err(ThymeError::ImageError(
                "Only plain numpy arrays are currentled supported.",
            )),
        }
    }

    /// Save image
    ///
    /// # Arguments
    ///
    /// * `path` - A path to an image with a valid extension
    ///
    /// ```no_run
    /// use thyme_core::im::ThymeImage;
    /// let image = ThymeImage::open("image.png").unwrap();
    /// image.save("image.npy").unwrap();
    /// ```
    pub fn save<P: AsRef<Path>>(self, path: P) -> Result<(), ThymeError> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if let Some(ext) = extension {
            if ext == "npy" {
                return self.save_as_numpy(path);
            }

            if constant::IMAGE_DYNAMIC_FORMATS.iter().any(|e| e == &ext) {
                return self.save_as_default(path);
            }
        }

        Err(ThymeError::ImageExtensionError)
    }

    /// Save image as a default image format
    ///
    /// # Arguments
    ///
    /// * `path` - Path to output image
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use image::{GrayImage, DynamicImage};
    /// use thyme_core::im::ThymeImage;
    ///
    /// let gray = GrayImage::new(10, 10);
    /// let dynamic = DynamicImage::ImageLuma8(gray);
    /// let image = ThymeImage::new_from_default(dynamic).unwrap();
    /// image.save_as_default("new_image.png").unwrap();
    /// ```
    pub fn save_as_default<P: AsRef<Path>>(self, path: P) -> Result<(), ThymeError> {
        let channels = self.channels();
        match (self, channels) {
            (ThymeImage::U8(buffer), 1) => {
                let image_buffer = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                    buffer.width(),
                    buffer.height(),
                    buffer.into_raw(),
                )
                .ok_or(ThymeError::ImageWriteError)?;

                image_buffer
                    .save(path)
                    .map_err(|_| ThymeError::ImageWriteError)
            }
            (ThymeImage::U16(buffer), 1) => {
                let image_buffer = ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(
                    buffer.width(),
                    buffer.height(),
                    buffer.into_raw(),
                )
                .ok_or(ThymeError::ImageWriteError)?;

                image_buffer
                    .save(path)
                    .map_err(|_| ThymeError::ImageWriteError)
            }
            (ThymeImage::U8(buffer), 3) => {
                let image_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                    buffer.width(),
                    buffer.height(),
                    buffer.into_raw(),
                )
                .ok_or(ThymeError::ImageWriteError)?;

                image_buffer
                    .save(path)
                    .map_err(|_| ThymeError::ImageWriteError)
            }
            (ThymeImage::U16(buffer), 3) => {
                let image_buffer = ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(
                    buffer.width(),
                    buffer.height(),
                    buffer.into_raw(),
                )
                .ok_or(ThymeError::ImageWriteError)?;

                image_buffer
                    .save(path)
                    .map_err(|_| ThymeError::ImageWriteError)
            }
            (ThymeImage::F32(buffer), 3) => {
                let image_buffer = ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(
                    buffer.width(),
                    buffer.height(),
                    buffer.into_raw(),
                )
                .ok_or(ThymeError::ImageWriteError)?;

                image_buffer
                    .save(path)
                    .map_err(|_| ThymeError::ImageWriteError)
            }
            _ => Err(ThymeError::ImageError(
                "Only 1 or 3 channel RGB/grayscale images can be saved as a default image format (e.g. png).",
            )),
        }
    }

    /// Save image as a numpy format
    ///
    /// # Arguments
    ///
    /// * `path` - Path to output image
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use image::{GrayImage, DynamicImage};
    /// use thyme_core::im::ThymeImage;
    ///
    /// let gray = GrayImage::new(10, 10);
    /// let dynamic = DynamicImage::ImageLuma8(gray);
    /// let image = ThymeImage::new_from_default(dynamic).unwrap();
    /// image.save_as_numpy("new_image.npy").unwrap();
    /// ```
    pub fn save_as_numpy<P: AsRef<Path>>(self, path: P) -> Result<(), ThymeError> {
        let shape = vec![
            self.height() as u64,
            self.width() as u64,
            self.channels() as u64,
        ];

        match self {
            ThymeImage::U8(buffer) => write_numpy(path.as_ref(), buffer.into_raw(), shape),
            ThymeImage::U16(buffer) => write_numpy(path.as_ref(), buffer.into_raw(), shape),
            ThymeImage::U32(buffer) => write_numpy(path.as_ref(), buffer.into_raw(), shape),
            ThymeImage::U64(buffer) => write_numpy(path.as_ref(), buffer.into_raw(), shape),
            ThymeImage::I32(buffer) => write_numpy(path.as_ref(), buffer.into_raw(), shape),
            ThymeImage::I64(buffer) => write_numpy(path.as_ref(), buffer.into_raw(), shape),
            ThymeImage::F32(buffer) => write_numpy(path.as_ref(), buffer.into_raw(), shape),
            ThymeImage::F64(buffer) => write_numpy(path.as_ref(), buffer.into_raw(), shape),
        }
    }
}

// <<< I/O METHODS

// >>> PROPERTY METHODS

impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; width(&self) -> u32);
impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; height(&self) -> u32);
impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; channels(&self) -> u32);
impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; shape(&self) -> (u32, u32, u32));
impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; len(&self) -> usize);
impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; is_empty(&self) -> bool);

impl ThymeImage {
    /// Get the minimum value for the image data type
    pub fn dtype_min(&self) -> f64 {
        match self {
            ThymeImage::U8(_) => u8::MIN as f64,
            ThymeImage::U16(_) => u16::MIN as f64,
            ThymeImage::U32(_) => u32::MIN as f64,
            ThymeImage::U64(_) => u64::MIN as f64,
            ThymeImage::I32(_) => i32::MIN as f64,
            ThymeImage::I64(_) => i64::MIN as f64,
            ThymeImage::F32(_) => f32::MIN as f64,
            ThymeImage::F64(_) => f64::MIN,
        }
    }

    /// Get the maximum value for the image data type
    pub fn dtype_max(&self) -> f64 {
        match self {
            ThymeImage::U8(_) => u8::MAX as f64,
            ThymeImage::U16(_) => u16::MAX as f64,
            ThymeImage::U32(_) => u32::MAX as f64,
            ThymeImage::U64(_) => u64::MAX as f64,
            ThymeImage::I32(_) => i32::MAX as f64,
            ThymeImage::I64(_) => i64::MAX as f64,
            ThymeImage::F32(_) => f32::MAX as f64,
            ThymeImage::F64(_) => f64::MAX,
        }
    }
}

// <<< PROPERTY METHODS

// >>> CONVERSION METHODS

impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; to_u8(&self) -> Vec<u8>);
impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; to_u16(&self) -> Vec<u16>);
impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; to_u32(&self) -> Vec<u32>);
impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; to_f32(&self) -> Vec<f32>);
impl_enum_dispatch!(ThymeImage, U8, U16, U32, I32, I64, U64, F32, F64; to_f64(&self) -> Vec<f64>);

// <<< CONVERSION METHODS

// >>> TRANSFORM METHODS

impl ThymeImage {
    /// Generate a zero-copy crop of an image subregion
    ///
    /// # Arguments
    ///
    /// * `x` - Minimum x-coordinate (left)
    /// * `y` - Minimum y-coordinate (bottom)
    /// * `w` - Width of crop
    /// * `h` - Height of crop
    pub fn crop_view(&self, x: u32, y: u32, w: u32, h: u32) -> ThymeView {
        match self {
            ThymeImage::U8(buffer) => ThymeView::U8(buffer.crop_view(x, y, w, h)),
            ThymeImage::U16(buffer) => ThymeView::U16(buffer.crop_view(x, y, w, h)),
            ThymeImage::U32(buffer) => ThymeView::U32(buffer.crop_view(x, y, w, h)),
            ThymeImage::U64(buffer) => ThymeView::U64(buffer.crop_view(x, y, w, h)),
            ThymeImage::I32(buffer) => ThymeView::I32(buffer.crop_view(x, y, w, h)),
            ThymeImage::I64(buffer) => ThymeView::I64(buffer.crop_view(x, y, w, h)),
            ThymeImage::F32(buffer) => ThymeView::F32(buffer.crop_view(x, y, w, h)),
            ThymeImage::F64(buffer) => ThymeView::F64(buffer.crop_view(x, y, w, h)),
        }
    }

    /// Create a new image with copied cropped contents
    ///
    /// # Arguments
    ///
    /// * `x` - Minimum x-coordinate (left)
    /// * `y` - Minimum y-coordinate (bottom)
    /// * `w` - Width of crop
    /// * `h` - Height of crop
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Result<ThymeImage, ThymeError> {
        match self {
            ThymeImage::U8(buffer) => Ok(ThymeImage::U8(buffer.crop(x, y, w, h)?)),
            ThymeImage::U16(buffer) => Ok(ThymeImage::U16(buffer.crop(x, y, w, h)?)),
            ThymeImage::U32(buffer) => Ok(ThymeImage::U32(buffer.crop(x, y, w, h)?)),
            ThymeImage::U64(buffer) => Ok(ThymeImage::U64(buffer.crop(x, y, w, h)?)),
            ThymeImage::I32(buffer) => Ok(ThymeImage::I32(buffer.crop(x, y, w, h)?)),
            ThymeImage::I64(buffer) => Ok(ThymeImage::I64(buffer.crop(x, y, w, h)?)),
            ThymeImage::F32(buffer) => Ok(ThymeImage::F32(buffer.crop(x, y, w, h)?)),
            ThymeImage::F64(buffer) => Ok(ThymeImage::F64(buffer.crop(x, y, w, h)?)),
        }
    }

    /// Crops the image while applying a mask to either foreground or background pixels
    ///
    /// # Arguments
    ///
    /// * `x` - Minimum x-coordinate (left)
    /// * `y` - Minimum y-coordinate (bottom)
    /// * `w` - Width of crop
    /// * `h` - Height of crop
    /// * `mask` - A cropped mask view
    /// * `mask_style` - Foreground or background masking style
    pub fn crop_masked(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        mask: &ThymeMaskView,
        mask_style: MaskingStyle,
    ) -> Result<ThymeImage, ThymeError> {
        match self {
            ThymeImage::U8(buffer) => Ok(ThymeImage::U8(
                buffer.crop_masked(x, y, w, h, mask, mask_style)?,
            )),
            ThymeImage::U16(buffer) => Ok(ThymeImage::U16(
                buffer.crop_masked(x, y, w, h, mask, mask_style)?,
            )),
            ThymeImage::U32(buffer) => Ok(ThymeImage::U32(
                buffer.crop_masked(x, y, w, h, mask, mask_style)?,
            )),
            ThymeImage::U64(buffer) => Ok(ThymeImage::U64(
                buffer.crop_masked(x, y, w, h, mask, mask_style)?,
            )),
            ThymeImage::I32(buffer) => Ok(ThymeImage::I32(
                buffer.crop_masked(x, y, w, h, mask, mask_style)?,
            )),
            ThymeImage::I64(buffer) => Ok(ThymeImage::I64(
                buffer.crop_masked(x, y, w, h, mask, mask_style)?,
            )),
            ThymeImage::F32(buffer) => Ok(ThymeImage::F32(
                buffer.crop_masked(x, y, w, h, mask, mask_style)?,
            )),
            ThymeImage::F64(buffer) => Ok(ThymeImage::F64(
                buffer.crop_masked(x, y, w, h, mask, mask_style)?,
            )),
        }
    }

    /// Resize the image
    ///
    /// # Arguments
    ///
    /// * `width` - Width of resized image
    /// * `height` - Height of resized image
    pub fn resize(&self, width: u32, height: u32) -> Result<ThymeImage, ThymeError> {
        let channels = self.channels();
        match (self, channels) {
            (ThymeImage::U8(buffer), 1) => Ok(ThymeImage::U8(ThymeBuffer::new(
                width,
                height,
                1,
                transform::resize_bilinear_fast(
                    &DynamicImage::ImageLuma8(
                        ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                            buffer.width(),
                            buffer.height(),
                            buffer.as_raw().to_vec(),
                        )
                        .ok_or(ThymeError::ImageError("Failed to resize image"))?,
                    ),
                    width,
                    height,
                    PixelType::U8,
                ),
            )?)),
            (ThymeImage::U8(buffer), 3) => Ok(ThymeImage::U8(
                ThymeBuffer::new(
                    width,
                    height,
                    3,
                    transform::resize_bilinear_fast(
                        &DynamicImage::ImageRgb8(
                            ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                                buffer.width(),
                                buffer.height(),
                                buffer.as_raw().to_vec(),
                            )
                            .ok_or(ThymeError::ImageError("Failed to resize image"))?,
                        ),
                        width,
                        height,
                        PixelType::U8x3,
                    ),
                )
                .map_err(|_| ThymeError::ImageError("Failed to resize image."))?,
            )),
            (ThymeImage::U16(buffer), 1) => Ok(ThymeImage::U16(ThymeBuffer::new(
                width,
                height,
                1,
                transform::resize_bilinear_default(
                    &ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(
                        buffer.width(),
                        buffer.height(),
                        buffer.as_raw().to_vec(),
                    )
                    .ok_or(ThymeError::ImageError("Failed to resize image"))?,
                    width,
                    height,
                )
                .into_raw(),
            )?)),
            (ThymeImage::U16(buffer), 3) => Ok(ThymeImage::U16(ThymeBuffer::new(
                width,
                height,
                3,
                transform::resize_bilinear_default(
                    &ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(
                        buffer.width(),
                        buffer.height(),
                        buffer.as_raw().to_vec(),
                    )
                    .ok_or(ThymeError::ImageError("Failed to resize image"))?,
                    width,
                    height,
                )
                .into_raw(),
            )?)),
            (ThymeImage::F32(buffer), 3) => Ok(ThymeImage::F32(ThymeBuffer::new(
                width,
                height,
                3,
                transform::resize_bilinear_default(
                    &ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(
                        buffer.width(),
                        buffer.height(),
                        buffer.as_raw().to_vec(),
                    )
                    .ok_or(ThymeError::ImageError("Failed to resize image"))?,
                    width,
                    height,
                )
                .into_raw(),
            )?)),
            (ThymeImage::U8(buffer), _) => Ok(ThymeImage::U8(ThymeBuffer::new(
                width,
                height,
                channels,
                transform::resize_bilinear_general::<u8>(
                    buffer.as_raw(),
                    buffer.width() as usize,
                    buffer.height() as usize,
                    channels as usize,
                    width as usize,
                    height as usize,
                    true,
                ),
            )?)),
            (ThymeImage::U16(buffer), _) => Ok(ThymeImage::U16(ThymeBuffer::new(
                width,
                height,
                channels,
                transform::resize_bilinear_general::<u16>(
                    buffer.as_raw(),
                    buffer.width() as usize,
                    buffer.height() as usize,
                    channels as usize,
                    width as usize,
                    height as usize,
                    true,
                ),
            )?)),
            (ThymeImage::U32(buffer), _) => Ok(ThymeImage::U32(ThymeBuffer::new(
                width,
                height,
                channels,
                transform::resize_bilinear_general::<u32>(
                    buffer.as_raw(),
                    buffer.width() as usize,
                    buffer.height() as usize,
                    channels as usize,
                    width as usize,
                    height as usize,
                    true,
                ),
            )?)),
            (ThymeImage::U64(buffer), _) => Ok(ThymeImage::U64(ThymeBuffer::new(
                width,
                height,
                channels,
                transform::resize_bilinear_general::<u64>(
                    buffer.as_raw(),
                    buffer.width() as usize,
                    buffer.height() as usize,
                    channels as usize,
                    width as usize,
                    height as usize,
                    true,
                ),
            )?)),
            (ThymeImage::I32(buffer), _) => Ok(ThymeImage::I32(ThymeBuffer::new(
                width,
                height,
                channels,
                transform::resize_bilinear_general::<i32>(
                    buffer.as_raw(),
                    buffer.width() as usize,
                    buffer.height() as usize,
                    channels as usize,
                    width as usize,
                    height as usize,
                    true,
                ),
            )?)),
            (ThymeImage::I64(buffer), _) => Ok(ThymeImage::I64(ThymeBuffer::new(
                width,
                height,
                channels,
                transform::resize_bilinear_general::<i64>(
                    buffer.as_raw(),
                    buffer.width() as usize,
                    buffer.height() as usize,
                    channels as usize,
                    width as usize,
                    height as usize,
                    true,
                ),
            )?)),
            (ThymeImage::F32(buffer), _) => Ok(ThymeImage::F32(ThymeBuffer::new(
                width,
                height,
                channels,
                transform::resize_bilinear_general::<f32>(
                    buffer.as_raw(),
                    buffer.width() as usize,
                    buffer.height() as usize,
                    channels as usize,
                    width as usize,
                    height as usize,
                    true,
                ),
            )?)),
            (ThymeImage::F64(buffer), _) => Ok(ThymeImage::F64(ThymeBuffer::new(
                width,
                height,
                channels,
                transform::resize_bilinear_general::<f64>(
                    buffer.as_raw(),
                    buffer.width() as usize,
                    buffer.height() as usize,
                    channels as usize,
                    width as usize,
                    height as usize,
                    true,
                ),
            )?)),
        }
    }
}

// <<< TRANSFORM METHODS

#[cfg(test)]
mod test {

    use super::*;
    use image::{
        DynamicImage, GrayAlphaImage, GrayImage, ImageBuffer, Luma, LumaA, Rgb, Rgb32FImage,
        RgbImage, Rgba, Rgba32FImage, RgbaImage,
    };

    const TEST_GRAY: &str = "../data/tests/test_grayscale";
    const TEST_RGB: &str = "../data/tests/test_rgb";

    #[test]
    fn test_grayscale_open() {
        let extensions = [
            ".jpeg", ".npy", ".pbm", ".png", ".tga", ".tif", "_f32.npy", "_f64.npy", "_i32.npy",
            "_i64.npy", "_u16.npy",
        ];

        for ext in extensions.into_iter() {
            let img = ThymeImage::open(format!("{}{}", TEST_GRAY, ext));
            assert!(img.is_ok(), "{}", ext);

            let img = img.unwrap();
            assert_eq!(img.width(), 621);
            assert_eq!(img.height(), 621);
            assert_eq!(img.channels(), 1, "{}", ext);
        }

        // Grayscale .bmp and .webp default to 3-channel
        // RGB when saved using python Pillow or cv2
        let extensions = [".bmp", ".webp"];
        for ext in extensions.into_iter() {
            let img = ThymeImage::open(format!("{}{}", TEST_GRAY, ext));
            assert!(img.is_ok(), "{}", ext);

            let img = img.unwrap();
            assert_eq!(img.width(), 621);
            assert_eq!(img.height(), 621);
            assert_eq!(img.channels(), 3, "{}", ext);
        }
    }

    #[test]
    fn test_rgb_open() {
        let extensions = [
            ".bmp", ".jpeg", ".npy", ".pbm", ".png", ".tga", ".tif", ".webp", "_f32.npy",
            "_f64.npy", "_i32.npy", "_i64.npy",
        ];

        for ext in extensions.into_iter() {
            let img = ThymeImage::open(format!("{}{}", TEST_RGB, ext));
            assert!(img.is_ok(), "{}", ext);

            let img = img.unwrap();
            assert_eq!(img.width(), 621);
            assert_eq!(img.height(), 621);
            assert_eq!(img.channels(), 3, "{}", ext);
        }
    }

    #[test]
    fn test_grayscale_save() {
        const TEST_DEFAULT: &str = "TEST_SAVE_DEFAULT_GRAY.png";
        const TEST_NUMPY: &str = "TEST_SAVE_NUMPY_GRAY.npy";

        let img =
            ThymeImage::U8(ThymeBuffer::<u8, Vec<u8>>::new(2, 2, 1, vec![0, 1, 2, 3]).unwrap());

        img.clone().save(TEST_DEFAULT).unwrap();
        img.clone().save(TEST_NUMPY).unwrap();

        let img_default = ThymeImage::open(TEST_DEFAULT).unwrap();
        let img_numpy = ThymeImage::open(TEST_NUMPY).unwrap();

        assert_eq!(img.to_u8(), img_default.to_u8());
        assert_eq!(img.to_u8(), img_numpy.to_u8());

        std::fs::remove_file(TEST_DEFAULT).unwrap();
        std::fs::remove_file(TEST_NUMPY).unwrap();
    }

    #[test]
    fn test_rgb_save() {
        const TEST_DEFAULT: &str = "TEST_SAVE_DEFAULT_RGB.png";
        const TEST_NUMPY: &str = "TEST_SAVE_NUMPY_RGB.npy";

        let img = ThymeImage::U8(
            ThymeBuffer::<u8, Vec<u8>>::new(2, 2, 3, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                .unwrap(),
        );

        img.clone().save(TEST_DEFAULT).unwrap();
        img.clone().save(TEST_NUMPY).unwrap();

        let img_default = ThymeImage::open(TEST_DEFAULT).unwrap();
        let img_numpy = ThymeImage::open(TEST_NUMPY).unwrap();

        assert_eq!(img.to_u8(), img_default.to_u8());
        assert_eq!(img.to_u8(), img_numpy.to_u8());

        std::fs::remove_file(TEST_DEFAULT).unwrap();
        std::fs::remove_file(TEST_NUMPY).unwrap();
    }

    #[test]
    fn test_multichannel_save() {
        const TEST_DEFAULT: &str = "TEST_SAVE_DEFAULT_MULTI.png";
        const TEST_NUMPY: &str = "TEST_SAVE_NUMPY_MULTI.npy";

        let img = ThymeImage::U8(
            ThymeBuffer::<u8, Vec<u8>>::new(
                2,
                2,
                4,
                vec![0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11],
            )
            .unwrap(),
        );

        let failure = img.clone().save(TEST_DEFAULT);
        assert!(failure.is_err());

        img.clone().save(TEST_NUMPY).unwrap();

        let img_numpy = ThymeImage::open(TEST_NUMPY).unwrap();
        assert_eq!(img.to_u8(), img_numpy.to_u8());
        std::fs::remove_file(TEST_NUMPY).unwrap();
    }

    #[test]
    fn test_dynamic_gray_u8() {
        let dynamic = DynamicImage::ImageLuma8(GrayImage::from_fn(10, 10, |x, _| Luma([x as u8])));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);
    }

    #[test]
    fn test_dynamic_gray_alpha_u8() {
        let dynamic =
            DynamicImage::ImageLumaA8(GrayAlphaImage::from_fn(10, 10, |x, _| LumaA([x as u8, 10])));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);

        let data = image.to_u8();
        assert_eq!(data[0], 0);
        assert_eq!(data[1], 1);
    }

    #[test]
    fn test_dynamic_gray_u16() {
        let dynamic = DynamicImage::ImageLuma16(ImageBuffer::<Luma<u16>, Vec<u16>>::from_fn(
            10,
            10,
            |x, _| Luma([x as u16]),
        ));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);
    }

    #[test]
    fn test_dynamic_gray_alpha_u16() {
        let dynamic = DynamicImage::ImageLumaA16(ImageBuffer::<LumaA<u16>, Vec<u16>>::from_fn(
            10,
            10,
            |x, _| LumaA([x as u16, 10]),
        ));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);

        let data = image.to_u16();
        assert_eq!(data[0], 0);
        assert_eq!(data[1], 1);
    }

    #[test]
    fn test_dynamic_rgb_u8() {
        let dynamic = DynamicImage::ImageRgb8(RgbImage::from_fn(10, 10, |x, _| {
            Rgb([x as u8, x as u8, x as u8])
        }));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);

        let data = image.to_u8();
        assert_eq!((data[0], data[1], data[2]), (0, 0, 0));
        assert_eq!((data[3], data[4], data[5]), (1, 1, 1));
    }

    #[test]
    fn test_dynamic_rgb_alpha_u8() {
        let dynamic = DynamicImage::ImageRgba8(RgbaImage::from_fn(10, 10, |x, _| {
            Rgba([x as u8, x as u8, x as u8, 10])
        }));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);

        let data = image.to_u8();
        assert_eq!((data[0], data[1], data[2]), (0, 0, 0));
        assert_eq!((data[3], data[4], data[5]), (1, 1, 1));
    }

    #[test]
    fn test_dynamic_rgb_u16() {
        let dynamic = DynamicImage::ImageRgb16(ImageBuffer::<Rgb<u16>, Vec<u16>>::from_fn(
            10,
            10,
            |x, _| Rgb([x as u16, x as u16, x as u16]),
        ));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);

        let data = image.to_u16();
        assert_eq!((data[0], data[1], data[2]), (0, 0, 0));
        assert_eq!((data[3], data[4], data[5]), (1, 1, 1));
    }

    #[test]
    fn test_dynamic_rgb_alpha_u16() {
        let dynamic = DynamicImage::ImageRgba16(ImageBuffer::<Rgba<u16>, Vec<u16>>::from_fn(
            10,
            10,
            |x, _| Rgba([x as u16, x as u16, x as u16, 10]),
        ));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);

        let data = image.to_u16();
        assert_eq!((data[0], data[1], data[2]), (0, 0, 0));
        assert_eq!((data[3], data[4], data[5]), (1, 1, 1));
    }

    #[test]
    fn test_dynamic_rgb_f32() {
        let dynamic = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(10, 10, |x, _| {
            Rgb([x as f32, x as f32, x as f32])
        }));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);
    }

    #[test]
    fn test_dynamic_rgb_alpha_f32() {
        let dynamic = DynamicImage::ImageRgba32F(Rgba32FImage::from_fn(10, 10, |x, _| {
            Rgba([x as f32, x as f32, x as f32, 10.0])
        }));

        let image = ThymeImage::new_from_default(dynamic);
        assert!(image.is_ok());

        let image = image.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);
    }

    #[test]
    fn test_resize_u8() {
        let dynamic = DynamicImage::ImageLuma8(GrayImage::from_fn(10, 10, |x, _| Luma([x as u8])));

        let one_channel = ThymeImage::new_from_default(dynamic).unwrap();

        let downsampled = one_channel.resize(3, 4).unwrap();
        assert_eq!(downsampled.width(), 3);
        assert_eq!(downsampled.height(), 4);

        let upsampled = one_channel.resize(23, 24).unwrap();
        assert_eq!(upsampled.width(), 23);
        assert_eq!(upsampled.height(), 24);

        let dynamic = DynamicImage::ImageRgb8(RgbImage::from_fn(10, 10, |x, _| {
            Rgb([x as u8, x as u8, x as u8])
        }));

        let three_channel = ThymeImage::new_from_default(dynamic).unwrap();

        let downsampled = three_channel.resize(3, 4).unwrap();
        assert_eq!(downsampled.width(), 3);
        assert_eq!(downsampled.height(), 4);

        let upsampled = three_channel.resize(23, 24).unwrap();
        assert_eq!(upsampled.width(), 23);
        assert_eq!(upsampled.height(), 24);

        let two_channel = ThymeImage::U8(
            ThymeBuffer::<u8, Vec<u8>>::new(2, 2, 2, vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
        );

        let downsampled = two_channel.resize(3, 4).unwrap();
        assert_eq!(downsampled.width(), 3);
        assert_eq!(downsampled.height(), 4);

        let upsampled = two_channel.resize(23, 24).unwrap();
        assert_eq!(upsampled.width(), 23);
        assert_eq!(upsampled.height(), 24);
    }

    #[test]
    fn test_resize_u16() {
        let dynamic = DynamicImage::ImageLuma16(ImageBuffer::<Luma<u16>, Vec<u16>>::from_fn(
            10,
            10,
            |x, _| Luma([x as u16]),
        ));

        let one_channel = ThymeImage::new_from_default(dynamic).unwrap();

        let downsampled = one_channel.resize(3, 4).unwrap();
        assert_eq!(downsampled.width(), 3);
        assert_eq!(downsampled.height(), 4);

        let upsampled = one_channel.resize(23, 24).unwrap();
        assert_eq!(upsampled.width(), 23);
        assert_eq!(upsampled.height(), 24);

        let dynamic = DynamicImage::ImageRgb16(ImageBuffer::<Rgb<u16>, Vec<u16>>::from_fn(
            10,
            10,
            |x, _| Rgb([x as u16, x as u16, x as u16]),
        ));

        let three_channel = ThymeImage::new_from_default(dynamic).unwrap();

        let downsampled = three_channel.resize(3, 4).unwrap();
        assert_eq!(downsampled.width(), 3);
        assert_eq!(downsampled.height(), 4);

        let upsampled = three_channel.resize(23, 24).unwrap();
        assert_eq!(upsampled.width(), 23);
        assert_eq!(upsampled.height(), 24);

        let two_channel = ThymeImage::U16(
            ThymeBuffer::<u16, Vec<u16>>::new(2, 2, 2, vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
        );

        let downsampled = two_channel.resize(3, 4).unwrap();
        assert_eq!(downsampled.width(), 3);
        assert_eq!(downsampled.height(), 4);

        let upsampled = two_channel.resize(23, 24).unwrap();
        assert_eq!(upsampled.width(), 23);
        assert_eq!(upsampled.height(), 24);
    }
}
