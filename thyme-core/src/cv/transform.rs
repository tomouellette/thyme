// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::ops::{Add, Div, Mul, Sub};

use fast_image_resize;
use fast_image_resize::{FilterType, PixelType, images::Image};
use image::{DynamicImage, GenericImage, ImageBuffer, Pixel};
use num::{FromPrimitive, ToPrimitive};

/// Resize a 2D image-rs ImageBuffer
///
/// # Arguments
///
/// * `image` - A u8 or u16 Luma or RGB ImageBuffer
/// * `new_width` - New width following resizing
/// * `new_height` - New height following resizing
pub fn resize_bilinear_default<I, P>(
    image: &I,
    new_width: u32,
    new_height: u32,
) -> ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>
where
    I: GenericImage<Pixel = P>,
    P: Pixel + 'static,
{
    image::imageops::resize(
        image,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    )
}

/// Resize a 2D u8 image using the SIMD-accelerated fast-image-resize crate
///
/// # Arguments
///
/// * `source` - A DynamicImage with u8 subpixel type
/// * `new_width` - New width following resizing
/// * `new_height` - New height following resizing
/// * `pixel_type` - RGB or Luma pixel type
pub fn resize_bilinear_fast(
    source: &DynamicImage,
    new_width: u32,
    new_height: u32,
    pixel_type: PixelType,
) -> Vec<u8> {
    let mut destination = Image::new(new_width, new_height, pixel_type);

    let mut resizer = fast_image_resize::Resizer::new();
    let option = fast_image_resize::ResizeOptions {
        algorithm: fast_image_resize::ResizeAlg::Convolution(FilterType::Bilinear),
        cropping: fast_image_resize::SrcCropping::None,
        mul_div_alpha: false,
    };

    resizer.resize(source, &mut destination, &option).unwrap();

    destination.into_vec()
}

/// Resizes a 2D image buffer using bilinear interpolation
///
/// This is going to be pretty inefficient but will only be called for
/// a small subset of images that aren't u8 or u16 type. If an application
/// with a lot of float type images are used, then we can re-implement a
/// faster/more efficient approach.
///
/// # Arguments
///
/// * `buffer` - Input image buffer in row-major order (width * height * channels)
/// * `width` - Current width of the image
/// * `height` - Current height of the image
/// * `channels` - Number of channels (1 for grayscale, 3 for RGB, etc.)
/// * `new_width` - Target width
/// * `new_height` - Target height
/// * `round` - Round values before casting to original type
pub fn resize_bilinear_general<T>(
    buffer: &[T],
    width: usize,
    height: usize,
    channels: usize,
    new_width: usize,
    new_height: usize,
    round: bool,
) -> Vec<T>
where
    T: Copy + FromPrimitive + ToPrimitive + 'static,
    f64: Add<Output = f64> + Mul<Output = f64> + Sub<Output = f64> + Div<Output = f64>,
{
    assert_eq!(buffer.len(), width * height * channels);
    let mut result = vec![T::from_u8(0).unwrap(); new_width * new_height * channels];

    if width == new_width && height == new_height {
        return buffer.to_vec();
    }

    let x_ratio = (width as f64).max(1.0) / (new_width as f64).max(1.0);
    let y_ratio = (height as f64).max(1.0) / (new_height as f64).max(1.0);

    for y in 0..new_height {
        let y_f = (y as f64 - 0.5) * y_ratio;
        let y1 = (y_f.floor() as usize).max(0);
        let y2 = (y_f.ceil() as usize).min(height - 1);
        let y_diff = y_f - y1 as f64;

        for x in 0..new_width {
            let x_f = (x as f64 - 0.5) * x_ratio;
            let x1 = (x_f.floor() as usize).max(0);
            let x2 = (x_f.ceil() as usize).min(width - 1);
            let x_diff = x_f - x1 as f64;

            for c in 0..channels {
                let a = buffer[(y1 * width + x1) * channels + c].to_f64().unwrap();
                let b = buffer[(y1 * width + x2) * channels + c].to_f64().unwrap();
                let c_val = buffer[(y2 * width + x1) * channels + c].to_f64().unwrap();
                let d = buffer[(y2 * width + x2) * channels + c].to_f64().unwrap();

                let interpolant = if x_diff < 1e-5 && y_diff < 1e-5 {
                    a // Snap to exact pixel if very close
                } else {
                    a * (1.0 - x_diff) * (1.0 - y_diff)
                        + b * x_diff * (1.0 - y_diff)
                        + c_val * (1.0 - x_diff) * y_diff
                        + d * x_diff * y_diff
                };

                let val = if round {
                    interpolant.round()
                } else {
                    interpolant
                };

                result[(y * new_width + x) * channels + c] = T::from_f64(val).unwrap();
            }
        }
    }

    result
}

#[cfg(test)]
mod test {

    use super::*;
    use fast_image_resize::PixelType;
    use image::Luma;

    #[test]
    fn test_resize_consistency() {
        // There is going to be some variability in how integer types are
        // handled across various resize implementations so we accept consistency
        // at some specified error.
        const MAX_ERROR: f32 = 5.0;

        let buffer_u8 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];

        let image_buffer =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(3, 3, buffer_u8.clone()).unwrap();
        let dynamic_image = DynamicImage::ImageLuma8(image_buffer.clone());

        let resize_default = resize_bilinear_default(&image_buffer, 5, 5)
            .as_raw()
            .clone();
        let resize_fast = resize_bilinear_fast(&dynamic_image, 5, 5, PixelType::U8);
        let resize_general = resize_bilinear_general(&buffer_u8, 3, 3, 1, 5, 5, true);

        assert!(
            resize_default
                .iter()
                .zip(&resize_fast)
                .map(|(x, y)| (x.to_f32().unwrap() - y.to_f32().unwrap()).abs())
                .sum::<f32>()
                < MAX_ERROR
        );

        assert!(
            resize_default
                .iter()
                .zip(&resize_general)
                .map(|(x, y)| (x.to_f32().unwrap() - y.to_f32().unwrap()).abs())
                .sum::<f32>()
                < MAX_ERROR
        );

        assert!(
            resize_fast
                .iter()
                .zip(&resize_general)
                .map(|(x, y)| (x.to_f32().unwrap() - y.to_f32().unwrap()).abs())
                .sum::<f32>()
                < MAX_ERROR
        );
    }

    #[test]
    fn test_resize_general() {
        let buffer_u8_symmetric = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];

        assert_eq!(
            resize_bilinear_general(&buffer_u8_symmetric, 3, 3, 1, 5, 5, true).len(),
            25
        );

        assert_eq!(
            resize_bilinear_general(&buffer_u8_symmetric, 3, 3, 1, 3, 4, true).len(),
            12
        );

        assert_eq!(
            resize_bilinear_general(&buffer_u8_symmetric, 3, 3, 1, 5, 7, true).len(),
            35
        );

        let buffer_u8_symmetric_two_channels = vec![0, 1, 2, 3, 4, 5, 6, 7];

        assert_eq!(
            resize_bilinear_general(&buffer_u8_symmetric_two_channels, 2, 2, 2, 3, 4, true).len(),
            24
        );

        assert_eq!(
            resize_bilinear_general(&buffer_u8_symmetric_two_channels, 2, 2, 2, 3, 4, true).len(),
            24
        );

        assert_eq!(
            resize_bilinear_general(&buffer_u8_symmetric_two_channels, 2, 2, 2, 23, 24, true).len(),
            23 * 24 * 2
        );
    }
}
