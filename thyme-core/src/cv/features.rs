// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::ops::Deref;

use num::{FromPrimitive, ToPrimitive};

use crate::constant::{GLCM_ARRAY_SIZE, GLCM_LEVELS};
use crate::im::ThymeViewBuffer;

#[derive(Debug, Clone)]
pub struct GLCM {
    data: [f32; GLCM_ARRAY_SIZE],
    rows: usize,
    cols: usize,
}

impl GLCM {
    /// Create a new normalized gray-level co-occurence matrix
    ///
    /// # Arguments
    ///
    /// * `pixels` - A row-major raw pixel buffer
    /// * `width` - Width of image
    /// * `height` - Height of image
    /// * `channel` - Which channel to compute the comatrix
    /// * `channels` - Number of channels in image
    /// * `angle` - Angle (in degrees) for computing neighbour co-occurence
    /// * `distance` - Number of pixels to neighbouring pixels
    ///
    /// # Examples
    ///
    /// ```
    /// use thyme_core::cv::features::GLCM;
    /// let buffer: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let comatrix = GLCM::new(&buffer, 3, 3, 0, 1, 0.0, 1.0);
    /// ```
    pub fn new<T>(
        pixels: &[T],
        width: usize,
        height: usize,
        channel: usize,
        channels: usize,
        angle: f32,
        distance: f32,
    ) -> GLCM
    where
        T: ToPrimitive,
    {
        let radians = angle.to_radians();

        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        let pixel_vec: Vec<f32> = pixels
            .iter()
            .skip(channel)
            .step_by(channels)
            .map(|p| {
                let value = p.to_f32().unwrap();
                min_val = min_val.min(value);
                max_val = max_val.max(value);
                value
            })
            .collect();

        let (sa, sb, sc) = if max_val != GLCM_LEVELS as f32 - 1.0 || min_val != 0.0 {
            (min_val, max_val, GLCM_LEVELS as f32 - 1.0)
        } else if min_val == max_val {
            // Homogeneous images are set to zero
            (0.0, 1.0, 0.0)
        } else {
            (0.0, 1.0, 1.0)
        };

        let (w, h) = (width as i32, height as i32);
        let offset_x = (radians.cos() * distance).round() as i32;
        let offset_y = (radians.sin() * distance).round() as i32;

        let mut comatrix = [0.0; GLCM_ARRAY_SIZE];

        let scale_pixel =
            |pixel: f32| -> usize { ((pixel - sa) / (sb - sa) * sc).round() as usize };

        let mut comatrix_sum = 0f32;

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let root = pixel_vec[idx];

                let i_offset = x + offset_x;
                let j_offset = y + offset_y;

                if i_offset >= w || i_offset < 0 || j_offset >= h || j_offset < 0 {
                    continue;
                }

                let neighbour_idx = (j_offset * w + i_offset) as usize;
                let neighbour = pixel_vec[neighbour_idx];

                let root_scaled = scale_pixel(root);
                let neighbour_scaled = scale_pixel(neighbour);

                comatrix[root_scaled * GLCM_LEVELS + neighbour_scaled] += 1.0;
                comatrix[neighbour_scaled * GLCM_LEVELS + root_scaled] += 1.0;

                comatrix_sum += 2.0;
            }
        }

        comatrix.iter_mut().for_each(|v| *v /= comatrix_sum);

        GLCM {
            data: comatrix,
            rows: GLCM_LEVELS,
            cols: GLCM_LEVELS,
        }
    }

    /// Create a new normalized gray-level co-occurence matrix from aa ThymeObjectBuffer
    ///
    /// # Arguments
    ///
    /// * `object` - A ThymeObjectBuffer
    /// * `channel` - Which channel to compute the comatrix
    /// * `angle` - Angle (in degrees) for computing neighbour co-occurence
    /// * `distance` - Number of pixels to neighbouring pixels
    ///
    /// # Note
    ///
    /// This is pretty redunant with the default constructor. We could possibly
    /// just accept a Vec<f32> instead and perform the other operations in the
    /// glcm_multichannel function. This would avoid the need for object specific
    /// functions.
    pub fn new_from_object<T, Container>(
        object: &ThymeViewBuffer<T, Container>,
        channel: usize,
        angle: f32,
        distance: f32,
    ) -> GLCM
    where
        T: ToPrimitive + FromPrimitive,
        Container: Deref<Target = [T]>,
    {
        let radians = angle.to_radians();

        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        let pixel_vec: Vec<f32> = object
            .iter()
            .skip(channel)
            .step_by(object.channels())
            .map(|p| {
                let value = p.to_f32().unwrap();
                min_val = min_val.min(value);
                max_val = max_val.max(value);
                value
            })
            .collect();

        let (sa, sb, sc) = if max_val != GLCM_LEVELS as f32 - 1.0 || min_val != 0.0 {
            (min_val, max_val, GLCM_LEVELS as f32 - 1.0)
        } else if min_val == max_val {
            // Homogeneous images are set to zero
            (0.0, 1.0, 0.0)
        } else {
            (0.0, 1.0, 1.0)
        };

        let (w, h) = (object.width() as i32, object.height() as i32);
        let offset_x = (radians.cos() * distance).round() as i32;
        let offset_y = (radians.sin() * distance).round() as i32;

        let mut comatrix = [0.0; GLCM_ARRAY_SIZE];

        let scale_pixel =
            |pixel: f32| -> usize { ((pixel - sa) / (sb - sa) * sc).round() as usize };

        let mut comatrix_sum = 0f32;

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let root = pixel_vec[idx];

                let i_offset = x + offset_x;
                let j_offset = y + offset_y;

                if i_offset >= w || i_offset < 0 || j_offset >= h || j_offset < 0 {
                    continue;
                }

                let neighbour_idx = (j_offset * w + i_offset) as usize;
                let neighbour = pixel_vec[neighbour_idx];

                let root_scaled = scale_pixel(root);
                let neighbour_scaled = scale_pixel(neighbour);

                comatrix[root_scaled * GLCM_LEVELS + neighbour_scaled] += 1.0;
                comatrix[neighbour_scaled * GLCM_LEVELS + root_scaled] += 1.0;

                comatrix_sum += 2.0;
            }
        }

        comatrix.iter_mut().for_each(|v| *v /= comatrix_sum);

        GLCM {
            data: comatrix,
            rows: GLCM_LEVELS,
            cols: GLCM_LEVELS,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f32)> + '_ {
        self.data.iter().enumerate().map(|(index, &value)| {
            let i = index / self.cols;
            let j = index % self.cols;
            (i, j, value)
        })
    }

    pub fn margin_sums(&self) -> (Vec<f32>, Vec<f32>) {
        let mut row_sums = vec![0.0; self.rows];
        let mut col_sums = vec![0.0; self.cols];

        for (i, j, value) in self.iter() {
            row_sums[i] += value;
            col_sums[j] += value;
        }

        (row_sums, col_sums)
    }
}

/// Compute a normalized gray-level co-occurence matrix for each image channel
///
/// # Arguments
///
/// * `pixels` - A row-major raw pixel buffer
/// * `width` - Width of image
/// * `height` - Height of image
/// * `channels` - Number of channels in image
/// * `angle` - Angle (in degrees) for computing neighbour co-occurence
/// * `distance` - Number of pixels to neighbouring pixels
///
/// # Examples
///
/// ```
/// use thyme_core::cv::features::glcm_multichannel;
/// let buffer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
/// let comatrices = glcm_multichannel(&buffer, 2, 2, 3, 0.0, 1.0);
/// ```
pub fn glcm_multichannel<T>(
    pixels: &[T],
    width: usize,
    height: usize,
    channels: usize,
    angle: f32,
    distance: f32,
) -> Vec<GLCM>
where
    T: ToPrimitive,
{
    (0..channels)
        .map(|channel| GLCM::new(pixels, width, height, channel, channels, angle, distance))
        .collect()
}

/// Compute channel-wise normalized gray-level co-occurence matrix from a ThymeObjectBuffer
///
/// # Arguments
///
/// * `pixels` - A row-major raw pixel buffer
/// * `width` - Width of image
/// * `height` - Height of image
/// * `channels` - Number of channels in image
/// * `angle` - Angle (in degrees) for computing neighbour co-occurence
/// * `distance` - Number of pixels to neighbouring pixels
pub fn glcm_multichannel_object<T, Container>(
    object: &ThymeViewBuffer<T, Container>,
    angle: f32,
    distance: f32,
) -> Vec<GLCM>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    (0..object.channels())
        .map(|channel| GLCM::new_from_object(object, channel, angle, distance))
        .collect()
}
