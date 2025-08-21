// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::ops::Deref;

use num::{FromPrimitive, ToPrimitive};

use crate::cv::features::{GLCM, glcm_multichannel, glcm_multichannel_object};
use crate::im::ThymeViewBuffer;

#[inline]
pub fn texture_energy(glcm: &GLCM) -> f32 {
    glcm.iter().fold(0.0, |acc, (_, _, x)| acc + x * x)
}

#[inline]
pub fn texture_contrast(glcm: &GLCM) -> f32 {
    let mut contrast = 0.0;
    for (i, j, g_ij) in glcm.iter() {
        let i_minus_j = i as f32 - j as f32;
        contrast += i_minus_j * i_minus_j * g_ij;
    }
    contrast
}

#[inline]
pub fn texture_correlation(glcm: &GLCM) -> f32 {
    let (px, py) = glcm.margin_sums();

    let ux = px
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, &x)| acc + ((i as f32 + 1.0) * x));

    let uy = py
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, &y)| acc + ((i as f32 + 1.0) * y));

    let sx = px
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, &x)| {
            acc + (i as f32 + 1.0 - ux) * (i as f32 + 1.0 - ux) * x
        })
        .sqrt();

    let sy = py
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, &y)| {
            acc + (i as f32 + 1.0 - uy) * (i as f32 + 1.0 - uy) * y
        })
        .sqrt();

    let mut correlation = 0.0;
    for (i, j, g_ij) in glcm.iter() {
        correlation += ((i as f32 + 1.0 - ux) * (j as f32 + 1.0 - uy) * g_ij) / (sx * sy);
    }

    correlation
}

#[inline]
pub fn texture_sum_of_squares(glcm: &GLCM) -> f32 {
    let (px, _) = glcm.margin_sums();

    let u = px
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, &x)| acc + ((i as f32 + 1.0) * x));

    let mut sum_of_squares = 0.0;
    for (i, _, g_ij) in glcm.iter() {
        sum_of_squares += (i as f32 + 1.0 - u) * (i as f32 + 1.0 - u) * g_ij;
    }

    sum_of_squares
}

#[inline]
pub fn texture_inverse_difference(glcm: &GLCM) -> f32 {
    let mut inverse_difference_moment = 0.0;
    for (i, j, g_ij) in glcm.iter() {
        inverse_difference_moment +=
            (1.0 / (1.0 + ((i as f32 - j as f32) * (i as f32 - j as f32)))) * g_ij;
    }

    inverse_difference_moment
}

#[inline]
pub fn texture_sum_average(glcm: &GLCM) -> f32 {
    let mut px_plus_y = vec![0.0; 2 * glcm.rows()];
    for (i, j, g_ij) in glcm.iter() {
        px_plus_y[i + j] += g_ij
    }

    let mut sum_average = 0.0;
    for (i, px_plus_y_k) in px_plus_y.iter().enumerate().take(2 * glcm.rows()) {
        sum_average += i as f32 * px_plus_y_k;
    }

    sum_average
}

#[inline]
pub fn texture_sum_variance(glcm: &GLCM) -> f32 {
    let mut px_plus_y = vec![0.0; 2 * glcm.rows()];
    for (i, j, g_ij) in glcm.iter() {
        px_plus_y[i + j] += g_ij;
    }

    let mut sum_average = 0.0;
    let mut sum_variance = 0.0;
    for (i, px_plus_y_k) in px_plus_y.iter().enumerate().take(2 * glcm.rows()) {
        let k = i as f32;
        sum_average += k * px_plus_y_k;
        sum_variance += k * k * px_plus_y_k;
    }

    sum_variance - sum_average * sum_average
}

#[inline]
pub fn texture_sum_entropy(glcm: &GLCM) -> f32 {
    let mut px_plus_y = vec![0.0; 2 * glcm.rows()];
    for (i, j, g_ij) in glcm.iter() {
        px_plus_y[i + j] += g_ij;
    }

    let mut sum_entropy = 0.0;
    for px_plus_y_k in px_plus_y.iter().take(2 * glcm.rows()) {
        let buffer = if *px_plus_y_k <= f32::EPSILON {
            1.0
        } else {
            0.0
        };

        sum_entropy += px_plus_y_k * (px_plus_y_k + buffer).log2();
    }

    -sum_entropy
}

#[inline]
pub fn texture_entropy(glcm: &GLCM) -> f32 {
    -glcm.iter().fold(0.0, |acc, (_, _, g_ij)| {
        acc + g_ij * (g_ij + f32::EPSILON).log2()
    })
}

#[inline]
pub fn texture_difference_variance(glcm: &GLCM) -> f32 {
    let mut px_minus_y = vec![0.0; glcm.rows()];
    for (i, j, g_ij) in glcm.iter() {
        let index = (i as i32 - j as i32).unsigned_abs() as usize;
        px_minus_y[index] += g_ij;
    }

    let mean_x_minus_y = px_minus_y.iter().sum::<f32>() / px_minus_y.len() as f32;
    let variance = px_minus_y.iter().enumerate().fold(0.0, |acc, (_, &x)| {
        acc + (x - mean_x_minus_y) * (x - mean_x_minus_y)
    });

    variance / px_minus_y.len() as f32
}

#[inline]
pub fn texture_difference_entropy(glcm: &GLCM) -> f32 {
    let mut px_minus_y = vec![0.0; glcm.rows()];
    for (i, j, g_ij) in glcm.iter() {
        let index = (i as i32 - j as i32).unsigned_abs() as usize;
        px_minus_y[index] += g_ij;
    }

    -px_minus_y
        .iter()
        .fold(0.0, |acc, &x| acc + x * (x + f32::EPSILON).log2())
}

#[inline]
pub fn texture_infocorr_1(glcm: &GLCM) -> f32 {
    let (px, py) = glcm.margin_sums();

    let hx = -px
        .iter()
        .fold(0.0, |acc, &x| acc + x * (x + f32::EPSILON).log2());
    let hy = -py
        .iter()
        .fold(0.0, |acc, &y| acc + y * (y + f32::EPSILON).log2());

    let mut hxy1 = 0.0;
    let mut hxy2 = 0.0;
    for (i, j, g_ij) in glcm.iter() {
        hxy1 += g_ij * (g_ij + f32::EPSILON).log2();
        hxy2 += px[i] * py[j] * (px[i] * py[j] + f32::EPSILON).log2();
    }

    (hxy2 - hxy1) / hx.max(hy)
}

#[inline]
pub fn texture_infocorr_2(glcm: &GLCM) -> f32 {
    let (px, py) = glcm.margin_sums();

    let mut hxy1 = 0.0;
    let mut hxy2 = 0.0;
    for (i, j, g_ij) in glcm.iter() {
        hxy1 += g_ij * (g_ij + f32::EPSILON).log2();
        hxy2 += px[i] * py[j] * (px[i] * py[j] + f32::EPSILON).log2();
    }

    (1.0 - (-2.0 * (hxy1 - hxy2)).exp()).sqrt()
}

#[inline]
pub fn haralick_features(glcm: &GLCM) -> [f32; 13] {
    let (px, py) = glcm.margin_sums();

    let (mut ux, mut uy) = (0.0, 0.0);
    let (mut hx, mut hy) = (0.0, 0.0);

    for i in 0..px.len() {
        ux += (i as f32 + 1.0) * px[i];
        uy += (i as f32 + 1.0) * py[i];
        hx -= px[i] * (px[i] + f32::EPSILON).log2();
        hy -= py[i] * (py[i] + f32::EPSILON).log2();
    }

    let sx = px
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, &x)| {
            acc + (i as f32 + 1.0 - ux) * (i as f32 + 1.0 - ux) * x
        })
        .sqrt();

    let sy = py
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, &y)| {
            acc + (i as f32 + 1.0 - uy) * (i as f32 + 1.0 - uy) * y
        })
        .sqrt();

    let mut hxy1 = 0.0;
    let mut hxy2 = 0.0;

    let mut px_plus_y = vec![0.0; 2 * glcm.rows()];
    let mut px_minus_y = vec![0.0; glcm.rows()];

    let mut energy = 0.0;
    let mut contrast = 0.0;
    let mut correlation = 0.0;
    let mut sum_of_squares = 0.0;
    let mut inverse_difference_moment = 0.0;
    let mut sum_average = 0.0;
    let mut sum_variance = 0.0;
    let mut sum_entropy = 0.0;
    let mut entropy = 0.0;
    let mut difference_variance = 0.0;
    let mut difference_entropy = 0.0;

    for (i, j, g_ij) in glcm.iter() {
        hxy1 += g_ij * (g_ij + f32::EPSILON).log2();
        hxy2 += px[i] * py[j] * (px[i] * py[j] + f32::EPSILON).log2();

        let index = (i as i32 - j as i32).unsigned_abs() as usize;
        px_minus_y[index] += g_ij;
        px_plus_y[i + j] += g_ij;

        let i = i as f32;
        let j = j as f32;
        let d = i - j;
        let dsq = d * d;

        energy += g_ij * g_ij;
        contrast += dsq * g_ij;
        correlation += ((i + 1.0 - ux) * (j + 1.0 - uy) * g_ij) / (sx * sy);
        sum_of_squares += (i + 1.0 - ux) * (i + 1.0 - ux) * g_ij;
        inverse_difference_moment += (1.0 / (1.0 + dsq)) * g_ij;
        entropy += g_ij * (g_ij + f32::EPSILON).log2();
    }

    for (i, px_plus_y_k) in px_plus_y.iter().enumerate().take(2 * glcm.rows()) {
        let k = i as f32;
        sum_average += k * px_plus_y_k;
        sum_variance += k * k * px_plus_y_k;

        let buffer = if *px_plus_y_k <= f32::EPSILON {
            1.0
        } else {
            0.0
        };
        sum_entropy += px_plus_y_k * (px_plus_y_k + buffer).log2();
    }

    sum_variance -= sum_average * sum_average;

    let u_x_minus_y = px_minus_y.iter().sum::<f32>() / px_minus_y.len() as f32;

    for px_minus_y_k in px_minus_y.iter().take(glcm.rows()) {
        difference_variance += (px_minus_y_k - u_x_minus_y) * (px_minus_y_k - u_x_minus_y);
        difference_entropy += px_minus_y_k * (px_minus_y_k + f32::EPSILON).log2();
    }

    difference_variance /= px_minus_y.len() as f32;

    let information_measure_of_correlation_1 = (hxy2 - hxy1) / hx.max(hy);
    let information_measure_of_correlation_2 = (1.0 - (-2.0 * (hxy1 - hxy2)).exp()).sqrt();

    [
        energy,
        contrast,
        correlation,
        sum_of_squares,
        inverse_difference_moment,
        sum_average,
        sum_variance,
        -sum_entropy,
        -entropy,
        difference_variance,
        -difference_entropy,
        information_measure_of_correlation_1,
        information_measure_of_correlation_2,
    ]
}

#[inline]
pub fn descriptors<T>(pixels: &[T], width: usize, height: usize, channels: usize) -> [f32; 13]
where
    T: ToPrimitive,
{
    let mut haralick: [f32; 13] = [0.0; 13];
    for i in [0, 45, 90, 135].iter() {
        for glcm in glcm_multichannel(pixels, width, height, channels, *i as f32, 1.0).iter() {
            let features = haralick_features(glcm);
            for j in 0..13 {
                haralick[j] += features[j] / (4.0 * channels as f32);
            }
        }
    }

    haralick
}

#[inline]
pub fn objects<T, Container>(object: &ThymeViewBuffer<T, Container>) -> [f32; 13]
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    let mut haralick: [f32; 13] = [0.0; 13];
    for i in [0, 45, 90, 135].iter() {
        for glcm in glcm_multichannel_object(object, *i as f32, 1.0).iter() {
            let features = haralick_features(glcm);
            for j in 0..13 {
                haralick[j] += features[j] / (4.0 * object.channels() as f32);
            }
        }
    }

    haralick
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::im::ThymeBuffer;

    fn square_image() -> [u8; 4] {
        [0, 0, 255, 255]
    }

    const EPS: f32 = 1e-6;

    #[test]
    fn test_texture_energy() {
        let energy = texture_energy(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert_eq!(energy, 0.5);
    }

    #[test]
    fn test_texture_contrast() {
        let contrast = texture_contrast(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert_eq!(contrast, 0.0);
    }

    #[test]
    fn test_texture_correlation() {
        let correlation = texture_correlation(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert_eq!(correlation, 1.0);
    }

    #[test]
    fn test_texture_sum_of_squares() {
        let sum_of_squares =
            texture_sum_of_squares(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert_eq!(sum_of_squares, 992.25);
    }

    #[test]
    fn test_texture_inverse_difference() {
        let inverse_difference =
            texture_inverse_difference(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));

        assert_eq!(inverse_difference, 1.0);
    }

    #[test]
    fn test_texture_sum_average() {
        let sum_average = texture_sum_average(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert_eq!(sum_average, 63.0);
    }

    #[test]
    fn test_texture_sum_variance() {
        let sum_variance = texture_sum_variance(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert_eq!(sum_variance, 3969.0);
    }

    #[test]
    fn test_texture_sum_entropy() {
        let sum_entropy = texture_sum_entropy(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert_eq!(sum_entropy, 1.0);
    }

    #[test]
    fn test_texture_entropy() {
        let entropy = texture_entropy(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert!((entropy - 1.0).abs() < EPS);
    }

    #[test]
    fn test_texture_difference_variance() {
        let difference_variance =
            texture_difference_variance(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));

        assert!((difference_variance - 0.015380859).abs() < EPS);
    }

    #[test]
    fn test_texture_difference_entropy() {
        let difference_entropy =
            texture_difference_entropy(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));

        assert!((difference_entropy - 0.0).abs() < EPS);
    }

    #[test]
    fn test_texture_information_measure_of_correlation_1() {
        let imc1 = texture_infocorr_1(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert_eq!(imc1, -1.0);
    }

    #[test]
    fn test_texture_information_measure_of_correlation_2() {
        let imc2 = texture_infocorr_2(&GLCM::new(&square_image(), 2, 2, 0, 1, 0.0, 1.0));
        assert!((imc2 - 0.92987347).abs() < EPS);
    }

    #[test]
    fn test_haralick_features() {
        let pixels = square_image();
        let comatrix = GLCM::new(&pixels, 2, 2, 0, 1, 0.0, 1.0);
        let features = haralick_features(&comatrix);

        assert_eq!(features.len(), 13);

        let energy = texture_energy(&comatrix);
        let contrast = texture_contrast(&comatrix);
        let correlation = texture_correlation(&comatrix);
        let sum_of_squares = texture_sum_of_squares(&comatrix);
        let inverse_difference = texture_inverse_difference(&comatrix);
        let sum_average = texture_sum_average(&comatrix);
        let sum_variance = texture_sum_variance(&comatrix);
        let sum_entropy = texture_sum_entropy(&comatrix);
        let entropy = texture_entropy(&comatrix);
        let difference_variance = texture_difference_variance(&comatrix);
        let difference_entropy = texture_difference_entropy(&comatrix);
        let imc1 = texture_infocorr_1(&comatrix);
        let imc2 = texture_infocorr_2(&comatrix);

        assert_eq!(features[0], energy);
        assert_eq!(features[1], contrast);
        assert_eq!(features[2], correlation);
        assert_eq!(features[3], sum_of_squares);
        assert_eq!(features[4], inverse_difference);
        assert_eq!(features[5], sum_average);
        assert_eq!(features[6], sum_variance);
        assert_eq!(features[7], sum_entropy);
        assert_eq!(features[8], entropy);
        assert_eq!(features[9], difference_variance);
        assert_eq!(features[10], difference_entropy);
        assert_eq!(features[11], imc1);
        assert_eq!(features[12], imc2);
    }

    #[test]
    fn test_object_texture() {
        let pixels = square_image();
        let buffer = ThymeBuffer::new(2, 2, 1, pixels.to_vec()).unwrap();
        let object = ThymeViewBuffer::new(0, 0, 2, 2, &buffer);

        let texture_array = descriptors(&pixels, 2, 2, 1);
        let texture_object = objects(&object);

        assert_eq!(texture_array, texture_object);
    }
}
