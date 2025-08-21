// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::cmp::Ordering;
use std::ops::Deref;

use num::{FromPrimitive, ToPrimitive};

use crate::im::ThymeViewBuffer;

#[inline]
pub fn intensity_min<T>(pixels: &[T], channels: usize) -> Vec<f32>
where
    T: ToPrimitive,
{
    let mut min = vec![f32::INFINITY; channels];
    for pixel in pixels.chunks_exact(channels) {
        for (i, v) in pixel.iter().enumerate() {
            let v = v.to_f32().unwrap();
            if v < min[i] && v > 0. {
                min[i] = v;
            }
        }
    }

    for v in min.iter_mut() {
        if *v == f32::INFINITY {
            *v = 0.0;
        }
    }

    min
}

#[inline]
pub fn intensity_max<T>(pixels: &[T], channels: usize) -> Vec<f32>
where
    T: ToPrimitive,
{
    let mut max = vec![f32::NEG_INFINITY; channels];
    for pixel in pixels.chunks_exact(channels) {
        for (i, v) in pixel.iter().enumerate() {
            let v = v.to_f32().unwrap();
            if v > max[i] && v > 0. {
                max[i] = v;
            }
        }
    }

    for v in max.iter_mut() {
        if *v == f32::NEG_INFINITY {
            *v = 0.0;
        }
    }

    max
}

#[inline]
pub fn intensity_mean<T>(pixels: &[T], channels: usize) -> Vec<f32>
where
    T: ToPrimitive,
{
    let mut n = vec![0; channels];
    let mut mean = vec![0.0; channels];
    for pixel in pixels.chunks_exact(channels) {
        for (i, v) in pixel.iter().enumerate() {
            let v = v.to_f32().unwrap();
            if v > 0. {
                n[i] += 1;
                mean[i] += v;
            }
        }
    }

    for i in 0..channels {
        if n[i] > 0 {
            mean[i] *= 1.0 / n[i] as f32;
        }
    }

    mean
}

#[inline]
pub fn intensity_sum<T>(pixels: &[T], channels: usize) -> Vec<f32>
where
    T: ToPrimitive,
{
    let mut sum = vec![0.0; channels];
    for pixel in pixels.chunks_exact(channels) {
        for (i, v) in pixel.iter().enumerate() {
            sum[i] += v.to_f32().unwrap();
        }
    }

    sum
}

#[inline]
pub fn intensity_std<T>(pixels: &[T], channels: usize) -> Vec<f32>
where
    T: ToPrimitive,
{
    let mut n = vec![0; channels];
    let mean = intensity_mean(pixels, channels);

    let mut std = vec![0.0; channels];
    for pixel in pixels.chunks_exact(channels) {
        for (i, v) in pixel.iter().enumerate() {
            let v = v.to_f32().unwrap();
            if v > 0. {
                n[i] += 1;
                std[i] += (v - mean[i]) * (v - mean[i]);
            }
        }
    }

    for i in 0..channels {
        if n[i] > 0 {
            std[i] = (std[i] * 1.0 / n[i] as f32).sqrt()
        }
    }

    std
}

#[inline]
pub fn intensity_median<T>(pixels: &[T]) -> f32
where
    T: Copy + Into<f32> + PartialOrd + ToPrimitive,
{
    let mut pixels: Vec<f32> = pixels
        .to_vec()
        .iter()
        .map(|x| x.to_f32().unwrap())
        .filter(|x| *x > 0.)
        .collect();

    pixels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let len = pixels.len();
    let mid = len / 2;

    if len % 2 == 0 {
        let left = pixels[mid - 1];
        let right = pixels[mid];
        (left + right) / 2.0
    } else {
        pixels[mid]
    }
}

#[inline]
pub fn intensity_mad<T>(pixels: &[T]) -> f32
where
    T: Copy + Into<f32> + PartialOrd + ToPrimitive,
{
    let median = intensity_median(pixels);
    let mut mad = Vec::new();
    for pixel in pixels.iter() {
        let pixel = pixel.to_f32().unwrap();
        if pixel > 0. {
            mad.push((pixel - median).abs());
        }
    }

    mad.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = mad.len() / 2;
    if mad.len() % 2 == 0 {
        (mad[mid] + mad[mid - 1]) / 2.0
    } else {
        mad[mid]
    }
}

#[inline]
#[allow(clippy::all)]
pub fn descriptors<T>(pixels: &[T], channels: usize) -> Vec<f32>
where
    T: ToPrimitive,
{
    let mut n = vec![0; channels];

    // Initial allocation for all intensity measurements. The
    // intensity min, max, sum, mean, and standard deviation
    // are stored in chunks that span the number of channels.
    // The last two spots are for median and mad descriptors.
    let mut results = vec![0.0; channels * 5 + 2];

    for i in 0..channels {
        results[i + 0 * channels] = f32::INFINITY;
        results[i + 1 * channels] = f32::NEG_INFINITY;
    }

    let mut store: Vec<f32> = Vec::with_capacity(pixels.len());

    for pixel in pixels.chunks_exact(channels) {
        for (i, v) in pixel.iter().enumerate() {
            let v = v.to_f32().unwrap();

            if v > 0. {
                n[i] += 1;

                // Intensity minimum
                results[i + 0 * channels] = results[i + 0 * channels].min(v);

                // Intensity maximum
                results[i + 1 * channels] = results[i + 1 * channels].max(v);

                // Intensity sum/integrated
                results[i + 2 * channels] += v;
            }

            store.push(v);
        }
    }

    for v in results.iter_mut().take(channels * 2) {
        if *v == f32::NEG_INFINITY || *v == f32::INFINITY {
            *v = 0.
        }
    }

    // Intensity mean
    for i in 0..channels {
        if n[i] > 0 {
            results[i + 3 * channels] = results[i + 2 * channels] * 1.0 / n[i] as f32;
        }
    }

    // Intensity standard deviation
    for pixel in store.chunks_exact(channels) {
        for (i, &v) in pixel.iter().enumerate() {
            results[i + 4 * channels] += (v - results[i + 3 * channels]).powi(2);
        }
    }

    for i in 0..channels {
        if n[i] > 0 {
            results[i + 4 * channels] = (results[i + 4 * channels] * 1.0 / n[i] as f32).sqrt();
        }
    }

    store.retain(|v| *v > 0.);

    if store.is_empty() {
        return results;
    }

    store.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let n = store.len();
    let mid = n / 2;
    let len = results.len();

    // Intensity median
    results[len - 2] = if n % 2 == 0 {
        (store[mid - 1] + store[mid]) / 2.0
    } else {
        store[mid]
    };

    store
        .iter_mut()
        .for_each(|pixel| *pixel = (*pixel - results[len - 2]).abs());

    store.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Intensity median absolute deviation
    results[len - 1] = if n % 2 == 0 {
        (store[mid] + store[mid - 1]) / 2.0
    } else {
        store[mid]
    };

    results
}

#[inline]
#[allow(clippy::all)]
pub fn objects<T, Container>(object: &ThymeViewBuffer<T, Container>) -> Vec<f32>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    let c = object.channels();
    let mut n = vec![0; c];

    // Initial allocation for all intensity measurements. The
    // intensity min, max, sum, mean, and standard deviation
    // are stored in chunks that span the number of channels.
    // The last two spots are for median and mad descriptors.
    let mut results = vec![0.0; c * 5 + 2];

    for i in 0..c {
        results[i + 0 * c] = f32::INFINITY;
        results[i + 1 * c] = f32::NEG_INFINITY;
    }

    let mut store: Vec<f32> = Vec::with_capacity(object.len());

    for pixel in object.iter_pixels() {
        for (i, v) in pixel.iter().enumerate() {
            let v = v.to_f32().unwrap();

            if v > 0. {
                n[i] += 1;

                // Intensity minimum
                results[i + 0 * c] = results[i + 0 * c].min(v);

                // Intensity maximum
                results[i + 1 * c] = results[i + 1 * c].max(v);

                // Intensity sum/integrated
                results[i + 2 * c] += v;
            }

            store.push(v);
        }
    }

    for v in results.iter_mut().take(c * 2) {
        if *v == f32::NEG_INFINITY || *v == f32::INFINITY {
            *v = 0.
        }
    }

    // Intensity mean
    for i in 0..c {
        if n[i] > 0 {
            results[i + 3 * c] = results[i + 2 * c] * 1.0 / n[i] as f32;
        }
    }

    // Intensity standard deviation
    for pixel in store.chunks_exact(c) {
        for (i, &v) in pixel.iter().enumerate() {
            if v > 0. {
                results[i + 4 * c] += (v - results[i + 3 * c]).powi(2);
            }
        }
    }

    for i in 0..c {
        if n[i] > 0 {
            results[i + 4 * c] = (results[i + 4 * c] * 1.0 / n[i] as f32).sqrt();
        }
    }

    store.retain(|x| *x > 0.);

    if store.is_empty() {
        return results;
    }

    store.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let n = store.len();
    let mid = n / 2;
    let len = results.len();

    // Intensity median
    results[len - 2] = if n % 2 == 0 {
        (store[mid - 1] + store[mid]) / 2.0
    } else {
        store[mid]
    };

    store
        .iter_mut()
        .for_each(|pixel| *pixel = (*pixel - results[len - 2]).abs());

    store.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Intensity median absolute deviation
    results[len - 1] = if n % 2 == 0 {
        (store[mid] + store[mid - 1]) / 2.0
    } else {
        store[mid]
    };

    results
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::im::ThymeBuffer;

    fn test_pixels() -> (Vec<u8>, usize) {
        let channels = 3;
        let pixels: Vec<u8> = vec![0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5];
        (pixels, channels)
    }

    fn test_object() -> ThymeBuffer<u8, Vec<u8>> {
        let pixels: Vec<u8> = vec![0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5];
        ThymeBuffer::new(2, 2, 3, pixels).unwrap()
    }

    #[test]
    fn test_intensity_min() {
        let (pixels, channels) = test_pixels();
        let min = intensity_min(&pixels, channels);
        assert_eq!(min, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_intensity_max() {
        let (pixels, channels) = test_pixels();
        let max = intensity_max(&pixels, channels);
        assert_eq!(max, vec![0.0, 1.0, 5.0]);
    }

    #[test]
    fn test_intensity_mean() {
        let (pixels, channels) = test_pixels();
        let mean = intensity_mean(&pixels, channels);
        assert_eq!(mean, vec![0.0, 1.0, 3.5]);
    }

    #[test]
    fn test_intensity_sum() {
        let (pixels, channels) = test_pixels();
        let sum = intensity_sum(&pixels, channels);
        assert_eq!(sum, vec![0.0, 4.0, 14.0]);
    }

    #[test]
    fn test_intensity_std() {
        let (pixels, channels) = test_pixels();
        let std = intensity_std(&pixels, channels);
        assert_eq!(std, vec![0.0, 0.0, 1.118034]);
    }

    #[test]
    fn test_intensity_median() {
        let (pixels, _) = test_pixels();
        let median = intensity_median(&pixels);
        assert_eq!(median, 1.5);
    }

    #[test]
    fn test_intensity_mad() {
        let (pixels, _) = test_pixels();
        let mad = intensity_mad(&pixels);
        assert_eq!(mad, 0.5);
    }

    #[test]
    fn test_descriptors() {
        let (pixels, channels) = test_pixels();

        let min = intensity_min(&pixels, channels);
        let max = intensity_max(&pixels, channels);
        let sum = intensity_sum(&pixels, channels);
        let mean = intensity_mean(&pixels, channels);
        let std = intensity_std(&pixels, channels);
        let median = intensity_median(&pixels);
        let mad = intensity_mad(&pixels);

        let results = descriptors(&pixels, channels);

        for i in 0..3 {
            assert_eq!(min[i], results[i]);
            assert_eq!(max[i], results[i + 3]);
            assert_eq!(sum[i], results[i + 6]);
            assert_eq!(mean[i], results[i + 9]);
            assert_eq!(std[i], results[i + 12]);
        }

        assert_eq!(median, results[3 + 12]);
        assert_eq!(mad, results[3 + 13]);
    }

    #[test]
    fn test_objects() {
        let (pixels, channels) = test_pixels();

        let min = intensity_min(&pixels, channels);
        let max = intensity_max(&pixels, channels);
        let sum = intensity_sum(&pixels, channels);
        let mean = intensity_mean(&pixels, channels);
        let std = intensity_std(&pixels, channels);
        let median = intensity_median(&pixels);
        let mad = intensity_mad(&pixels);

        let buffer = test_object();
        let object = buffer.crop_view(0, 0, 2, 2);
        let results = objects(&object);

        for i in 0..3 {
            assert_eq!(min[i], results[i]);
            assert_eq!(max[i], results[i + 3]);
            assert_eq!(sum[i], results[i + 6]);
            assert_eq!(mean[i], results[i + 9]);
            assert_eq!(std[i], results[i + 12]);
        }

        assert_eq!(median, results[3 + 12]);
        assert_eq!(mad, results[3 + 13]);
    }
}
