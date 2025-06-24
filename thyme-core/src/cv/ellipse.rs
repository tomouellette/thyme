// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use nalgebra::{DVector, MatrixXx5};

use crate::cv::points::resample_points;

/// Fit a best fitting ellipse to a set of points and extract elliptic parameters
///
/// # Arguments
///
/// * `points` - A set of ordered and deduplicated points
///
/// # Examples
///
/// ```
/// use thyme_core::cv::ellipse::fit_ellipse_lstsq;
/// let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
/// let params = fit_ellipse_lstsq(&points);
/// ```
#[inline]
pub fn fit_ellipse_lstsq(points: &[[f32; 2]]) -> [f32; 4] {
    let mut points = if points[0] != points[points.len() - 1] {
        let mut new_points = points.to_owned();
        new_points.push(points[0]);
        new_points
    } else {
        points.to_owned()
    };

    // We resample here as it seems to lead to more stable fits
    // when doing some anecdotal testing. We could come back to
    // this function and be a bit more rigorous.
    if points.len() < 32 {
        resample_points(&mut points, 32)
    };

    let (cx, cy) = points
        .iter()
        .skip(1)
        .fold((0.0, 0.0), |(cx, cy), p| (cx + p[0], cy + p[1]));

    let n = (points.len() - 1) as f32;
    let points: Vec<[f32; 2]> = points
        .iter()
        .map(|p| [p[0] - cx / n, p[1] - cy / n])
        .collect();

    let design: MatrixXx5<f32> = MatrixXx5::from_columns(&[
        DVector::from_iterator(points.len(), points.iter().map(|p| p[0] * p[0])),
        DVector::from_iterator(points.len(), points.iter().map(|p| p[0] * p[1])),
        DVector::from_iterator(points.len(), points.iter().map(|p| p[1] * p[1])),
        DVector::from_iterator(points.len(), points.iter().map(|p| p[0])),
        DVector::from_iterator(points.len(), points.iter().map(|p| p[1])),
    ]);

    let y = DVector::from_iterator(points.len(), points.iter().map(|_| 1.0_f32));

    let epsilon = 1e-8;
    let results = lstsq::lstsq(&design, &y, epsilon).unwrap();

    let a: f32 = results.solution[0];
    let b: f32 = results.solution[1] / 2.0;
    let c: f32 = results.solution[2];
    let d: f32 = results.solution[3] / 2.0;
    let f: f32 = results.solution[4] / 2.0;
    let g: f32 = -1.0;

    let denominator = b * b - a * c;
    let numerator = 2.0 * (a * f * f + c * d * d + g * b * b - 2.0 * b * d * f - a * c * g);
    let factor = ((a - c) * (a - c) + 4.0 * b * b).sqrt();

    let mut axis_length_major = (numerator / denominator / (factor - a - c)).sqrt();
    let mut axis_length_minor = (numerator / denominator / (-factor - a - c)).sqrt();

    let mut width_gt_height = true;
    if axis_length_major < axis_length_minor {
        width_gt_height = false;
        std::mem::swap(&mut axis_length_major, &mut axis_length_minor);
    }

    let mut r = (axis_length_minor / axis_length_major).powf(2.0);
    r = if r > 1.0 { 1.0 / r } else { r };
    let eccentricity = (1.0 - r).sqrt();

    let mut phi = if b == 0.0 {
        if a < c {
            0.0
        } else {
            std::f32::consts::PI / 2.0
        }
    } else {
        let mut inner = ((2.0 * b) / (a - c)).atan() / 2.0;
        inner += if a > c {
            std::f32::consts::PI / 2.0
        } else {
            0.0
        };
        inner
    };

    phi += if !width_gt_height {
        std::f32::consts::PI / 2.0
    } else {
        0.0
    };
    phi %= std::f32::consts::PI;

    [
        axis_length_major * 2.0,
        axis_length_minor * 2.0,
        eccentricity,
        phi,
    ]
}
