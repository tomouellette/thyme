// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use crate::cv::ellipse::fit_ellipse_lstsq;
use crate::cv::points::{convex_hull, point_to_segment_distance};

#[inline]
pub fn area(points: &[[f32; 2]]) -> f32 {
    let mut area = 0.0;

    let n = points.len();
    for i in 0..n - 1 {
        let p1 = points[i];
        let p2 = points[i + 1];
        area += p1[0] * p2[1] - p2[0] * p1[1];
    }

    if points[0] != points[n - 1] {
        let p1 = points[n - 1];
        let p2 = points[0];
        area += p1[0] * p2[1] - p2[0] * p1[1];
    }

    area.abs() / 2.0
}

#[inline]
pub fn area_bbox(points: &[[f32; 2]]) -> f32 {
    let (mut xmin, mut ymin) = (points[0][0], points[0][1]);
    let (mut xmax, mut ymax) = (points[0][0], points[0][1]);

    for point in points.iter().skip(1) {
        xmin = if point[0] < xmin { point[0] } else { xmin };
        ymin = if point[1] < ymin { point[1] } else { ymin };
        xmax = if point[0] > xmax { point[0] } else { xmax };
        ymax = if point[1] > ymax { point[1] } else { ymax };
    }

    (xmax - xmin) * (ymax - ymin)
}

#[inline]
pub fn area_convex(points: &[[f32; 2]]) -> f32 {
    area(&convex_hull(points))
}

#[inline]
pub fn perimeter(points: &[[f32; 2]]) -> f32 {
    let n_points = points.len();

    let mut perimeter = 0.0;

    for i in 0..n_points - 1 {
        let dx = points[i][0] - points[i + 1][0];
        let dy = points[i][1] - points[i + 1][1];
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    if points[0] != points[n_points - 1] {
        let dx = points[points.len() - 1][0] - points[0][0];
        let dy = points[points.len() - 1][1] - points[0][1];
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    perimeter
}

#[inline]
pub fn centroid(points: &[[f32; 2]]) -> [f32; 2] {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut area = 0.0;

    let n = points.len();
    let is_closed = points[0] == points[n - 1];
    let n_end = if is_closed { n - 1 } else { n };

    for i in 0..n_end {
        let j = (i + 1) % n_end;
        let p1 = points[i];
        let p2 = points[j];
        let cross = p1[0] * p2[1] - p2[0] * p1[1];
        sum_x += (p1[0] + p2[0]) * cross;
        sum_y += (p1[1] + p2[1]) * cross;
        area += cross;
    }

    area /= 2.0;
    [sum_x / (6.0 * area.abs()), sum_y / (6.0 * area.abs())]
}

#[inline]
pub fn center(points: &[[f32; 2]]) -> [f32; 2] {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;

    let n = points.len();
    let is_closed = points[0] == points[n - 1];
    let n_end = if is_closed { n - 1 } else { n };

    for point in points.iter().take(n_end) {
        sum_x += point[0];
        sum_y += point[1];
    }

    [sum_x / n_end as f32, sum_y / n_end as f32]
}

#[inline]
pub fn elongation(points: &[[f32; 2]]) -> f32 {
    let (mut xmin, mut ymin) = (points[0][0], points[0][1]);
    let (mut xmax, mut ymax) = (points[0][0], points[0][1]);

    for point in points.iter().skip(1) {
        xmin = if point[0] < xmin { point[0] } else { xmin };
        ymin = if point[1] < ymin { point[1] } else { ymin };
        xmax = if point[0] > xmax { point[0] } else { xmax };
        ymax = if point[1] > ymax { point[1] } else { ymax };
    }

    let elongation = (xmax - xmin) / (ymax - ymin);

    if elongation > 1.0 {
        1.0 / elongation
    } else {
        elongation
    }
}

#[inline]
pub fn thread_length(points: &[[f32; 2]]) -> f32 {
    let perimeter = perimeter(points);
    let area = area(points);

    let left = perimeter.powi(2);
    let right = 16.0 * area;

    let coefficient = if left <= right {
        0.0
    } else {
        (left - right).sqrt()
    };

    (perimeter + coefficient) / 4.0
}

#[inline]
pub fn thread_width(points: &[[f32; 2]]) -> f32 {
    let perimeter = perimeter(points);
    let area = area(points);

    let left = perimeter.powi(2);
    let right = 16.0 * area;

    let coefficient = if left <= right {
        0.0
    } else {
        (left - right).sqrt()
    };

    let thread_length = (perimeter + coefficient) / 4.0;
    area / thread_length
}

#[inline]
pub fn solidity(points: &[[f32; 2]]) -> f32 {
    let area = area(points);
    let area_convex = area_convex(points);

    area / area_convex
}

#[inline]
pub fn extent(points: &[[f32; 2]]) -> f32 {
    let area = area(points);
    let area_bbox = area_bbox(points);

    area / area_bbox
}

#[inline]
pub fn form_factor(points: &[[f32; 2]]) -> f32 {
    let perimeter = perimeter(points);
    let area = area(points);

    (4.0 * std::f32::consts::PI * area) / (perimeter * perimeter)
}

#[inline]
pub fn equivalent_diameter(points: &[[f32; 2]]) -> f32 {
    (area(points) / std::f32::consts::PI).sqrt() * 2.0
}

#[inline]
pub fn eccentricity(points: &[[f32; 2]]) -> f32 {
    let ellipse = fit_ellipse_lstsq(points);
    ellipse[2]
}

#[inline]
pub fn major_axis_length(points: &[[f32; 2]]) -> f32 {
    let ellipse = fit_ellipse_lstsq(points);
    ellipse[0]
}

#[inline]
pub fn minor_axis_length(points: &[[f32; 2]]) -> f32 {
    let ellipse = fit_ellipse_lstsq(points);
    ellipse[1]
}

#[inline]
pub fn min_radius(points: &[[f32; 2]]) -> f32 {
    let [cx, cy] = centroid(points);
    let mut min_radius = f32::MAX;

    for i in 0..points.len() {
        let p1 = points[i];
        let p2 = points[(i + 1) % points.len()];

        let distance = point_to_segment_distance(cx, cy, p1, p2);
        if distance < min_radius {
            min_radius = distance;
        }
    }

    min_radius
}

#[inline]
pub fn max_radius(points: &[[f32; 2]]) -> f32 {
    let [x_centroid, y_centroid] = centroid(points);
    let mut maximum_radius = 0.0;

    for point in points.iter() {
        let (x, y) = (point[0], point[1]);
        let distance = (x_centroid - x) * (x_centroid - x) + (y_centroid - y) * (y_centroid - y);
        if distance > maximum_radius {
            maximum_radius = distance;
        }
    }

    maximum_radius.sqrt()
}

#[inline]
pub fn mean_radius(points: &[[f32; 2]]) -> f32 {
    let [x_centroid, y_centroid] = centroid(points);
    let include_last = (points.last().unwrap() == &points[0]) as usize;

    let mut mean_radius = 0.0;
    for point in points.iter().take(points.len() - include_last) {
        let (x, y) = (point[0], point[1]);
        let distance = (x_centroid - x) * (x_centroid - x) + (y_centroid - y) * (y_centroid - y);
        mean_radius += distance.sqrt();
    }

    mean_radius / (points.len() - include_last) as f32
}

#[inline]
pub fn min_feret(points: &[[f32; 2]]) -> f32 {
    let norm = |x: [f32; 2]| -> f32 { (x[0] * x[0] + x[1] * x[1]).sqrt() };
    let cross_product = |x: [f32; 2], y: [f32; 2]| -> f32 { x[0] * y[1] - x[1] * y[0] };
    let n_points = points.len();

    let mut feret_diameter_minimum = f32::MAX;
    for i in 0..n_points {
        let p1 = points[i];
        let p2 = points[(i + 1) % n_points];
        let diff_a = [p2[0] - p1[0], p2[1] - p1[1]];

        let mut distance = 0.0;
        for point in points.iter().take(n_points) {
            let diff_b = [point[0] - p1[0], point[1] - p1[1]];
            let d = (cross_product(diff_a, diff_b) / norm(diff_a)).abs();

            if d > distance {
                distance = d;
            }
        }

        if distance < feret_diameter_minimum && distance > f32::EPSILON {
            feret_diameter_minimum = distance;
        }
    }

    feret_diameter_minimum
}

#[inline]
pub fn max_feret(points: &[[f32; 2]]) -> f32 {
    let mut max_diameter = 0f32;
    let n_points = points.len();

    for i in 0..n_points {
        for j in (i + 1)..n_points {
            let dx = points[j][0] - points[i][0];
            let dy = points[j][1] - points[i][1];

            if dx == 0.0 && dy == 0.0 {
                continue;
            }

            let mut min_proj = f32::INFINITY;
            let mut max_proj = f32::NEG_INFINITY;

            for k in 0..n_points {
                let proj = (points[k][0] - points[i][0]) * dx + (points[k][1] - points[i][1]) * dy;

                min_proj = min_proj.min(proj);
                max_proj = max_proj.max(proj);
            }

            let diameter = ((max_proj - min_proj) / ((dx * dx + dy * dy).sqrt())).abs();
            max_diameter = max_diameter.max(diameter);
        }
    }

    max_diameter
}

#[inline]
pub fn descriptors(points: &[[f32; 2]]) -> [f32; 23] {
    let n = points.len();
    let is_closed = points[0] == points[n - 1];
    let n_end = if is_closed { n - 1 } else { n };

    let mut area = 0f32;
    let mut perimeter = 0f32;

    let mut sum_x = 0f32;
    let mut sum_y = 0f32;
    let mut mean_x = 0f32;
    let mut mean_y = 0f32;

    let (mut xmin, mut ymin) = (points[0][0], points[0][1]);
    let (mut xmax, mut ymax) = (points[0][0], points[0][1]);

    for i in 0..n {
        let p1 = points[i];
        let p2 = points[(i + 1) % n];

        // Area and centroid
        if i < n_end {
            let cross = p1[0] * p2[1] - p2[0] * p1[1];
            area += cross;
            sum_x += (p1[0] + p2[0]) * cross;
            sum_y += (p1[1] + p2[1]) * cross;
        }

        // Perimeter
        if !is_closed || (i + 1) % n != 0 {
            let dx = p1[0] - p2[0];
            let dy = p1[1] - p2[1];
            perimeter += (dx * dx + dy * dy).sqrt();
        }

        // Bounding box
        xmin = xmin.min(p1[0]);
        ymin = ymin.min(p1[1]);
        xmax = xmax.max(p1[0]);
        ymax = ymax.max(p1[1]);

        // Center
        if i < n_end {
            mean_x += p1[0];
            mean_y += p1[1];
        }
    }

    let area = area.abs() / 2.0;
    let area_bbox = (xmax - xmin) * (ymax - ymin);

    let centroid_x = sum_x / (6.0 * area.abs());
    let centroid_y = sum_y / (6.0 * area.abs());
    let center_x = mean_x / n_end as f32;
    let center_y = mean_y / n_end as f32;

    let elongation = {
        let e = (xmax - xmin) / (ymax - ymin);
        if e > 1.0 { 1.0 / e } else { e }
    };

    let mut minimum_radius = f32::MAX;
    let mut maximum_radius = 0f32;
    let mut mean_radius = 0f32;

    let mut min_feret = f32::MAX;
    let mut max_feret = 0f32;

    for i in 0..n {
        let p1 = points[i];
        let p2 = points[(i + 1) % n];

        // Min radius
        minimum_radius =
            minimum_radius.min(point_to_segment_distance(centroid_x, centroid_y, p1, p2));

        // Max and mean radius
        if !is_closed || i < n - 1 {
            let dx = centroid_x - p1[0];
            let dy = centroid_y - p1[1];
            let distance_sq = dx * dx + dy * dy;
            maximum_radius = maximum_radius.max(distance_sq);
            mean_radius += distance_sq.sqrt();
        }

        // Feret diameters
        let diff_a = [p2[0] - p1[0], p2[1] - p1[1]];
        let norm_a = (diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1]).sqrt();

        let mut max_distance = 0f32;
        for point in points {
            let diff_b = [point[0] - p1[0], point[1] - p1[1]];
            let d = (diff_a[0] * diff_b[1] - diff_a[1] * diff_b[0]).abs() / norm_a;
            max_distance = max_distance.max(d);
        }

        if max_distance < min_feret && max_distance > f32::EPSILON {
            min_feret = max_distance;
        }
    }

    maximum_radius = maximum_radius.sqrt();
    mean_radius /= n_end as f32;

    // Max feret
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[j][0] - points[i][0];
            let dy = points[j][1] - points[i][1];

            if dx == 0.0 && dy == 0.0 {
                continue;
            }

            let norm = (dx * dx + dy * dy).sqrt();

            let mut min_proj = f32::INFINITY;
            let mut max_proj = f32::NEG_INFINITY;

            for point in points {
                let proj =
                    ((point[0] - points[i][0]) * dx + ((point[1] - points[i][1]) * dy)) / norm;
                min_proj = min_proj.min(proj);
                max_proj = max_proj.max(proj);
            }

            max_feret = max_feret.max((max_proj - min_proj).abs());
        }
    }

    // Convex hull
    let area_convex = {
        let convex_hull_points = convex_hull(points);

        let mut area = 0.0;
        let n_hull = convex_hull_points.len();
        for i in 0..n_hull - 1 {
            let p1 = convex_hull_points[i];
            let p2 = convex_hull_points[i + 1];
            area += p1[0] * p2[1] - p2[0] * p1[1];
        }

        if convex_hull_points[0] != convex_hull_points[n_hull - 1] {
            let p1 = convex_hull_points[n_hull - 1];
            let p2 = convex_hull_points[0];
            area += p1[0] * p2[1] - p2[0] * p1[1];
        }

        area.abs() / 2.0
    };

    // Ellipse fitting
    let ellipse = fit_ellipse_lstsq(points);
    let major_axis = ellipse[0];
    let minor_axis = ellipse[1];
    let eccentricity = ellipse[2];

    // Thread width and height
    let thread_left = perimeter.powi(2);
    let thread_right = 16.0 * area;
    let thread_coefficient = if thread_left <= thread_right {
        0.0
    } else {
        (thread_left - thread_right).sqrt()
    };
    let thread_length = (perimeter + thread_coefficient) / 4.0;
    let thread_width = area / thread_length;

    // Refine
    let solidity = area / area_convex;
    let extent = area / area_bbox;
    let form_factor = (4.0 * std::f32::consts::PI * area) / (perimeter * perimeter);
    let equivalent_diameter = (area / std::f32::consts::PI).sqrt() * 2.0;

    [
        centroid_x,
        centroid_y,
        center_x,
        center_y,
        area,
        area_bbox,
        area_convex,
        perimeter,
        elongation,
        thread_length,
        thread_width,
        solidity,
        extent,
        form_factor,
        equivalent_diameter,
        eccentricity,
        major_axis,
        minor_axis,
        minimum_radius,
        maximum_radius,
        mean_radius,
        min_feret,
        max_feret,
    ]
}

#[cfg(test)]
mod test {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn unit_circle(close: bool) -> Vec<[f32; 2]> {
        let mut points = Vec::with_capacity(360);
        for i in 0..360 {
            let t = 2.0 * std::f32::consts::PI * i as f32 / 360f32;
            points.push([t.cos(), t.sin()]);
        }

        if close {
            points.push(points[0]);
        }

        points
    }

    fn unit_circle_shifted(close: bool) -> Vec<[f32; 2]> {
        let mut points = Vec::with_capacity(360);
        for i in 0..360 {
            let t = 2.0 * std::f32::consts::PI * i as f32 / 360f32;
            points.push([t.cos() + 1.0, t.sin() + 2.0]);
        }

        if close {
            points.push(points[0]);
        }

        points
    }

    fn unit_square(close: bool) -> Vec<[f32; 2]> {
        let mut points = vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        if close {
            points.push(points[0]);
        }
        points
    }

    fn test_equivalence(f: fn(&[[f32; 2]]) -> f32) {
        let open_circle = unit_circle(false);
        let open_square = unit_square(false);
        let closed_circle = unit_circle(true);
        let closed_square = unit_square(true);

        assert_eq!(f(&open_circle), f(&closed_circle));
        assert_eq!(f(&open_square), f(&closed_square));
    }

    #[test]
    fn test_centroid() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let xy = centroid(&circle);
            assert!((xy[0] - 0.0).abs() < EPSILON);
            assert!((xy[1] - 0.0).abs() < EPSILON);

            let square = unit_square(close);
            let xy = centroid(&square);
            assert!((xy[0] - 0.0).abs() < EPSILON);
            assert!((xy[1] - 0.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_centroid_shifted() {
        for close in [true, false] {
            let circle = unit_circle_shifted(close);

            let xy = centroid(&circle);
            assert!((xy[0] - 1.0).abs() < EPSILON);
            assert!((xy[1] - 2.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_center() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let xy = center(&circle);
            assert!((xy[0] - 0.0).abs() < EPSILON);
            assert!((xy[1] - 0.0).abs() < EPSILON);

            let square = unit_square(close);
            let xy = center(&square);
            assert!((xy[0] - 0.0).abs() < EPSILON);
            assert!((xy[1] - 0.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_area() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let area_polygon = area(&circle);
            let circle_area = 0.5 * 360f32 * (2.0 * std::f32::consts::PI / 360f32).sin();
            assert!((area_polygon - circle_area).abs() < EPSILON);

            let square = unit_square(close);
            let area_polygon = area(&square);
            assert!((area_polygon - 4.0).abs() < EPSILON);
        }

        test_equivalence(area);
    }

    #[test]
    fn test_area_bbox() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let area = area_bbox(&circle);
            assert!((area - 4.0).abs() < EPSILON);

            let square = unit_square(close);
            let area = area_bbox(&square);
            assert!((area - 4.0).abs() < EPSILON);
        }

        test_equivalence(area_bbox);
    }

    #[test]
    fn test_area_convex() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let area = area_convex(&circle);
            let circle_area = 0.5 * 360f32 * (2.0 * std::f32::consts::PI / 360f32).sin();
            assert!((area - circle_area).abs() < EPSILON);

            let square = unit_square(close);
            let area = area_convex(&square);
            assert!((area - 4.0).abs() < EPSILON);
        }

        test_equivalence(area_convex);
    }

    #[test]
    fn test_perimeter() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let perimeter_polygon = perimeter(&circle);
            let circle_perimeter = 360f32 * 2.0 * (std::f32::consts::PI / 360f32).sin();
            assert!((perimeter_polygon - circle_perimeter).abs() < 1e-4);

            let square = unit_square(close);
            let perimeter_polygon = perimeter(&square);
            assert!((perimeter_polygon - 8.0).abs() < EPSILON);
        }

        test_equivalence(perimeter);
    }

    #[test]
    fn test_elongation() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let elongation_polygon = elongation(&circle);
            assert!((elongation_polygon - 1.0).abs() < EPSILON);

            let square = unit_square(close);
            let elongation_polygon = elongation(&square);
            assert!((elongation_polygon - 1.0).abs() < EPSILON);
        }

        test_equivalence(elongation);
    }

    #[test]
    fn test_solidity() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let solidity_polygon = solidity(&circle);
            assert!((solidity_polygon - 1.0).abs() < EPSILON);

            let square = unit_square(close);
            let solidity_polygon = solidity(&square);
            assert!((solidity_polygon - 1.0).abs() < EPSILON);
        }

        test_equivalence(solidity);
    }

    #[test]
    fn test_extent() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let extent_polygon = extent(&circle);
            let circle_area = 0.5 * 360f32 * (2.0 * std::f32::consts::PI / 360f32).sin();
            assert!((extent_polygon - circle_area / 4.0).abs() < EPSILON);

            let square = unit_square(close);
            let extent_polygon = extent(&square);
            assert!((extent_polygon - 1.0).abs() < EPSILON);
        }

        test_equivalence(extent);
    }

    #[test]
    fn test_form_factor() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let form_factor_polygon = form_factor(&circle);
            let circle_area = 0.5 * 360f32 * (2.0 * std::f32::consts::PI / 360f32).sin();
            let circle_perimeter = 360f32 * 2.0 * (std::f32::consts::PI / 360f32).sin();
            assert!(
                (form_factor_polygon
                    - (4.0 * std::f32::consts::PI * circle_area)
                        / (circle_perimeter * circle_perimeter))
                    .abs()
                    < EPSILON
            );

            let square = unit_square(close);
            let form_factor_polygon = form_factor(&square);
            assert!((form_factor_polygon - std::f32::consts::PI / 4.0).abs() < EPSILON);
        }

        test_equivalence(form_factor);
    }

    #[test]
    fn test_thread_length() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let thread_length_polygon = thread_length(&circle);
            let circle_perimeter = 360f32 * 2.0 * (std::f32::consts::PI / 360f32).sin();
            assert!((thread_length_polygon - circle_perimeter / 4.0).abs() < EPSILON);

            let square = unit_square(close);
            let thread_length_polygon = thread_length(&square);
            assert!((thread_length_polygon - 2.0).abs() < EPSILON);
        }

        test_equivalence(thread_length);
    }

    #[test]
    fn test_thread_width() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let thread_width_polygon = thread_width(&circle);
            let circle_area = 0.5 * 360f32 * (2.0 * std::f32::consts::PI / 360f32).sin();
            let circle_perimeter = 360f32 * 2.0 * (std::f32::consts::PI / 360f32).sin();
            assert!((thread_width_polygon - (4.0 * circle_area) / circle_perimeter).abs() < 1e-4);

            let square = unit_square(close);
            let thread_width_polygon = thread_width(&square);
            assert!((thread_width_polygon - 2.0).abs() < EPSILON);
        }

        test_equivalence(thread_width);
    }

    #[test]
    fn feret_diameter_maximum() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let feret_diameter_maximum = max_feret(&circle);
            // The discrete approximation of a unit circle with points
            // leads to slightly higher error from continuous expectation
            assert!((feret_diameter_maximum - 2.0).abs() < 1e-4);

            let square = unit_square(close);
            let feret_diameter_maximum = max_feret(&square);
            assert!((feret_diameter_maximum - 8_f32.sqrt()).abs() < EPSILON);
        }

        test_equivalence(max_feret);
    }

    #[test]
    fn test_feret_diameter_minimum() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let feret_diameter_minimum = min_feret(&circle);
            // The discrete approximation of a unit circle with points
            // leads to slightly higher error from continuous expectation
            assert!((feret_diameter_minimum - 2.0).abs() < 1e-4);

            let square = unit_square(close);
            let feret_diameter_minimum = min_feret(&square);
            assert!((feret_diameter_minimum - 2.0).abs() < EPSILON);
        }

        test_equivalence(min_feret);
    }

    #[test]
    fn test_min_radius() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let min_radius_polygon = min_radius(&circle);
            assert!((min_radius_polygon - 1.0).abs() < 1e-4);

            let square = unit_square(close);
            let min_radius_polygon = min_radius(&square);
            assert!((min_radius_polygon - 1.0).abs() < EPSILON);
        }

        test_equivalence(min_radius);
    }

    #[test]
    fn test_min_radius_shifted() {
        for close in [true, false] {
            let circle = unit_circle_shifted(close);
            let min_radius_polygon = min_radius(&circle);
            assert!((min_radius_polygon - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_max_radius() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let max_radius_polygon = max_radius(&circle);
            assert!((max_radius_polygon - 1.0).abs() < EPSILON);

            let square = unit_square(close);
            let max_radius_polygon = max_radius(&square);
            let h = (2.0_f32).sqrt();
            assert!((max_radius_polygon - h).abs() < EPSILON);
        }

        test_equivalence(max_radius);
    }

    #[test]
    fn test_max_radius_shifted() {
        for close in [true, false] {
            let circle = unit_circle_shifted(close);
            let max_radius_polygon = min_radius(&circle);
            assert!((max_radius_polygon - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_mean_radius() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let mean_radius_polygon = mean_radius(&circle);
            assert!((mean_radius_polygon - 1.0).abs() < EPSILON);

            let square = unit_square(close);
            let mean_radius_polygon = mean_radius(&square);
            let h = (2.0_f32).sqrt();
            assert!((mean_radius_polygon - h).abs() < EPSILON);
        }

        test_equivalence(mean_radius);
    }

    #[test]
    fn test_eccentricity() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let eccentricity_polygon = eccentricity(&circle);
            // The discrete approximation to the unit circle results
            // in a higher error on eccentricity than if the circle
            // was completely continuous
            assert!((eccentricity_polygon - 0.0).abs() < 1e-2);
        }

        test_equivalence(eccentricity);
    }

    #[test]
    fn test_major_axis_length() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let major_axis_length_polygon = major_axis_length(&circle);
            assert!((major_axis_length_polygon - 2.0).abs() < EPSILON);
        }

        test_equivalence(major_axis_length);
    }

    #[test]
    fn test_minor_axis_length() {
        for close in [true, false] {
            let circle = unit_circle(close);
            let minor_axis_length_polygon = minor_axis_length(&circle);
            assert!((minor_axis_length_polygon - 2.0).abs() < EPSILON);
        }

        test_equivalence(minor_axis_length);
    }

    #[test]
    fn test_descriptors() {
        for close in [true, false] {
            for shifted in [true, false] {
                let points = if !shifted {
                    unit_circle(close)
                } else {
                    let mut points = vec![];
                    for point in unit_circle(close).iter().copied() {
                        points.push([point[0] + 10f32, point[1] + 12f32])
                    }
                    points
                };

                let descriptors = descriptors(&points);
                let centroid = centroid(&points);
                let center = center(&points);
                let area_polygon = area(&points);
                let area_bbox = area_bbox(&points);
                let area_convex = area_convex(&unit_circle(close));
                let perimeter = perimeter(&points);
                let elongation = elongation(&points);
                let solidity = solidity(&points);
                let extent = extent(&points);
                let form_factor = form_factor(&points);
                let equivalent_diameter = equivalent_diameter(&points);
                let major_axis = major_axis_length(&points);
                let minor_axis = minor_axis_length(&points);
                let thread_length = thread_length(&points);
                let thread_width = thread_width(&points);
                let minimum_radius = min_radius(&points);
                let maximum_radius = max_radius(&points);
                let mean_radius = mean_radius(&points);
                let eccentricity = eccentricity(&points);
                let min_feret = min_feret(&points);
                let max_feret = max_feret(&points);

                assert_eq!(descriptors[0], centroid[0]);
                assert_eq!(descriptors[1], centroid[1]);
                assert_eq!(descriptors[2], center[0]);
                assert_eq!(descriptors[3], center[1]);
                assert_eq!(descriptors[4], area_polygon);
                assert_eq!(descriptors[5], area_bbox);
                if shifted {
                    // Small precision difference when comparing single
                    // function implementation and batch descriptors calculation.
                    // This shouldn't materially impact interpretation of the output
                    // values - but keeping this here to make note of it.
                    assert!((descriptors[6] - area_convex).abs() < 1e-4);
                } else {
                    assert_eq!(descriptors[6], area_convex);
                }
                assert_eq!(descriptors[7], perimeter);
                assert_eq!(descriptors[8], elongation);
                assert_eq!(descriptors[9], thread_length);
                assert_eq!(descriptors[10], thread_width);
                assert_eq!(descriptors[11], solidity);
                assert_eq!(descriptors[12], extent);
                assert_eq!(descriptors[13], form_factor);
                assert_eq!(descriptors[14], equivalent_diameter);
                assert_eq!(descriptors[15], eccentricity);
                assert_eq!(descriptors[16], major_axis);
                assert_eq!(descriptors[17], minor_axis);
                assert_eq!(descriptors[18], minimum_radius);
                assert_eq!(descriptors[19], maximum_radius);
                assert_eq!(descriptors[20], mean_radius);
                assert_eq!(descriptors[21], min_feret);
                assert_eq!(descriptors[22], max_feret);
            }
        }
    }
}
