// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

/// Compute the convex hull for a set of (x, y) points
///
/// # Examples
///
/// ```
/// use thyme_core::cv::points::convex_hull;
///
/// let points = [[0., 1.], [1., 1.], [0.5, 0.5], [1., 0.], [0., 0.]];
/// let hull = convex_hull(&points);
///
/// assert_eq!(hull, [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
/// ```
pub fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });

    fn _ccw(p: [f32; 2], q: [f32; 2], r: [f32; 2]) -> bool {
        (q[1] - p[1]) * (r[0] - q[0]) > (q[0] - p[0]) * (r[1] - q[1])
    }

    let mut lower_hull = Vec::new();
    for &point in sorted_points.iter() {
        while lower_hull.len() >= 2
            && !_ccw(
                lower_hull[lower_hull.len() - 2],
                lower_hull[lower_hull.len() - 1],
                point,
            )
        {
            lower_hull.pop();
        }
        lower_hull.push(point);
    }

    let mut upper_hull = Vec::new();
    for &point in sorted_points.iter().rev() {
        while upper_hull.len() >= 2
            && !_ccw(
                upper_hull[upper_hull.len() - 2],
                upper_hull[upper_hull.len() - 1],
                point,
            )
        {
            upper_hull.pop();
        }
        upper_hull.push(point);
    }

    lower_hull.pop();
    upper_hull.pop();

    lower_hull.append(&mut upper_hull);
    lower_hull
}

/// Deduplicate redundant points for a set of (x,y) points
///
/// # Examples
///
/// ```
/// use thyme_core::cv::points::dedup_points;
///
/// let mut points = vec![[0., 1.], [1., 1.], [1., 1.], [1., 0.], [0., 0.]];
/// dedup_points(&mut points);
///
/// assert_eq!(points, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
/// ```
pub fn dedup_points(points: &mut Vec<[f32; 2]>) {
    const EPSILON: f32 = f32::EPSILON;

    points.sort_unstable_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });

    points.dedup_by(|a, b| (a[0] - b[0]).abs() < EPSILON && (a[1] - b[1]).abs() < EPSILON);
}

/// Resample points to a specified number of equidistant points
///
/// # Examples
///
/// ```
/// use thyme_core::cv::points::resample_points;
///
/// let mut points = vec![[0., 1.], [1., 1.], [1., 1.], [1., 0.], [0., 0.]];
/// resample_points(&mut points, 4);
///
/// assert_eq!(points, [[0.0, 1.0], [1.0, 0.6666666], [0.33333325, 0.0], [0.0, 1.0]]);
/// ```
pub fn resample_points(points: &mut Vec<[f32; 2]>, n_points: usize) {
    let is_closed = points[0] == points[points.len() - 1];
    if !is_closed {
        points.push(points[0]);
    }

    // Compute segment lengths
    let mut distances = Vec::with_capacity(points.len() - 1);
    for i in 0..points.len() - 1 {
        let dx = points[i + 1][0] - points[i][0];
        let dy = points[i + 1][1] - points[i][1];
        distances.push((dx * dx + dy * dy).sqrt());
    }

    // If very short path or repeated points, we replicate points
    let total_length: f32 = distances.iter().sum();
    if total_length == 0.0 {
        points.clear();
        points.extend(std::iter::repeat_n(points[0], n_points).take(n_points));
        return;
    }

    let mut cum_distances = Vec::with_capacity(points.len());
    cum_distances.push(0.0);
    for i in 0..distances.len() {
        cum_distances.push(cum_distances[i] + distances[i]);
    }

    let sample_distances: Vec<f32> = (0..n_points)
        .map(|i| i as f32 * total_length / (n_points - 1) as f32)
        .collect();

    let mut resampled = Vec::with_capacity(n_points);
    let mut j = 0;
    for &d in &sample_distances {
        while j < cum_distances.len() - 2 && d > cum_distances[j + 1] {
            j += 1;
        }

        let segment_length = distances[j];
        let t = if segment_length == 0.0 {
            0.0
        } else {
            (d - cum_distances[j]) / segment_length
        };

        let x = points[j][0] + t * (points[j + 1][0] - points[j][0]);
        let y = points[j][1] + t * (points[j + 1][1] - points[j][1]);
        resampled.push([x, y]);
    }

    if !is_closed {
        points.pop();
    }

    points.clear();
    points.extend(resampled);
}

/// Re-order outline points
///
/// # Examples
///
/// ```
/// use thyme_core::cv::points::order_points;
///
/// let mut points = [[1., 1.], [0., 1.], [1., 0.], [0., 0.]];
/// order_points(&mut points);
///
/// assert_eq!(points, [[0., 1.], [1., 1.], [1., 0.], [0., 0.]]);
/// ```
pub fn order_points(points: &mut [[f32; 2]]) {
    let n = points.len() as f32;

    let centroid = points
        .iter()
        .fold([0.0; 2], |acc, p| [acc[0] + p[0] / n, acc[1] + p[1] / n]);

    points.sort_by(|a, b| {
        let theta_a = (a[1] - centroid[1]).atan2(a[0] - centroid[0]);
        let theta_b = (b[1] - centroid[1]).atan2(b[0] - centroid[0]);
        if theta_a == theta_b {
            let dist_a = (a[0] - centroid[0]).powi(2) + (a[1] - centroid[1]).powi(2);
            let dist_b = (b[0] - centroid[0]).powi(2) + (b[1] - centroid[1]).powi(2);
            dist_a.partial_cmp(&dist_b).unwrap()
        } else {
            theta_b.partial_cmp(&theta_a).unwrap()
        }
    });
}

/// Mutably draw points onto a row-major canvas of specified size
///
/// # Arguments
///
/// * `buffer` - A row-major canvas for drawing points onto
/// * `width` - Width of canvas
/// * `height` - Height of canvas
/// * `points` - A set of (x, y) points
/// * `color` - A positive integer specifying fill color
///
/// # References
///
/// Adapted/modified from: https://github.com/image-rs/imageproc
///
/// # Examples
///
/// ```
/// use thyme_core::cv::points::draw_points_mut;
///
/// let width = 3;
/// let height = 3;
/// let mut buffer = vec![0, 0, 0, 0, 0, 0, 0, 0, 0];
/// let points = [[1., 1.], [0., 1.], [1., 0.], [0., 0.]];
///
/// draw_points_mut(&mut buffer, width, height, &points, 1);
///
/// assert_eq!(buffer, vec![1, 1, 0, 1, 1, 0, 0, 0, 0]);
/// ```
pub fn draw_points_mut(
    buffer: &mut [u32],
    width: u32,
    height: u32,
    points: &[[f32; 2]],
    color: u32,
) {
    // Helper function to draw a line between two points
    fn draw_line(
        buffer: &mut [u32],
        width: u32,
        height: u32,
        start: [f32; 2],
        end: [f32; 2],
        color: u32,
    ) {
        let (x0, y0) = (start[0] as i32, start[1] as i32);
        let (x1, y1) = (end[0] as i32, end[1] as i32);

        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut x = x0;
        let mut y = y0;

        loop {
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let index = (y as u32 * width + x as u32) as usize;
                buffer[index] = color;
            }

            if x == x1 && y == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }

    // Find the minimum and maximum y-coordinates
    let mut y_min = i32::MAX;
    let mut y_max = i32::MIN;
    for p in points {
        y_min = y_min.min(p[1] as i32);
        y_max = y_max.max(p[1] as i32);
    }

    y_min = y_min.max(0).min(height as i32 - 1);
    y_max = y_max.max(0).min(height as i32 - 1);

    // Close the polygon by connecting the last point to the first
    let mut closed: Vec<[f32; 2]> = points.to_vec();
    closed.push(points[0]);

    // Collect edges
    let edges: Vec<&[[f32; 2]]> = closed.windows(2).collect();
    let mut intersections = Vec::new();

    // Scanline algorithm to fill the polygon
    for y in y_min..=y_max {
        for edge in &edges {
            let p0 = edge[0];
            let p1 = edge[1];

            if (p0[1] <= y as f32 && p1[1] >= y as f32) || (p1[1] <= y as f32 && p0[1] >= y as f32)
            {
                if p0[1] == p1[1] {
                    intersections.push(p0[0] as i32);
                    intersections.push(p1[0] as i32);
                } else if p0[1] == y as f32 || p1[1] == y as f32 {
                    if p1[1] > y as f32 {
                        intersections.push(p0[0] as i32);
                    }
                    if p0[1] > y as f32 {
                        intersections.push(p1[0] as i32);
                    }
                } else {
                    let fraction = (y as f32 - p0[1]) / (p1[1] - p0[1]);
                    let inter = p0[0] + fraction * (p1[0] - p0[0]);
                    intersections.push(inter.round() as i32);
                }
            }
        }

        intersections.sort_unstable();
        intersections.chunks(2).for_each(|range| {
            let mut from = range[0].min(width as i32);
            let mut to = range[1].min(width as i32 - 1);
            if from < width as i32 && to >= 0 {
                from = from.max(0);
                to = to.max(0);

                for x in from..=to {
                    let index = (y as u32 * width + x as u32) as usize;
                    buffer[index] = color;
                }
            }
        });

        intersections.clear();
    }

    // Draw the edges of the polygon
    for edge in &edges {
        let start = edge[0];
        let end = edge[1];
        draw_line(buffer, width, height, start, end, color);
    }
}

/// Draw points onto a row-major canvas with padding
///
/// For any set of input points, the bounding box is estimated
/// and then a canvas the size of the bounding box plus specified
/// padding is initialized. The points are drawn in the approximate
/// center of this mask by shifting points back to the origin.
///
/// # Arguments
///
/// * `points` - A set of (x, y) points
/// * `pad` - Number of pixels to pad all sides of the canvas
///
/// # Examples
///
/// ```
/// use thyme_core::cv::points::draw_points;
///
/// let points = [[1., 1.], [0., 1.], [1., 0.], [0., 0.]];
/// let buffer = draw_points(&points, 1);
///
/// assert_eq!(buffer, vec![0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]);
/// ```
pub fn draw_points(points: &[[f32; 2]], pad: u32) -> Vec<u32> {
    let &[fx, fy] = &points[0];

    let mut min_x = fx;
    let mut min_y = fy;
    let mut max_x = fx;
    let mut max_y = fy;

    for &[x, y] in points {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    let (w, h, pad) = (max_x - min_x, max_y - min_y, pad as f32);

    let scale_x = w / (max_x - min_x);
    let scale_y = h / (max_y - min_y);
    let scale = scale_x.min(scale_y);

    let offset_x = pad + (w - (max_x - min_x) * scale) / 2.0;
    let offset_y = pad + (h - (max_y - min_y) * scale) / 2.0;

    let scaled: Vec<[f32; 2]> = points
        .iter()
        .map(|p| {
            let px = ((p[0] - min_x) * scale + offset_x).round();
            let py = ((p[1] - min_y) * scale + offset_y).round();
            [px, py]
        })
        .collect();

    let cw = (w + 1.0 + 2.0 * pad) as u32;
    let ch = (h + 1.0 + 2.0 * pad) as u32;

    let mut buffer = vec![0u32; (cw * ch) as usize];

    draw_points_mut(&mut buffer, cw, ch, &scaled, 1u32);

    buffer
}

/// Draw centered and filled points onto a canvas
pub fn draw_centered_points(
    width: u32,
    height: u32,
    points: &[[f32; 2]],
    color: u32,
    pad: u32,
) -> Vec<u32> {
    let mut buffer = vec![0; (width * height) as usize];

    let draw_width = width - 2 * pad;
    let draw_height = height - 2 * pad;

    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;

    for p in points {
        x_min = x_min.min(p[0]);
        x_max = x_max.max(p[0]);
        y_min = y_min.min(p[1]);
        y_max = y_max.max(p[1]);
    }

    let bbox_width = x_max - x_min;
    let bbox_height = y_max - y_min;

    let scale_x = draw_width as f32 / bbox_width;
    let scale_y = draw_height as f32 / bbox_height;
    let scale = scale_x.min(scale_y);

    let offset_x = (draw_width as f32 - bbox_width * scale) / 2.0 - x_min * scale + pad as f32;
    let offset_y = (draw_height as f32 - bbox_height * scale) / 2.0 - y_min * scale + pad as f32;

    let transformed_points: Vec<[f32; 2]> = points
        .iter()
        .map(|p| [p[0] * scale + offset_x, p[1] * scale + offset_y])
        .collect();

    draw_points_mut(&mut buffer, width, height, &transformed_points, color);

    buffer
}

/// Compute the distance from a point to a line segment
pub fn point_to_segment_distance(px: f32, py: f32, p1: [f32; 2], p2: [f32; 2]) -> f32 {
    let (x1, y1, x2, y2) = (p1[0], p1[1], p2[0], p2[1]);

    let dx = x2 - x1;
    let dy = y2 - y1;

    if dx == 0.0 && dy == 0.0 {
        return ((px - x1).powi(2) + (py - y1).powi(2)).sqrt();
    }

    let t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
    let t_clamped = t.clamp(0.0, 1.0);

    let closest_x = x1 + t_clamped * dx;
    let closest_y = y1 + t_clamped * dy;

    ((px - closest_x).powi(2) + (py - closest_y).powi(2)).sqrt()
}
