// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::cmp::Ordering;
use std::collections::VecDeque;

/// Find contours using 8-connectivity
///
/// # Arguments
///
/// * `width` - Width of mask
/// * `height` - Height of mask
/// * `buffer` - A row-major mask buffer
/// * `threshold` - Threshold value for foreground/background pixels
/// * `order` - An ordering specifying how a pixel will be compared to threshold
///
/// # References
///
/// Adapted/modified from: https://github.com/image-rs/imageproc
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering::Greater;
/// use thyme_core::cv::find_contours;
///
/// let width = 3;
/// let height = 3;
/// let buffer_one: Vec<u32> = vec![12, 12, 0, 12, 12, 0, 0, 0, 0];
/// let contours_one = find_contours(width, height, &buffer_one, &0, Greater);
///
/// assert_eq!(contours_one, [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]])
/// ```
pub fn find_contours(
    width: u32,
    height: u32,
    pixels: &[u32],
    threshold: &u32,
    order: Ordering,
) -> Vec<Vec<[f32; 2]>> {
    let width = width as usize;
    let height = height as usize;
    let padded_width = width + 2;
    let padded_height = height + 2;

    let at = |x: usize, y: usize| x + padded_width * y;

    let mut image_values = vec![0i32; padded_height * padded_width];

    for y in 0..height {
        for x in 0..width {
            let pixel = pixels[y * width + x];
            image_values[at(x + 1, y + 1)] = if pixel.cmp(threshold) == order { 1 } else { 0 };
        }
    }

    let mut diffs = VecDeque::from(vec![
        [-1, 0],  // West
        [-1, -1], // Northwest
        [0, -1],  // North
        [1, -1],  // Northeast
        [1, 0],   // East
        [1, 1],   // Southeast
        [0, 1],   // South
        [-1, 1],  // Southwest
    ]);

    let mut contours: Vec<Contour> = Vec::new();
    let mut curr_border_num = 1;
    let mut parent_border_num = 1;

    for y in 1..=height {
        for x in 1..=width {
            if image_values[at(x, y)] == 0 {
                continue;
            }

            let curr = (x as i32, y as i32);

            let (is_outer_border, adjacent_point) = if image_values[at(x, y)] == 1
                && x > 0
                && image_values[at(x - 1, y)] == 0
            {
                (true, (x as i32 - 1, y as i32))
            } else if image_values[at(x, y)] > 0 && x + 1 < width && image_values[at(x + 1, y)] == 0
            {
                if image_values[at(x, y)] > 1 {
                    parent_border_num = image_values[at(x, y)] as usize;
                }
                (false, (x as i32 + 1, y as i32))
            } else {
                continue;
            };

            curr_border_num += 1;

            let border_type = if is_outer_border {
                BorderType::Outer
            } else {
                BorderType::Hole
            };

            let parent = if parent_border_num > 1 {
                let parent_index = parent_border_num - 2;
                let parent_contour = &contours[parent_index];
                if (border_type == BorderType::Outer)
                    ^ (parent_contour.border_type == BorderType::Outer)
                {
                    Some(parent_index)
                } else {
                    parent_contour.parent
                }
            } else {
                None
            };

            let mut contour_points: Vec<[f32; 2]> = Vec::new();
            rotate_to_value(
                &mut diffs,
                [adjacent_point.0 - curr.0, adjacent_point.1 - curr.1],
            );

            let pos1_option = diffs.iter().find_map(|&diff| {
                let nx = curr.0 + diff[0];
                let ny = curr.1 + diff[1];
                if nx >= 0
                    && nx < padded_width as i32
                    && ny >= 0
                    && ny < padded_height as i32
                    && image_values[at(nx as usize, ny as usize)] != 0
                {
                    Some((nx, ny))
                } else {
                    None
                }
            });

            if let Some(pos1) = pos1_option {
                let mut pos2 = pos1;
                let mut pos3 = curr;

                loop {
                    contour_points.push([pos3.0 as f32 - 1.0, pos3.1 as f32 - 1.0]);
                    rotate_to_value(&mut diffs, [pos2.0 - pos3.0, pos2.1 - pos3.1]);
                    let pos4 = diffs
                        .iter()
                        .rev()
                        .find_map(|&diff| {
                            let nx = pos3.0 + diff[0];
                            let ny = pos3.1 + diff[1];
                            if nx >= 0
                                && nx < padded_width as i32
                                && ny >= 0
                                && ny < padded_height as i32
                                && image_values[at(nx as usize, ny as usize)] != 0
                            {
                                Some((nx, ny))
                            } else {
                                None
                            }
                        })
                        .unwrap();

                    let mut is_right_edge = false;
                    for &diff in diffs.iter().rev() {
                        if diff == [pos4.0 - pos3.0, pos4.1 - pos3.1] {
                            break;
                        }
                        if diff == [1, 0] {
                            is_right_edge = true;
                            break;
                        }
                    }

                    if pos3.0 as usize + 1 == padded_width || is_right_edge {
                        image_values[at(pos3.0 as usize, pos3.1 as usize)] = -curr_border_num;
                    } else if image_values[at(pos3.0 as usize, pos3.1 as usize)] == 1 {
                        image_values[at(pos3.0 as usize, pos3.1 as usize)] = curr_border_num;
                    }

                    if pos4 == curr && pos3 == pos1 {
                        break;
                    }

                    pos2 = pos3;
                    pos3 = pos4;
                }
            } else {
                contour_points.push([x as f32 - 1.0, y as f32 - 1.0]);
                image_values[at(x, y)] = -curr_border_num;
            }

            contours.push(Contour::new(contour_points, border_type, parent));
        }
    }

    contours
        .into_iter()
        .filter(|contour| contour.border_type() == &BorderType::Outer)
        .map(|contour| contour.into_points())
        .collect()
}

///  Contour for storiing outlines of segmented objects
#[derive(Debug, Clone)]
pub struct Contour {
    points: Vec<[f32; 2]>,
    border_type: BorderType,
    parent: Option<usize>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BorderType {
    Outer,
    Hole,
}

impl Contour {
    pub fn new(points: Vec<[f32; 2]>, border_type: BorderType, parent: Option<usize>) -> Self {
        Contour {
            points,
            border_type,
            parent,
        }
    }

    pub fn as_points(&self) -> &Vec<[f32; 2]> {
        &self.points
    }

    pub fn into_points(self) -> Vec<[f32; 2]> {
        self.points
    }

    pub fn border_type(&self) -> &BorderType {
        &self.border_type
    }

    pub fn parent(&self) -> Option<usize> {
        self.parent
    }
}

fn rotate_to_value(values: &mut VecDeque<[i32; 2]>, value: [i32; 2]) {
    if let Some(pos) = values.iter().position(|&v| v == value) {
        values.rotate_left(pos);
    }
}

/// Find contours incrementally for each object denoted by a unique non-zero integer
///
/// # Arguments
///
/// * `width` - Width of mask
/// * `height` - Height of mask
/// * `buffer` - A row-major mask buffer
/// * `labels` - Positive non-zero integers specifying unique segmented objects
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering::Greater;
/// use thyme_core::cv::find_labeled_contours;
///
/// let width = 3;
/// let height = 3;
/// let buffer: Vec<u32> = vec![12, 12, 0, 12, 0, 10, 0, 10, 10];
/// let (labels, contours) = find_labeled_contours(width, height, &buffer, &vec![10u32, 12u32]);
///
/// assert_eq!(contours, [[[2.0, 1.0], [1.0, 2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]);
/// ```
pub fn find_labeled_contours(
    width: u32,
    height: u32,
    pixels: &[u32],
    labels: &Vec<u32>,
) -> (Vec<u32>, Vec<Vec<[f32; 2]>>) {
    let mut contours = Vec::with_capacity(labels.len());
    let mut retained = Vec::with_capacity(labels.len());

    for label in labels {
        let contour = find_contours(width, height, pixels, label, Ordering::Equal)
            .into_iter()
            .max_by_key(|contour| contour.len());

        if let Some(contour) = contour {
            if contour.len() > 2 {
                contours.push(contour);
                retained.push(*label)
            }
        }
    }

    (retained, contours)
}

#[cfg(test)]
mod test {

    use super::*;

    fn four_regions_small() -> (u32, u32, [u32; 9]) {
        let mut buffer = [0u32; 9];

        buffer[0] = 1u32;
        buffer[2] = 1u32;
        buffer[6] = 1u32;
        buffer[8] = 1u32;

        (3, 3, buffer)
    }

    fn four_regions_big() -> (u32, u32, [u32; 25]) {
        let mut buffer = [0u32; 25];

        buffer[0] = 1u32;
        buffer[1] = 1u32;
        buffer[5] = 1u32;
        buffer[6] = 1u32;

        buffer[3] = 1u32;
        buffer[4] = 1u32;
        buffer[8] = 1u32;
        buffer[9] = 1u32;

        buffer[15] = 2u32;
        buffer[16] = 2u32;
        buffer[20] = 2u32;
        buffer[21] = 2u32;

        buffer[18] = 3u32;
        buffer[19] = 3u32;
        buffer[23] = 3u32;
        buffer[24] = 3u32;

        (5, 5, buffer)
    }

    fn three_regions() -> (u32, u32, [u32; 9]) {
        let mut buffer = [0u32; 9];

        buffer[0] = 1u32;
        buffer[2] = 1u32;
        buffer[6] = 1u32;
        buffer[7] = 1u32;
        buffer[8] = 1u32;

        (3, 3, buffer)
    }

    fn two_squares() -> (u32, u32, [u32; 100]) {
        let mut buffer = [0u32; 100];

        for i in 0..10 {
            for j in 0..10 {
                let idx = j * 10 + i;
                if i < 4 && j < 4 {
                    buffer[idx] = 1u32;
                } else if i >= 6 && j >= 6 {
                    buffer[idx] = 2u32;
                } else {
                    buffer[idx] = 0u32;
                }
            }
        }

        (10, 10, buffer)
    }

    fn two_squares_touching() -> (u32, u32, [u32; 100]) {
        let mut buffer = [0u32; 100];

        for i in 0..10 {
            for j in 0..10 {
                let idx = j * 10 + i;
                if i < 5 {
                    buffer[idx] = 1u32;
                } else if i >= 5 {
                    buffer[idx] = 2u32;
                } else {
                    buffer[idx] = 0u32;
                }
            }
        }

        (10, 10, buffer)
    }

    #[test]
    fn test_four_regions_small() {
        let (w, h, buffer) = four_regions_small();
        let contours = find_contours(w, h, &buffer, &0, Ordering::Greater);

        let n_contours = contours.len();
        assert_eq!(n_contours, 4);

        let p0 = &contours[0];
        let p1 = &contours[1];
        let p2 = &contours[2];
        let p3 = &contours[3];

        assert_eq!(*p0, vec![[0., 0.]]);
        assert_eq!(*p1, vec![[2., 0.]]);
        assert_eq!(*p2, vec![[0., 2.]]);
        assert_eq!(*p3, vec![[2., 2.]]);
    }

    #[test]
    fn test_four_regions_big() {
        let (w, h, buffer) = four_regions_big();
        let contours = find_contours(w, h, &buffer, &0, Ordering::Greater);

        let n_contours = contours.len();
        assert_eq!(n_contours, 4);

        let p0 = &contours[0];
        let p1 = &contours[1];
        let p2 = &contours[2];
        let p3 = &contours[3];

        assert_eq!(*p0, vec![[0., 0.], [0., 1.], [1., 1.], [1., 0.]]);
        assert_eq!(*p1, vec![[3., 0.], [3., 1.], [4., 1.], [4., 0.]]);
        assert_eq!(*p2, vec![[0., 3.], [0., 4.], [1., 4.], [1., 3.]]);
        assert_eq!(*p3, vec![[3., 3.], [3., 4.], [4., 4.], [4., 3.]]);
    }

    #[test]
    fn test_three_regions() {
        let (w, h, buffer) = three_regions();
        let contours = find_contours(w, h, &buffer, &0, Ordering::Greater);

        let n_contours = contours.len();
        assert_eq!(n_contours, 3);

        let p0 = &contours[0];
        let p1 = &contours[1];
        let p2 = &contours[2];

        assert_eq!(*p0, vec![[0., 0.]]);
        assert_eq!(*p1, vec![[2., 0.]]);
        assert_eq!(*p2, vec![[0., 2.], [1., 2.], [2., 2.], [1., 2.]]);
    }

    #[test]
    fn test_two_squares() {
        let (w, h, buffer) = two_squares();
        let contours = find_contours(w, h, &buffer, &0, Ordering::Greater);

        let n_contours = contours.len();
        assert_eq!(n_contours, 2);

        let p0 = &contours[0];
        let p1 = &contours[1];

        assert_eq!(
            *p0,
            vec![
                [0., 0.],
                [0., 1.],
                [0., 2.],
                [0., 3.],
                [1., 3.],
                [2., 3.],
                [3., 3.],
                [3., 2.],
                [3., 1.],
                [3., 0.],
                [2., 0.],
                [1., 0.],
            ]
        );

        assert_eq!(
            *p1,
            vec![
                [6., 6.],
                [6., 7.],
                [6., 8.],
                [6., 9.],
                [7., 9.],
                [8., 9.],
                [9., 9.],
                [9., 8.],
                [9., 7.],
                [9., 6.],
                [8., 6.],
                [7., 6.],
            ]
        );
    }

    #[test]
    fn test_two_squares_touching() {
        let (w, h, buffer) = two_squares_touching();
        let contours = find_contours(w, h, &buffer, &0, Ordering::Greater);

        assert_eq!(contours.len(), 1);

        let (_, contours) = find_labeled_contours(w, h, &buffer, &vec![1u32, 2u32]);

        let p0 = &contours[0];
        let p1 = &contours[1];

        assert_eq!(
            *p0,
            vec![
                [0., 0.],
                [0., 1.],
                [0., 2.],
                [0., 3.],
                [0., 4.],
                [0., 5.],
                [0., 6.],
                [0., 7.],
                [0., 8.],
                [0., 9.],
                [1., 9.],
                [2., 9.],
                [3., 9.],
                [4., 9.],
                [4., 8.],
                [4., 7.],
                [4., 6.],
                [4., 5.],
                [4., 4.],
                [4., 3.],
                [4., 2.],
                [4., 1.],
                [4., 0.],
                [3., 0.],
                [2., 0.],
                [1., 0.]
            ]
        );

        assert_eq!(
            *p1,
            vec![
                [5., 0.],
                [5., 1.],
                [5., 2.],
                [5., 3.],
                [5., 4.],
                [5., 5.],
                [5., 6.],
                [5., 7.],
                [5., 8.],
                [5., 9.],
                [6., 9.],
                [7., 9.],
                [8., 9.],
                [9., 9.],
                [9., 8.],
                [9., 7.],
                [9., 6.],
                [9., 5.],
                [9., 4.],
                [9., 3.],
                [9., 2.],
                [9., 1.],
                [9., 0.],
                [8., 0.],
                [7., 0.],
                [6., 0.],
            ]
        );
    }
}
