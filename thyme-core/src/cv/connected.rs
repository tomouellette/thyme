// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::cmp::Ordering;

/// A union-find structure for finding and merging connected components
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Initialize a new union-find object with `n` elements in `n` sets
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![1; n],
        }
    }

    /// Find the root of the set containing `x`
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            // Path compression
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Merge sets containing `x` and `y`
    pub fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            match self.rank[root_x].cmp(&self.rank[root_y]) {
                Ordering::Greater => self.parent[root_y] = root_x,
                Ordering::Less => self.parent[root_x] = root_y,
                Ordering::Equal => {
                    self.parent[root_y] = root_x;
                    self.rank[root_x] += 1;
                }
            }
        }
    }

    /// Check if `x` and `y` belong to the same set
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

/// Two-pass 8-connected component labeling on mask buffers
///
/// # Arguments
///
/// * `width` - Width of mask
/// * `height` - Height of mask
/// * `buffer` - A row-major mask buffer
///
/// # Examples
///
/// ```
/// use thyme_core::im::ThymeMask;
/// use thyme_core::cv::connected_components;
///
/// let width = 3;
/// let height = 3;
///
/// let buffer_one: Vec<u32> = vec![10, 10, 0, 10, 0, 20, 0, 20, 20];
/// let labels_one = connected_components(width, height, &buffer_one);
/// assert_eq!(labels_one, [1, 1, 0, 1, 0, 1, 0, 1, 1]);
///
/// let buffer_two: Vec<u32> = vec![10, 10, 10, 0, 0, 0, 20, 20, 20];
/// let labels_two = connected_components(width, height, &buffer_two);
/// assert_eq!(labels_two, [1, 1, 1, 0, 0, 0, 2, 2, 2]);
/// ```
pub fn connected_components(width: u32, height: u32, buffer: &[u32]) -> Vec<u32> {
    let width = width as usize;
    let height = height as usize;
    let size = width * height;

    let mut labels = vec![0u32; size];
    let mut next_label = 1;
    let mut uf = UnionFind::new(size);

    // Assign preliminary labels (1st pass)
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if buffer[idx] == 0 {
                // Ignore background pixels
                continue;
            }

            let mut neighbors = vec![];

            // Check left neighbor
            if x > 0 && buffer[idx - 1] > 0 {
                neighbors.push(labels[idx - 1]);
            }

            // Check top neighbor
            if y > 0 && buffer[idx - width] > 0 {
                neighbors.push(labels[idx - width]);
            }

            // Check top-left neighbor (diagonal)
            if x > 0 && y > 0 && buffer[idx - width - 1] > 0 {
                neighbors.push(labels[idx - width - 1]);
            }

            // Check top-right neighbor (diagonal)
            if x < width - 1 && y > 0 && buffer[idx - width + 1] > 0 {
                neighbors.push(labels[idx - width + 1]);
            }

            if neighbors.is_empty() {
                labels[idx] = next_label;
                next_label += 1;
            } else {
                let min_label = *neighbors.iter().min().unwrap();
                labels[idx] = min_label;

                // Take union of neighbors
                for &label in &neighbors {
                    uf.union(min_label as usize, label as usize);
                }
            }
        }
    }

    // Resolve labels using union-find (2nd pass)
    for label in labels.iter_mut().take(size) {
        if label != &0 {
            *label = uf.find(*label as usize) as u32;
        }
    }

    labels
}

#[cfg(test)]
mod test {

    use super::*;

    fn four_regions() -> (u32, u32, [u32; 9]) {
        let mut buffer = [0u32; 9];

        buffer[0] = 1u32;
        buffer[2] = 2u32;
        buffer[6] = 3u32;
        buffer[8] = 3u32;

        (3, 3, buffer)
    }

    fn three_regions() -> (u32, u32, [u32; 9]) {
        let mut buffer = [0u32; 9];

        buffer[0] = 1u32;
        buffer[2] = 2u32;
        buffer[6] = 3u32;
        buffer[7] = 3u32;
        buffer[8] = 3u32;

        (3, 3, buffer)
    }

    fn touching_regions() -> (u32, u32, [u32; 9]) {
        let mut buffer = [0u32; 9];

        buffer[0] = 1u32;
        buffer[2] = 2u32;
        buffer[4] = 4u32;
        buffer[6] = 3u32;
        buffer[7] = 3u32;
        buffer[8] = 3u32;

        (3, 3, buffer)
    }

    #[test]
    fn test_four_regions() {
        let (w, h, buffer) = four_regions();

        let mut labels = connected_components(w, h, &buffer);
        labels.sort();
        labels.dedup();

        assert_eq!(labels, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_three_regions() {
        let (w, h, buffer) = three_regions();

        let mut labels = connected_components(w, h, &buffer);
        labels.sort();
        labels.dedup();

        assert_eq!(labels, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_middle_regions() {
        let (w, h, buffer) = touching_regions();

        let mut labels = connected_components(w, h, &buffer);
        labels.sort();
        labels.dedup();

        assert_eq!(labels, vec![0, 1]);
    }
}
