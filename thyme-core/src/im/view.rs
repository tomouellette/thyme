// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::ops::Deref;

use num::{FromPrimitive, ToPrimitive};

use crate::im::ThymeBuffer;
use crate::impl_enum_dispatch;
use crate::mp::{intensity, moments, texture, zernike};

/// A wrapper around valid view types
pub enum ThymeView<'a> {
    U8(ThymeViewBuffer<'a, u8, Vec<u8>>),
    U16(ThymeViewBuffer<'a, u16, Vec<u16>>),
    U32(ThymeViewBuffer<'a, u32, Vec<u32>>),
    U64(ThymeViewBuffer<'a, u64, Vec<u64>>),
    I32(ThymeViewBuffer<'a, i32, Vec<i32>>),
    I64(ThymeViewBuffer<'a, i64, Vec<i64>>),
    F32(ThymeViewBuffer<'a, f32, Vec<f32>>),
    F64(ThymeViewBuffer<'a, f64, Vec<f64>>),
}

// >>> PROPERTY METHODS

impl_enum_dispatch!(ThymeView<'a>, U8, U16, U32, U64, I32, I64, F32, F64; width(&'a self) -> usize);
impl_enum_dispatch!(ThymeView<'a>, U8, U16, U32, U64, I32, I64, F32, F64; height(&'a self) -> usize);
impl_enum_dispatch!(ThymeView<'a>, U8, U16, U32, U64, I32, I64, F32, F64; channels(&'a self) -> usize);

// <<< PROPERTY METHODS

// >>> MEASURE METHODS

impl_enum_dispatch!(ThymeView<'a>, U8, U16, U32, U64, I32, I64, F32, F64; intensity(&'a self) -> [f32; 7]);
impl_enum_dispatch!(ThymeView<'a>, U8, U16, U32, U64, I32, I64, F32, F64; moments(&'a self) -> [f32; 24]);
impl_enum_dispatch!(ThymeView<'a>, U8, U16, U32, U64, I32, I64, F32, F64; texture(&'a self) -> [f32; 13]);
impl_enum_dispatch!(ThymeView<'a>, U8, U16, U32, U64, I32, I64, F32, F64; zernike(&'a self) -> [f32; 30]);
impl_enum_dispatch!(ThymeView<'a>, U8, U16, U32, U64, I32, I64, F32, F64; descriptors(&'a self) -> Vec<f32>);

// <<< MEASURE METHODS

/// A row-major buffer that defines an image view/crop/subregion
///
/// The cropped object represents a zero-copy reference to a larger
/// ThymeBuffer. To enable zero-copy, the full image has to share
/// the same lifetime as the cropped object. This should generally
/// always be the case since we perform operations on cropped objects
/// iteratively across all segmented objects from the same image.
///
/// # Examples
///
/// ```
/// use thyme_core::im::{ThymeBuffer, ThymeViewBuffer};
///
/// let width = 3;
/// let height = 3;
/// let channels = 1;
/// let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
///
/// let buffer = ThymeBuffer::<u8, Vec<u8>>::new(width, height, channels, data).unwrap();
/// let crop = ThymeViewBuffer::new(1, 1, 2, 2, &buffer);
///
/// for subpixel in crop.iter() {
///     let _ = subpixel;
/// }
/// ```
#[derive(Clone)]
pub struct ThymeViewBuffer<'a, T, Container> {
    buffer: &'a ThymeBuffer<T, Container>,
    width: usize,    // Full image width
    channels: usize, // Full image channels
    x: usize,        // Minimum x-value of crop
    y: usize,        // Minimum y-value of crop
    w: usize,        // Width of crop
    h: usize,        // Height of crop
}

impl<'a, T, Container> ThymeViewBuffer<'a, T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    /// Initialize a copy-free object specifying a cropped/subregion of an image
    ///
    /// # Examples
    ///
    /// ```
    /// use thyme_core::im::{ThymeBuffer, ThymeViewBuffer};
    ///
    /// let width = 3;
    /// let height = 3;
    /// let channels = 1;
    /// let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
    ///
    /// let buffer = ThymeBuffer::<u8, Vec<u8>>::new(width, height, channels, data).unwrap();
    /// let crop = ThymeViewBuffer::new(1, 1, 2, 2, &buffer);
    /// ```
    pub fn new(x: u32, y: u32, w: u32, h: u32, buffer: &'a ThymeBuffer<T, Container>) -> Self {
        let x = std::cmp::min(x, buffer.width());
        let y = std::cmp::min(y, buffer.height());
        let h = std::cmp::min(h, buffer.height() - y);
        let w = std::cmp::min(w, buffer.width() - x);

        ThymeViewBuffer {
            buffer,
            width: buffer.width() as usize,
            channels: buffer.channels() as usize,
            x: x as usize,
            y: y as usize,
            w: w as usize,
            h: h as usize,
        }
    }
}

// >>> PROPERTY METHODS

impl<T, Container> ThymeViewBuffer<'_, T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    /// Length of cropped buffer
    pub fn len(&self) -> usize {
        self.w * self.h * self.channels
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Width of object
    #[allow(clippy::all)]
    pub fn width(&self) -> usize {
        self.w
    }

    /// Height of object
    #[allow(clippy::all)]
    pub fn height(&self) -> usize {
        self.h
    }

    /// Number of channels
    pub fn channels(&self) -> usize {
        self.channels
    }
}

// <<< PROPERTY METHODS

// >>> MEASURE METHODS

impl<'a, T, Container> ThymeViewBuffer<'a, T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    /// Compute the intensity descriptors for the object
    #[allow(clippy::identity_op, clippy::erasing_op)]
    pub fn intensity(&'a self) -> [f32; 7] {
        let results = intensity::objects(self);

        let c = self.channels();
        let rc = 1f32 / c as f32;
        let len = results.len();

        // We average over channel values to avoid variable
        //sized outputs in variable channel experiments
        let mut average: [f32; 7] = [0f32; 7];

        average[5] = results[len - 2];
        average[6] = results[len - 1];

        for i in 0..c {
            average[0] = results[i + 0 * c] * rc;
            average[1] = results[i + 1 * c] * rc;
            average[2] = results[i + 2 * c] * rc;
            average[3] = results[i + 3 * c] * rc;
            average[4] = results[i + 3 * c] * rc;
        }

        average
    }

    /// Compute the image moments for the object
    pub fn moments(&'a self) -> [f32; 24] {
        moments::objects(self)
    }

    /// Compute the texture descriptors for the object
    pub fn texture(&'a self) -> [f32; 13] {
        texture::objects(self)
    }

    /// Compute the zernike moments for the object
    pub fn zernike(&'a self) -> [f32; 30] {
        zernike::objects(self)
    }

    /// Compute all view descriptors
    pub fn descriptors(&'a self) -> Vec<f32> {
        self.intensity()
            .into_iter()
            .chain(self.moments())
            .chain(self.texture())
            .chain(self.zernike())
            .collect()
    }
}

// <<< MEASURE METHODS

// >>> ITERATOR METHODS

impl<'a, T, Container> ThymeViewBuffer<'a, T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    /// Return an iterator over pixels containing all channels
    pub fn iter(&'a self) -> SubpixelIterator<'a, T, Container> {
        SubpixelIterator {
            buffer: self.buffer,
            width: self.width,
            channels: self.channels,
            x: self.x,
            y: self.y,
            w: self.w,
            h: self.h,
            i: self.y,
            j: self.w * self.channels,
        }
    }

    /// Return an iterator over pixels containing all channels
    pub fn iter_pixels(&'a self) -> PixelIterator<'a, T, Container> {
        PixelIterator {
            buffer: self.buffer,
            width: self.width,
            channels: self.channels,
            x: self.x,
            y: self.y,
            w: self.w,
            h: self.h,
            i: 0,
            j: 0,
        }
    }
}

// <<< ITERATOR METHODS

/// An iterator over subpixels
pub struct SubpixelIterator<'a, T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    buffer: &'a ThymeBuffer<T, Container>,
    width: usize,
    channels: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    i: usize,
    j: usize,
}

impl<'a, T, Container> Iterator for SubpixelIterator<'a, T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.y + self.h {
            return None;
        }

        let j_mod = self.j % (self.w * self.channels);

        let idx = self.i * (self.width * self.channels) + (self.x * self.channels) + j_mod;

        if j_mod == (self.w * self.channels) - 1 {
            self.i += 1;
        }

        self.j += 1;

        Some(&self.buffer.as_raw()[idx])
    }
}

/// An iterator over pixels containing all channels
pub struct PixelIterator<'a, T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    buffer: &'a ThymeBuffer<T, Container>,
    width: usize,
    channels: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    i: usize,
    j: usize,
}

impl<'a, T, Container> Iterator for PixelIterator<'a, T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.j >= self.h {
            return None;
        }

        let idx = ((self.y + self.j) * self.width + (self.x + self.i)) * self.channels;

        self.i += 1;

        if self.i >= self.w {
            self.i = 0;
            self.j += 1;
        }

        Some(&self.buffer.as_raw()[idx..idx + self.channels])
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_crop_in_bounds() {
        let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let buffer = ThymeBuffer::<u8, Vec<u8>>::new(3, 3, 1, data).unwrap();

        let crop = ThymeViewBuffer::new(1, 1, 2, 2, &buffer);
        let mut step = crop.iter();
        assert_eq!(step.next().unwrap(), &4);
        assert_eq!(step.next().unwrap(), &5);
        assert_eq!(step.next().unwrap(), &7);
        assert_eq!(step.next().unwrap(), &8);
    }

    #[test]
    fn test_crop_in_bounds_multichannel() {
        let data = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8];
        let buffer = ThymeBuffer::<u8, Vec<u8>>::new(3, 3, 2, data).unwrap();

        let crop = ThymeViewBuffer::new(1, 1, 2, 2, &buffer);
        let mut step = crop.iter();
        assert_eq!(step.next().unwrap(), &4);
        assert_eq!(step.next().unwrap(), &4);
        assert_eq!(step.next().unwrap(), &5);
        assert_eq!(step.next().unwrap(), &5);
        assert_eq!(step.next().unwrap(), &7);
        assert_eq!(step.next().unwrap(), &7);
        assert_eq!(step.next().unwrap(), &8);
        assert_eq!(step.next().unwrap(), &8);
    }

    #[test]
    fn test_crop_out_bounds() {
        let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let buffer = ThymeBuffer::<u8, Vec<u8>>::new(3, 3, 1, data).unwrap();

        let crop = ThymeViewBuffer::new(2, 2, 4, 4, &buffer);
        let mut step = crop.iter();
        assert_eq!(step.next().unwrap(), &8);
        assert_eq!(step.next(), None);
    }

    #[test]
    fn test_crop_iter() {
        let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let buffer = ThymeBuffer::<u8, Vec<u8>>::new(3, 3, 1, data).unwrap();

        let crop = ThymeViewBuffer::new(0, 0, 3, 3, &buffer);
        for (i, c) in crop.iter().enumerate() {
            let h = i / 3;
            let w = i % 3;
            assert_eq!(*c, (3 * h + w) as u8);
        }
    }

    #[test]
    fn test_crop_iter_pixels() {
        let data = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8];
        let buffer = ThymeBuffer::<u8, Vec<u8>>::new(3, 3, 2, data).unwrap();

        let crop = ThymeViewBuffer::new(0, 0, 3, 3, &buffer);

        for (i, pixel) in crop.iter_pixels().enumerate() {
            assert_eq!(pixel, &[i as u8, i as u8]);
        }
    }

    #[test]
    fn test_iter_consistency() {
        let size_32 = ThymeBuffer::<u8, Vec<u8>>::new(3, 2, 1, vec![0, 0, 0, 1, 1, 1]).unwrap();
        let size_23 = ThymeBuffer::<u8, Vec<u8>>::new(2, 3, 1, vec![0, 0, 1, 1, 2, 2]).unwrap();

        let size_32_crop = ThymeViewBuffer::new(0, 0, 3, 2, &size_32);
        let size_23_crop = ThymeViewBuffer::new(0, 0, 2, 3, &size_23);

        assert_eq!(
            size_32_crop.iter().count(),
            size_32_crop.iter_pixels().count()
        );

        assert_eq!(
            size_23_crop.iter().count(),
            size_23_crop.iter_pixels().count()
        );
    }
}
