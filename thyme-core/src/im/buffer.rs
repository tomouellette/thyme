// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::iter::Iterator;
use std::marker::PhantomData;
use std::ops::Deref;
use std::slice::ChunksExact;

use num::{FromPrimitive, ToPrimitive};

use crate::error::ThymeError;
use crate::im::{MaskingStyle, ThymeMaskView, ThymeViewBuffer};

/// A row-major container storing an image buffer or grid of pixels.
///
/// The struct is generic over the data type `T` and over the container that
/// holds raw pixel/subpixel data as a slice (`[T]`) or vector (`Vec<T>`).
/// The container holding the pixel data must implement `Deref<Target = [T]>`
/// to allow for slice-like access to the data. The length of the container
/// must also be equal to the product of `w` * `h` * `c`.
///
/// # Examples
///
/// ```
/// use thyme_core::im::ThymeBuffer;
///
/// let width = 10;
/// let height = 10;
/// let channels = 3; // RGB
/// let data = vec![0u8; (width * height * channels) as usize];
///
/// let buffer = ThymeBuffer::new(width, height, channels, data);
///
/// assert_eq!(buffer.unwrap().len(), (width * height * channels) as usize);
/// ```
///
/// ```
/// use thyme_core::im::ThymeBuffer;
///
/// let width = 10;
/// let height = 10;
/// let channels = 3; // RGB
/// let data = vec![0u8; (width * height * 3 * channels) as usize];
///
/// let buffer = ThymeBuffer::new(width, height, channels, data);
///
/// assert!(buffer.is_err()); // Buffer size does not match dimensions
/// ```
#[derive(Debug, Clone)]
pub struct ThymeBuffer<T, Container> {
    w: u32,                   // Width
    h: u32,                   // Height
    c: u32,                   // Channels
    pub buffer: Container,    // Slice
    _phantom: PhantomData<T>, // Pixel
}

impl<T, Container> ThymeBuffer<T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    /// Initializes a buffer from a generic data container
    ///
    /// # Arguments
    ///
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `channels` - Number of image channels (e.g. 1 for grayscale)
    /// * `buffer` - A generic container (e.g. `Vec` or slice)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use thyme_core::im::ThymeBuffer;
    /// let buffer = [0, 1, 2, 3, 4];
    /// let buffer = ThymeBuffer::new(2, 2, 1, buffer.as_slice());
    /// ```
    pub fn new(
        width: u32,
        height: u32,
        channels: u32,
        buffer: Container,
    ) -> Result<ThymeBuffer<T, Container>, ThymeError> {
        if width * height * channels == buffer.len() as u32 {
            Ok(ThymeBuffer {
                w: width,
                h: height,
                c: channels,
                buffer,
                _phantom: PhantomData,
            })
        } else {
            Err(ThymeError::BufferSizeError)
        }
    }
}

// >>> PROPERTY METHODS

impl<T, Container> ThymeBuffer<T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    /// Width of the image
    pub fn width(&self) -> u32 {
        self.w
    }

    /// Height of the image
    pub fn height(&self) -> u32 {
        self.h
    }

    /// Number of channels in the image
    pub fn channels(&self) -> u32 {
        self.c
    }

    /// Shape/dimensions of the image
    pub fn shape(&self) -> (u32, u32, u32) {
        (self.h, self.w, self.c)
    }

    /// Length of the raw image
    pub fn len(&self) -> usize {
        (self.w * self.h * self.c) as usize
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// <<< PROPERTY METHODS

// >>> CONVERSION METHODS

impl<T, Container> ThymeBuffer<T, Container>
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    /// Returns the raw image
    pub fn into_raw(self) -> Container {
        self.buffer
    }

    /// Returns a reference to the raw image
    pub fn as_raw(&self) -> &Container {
        &self.buffer
    }

    /// Cast subpixels to u8 and return the buffer
    pub fn to_u8(&self) -> Vec<u8> {
        self.buffer
            .iter()
            .map(|x| x.to_u8().unwrap_or(0u8))
            .collect()
    }

    /// Cast subpixels to u16 and return the buffer
    pub fn to_u16(&self) -> Vec<u16> {
        self.buffer
            .iter()
            .map(|x| x.to_u16().unwrap_or(0u16))
            .collect()
    }

    /// Cast subpixels to u16 and return the buffer
    pub fn to_u32(&self) -> Vec<u32> {
        self.buffer
            .iter()
            .map(|x| x.to_u32().unwrap_or(0u32))
            .collect()
    }

    /// Cast subpixels to f32 and return the buffer
    pub fn to_f32(&self) -> Vec<f32> {
        self.buffer
            .iter()
            .map(|x| x.to_f32().unwrap_or(0f32))
            .collect()
    }

    /// Cast subpixels to f64 and return the buffer
    pub fn to_f64(&self) -> Vec<f64> {
        self.buffer
            .iter()
            .map(|x| x.to_f64().unwrap_or(0f64))
            .collect()
    }

    // An iterator over the raw buffer
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }

    // An iterator over a raw channel buffer
    pub fn iter_channel(&self, channel: u32) -> Result<impl Iterator<Item = &T>, ThymeError>
    where
        Container: Deref<Target = [T]>,
    {
        if channel >= self.channels() {
            return Err(ThymeError::ChannelBoundsError);
        }

        Ok(self
            .iter()
            .skip(channel as usize)
            .step_by(self.channels() as usize))
    }

    // An iterator over pixel-level chunks of the raw buffer
    pub fn iter_pixels(&self) -> ChunksExact<T> {
        self.buffer.chunks_exact(self.channels() as usize)
    }
}

// <<< CONVERSION METHODS

// >>> TRANSFORM METHODS

impl<T, Container> ThymeBuffer<T, Container>
where
    Container: Deref<Target = [T]> + FromIterator<T>,
    T: Clone + ToPrimitive + FromPrimitive,
{
    /// Generate a zero-copy crop of an image subregion
    ///
    /// # Arguments
    ///
    /// * `x` - Minimum x-coordinate (left)
    /// * `y` - Minimum y-coordinate (bottom)
    /// * `w` - Width of crop
    /// * `h` - Height of crop
    pub fn crop_view(&self, x: u32, y: u32, w: u32, h: u32) -> ThymeViewBuffer<T, Container> {
        ThymeViewBuffer::new(x, y, w, h, self)
    }

    /// Create a new buffer with copied cropped contents
    ///
    /// # Arguments
    ///
    /// * `x` - Minimum x-coordinate (left)
    /// * `y` - Minimum y-coordinate (bottom)
    /// * `w` - Width of crop
    /// * `h` - Height of crop
    pub fn crop(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<ThymeBuffer<T, Container>, ThymeError> {
        if x + w > self.w || y + h > self.h {
            return Err(ThymeError::ImageError("Cropping coordinates out of bounds"));
        }

        let c = self.c as usize;
        let orig_w = self.w as usize;
        let orig_buffer = self.buffer.as_ref();

        let mut new_buffer = Vec::with_capacity((w * h * self.c) as usize);

        for row in y..y + h {
            let start = ((row as usize) * orig_w + (x as usize)) * c;
            let end = start + (w as usize) * c;
            new_buffer.extend_from_slice(&orig_buffer[start..end]);
        }

        Ok(ThymeBuffer {
            w,
            h,
            c: self.c,
            buffer: Container::from_iter(new_buffer),
            _phantom: PhantomData,
        })
    }

    /// Crops the buffer while applying a mask to either foreground or background pixels
    ///
    /// # Arguments
    ///
    /// * `x` - Minimum x-coordinate (left)
    /// * `y` - Minimum y-coordinate (bottom)
    /// * `w` - Width of crop
    /// * `h` - Height of crop
    /// * `mask` - A cropped mask view
    /// * `mask_style` - Foreground or background masking style
    pub fn crop_masked(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        mask: &ThymeMaskView,
        mask_style: MaskingStyle,
    ) -> Result<ThymeBuffer<T, Container>, ThymeError> {
        let crop = self.crop_view(x, y, w, h);
        let mut masked = Vec::with_capacity(crop.len());

        match mask_style {
            MaskingStyle::Foreground => {
                for (m, b) in mask.iter().zip(crop.iter_pixels()) {
                    if m == &0u32 {
                        for _ in 0..self.c {
                            masked.push(T::from_u32(0u32).unwrap());
                        }
                    } else {
                        masked.extend_from_slice(b);
                    }
                }
            }
            MaskingStyle::Background => {
                for (m, b) in mask.iter().zip(crop.iter_pixels()) {
                    if m != &0u32 {
                        for _ in 0..self.c {
                            masked.push(T::from_u32(0u32).unwrap());
                        }
                    } else {
                        masked.extend_from_slice(b);
                    }
                }
            }
        }

        Ok(ThymeBuffer {
            w,
            h,
            c: self.c,
            buffer: Container::from_iter(masked),
            _phantom: PhantomData,
        })
    }
}

// <<< TRANSFORM METHODS

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_buffer_new_success() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice());
        assert!(buffer.is_ok());
    }

    #[test]
    fn test_buffer_new_error() {
        let buffer = ThymeBuffer::new(2, 3, 2, [1, 2, 3, 4, 5, 6].as_slice());
        assert!(buffer.is_err());
    }

    #[test]
    fn test_buffer_width() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice());
        assert_eq!(buffer.unwrap().width(), 1);
    }

    #[test]
    fn test_buffer_height() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice());
        assert_eq!(buffer.unwrap().height(), 3);
    }

    #[test]
    fn test_buffer_channels() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice());
        assert_eq!(buffer.unwrap().channels(), 2);
    }

    #[test]
    fn test_buffer_shape() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice());
        assert_eq!(buffer.unwrap().shape(), (3, 1, 2));
    }

    #[test]
    fn test_buffer_len() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice());
        assert_eq!(buffer.unwrap().len(), 6);
    }

    #[test]
    fn test_buffer_into_raw() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice());
        assert_eq!(buffer.unwrap().into_raw(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_buffer_as_raw() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice());
        assert_eq!(buffer.unwrap().as_raw(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_buffer_to_u8() {
        let buffer = ThymeBuffer::new(1, 2, 2, [2.5, 3.9, 4.8, 2.2].as_slice()).unwrap();

        let u8_vec = buffer.to_u8();
        assert_eq!(u8_vec, [2, 3, 4, 2]);
    }

    #[test]
    fn test_buffer_to_u16() {
        let buffer = ThymeBuffer::new(1, 2, 2, [2.5, 3.9, 4.8, 2.2].as_slice()).unwrap();

        let u16_vec = buffer.to_u16();
        assert_eq!(u16_vec, [2, 3, 4, 2]);
    }

    #[test]
    fn test_iter() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice()).unwrap();

        for (a, b) in buffer.iter().zip([1, 2, 3, 4, 5, 6]) {
            assert_eq!(a, &b);
        }
    }

    #[test]
    fn test_iter_channel() {
        let buffer = ThymeBuffer::new(1, 3, 2, [1, 2, 3, 4, 5, 6].as_slice()).unwrap();

        for (a, b) in buffer.iter_channel(0).unwrap().zip([1, 3, 5]) {
            assert_eq!(a, &b)
        }

        for (a, b) in buffer.iter_channel(1).unwrap().zip([2, 4, 6]) {
            assert_eq!(a, &b)
        }

        let buffer = ThymeBuffer::new(2, 1, 3, [1, 2, 3, 4, 5, 6].as_slice()).unwrap();

        for (a, b) in buffer.iter_channel(0).unwrap().zip([1, 4]) {
            assert_eq!(a, &b)
        }

        for (a, b) in buffer.iter_channel(1).unwrap().zip([2, 5]) {
            assert_eq!(a, &b)
        }

        for (a, b) in buffer.iter_channel(2).unwrap().zip([3, 6]) {
            assert_eq!(a, &b)
        }
    }

    #[test]
    fn test_iter_pixels() {
        let buffer = ThymeBuffer::new(1, 4, 2, [1, 2, 3, 4, 5, 6, 7, 8].as_slice()).unwrap();

        for (a, b) in buffer.iter_pixels().zip([[1, 2], [3, 4], [5, 6], [7, 8]]) {
            assert_eq!(a[0], b[0]);
            assert_eq!(a[1], b[1]);
        }
    }
}
