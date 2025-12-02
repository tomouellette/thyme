// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::fmt;

#[derive(Debug, Clone)]
pub enum PineappleError {
    BoundingBoxError,
    BufferSizeError,
    ChannelBoundsError,
    ConversionError,
    ImageError(&'static str),
    ImageReadError,
    ImageWriteError,
    ImageFormatError,
    ImageExtensionError,
    MaskError(&'static str),
    MaskFormatError,
    PolygonsSizeError,
    PolygonsReadError,
    PolygonsWriteError,
    BoxesSizeError,
    BoxesReadError,
    BoxesWriteError,
    NoFileError(String),
    DirError(String),
    OtherError(String),
}

impl fmt::Display for PineappleError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PineappleError::BoundingBoxError => {
                write!(
                    f,
                    "[pineapple::BoundingBoxError] The bounding box is invalid as max_x (max_y) must be greater than min_x (min_y)."
                )
            }
            PineappleError::BufferSizeError => {
                write!(
                    f,
                    "[pineapple::BufferSizeError] The buffer does not match provided size"
                )
            }
            PineappleError::ChannelBoundsError => {
                write!(
                    f,
                    "[pineapple::ChannelBoundsError] The indexed channel is out of bounds."
                )
            }
            PineappleError::ConversionError => {
                write!(
                    f,
                    "[pineapple::ConversionError] Failed to convert value to f32."
                )
            }
            PineappleError::ImageError(message) => {
                write!(f, "[pineapple::ImageError] Failed to create image. {}", message)
            }
            PineappleError::ImageReadError => {
                write!(f, "[pineapple::ImageReadError] Failed to read image.",)
            }
            PineappleError::ImageWriteError => {
                write!(f, "[pineapple::ImageWriteError] Failed to write image.",)
            }
            PineappleError::ImageFormatError => {
                write!(
                    f,
                    "[pineapple::ImageFormatError] Only 1 and 3-channel u8 and u16 images are currently supported."
                )
            }
            PineappleError::ImageExtensionError => {
                write!(
                    f,
                    "[pineapple::ImageExtensionError] Could not detect a valid image extension for input."
                )
            }
            PineappleError::MaskError(message) => {
                write!(f, "[pineapple::MaskError] Failed to create mask. {}", message)
            }
            PineappleError::MaskFormatError => {
                write!(
                    f,
                    "[pineapple::MaskFormatError] Only 1-channel u8 and u16 masks are currently supported."
                )
            }
            PineappleError::PolygonsSizeError => {
                write!(
                    f,
                    "[pineapple::PolygonsSizeError] No polygons with length > 3 were detected in input.",
                )
            }
            PineappleError::PolygonsReadError => {
                write!(f, "[pineapple::PolygonsReadError] Polygons could not be read.")
            }
            PineappleError::PolygonsWriteError => {
                write!(
                    f,
                    "[pineapple::PolygonsWriteError] Failed to successfully write polygons to output."
                )
            }
            PineappleError::BoxesSizeError => {
                write!(
                    f,
                    "[pineapple::BoxesSizeError] Bounding box must satisfy x_min < x_max and y_min < y_max.",
                )
            }
            PineappleError::BoxesReadError => {
                write!(
                    f,
                    "[pineapple::BoxesReadError] Bounding boxes could not be read."
                )
            }
            PineappleError::BoxesWriteError => {
                write!(
                    f,
                    "[pineapple::BoxesWriteError] Failed to successfully write boundng boxes to output."
                )
            }
            PineappleError::NoFileError(message) => {
                write!(
                    f,
                    "[pineapple::NoFileError] File could not be found. {}.",
                    message
                )
            }
            PineappleError::DirError(message) => {
                write!(
                    f,
                    "[pineapple::DirError] Directory could not be read. {}.",
                    message
                )
            }
            PineappleError::OtherError(message) => {
                write!(f, "[pineapple::OtherError] Error: {}.", message)
            }
        }
    }
}

impl std::error::Error for PineappleError {}
