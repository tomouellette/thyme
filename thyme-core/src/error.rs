// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use std::fmt;

#[derive(Debug, Clone)]
pub enum ThymeError {
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

impl fmt::Display for ThymeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ThymeError::BoundingBoxError => {
                write!(
                    f,
                    "[thyme::BoundingBoxError] The bounding box is invalid as max_x (max_y) must be greater than min_x (min_y)."
                )
            }
            ThymeError::BufferSizeError => {
                write!(
                    f,
                    "[thyme::BufferSizeError] The buffer does not match provided size"
                )
            }
            ThymeError::ChannelBoundsError => {
                write!(
                    f,
                    "[thyme::ChannelBoundsError] The indexed channel is out of bounds."
                )
            }
            ThymeError::ConversionError => {
                write!(
                    f,
                    "[thyme::ConversionError] Failed to convert value to f32."
                )
            }
            ThymeError::ImageError(message) => {
                write!(f, "[thyme::ImageError] Failed to create image. {}", message)
            }
            ThymeError::ImageReadError => {
                write!(f, "[thyme::ImageReadError] Failed to read image.",)
            }
            ThymeError::ImageWriteError => {
                write!(f, "[thyme::ImageWriteError] Failed to write image.",)
            }
            ThymeError::ImageFormatError => {
                write!(
                    f,
                    "[thyme::ImageFormatError] Only 1 and 3-channel u8 and u16 images are currently supported."
                )
            }
            ThymeError::ImageExtensionError => {
                write!(
                    f,
                    "[thyme::ImageExtensionError] Could not detect a valid image extension for input."
                )
            }
            ThymeError::MaskError(message) => {
                write!(f, "[thyme::MaskError] Failed to create mask. {}", message)
            }
            ThymeError::MaskFormatError => {
                write!(
                    f,
                    "[thyme::MaskFormatError] Only 1-channel u8 and u16 masks are currently supported."
                )
            }
            ThymeError::PolygonsSizeError => {
                write!(
                    f,
                    "[thyme::PolygonsSizeError] No polygons with length > 3 were detected in input.",
                )
            }
            ThymeError::PolygonsReadError => {
                write!(f, "[thyme::PolygonsReadError] Polygons could not be read.")
            }
            ThymeError::PolygonsWriteError => {
                write!(
                    f,
                    "[thyme::PolygonsWriteError] Failed to successfully write polygons to output."
                )
            }
            ThymeError::BoxesSizeError => {
                write!(
                    f,
                    "[thyme::BoxesSizeError] Bounding box must satisfy x_min < x_max and y_min < y_max.",
                )
            }
            ThymeError::BoxesReadError => {
                write!(
                    f,
                    "[thyme::BoxesReadError] Bounding boxes could not be read."
                )
            }
            ThymeError::BoxesWriteError => {
                write!(
                    f,
                    "[thyme::BoxesWriteError] Failed to successfully write boundng boxes to output."
                )
            }
            ThymeError::NoFileError(message) => {
                write!(
                    f,
                    "[thyme::NoFileError] File could not be found. {}.",
                    message
                )
            }
            ThymeError::DirError(message) => {
                write!(
                    f,
                    "[thyme::DirError] Directory could not be read. {}.",
                    message
                )
            }
            ThymeError::OtherError(message) => {
                write!(f, "[thyme::OtherError] Error: {}.", message)
            }
        }
    }
}

impl std::error::Error for ThymeError {}
