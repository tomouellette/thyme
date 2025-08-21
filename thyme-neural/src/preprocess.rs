// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use candle_core::{DType, Device, Result, Tensor};

use thyme_core::im::ThymeImage;

/// Convert a ThymeImage to a 3-channel "RGB" tensor
///
/// Any 1-channel image is simply repeated three times to generate
/// a 3-channel image. Multi-channel images that aren't 1 or 3 channels
/// will be averaged then stacked to 3 channels.
fn to_tensor_rgb(image: &ThymeImage, device: &Device) -> Result<Tensor> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let c = image.channels() as usize;

    let tensor = Tensor::from_vec(image.to_f32(), (h, w, c), device)?.permute((2, 0, 1))?;

    if c == 3 {
        return Ok(tensor);
    }

    if c == 1 {
        return Tensor::cat(&[&tensor; 3], 0);
    }

    let averaged = tensor.mean_keepdim(0).unwrap();

    Tensor::cat(&[&averaged; 3], 0)
}

/// Perform imagenet standardization on an input ThymeImage
pub fn preprocess_imagenet(image: &ThymeImage, device: &Device) -> Result<Tensor> {
    pub const IMAGENET_MEAN: [f32; 3] = [0.485f32, 0.456, 0.406];
    pub const IMAGENET_STD: [f32; 3] = [0.229f32, 0.224, 0.225];

    let tensor = if image.width() == 224 && image.height() == 224 {
        to_tensor_rgb(image, device)?
    } else {
        to_tensor_rgb(&image.resize(224, 224).unwrap(), device)?
    };

    let mean = Tensor::new(&IMAGENET_MEAN, device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&IMAGENET_STD, device)?.reshape((3, 1, 1))?;

    (tensor.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

/// Perform subcell standardization on an input ThymeImage
///
/// Note that subcell used min-max normalization for some reason
/// https://github.com/CellProfiling/SubCellPortable/blob/main/inference.py#L76C1-L81C14
pub fn preprocess_subcell(image: &ThymeImage, device: &Device) -> Result<Tensor> {
    let eps: Tensor = Tensor::new(1e-6f32, device)?;

    let tensor = if image.width() == 448 && image.height() == 448 {
        to_tensor_rgb(image, device)?
    } else {
        to_tensor_rgb(&image.resize(448, 448).unwrap(), device)?
    };

    // Not sure if there's an implementation to take min over
    // multiple dimensions in candle - need to re-check docs
    let min_val = tensor.min(0)?.min(0)?.min(0)?;
    let max_val = tensor.max(0)?.max(0)?.max(0)?;

    tensor
        .broadcast_sub(&min_val)?
        .broadcast_div(&(max_val - min_val + eps)?)
}

#[cfg(test)]
mod test {
    use super::*;

    use thyme_core::im::ThymeBuffer;

    #[test]
    fn test_to_tensor_rgb_1channel() {
        let buffer: Vec<u8> = vec![0, 1, 2, 3];
        let image = ThymeImage::U8(ThymeBuffer::new(2, 2, 1, buffer).unwrap());
        let tensor = to_tensor_rgb(&image, &Device::Cpu);

        let shape = tensor.unwrap().shape().clone().into_dims();
        assert_eq!(shape[0], 3);
        assert_eq!(shape[1], 2);
        assert_eq!(shape[2], 2);
    }

    #[test]
    fn test_to_tensor_rgb_2channel() {
        let buffer: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let image = ThymeImage::U8(ThymeBuffer::new(2, 2, 2, buffer).unwrap());
        let tensor = to_tensor_rgb(&image, &Device::Cpu);

        let shape = tensor.unwrap().shape().clone().into_dims();
        assert_eq!(shape[0], 3);
        assert_eq!(shape[1], 2);
        assert_eq!(shape[2], 2);
    }

    #[test]
    fn test_to_tensor_rgb_3channel() {
        let buffer: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let image = ThymeImage::U8(ThymeBuffer::new(2, 2, 3, buffer).unwrap());
        let tensor = to_tensor_rgb(&image, &Device::Cpu);

        let shape = tensor.unwrap().shape().clone().into_dims();
        assert_eq!(shape[0], 3);
        assert_eq!(shape[1], 2);
        assert_eq!(shape[2], 2);
    }

    #[test]
    fn test_to_tensor_rgb_nchannel() {
        let buffer: Vec<u8> = (0..20).collect();
        let image = ThymeImage::U8(ThymeBuffer::new(2, 2, 5, buffer).unwrap());
        let tensor = to_tensor_rgb(&image, &Device::Cpu);

        let shape = tensor.unwrap().shape().clone().into_dims();
        assert_eq!(shape[0], 3);
        assert_eq!(shape[1], 2);
        assert_eq!(shape[2], 2);
    }
}
