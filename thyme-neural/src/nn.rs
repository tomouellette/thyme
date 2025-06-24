// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use candle_core::Device;
use candle_core::{Module, Result, Tensor};

use thyme_core::im::ThymeImage;

use crate::models::DinoVisionTransformer;
use crate::models::StandardVisionTransformer;

use crate::load::{
    load_dinobloom_vit_base, load_dinov2_vit_base, load_dinov2_vit_small, load_scdino_vit_small,
    load_subcell_vit_base,
};

use crate::preprocess::{preprocess_imagenet, preprocess_subcell};

pub enum Models {
    DinoVitSmall(DinoVisionTransformer),
    DinoVitBase(DinoVisionTransformer),
    DinobloomVitBase(DinoVisionTransformer),
    ScdinoVitSmall(StandardVisionTransformer),
    SubcellVitSmall(StandardVisionTransformer),
}

impl Models {
    pub fn load(model_name: &str, device: &Device, verbose: bool) -> Self {
        match model_name {
            "dino_vit_small" => {
                let model = load_dinov2_vit_small(device, verbose).unwrap();
                Models::DinoVitSmall(model)
            }
            "dino_vit_base" => {
                let model = load_dinov2_vit_base(device, verbose).unwrap();
                Models::DinoVitBase(model)
            }
            "dinobloom_vit_base" => {
                let model = load_dinobloom_vit_base(device, verbose).unwrap();
                Models::DinobloomVitBase(model)
            }
            "scdino_vit_small" => {
                let model = load_scdino_vit_small(device, verbose).unwrap();
                Models::ScdinoVitSmall(model)
            }
            "subcell_vit_base" => {
                let model = load_subcell_vit_base(device, verbose).unwrap();
                Models::SubcellVitSmall(model)
            }
            _ => {
                eprintln!("[thyme::nn::models] Model name not found.");
                std::process::exit(1);
            }
        }
    }

    pub fn preprocess(&self, image: &ThymeImage, device: &Device) -> Result<Tensor> {
        match self {
            Models::DinoVitSmall(_) => preprocess_imagenet(image, device),
            Models::DinoVitBase(_) => preprocess_imagenet(image, device),
            Models::DinobloomVitBase(_) => preprocess_imagenet(image, device),
            Models::ScdinoVitSmall(_) => preprocess_imagenet(image, device),
            Models::SubcellVitSmall(_) => preprocess_subcell(image, device),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input = input.unsqueeze(0).unwrap();
        match self {
            Models::DinoVitSmall(model) => model.forward(&input),
            Models::DinoVitBase(model) => model.forward(&input),
            Models::DinobloomVitBase(model) => model.forward(&input),
            Models::ScdinoVitSmall(model) => model.forward(&input),
            Models::SubcellVitSmall(model) => model.forward(&input),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use thyme_core::im::ThymeImage;

    fn load_rgb() -> ThymeImage {
        ThymeImage::open("../data/tests/test_rgb.png").unwrap()
    }

    fn load_grayscale() -> ThymeImage {
        ThymeImage::open("../data/tests/test_grayscale.png").unwrap()
    }

    fn test_model(name: &str, color: &str, n_embed: usize) {
        let image = if color == "rgb" {
            load_rgb()
        } else {
            load_grayscale()
        };

        let model = Models::load(name, &Device::Cpu, true);
        let image = model.preprocess(&image, &Device::Cpu).unwrap();
        let logits = model.forward(&image).unwrap();

        let (n_row, n_columns) = logits.shape().dims2().unwrap();

        assert_eq!(n_row, 1);
        assert_eq!(n_columns, n_embed);
    }

    #[test]
    fn test_dinov2_small_rgb() {
        test_model("dino_vit_small", "rgb", 384);
    }

    #[test]
    fn test_dinov2_small_grayscale() {
        test_model("dino_vit_small", "grayscale", 384);
    }

    #[test]
    fn test_dinov2_base_rgb() {
        test_model("dino_vit_base", "rgb", 768);
    }

    #[test]
    fn test_dinov2_base_grayscale() {
        test_model("dino_vit_base", "grayscale", 768);
    }

    #[test]
    fn test_dinobloom_rgb() {
        test_model("dinobloom_vit_base", "rgb", 768);
    }

    #[test]
    fn test_dinobloom_grayscale() {
        test_model("dinobloom_vit_base", "grayscale", 768);
    }

    #[test]
    fn test_subcell_rgb() {
        test_model("subcell_vit_base", "rgb", 768);
    }

    #[test]
    fn test_subcell_grayscale() {
        test_model("subcell_vit_base", "grayscale", 768);
    }

    #[test]
    fn test_scdino_rgb() {
        test_model("scdino_vit_small", "rgb", 384);
    }

    #[test]
    fn test_scdino_grayscale() {
        test_model("scdino_vit_small", "grayscale", 384);
    }
}
