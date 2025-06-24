// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;

use thyme_data::data::Weights;

use crate::models::DinoVisionTransformer;
use crate::models::{StandardVisionTransformer, StandardVisionTransformerConfig};

pub fn load_dinov2_vit_small(device: &Device, verbose: bool) -> Result<DinoVisionTransformer> {
    let weights = Weights::DinoVitSmall;
    let path = weights.path();
    weights.download(verbose);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    let model = DinoVisionTransformer::new(vb, 12, 384, 6, 14, 518).unwrap();

    Ok(model)
}

pub fn load_dinov2_vit_base(device: &Device, verbose: bool) -> Result<DinoVisionTransformer> {
    let weights = Weights::DinoVitBase;
    let path = weights.path();
    weights.download(verbose);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    let model = DinoVisionTransformer::new(vb, 12, 768, 12, 14, 518).unwrap();

    Ok(model)
}

pub fn load_dinobloom_vit_base(device: &Device, verbose: bool) -> Result<DinoVisionTransformer> {
    let weights = Weights::DinobloomVitBase;
    let path = weights.path();
    weights.download(verbose);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    let model = DinoVisionTransformer::new(vb, 12, 768, 12, 14, 224).unwrap();

    Ok(model)
}

pub fn load_subcell_vit_base(device: &Device, verbose: bool) -> Result<StandardVisionTransformer> {
    let weights = Weights::SubcellVitBase;
    let path = weights.path();
    weights.download(verbose);

    let config = StandardVisionTransformerConfig::vit_base_subcell();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    let model = StandardVisionTransformer::new(&config, vb).unwrap();

    Ok(model)
}

pub fn load_scdino_vit_small(device: &Device, verbose: bool) -> Result<StandardVisionTransformer> {
    let weights = Weights::ScdinoVitSmall;
    let path = weights.path();
    weights.download(verbose);

    let config = StandardVisionTransformerConfig::vit_base_scdino();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    let model = StandardVisionTransformer::new(&config, vb).unwrap();

    Ok(model)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_load_dinov2_small() {
        let model = load_dinov2_vit_small(&Device::Cpu, true);
        assert!(model.is_ok());
    }

    #[test]
    fn test_load_dinov2_base_imagenet() {
        let model = load_dinov2_vit_base(&Device::Cpu, true);
        assert!(model.is_ok());
    }

    #[test]
    fn test_load_dinobloom_vit_base() {
        let model = load_dinobloom_vit_base(&Device::Cpu, true);
        assert!(model.is_ok());
    }

    #[test]
    fn test_load_subcell_vit_base() {
        let model = load_subcell_vit_base(&Device::Cpu, true);
        assert!(model.is_ok());
    }

    #[test]
    fn test_load_scdino_vit_small() {
        let model = load_scdino_vit_small(&Device::Cpu, true);
        assert!(model.is_ok());
    }
}
