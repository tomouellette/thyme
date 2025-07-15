// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use crate::get_thyme_cache;
use crate::request;

// NOTE: This download scheme isn't really good practice as any
// new dataset additions will require a new release of the library.
// BUT, if there's request, then I'll move the identifiers outside
// the library to allow for arbitrary data/dataset additions.

/// Available neural network models in the thyme library.
pub enum Weights {
    DinoVitSmall,
    DinoVitBase,
    DinobloomVitBase,
    ScdinoVitSmall,
    SubcellVitBase,
}

impl Weights {
    /// Select a weights from the available weights.
    pub fn select(weights_name: &str) -> Self {
        match weights_name {
            "dino_vit_small" => Weights::DinoVitSmall,
            "dino_vit_base" => Weights::DinoVitBase,
            "dinobloom_vit_base" => Weights::DinobloomVitBase,
            "scdino_vit_small" => Weights::ScdinoVitSmall,
            "subcell_vit_base" => Weights::SubcellVitBase,
            _ => {
                let msg = format!(
                    "[thyme::data::weights] Weights {} not found. Avalaible weights include: {}",
                    weights_name,
                    "dino_vit_small, dino_vit_base, dinobloom_vit_base, scdino_vit_small, subcell_vit_base."
                );
                eprintln!("{}", msg);
                std::process::exit(1);
            }
        }
    }

    /// Return an iterator over the enum members.
    pub fn iter() -> impl Iterator<Item = &'static Weights> {
        static WEIGHTS: [Weights; 5] = [
            Weights::DinoVitSmall,
            Weights::DinoVitBase,
            Weights::DinobloomVitBase,
            Weights::ScdinoVitSmall,
            Weights::SubcellVitBase,
        ];

        WEIGHTS.iter()
    }

    /// Get the name of the model.
    pub fn model_name(&self) -> &str {
        match self {
            Weights::DinoVitSmall => "dino_vit_small",
            Weights::DinoVitBase => "dino_vit_base",
            Weights::DinobloomVitBase => "dinobloom_vit_base",
            Weights::ScdinoVitSmall => "scdino_vit_small",
            Weights::SubcellVitBase => "subcell_vit_base",
        }
    }

    /// Get the file name of the model saved on Google drive.
    fn file_name(&self) -> &str {
        match self {
            Weights::DinoVitSmall => "dinov2_vits14_imagenet.safetensors",
            Weights::DinoVitBase => "dinov2_vitb14_imagenet.safetensors",
            Weights::DinobloomVitBase => "dinov2_vitb14_dinobloom.safetensors",
            Weights::ScdinoVitSmall => "scdino_vit_small.safetensors",
            Weights::SubcellVitBase => "subcell_vit_base.safetensors",
        }
    }

    /// Get the Google drive file identifier for the saved model.
    pub fn file_id(&self) -> &str {
        match self {
            Weights::DinoVitSmall => "1xuyTyPsuPiDtec8ojZwAyXSDq9AzyPQX",
            Weights::DinoVitBase => "19vy-A-KTaaF3vsWKxu0JpA0gaATU52Gh",
            Weights::DinobloomVitBase => "1XhzSiO2IDKppr2UCTAio_niLSk5QA6hG",
            Weights::ScdinoVitSmall => "1omwQNJVMkrbYCstSF11p5HsErHzINTz6",
            Weights::SubcellVitBase => "1LZn3xlgVVd2jQIpXst4CMCYN58F-VG-x",
        }
    }

    /// Get the usage license for a model.
    pub fn license(&self) -> &str {
        match self {
            Weights::DinoVitSmall => "Apache License 2.0",
            Weights::DinoVitBase => "Apache License 2.0",
            Weights::DinobloomVitBase => "Apache License 2.0",
            Weights::ScdinoVitSmall => "Apache License 2.0",
            Weights::SubcellVitBase => "MIT License",
        }
    }

    /// Get the authors of the model weights.
    pub fn data_authors(&self) -> &str {
        match self {
            Weights::DinoVitSmall => "Huggingface/candle",
            Weights::DinoVitBase => "Huggingface/candle",
            Weights::DinobloomVitBase => "Marr Lab",
            Weights::ScdinoVitSmall => "Snijder Lab",
            Weights::SubcellVitBase => "Lundberg Lab",
        }
    }

    /// Get the size of the model in GB.
    pub fn data_size(&self) -> &str {
        match self {
            Weights::DinoVitSmall => "0.097",
            Weights::DinoVitBase => "0.330",
            Weights::DinobloomVitBase => "0.330",
            Weights::SubcellVitBase => "0.330",
            Weights::ScdinoVitSmall => "0.097",
        }
    }

    /// Download the model to the thyme cache.
    pub fn download(&self, verbose: bool) {
        let cache = get_thyme_cache();
        let model_name = cache.join(self.file_name());
        if !model_name.exists() {
            request::download_file(self.file_id(), cache.as_path(), self.file_name(), !verbose)
                .unwrap();

            if !model_name.exists() {
                eprintln!("[thyme::data::weights] Failed to download model weights.");
                std::process::exit(1);
            }
        }
    }

    /// Get path to model weights.
    pub fn path(&self) -> std::path::PathBuf {
        let cache = get_thyme_cache();
        cache.join(self.file_name())
    }
}
