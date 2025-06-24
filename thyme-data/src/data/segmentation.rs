// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use crate::request;
use std::path::Path;

// NOTE: This download scheme isn't really good practice as any
// new dataset additions will require a new release of the library.
// BUT, if there's request, then I'll move the identifiers outside
// the library to allow for arbitrary data/dataset additions.

/// Available annotated/segmentation datasets in the thyme library.
pub enum SegmentationDatasets {
    Almeida2023,
    Arvidsson2022,
    Cellpose2021,
    Conic2022,
    Cryonuseg2021,
    Dsb2019,
    Hpa2022,
    Livecell2021,
    Nuinseg2024,
    Pannuke2020,
    Tissuenet2022,
    Vicar2021,
}

impl SegmentationDatasets {
    /// Select a dataset from the available datasets.
    pub fn select(name: &str) -> Self {
        match name {
            "almeida_2023" => SegmentationDatasets::Almeida2023,
            "arvidsson_2022" => SegmentationDatasets::Arvidsson2022,
            "cellpose_2021" => SegmentationDatasets::Cellpose2021,
            "conic_2022" => SegmentationDatasets::Conic2022,
            "cryonuseg_2021" => SegmentationDatasets::Cryonuseg2021,
            "dsb_2019" => SegmentationDatasets::Dsb2019,
            "hpa_2022" => SegmentationDatasets::Hpa2022,
            "livecell_2021" => SegmentationDatasets::Livecell2021,
            "nuinseg_2024" => SegmentationDatasets::Nuinseg2024,
            "pannuke_2020" => SegmentationDatasets::Pannuke2020,
            "tissuenet_2022" => SegmentationDatasets::Tissuenet2022,
            "vicar_2021" => SegmentationDatasets::Vicar2021,
            _ => {
                let msg = format!(
                    "[thyme::data::segmentation] Dataset {} not found. Avalaible datasets include: {}",
                    name,
                    "almeida_2023, arvidsson_2022, cellpose_2021, conic_2022, cryonuseg_2021, dsb_2019, hpa_2022, livecell_2021, nuinseg_2024, pannuke_2020, tissuenet_2022, vicar_2021."
                );
                eprintln!("{}", msg);
                std::process::exit(1);
            }
        }
    }

    /// Return an iterator over the enum members.
    pub fn iter() -> impl Iterator<Item = &'static SegmentationDatasets> {
        static ANNOTATED: [SegmentationDatasets; 12] = [
            SegmentationDatasets::Almeida2023,
            SegmentationDatasets::Arvidsson2022,
            SegmentationDatasets::Cellpose2021,
            SegmentationDatasets::Conic2022,
            SegmentationDatasets::Cryonuseg2021,
            SegmentationDatasets::Dsb2019,
            SegmentationDatasets::Hpa2022,
            SegmentationDatasets::Livecell2021,
            SegmentationDatasets::Nuinseg2024,
            SegmentationDatasets::Pannuke2020,
            SegmentationDatasets::Tissuenet2022,
            SegmentationDatasets::Vicar2021,
        ];

        ANNOTATED.iter()
    }

    /// Get the name of the model saved on Google drive.
    pub fn name(&self) -> &str {
        match self {
            SegmentationDatasets::Almeida2023 => "almeida_2023",
            SegmentationDatasets::Arvidsson2022 => "arvidsson_2022",
            SegmentationDatasets::Cellpose2021 => "cellpose_2021",
            SegmentationDatasets::Conic2022 => "conic_2022",
            SegmentationDatasets::Cryonuseg2021 => "cryonuseg_2021",
            SegmentationDatasets::Dsb2019 => "dsb_2019",
            SegmentationDatasets::Hpa2022 => "hpa_2022",
            SegmentationDatasets::Livecell2021 => "livecell_2021",
            SegmentationDatasets::Nuinseg2024 => "nuinseg_2024",
            SegmentationDatasets::Pannuke2020 => "pannuke_2020",
            SegmentationDatasets::Tissuenet2022 => "tissuenet_2022",
            SegmentationDatasets::Vicar2021 => "vicar_2021",
        }
    }

    /// Get the Google drive file identifier for the saved model.
    pub fn file_id(&self) -> &str {
        match self {
            SegmentationDatasets::Almeida2023 => "1BlHvG0MkWwuqGA3ImUJ09D9E7rsVt5ER",
            SegmentationDatasets::Arvidsson2022 => "12Cwk5MX3V9z_2KmBJyn5jXc-JuW7-e2k",
            SegmentationDatasets::Cellpose2021 => "12Z9PpJEdSE0bHALNxpAAeD6WA9aBhMEO",
            SegmentationDatasets::Conic2022 => "1nXOnDkWpRfU5iGXFZe06-CQaAMFq13f_",
            SegmentationDatasets::Cryonuseg2021 => "1cfIY9BSlTe0RNaq1V8fZmKJwWyBs4WEj",
            SegmentationDatasets::Dsb2019 => "1qgAyMcrZwLudlA4vjy7jwuKTjAxT7Ky2",
            SegmentationDatasets::Hpa2022 => "1NyV6xuIAIuaSiXp0H-4VCV8tjaNSpXtX",
            SegmentationDatasets::Livecell2021 => "1JNXkZS0QSQW25b-opoyKomPKCfO_3pkx",
            SegmentationDatasets::Nuinseg2024 => "1gSmbsfhO7aP1yBB5R9XMMrAH4hy-Thmm",
            SegmentationDatasets::Pannuke2020 => "1J9CeH9t23EpottNyUKeBBkYpTfMR3EgT",
            SegmentationDatasets::Tissuenet2022 => "1ilHrzUuGfobSdFmTezyynCWCLIoJwaHQ",
            SegmentationDatasets::Vicar2021 => "12tJOlIHZPFqp8GLek_jV__Uhhgsa530_",
        }
    }

    /// Get the usage license for a model.
    pub fn license(&self) -> &str {
        match self {
            SegmentationDatasets::Almeida2023 => "CC BY 4.0",
            SegmentationDatasets::Arvidsson2022 => "CC BY 4.0",
            SegmentationDatasets::Cellpose2021 => "Custom NC",
            SegmentationDatasets::Conic2022 => "CC BY-NC 4.0",
            SegmentationDatasets::Cryonuseg2021 => "MIT",
            SegmentationDatasets::Dsb2019 => "CC0 1.0 Universal",
            SegmentationDatasets::Hpa2022 => "CC BY 4.0",
            SegmentationDatasets::Livecell2021 => "CC BY-NC 4.0",
            SegmentationDatasets::Nuinseg2024 => "MIT",
            SegmentationDatasets::Pannuke2020 => "CC BY-NC-SA 4.0",
            SegmentationDatasets::Tissuenet2022 => "Modified NC Apache",
            SegmentationDatasets::Vicar2021 => "CC BY 4.0",
        }
    }

    /// Get the authors of the model weights.
    pub fn data_authors(&self) -> &str {
        match self {
            SegmentationDatasets::Almeida2023 => "Almeida et al. 2023",
            SegmentationDatasets::Arvidsson2022 => "Arvidsson et al. 2022",
            SegmentationDatasets::Cellpose2021 => "Stringer et al. 2021",
            SegmentationDatasets::Conic2022 => "Graham et al. 2022",
            SegmentationDatasets::Cryonuseg2021 => "Mahbod et al. 2021",
            SegmentationDatasets::Dsb2019 => "Caicedo et al. 2019",
            SegmentationDatasets::Hpa2022 => "HPA 2022",
            SegmentationDatasets::Livecell2021 => "Edlund et al. 2021",
            SegmentationDatasets::Nuinseg2024 => "Mahbod et al. 2024",
            SegmentationDatasets::Pannuke2020 => "Gamper et al. 2020",
            SegmentationDatasets::Tissuenet2022 => "Greenwald et al. 2022",
            SegmentationDatasets::Vicar2021 => "Vicar et al. 2021",
        }
    }

    /// Get the size of the model in GB.
    pub fn data_size(&self) -> &str {
        match self {
            SegmentationDatasets::Almeida2023 => "0.927",
            SegmentationDatasets::Arvidsson2022 => "0.028",
            SegmentationDatasets::Cellpose2021 => "0.356",
            SegmentationDatasets::Conic2022 => "1.920",
            SegmentationDatasets::Cryonuseg2021 => "0.031",
            SegmentationDatasets::Dsb2019 => "0.112",
            SegmentationDatasets::Hpa2022 => "1.630",
            SegmentationDatasets::Livecell2021 => "3.260",
            SegmentationDatasets::Nuinseg2024 => "0.347",
            SegmentationDatasets::Pannuke2020 => "1.250",
            SegmentationDatasets::Tissuenet2022 => "4.270",
            SegmentationDatasets::Vicar2021 => "0.113",
        }
    }

    /// Download the dataset to an output directory.
    pub fn download(&self, output: &Path, verbose: bool) {
        let filename = format!("{}.tar.gz", self.name().replace("_", "-"));

        if !output.join(&filename).exists() {
            request::download_file(self.file_id(), output, &filename, !verbose).unwrap();

            if !output.join(&filename).exists() {
                eprintln!("[thyme::data::segmentation] Failed to download segmentation dataset.");
                std::process::exit(1);
            }
        }
    }
}
