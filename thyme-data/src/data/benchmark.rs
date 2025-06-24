// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use crate::request;
use std::path::Path;

// NOTE: This download scheme isn't really good practice as any
// new dataset additions will require a new release of the library.
// BUT, if there's request, then I'll move the identifiers outside
// the library to allow for arbitrary data/dataset additions.

/// Available neural network models in the thyme library.
pub enum BenchmarkDatasets {
    Amgad2022,
    Cnmc2019,
    Fracatlas2023,
    Isic2019,
    Kermany2018,
    Kromp2023,
    Matek2021,
    Murphy2001,
    Opencell2024,
    Phillip2021,
    Recursion2019,
    Verma2021,
    Runtime,
}

impl BenchmarkDatasets {
    /// Select a dataset from the available benchmarks.
    pub fn select(name: &str) -> Self {
        match name {
            "amgad_2022" => BenchmarkDatasets::Amgad2022,
            "cnmc_2019" => BenchmarkDatasets::Cnmc2019,
            "fracatlas_2023" => BenchmarkDatasets::Fracatlas2023,
            "isic_2019" => BenchmarkDatasets::Isic2019,
            "kermany_2018" => BenchmarkDatasets::Kermany2018,
            "kromp_2023" => BenchmarkDatasets::Kromp2023,
            "matek_2021" => BenchmarkDatasets::Matek2021,
            "murphy_2001" => BenchmarkDatasets::Murphy2001,
            "opencell_2024" => BenchmarkDatasets::Opencell2024,
            "phillip_2021" => BenchmarkDatasets::Phillip2021,
            "recursion_2019" => BenchmarkDatasets::Recursion2019,
            "verma_2021" => BenchmarkDatasets::Verma2021,
            "runtime" => BenchmarkDatasets::Runtime,
            _ => {
                let msg = format!(
                    "[thyme::data::benchmark] Dataset {} not found. Avalaible benchmark datasets include: {}",
                    name,
                    "amgad_2022, cnmc_2019, fracatlas_2023, isic_2019, kermany_2018, kromp_2023, matek_2021, murphy_2001, opencell_2024, phillip_2021, recursion_2019, verma_2021, runtime."
                );
                eprintln!("{}", msg);
                std::process::exit(1);
            }
        }
    }

    /// Return an iterator over the enum members.
    pub fn iter() -> impl Iterator<Item = &'static BenchmarkDatasets> {
        static BENCHMARKS: [BenchmarkDatasets; 13] = [
            BenchmarkDatasets::Amgad2022,
            BenchmarkDatasets::Cnmc2019,
            BenchmarkDatasets::Fracatlas2023,
            BenchmarkDatasets::Isic2019,
            BenchmarkDatasets::Kermany2018,
            BenchmarkDatasets::Kromp2023,
            BenchmarkDatasets::Matek2021,
            BenchmarkDatasets::Murphy2001,
            BenchmarkDatasets::Opencell2024,
            BenchmarkDatasets::Phillip2021,
            BenchmarkDatasets::Recursion2019,
            BenchmarkDatasets::Verma2021,
            BenchmarkDatasets::Runtime,
        ];

        BENCHMARKS.iter()
    }

    /// Get the name of the model saved on Google drive.
    pub fn name(&self) -> &str {
        match self {
            BenchmarkDatasets::Amgad2022 => "amgad_2022",
            BenchmarkDatasets::Cnmc2019 => "cnmc_2019",
            BenchmarkDatasets::Fracatlas2023 => "fracatlas_2023",
            BenchmarkDatasets::Isic2019 => "isic_2019",
            BenchmarkDatasets::Kermany2018 => "kermany_2018",
            BenchmarkDatasets::Kromp2023 => "kromp_2023",
            BenchmarkDatasets::Matek2021 => "matek_2021",
            BenchmarkDatasets::Murphy2001 => "murphy_2001",
            BenchmarkDatasets::Opencell2024 => "opencell_2024",
            BenchmarkDatasets::Phillip2021 => "phillip_2021",
            BenchmarkDatasets::Recursion2019 => "recursion_2019",
            BenchmarkDatasets::Verma2021 => "verma_2021",
            BenchmarkDatasets::Runtime => "runtime",
        }
    }

    /// Get the Google drive file identifier for the saved model.
    pub fn file_id(&self) -> &str {
        match self {
            BenchmarkDatasets::Amgad2022 => "1JHlGon82bYPhpeOwbRYxz4uRxhcxasr3",
            BenchmarkDatasets::Cnmc2019 => "1a7Wt0kwt3Uq1NKMtBsWmesMH4tCwqkgi",
            BenchmarkDatasets::Fracatlas2023 => "1vyXNA4bxMFk-7Hw59TPiWIfX-BzTSmCd",
            BenchmarkDatasets::Isic2019 => "1CDGbcBxs7SUemGwpBtoVYJblqCNgw469",
            BenchmarkDatasets::Kermany2018 => "1Xk7LWa7HWzTN7Nxsa8MuefBmzNsz4VuM",
            BenchmarkDatasets::Kromp2023 => "16RXNWQXlw_scJ75DwowngxJHYB2rZsW7",
            BenchmarkDatasets::Matek2021 => "1BDYtZoqSUUZQmWgcEBtopaJTWgwrAXGz",
            BenchmarkDatasets::Murphy2001 => "1fl4dwbjX11SpDRhwbIi2-lbcswvyr1F_",
            BenchmarkDatasets::Opencell2024 => "1nlqt7ujUPciEoAKriIu_bqE5fUJZX4nx",
            BenchmarkDatasets::Phillip2021 => "1yE4BblXBAPJDT1AnK3gghHAS3cZUFCd6",
            BenchmarkDatasets::Recursion2019 => "1209hlaKcOqKdEGOwvlRhJakX8ciN-SX8",
            BenchmarkDatasets::Verma2021 => "1AyU-4-doJY2GX3dmf7ryPDCvDA_x4PPD",
            BenchmarkDatasets::Runtime => "1BlXIv49oxj2dsiiEASbTEiyh7QpIjYb_",
        }
    }

    /// Get the usage license for a model.
    pub fn license(&self) -> &str {
        match self {
            BenchmarkDatasets::Amgad2022 => "CC0 1.0",
            BenchmarkDatasets::Cnmc2019 => "CC BY 3.0",
            BenchmarkDatasets::Fracatlas2023 => "CC BY 4.0",
            BenchmarkDatasets::Isic2019 => "CC BY-NC 4.0",
            BenchmarkDatasets::Kermany2018 => "CC BY 4.0",
            BenchmarkDatasets::Kromp2023 => "CC BY 4.0",
            BenchmarkDatasets::Matek2021 => "CC BY 4.0",
            BenchmarkDatasets::Murphy2001 => "MIT",
            BenchmarkDatasets::Opencell2024 => "MIT",
            BenchmarkDatasets::Phillip2021 => "MIT",
            BenchmarkDatasets::Recursion2019 => "CC BY-NC-SA 4.0",
            BenchmarkDatasets::Verma2021 => "CC BY-NC-SA 4.0",
            BenchmarkDatasets::Runtime => "MIT",
        }
    }

    /// Get the authors of the model weights.
    pub fn data_authors(&self) -> &str {
        match self {
            BenchmarkDatasets::Amgad2022 => "Amgad et al. 2022",
            BenchmarkDatasets::Cnmc2019 => "C-NMC Challenge",
            BenchmarkDatasets::Fracatlas2023 => "Abedeen et al. 2023",
            BenchmarkDatasets::Isic2019 => "ISIC",
            BenchmarkDatasets::Kermany2018 => "Kermany et al. 2018",
            BenchmarkDatasets::Kromp2023 => "Kromp et al. 2023",
            BenchmarkDatasets::Matek2021 => "Matek et al. 2021",
            BenchmarkDatasets::Murphy2001 => "Murphy et al. 2001",
            BenchmarkDatasets::Opencell2024 => "OpenCell",
            BenchmarkDatasets::Phillip2021 => "Phillip et al. 2021",
            BenchmarkDatasets::Recursion2019 => "Recursion",
            BenchmarkDatasets::Verma2021 => "Verma et al. 2021",
            BenchmarkDatasets::Runtime => "MIT",
        }
    }

    /// Get the size of the model in GB.
    pub fn data_size(&self) -> &str {
        match self {
            BenchmarkDatasets::Amgad2022 => "0.062",
            BenchmarkDatasets::Cnmc2019 => "0.182",
            BenchmarkDatasets::Fracatlas2023 => "0.247",
            BenchmarkDatasets::Isic2019 => "1.140",
            BenchmarkDatasets::Kermany2018 => "0.638",
            BenchmarkDatasets::Kromp2023 => "0.025",
            BenchmarkDatasets::Matek2021 => "0.508",
            BenchmarkDatasets::Murphy2001 => "0.033",
            BenchmarkDatasets::Opencell2024 => "1.030",
            BenchmarkDatasets::Phillip2021 => "0.032",
            BenchmarkDatasets::Recursion2019 => "0.037",
            BenchmarkDatasets::Verma2021 => "0.021",
            BenchmarkDatasets::Runtime => "0.017",
        }
    }

    /// Download the dataset to an output directory.
    pub fn download(&self, output: &Path, verbose: bool) {
        let filename = format!("{}.tar.gz", self.name().replace("_", "-"));

        if !output.join(&filename).exists() {
            request::download_file(self.file_id(), output, &filename, !verbose).unwrap();

            if !output.join(&filename).exists() {
                eprintln!("[thyme::data::benchmark] Failed to download benchmark dataset.");
                std::process::exit(1);
            }
        }
    }
}
