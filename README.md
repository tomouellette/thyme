<p align='center'>    
    <img width="65%" align='center' src="data/img/thyme-light.png#gh-light-mode-only"/>
    <img width="65%" align='center' src="data/img/thyme-dark.png#gh-dark-mode-only"/>
</p>

<hr>

`thyme` is a command-line tool for processing and profiling morphological data in bio-imaging datasets.

- [Installation](#installation)
  - [Cargo](#cargo)
  - [Pre-compiled binaries](#pre-compiled-binaries)
  - [Source](#source)
- [Usage](#usage)
  - [`thyme process`](#thyme-process)
  - [`thyme profile`](#thyme-profile)
  - [`thyme neural`](#thyme-neural)
  - [`thyme measure`](#thyme-measure)
  - [`thyme utils`](#thyme-utils)
  - [`thyme download`](#thyme-download)
- [License](#license)
- [Citation](#citation)

## Installation

### Cargo

`thyme` can be installed using the [rust](https://www.rust-lang.org/) package manager [cargo](https://github.com/rust-lang/cargoinstall):

```bash
cargo install --git https://github.com/tomouellette/thyme thyme-cli
```

Assuming `cargo` properly manages your `PATH`, you can verify the installation as follows.

```bash
thyme --help
```

### Pre-compiled binaries

Pre-built binaries for x86-64 linux, x86-64 apple, aarch-64 apple, and x86-64 windows are available for download in [releases](https://github.com/tomouellette/thyme/releases). Note that the pre-built binaries are untested on windows.

### Source

If you have [rust](https://www.rust-lang.org/) (tested on `1.86.0`) and [cargo](https://github.com/rust-lang/cargoinstall) installations, you can build directly from source as follows.

```bash
git clone https://github.com/tomouellette/thyme
cargo build --release
./target/release/thyme --help
```

If you want to reduce compile time, `lto` can be disabled in `Cargo.toml`. However this may increase binary size (particularly on linux) and possibly reduce performance slightly.

### Conda

A future [`conda`](https://anaconda.org/) release is being planned.

## Usage

### `thyme process`

Given a set of [valid input images](https://github.com/tomouellette/thyme/tree/main/thyme-core), `thyme` can extract object-level data for a variety of [segmentation formats](https://github.com/tomouellette/thyme/tree/main/thyme-core) including masks, polygons, and bounding boxes. Object-level data (e.g. cropped objects, polygons, etc.) can be extracted and saved as follows.

```bash
# Profile image-mask pairs stored in different directories
thyme process mask -i images/ -s masks/ -o data/ -v

# Profile image-mask pairs stored in the same directory
thyme process mask -i images/ --image-substring _image --mask-substring _mask -o data/ -v

# Profile image-polygon pairs stored in different directories
thyme process polygons -i images/ -s polygons/ -o data -v

# Profile image-polygon pairs stored in the same directory
thyme process polygons -i images/ --image-substring _image --polygon-substring _polygon -o data/ -v

# Profile image-bounding-box pairs stored in different directories
thyme process boxes -i images/ -s bounding_boxes/ -o data/ -v

# Profile image-bounding-box pairs stored in the same directory
thyme process boxes -i images/ --image-substring _image --box-substring _bounding_box -o data/ -v
```

For a more controlled run, a variety of flags can be set for all the process commands.

```bash
thyme process mask \
    -i images/ \            # Directory containing images
    -s masks/ \             # Directory containing masks
    -o data/ \              # Output directory
    --image-substring _red  # Only process images with this substring
    --mask-substring _run1  # Only process masks with this substring
    --mode cfbmpx \         # Extract specific object features (e.g. b = background pixels)
    --pad 10 \              # Padding around the object mask
    --min-size 5 \          # Minimum size (width/height) of analyzed objects
    --drop-borders \        # Drop objects that touch the image border
    --image-format png \    # Output format for object images
    --array-format json \   # Output format for polygons and bounding boxes
    --threads 8 \           # Max number of concurrent tasks (defaults to 8)
    -v                      # Verbose output
```

### `thyme profile`

Given a set of [valid input images](https://github.com/tomouellette/thyme/tree/main/thyme-core), `thyme` can compute object-level morphological descriptors across a variety of paired [segmentation formats](https://github.com/tomouellette/thyme/tree/main/thyme-core) including masks, polygons, and bounding boxes. Descriptors can be computed and saved as follows.

```bash
# Profile image-mask pairs stored in different directories
thyme profile mask -i images/ -s masks/ -o descriptors.csv -v

# Profile image-mask pairs stored in the same directory
thyme profile mask -i images/ --image-substring _image --mask-substring _mask -o descriptors.csv -v

# Profile image-polygon pairs stored in different directories
thyme profile polygons -i images/ -s polygons/ -o descriptors.csv -v

# Profile image-polygon pairs stored in the same directory
thyme profile polygons -i images/ --image-substring _image --polygon-substring _polygon -o descriptors.csv -v

# Profile image-bounding-box pairs stored in different directories
thyme profile boxes -i images/ -s bounding_boxes/ -o descriptors.csv -v

# Profile image-bounding-box pairs stored in the same directory
thyme profile boxes -i images/ --image-substring _image --box-substring _bounding_box -o descriptors.csv -v
```

For a more controlled run, a variety of flags can be set for all the profile commands.

```bash
thyme profile mask \
    -i images/ \            # Directory containing images
    -s masks/ \             # Directory containing masks
    -o descriptors.csv \    # Output directory or file (.csv, .txt, .tsv, .pq)
    --image-substring _red  # Only process images with this substring
    --mask-substring _dark  # Only process masks with this substring
    --mode cmfbp \          # Compute descriptors on different image features (eg f = foreground pixels)
    --pad 10 \              # Padding around the object mask
    --min-size 5.0 \        # Minimum size (width/height) of analyzed objects
    --drop-borders \        # Drop objects that touch the image border
    --threads 8 \           # Optional number of threads (or automatically selects)
    -v                      # Verbose output
```

### `thyme neural`

Given a set of [valid input images](https://github.com/tomouellette/thyme/tree/main/thyme-core), `thyme` can compute object-level self-supervised features (aka. 'deep profiles') across a variety of paired [segmentation formats](https://github.com/tomouellette/thyme/tree/main/thyme-core) including masks, polygons, and bounding boxes. Features can be computed and saved as follows.

```bash
# Generate features from image-mask pairs stored in different directories
thyme neural mask -i images/ -s masks/ -o features.npz --model dino_vit_small -v

# Generate features from image-mask pairs stored in the same directory
thyme neural mask -i images/ --image-substring _image --mask-substring _mask -o features.npz --model dino_vit_small -v

# Generate features from image-polygon pairs stored in different directories
thyme neural polygons -i images/ -s polygons/ -o features.npz --model dino_vit_small -v

# Generate features from image-polygon pairs stored in the same directory
thyme neural polygons -i images/ --image-substring _image --polygon-substring _polygon -o features.npz --model dino_vit_small -v

# Generate features from image-bounding-box pairs stored in different directories
thyme neural boxes -i images/ -s bounding-boxes/ -o features.npz --model dino_vit_small -v

# Generate features from image-bounding-box pairs stored in the same directory
thyme neural boxes -i images/ --image-substring _image --box-substring _bounding_box -o features.npz --model dino_vit_small -v
```

For a more controlled run, a variety of flags can be set for all the neural commands.

```bash
thyme neural mask \
    -i images/ \              # Directory containing images
    -s masks/ \               # Directory containing masks
    -o features.npz \         # Output directory or file (.csv, .txt, .pq, .npy, .npz)
    --image-substring _red    # Only process images with this substring
    --mask-substring _dark    # Only process masks with this substring
    --model dino_vit_small \  # Compute features using different self-supervised models
    --pad 10 \                # Padding around the object mask
    --min-size 5.0 \          # Minimum size (width/height) of analyzed objects
    --drop-borders \          # Drop objects that touch the image border
    --threads 8 \             # Optional number of threads (or automatically selects)
    -v                        # Verbose output
```

### `thyme measure`

If you want to compute quantitative features directly from images or polygons without associated segmentation data, then you can use `thyme measure`. Various quantitative features can be computed and saved as follows.

```bash
# Measure intensity descriptors for a single image (to stdout)
thyme measure intensity -i image.png

# Measure intensity descriptors for images stored in a directory
thyme measure intensity -i images/ -o descriptors.csv --image-substring _image -v

# Measure moment descriptors for a single image (to stdout)
thyme measure moments -i image.png

# Measure moment descriptors for images stored in a directory
thyme measure moments -i images/ -o descriptors.csv --image-substring _image -v

# Measure texture descriptors for a single image (to stdout)
thyme measure texture -i image.png

# Measure texture descriptors for images stored in a directory
thyme measure texture -i images/ -o descriptors.csv --image-substring _image -v

# Measure zernike descriptors for a single image (to stdout)
thyme measure zernike -i image.png

# Measure zernike descriptors for images stored in a directory
thyme measure zernike -i images/ -o descriptors.csv --image-substring _image -v

# Measure form descriptors for a single set of polygons (to stdout)
thyme measure form -i polygons.json

# Measure form descriptors for polygons stored in a directory
thyme measure form -i polygons/ -o descriptors.csv --polygon-substring _polygon -v
```

Self-supervised features from a variety of pre-trained models can also easily be computed using `thyme measure`. 

```bash
# Measure self-supervised features for a single image (to stdout)
thyme measure neural -i image.png --model dino_vit_small

# Measure self-supervised features for images stored in a directory
thyme measure neural -i images/ -o embeddings.npz --model dino_vit_small --image-substring _image -v

# Measure self-supervised features for images stored in a directory on leveraing your apple silicon GPU
thyme measure neural -i images/ -o embeddings.npz --model dino_vit_small --image-substring _image --device metal -v
```

Of note, generating self-supervised embeddings from pre-extracted images will be much faster (on GPU or via multi-threading) than performing object-level computation on image-segment pairs. Therefore we recommend using `thyme neural [segment]` for cases where you are storage-constrained and `thyme process [segment]` then `thyme measure neural` for cases where you require faster object-level embeddings.

### `thyme utils`

Additional utilities to convert between data formats (e.g. images to zarr arrays, segmentation masks to polygons, etc.) are available using `thyme utils`. Various non-destructive conversions can be performed as follows.

```bash
# Convert images with the same number of channels to a zarr v3 group
thyme utils images2zarr -i images/ -o images.zarr --width 32 --height 32 --channels 1 --dtype f32 --gzip-compression 5 --image-substring _image -v

# Convert a single segmentation mask to polygon format
thyme utils mask2polygons -i mask.png -o polygons.json

# Convert a folder of segmentation masks to polygon format
thyme utils mask2polygons -i masks/ -o polygons/ --mask-substring _mask -v

# Convert a single segmentation mask to bounding boxes format
thyme utils mask2boxes -i mask.png -o boxes.json

# Convert a folder of segmentation masks to bounding boxes format
thyme utils mask2boxes -i masks/ -o boxes/ --mask-substring _mask -v
```

Note that `images2zarrs` encodes image name strings as fixed-width numpy-style arrays (max length of 100). We currently do this as current zarr string decoding is inconsistent across different implementations. If you are loading the data in python, the saved image names can be mapped to strings via utf8 decoding as follows.

```python
import zarr
images2zarr = zarr.open("zarr_images.zarr")
strings = [bytes(row).split(b"\x00", 1)[0].decode("utf-8") for row in images2zarr["names"][:]]
```

### `thyme download`

To enable easier testing and model development/evaluation, we have curated and standardized a variety of previously annotated or generated bio-imaging datasets. We have also collected a variety of pre-trained neural network models for generating self-supervised embeddings. Below we provide an overview of the available datasets and pre-trained weights.

### Segmentation datasets

Each segmentation dataset was preprocessed and standardized to include images, segmentation masks, segmentation polygons, and object bounding boxes. Please check the original references and licenses to ensure the license supports your use case. You can download a segmentation dataset as follows.

```bash
# List all available segmentation datasets
thyme download segmentation --list

# Download all available segmentation datasets
thyme download segmentation --all -o datasets/ -v

# Download a specific segmentation dataset
thyme download segmentation -n vicar_2021 -o datasets/ -v
```

Below we provide a table of the available segmentation datasets in the current `thyme` release.

| Dataset | Author | Size (GB) | License |
| ------- | ------ | --------- | ------- |
| [almeida_2023](https://www.nature.com/articles/s41467-023-39676-y) | Almeida et al. 2023 | 0.927 | CC BY 4.0 |
| [arvidsson_2022](https://www.sciencedirect.com/science/article/pii/S2352340922009726?via%3Dihub) | Arvidsson et al. 2022 | 0.028 | CC BY 4.0 |
| [cellpose_2021](https://www.nature.com/articles/s41592-020-01018-x) | Stringer et al. 2021 | 0.356 | Custom NC |
| [conic_2022](https://conic-challenge.grand-challenge.org/) | Graham et al. 2022 | 1.920 | CC BY-NC 4.0 |
| [cryonuseg_2021](https://www.sciencedirect.com/science/article/pii/S0010482521001438) | Mahbod et al. 2021 | 0.031 | MIT |
| [dsb_2019](https://www.nature.com/articles/s41592-019-0612-7) | Caicedo et al. 2019 | 0.112 | CC0 1.0 Universal |
| [hpa_2022](https://zenodo.org/records/6538890) | HPA 2022 | 1.630 | CC BY 4.0 |
| [livecell_2021](https://www.nature.com/articles/s41592-021-01249-6) | Edlund et al. 2021 | 3.260 | CC BY-NC 4.0 |
| [nuinseg_2024](https://www.nature.com/articles/s41597-024-03117-2) | Mahbod et al. 2024 | 0.347 | MIT |
| [pannuke_2020](https://arxiv.org/abs/2003.10778) | Gamper et al. 2020 | 1.250 | CC BY-NC-SA 4.0 |
| [tissuenet_2022](https://www.nature.com/articles/s41587-021-01094-0) | Greenwald et al. 2022 | 4.270 | Modified NC Apache |
| [vicar_2021](https://zenodo.org/records/5153251) | Vicar et al. 2021 | 0.113 | CC BY 4.0 |

### Benchmark datasets

Benchmark datasets provide single cell or single object images to evaluate the predictive performance of descriptors or self-supervised embeddings. Each dataset includes single object images, masks, polygons, and bounding boxes. Note that some of the single object segmentation masks were generated roughly and can be improved if desired. We also provide a synthetic image dataset for evaluating runtime performance of various processing/profiling methods. You can download a benchmark dataset as follows.

```bash
# List all available classification datasets
thyme download benchmark --list

# Download all available classification datasets
thyme download benchmark --all -o datasets/ -v

# Download a specific classification dataset
thyme download benchmark -n amgad_2022 -o datasets/ -v
```

Below we provide a table of the available classification datasets in the current `thyme` release.

| Dataset | Author | Size (GB) | License |
| ------- | ------ | --------- | ------- |
| [amgad_2022](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac037/6586817) | Amgad et al. 2022 | 0.062 | CC0 1.0 |
| [cnmc_2019](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223) | C-NMC Challenge | 0.182 | CC BY 3.0 |
| [fracatlas_2023](https://www.nature.com/articles/s41597-023-02432-4) | Abedeen et al. 2023 | 0.247 | CC BY 4.0 |
| [isic_2019](https://challenge.isic-archive.com/data/#2019) | ISIC | 1.140 | CC BY-NC 4.0 |
| [kermany_2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) | Kermany et al. 2018 | 0.638 | CC BY 4.0 |
| [kromp_2023](https://www.nature.com/articles/s41597-023-02182-3) | Kromp et al. 2023 | 0.025 | CC BY 4.0 |
| [matek_2021](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/) | Matek et al. 2021 | 0.508 | CC BY 4.0 |
| [murphy_2001](https://murphylab.web.cmu.edu/data/) | Murphy et al. 2001 | 0.033 | MIT |
| [opencell_2024](https://virtualcellmodels.cziscience.com/dataset/0192bb9e-780b-74fc-a16b-d592aa89cacd?utm_source=czi&utm_campaign=MVP_launch&utm_medium=blog) | OpenCell | 1.030 | MIT |
| [phillip_2021](https://www.nature.com/articles/s41596-020-00432-x) | Phillip et al. 2021 | 0.032 | MIT |
| [recursion_2019](https://www.rxrx.ai/) | Recursion | 0.037 | CC BY-NC-SA 4.0 |
| [verma_2021](https://monusac-2020.grand-challenge.org/) | Verma et al. 2021 | 0.021 | CC BY-NC-SA 4.0 |
| runtime | Ouellette et al. 2025 | 0.018 | MIT |

### Weights

The self-supervised embeddings generated via `thyme neural` or `thyme measure neural` are made possible by leveraging a variety of open source pre-trained neural networks. Models can be pre-downloaded or will be downloaded on first use. You can download pre-trained weights as follows.

```bash
# Optionally set a cache to save models (defaults to ~/.thyme_cache)
export THYME_CACHE=/your/new/cache/location

# List all available pre-trained weights
thyme download weights --list

# Download all available pre-trained weightsdatasets
thyme download weights --all -v

# Download specific pre-trained weights dataset
thyme download weights -n dino_vit_small -v
```

Below we provide a table of the available weights in the current `thyme` release.

|       Model        |       Author        | Size (GB)  |      License       |
| ------------------ | ------------------- | ---------- | ------------------ |
|   [dino_vit_small](https://github.com/huggingface/candle)   | Huggingface/Candle  |   0.097    | Apache License 2.0 |
|   [dino_vit_base](https://github.com/huggingface/candle) | Huggingface/Candle  |   0.330    | Apache License 2.0 |
| [dinobloom_vit_base](https://github.com/marrlab/DinoBloom) |      Marr Lab       |   0.330    | Apache License 2.0 |
|  [scdino_vit_small](https://github.com/JacobHanimann/scDINO) |     Snijder Lab     |   0.097    | Apache License 2.0 |
|  [subcell_vit_base](https://github.com/CellProfiling/SubCellPortable/tree/main) |    Lundberg Lab     |   0.330    |    MIT License     |

If you would like another model added to `thyme`, please open an issue providing a link to the original model implementation and the associated open source weights. For each new model, we have to generate a rust implementation for compatibility with `thyme neural` and `thyme measure neural`.

## License

`thyme` is licensed under the [BSD 3-Clause License](https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_(%22BSD_License_2.0%22,_%22Revised_BSD_License%22,_%22New_BSD_License%22,_or_%22Modified_BSD_License%22)) (see [LICENSE](https://github.com/tomouellette/thyme/blob/main/LICENSE.txt)).

You may not use this file except in compliance with the license. A copy of the license has been included in the root of the repository. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the BSD 3-Clause license, shall be licensed as above, without any additional terms or conditions.

## Citation

> T.W. Ouellette, Y. Shao, P. Awadalla. Scalable processing for image-based cell profiling with thyme. bioRxiv (2025).
