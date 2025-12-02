# Command-Line Help for `pineapple`

This document contains the help content for the `pineapple` command-line program.

**Command Overview:**

* [`pineapple`↴](#pineapple)
* [`pineapple download`↴](#pineapple-download)
* [`pineapple download segmentation`↴](#pineapple-download-segmentation)
* [`pineapple download benchmark`↴](#pineapple-download-benchmark)
* [`pineapple download weights`↴](#pineapple-download-weights)
* [`pineapple measure`↴](#pineapple-measure)
* [`pineapple measure form`↴](#pineapple-measure-form)
* [`pineapple measure intensity`↴](#pineapple-measure-intensity)
* [`pineapple measure moments`↴](#pineapple-measure-moments)
* [`pineapple measure neural`↴](#pineapple-measure-neural)
* [`pineapple measure texture`↴](#pineapple-measure-texture)
* [`pineapple measure zernike`↴](#pineapple-measure-zernike)
* [`pineapple neural`↴](#pineapple-neural)
* [`pineapple neural boxes`↴](#pineapple-neural-boxes)
* [`pineapple neural mask`↴](#pineapple-neural-mask)
* [`pineapple neural polygons`↴](#pineapple-neural-polygons)
* [`pineapple process`↴](#pineapple-process)
* [`pineapple process boxes`↴](#pineapple-process-boxes)
* [`pineapple process mask`↴](#pineapple-process-mask)
* [`pineapple process polygons`↴](#pineapple-process-polygons)
* [`pineapple profile`↴](#pineapple-profile)
* [`pineapple profile boxes`↴](#pineapple-profile-boxes)
* [`pineapple profile mask`↴](#pineapple-profile-mask)
* [`pineapple profile polygons`↴](#pineapple-profile-polygons)
* [`pineapple utils`↴](#pineapple-utils)
* [`pineapple utils images2zarr`↴](#pineapple-utils-images2zarr)
* [`pineapple utils mask2boxes`↴](#pineapple-utils-mask2boxes)
* [`pineapple utils mask2polygons`↴](#pineapple-utils-mask2polygons)

## `pineapple`

Simplified processing for image-based cell profiling with pineapple

**Usage:** `pineapple [NAME] [COMMAND]`

###### **Subcommands:**

* `download` — Download standardized bioimaging datasets or pre-trained neural networks.
* `measure` — Measure quantitative features from bio-imaging data.
* `neural` — Compute object-level self-supervised features from image and segment pairs.
* `process` — Extract object-level data from image and segment pairs.
* `profile` — Compute object-level morphological descriptors from image and segment pairs.
* `utils` — General utilities for converting and transforming image/image-related data.

###### **Arguments:**

* `<NAME>`



## `pineapple download`

Download standardized bioimaging datasets or pre-trained neural networks.

**Usage:** `pineapple download
       download segmentation [OPTIONS]
       download benchmark [OPTIONS]
       download weights [OPTIONS]
       download help [COMMAND]...`

###### **Subcommands:**

* `segmentation` — Download standardized bioimaging segmentation datasets.
* `benchmark` — Download standardized single-object benchmark datasets.
* `weights` — Download pre-trained neural network weights.



## `pineapple download segmentation`

Download standardized bioimaging segmentation datasets.

**Usage:** `pineapple download segmentation [OPTIONS]`

###### **Options:**

* `-n`, `--name <NAME>` — Dataset name. Run pineapple download segmentation --list` to see all available datasets. Run `pineapple download segementation --all` to download all segmentation datasets
* `-o`, `--output <OUTPUT>` — Output path to save downloaded dataset.
* `-v`, `--verbose` — Verbose output.
* `--list` — List all available segmentation datasets.
* `--all` — Download all available segmentation datasets.



## `pineapple download benchmark`

Download standardized single-object benchmark datasets.

**Usage:** `pineapple download benchmark [OPTIONS]`

###### **Options:**

* `-n`, `--name <NAME>` — Dataset name. Run pineapple download benchmark --list` to see all available benchmarks. Run `pineapple download benchmark --all` to download all benchmark datasets
* `-o`, `--output <OUTPUT>` — Output path to save downloaded dataset.
* `-v`, `--verbose` — Verbose output.
* `--list` — List all available benchmark datasets.
* `--all` — Download all available benchmark datasets.



## `pineapple download weights`

Download pre-trained neural network weights.

**Usage:** `pineapple download weights [OPTIONS]`

###### **Options:**

* `-n`, `--name <NAME>` — Weights name.
* `-v`, `--verbose` — Verbose output.
* `--list` — List all available neural net weights.
* `--all` — Download all available neural net weights.



## `pineapple measure`

Measure quantitative features from bio-imaging data.

**Usage:** `pineapple measure
       measure form [OPTIONS] --polygons <POLYGONS>
       measure intensity [OPTIONS] --images <IMAGES>
       measure moments [OPTIONS] --images <IMAGES>
       measure neural [OPTIONS] --images <IMAGES>
       measure texture [OPTIONS] --images <IMAGES>
       measure zernike [OPTIONS] --images <IMAGES>
       measure help [COMMAND]...`

###### **Subcommands:**

* `form` — 
* `intensity` — 
* `moments` — 
* `neural` — 
* `texture` — 
* `zernike` — 



## `pineapple measure form`

**Usage:** `pineapple measure form [OPTIONS] --polygons <POLYGONS>`

###### **Options:**

* `-i`, `--polygons <POLYGONS>` — Polygons or polygons directory.
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq).
* `-v`, `--verbose` — Verbose output.
* `--polygon-substring <POLYGON_SUBSTRING>` — Substring specifying polygons (e.g. _polygons).
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple measure intensity`

**Usage:** `pineapple measure intensity [OPTIONS] --images <IMAGES>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image or image directory.
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq).
* `-v`, `--verbose` — Verbose output.
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple measure moments`

**Usage:** `pineapple measure moments [OPTIONS] --images <IMAGES>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image or image directory.
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq).
* `-v`, `--verbose` — Verbose output.
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple measure neural`

**Usage:** `pineapple measure neural [OPTIONS] --images <IMAGES>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image or image directory.
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq, .npy, .npz).
* `-m`, `--model <MODEL>` — Model name.

  Default value: `dino_vit_small`
* `-d`, `--device <DEVICE>` — Device (cpu, cuda, metal).

  Default value: `cpu`
* `-v`, `--verbose` — Verbose output.
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple measure texture`

**Usage:** `pineapple measure texture [OPTIONS] --images <IMAGES>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image or image directory.
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq).
* `-v`, `--verbose` — Verbose output.
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple measure zernike`

**Usage:** `pineapple measure zernike [OPTIONS] --images <IMAGES>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image or image directory.
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq).
* `-v`, `--verbose` — Verbose output.
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple neural`

Compute object-level self-supervised features from image and segment pairs.

**Usage:** `pineapple neural
       neural boxes [OPTIONS] --images <IMAGES> --output <OUTPUT>
       neural mask [OPTIONS] --images <IMAGES> --output <OUTPUT>
       neural polygons [OPTIONS] --images <IMAGES> --output <OUTPUT>
       neural help [COMMAND]...`

###### **Subcommands:**

* `boxes` — 
* `mask` — 
* `polygons` — 



## `pineapple neural boxes`

**Usage:** `pineapple neural boxes [OPTIONS] --images <IMAGES> --output <OUTPUT>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-s`, `--boxes <BOXES>` — Bounding box directory.
* `--device <DEVICE>` — Device (cpu, cuda, metal).

  Default value: `cpu`
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq, .npy, .npz).
* `-v`, `--verbose` — Verbose output.
* `-d`, `--drop-borders` — Exclude objects touching edge of image.
* `-m`, `--model <MODEL>` — Model name.

  Default value: `dino_vit_small`
* `-p`, `--pad <PAD>` — Add padding around objects before computing self-supervised features.

  Default value: `1`
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `--box-substring <BOX_SUBSTRING>` — Substring specifying bounding boxes (e.g. _boxes).
* `--min-size <MIN_SIZE>` — Exclude objects smaller than a minimum size.

  Default value: `1`
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple neural mask`

**Usage:** `pineapple neural mask [OPTIONS] --images <IMAGES> --output <OUTPUT>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-s`, `--masks <MASKS>` — Mask directory.
* `--device <DEVICE>` — Device (cpu, cuda, metal).

  Default value: `cpu`
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq, .npy, .npz).
* `-v`, `--verbose` — Verbose output.
* `-d`, `--drop-borders` — Exclude objects touching edge of image.
* `-m`, `--model <MODEL>` — Model name.

  Default value: `dino_vit_small`
* `-p`, `--pad <PAD>` — Add padding around objects before computing self-supervised features.

  Default value: `1`
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `--mask-substring <MASK_SUBSTRING>` — Substring specifying masks (e.g. _mask).
* `--min-size <MIN_SIZE>` — Exclude objects smaller than a minimum size.

  Default value: `1`
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple neural polygons`

**Usage:** `pineapple neural polygons [OPTIONS] --images <IMAGES> --output <OUTPUT>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-s`, `--polygons <POLYGONS>` — Polygons directory.
* `--device <DEVICE>` — Device (cpu, cuda, metal).

  Default value: `cpu`
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq, .npy, .npz).
* `-v`, `--verbose` — Verbose output.
* `-d`, `--drop-borders` — Exclude objects touching edge of image.
* `-m`, `--model <MODEL>` — Model name.

  Default value: `dino_vit_small`
* `-p`, `--pad <PAD>` — Add padding around objects before computing self-supervised features.

  Default value: `1`
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `--polygon-substring <POLYGON_SUBSTRING>` — Substring specifying polygons (e.g. _polygons).
* `--min-size <MIN_SIZE>` — Exclude objects smaller than a minimum size.

  Default value: `1`
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple process`

Extract object-level data from image and segment pairs.

**Usage:** `pineapple process
       process boxes [OPTIONS] --images <IMAGES> --output <OUTPUT>
       process mask [OPTIONS] --images <IMAGES> --output <OUTPUT>
       process polygons [OPTIONS] --images <IMAGES> --output <OUTPUT>
       process help [COMMAND]...`

###### **Subcommands:**

* `boxes` — 
* `mask` — 
* `polygons` — 



## `pineapple process boxes`

**Usage:** `pineapple process boxes [OPTIONS] --images <IMAGES> --output <OUTPUT>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-s`, `--boxes <BOXES>` — Bounding boxes directory.
* `-o`, `--output <OUTPUT>` — Output directory.
* `-v`, `--verbose` — Verbose output.
* `-d`, `--drop-borders` — Exclude objects touching edge of image.
* `-m`, `--mode <MODE>` — Mode. One or more of c (complete pixels) and/or x (bounding boxes).

  Default value: `cx`
* `-p`, `--pad <PAD>` — Add padding around extracted objects

  Default value: `1`
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `--box-substring <BOX_SUBSTRING>` — Substring specifying boxes (e.g. _boxes).
* `--min-size <MIN_SIZE>` — Exclude objects smaller than a minimum size.

  Default value: `1`
* `-e`, `--image-format <IMAGE_FORMAT>` — Format to save extracted object images (e.g. png, jpeg, npy).

  Default value: `png`
* `-a`, `--array-format <ARRAY_FORMAT>` — Format to save extracted polygons and/or bounding boxes (e.g. json).

  Default value: `json`
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple process mask`

**Usage:** `pineapple process mask [OPTIONS] --images <IMAGES> --output <OUTPUT>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-s`, `--masks <MASKS>` — Mask directory.
* `-o`, `--output <OUTPUT>` — Output directory.
* `-v`, `--verbose` — Verbose output.
* `-d`, `--drop-borders` — Exclude objects touching edge of image.
* `-m`, `--mode <MODE>` — Mode. One or more of c (complete pixels), f (foreground pixels), b (background pixels), m (binary mask), p (polygons), and x (bounding boxes).

  Default value: `cm`
* `-p`, `--pad <PAD>` — Add padding around extracted objects

  Default value: `1`
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `--mask-substring <MASK_SUBSTRING>` — Substring specifying masks (e.g. _mask).
* `--min-size <MIN_SIZE>` — Exclude objects smaller than a minimum size.

  Default value: `1`
* `-e`, `--image-format <IMAGE_FORMAT>` — Format to save extracted object images (e.g. png, jpeg, npy).

  Default value: `png`
* `-a`, `--array-format <ARRAY_FORMAT>` — Format to save extracted polygons and/or bounding boxes (e.g. json).

  Default value: `json`
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple process polygons`

**Usage:** `pineapple process polygons [OPTIONS] --images <IMAGES> --output <OUTPUT>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-s`, `--polygons <POLYGONS>` — Polygons directory.
* `-o`, `--output <OUTPUT>` — Output directory.
* `-v`, `--verbose` — Verbose output.
* `-d`, `--drop-borders` — Exclude objects touching edge of image.
* `-m`, `--mode <MODE>` — Mode. One or more of c (complete pixels), f (foreground pixels), b (background pixels), m (binary mask), p (polygons), and x (bounding boxes).

  Default value: `cm`
* `-p`, `--pad <PAD>` — Add padding around extracted objects

  Default value: `1`
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `--polygon-substring <POLYGON_SUBSTRING>` — Substring specifying polygons (e.g. _polygon).
* `--min-size <MIN_SIZE>` — Exclude objects smaller than a minimum size.

  Default value: `1`
* `-e`, `--image-format <IMAGE_FORMAT>` — Format to save extracted object images (e.g. png, jpeg, npy).

  Default value: `png`
* `-a`, `--array-format <ARRAY_FORMAT>` — Format to save extracted polygons and/or bounding boxes (e.g. json).

  Default value: `json`
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple profile`

Compute object-level morphological descriptors from image and segment pairs.

**Usage:** `pineapple profile
       profile boxes [OPTIONS] --images <IMAGES> --output <OUTPUT>
       profile mask [OPTIONS] --images <IMAGES> --output <OUTPUT>
       profile polygons [OPTIONS] --images <IMAGES> --output <OUTPUT>
       profile help [COMMAND]...`

###### **Subcommands:**

* `boxes` — 
* `mask` — 
* `polygons` — 



## `pineapple profile boxes`

**Usage:** `pineapple profile boxes [OPTIONS] --images <IMAGES> --output <OUTPUT>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-s`, `--boxes <BOXES>` — Bounding boxes directory.
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq).
* `-v`, `--verbose` — Verbose output.
* `-d`, `--drop-borders` — Exclude objects touching edge of image.
* `-m`, `--mode <MODE>` — Mode. Compute descriptors across one or more features including c (complete pixels) and/or x (bounding boxes).

  Default value: `cx`
* `-p`, `--pad <PAD>` — Add padding around extracted objects before computing profiles.

  Default value: `1`
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `--box-substring <BOX_SUBSTRING>` — Substring specifying bounding boxes (e.g. _boxes).
* `--min-size <MIN_SIZE>` — Exclude objects smaller than a minimum size.

  Default value: `1`
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple profile mask`

**Usage:** `pineapple profile mask [OPTIONS] --images <IMAGES> --output <OUTPUT>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-s`, `--masks <MASKS>` — Mask directory.
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq).
* `-v`, `--verbose` — Verbose output.
* `-d`, `--drop-borders` — Exclude objects touching edge of image.
* `-m`, `--mode <MODE>` — Mode. Compute descriptors across one or more features including c (complete pixels), f (foreground pixels), b (background pixels), m (binary mask), p (polygons), and x (bounding boxes).

  Default value: `cm`
* `-p`, `--pad <PAD>` — Add padding around extracted objects before computing profiles.

  Default value: `1`
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `--mask-substring <MASK_SUBSTRING>` — Substring specifying masks (e.g. _mask).
* `--min-size <MIN_SIZE>` — Exclude objects smaller than a minimum size.

  Default value: `1`
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple profile polygons`

**Usage:** `pineapple profile polygons [OPTIONS] --images <IMAGES> --output <OUTPUT>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-s`, `--polygons <POLYGONS>` — Polygons directory.
* `-o`, `--output <OUTPUT>` — Output directory or file (.csv, .txt, .tsv, .pq).
* `-v`, `--verbose` — Verbose output.
* `-d`, `--drop-borders` — Exclude objects touching edge of image.
* `-m`, `--mode <MODE>` — Mode. Compute descriptors across one or more features including c (complete pixels), f (foreground pixels), b (background pixels), m (binary mask), p (polygons), and x (bounding boxes).

  Default value: `cm`
* `-p`, `--pad <PAD>` — Add padding around extracted objects before computing profiles.

  Default value: `1`
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `--polygon-substring <POLYGON_SUBSTRING>` — Substring specifying polygons (e.g. _polygons).
* `--min-size <MIN_SIZE>` — Exclude objects smaller than a minimum size.

  Default value: `1`
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple utils`

General utilities for converting and transforming image/image-related data.

**Usage:** `pineapple utils
       utils images2zarr [OPTIONS] --images <IMAGES> --output <OUTPUT> --resize-width <RESIZE_WIDTH> --resize-height <RESIZE_HEIGHT> --channels <CHANNELS>
       utils mask2boxes [OPTIONS] --mask <MASK> --output <OUTPUT>
       utils mask2polygons [OPTIONS] --mask <MASK> --output <OUTPUT>
       utils help [COMMAND]...`

###### **Subcommands:**

* `images2zarr` — 
* `mask2boxes` — 
* `mask2polygons` — 



## `pineapple utils images2zarr`

**Usage:** `pineapple utils images2zarr [OPTIONS] --images <IMAGES> --output <OUTPUT> --resize-width <RESIZE_WIDTH> --resize-height <RESIZE_HEIGHT> --channels <CHANNELS>`

###### **Options:**

* `-i`, `--images <IMAGES>` — Image directory.
* `-o`, `--output <OUTPUT>` — Output zarr file.
* `--resize-width <RESIZE_WIDTH>` — Resize each image to specified width.
* `--resize-height <RESIZE_HEIGHT>` — Resize each image to specified height.
* `--channels <CHANNELS>` — Number of image channels.
* `--dtype <DTYPE>` — Cast subpixels to specified data type (u8, u16, u32, f32, or f64).

  Default value: `f32`
* `--gzip-compression <GZIP_COMPRESSION>` — Gzip compression level (0 - 9)

  Default value: `5`
* `-v`, `--verbose` — Verbose output.
* `--image-substring <IMAGE_SUBSTRING>` — Substring specifying images (e.g. _image).
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple utils mask2boxes`

**Usage:** `pineapple utils mask2boxes [OPTIONS] --mask <MASK> --output <OUTPUT>`

###### **Options:**

* `-i`, `--mask <MASK>` — Mask or mask directory.
* `-o`, `--output <OUTPUT>` — Output bounding boxes file.
* `-v`, `--verbose` — Verbose output.
* `--mask-substring <MASK_SUBSTRING>` — Substring specifying masks (e.g. _mask).
* `-t`, `--threads <THREADS>` — Number of threads.



## `pineapple utils mask2polygons`

**Usage:** `pineapple utils mask2polygons [OPTIONS] --mask <MASK> --output <OUTPUT>`

###### **Options:**

* `-i`, `--mask <MASK>` — Mask or mask directory.
* `-o`, `--output <OUTPUT>` — Output polygons file.
* `-v`, `--verbose` — Verbose output.
* `--mask-substring <MASK_SUBSTRING>` — Substring specifying masks (e.g. _mask).
* `-t`, `--threads <THREADS>` — Number of threads.



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>

