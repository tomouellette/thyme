# thyme-core

`thyme-core` defines functions and algorithms for reading, processing, and profilingbio-imaging datasets.

## Supported image formats

`thyme` currently supports 1 or 3 channel images of the following formats (plus experimental support for N-channel numpy images).

| Format | Dependencies |
| ------ | ------------ |
| `jpeg/jpg` | [image-rs](https://github.com/image-rs/image) |
| `png` | [image-rs](https://github.com/image-rs/image) |
| `bmp` | [image-rs](https://github.com/image-rs/image) | 
| `tiff/tif` | [image-rs](https://github.com/image-rs/image) |
| `hdr` | [image-rs](https://github.com/image-rs/image) |
| `pbm` | [image-rs](https://github.com/image-rs/image) |
| `avif` | [image-rs](https://github.com/image-rs/image) |
| `tga` | [image-rs](https://github.com/image-rs/image) |
| `qoi` | [image-rs](https://github.com/image-rs/image) |
| `exr` | [image-rs](https://github.com/image-rs/image) |
| `webp` | [image-rs](https://github.com/image-rs/image) |
| `npy` | [npyz](https://github.com/ExpHP/npyz) |

## Supported segmentation formats

`thyme` currently supports reading and writing of masks, polygons, and bounding boxes.

| Format | Description |
| ----------------- | ----------- |
| [Binary mask](https://github.com/tomouellette/thyme/blob/main/data/tests/test_mask_binary.png) | `u8`, `u16`, or `u32` image where `0` indicates background and a positive integer indicates foreground. |
| [Integer mask](https://github.com/tomouellette/thyme/blob/main/data/tests/test_mask_integer.png) | `u8`, `u16`, or `u32` image where `0` indicates background and unique positive integers specifiy different objects. |
| [Polygons](https://github.com/tomouellette/thyme/blob/main/data/tests/test_polygons.json) | `(N, K, 2)` `json` with a valid key: `polygons`, `contours`, `outlines`, `shapes`, `points`. |
| [Bounding boxes](https://github.com/tomouellette/thyme/blob/main/data/tests/test_boxes.json) | `(N, [x_min, y_min, x_max, y_max])` `json` with a valid key: `bounding_boxes`, `bboxes`, `bbox`, `bounding_box`, `boxes`, `xyxy`. |

## Future support

- Image formats
    - N-channel `tiff/tif` and N-channel `zarr` images.
    - Optional compilation flag for bioformats (see [rust bindings](https://github.com/AzHicham/bioformats-rs))
- Segmentation formats
    - Additional formats for polygons or bounding boxes will be added on request (e.g. `npy`).
- No planned support
    - Three-dimensional imaging
    - Gigapixel or whole-slide histology images (currently developing other crates for this - reach out if interested)
    - Python bindings to the `thyme` library (currently developing a python/rust specific library - reach out if interested)
