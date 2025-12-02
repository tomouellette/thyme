// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

// All currently supported image formats
pub const SUPPORTED_IMAGE_FORMATS: [&str; 18] = [
    "avif", "bmp", "dds", "hdr", "ico", "jpeg", "jpg", "exr", "png", "pbm", "pgm", "ppm", "qoi",
    "tga", "tif", "tiff", "webp", "npy",
];

// All currently supported array formats
pub const SUPPORTED_ARRAY_FORMATS: [&str; 1] = ["json"];

// The currently supported common image formats
pub const IMAGE_DYNAMIC_FORMATS: [&str; 17] = [
    "avif", "bmp", "dds", "hdr", "ico", "jpeg", "jpg", "exr", "png", "pbm", "pgm", "ppm", "qoi",
    "tga", "tif", "tiff", "webp",
];

// The valid json keys indicating bounding box values
pub const BOUNDING_BOX_JSON_VALID_KEYS: [&str; 7] = [
    "bounding_boxes",
    "bboxes",
    "bbox",
    "bounding_box",
    "boxes",
    "box",
    "xyxy",
];

// The valid json keys indicating polygon values
pub const POLYGON_JSON_VALID_KEYS: [&str; 5] =
    ["polygons", "contours", "outlines", "shapes", "points"];

// Factorial constants used currently in zernike descriptor calculations
pub const FACTORIAL: [f32; 10] = [
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
];

// Hard default settings for the gray-level co-occurence matrix calculations
pub const GLCM_LEVELS: usize = 64;
pub const GLCM_ARRAY_SIZE: usize = GLCM_LEVELS * GLCM_LEVELS;

// Names for morphological descriptors
pub const FORM_DESCRIPTOR_NAMES: [&str; 23] = [
    "form_centroid_x",
    "form_centroid_y",
    "form_center_x",
    "form_center_y",
    "form_area",
    "form_area_bbox",
    "form_area_convex",
    "form_perimeter",
    "form_elongation",
    "form_thread_length",
    "form_thread_width",
    "form_solidity",
    "form_extent",
    "form_form_factor",
    "form_equivalent_diameter",
    "form_eccentricity",
    "form_major_axis",
    "form_minor_axis",
    "form_minimum_radius",
    "form_maximum_radius",
    "form_mean_radius",
    "form_min_feret",
    "form_max_feret",
];

pub const INTENSITY_DESCRIPTOR_NAMES: [&str; 7] = [
    "intensity_min",
    "intensity_max",
    "intensity_sum",
    "intensity_mean",
    "intensity_std",
    "intensity_median",
    "intensity_mad",
];

pub const MOMENTS_DESCRIPTOR_NAMES: [&str; 24] = [
    "moments_m00",
    "moments_m10",
    "moments_m01",
    "moments_m11",
    "moments_m20",
    "moments_m02",
    "moments_m21",
    "moments_m12",
    "moments_m30",
    "moments_m03",
    "moments_u11",
    "moments_u20",
    "moments_u02",
    "moments_u21",
    "moments_u12",
    "moments_u30",
    "moments_u03",
    "moments_i1",
    "moments_i2",
    "moments_i3",
    "moments_i4",
    "moments_i5",
    "moments_i6",
    "moments_i7",
];

pub const TEXTURE_DESCRIPTOR_NAMES: [&str; 13] = [
    "texture_energy",
    "texture_contrast",
    "texture_correlation",
    "texture_sum_of_squares",
    "texture_inverse_difference_moment",
    "texture_sum_average",
    "texture_sum_variance",
    "texture_sum_entropy",
    "texture_entropy",
    "texture_difference_variance",
    "texture_difference_entropy",
    "texture_infocorr1",
    "texture_infocorr2",
];

pub const ZERNIKE_DESCRIPTOR_NAMES: [&str; 30] = [
    "zernike_00",
    "zernike_11",
    "zernike_20",
    "zernike_22",
    "zernike_31",
    "zernike_33",
    "zernike_40",
    "zernike_42",
    "zernike_44",
    "zernike_51",
    "zernike_53",
    "zernike_55",
    "zernike_60",
    "zernike_62",
    "zernike_64",
    "zernike_66",
    "zernike_71",
    "zernike_73",
    "zernike_75",
    "zernike_77",
    "zernike_80",
    "zernike_82",
    "zernike_84",
    "zernike_86",
    "zernike_88",
    "zernike_91",
    "zernike_93",
    "zernike_95",
    "zernike_97",
    "zernike_99",
];
