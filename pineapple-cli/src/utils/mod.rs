// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use clap::{Args, Subcommand};

mod images2zarr;
mod mask2boxes;
mod mask2polygons;

use images2zarr::{Images2zarrArgs, utils_images2zarr};
use mask2boxes::{Mask2boxesArgs, utils_mask2boxes};
use mask2polygons::{Mask2polygonsArgs, utils_mask2polygons};

#[derive(Debug, Args)]
#[command(about = "General utilities for converting and transforming image/image-related data.")]
#[command(args_conflicts_with_subcommands = true)]
#[command(arg_required_else_help = true)]
#[command(flatten_help = true)]
pub struct UtilsArgs {
    #[command(subcommand)]
    command: Option<UtilsCommands>,
}

#[derive(Debug, Subcommand)]
enum UtilsCommands {
    Images2zarr(Images2zarrArgs),
    Mask2boxes(Mask2boxesArgs),
    Mask2polygons(Mask2polygonsArgs),
}

pub fn utils(args: &UtilsArgs) {
    match args.command.as_ref().unwrap() {
        UtilsCommands::Images2zarr(images2zarr_args) => utils_images2zarr(images2zarr_args),
        UtilsCommands::Mask2boxes(mask2boxes_args) => utils_mask2boxes(mask2boxes_args),
        UtilsCommands::Mask2polygons(mask2polygons_args) => utils_mask2polygons(mask2polygons_args),
    }
}
