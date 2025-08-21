// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use clap::{Args, Subcommand};

mod boxes;
mod mask;
mod polygons;

use boxes::{ProfileBoxesArgs, profile_image_boxes};
use mask::{ProfileMaskArgs, profile_image_mask};
use polygons::{ProfilePolygonsArgs, profile_image_polygons};

#[derive(Debug, Args)]
#[command(about = "Compute object-level morphological descriptors from image and segment pairs.")]
#[command(args_conflicts_with_subcommands = true)]
#[command(arg_required_else_help = true)]
#[command(flatten_help = true)]
pub struct ProfileArgs {
    #[command(subcommand)]
    command: Option<ProfileCommands>,
}

#[derive(Debug, Subcommand)]
enum ProfileCommands {
    Boxes(ProfileBoxesArgs),
    Mask(ProfileMaskArgs),
    Polygons(ProfilePolygonsArgs),
}

pub fn profile(args: &ProfileArgs) {
    match args.command.as_ref().unwrap() {
        ProfileCommands::Boxes(boxes) => profile_image_boxes(boxes),
        ProfileCommands::Mask(masks) => profile_image_mask(masks),
        ProfileCommands::Polygons(polygons) => profile_image_polygons(polygons),
    }
}
