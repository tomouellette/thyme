// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use clap::{Args, Subcommand};

mod boxes;
mod mask;
mod polygons;

use boxes::{ProcessBoxesArgs, process_image_boxes};
use mask::{ProcessMaskArgs, process_image_mask};
use polygons::{ProcessPolygonsArgs, process_image_polygons};

#[derive(Debug, Args)]
#[command(about = "Extract object-level data from image and segment pairs.")]
#[command(args_conflicts_with_subcommands = true)]
#[command(arg_required_else_help = true)]
#[command(flatten_help = true)]
pub struct ProcessArgs {
    #[command(subcommand)]
    command: Option<ProcessCommands>,
}

#[derive(Debug, Subcommand)]
enum ProcessCommands {
    Boxes(ProcessBoxesArgs),
    Mask(ProcessMaskArgs),
    Polygons(ProcessPolygonsArgs),
}

pub fn process(args: &ProcessArgs) {
    match args.command.as_ref().unwrap() {
        ProcessCommands::Boxes(boxes) => process_image_boxes(boxes),
        ProcessCommands::Mask(masks) => process_image_mask(masks),
        ProcessCommands::Polygons(polygons) => process_image_polygons(polygons),
    }
}
