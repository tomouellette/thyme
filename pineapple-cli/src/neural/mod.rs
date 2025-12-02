// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use clap::{Args, Subcommand};

mod boxes;
mod mask;
mod polygons;

use boxes::{NeuralBoxesArgs, neural_image_boxes};
use mask::{NeuralMaskArgs, neural_image_mask};
use polygons::{NeuralPolygonsArgs, neural_image_polygons};

#[derive(Debug, Args)]
#[command(about = "Compute object-level self-supervised features from image and segment pairs.")]
#[command(args_conflicts_with_subcommands = true)]
#[command(arg_required_else_help = true)]
#[command(flatten_help = true)]
pub struct NeuralArgs {
    #[command(subcommand)]
    command: Option<NeuralCommands>,
}

#[derive(Debug, Subcommand)]
enum NeuralCommands {
    Boxes(NeuralBoxesArgs),
    Mask(NeuralMaskArgs),
    Polygons(NeuralPolygonsArgs),
}

pub fn neural(args: &NeuralArgs) {
    match args.command.as_ref().unwrap() {
        NeuralCommands::Boxes(boxes) => neural_image_boxes(boxes),
        NeuralCommands::Mask(masks) => neural_image_mask(masks),
        NeuralCommands::Polygons(polygons) => neural_image_polygons(polygons),
    }
}
