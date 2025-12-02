// Copyright (c) 2025-2026, Tom Ouellette
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

use clap::{Args, Subcommand};

mod form;
mod intensity;
mod moments;
mod neural;
mod texture;
mod zernike;

use form::{measure_form, FormArgs};
use intensity::{measure_intensity, IntensityArgs};
use moments::{measure_moments, MomentsArgs};
use neural::{measure_neural, NeuralArgs};
use texture::{measure_texture, TextureArgs};
use zernike::{measure_zernike, ZernikeArgs};

#[derive(Debug, Args)]
#[command(about = "Measure quantitative features from bio-imaging data.")]
#[command(args_conflicts_with_subcommands = true)]
#[command(arg_required_else_help = true)]
#[command(flatten_help = true)]
pub struct MeasureArgs {
    #[command(subcommand)]
    command: Option<MeasureCommands>,
}

#[derive(Debug, Subcommand)]
enum MeasureCommands {
    Form(FormArgs),
    Intensity(IntensityArgs),
    Moments(MomentsArgs),
    Neural(NeuralArgs),
    Texture(TextureArgs),
    Zernike(ZernikeArgs),
}

pub fn measure(args: &MeasureArgs) {
    match args.command.as_ref().unwrap() {
        MeasureCommands::Form(form) => measure_form(form),
        MeasureCommands::Intensity(intensity) => measure_intensity(intensity),
        MeasureCommands::Moments(moments) => measure_moments(moments),
        MeasureCommands::Neural(neural) => measure_neural(neural),
        MeasureCommands::Texture(texture) => measure_texture(texture),
        MeasureCommands::Zernike(zernike) => measure_zernike(zernike),
    }
}
