// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use chrono;
use colored::*;
use kdam::{Bar, tqdm};

/// A basic progress bar for tracking iterations
pub fn progress_bar(n: usize, desc: &str, verbose: bool) -> Bar {
    if !verbose {
        return tqdm!(disable = true);
    }

    let pb = tqdm!(
        total = n,
        force_refresh = false,
        desc = progress_timestamp(desc),
        bar_format =
            "{desc suffix=' '}[{percentage:.0}%] ({rate:.1}/s, eta: {remaining human=true})"
    );
    pb
}

/// A progress bar with a standardized timestamp for tracking time
pub fn progress_timestamp(desc: &str) -> String {
    let time = chrono::Local::now();
    let ymd = time.format("%Y-%m-%dT").to_string();
    let ymd = &ymd[..ymd.len() - 1];
    let hms = time.format("%H:%M:%S").to_string();
    let time = format!("{} | {}", ymd, hms);

    format!(
        "{} {} {} {} {} {}",
        "[".bold(),
        time,
        "|".bold(),
        "thyme".truecolor(103, 194, 69).bold(),
        "]".bold(),
        desc,
    )
}

/// Print timestamped statements to console
pub fn progress_log(desc: &str, verbose: bool) {
    if !verbose {
        return;
    }

    println!("{}", progress_timestamp(desc));
}

/// Format numbers to readaable thousands format
pub fn thousands_format<T>(number: T) -> String
where
    T: std::fmt::Display,
{
    let number = number.to_string();
    if number.len() > 4 {
        number
            .as_bytes()
            .rchunks(3)
            .rev()
            .map(std::str::from_utf8)
            .collect::<Result<Vec<&str>, _>>()
            .unwrap()
            .join(",")
    } else {
        number.to_string()
    }
}
