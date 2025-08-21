// Copyright (c) 2025, Tom Ouellette
// Licensed under the MIT License

use anyhow::{Context, Result, anyhow};
use kdam::BarExt;
use reqwest::{Client, redirect::Policy};
use scraper::{Html, Selector};
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

use thyme_core::ut::track::{progress_bar, progress_log};

/// Download a file from Google drive
///
/// # Arguments
///
/// * `file_id` - Unique google drive file identifier
/// * `output_dir` - Directory to download file to
/// * `filename` - Filename of downloaded file
/// * `silent` - Turn off download messages
#[tokio::main]
pub async fn download_file(
    file_id: &str,
    output_dir: &Path,
    filename: &str,
    silent: bool,
) -> Result<()> {
    let client = create_http_client()?;
    let initial_url = format!("https://drive.google.com/uc?id={}&export=download", file_id);
    let download_url = handle_virus_scan_warning(&client, &initial_url).await?;
    download_file_with_progress(&client, &download_url, output_dir, filename, silent).await?;
    if !silent {
        println!();
    }
    progress_log("Complete", !silent);
    Ok(())
}

fn create_http_client() -> Result<Client> {
    Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        .cookie_store(true)
        .redirect(Policy::limited(10))
        .build()
        .context("Failed to create HTTP client")
}

/// Allows for passing Google drive virus scan warning
async fn handle_virus_scan_warning(client: &Client, url: &str) -> Result<String> {
    let resp = client
        .get(url)
        .send()
        .await
        .context("Failed to send initial request")?;

    let body = resp.text().await.context("Failed to get response body")?;

    if body.contains("Google Drive can't scan this file for viruses") {
        extract_download_url(&body)
    } else {
        Ok(url.to_string())
    }
}

fn extract_download_url(body: &str) -> Result<String> {
    let document = Html::parse_document(body);
    let form_selector = Selector::parse("form#download-form").unwrap();
    let input_selector = Selector::parse("input[name]").unwrap();

    let form = document
        .select(&form_selector)
        .next()
        .ok_or_else(|| anyhow!("Download form not found in the HTML"))?;

    let action = form
        .value()
        .attr("action")
        .ok_or_else(|| anyhow!("No form action found"))?;

    let params: Vec<(String, String)> = form
        .select(&input_selector)
        .filter_map(
            |input| match (input.value().attr("name"), input.value().attr("value")) {
                (Some(name), Some(value)) => Some((name.to_string(), value.to_string())),
                _ => None,
            },
        )
        .collect();

    let query_string = params
        .into_iter()
        .map(|(name, value)| format!("{}={}", name, value))
        .collect::<Vec<_>>()
        .join("&");

    Ok(format!("{}?{}", action, query_string))
}

async fn download_file_with_progress(
    client: &Client,
    url: &str,
    output_dir: &Path,
    filename: &str,
    silent: bool,
) -> Result<()> {
    let mut resp = client
        .get(url)
        .send()
        .await
        .context("Failed to send download request")?;

    let total_size = resp.content_length().unwrap_or(0);
    let mut pb = progress_bar(
        total_size as usize,
        format!("Downloading {}", filename).as_str(),
        !silent,
    );

    let total_gigabytes = total_size as f64 / 1e9;

    if !silent {
        progress_log(
            format!("Starting {} download ({:.2} GB)", filename, total_gigabytes).as_str(),
            !silent,
        );
    }

    if total_gigabytes == 0.0 {
        println!("Download could not be started. Please check connection and try again.");
        std::process::exit(1);
    }

    tokio::fs::create_dir_all(output_dir)
        .await
        .context("Failed to create output directory")?;
    let filepath = output_dir.join(filename);
    let mut file = File::create(&filepath)
        .await
        .context("Failed to create output file")?;

    while let Some(chunk) = resp.chunk().await.context("Failed to read chunk")? {
        file.write_all(&chunk)
            .await
            .context("Failed to write chunk to file")?;

        if !silent {
            pb.update(chunk.len())?;
        }
    }

    Ok(())
}
