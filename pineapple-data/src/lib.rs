// Copyright (c) 2025-2026, Tom Ouellette
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

use dirs::home_dir;

pub fn get_pineapple_cache() -> std::path::PathBuf {
    if let Ok(pineapple_cache) = std::env::var("THYME_CACHE")
        && !pineapple_cache.is_empty() {
        return std::path::PathBuf::from(pineapple_cache);
    }

    if let Some(home) = home_dir() {
        return home.join(".pineapple_cache");
    }

    std::path::PathBuf::from("/.pineapple_cache")
}

pub mod data;
pub mod request;
