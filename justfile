#!/usr/bin/env -S just --justfile

[group: 'dev']
docs:
  cargo run -p thyme-cli --bin docs --features docs > docs/docs.md

[group: 'dev']
clippy:
  cargo clippy --all --all-targets -- --deny warnings
