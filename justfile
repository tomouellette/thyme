#!/usr/bin/env -S just --justfile

[group: 'dev']
docs:
  cargo run -p pineapple-cli --bin docs --features docs > docs/docs.md

[group: 'dev']
clippy:
  cargo clippy --all --all-targets -- --deny warnings

[group: 'build']
tag:
    # Delete and re-tag if build fails
    # git tag -d v0.0.2
    git push origin :refs/tags/v0.0.2
    git tag -a v0.0.2 -m "v0.0.2"
    git push origin v0.0.2
