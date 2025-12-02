#!/usr/bin/env -S just --justfile

[group: 'dev']
docs:
  cargo run -p pineapple-cli --bin docs --features docs > docs/docs.md

[group: 'dev']
clippy:
  cargo clippy --all --all-targets -- --deny warnings

[group: 'dev']
test:
  cargo check --workspace
  cargo test --workspace

[group: 'dev']
package:
  cargo publish --dry-run --manifest-path pineapple-core/Cargo.toml
  cargo publish --dry-run --manifest-path pineapple-data/Cargo.toml
  cargo publish --dry-run --manifest-path pineapple-neural/Cargo.toml
  cargo publish --dry-run --manifest-path pineapple-cli/Cargo.toml

[group: 'build']
publish:
  #!/bin/bash
  cargo publish --manifest-path pineapple-core/Cargo.toml
  sleep 30s
  cargo publish --manifest-path pineapple-data/Cargo.toml
  sleep 30s
  cargo publish --manifest-path pineapple-neural/Cargo.toml
  sleep 30s
  cargo publish --manifest-path pineapple-cli/Cargo.toml

[group: 'build']
tag:
    # Delete and re-tag if build fails
    git tag -d v0.0.5
    git push origin :refs/tags/v0.0.5
    git tag -a v0.0.5 -m "v0.0.5"
    git push origin v0.0.5
