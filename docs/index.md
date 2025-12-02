# Get started

<img style="color: gray; float: left; width: 5em; margin-left: 0em; margin-right: 2em;" src="assets/logo.svg#only-light"></img>
<img style="color: gray; float: left; width: 5em; margin-left: 0em; margin-right: 2em;" src="assets/logo_dark.svg#only-dark"></img>

`pineapple` is a command-line tool for processing and profiling morphological data in bio-imaging datasets.
`pineapple` is written in the Rust programming language and can be used for extracting segmented objects, computing morphological descriptors, generating self-supervised embeddings of cells and nuclei, and more.

## Installation

#### Cargo

`pineapple` can be installed using the [rust](https://www.rust-lang.org/) package manager [cargo](https://github.com/rust-lang/cargoinstall):

```bash
cargo install pineapple 
```

#### Pre-compiled binaries

Pre-built binaries for x86-64 linux, x86-64 apple, aarch-64 apple, and x86-64 windows are available for download in GitHub [releases](https://github.com/tomouellette/pineapple/releases). Note that the pre-built binaries are untested on windows.

## License

`pineapple` is licensed under the `BSD 3-Clause` license (see [LICENSE](https://github.com/tomouellette/pineapple/blob/main/LICENSE.txt)).
