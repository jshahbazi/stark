# Installing Stark

Stark runs on an Apple silicon Mac with macOS 26.4 or newer. The native engine currently compiles Metal 4 kernels for that minimum OS. Use a native arm64 terminal; Intel Macs and Rosetta terminals are rejected.

Install the [v0.1.0 prebuilt release](https://github.com/jshahbazi/stark/releases/tag/v0.1.0) without a source checkout or developer tools:

```sh
curl -fL https://github.com/jshahbazi/stark/releases/download/v0.1.0/install.sh -o stark-install.sh
sh stark-install.sh --version 0.1.0
export PATH="$HOME/.local/bin:$PATH"
stark doctor
```

The installer downloads and verifies the versioned archive and installs to `~/.local`, without sudo. Add `~/.local/bin` to your shell profile to keep it on PATH. The runtime includes Python and the converter; Rust, Xcode and uv are only needed to build Stark itself.

The [2026-10-21 installation validation](INSTALLATION_VALIDATION_2026_10_21.md) records 41 passing tests plus a real source install, upgrade, offline conversion, relocation and uninstall on the named build.

## Install from a source checkout

The source build requires Xcode with its Metal Toolchain, [rustup](https://rustup.rs), and [uv 0.12.24](https://github.com/astral-sh/uv/releases/tag/0.12.24). Select Xcode in **Xcode > Settings > Locations > Command Line Tools**. Command Line Tools alone do not provide the required Metal compiler.

From the Stark checkout:

```sh
./install.sh
```

This installs Rust 1.98.1 through your existing rustup, builds the workers with locked dependencies, downloads Python 3.13.13 and the locked converter dependencies, and creates a standalone installation in `~/.local`. It does not require an administrator password. The build can take several minutes. Each source build gets a version containing its source identity and UTC build time. The resulting installation runs independently of the source checkout, Rust, uv, Xcode and your system Python.

If `~/.local/bin` is absent from your shell's PATH, add this line to `~/.zprofile`, then open a new terminal:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

Check the installation:

```sh
stark --version
stark doctor
```

`doctor` checks the platform, isolated converter imports and both worker executables. It does not load a model or qualify inference on your GPU.

## Install a prebuilt archive

A bundle contains the native scoring and generation workers, embedded Metal kernels, a private Python runtime, the converter with locked dependencies, provenance and license notices. No compiler, package manager or Python setup is needed on the receiving Mac. Installing a local archive requires no network access.

Keep the archive, its adjacent `.sha256` file and the companion `install.sh` together. The builder emits all three; the receiving Mac does not need a source checkout:

```sh
sh install.sh --archive /path/to/stark-VERSION-aarch64-apple-darwin.tar.gz
```

Alternatively, provide a trusted hash explicitly:

```sh
sh install.sh --archive /path/to/stark-VERSION-aarch64-apple-darwin.tar.gz \
  --sha256 SHA256
```

Replace `VERSION` and `SHA256` with the archive's actual version and hash. Checksums detect corruption and can verify an independently obtained expected hash. An adjacent checksum has the same publisher trust as the archive. Archives are not Developer ID signed or notarized by this tooling. If macOS blocks a downloaded executable, use the normal macOS approval flow only after verifying its source; the installer does not remove quarantine attributes or change Gatekeeper settings.

For a published GitHub release tagged `vVERSION` containing the three files, download its installer, then let it download and verify the archive and checksum:

```sh
curl --proto '=https' --tlsv1.2 --fail --location \
  https://github.com/jshahbazi/stark/releases/download/vVERSION/install.sh \
  --output install.sh
sh install.sh --version VERSION
```

The version is explicit. A missing release or checksum stops the installation with an error; the installer never substitutes another build.

## Use Stark

Convert a supported Hugging Face model:

```sh
stark convert Qwen/Qwen3-0.6B --output models/qwen3
```

Convert a complete local checkpoint:

```sh
stark convert /path/to/downloaded-model --metadata-profile hf-generic \
  --offline --output models/local-model
```

Use a new output directory each time. Gated downloads use your existing Hugging Face credentials. See the [converter guide](../tools/converter/README.md) for model coverage, source pinning and qualification limits.

Start a worker for an application that implements its private JSON-lines protocol:

```sh
stark score --score-model-path /absolute/path/to/converted-model
stark generate --model-path /absolute/path/to/converted-model
```

These launch the existing [scoring](SCORING.md) and [generation](GENERATION.md) workers. The application supplies requests over private pipes. Host integration stays in a separate repository. The aliases `stark-convert`, `stark-score` and `stark-generate` are also installed.

## Update, choose a location, or uninstall

Install a newer archive with the same command and prefix. The installer verifies its checksum, extracts into a temporary directory and runs `doctor` before switching the `current` link. A failed self-check keeps the previous version active. Version directories are immutable: installing different contents under an existing version is refused. Reinstalling the same archive is allowed.

Choose a prefix consistently for installation and updates:

```sh
./install.sh --prefix "$HOME/tools"
sh install.sh --archive /path/to/stark-VERSION-aarch64-apple-darwin.tar.gz \
  --prefix "$HOME/tools"
```

Commands go into `PREFIX/bin`; bundles go into `PREFIX/lib/stark/releases/VERSION`. Other commands at that prefix are preserved, and collisions with existing Stark commands are refused. Old versions remain available in the release directory until uninstall. Models and Hugging Face caches remain at the locations you selected.

```sh
stark uninstall
```

If the launcher is broken, use `sh install.sh --uninstall --prefix /absolute/prefix`. Uninstall removes the managed binaries and command links, while keeping models and other commands. An interrupted install can leave `PREFIX/lib/stark/.install-lock`. After confirming no installer is running, remove that empty lock directory and rerun the installer.

## Build a distributable archive

On a configured build Mac, commit the intended source and use:

```sh
uv run --no-project --python 3.13.13 python scripts/build_distribution.py \
  --version VERSION
```

Output stays in ignored `artifacts/distributions/`:

- `stark-VERSION-aarch64-apple-darwin.tar.gz`
- `stark-VERSION-aarch64-apple-darwin.tar.gz.sha256`
- `install.sh`

The builder uses separate Cargo invocations for the optimized scorer and default generation worker, avoiding feature unification. It checks arm64 architecture, system-only native dynamic libraries, fixed model-free worker responses, relocated Python imports and converter entry points. Its manifest records source identity, lockfiles, build configuration and file hashes. Native and Python notices travel with the archive. `--allow-dirty` creates an explicitly identified development bundle; it does not extend existing source-bound inference validation to the new build.

Independent builds are not promised to produce identical archive bytes. uv embeds its managed installation path in Python's dynamic-library identity, and installed package metadata can vary. Each fresh build needs its own version. The archive hash and complete manifest identify the exact artifact that was checked; a published version must retain those exact bytes.

The builder creates local archives. The v0.1.0 assets are published through a manual GitHub release. The inherited Uzu release, cloud and signing workflows remain disabled.

## What other engines do

Reviewed on 2026-10-21 against upstream documentation:

| Engine | Documented installation | Applicable pattern |
| --- | --- | --- |
| [Ollama](https://docs.ollama.com/macos) | Mac application download with a CLI link | Ship a complete runtime and expose commands on PATH. Stark remains a standalone engine. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md) | Homebrew, conda-forge, MacPorts and Nix packages | Treat building as a maintainer task; give consumers prebuilt executables. |
| [MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/README.md) | Python package with generation, chat and conversion entry points | Keep entry points discoverable and the conversion environment isolated. |

Stark's first distribution format is a complete Mac CLI archive because its kernels are embedded and its workers link only Apple system libraries. Bundling the converter's Python runtime removes an additional setup step. A package-manager formula can consume the same versioned archive when public releases exist.
