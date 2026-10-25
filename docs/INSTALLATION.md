# Installing Stark

Stark runs on an Apple silicon Mac with macOS 26.4 or newer. The native engine currently compiles Metal 4 kernels for that minimum OS. Use a native arm64 terminal; Intel Macs and Rosetta terminals are rejected.

Install the [latest signed release](https://github.com/jshahbazi/stark/releases/latest), including terminal chat and the menu bar app, without a source checkout or developer tools:

```sh
curl -fL https://github.com/jshahbazi/stark/releases/latest/download/install.sh -o stark-install.sh
sh stark-install.sh
export PATH="$HOME/.local/bin:$PATH"
stark doctor
open "$HOME/Applications/Stark.app"
```

The installer downloads and verifies the versioned archive, installs the CLI to `~/.local` and the self-contained app to `~/Applications/Stark.app`, without sudo. Add `~/.local/bin` to your shell profile to keep it on PATH. Both interfaces include their private runtimes; Rust, Xcode and uv are only needed to build Stark itself. The signed release includes Developer ID signatures, hardened runtime and secure timestamps for every runtime-loadable native component, including private Python interpreters and extensions. The app's notarization ticket is stapled and validated.

The [2026-10-21 installation validation](INSTALLATION_VALIDATION_2026_10_21.md) records 41 passing tests plus a real source install, upgrade, offline conversion, relocation and uninstall on the named build.

## Menu bar application

Download the signed, notarized `.dmg` from the [latest release](https://github.com/jshahbazi/stark/releases/latest), open it, and drag `Stark.app` to Applications. Eject the disk image, then open the installed app. The installer script above also installs the terminal tools.

The installed app provides model installation, Run/Stop controls, a small chat window, status and last-request input/output token speeds. Open `~/Applications/Stark.app`, click the cat icon, choose **Install Model**, select a supported folder or Hugging Face repository, then choose **Run** to open chat. **Stop** unloads the worker. Models download separately and remain outside the application bundle. See the [menu bar guide](../apps/menu-bar/README.md).

To rebuild only the app during development, use a configured source checkout with a standalone CLI runtime already installed:

```sh
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-generate
uv run --no-project --python 3.13.13 python scripts/build_menubar.py --install --open
```

This installs a self-contained `~/Applications/Stark.app` using a copy of the existing runtime. See the [menu bar guide](../apps/menu-bar/README.md) for model support, timing definitions, build inputs and installation details.

## Install from a source checkout

The source build requires Xcode with its Metal Toolchain, [rustup](https://rustup.rs), and [uv 0.12.24](https://github.com/astral-sh/uv/releases/tag/0.12.24). Select Xcode in **Xcode > Settings > Locations > Command Line Tools**. Command Line Tools alone do not provide the required Metal compiler.

From the Stark checkout:

```sh
./install.sh --from-source
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

A bundle contains the native scoring and generation workers, embedded Metal kernels, a private Python runtime, the converter with locked dependencies, provenance and license notices. No compiler, package manager or Python setup is needed on the receiving Mac. A local archive supplies the installation files; macOS may need a network connection to retrieve its notarization tickets on first execution.

Combined releases also contain `app/Stark.app` with an independent embedded runtime. The installer copies this app to the selected Applications folder. Earlier CLI-only archives remain supported.

Keep the archive, its adjacent `.sha256` file and the companion `install.sh` together. The builder emits all three; the receiving Mac does not need a source checkout:

```sh
sh install.sh --archive /path/to/stark-VERSION-aarch64-apple-darwin.tar.gz
```

Alternatively, provide a trusted hash explicitly:

```sh
sh install.sh --archive /path/to/stark-VERSION-aarch64-apple-darwin.tar.gz \
  --sha256 SHA256
```

Replace `VERSION` and `SHA256` with the archive's actual version and hash. Checksums detect corruption and can verify an independently obtained expected hash. An adjacent checksum has the same publisher trust as the archive. Release `v0.1.0` and direct source builds remain unsigned. The separate signing action creates signed releases and records Apple's acceptance in an attached notarization receipt. Apple issues online tickets for standalone executables but cannot staple tickets to them or to ZIP files. The installer does not remove quarantine attributes or change Gatekeeper settings. See [Apple's notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow).

For a published GitHub release tagged `vVERSION` containing the three files, download its installer, then let it download and verify the archive and checksum:

```sh
curl --proto '=https' --tlsv1.2 --fail --location \
  https://github.com/jshahbazi/stark/releases/download/vVERSION/install.sh \
  --output install.sh
sh install.sh --version VERSION
```

By default, running the installer selects the latest stable GitHub release, validates its version, then downloads that exact archive and checksum. It resolves the version once so a newly published release cannot change the selected archive halfway through installation. A missing release or checksum stops installation. Use `--version VERSION` when you want to pin a specific release; source builds require `--from-source`.

## Use Stark

Start an interactive terminal conversation with a converted model:

```sh
stark chat --model-path /absolute/path/to/converted-model
```

Enter one message per line; `/reset` clears history and `/exit` quits. Chat uses the model's chat template and keeps history in memory. See [chat options](GENERATION.md#interactive-cli-chat).

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

Apps built with the integrated updater offer **Check for Updates…** in the menu bar dropdown. Choose the offered upgrade to download and verify the latest stable signed app, then **Install and Relaunch**. This updates the complete app and preserves models; the terminal CLI keeps its existing version. Older apps need one normal installation to acquire the updater. See [app updates](../apps/menu-bar/README.md#app-updates) for verification, ownership and recovery behavior.

Quit the menu bar app, then install a newer archive with the same command, prefix and app directory. The installer verifies its checksum, extracts into a temporary directory, runs `doctor` and validates the app before activation. Combined installation failures roll back the app, command links and active CLI version. Version directories are immutable: installing different contents under an existing version is refused. Reinstalling the same archive is allowed. A running app or unrelated application at the destination stops installation.

Choose a prefix consistently for installation and updates:

```sh
./install.sh --from-source --prefix "$HOME/tools"
sh install.sh --archive /path/to/stark-VERSION-aarch64-apple-darwin.tar.gz \
  --prefix "$HOME/tools" --app-dir "$HOME/Applications"
```

Commands go into `PREFIX/bin`; bundles go into `PREFIX/lib/stark/releases/VERSION`. `--app-dir` selects the parent folder for `Stark.app`. Other commands at that prefix are preserved, and collisions with existing Stark commands are refused. Old CLI versions remain available in the release directory until uninstall. The installer records ownership of the app separately so one CLI installation cannot remove another installation's app. Models and Hugging Face caches remain at the locations you selected.

```sh
stark uninstall
```

Quit the app before uninstalling. If the launcher is broken, use `sh install.sh --uninstall --prefix /absolute/prefix`. Uninstall removes the managed app, binaries and command links, while keeping models, Application Support data and other commands. An interrupted install can leave `PREFIX/lib/stark/.install-lock`. After confirming no installer is running, remove that empty lock directory and rerun the installer.

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

The builder creates combined local archives by default; `--cli-only` creates a development bundle without the app. The [signed release workflow](RELEASING.md) builds from the dispatched protected `main` commit on a GitHub-hosted Mac, signs and notarizes the combined distribution, staples the app ticket, verifies the installation and DMG, then tags the source and publishes the latest stable release. The signing action retains the engine's original source identity and records its signing recipe separately. The inherited Uzu release, cloud and signing workflows remain disabled.

## What other engines do

Reviewed on 2026-10-21 against upstream documentation:

| Engine | Documented installation | Applicable pattern |
| --- | --- | --- |
| [Ollama](https://docs.ollama.com/macos) | Mac application download with a CLI link | Ship a complete runtime and expose commands on PATH. Stark remains a standalone engine. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md) | Homebrew, conda-forge, MacPorts and Nix packages | Treat building as a maintainer task; give consumers prebuilt executables. |
| [MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/README.md) | Python package with generation, chat and conversion entry points | Keep entry points discoverable and the conversion environment isolated. |

Stark's first distribution format is a complete Mac CLI archive because its kernels are embedded and its workers link only Apple system libraries. Bundling the converter's Python runtime removes an additional setup step. A package-manager formula can consume the same versioned archive when public releases exist.
