# Installing Stark

Stark runs on an Apple silicon Mac with macOS 26.4 or newer; the engine compiles Metal 4 kernels for that minimum. Use a native arm64 terminal. Intel Macs and Rosetta terminals are rejected.

There are three ways to install, each verifying something different; see [what each route verifies](#what-each-install-route-verifies).

## Install the menu bar app

Download the signed, notarized `.dmg` from the [latest release](https://github.com/jshahbazi/stark/releases/latest), open it, drag `Stark.app` to Applications, eject the image and open the app. Click the cat icon, choose **Install Model**, select a supported folder or Hugging Face repository, then **Run**. See the [menu bar guide](../apps/menu-bar/README.md).

## Install the app and terminal tools

```sh
curl -fL https://github.com/jshahbazi/stark/releases/latest/download/install.sh -o stark-install.sh
sh stark-install.sh
export PATH="$HOME/.local/bin:$PATH"
stark doctor
open "$HOME/Applications/Stark.app"
```

The script resolves the latest stable release, downloads its versioned archive and checksum, installs the CLI under `~/.local` and the app at `~/Applications/Stark.app`, and needs no sudo. Add the `PATH` line to `~/.zprofile` to keep `stark` available in new terminals. `stark doctor` should print `Installation checks passed. No model was loaded.` Both the CLI and the app embed private Python runtimes; Rust, Xcode and uv are needed only to build Stark.

To pin a release instead of taking the latest:

```sh
curl -fL https://github.com/jshahbazi/stark/releases/download/vVERSION/install.sh -o stark-install.sh
sh stark-install.sh --version VERSION
```

## What each install route verifies

| Route | Verification |
| --- | --- |
| DMG drag-and-drop | The browser download carries a quarantine attribute, so macOS Gatekeeper checks the Developer ID signature and the stapled notarization ticket before the app first opens. |
| `install.sh` | Downloads the archive and its `.sha256` from the same GitHub release over TLS and refuses a mismatch. This detects corruption and a wrong file; it does not detect substitution by someone able to publish release assets, because the checksum and archive share an origin. Files written by `curl` carry no quarantine attribute and the script does not check code signatures. Pass `--sha256 HASH` to check against a hash you obtained elsewhere. |
| **Check for Updates** in the app | Requires the exact GitHub asset digest, a Developer ID signature from Stark's team, a passing Gatekeeper assessment and a bundle version equal to the release tag, re-checked immediately before the swap. See [app updates](../apps/menu-bar/README.md#app-updates). |

Release archives are signed with Developer ID, hardened runtime and secure timestamps for every native image, and the app's notarization ticket is stapled; `install.sh` just does not check them. The script never removes quarantine attributes or changes Gatekeeper settings.

## Install from a source checkout

Install Xcode with its Metal Toolchain, [rustup](https://rustup.rs) and [uv 0.12.24](https://github.com/astral-sh/uv/releases/tag/0.12.24). Select Xcode under **Xcode > Settings > Locations > Command Line Tools**; the Command Line Tools package alone has no Metal compiler. Then, from the checkout:

```sh
./install.sh --from-source
```

This installs Rust 1.98.1 through rustup, builds both workers with locked dependencies, downloads Python 3.13.13 and the locked converter dependencies, and creates a standalone installation under `~/.local` that no longer depends on the checkout, Rust, uv, Xcode or the system Python. The build takes several minutes. Each source build gets a version containing its source identity and build time. Check it with:

```sh
stark --version
stark doctor
```

`doctor` checks the platform, the bundled converter and both worker executables without loading a model.

To rebuild only the app on top of an existing CLI installation:

```sh
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-generate
uv run --no-project --python 3.13.13 python scripts/build_menubar.py --install --open
```

## Install a prebuilt archive

A bundle contains both workers with embedded Metal kernels, a private Python runtime, the converter with locked dependencies, provenance and license notices. Combined bundles also contain `app/Stark.app`. Keep the archive, its `.sha256` file and the companion `install.sh` together:

```sh
sh install.sh --archive /path/to/stark-VERSION-aarch64-apple-darwin.tar.gz
```

Or supply the expected hash explicitly:

```sh
sh install.sh --archive /path/to/stark-VERSION-aarch64-apple-darwin.tar.gz --sha256 HASH
```

macOS may need a network connection to fetch notarization tickets for standalone executables the first time they run.

## Use Stark

```sh
stark chat --model-path /absolute/path/to/converted-model
stark convert Qwen/Qwen3-0.6B --output models/qwen3
stark convert /path/to/downloaded-model --metadata-profile hf-generic --offline --output models/local-model
stark generate --model-path /absolute/path/to/converted-model
stark score --score-model-path /absolute/path/to/converted-model
```

`chat` is described in [text generation](GENERATION.md#interactive-cli-chat), `convert` in the [converter guide](../tools/converter/README.md), and the two workers in [worker protocols](WORKER_PROTOCOLS.md). `stark help` lists every command; `stark-convert`, `stark-score` and `stark-generate` are installed as aliases.

## Update

Apps with the integrated updater offer **Check for Updates** in the menu. Choosing the offered version downloads, verifies and stages the new app, then **Install and Relaunch** swaps it in. This updates only the app and preserves models and preferences; the CLI keeps its version.

To update the CLI and app together, quit Stark and run the installer again with the same prefix and app directory. The installer verifies the checksum, extracts to a temporary directory, runs `doctor` and validates the app before activating anything. A failed combined installation rolls back the app, command links and active CLI version. Version directories are immutable: installing different bytes under an existing version is refused, and reinstalling the same archive is allowed. A running app or an unrelated application at the destination stops the installation.

## Choose a location

```sh
./install.sh --from-source --prefix "$HOME/tools"
sh install.sh --archive /path/to/stark-VERSION-aarch64-apple-darwin.tar.gz \
  --prefix "$HOME/tools" --app-dir "$HOME/Applications"
```

Commands go in `PREFIX/bin`; bundles in `PREFIX/lib/stark/releases/VERSION`; `--app-dir` selects the parent folder for `Stark.app`. Use the same prefix for later updates. Other commands at the prefix are preserved, and collisions with existing Stark commands are refused. The installer records app ownership separately so one CLI installation cannot remove another installation's app. Models and Hugging Face caches stay where you put them.

## Uninstall

```sh
stark uninstall
```

Quit the app first. If the launcher is broken, use `sh install.sh --uninstall --prefix /absolute/prefix`. Uninstall removes the managed app, binaries and command links and keeps models, Application Support data and unrelated commands.

Two lock directories guard installations. An interrupted CLI install can leave `PREFIX/lib/stark/.install-lock`; an interrupted app install or update can leave `.Stark.app.install-lock` next to the app. After confirming no installer is running, remove the empty lock directory and retry.

## Build a distributable archive

On a configured build Mac, commit the intended source and run:

```sh
uv run --no-project --python 3.13.13 python scripts/build_distribution.py --version VERSION
```

Output goes to the ignored `artifacts/distributions/` directory: the archive, its `.sha256` and `install.sh`. The builder compiles the scorer with `--features optimized` and the generator without it in separate Cargo invocations, checks arm64 architecture, system-only dynamic libraries, model-free worker replies, relocated Python imports and converter entry points, and writes a manifest of source identity, lockfiles, configuration and file hashes. `--allow-dirty` builds an explicitly marked development bundle; `--cli-only` omits the app. Independent builds are not byte-identical, so every build needs its own version. Signing, notarization and publication are described in [releasing](RELEASING.md).
