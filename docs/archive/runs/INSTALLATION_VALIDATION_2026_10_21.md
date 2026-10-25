# Standalone installation validation — 2026-10-21

The installer, CLI and standalone bundle passed model-free checks on one Apple silicon Mac running macOS 26.7.1. No model was loaded and no native inference calls were made. This packet qualifies the installation path on the named host and artifact; existing inference and model-conversion qualification records retain their original source boundaries.

## Exact build

| Field | Value |
| --- | --- |
| Source revision | `14ffff1414ac0320dcd9e1e0b111a0d6bc5c2cf0` |
| Tracked source state | Clean |
| Bundle version | `0.1.0-dev.14ffff1414ac.6b9b4111869a.20261021184753` |
| Target and minimum OS | `aarch64-apple-darwin`, macOS 26.4 |
| Tooling | Rust 1.98.1, uv 0.12.24, Python 3.13.13 |
| Native configuration | Optimized scorer; separate default Metal generation build |
| Converter | `stark-convert` 0.9.1, dependencies installed with lockfile hashes |
| Archive size | 90,562,522 bytes |
| Manifest entries | 3,566 files |
| Archive SHA-256 | `f9aa75da437aec3b3fdeafbee6a6e77b2fcc350ffbdd32351430976bc750a149` |
| Manifest SHA-256 | `dc5312356ed1c553b328d3be432d6a9e2dd90456fc413d71a456a378afcd0ea8` |
| Source file manifest SHA-256 | `5e0b41a56d1538a3ee2acaea15c417988ada87de764c49631b0903b04d0682ba` |
| Scorer SHA-256 | `b364200e1a1f7dc657c51a63ef6efd60642749816720e8cf5eb345511d6a92e9` |
| Generation worker SHA-256 | `d28aee32a56a49f0386c2d61831975c6a179fc93eaf262b77c49dbd727a44c0d` |

The archive, checksum, companion installer and private integration helper remain under ignored `artifacts/`. No binaries, runtime files or model data are committed. This change publishes no GitHub release or package.

## Checks and results

| Check | Result |
| --- | --- |
| Installer, launcher and builder tests | 41 passed |
| Shell syntax, Ruff lint/format, repository policy and diff whitespace | Passed |
| Real `./install.sh --prefix ...` | Passed twice from the same clean source; each build received a distinct UTC version |
| Upgrade under a prefix containing spaces | Passed; the second source build became current and the previous version remained available |
| Companion installer and exact-archive reinstall | Passed without rebuilding or downloading dependencies |
| Native build inspection | Both workers arm64, Mach-O minimum macOS 26.4, system-only dynamic dependencies |
| Standalone doctor | Passed with `PATH=/usr/bin:/bin` and deliberately invalid developer `PYTHONHOME`/`PYTHONPATH` |
| Converter command names | Help/version expose `stark-convert`; aliases and argument forwarding passed |
| Offline conversion | A tiny synthetic sharded checkpoint converted outside the checkout through the installed CLI |
| Installed file integrity | Every manifest entry matched its SHA-256 after the checks |
| Prefix relocation | Moving the complete installation preserved doctor and conversion entry points |
| Uninstall | The real installed launcher removed managed commands/runtime and kept a synthetic model-directory marker |
| Native protocol | Both worker subcommands returned only their expected fixed invalid-argument reply; no model initialization |
| License collection | 142 reachable Cargo dependency notices, seven explicitly attributed supplemental notice mappings, Rust standard-library notices and bundled Python notices |

The seven private integration checks passed as a group. Synthetic unit tests separately cover unsupported OS/architecture, command collisions, corrupt or missing checksums, unavailable releases, unsafe archive paths/links, immutable-version collisions, failed-upgrade preservation and unrelated-file preservation.

## Reproduction

Use the source revision above and the prerequisites in [the installation guide](../../INSTALLATION.md):

```sh
uv run --no-project --python 3.13.13 python -m unittest scripts/test_install.py scripts/test_stark_cli.py scripts/test_build_distribution.py
./install.sh --prefix "$PWD/artifacts/install verification"
"$PWD/artifacts/install verification/bin/stark" doctor
```

The source installer creates a new build version and archive each time. Independent builds can differ in Python's managed dynamic-library identity/signature and package installation metadata. Archive/container normalization does not establish complete build reproducibility. Preserve a published version's exact archive bytes and checksum.

Installation checks do not qualify another Mac, GPU, model, numerical output or performance result. Developer ID signing/notarization and public release hosting remain separate operations. Inherited release, signing and cloud workflows remain disabled.
