# Publishing a standalone Mac release

The `Signed release` workflow signs and notarizes a pinned prebuilt bundle from protected `main`. The separate `Prebuilt release` workflow constructs unsigned inputs. Both create drafts for receiving-Mac verification before publication. Neither runs the inherited Uzu release, cloud or signing workflows.

## Sign the existing prebuilt engine

The first signed distribution is `v0.1.1`. Its input is the immutable `v0.1.0` archive with SHA-256 `20f43b0f62fbe58dd026241031856a5f901067a88ed0f9cf27f2c3ea67d9e05b`, built from engine source `8c6f78c473c12fb9f86474d5c1aa8de4fb2d4b7e`. The signing workflow pins both identities in reviewed source. Repackaging changes signatures, distribution version and manifest hashes; it does not rebuild the engine or replace the original archive.

After merging the signing recipe with successful required checks, tag that exact commit and dispatch from `main`:

```sh
git switch main
git pull --ff-only
git tag -a v0.1.1 -m 'Stark 0.1.1 signed and notarized Mac release'
git push origin v0.1.1
gh workflow run signed-release.yml --ref main -f version=0.1.1
```

The tag must identify the workflow's exact source commit. A new input engine requires updating the pinned input archive hash and engine revision in a reviewed workflow change. Never move a published tag or replace published archive bytes.

The `macos-release` environment is restricted to protected branches. It contains these six secrets:

- `MACOS_CERTIFICATE_P12_BASE64`
- `MACOS_CERTIFICATE_PASSWORD`
- `MACOS_SIGNING_IDENTITY`
- `APPLE_API_KEY_P8_BASE64`
- `APPLE_API_KEY_ID`
- `APPLE_API_ISSUER_ID`

Keep the certificate export, API key and passwords outside Git and command output. The action imports them into a temporary Keychain, removes the private files, and unconditionally restores the previous Keychain search list and deletes the temporary Keychain. Only the import step receives all six secrets. Signing receives the public certificate identity and accesses the temporary Keychain profile. A separate publishing job receives repository write access and has no Apple credentials.

Before importing credentials, the action verifies the pinned input's checksum, source, exact file inventory and standalone installation. It then signs every runtime Mach-O image individually, including the workers, interpreter, libpython, native Python extensions and bundled dynamic libraries. It requires the exact Developer ID Application certificate, hardened runtime, secure timestamps and no extra entitlements. Relocatable object files and static libraries are build support, not runtime code-signing targets; skipped Mach-O object paths are recorded explicitly.

After signing, the finalizer refreshes the file manifest and creates a ZIP of the exact signed tree for Apple. It requires notarization status `Accepted` and online ticket checks for both workers and the interpreter. Signing or notarization failure publishes no release outputs. A source-bound aggregate receipt records the engine revision, signing recipe revision and hashes, signer certificate and Team ID, native image count, Apple submission identifier and exact archive hashes. It contains no credentials or runtime payloads.

The signed draft has seven assets: the signed TAR archive and checksum, notarized ZIP and checksum, aggregate notarization receipt, installation verification JSON and `install.sh`. Download them and verify the actual archive on a receiving Mac:

```sh
gh release download v0.1.1 --dir artifacts/release-0.1.1
uv run --project tools/converter --locked --python 3.13.13 python scripts/verify_distribution.py \
  --archive artifacts/release-0.1.1/stark-0.1.1-aarch64-apple-darwin.tar.gz \
  --version 0.1.1 --expected-source 8c6f78c473c12fb9f86474d5c1aa8de4fb2d4b7e \
  --expected-distribution-source SIGNING_RECIPE_COMMIT \
  --expected-signer EFE6CD2666B9106E68724C38ADD1C405267A2F94 --require-notarized \
  --output artifacts/release-0.1.1/receiving-mac-verification.json
```

Replace `SIGNING_RECIPE_COMMIT` with the full tagged commit. This adds checks of all native signatures and executable notarization tickets to the normal installation, conversion, hashes, relocation and uninstall checks. After the actual draft passes:

```sh
gh release edit v0.1.1 --draft=false --latest
```

Test the public installer download and `--version 0.1.1` route from outside a checkout after publication. Preserve the earlier validation packets; record this run separately.

Apple cannot staple tickets to standalone command-line binaries or ZIP files. Gatekeeper retrieves their tickets online. The distributed TAR contains the same signed native bytes submitted in the ZIP. See [Apple's notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow). The installer never removes quarantine or changes Gatekeeper policy.

## Build an unsigned input from source

Merge the intended changes with successful required CI and Security checks. Choose a new version, then create its tag on the exact merged source. Do not move a published version tag or replace its archive bytes.

For the first release:

```sh
git switch main
git pull --ff-only
git tag -a v0.1.0 -m 'Stark 0.1.0 prebuilt Mac release'
git push origin v0.1.0
gh workflow run prebuilt-release.yml --ref v0.1.0 -f version=0.1.0
```

The workflow checks the tag, source commit and ancestry before building. It uses Rust 1.98.1, uv 0.12.24 and Python 3.13.13 on `macos-26`, with locked dependencies. Build tools and package managers stay outside the consumer runtime; pip and ensurepip are removed from the copied Python runtime after construction.

The build runs without a publishing token in its process environment. The final upload step receives the repository token and creates a draft release. The job's `contents: write` permission is required for that upload; checkout retains no credential. Existing releases are never overwritten by the workflow.

## Verify an unsigned input draft

The workflow's archive verifier checks checksum installation under a temporary prefix containing spaces, isolated doctor and CLI identity, absence of bootstrap installers, fixed model-free worker replies, offline synthetic conversion, every manifest hash, exact-archive reinstall, prefix relocation and uninstall with model-directory preservation. It prints aggregate results and uploads only these three assets to the draft:

- `stark-VERSION-aarch64-apple-darwin.tar.gz`
- `stark-VERSION-aarch64-apple-darwin.tar.gz.sha256`
- `install.sh`

Download and verify the actual draft artifacts on a receiving Mac before publication:

```sh
gh release download v0.1.0 --dir artifacts/release-0.1.0
uv run --project tools/converter --locked --python 3.13.13 python scripts/verify_distribution.py \
  --archive artifacts/release-0.1.0/stark-0.1.0-aarch64-apple-darwin.tar.gz \
  --version 0.1.0 --expected-source SOURCE_COMMIT \
  --output artifacts/release-0.1.0/verification.json
```

Replace `SOURCE_COMMIT` with the tagged commit's full hash. Record aggregate verification results and archive/manifest hashes separately from earlier validation packets. Review the release notes and their installation command, then publish the verified draft:

```sh
gh release edit v0.1.0 --draft=false --latest
```

Finally, test the public installer download and its `--version` route from outside a source checkout. A local device policy that denies fresh dependency installation must not be bypassed; hosted construction and local testing of the resulting standalone archive keep package installation off that device.

## Scope

The release supports Apple silicon and macOS 26.4 or newer. It contains embedded Metal kernels, the two private-protocol workers, the converter, its Python runtime, license notices and a source/file-hash manifest. It contains no models or private evaluation artifacts. Model loading, numerical correctness and performance retain their separate source-bound qualification requirements.

The source builder and `Prebuilt release` action produce unsigned archives. `Signed release` is the separate signing and notarization boundary. Complete byte reproducibility across independent builds or signing runs is not promised; secure timestamps and the immutable archive checksum identify the exact tested distribution.
