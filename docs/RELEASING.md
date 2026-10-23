# Publishing a standalone Mac release

The `Signed release` workflow builds, signs and notarizes a combined CLI and menu bar application distribution from protected `main`. The separate `Prebuilt release` workflow constructs unsigned combined archives. Both create drafts for receiving-Mac verification before publication. Neither runs the inherited Uzu release, cloud or signing workflows.

## Release the combined CLI and application distribution

The original signed `v0.1.1` release used the immutable `v0.1.0` archive with SHA-256 `20f43b0f62fbe58dd026241031856a5f901067a88ed0f9cf27f2c3ea67d9e05b`, built from engine source `8c6f78c473c12fb9f86474d5c1aa8de4fb2d4b7e`. That historical input and receipt remain immutable; they are not the recipe for the combined application release.

After merging the reviewed changes with successful required checks, open **Actions → Signed release → Run workflow**, select `main`, and choose a `major`, `minor` or `patch` bump (default: `patch`). The equivalent command is:

```sh
gh workflow run signed-release.yml --ref main -f bump=patch
```

The workflow selects the highest stable `vMAJOR.MINOR.PATCH` Git tag or Cargo workspace version, then applies the selected bump. For example, a baseline of `0.1.2` produces `1.0.0`, `0.2.0` or `0.1.3`. Prerelease tags are excluded. All stable tags count, including tags belonging to drafts, so reserved versions are never reused. The selected version appears in the action log.

The workflow builds `stark-VERSION-unsigned-aarch64-apple-darwin.tar.gz` from the exact dispatched `main` commit, including the CLI and `Stark.app`; it verifies that combined archive before importing credentials. After signing and verification succeed, the separate publishing job creates `vVERSION` at that same commit and uploads a draft. No manual tag is required. An existing tag or release causes publication to fail; tags and assets are never overwritten. A failure after tag creation leaves that version reserved; a fresh dispatch selects the next version. Never move a published tag or replace published archive bytes.

The `macos-release` environment is restricted to protected branches. It contains these six secrets:

- `MACOS_CERTIFICATE_P12_BASE64`
- `MACOS_CERTIFICATE_PASSWORD`
- `MACOS_SIGNING_IDENTITY`
- `APPLE_API_KEY_P8_BASE64`
- `APPLE_API_KEY_ID`
- `APPLE_API_ISSUER_ID`

Keep the certificate export, API key and passwords outside Git and command output. The action imports them into a temporary Keychain, removes the private files, and unconditionally restores the previous Keychain search list and deletes the temporary Keychain. Only the import step receives all six secrets. Signing receives the public certificate identity and accesses the temporary Keychain profile. A separate publishing job receives repository write access and has no Apple credentials.

Before importing credentials, the action verifies the combined archive, application identity, exact file inventory and standalone installation. It then signs every runtime Mach-O image inside the CLI and application, signing nested code before its containing app, including the workers, interpreter, libpython, native Python extensions and bundled dynamic libraries. It requires the exact Developer ID Application certificate, hardened runtime, secure timestamps and no extra entitlements. Relocatable object files and static libraries are build support, not runtime code-signing targets; skipped Mach-O object paths are recorded explicitly.

After signing, the finalizer refreshes the file manifest, submits the application package for notarization, staples and validates the app ticket, then creates the final TAR and ZIP hashes. It requires notarization status `Accepted`, a valid stapled app ticket and online ticket checks for standalone executables. Signing or notarization failure publishes no release outputs. A source-bound aggregate receipt records the source and signing-recipe revisions and hashes, signer certificate and Team ID, native image count, Apple submission identifier and exact archive hashes. It contains no credentials or runtime payloads.

The signed draft includes the signed TAR archive and checksum, notarized ZIP and checksum, signed/notarized versioned DMG and checksum, stable `Stark.dmg` and `Stark.dmg.sha256` aliases, aggregate notarization receipts, installation and DMG verification JSON, and `install.sh`. The stable aliases are byte-for-byte copies of that release's verified versioned DMG; the stable checksum names `Stark.dmg`, and the publishing job checks both the checksum and exact byte equality before creating the draft. Existing versioned assets remain in the release. The DMG contains the unchanged, signed and stapled `Stark.app` plus an `Applications` shortcut. Users drag the application to Applications; the script installs both the application and CLI tools.

The website can therefore use the stable download URL without a site update:

```text
https://github.com/jshahbazi/stark/releases/latest/download/Stark.dmg
```

The signing job creates the DMG from the final signed TAR, signs the container with Developer ID, submits the DMG to Apple, staples its own ticket, and checks Gatekeeper acceptance. It mounts the image read-only to verify that the application file inventory, source revision, signature, and app ticket are preserved, then detaches it. A separate verification pass runs after credentials are removed. No application is opened or installed during DMG verification. This follows [Apple's distribution packaging guidance](https://developer.apple.com/documentation/xcode/packaging-mac-software-for-distribution).

Download the assets and verify the actual archive on a receiving Mac:

```sh
gh release download v0.1.2 --dir artifacts/release-0.1.2
uv run --project tools/converter --locked --python 3.13.13 python scripts/verify_distribution.py \
  --archive artifacts/release-0.1.2/stark-0.1.2-aarch64-apple-darwin.tar.gz \
  --version 0.1.2 --expected-source TAGGED_SOURCE_COMMIT \
  --expected-distribution-source SIGNING_RECIPE_COMMIT \
  --expected-signer EFE6CD2666B9106E68724C38ADD1C405267A2F94 --require-notarized --require-application \
  --output artifacts/release-0.1.2/receiving-mac-verification.json
```

Replace `TAGGED_SOURCE_COMMIT` and `SIGNING_RECIPE_COMMIT` with the full tagged commit. This adds checks of all native signatures, the installed app's seal and stapled ticket, and standalone executable notarization tickets to the normal installation, conversion, hashes, relocation and uninstall checks. After the actual draft passes:

```sh
gh release edit v0.1.2 --draft=false --latest
```

Test the public installer download, `stark doctor`, `stark --version` and `open "$HOME/Applications/Stark.app"` route from outside a checkout after publication. Preserve the earlier validation packets; record this run separately.

Verify the actual DMG separately before publishing the draft:

```sh
uv run --no-project --python 3.13.13 python scripts/build_dmg.py \
  --verify-dmg artifacts/release-VERSION/stark-VERSION-aarch64-apple-darwin.dmg \
  --receipt artifacts/release-VERSION/stark-VERSION-aarch64-apple-darwin.dmg.notarization.json \
  --expected-source TAGGED_SOURCE_COMMIT --version VERSION \
  --expected-signer EFE6CD2666B9106E68724C38ADD1C405267A2F94 \
  --output artifacts/release-VERSION/receiving-mac-dmg-verification.json
```

The release installer resolves GitHub's latest stable release by default and pins that validated version for both the archive and checksum downloads. The public install command requires no version argument:

```sh
curl -fL https://github.com/jshahbazi/stark/releases/latest/download/install.sh -o stark-install.sh
sh stark-install.sh
```

`--version VERSION` remains available for a pinned installation. Source builds require `--from-source` explicitly.

## Add a DMG to an existing signed release

Use the same protected signing workflow with an existing stable version:

```sh
gh workflow run signed-release.yml --ref main -f bump=patch -f existing_version=0.3.0
```

In this mode, `bump` is ignored. The workflow resolves the immutable release tag, requires its source to be an ancestor of the protected workflow source, downloads the existing TAR and checksum, and packages its signed application. It creates no tag or release and changes no existing archive bytes. After signing, notarization and credential-free verification, the new DMG assets are available in the workflow artifact `stark-signed-WORKFLOW_SOURCE_COMMIT`.

Download that artifact, run the receiving-Mac DMG verification above using the existing tag's source revision, then add the versioned DMG, checksum, notarization receipt, and DMG verification receipts with `gh release upload vVERSION FILE...`. Do not use `--clobber` for versioned archives or disk images. Existing release artifacts and source tags remain immutable.

After the verified image is available locally, backfill the stable aliases manually in the same release. This mode intentionally creates no release and does not upload aliases automatically:

```sh
cp artifacts/release-VERSION/stark-VERSION-aarch64-apple-darwin.dmg artifacts/release-VERSION/Stark.dmg
shasum -a 256 artifacts/release-VERSION/Stark.dmg | awk '{print $1 "  Stark.dmg"}' > artifacts/release-VERSION/Stark.dmg.sha256
cmp artifacts/release-VERSION/stark-VERSION-aarch64-apple-darwin.dmg artifacts/release-VERSION/Stark.dmg
shasum -a 256 --check artifacts/release-VERSION/Stark.dmg.sha256
gh release upload vVERSION \
  artifacts/release-VERSION/Stark.dmg artifacts/release-VERSION/Stark.dmg.sha256
```

Confirm the versioned DMG's original SHA-256 matches the stable alias before uploading. Never use `--clobber`; if `Stark.dmg` already exists, stop and investigate the release state.

The small `install.sh` release entrypoint may be refreshed to provide the latest stable installer behavior. Record its new source revision and SHA-256 in a separate verification receipt; preserve earlier receipts. Refreshing this entrypoint does not replace a versioned application archive or disk image.

Apple staples the ticket to `Stark.app`; standalone command-line binaries and ZIP files use online tickets. The distributed TAR and ZIP contain the same final signed and stapled application bytes. See [Apple's notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow). The installer never removes quarantine or changes Gatekeeper policy.

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

The release supports Apple silicon and macOS 26.4 or newer. It contains terminal chat, the menu bar app, embedded Metal kernels, the two private-protocol workers, the converter, private Python runtimes, license notices and source/file-hash manifests. It contains no models or private evaluation artifacts. Model loading, numerical correctness and performance retain their separate source-bound qualification requirements.

The source builder and `Prebuilt release` action produce unsigned archives. `Signed release` is the separate signing and notarization boundary. Complete byte reproducibility across independent builds or signing runs is not promised; secure timestamps and the immutable archive checksum identify the exact tested distribution.
