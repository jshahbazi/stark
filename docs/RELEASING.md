# Publishing a standalone Mac release

The `Prebuilt release` workflow builds the current standalone workers and converter. It is manually dispatched on an existing version tag whose commit is on `main`. It does not run the inherited Uzu release, cloud or signing workflows.

## Prepare the source

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

## Verify and publish

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

These archives are not Developer ID signed or notarized. The installer does not remove quarantine attributes or change Gatekeeper settings. Complete byte reproducibility across independent Python builds is not promised; the immutable archive checksum identifies the exact tested build.
