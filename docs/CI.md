# Continuous integration and security checks

CI, Security and CodeQL run for pull requests to `main`, pushes to `main` and manual dispatches; Security and CodeQL also run weekly. To check a branch before opening a pull request, choose the branch under **Actions > CI > Run workflow**. The pull-request `CI completed` gate requires the `engine-unit` job as well as the other required CI jobs, except that the macOS Rust jobs are skipped for changes that cannot affect them; see [skipped Rust jobs](#skipped-rust-jobs). All CI inputs are synthetic: no workflow downloads a model, and private validation artifacts never enter hosted runners.

| Workflow | Coverage |
| --- | --- |
| CI | Repository policy, Rust formatting, locked dependency resolution, workflow syntax, native template evaluator build and tests, converter tests and lint, installer and app script tests, process lifecycle tests on Linux and macOS, Apple silicon builds, Clippy and tests of both inference workers in the default and `optimized` configurations, the CPU/grammar engine suite and compilation of Metal kernel tests on the pinned nightly compiler, and the vendored cxx panic-safety test with the engine's `grammar` feature. |
| Build | Builds an unsigned development bundle on pushes to `codex/**` that touch source or build inputs, and on manual dispatch, verifies it, and uploads it as a seven-day artifact. Pushes to `main` do not build; see [macOS runner budget](#macos-runner-budget). |
| Security | cargo-deny advisories, soundness, licenses and sources across all Cargo features; OSV scans of the seven tracked lockfiles; dependency review on pull requests. |
| CodeQL | Rust, C/C++, Python, JavaScript/TypeScript and GitHub Actions in build mode `none`. Metal kernels and Swift are not analyzed. |
| Prebuilt release | Manual; builds an unsigned archive from an existing version tag and uploads a draft. See [releasing](RELEASING.md). |
| Signed release | Manual from protected `main`; builds a combined CLI and app distribution from the dispatched `main` commit, verifies it before credentials are imported, signs and notarizes, verifies the installation and DMG, and in a separate publishing job creates the version tag and publishes the latest stable release. |

The obsolete inherited `Tests` workflow was removed. The remaining inherited `Release`, `Run Benchmark` and `Run Stark Cargo Command` workflows are present but every job is `if: false`; the repository policy check requires them to stay disabled.

## Toolchains and runners

- Rust code builds with Rust 1.98.1 on the `macos-26` Apple silicon runner, whose Xcode must support the engine's Metal 4 / macOS 26.4 target. Formatting uses `nightly-2026-10-01` rustfmt because the inherited rustfmt configuration needs nightly options, and the engine unit suite builds with the same nightly compiler because it uses `custom_test_frameworks`.
- Python checks use uv 0.12.24. Repository policy scripts run on Python 3.12.13 with an adjacent script lockfile; the converter and the installer tests run on Python 3.13.13.
- actionlint, cargo-deny and OSV Scanner downloads are verified against pinned hashes. External Actions are pinned to full commit SHAs, checkout does not persist credentials, build jobs hold read-only tokens, and no job uses `pull_request_target` or a self-hosted runner.

## macOS runner budget

Hosted macOS jobs share five concurrent runners across the whole repository, and queued jobs start in order with no priority. A Signed release dispatched after a merge therefore waits behind every macOS job already queued. Two rules keep one runner free for it when no other branch is building:

- Build does not run on pushes to `main`. The branch build covers the merged tree before the merge and the Signed release rebuilds and verifies the same commit. Dispatch Build on `main` under **Actions > Build > Run workflow** when a development bundle of `main` is needed.
- The CI `cxx` job starts after `lifecycle`, so a CI run holds at most four macOS runners at once. The dependency is scheduling only: `cxx` still runs when `lifecycle` fails.

Pull-request CI and branch builds from other branches still occupy runners. Dispatch a release before updating other open branches to avoid queueing behind them. When adding a macOS job, keep the peak for one CI run at four.

### Skipped Rust jobs

The `scoring`, `cxx` and `engine-unit` jobs account for most macOS runner time. The `changes` job skips all three when every changed path is on a short list of paths those jobs never read (`scripts/ci/rust-jobs-needed.sh`):

- `docs/`, `website/`, `apps/` and `provenance/`;
- top-level Markdown, `LICENSE*` and `install.sh`;
- `scripts/`, except `scripts/ci/` and `scripts/test_generation_process.py`;
- Python sources, Markdown, `pyproject.toml` and `uv.lock` under `tools/`.

Any other path runs the jobs, including a new top-level directory, workflow files, and data files under `tools/` because Rust sources include converter fixtures and memory profiles. A manual dispatch always runs them. The `lifecycle` job always runs because it tests the installer and app scripts, and the Linux `quality` job always runs its Rust formatting, dependency and template evaluator checks.

A pull request is compared with the tip of `main` it would merge into. A push to `main` is compared with the last commit whose CI run on `main` succeeded, not with the previous push, so a Rust change whose run was cancelled by a later merge is still built by the run that replaced it. If no base can be found the jobs run.

The `CI completed` gate accepts the three jobs as skipped only when `changes` succeeded and reported that they are not needed; otherwise each must succeed. Before a Rust job starts reading a file on the list, remove that path from the script. `scripts/ci/test_rust_jobs_needed.py` fails when a crate includes a file from outside `crates/` and `vendor/` that the list would skip.

## What CI does and does not cover

**Product code.** Both inference workers are built, linted with Clippy at `-D warnings`, and tested with synthetic inputs; the generation worker's process isolation is checked by a Python harness. The portable `stark-template` evaluator is built, unit-tested and checked with Clippy in the Linux repository job before the converter suite. Converter tests use the native evaluator and Hugging Face offline mode; installed-package checks copy the release helper beside the temporary environment's Python executable. Standalone and app builds include the helper, verify its fixed self-test protocol and render synthetic data after relocation. Installer, app builder, signer, updater and release-version scripts run their unit tests. The C process reaper is compiled with warnings as errors and exercised on Linux and macOS.

**Engine unit tests.** The `engine-unit` job compiles the default `cpu,metal` target with `nightly-2026-10-01`, then runs the suite under `crates/stark-engine/unit` with `--no-default-features --features cpu,grammar` (the suite needs `custom_test_frameworks`). The compile check preserves coverage of the default native target; the hosted test command avoids requiring a Metal context because macOS 26 hosted runners currently fail with `CannotCreateResidencySet`. On a physical Apple silicon Mac with working Metal residency, run the full native command from [Contributing](../CONTRIBUTING.md#engine-tests) to exercise the kernel comparisons between Metal and the CPU reference backend in `src/backends/cpu`. The ignored tests that read a converted model from `STARK_TEST_MODEL_PATH` do not run in CI; run them locally as described in [Contributing](../CONTRIBUTING.md#engine-tests). See the [failed hosted run](https://github.com/jshahbazi/stark/actions/runs/40227058178/job/121270551189) for the current runner limitation.

**Grammar support.** The `cxx` job runs the vendored cxx panic-safety test with the `grammar` feature, which keeps the XGrammar bridge building although neither worker enables structured output.

**Not covered.** Model loading, numerical parity, GPU performance and converted-model compatibility need private, model-bound runs; see [validation](validation/SCORING_PARITY.md) and the [benchmarks index](benchmarks/README.md).

## Dependency gates

OSV Scanner downloads the public advisory databases and matches the seven tracked lockfiles locally: the root `Cargo.lock`, `tools/uv.lock`, `tools/converter/uv.lock`, three `tools/stock_benchmark` uv lockfiles, and the policy script lockfile. A strict JSON checker requires every declared lockfile to appear in the report; adding or removing a lockfile means editing `security.yml` and `SCANNED_LOCKFILES` in `scripts/ci/check_repository.py` in the same change. Every known vulnerability fails both cargo-deny and the OSV gate; no advisory is ignored. Unmaintained-package notices are warnings.

## Rust build caches

Rust jobs use `.github/actions/rust-cache`, a thin wrapper over the pinned `actions/cache`. Each job and configuration has its own cache of registry sources, Git dependencies and `target` output, keyed by runner image, compiler and Xcode identity, Cargo manifests and lockfiles, build scripts and native sources (`scripts/ci/rust-cache-key.sh`). Incremental compilation is off so no per-checkout incremental state is stored. Every build, Clippy and test step still runs after a restore; caches are saved only when the job succeeds. Pull requests can restore `main` caches, but caches saved on a merge ref do not populate `main`. A push to `main` that skips the Rust jobs saves no cache; none of the skippable paths is part of a cache key, so the existing `main` caches still match. Increment the `stark-rust-v1` namespace in the key script to force a fresh cache. The Build workflow does not use the cache and compiles from scratch each run.

## Download a development build

Open **Actions > Build**, select a successful run and download its artifact. Each build has a unique `0.0.0-dev.COMMIT.RUN.ATTEMPT` version with the source commit and file hashes recorded in the bundle manifest. The artifact holds the archive, its `.sha256`, `install.sh` and `verification.json`; install it with:

```sh
sh install.sh --archive "$PWD/stark-VERSION-aarch64-apple-darwin.tar.gz"
```

Development builds are unsigned and expire after seven days. Use the [published release](INSTALLATION.md) for normal installation.

The app displays **Development build NUMBER** and records its build number
separately from the archive identifier. Its application version is `0.0.0` with a
development channel marker. Only release signing assigns a stable app version;
local build numbers cannot prevent the app from offering the latest stable
release. See [app versioning](../apps/menu-bar/README.md#build-and-install).

## Release workflows

The prebuilt release job has `contents: write` only for its final upload step; build and verification receive no publishing token. It requires an existing version tag on a commit merged into `main`.

The signed release workflow runs in the protected `macos-release` environment. Only its credential-import step receives the six Apple secrets; the signing step receives the public identity; a separate job with `contents: write` creates the immutable version tag, uploads the verified assets and publishes the latest stable release after installation, DMG and checksum verification. The helper creates a temporary Keychain, removes credential files and restores the previous Keychain search list unconditionally. Repository policy permits `secrets.*` references only in this workflow and the disabled inherited ones, and rejects other events, branches, missing cleanup and alias reuse.

## Local checks

Run the same checks CI runs before opening a pull request. Repository policy and converter:

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py --test
cargo +1.98.1 build --release --locked -p stark-template
cargo +1.98.1 test --locked -p stark-template -p stark-resource-budget
cargo +1.98.1 clippy --locked -p stark-template -p stark-resource-budget --all-targets -- -D warnings
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q
uv run --project tools/converter --locked --python 3.13.13 ruff check tools/converter
uv run --project tools/converter --locked --python 3.13.13 ruff format --check tools/converter
uv lock --check --directory tools/converter
```

The Flash Next image qualification harness has model-free counterexamples in
the repository job's benchmark and qualification step. They check that cache
mismatches, missing MTP proposals, image speculation, incorrect color labels and
resource failures stop
qualification, preserve a sanitized partial receipt and close the owned worker.
Run them separately when changing the harness:

```sh
uv run --project tools/converter --locked --python 3.13.13 pytest -q scripts/test_qualify_flashnext_images.py
uv run --project tools/converter --locked --python 3.13.13 ruff check scripts/qualify_flashnext_images.py scripts/test_qualify_flashnext_images.py
uv run --project tools/converter --locked --python 3.13.13 ruff format --check scripts/qualify_flashnext_images.py scripts/test_qualify_flashnext_images.py
```

These checks use synthetic worker replies. The opt-in
`scripts/qualify_flashnext_images.py` command still requires a private, actual
model run for artifact-specific image, MTP and prefix lifecycle evidence.

Installer, app and release scripts (shell syntax plus every unit test CI runs):

```sh
sh -n install.sh && sh -n scripts/install-from-source.sh && sh -n scripts/stark
uv run --no-project --python 3.13.13 python -m unittest \
  scripts/test_install.py scripts/test_install_app.py scripts/test_stark_cli.py \
  scripts/test_build_distribution.py scripts/test_build_menubar.py scripts/test_sign_distribution.py \
  scripts/test_apple_credentials.py scripts/test_build_dmg.py scripts/test_release_version.py scripts/test_app_update.py
```

The native beginner-flow checks use isolated preferences and synthetic model
identities without a backend or production credentials:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/test_beginner_setup.py
uv run --project tools/converter --locked --python 3.13.13 python scripts/test_openwork_beginner_handoff.py
```

Rust formatting, dependencies and the workers (Apple silicon with Xcode's Metal tools):

```sh
cargo +nightly-2026-10-01 fmt -p stark-engine -p stark-engine-macros -p stark-score -p stark-generate -p stark-template -p stark-resource-budget --check
cargo metadata --locked --format-version=1 > /dev/null
cargo deny --locked --all-features check --warn unmaintained advisories licenses bans sources
cargo test --locked -p stark-generate --no-default-features
cargo build --release --locked -p stark-score --features optimized
cargo clippy --release --locked -p stark-score --all-targets --features optimized -- -D warnings
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
cargo build --release --locked -p stark-generate
cargo clippy --release --locked -p stark-generate --all-targets -- -D warnings
cargo test --release --locked -p stark-generate -- --test-threads=1
cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal --no-run
cargo +nightly-2026-10-01 test --locked -p stark-engine --no-default-features --features cpu,grammar -- --test-threads=1
mkdir -p runtime
cc -std=c11 -O2 -Wall -Wextra -Werror scripts/stark-validation-reap.c -o runtime/ci-stark-validation-reap
uv run --python 3.12.13 python scripts/test_scoring_process.py --reaper runtime/ci-stark-validation-reap
uv run --no-project --python 3.12.13 python scripts/test_generation_process.py --binary target/release/stark-generate
```

Run the scorer commands a second time without `--features optimized` to cover the default configuration. None of these commands loads a model.

## Repository controls

Keep secret scanning, push protection, Dependabot alerts, private vulnerability reporting and read-only default Actions permissions enabled. `main` requires a pull request, conversation resolution, the `CI completed` and `Security completed` checks with the branch up to date, and blocks force pushes and deletion, including for administrators. The Actions allowlist is limited to the pinned checkout, dependency review, CodeQL, setup-uv, cache and artifact actions used by the workflows; the `macos-release` environment is restricted to protected branches.

To update a tool or action, resolve its release to an immutable commit or verified archive hash, update the pin and any lockfile together, and record the validation run in the pull request. Never ignore an advisory to make CI pass. An exception the maintainer accepts needs its scope, rationale and expiry recorded and enforced in both cargo-deny and the OSV gate; the retired [rustls exception](archive/runs/RUSTLS_TEST_EXCEPTION.md) shows the previous arrangement.
