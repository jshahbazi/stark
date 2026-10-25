# Continuous integration and security checks

CI, Security and CodeQL run for pull requests to `main`, pushes to `main`, and manual workflow dispatches. Feature-branch pushes do not start a second copy of these workflows alongside the pull request. To validate a branch before opening a pull request, select the branch under **Actions → CI → Run workflow**; Security and CodeQL have the same manual control. Security and CodeQL also run weekly on the default branch. Build continues to produce development bundles for source changes pushed to `main` or `codex/**`. All test inputs are synthetic. Build artifacts contain the standalone distribution and aggregate installation checks; models, scoring payloads and private native validation artifacts stay outside public CI.

| Workflow | Coverage |
| --- | --- |
| CI | Tracked-file and workflow policy, native crate formatting, locked dependency resolution, workflow syntax, model-free converter tests and Ruff lint, Linux/macOS process lifecycle tests, Apple silicon scorer builds/Clippy/tests with default and optimized features, inherited Python binding and HTTP/1.1 server compatibility, synthetic server tests, and model-free TypeScript lint/type/test/package checks. |
| Build | Standalone Apple silicon development bundle, checksum, installer and aggregate installation verification; downloadable from the run for seven days. Runs on source/build changes pushed to `main` or `codex/**`, and manually on a selected branch. |
| Security | RustSec advisories, transitive soundness advisories, licenses and sources across all Cargo features; OSV checks of all seven dependency lockfiles, including the independent streaming converter; and dependency review for new vulnerabilities in pull requests. |
| CodeQL | Extended security queries for Rust, C/C++, Python, JavaScript/TypeScript, and GitHub Actions, using build mode `none`. Metal kernels and Swift bindings are not covered by these scans. |
| Prebuilt release | Manual version-tag dispatch on a merged commit; standalone Mac build, model-free archive verification and draft release upload. See [release operations](RELEASING.md). |
| Signed release | Manual dispatch from protected `main`; builds a fresh combined CLI and `Stark.app` distribution from the dispatched commit, verifies it before credentials, signs nested native code inside out, staples and validates the app notarization ticket, verifies the installation and DMG, then creates the immutable tag and publishes the latest stable release in a separate job. |

CodeQL runs on Ubuntu. Semantic coverage of platform-gated Rust Metal wrappers can be incomplete; a successful scan does not validate the macOS native path. See [GitHub's build-mode documentation](https://docs.github.com/en/code-security/reference/code-scanning/codeql/build-options-for-compiled-languages).

Scoring builds use Rust 1.98.1 on the standard `macos-26` Apple silicon runner. The runner's Xcode and Metal toolchain must support the engine's Metal 4.0 / macOS 26.4 target. Build failures from runner image changes should be investigated as tooling changes; they do not replace source-bound validation evidence.

Python checks use `uv` 0.12.24 and Python 3.12.13; the converter uses its locked Python 3.13.13 environment. The policy checker has an adjacent script lockfile. TypeScript checks use Node 24.19.0 and pnpm 10.18.1 with a frozen lockfile and installation scripts disabled. Actionlint 1.7.12, cargo-deny 0.20.2, and OSV Scanner 2.6.0 downloads are verified against pinned SHA-256 hashes; the pnpm archive is verified against its SHA-512 registry integrity value. External Actions use full commit SHAs, and checkout does not retain Git credentials. Build jobs have read-only repository tokens; only CodeQL analysis jobs receive `security-events: write`. Contribution code never runs through `pull_request_target` or a self-hosted runner. Dependency review leaves no PR comments.

OSV downloads public advisory databases and matches dependencies locally. The scan names the root Cargo lockfile, the inherited wasm example Cargo lockfile, the tools, converter, and Python binding uv lockfiles, the TypeScript pnpm lockfile, and the repository-checker script lockfile explicitly. It does not install project dependencies, execute call analysis, or send dependency metadata to OSV. A strict JSON checker requires every declared lockfile to appear in the scan. Optional VCS commit metadata must be a full 40-character hexadecimal SHA; unknown package identity fields remain rejected. Known vulnerabilities and soundness advisories fail the gate unless covered by an explicit, unexpired exception. Both dependency jobs validate the exception manifest against exact locked package identities and cargo-deny's ignore list before scanning. OSV reports accepted findings as visible warnings. The current [rustls exception](RUSTLS_TEST_EXCEPTION.md) covers non-production tests only. Exactly classified unmaintained-package notices remain visible warnings; their advisory IDs are not ignored. Cargo-deny also checks transitive soundness advisories with `unsound = "all"`.

Compatibility jobs use the CPU feature for inherited bindings and server code. The debug-profile CXX panic-safety and grammar tests run in a separate macOS job, in parallel with inherited dependency checks and release-profile server tests. This avoids putting both long builds on the same critical path. The completion gate requires both jobs to succeed, along with every other CI job; cancellation, failure and skipped checks do not pass it. TypeScript tests run on Ubuntu, stub the native callback interface and validate CommonJS, ESM, package exports, declarations, and source maps without loading an engine binary. Native integration tests require explicit `STARK_NATIVE_INTEGRATION=1` and stay outside public CI. The manual Rust binding-tool test command opts into those native tests. Python tool dependency updates have model-free local import/CLI validation; public CI checks their lockfile and advisories. Model export, inference, and numerical equivalence need separate private validation.

The existing repository quality job also runs converter synthetic tests with Hugging Face offline mode, Ruff, and a lock consistency check. It downloads no model and installs no Lalamo environment. Converter source/model parity and native loading remain separately recorded private validation evidence.

Standalone installation checks cover the launcher, bundle manifests and archive creation, checksum failures, unsupported platforms, command collisions and unsafe archive entries. Signing-tool tests reject altered input archives, invalid identities, missing native images, failed notarization and replacement of existing outputs. Application-builder and application-installer tests cover nested bundle inventories, app identity, archive installation and safe replacement. Credential-helper tests cover private-file permissions, partial setup, Keychain restoration and suppressed diagnostics. The macOS lifecycle job also exercises installation, relocation, upgrades and uninstall under a temporary prefix with spaces. These tests use synthetic workers and no model.

Formatting uses `nightly-2026-10-01` because the upstream import configuration requires nightly rustfmt. Scoring compilation still uses Rust 1.98.1. Legacy crate formatting is not a gate. Actionlint suppresses only the constant-false-condition diagnostic used by deliberately disabled inherited jobs; the repository checker independently requires those jobs to remain disabled.

The inherited Tests, Release, Run Benchmark, and Run Stark Cargo Command workflows stay disabled. Their action references are pinned for repository policy compliance; this does not qualify or enable their infrastructure.

Pull requests emit the required `CI completed` gate. Push and manual runs emit `Branch CI completed`, using the same aggregate checks. Distinct names prevent a cancelled duplicate branch run from blocking a passing pull request.

## CI timing evidence

The scheduling change was validated on 2026-10-22 UTC with identical native and TypeScript source, feature flags, test commands and toolchain pins. Only the workflows and this document changed between these commits.

| CI run | Source commit | Elapsed time |
| --- | --- | --- |
| [Before: pull request](https://github.com/jshahbazi/stark/actions/runs/39704758381) | `a60e554ce65eadd2f8f679729ed6590d161a74fd` | 31m26s |
| [After: manual branch validation](https://github.com/jshahbazi/stark/actions/runs/39708539187) | `09ede8daec2f8debf764a44c1be2f36254011b37` | 16m24s |

Both CI runs passed. Elapsed time includes runner queues and the completion gate; the observed reduction was 48%. The CXX test step took 12m31s in both runs. In the second run its separate job queued for 2m35s and finished while the inherited compatibility job was still running. TypeScript passed on Ubuntu in 23s, compared with 33s on macOS. These are individual hosted-run measurements; runner availability and image changes affect subsequent timings.

## Rust build caches

Rust CI jobs use the shared `.github/actions/rust-cache` action with GitHub's `actions/cache` v6.1.0 pinned to `55cc8345863c7cc4c66a329aec7e433d2d1c52a9`. Each job/configuration has its own cache of Cargo registry sources, Git dependencies and `target` outputs. The inherited compatibility job also populates the independent wasm target directory. Incremental compilation is disabled to avoid storing per-checkout incremental state. Models, private validation artifacts and Cargo credentials are outside the cache paths.

Cache compatibility includes runner OS, architecture and image version; Rust/C/C++ compiler identity; Xcode, SDK and applicable Metal compiler identity; compiler flags; Cargo manifests, lockfiles, profiles and configuration; build scripts, native sources and code generators; and the cache/workflow implementation. Metal's randomized cryptex `InstalledDir` is excluded from its version output; its compiler build and target versions remain covered. Native or dependency changes invalidate the fallback boundary. An ordinary Rust source change gets a new exact key and can restore a compatible prior build; Cargo fingerprints rebuild changed crates. Changes under `docs/` or to the root README reuse the exact key.

Every Cargo build, Clippy invocation and test still runs after a restore. Caches are saved only after the owning job succeeds. GitHub limits cache access by branch: pull requests can restore default-branch caches, while caches saved for a PR merge ref cannot populate `main`. See [GitHub's cache access rules](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#restrictions-for-accessing-a-cache). Cache misses and evictions trigger normal cold builds. Increment the `stark-rust-v1` namespace in `scripts/ci/rust-cache-key.sh` to force a fresh cache after an infrastructure issue.

### Hosted cache validation

The [cold population run](https://github.com/jshahbazi/stark/actions/runs/39874355777) and immediately following [warm run](https://github.com/jshahbazi/stark/actions/runs/39879305262) both passed on source commit `0f73fa9b557a131938d037478329a6190e5dd871`. All five Rust jobs saved caches after their cold builds and restored their exact primary keys on the warm run. Every existing build, Clippy and test step executed, including 56 server tests and seven CXX tests. [Security also passed](https://github.com/jshahbazi/stark/actions/runs/39874358763) on this commit.

| Cached job | Cold execution | Warm execution |
| --- | --- | --- |
| Repository checks | 1m57s | 2m13s |
| Scorer, default | 10m16s | 10m35s |
| Scorer, optimized | 7m12s | 6m27s |
| Inherited dependency compatibility | 18m12s | 4m35s |
| CXX panic safety and grammar compatibility | 16m24s | 2m42s |

Execution includes job setup, cache transfers and cleanup, and excludes time waiting for a runner. Summed execution across these parallel jobs fell from 54m01s to 26m32s, a 51% reduction. Compatibility execution fell by 75% and CXX by 84%. Default scorer execution did not improve: dependencies were reused, while Cargo rebuilt workspace crates and their native Metal work after the fresh checkout. Caching does not eliminate that remaining build cost.

Whole-workflow elapsed time, measured from workflow creation to its terminal update, was 47m48s cold and 38m16s warm. Runner queues dominated both runs: CXX waited 31m15s cold and 34m01s warm; warm compatibility and optimized scoring also waited over 30 minutes. These hosted measurements demonstrate cache reuse and reduced execution work; runner availability and image changes affect subsequent elapsed times.

## Download a development build

Open **Actions → Build**, select a successful run, then download its artifact. Each build has a unique `0.0.0-dev.COMMIT.RUN.ATTEMPT` version, with the full source commit and file hashes recorded in the bundle manifest. Documentation-only pushes do not start a build. The manual **Run workflow** control can build a selected branch once this workflow is on `main`.

The artifact contains four files: `stark-VERSION-aarch64-apple-darwin.tar.gz`, its `.sha256` checksum, `install.sh` and `verification.json`. After extracting the artifact, install the exact archive locally:

```sh
sh install.sh --archive "$PWD/stark-VERSION-aarch64-apple-darwin.tar.gz"
```

Replace `VERSION` with the version in the filename. The bundle supports Apple silicon and macOS 26.4 or newer. It includes both workers, the converter and its Python runtime. The workflow checks installation, offline synthetic conversion, file hashes, relocation and uninstall before uploading these four files. Failed verification uploads no artifact. Artifacts expire after seven days; download any build you need to keep.

Development builds are unsigned. Use the [published signed release](INSTALLATION.md) for normal installation. This read-only workflow uses no Apple credentials and creates no GitHub release. It reuses the release builder without repeating the scorer lint/test jobs already covered by CI.

## Release workflows

The separate manual prebuilt release job has `contents: write` for its draft upload. Checkout persists no credential, and only the final upload step receives `GH_TOKEN`; build and verification processes receive no publishing token. The workflow requires an existing matching version tag on a commit merged into `main`. Publication follows verification of the actual draft assets.

The signed-release workflow uses the protected `macos-release` environment on `main`. It builds a fresh combined CLI and app from the exact dispatched commit, verifies the app before credentials, and signs nested code inside out. Only its credential-import step receives the six exact Apple secrets; the signing step receives the public signing identity. The helper creates a temporary Keychain, removes credential files, and restores the previous Keychain search list during unconditional cleanup. Its signing job has read-only repository access. A separate job receives `contents: write` and `GH_TOKEN` to create the immutable version tag, upload the verified assets and publish the latest stable release after installation, DMG and checksum verification. Receiving-Mac checks provide additional validation after publication. Repository policy permits secret references only in this narrowly checked workflow and rejects additional secret references, other events, other branches, missing cleanup and credential reuse through YAML aliases. Existing inherited workflows remain disabled.

## Local checks

For installer changes, run:

```sh
sh -n install.sh
sh -n scripts/install-from-source.sh
sh -n scripts/stark
uv run --no-project --python 3.13.13 python -m unittest scripts/test_install.py scripts/test_stark_cli.py scripts/test_build_distribution.py
uv run --no-project --python 3.13.13 python -m unittest scripts/test_sign_distribution.py scripts/test_apple_credentials.py scripts/test_build_menubar.py scripts/test_install_app.py
uv run --project tools/converter --locked ruff check scripts/build_distribution.py scripts/test_build_distribution.py scripts/test_install.py scripts/test_stark_cli.py
```

The broader repository checks are:

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py --test
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q
uv run --project tools/converter --locked --python 3.13.13 ruff check tools/converter
uv lock --check --directory tools/converter
cargo +nightly-2026-10-01 fmt -p stark-engine -p stark-engine-macros -p stark-score --check
cargo metadata --locked --format-version=1 > /dev/null
uv run --no-project --python 3.12.13 python scripts/ci/check_dependency_exceptions.py
cargo deny --locked --all-features check --warn unmaintained advisories licenses bans sources
cc -std=c11 -O2 -Wall -Wextra -Werror scripts/stark-validation-reap.c -o runtime/ci-stark-validation-reap
uv run --python 3.12.13 python scripts/test_scoring_process.py --reaper runtime/ci-stark-validation-reap
cargo build --release --locked -p stark-score --features optimized
cargo clippy --release --locked -p stark-score --all-targets --features optimized -- -D warnings
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
```

Create the ignored `runtime/` directory before compiling the process owner locally. Scorer commands require Apple silicon and Xcode Metal tools. Also run the scorer commands without `--features optimized` to exercise the default configuration. None of these commands loads a model.

## Repository controls

Keep secret scanning, push protection, Dependabot alerts, private vulnerability reporting, and read-only default Actions permissions enabled. Automated dependency-fix PRs are not enabled by this change; updates require maintainer review and locked validation. Require successful `CI completed` and `Security completed` checks on `main`, with the branch up to date before merging. Prevent force pushes and branch deletion, and require conversation resolution. Review workflow, dependency, and security policy changes as carefully as native code.

The repository's live configuration was hardened on 2026-10-11: private reporting and Dependabot alerts were enabled, SHA pinning became mandatory, and the Actions allowlist was limited to checkout, dependency review, CodeQL, setup-uv, and setup-node. `main` requires a pull request and conversation resolution, blocks force pushes and deletion, and applies its protections to administrators. The sole maintainer can merge without a second reviewer. Required `CI completed` and `Security completed` checks are installed with strict branch freshness and are restricted to the GitHub Actions app (ID 15368). These workflow definitions must be included when integrating this branch; scheduled scanning and default-branch dependency alerts use `main` after integration.

The signing addition permits the exact reviewed upload-artifact and download-artifact commits for transfer between the signing and publishing jobs. Build uses the same permitted upload-artifact commit. The other Actions restrictions remain in place, and the `macos-release` environment is restricted to protected branches.

Rust CI caching additionally permits only the exact `actions/cache` commit above. Repository-token permissions, required merge checks and the signing environment are unchanged.

To update a tool or action, resolve its release to an immutable commit or verified archive hash, update the pin and any lockfile together, and record a new validation run. Never add an advisory exception solely to make CI pass; document any independently reviewed exception with its scope, rationale, and expiry.
