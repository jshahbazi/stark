# Continuous integration and security checks

Public checks run for pull requests to `main`, pushes to `main` and `codex/**`, and manual workflow dispatches. Security and CodeQL also run weekly on the default branch after these files are merged. All test inputs are synthetic. Build artifacts contain the standalone distribution and aggregate installation checks; models, scoring payloads and private native validation artifacts stay outside public CI.

| Workflow | Coverage |
| --- | --- |
| CI | Tracked-file and workflow policy, native crate formatting, locked dependency resolution, workflow syntax, model-free converter tests and Ruff lint, Linux/macOS process lifecycle tests, Apple silicon scorer builds/Clippy/tests with default and optimized features, inherited Python binding and HTTP/1.1 server compatibility, synthetic server tests, and model-free TypeScript lint/type/test/package checks. |
| Build | Standalone Apple silicon development bundle, checksum, installer and aggregate installation verification; downloadable from the run for seven days. Runs on source/build changes pushed to `main` or `codex/**`, and manually on a selected branch. |
| Security | RustSec advisories, transitive soundness advisories, licenses and sources across all Cargo features; OSV checks of all seven dependency lockfiles, including the independent streaming converter; and dependency review for new vulnerabilities in pull requests. |
| CodeQL | Extended security queries for Rust, C/C++, Python, JavaScript/TypeScript, and GitHub Actions, using build mode `none`. Metal kernels and Swift bindings are not covered by these scans. |
| Prebuilt release | Manual version-tag dispatch on a merged commit; standalone Mac build, model-free archive verification and draft release upload. See [release operations](RELEASING.md). |
| Signed release | Manual dispatch from protected `main`; verifies a pinned prebuilt input, signs all runtime native images, requires Apple notarization and online executable tickets, verifies the installation, and creates a draft in a separate publishing job. |

CodeQL runs on Ubuntu. Semantic coverage of platform-gated Rust Metal wrappers can be incomplete; a successful scan does not validate the macOS native path. See [GitHub's build-mode documentation](https://docs.github.com/en/code-security/reference/code-scanning/codeql/build-options-for-compiled-languages).

Scoring builds use Rust 1.98.1 on the standard `macos-26` Apple silicon runner. The runner's Xcode and Metal toolchain must support the engine's Metal 4.0 / macOS 26.4 target. Build failures from runner image changes should be investigated as tooling changes; they do not replace source-bound validation evidence.

Python checks use `uv` 0.12.24 and Python 3.12.13; the converter uses its locked Python 3.13.13 environment. The policy checker has an adjacent script lockfile. TypeScript checks use Node 24.19.0 and pnpm 10.18.1 with a frozen lockfile and installation scripts disabled. Actionlint 1.7.12, cargo-deny 0.20.2, and OSV Scanner 2.6.0 downloads are verified against pinned SHA-256 hashes; the pnpm archive is verified against its SHA-512 registry integrity value. External Actions use full commit SHAs, and checkout does not retain Git credentials. Build jobs have read-only repository tokens; only CodeQL analysis jobs receive `security-events: write`. Contribution code never runs through `pull_request_target` or a self-hosted runner. Dependency review leaves no PR comments.

OSV downloads public advisory databases and matches dependencies locally. The scan names the root Cargo lockfile, the inherited wasm example Cargo lockfile, the tools, converter, and Python binding uv lockfiles, the TypeScript pnpm lockfile, and the repository-checker script lockfile explicitly. It does not install project dependencies, execute call analysis, or send dependency metadata to OSV. A strict JSON checker requires every declared lockfile to appear in the scan. Optional VCS commit metadata must be a full 40-character hexadecimal SHA; unknown package identity fields remain rejected. Known vulnerabilities and soundness advisories fail the gate unless covered by an explicit, unexpired exception. Both dependency jobs validate the exception manifest against exact locked package identities and cargo-deny's ignore list before scanning. OSV reports accepted findings as visible warnings. The current [rustls exception](RUSTLS_TEST_EXCEPTION.md) covers non-production tests only. Exactly classified unmaintained-package notices remain visible warnings; their advisory IDs are not ignored. Cargo-deny also checks transitive soundness advisories with `unsound = "all"`.

Compatibility jobs use the CPU feature for inherited bindings and server code. TypeScript tests stub the native callback interface and validate CommonJS, ESM, package exports, declarations, and source maps without loading an engine binary. Native integration tests require explicit `STARK_NATIVE_INTEGRATION=1` and stay outside public CI. The manual Rust binding-tool test command opts into those native tests. Python tool dependency updates have model-free local import/CLI validation; public CI checks their lockfile and advisories. Model export, inference, and numerical equivalence need separate private validation.

The existing repository quality job also runs converter synthetic tests with Hugging Face offline mode, Ruff, and a lock consistency check. It downloads no model and installs no Lalamo environment. Converter source/model parity and native loading remain separately recorded private validation evidence.

Standalone installation checks cover the launcher, bundle manifests and archive creation, checksum failures, unsupported platforms, command collisions and unsafe archive entries. Signing-tool tests reject altered input archives, invalid identities, missing native images, failed notarization and replacement of existing outputs. Credential-helper tests cover private-file permissions, partial setup, Keychain restoration and suppressed diagnostics. The macOS lifecycle job also exercises installation, relocation, upgrades and uninstall under a temporary prefix with spaces. These tests use synthetic workers and no model.

Formatting uses `nightly-2026-10-01` because the upstream import configuration requires nightly rustfmt. Scoring compilation still uses Rust 1.98.1. Legacy crate formatting is not a gate. Actionlint suppresses only the constant-false-condition diagnostic used by deliberately disabled inherited jobs; the repository checker independently requires those jobs to remain disabled.

The inherited Tests, Release, Run Benchmark, and Run Stark Cargo Command workflows stay disabled. Their action references are pinned for repository policy compliance; this does not qualify or enable their infrastructure.

Pull requests emit the required `CI completed` gate. Push and manual runs emit `Branch CI completed`, using the same aggregate checks. Distinct names prevent a cancelled duplicate branch run from blocking a passing pull request.

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

The signed-release workflow uses the protected `macos-release` environment on `main`. Only its credential-import step receives the six exact Apple secrets; the signing step receives the public signing identity. The helper creates a temporary Keychain, removes credential files, and restores the previous Keychain search list during unconditional cleanup. Its signing job has read-only repository access. A separate job receives `contents: write` and `GH_TOKEN` to upload the verified assets to a draft. Repository policy permits secret references only in this narrowly checked workflow and rejects additional secret references, other events, other branches, missing cleanup and credential reuse through YAML aliases. Existing inherited workflows remain disabled.

## Local checks

For installer changes, run:

```sh
sh -n install.sh
sh -n scripts/install-from-source.sh
sh -n scripts/stark
uv run --no-project --python 3.13.13 python -m unittest scripts/test_install.py scripts/test_stark_cli.py scripts/test_build_distribution.py
uv run --no-project --python 3.13.13 python -m unittest scripts/test_sign_distribution.py scripts/test_apple_credentials.py
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

To update a tool or action, resolve its release to an immutable commit or verified archive hash, update the pin and any lockfile together, and record a new validation run. Never add an advisory exception solely to make CI pass; document any independently reviewed exception with its scope, rationale, and expiry.
