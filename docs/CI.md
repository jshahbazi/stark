# Continuous integration and security checks

Public checks run for pull requests to `main`, pushes to `main` and `codex/**`, and manual workflow dispatches. Security and CodeQL also run weekly on the default branch after these files are merged. All test inputs are synthetic. Workflows upload no model, scoring, or native validation artifacts.

| Workflow | Coverage |
| --- | --- |
| CI | Tracked-file and workflow policy, native crate formatting, locked dependency resolution, workflow syntax, Linux/macOS process lifecycle tests, Apple silicon scorer builds/Clippy/tests with default and optimized features, inherited Python binding and HTTP/1.1 server compatibility, synthetic server tests, and model-free TypeScript lint/type/test/package checks. |
| Security | RustSec advisories, transitive soundness advisories, licenses and sources across all Cargo features; OSV checks of all six dependency lockfiles; and dependency review for new vulnerabilities in pull requests. |
| CodeQL | Extended security queries for Rust, C/C++, Python, JavaScript/TypeScript, and GitHub Actions, using build mode `none`. Metal kernels and Swift bindings are not covered by these scans. |

CodeQL runs on Ubuntu. Semantic coverage of platform-gated Rust Metal wrappers can be incomplete; a successful scan does not validate the macOS native path. See [GitHub's build-mode documentation](https://docs.github.com/en/code-security/reference/code-scanning/codeql/build-options-for-compiled-languages).

Scoring builds use Rust 1.98.1 on the standard `macos-26` Apple silicon runner. The runner's Xcode and Metal toolchain must support the engine's Metal 4.0 / macOS 26.4 target. Build failures from runner image changes should be investigated as tooling changes; they do not replace source-bound validation evidence.

Python checks use `uv` 0.12.24 and Python 3.12.13. The policy checker has an adjacent script lockfile. TypeScript checks use Node 24.19.0 and pnpm 10.18.1 with a frozen lockfile and installation scripts disabled. Actionlint 1.7.12, cargo-deny 0.20.2, and OSV Scanner 2.6.0 downloads are verified against pinned SHA-256 hashes; the pnpm archive is verified against its SHA-512 registry integrity value. External Actions use full commit SHAs, and checkout does not retain Git credentials. Build jobs have read-only repository tokens; only CodeQL analysis jobs receive `security-events: write`. Contribution code never runs through `pull_request_target` or a self-hosted runner. Dependency review leaves no PR comments.

OSV downloads public advisory databases and matches dependencies locally. The scan names the root Cargo lockfile, the inherited wasm example Cargo lockfile, the tools and Python binding uv lockfiles, the TypeScript pnpm lockfile, and the repository-checker script lockfile explicitly. It does not install project dependencies, execute call analysis, or send dependency metadata to OSV. A strict JSON checker requires every declared lockfile to appear in the scan. Known vulnerabilities and soundness advisories fail the gate. Exactly classified unmaintained-package notices remain visible warnings; their advisory IDs are not ignored. Cargo-deny also checks transitive soundness advisories with `unsound = "all"`.

Compatibility jobs use the CPU feature for inherited bindings and server code. TypeScript tests stub the native callback interface and validate CommonJS, ESM, package exports, declarations, and source maps without loading an engine binary. Native integration tests require explicit `STARK_NATIVE_INTEGRATION=1` and stay outside public CI. The manual Rust binding-tool test command opts into those native tests. Python tool dependency updates have model-free local import/CLI validation; public CI checks their lockfile and advisories. Model export, inference, and numerical equivalence need separate private validation.

Formatting uses `nightly-2026-10-01` because the upstream import configuration requires nightly rustfmt. Scoring compilation still uses Rust 1.98.1. Legacy crate formatting is not a gate. Actionlint suppresses only the constant-false-condition diagnostic used by deliberately disabled inherited jobs; the repository checker independently requires those jobs to remain disabled.

The inherited Tests, Release, Run Benchmark, and Run Stark Cargo Command workflows stay disabled. Their action references are pinned for repository policy compliance; this does not qualify or enable their infrastructure.

## Local checks

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
cargo +nightly-2026-10-01 fmt -p stark-engine -p stark-engine-macros -p stark-score --check
cargo metadata --locked --format-version=1 > /dev/null
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

To update a tool or action, resolve its release to an immutable commit or verified archive hash, update the pin and any lockfile together, and record a new validation run. Never add an advisory exception solely to make CI pass; document any independently reviewed exception with its scope, rationale, and expiry.
