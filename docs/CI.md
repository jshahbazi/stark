# Continuous integration and security checks

Public checks run for pull requests to `main`, pushes to `main` and `codex/**`, and manual workflow dispatches. Security and CodeQL also run weekly on the default branch after these files are merged. All test inputs are synthetic. Workflows upload no model, scoring, or native validation artifacts.

| Workflow | Coverage |
| --- | --- |
| CI | Tracked-file and workflow policy, formatting of the native engine/scorer crates, locked dependency resolution, workflow syntax, Linux/macOS process lifecycle tests, and Apple silicon scorer builds, Clippy, and unit tests with default and optimized features. |
| Security | RustSec advisories, dependency licenses and sources across all Cargo features, and dependency review for new vulnerabilities in pull requests. |
| CodeQL | Extended security queries for Rust, C/C++, Python, JavaScript/TypeScript, and GitHub Actions, using build mode `none`. Metal kernels and Swift bindings are not covered by these scans. |

CodeQL runs on Ubuntu. Semantic coverage of platform-gated Rust Metal wrappers can be incomplete; a successful scan does not validate the macOS native path. See [GitHub's build-mode documentation](https://docs.github.com/en/code-security/reference/code-scanning/codeql/build-options-for-compiled-languages).

Scoring builds use Rust 1.98.1 on the standard `macos-26` Apple silicon runner. The runner's Xcode and Metal toolchain must support the engine's Metal 4.0 / macOS 26.4 target. Build failures from runner image changes should be investigated as tooling changes; they do not replace source-bound validation evidence.

Python checks use `uv` 0.12.24 and Python 3.12.13. The policy checker has an adjacent script lockfile. Actionlint 1.7.12 and cargo-deny 0.20.2 downloads are verified against pinned SHA-256 hashes. External Actions use full commit SHAs, and checkout does not retain Git credentials. Build jobs have read-only repository tokens; only CodeQL analysis jobs receive `security-events: write`. Contribution code never runs through `pull_request_target` or a self-hosted runner. Dependency review leaves no PR comments.

Formatting uses `nightly-2026-10-01` because the upstream import configuration requires nightly rustfmt. Scoring compilation still uses Rust 1.98.1. Legacy crate formatting is not a gate. Actionlint suppresses only the constant-false-condition diagnostic used by deliberately disabled inherited jobs; the repository checker independently requires those jobs to remain disabled.

The inherited Tests, Release, Run Benchmark, and Run Stark Cargo Command workflows stay disabled. Their action references are pinned for repository policy compliance; this does not qualify or enable their infrastructure.

## Local checks

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
cargo +nightly-2026-10-01 fmt -p stark-engine -p stark-engine-macros -p stark-score --check
cargo metadata --locked --format-version=1 > /dev/null
cargo deny --locked --all-features check advisories licenses bans sources
cc -std=c11 -O2 -Wall -Wextra -Werror scripts/stark-validation-reap.c -o runtime/ci-stark-validation-reap
uv run --python 3.12.13 python scripts/test_scoring_process.py --reaper runtime/ci-stark-validation-reap
cargo build --release --locked -p stark-score --features optimized
cargo clippy --release --locked -p stark-score --all-targets --features optimized -- -D warnings
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
```

Create the ignored `runtime/` directory before compiling the process owner locally. Scorer commands require Apple silicon and Xcode Metal tools. Also run the scorer commands without `--features optimized` to exercise the default configuration. None of these commands loads a model.

## Repository controls

Keep secret scanning, push protection, Dependabot alerts, private vulnerability reporting, and read-only default Actions permissions enabled. Automated dependency-fix PRs are not enabled by this change; updates require maintainer review and locked validation. Require successful `CI completed` and `Security completed` checks on `main` once the workflows are present there and have produced successful checks. Prevent force pushes and branch deletion, and require conversation resolution. Review workflow, dependency, and security policy changes as carefully as native code.

The repository's live configuration was hardened on 2026-10-11: private reporting and Dependabot alerts were enabled, SHA pinning became mandatory, and the Actions allowlist was limited to checkout, dependency review, CodeQL, and setup-uv. `main` requires a pull request and conversation resolution, blocks force pushes and deletion, and applies its protections to administrators. The sole maintainer can merge without a second reviewer. Required status checks are deferred until the new workflows are installed on `main` and their initial failures are resolved.

To update a tool or action, resolve its release to an immutable commit or verified archive hash, update the pin and any lockfile together, and record a new validation run. Never add an advisory exception solely to make CI pass; document any independently reviewed exception with its scope, rationale, and expiry.
