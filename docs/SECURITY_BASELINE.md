# Initial security and CI validation

This is a new tooling run dated 2026-10-11, separate from earlier model-dependent scoring validation. The dependency baseline is commit `97f43755`; `Cargo.lock` is unchanged with SHA-256 `303fc55ff149bada00ee751cac1b869b982b61fc50e1fc8bdd56e892be41d7dd`. Dependency checks used cargo-deny 0.20.2 and RustSec advisory database commit `7eebec69c352c7191b1f13eb95dd510eeca5d1de`.

## Local results

| Check | Result |
| --- | --- |
| Rust 1.98.1 release scorer tests, default features | 38 passed across the scorer and digest binary targets. |
| Rust 1.98.1 release scorer tests, optimized features | 38 passed across the scorer and digest binary targets. |
| Rust 1.98.1 release Clippy, both configurations and all scorer targets, warnings denied | Passed. |
| Native process-owner lifecycle | Eight model-free checks passed on Apple silicon macOS with Python 3.12.13. |
| Native crate formatting, nightly-2026-10-01 | Passed; formatter discovery covered 424 engine Rust files. |
| Repository workflow/tracked-file policy and actionlint 1.7.12 | Passed. |
| Dependency licenses, bans, and sources | Passed after adding the scorer's inherited MIT license metadata and allowing the existing LLVM exception and Unicode data license identifiers. |
| Full workspace RustSec advisories | Failed with three vulnerability advisories and ten unmaintained-package advisories. |

All tests used synthetic inputs. No model was loaded, no private evaluation was run, and no runtime artifacts were published. These checks establish CI wiring and model-free behavior only. Existing release-stripping warnings about `rust-objcopy` and a missing `libLLVM.dylib` did not fail builds or tests.

## Blocking dependency findings

| Advisory | Locked dependency | Required follow-up |
| --- | --- | --- |
| [RUSTSEC-2026-0258](https://rustsec.org/advisories/RUSTSEC-2026-0258) | h2 0.3.27 via hyper 0.14.32, rocket_http 0.5.1, Rocket, and the inherited CLI | Migrate the HTTP server dependency path to a fixed h2 version. The advisory specifies h2 >=0.4.16; it cannot be substituted into the old hyper dependency without a compatible migration. |
| [RUSTSEC-2026-0176](https://rustsec.org/advisories/RUSTSEC-2026-0176) | PyO3 0.28.3 in inherited Python bindings and binding-generation tools | Upgrade the coordinated PyO3, build-config, async-runtime, stub-generation, and NumPy dependency family to compatible releases using PyO3 >=0.29.0, then validate those bindings. |
| [RUSTSEC-2026-0177](https://rustsec.org/advisories/RUSTSEC-2026-0177) | PyO3 0.28.3 | The same coordinated migration is required for the closure thread-safety fix. |

The scan also reports unmaintained `backoff`, `instant`, `paste`, `smartstring`, and six `unic-*` crates reached through inherited tooling. Resolve those paths through dependency replacement or separately reviewed maintenance decisions. Existing upstream advisory ignores were removed, including the h2 vulnerability suppression. The new full-workspace security gate fails on these findings; no exception or accepted-risk decision is introduced here.

After Dependabot alerts were enabled, GitHub reported 48 open alerts on `main`: four critical, 20 high, 19 moderate, and five low. This is a broader advisory feed spanning Rust, Python tools, and inherited TypeScript dependencies; the counts overlap package/version lines and are not 48 demonstrated exploits.

| Critical/high alert location | Count | Affected dependencies |
| --- | --- | --- |
| `tools/uv.lock` | One critical, three high | anyio, fsspec, and transformers. |
| `Cargo.lock` | One high | PyO3. |
| `crates/legacy/stark/bindings/typescript/pnpm-lock.yaml` | Three critical, 16 high | handlebars, js-yaml, browserslist, brace-expansion, flatted, and minimatch across resolved versions. |

Review the current [Dependabot alerts](https://github.com/jshahbazi/stark/security/dependabot) before running inherited tooling. These dependency migrations are unresolved and are not covered by the model-free scorer tests.

Hosted CI execution and CodeQL analysis must be checked on the pushed branch. Required status checks on `main` remain deferred until the workflows are installed there and successful check names are observed. Private numerical parity and performance evidence remain separate.
