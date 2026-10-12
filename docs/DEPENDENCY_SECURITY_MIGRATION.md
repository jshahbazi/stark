# Dependency security migration

This run follows the initial 2026-10-11 scan of source revision `97f43755`. The changes are on `codex/security-ci`; the original scan remains in [Security baseline](SECURITY_BASELINE.md). Existing model-dependent validation results remain bound to their original source and toolchains.

## Dependency changes

- The PyO3 family moves from 0.28.3 to 0.29.3, with async runtimes 0.29.0, stub generation 0.23.1, and NumPy 0.29.0. This resolves RUSTSEC-2026-0176 and RUSTSEC-2026-0177. The Python 3.12 stable ABI and existing binding features are preserved.
- Rocket 0.5.1 now has default features disabled and retains its JSON feature. The inherited server supports HTTP/1.1, including its existing streaming API. HTTP/2 is disabled, removing the unpatched h2 0.3.27 dependency identified by RUSTSEC-2026-0258. Restoring HTTP/2 requires a compatible patched server stack.
- The root lockfile upgrades lru to 0.18.2 for the panic-safety/use-after-free issue RUSTSEC-2026-0253. The separate inherited wasm example lockfile updates h2 and rustls to fixed releases and refreshes local package identities against the current source. This resolves its additional RUSTSEC-2026-0258 and RUSTSEC-2026-0285 entries.
- The inherited CXX Git fork's string initialization was affected by RUSTSEC-2026-0202. A minimal vendored crate family preserves the fork's grammar bridge features and applies the official 1.0.195 exception-safety fixes. This is a documented backport, not a complete upstream release upgrade. See [CXX patch provenance](../vendor/cxx/PATCHES.md) for exact source revisions, patch commits, license hashes, and the local version boundary. A focused engine test target exercises its panic-safe initialization and existing string operations against the locked grammar build.
- Python tools use locked Transformers 5.15.1, Torch 2.14.1, AnyIO 4.14.2, fsspec 2026.9.0, idna 3.20, Pygments 2.21.0, and setuptools 84.0.0. Manifest minimums and constraints prevent returning to the vulnerable dependency ranges. The orphaned `tools/model_compare/uv.lock` is removed: that module is part of the root `tools` project and uses `tools/uv.lock`.
- TypeScript dependencies are regenerated under pnpm 10.18.1. Vitest 5.0.3 replaces Jest, and a local TypeScript compiler helper replaces tsc-multi. These replacements remove the unpatched braces and sprintf-js paths as well as the patchable inherited alerts. ESLint 10.12.0 and compatible TypeScript ESLint packages replace the earlier tooling. CommonJS, ESM, declarations, source maps, and NAPI adapter output contracts are checked with synthetic inputs.

## Security gate behavior

The new RustSec check includes transitive soundness advisories. OSV separately scans all six dependency lockfiles, including Git-sourced packages, with its all-packages mode. The report gate verifies declared source coverage and report structure; package extraction and version matching depend on the pinned scanner. Scanner errors, missing lockfile sources, malformed reports, unknown classifications, vulnerabilities, and unsoundness findings fail CI. Only notices explicitly classified as unmaintained packages become visible warnings. There are no vulnerability advisory exceptions.

Ten maintenance notices remain: backoff and instant through async-openai, paste through tokenizer/image dependencies, smartstring through the engine's Rhai build dependency, and six unic packages through the Python stub generator's parser. These require dependency replacement or upstream work. A passing vulnerability check does not resolve that maintenance debt.

## Validation scope

All checks use synthetic data or package metadata. No evaluation model, native inference session, private evaluation, or export workload was run. The dependency migrations do not establish numerical equivalence, GPU parity, performance, or converted-model compatibility. Python tool Ruff still reports 81 inherited issues in unchanged sources; it is not introduced as a passing gate by this migration. Public Dependabot alerts remain associated with `main` until the branch is integrated.

## CodeQL intake

The first completed CodeQL run on `e020464c737ced890cb6b6e4aabb9532ecf3be8b` reported 67 high-severity `rust/access-invalid-pointer` alerts in inherited N-API bindings. The completed [migration CodeQL run](https://github.com/jshahbazi/stark/actions/runs/38172755710) on `ddf904aa5ea96b1f509e847f8fbd346b9c3eeece` still reports 67 open instances of that rule. Its Actions, Python, JavaScript/TypeScript, and C/C++ analyses report zero results; all five analyses completed without errors. A bounded static review found 166 identical flows from `null_mut()` through the generated `wrapped_val` out parameter to a dereference. The pinned N-API generator checks the `napi_unwrap` return status before conversion; successful unwrap writes the out parameter, and normal native wrappers originate from `Box::into_raw()`.

Incomplete FFI modeling is a plausible explanation for the initial-null trace. No exploit path was confirmed by this review, and the alerts remain **needs review**. Foreign-wrapper type identity and lifetime safety were not qualified. No alert was dismissed or suppressed. These bindings are absent from the qualified scorer's dependency path, but remain reportable under the repository security policy. Follow-up should test ordinary, unwrapped, and foreign-class objects with a model-free native fixture before classifying the alerts or changing the shared generator. See the [Node-API unwrap contract](https://nodejs.org/api/n-api.html#napi_unwrap) and [CodeQL alerts](https://github.com/jshahbazi/stark/security/code-scanning).

## Migration validation

Local validation used Rust 1.98.1, uv 0.12.24, Node 24.19.0, and pnpm 10.18.1 on 2026-10-11.

| Check | Result |
| --- | --- |
| Coordinated PyO3 binding checks and CPU extension build | Passed; the built extension passed five import/constructor/export smoke checks. |
| PyO3 advisory boundaries | Two runtime iterator/callable checks and one compile-fail non-Sync closure check passed against 0.29.3. |
| HTTP/1.1 CLI compilation and synthetic server contracts | Passed; 56 server tests passed with the patched root lockfile. |
| CXX CPU/grammar bridge compilation and panic-safety tests | Passed; seven tests include both panicking value and panicking AsRef cases, plus ordinary string/async controls. |
| Inherited wasm example, wasm32-unknown-unknown | Locked compilation passed. |
| Rust binding tool command compilation | Passed. |
| TypeScript lint, types, clean dual-format build, package-name runtime/type resolution, and publint | Passed. |
| TypeScript model-free tool tests | Four passed; two native integration tests explicitly skipped. |
| Python tools | Locked sync/check, six module imports, three CLI help checks, and Transformers schema compatibility passed. |
| pnpm audit and pip-audit | No known vulnerabilities reported. |
| RustSec advisories, transitive soundness, licenses, bans, and sources | Passed with ten explicit maintenance warnings and no vulnerability exceptions. |
| OSV 2.6.0 and strict report gate | Six lockfiles, 1,489 extracted packages, ten maintenance warnings, zero blocking findings. |
| Dependency-report policy regression tests | 13 passed, including malformed reports, missing sources, and maintenance-versus-unsound classification. |
| Native formatting, repository policy, and workflow syntax | Passed. |

Before the dependency migration, the hosted [CI run](https://github.com/jshahbazi/stark/actions/runs/38171067521) and [CodeQL run](https://github.com/jshahbazi/stark/actions/runs/38171067533) both passed on `e020464c737ced890cb6b6e4aabb9532ecf3be8b`. CI included 38 scorer/digest tests in each of the default and optimized configurations and eight lifecycle checks on each operating system. Later pushed commits produce their own hosted runs; those results must be assessed against their recorded source revisions.

Hosted migration validation is bound to `ddf904aa5ea96b1f509e847f8fbd346b9c3eeece` on 2026-10-11:

| Workflow | Result |
| --- | --- |
| [CI](https://github.com/jshahbazi/stark/actions/runs/38172755695) | Passed both scorer builds and Clippy, 76 scorer test executions, eight lifecycle checks per operating system, 56 server tests, seven CXX tests, 13 report-policy tests, and four TypeScript tests with two native integration tests explicitly skipped. Python binding, binding-tool, wasm, repository-policy, and package-output checks also passed. |
| [Security](https://github.com/jshahbazi/stark/actions/runs/38172755731) | Passed RustSec, licenses, bans, sources, and the six-lockfile OSV gate with ten maintenance notices and zero blocking findings. |
| [CodeQL](https://github.com/jshahbazi/stark/actions/runs/38172755710) | All five language analyses completed successfully. The 67 Rust alerts remain open for review. |

The PR-only dependency-review job was not exercised by these push runs. A subsequent documentation-only commit records these results without changing the tested implementation or lockfiles.

| Lockfile | SHA-256 |
| --- | --- |
| Root Cargo | `65444f3e0e0707c43521539e43c95365cfc6a51ec6f9e90180bc43bc8fb4603e` |
| Inherited wasm example Cargo | `5c0cd1ee0708a3f16e718989405e11281ca8b2ea29ae8dccf972c8a7a722b99f` |
| Python tools uv | `acf7c45b6f7feac003986526a309a911d35bf68266e4455fbe33457b354378f9` |
| Python binding uv | `45462200247bd38b7ac09a826f4d6958a4fa8aa2812b0e47357edb740f3fcc3f` |
| TypeScript pnpm | `50f83e455c6e3f64a757fd4d77d17940ed198bcfb2af0cc984f19792b1f136ec` |
| Repository policy uv script | `0e1e290a7c4b246933c8d404fa0a993c3db36363fd9e423b87a36bf002fff7b8` |

## PR #2 lockfile remediation, 2026-10-12

The [failed Security run](https://github.com/jshahbazi/stark/actions/runs/38293010790) on source `c99f39e342310a4b2fcdace69b87469c68cbd10f` reported five blocking quinn-proto advisories: RUSTSEC-2026-0338 through RUSTSEC-2026-0342. Both Cargo lockfiles pinned 0.11.16. Updating their quinn-proto entries to the upstream patched 0.11.18 release resolves these findings without advisory exceptions or gate changes. All other package versions and dependency edges remain unchanged.

Local validation used Rust 1.98.1 and checksum-verified OSV Scanner 2.6.0 on macOS. Both `cargo metadata --locked` checks, `cargo check --locked -p download-manager`, and the repository policy checker passed. The same seven-lockfile offline scan and strict report gate used by CI reported 1,516 packages, ten maintenance notices, and zero blocking findings. These results apply to the lockfile hashes below; earlier source-bound validation remains unchanged. No model-dependent validation was run.

| Lockfile | SHA-256 |
| --- | --- |
| Root Cargo | `49a27c8d8b2eebc3bdf47d6957f5b2c8dbbb822ddd66090589a33bcbf1f35ec1` |
| Inherited wasm example Cargo | `6db0b0c0a8231cf5c44d4fffee1b526402d4df2b67e4f772e9664100ae0f7217` |
