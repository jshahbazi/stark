# Security policy

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/jshahbazi/stark/security/advisories/new). Include the affected commit or release version, platform, impact, and a minimal reproduction using synthetic inputs. Keep credentials, model weights, real prompts, responses, token vectors, decision probabilities, exception dumps, and private evaluation artifacts out of reports and public issues. Coordinate disclosure through the private report.

## Supported code

Security fixes target the current `main` branch and ship in the next signed release. Signed, notarized releases are published on the [GitHub releases page](https://github.com/jshahbazi/stark/releases); older releases are not patched. The supported product consists of:

- the native workers `stark-score` and `stark-generate` and the `stark-engine` crate they link;
- the `stark_convert` Python package (converter, terminal chat, the menu bar backend and its loopback API);
- the menu bar application in `apps/menu-bar`;
- the installer, build, signing and update scripts under `scripts/` and `install.sh`.

The inherited Uzu crates under `crates/legacy`, the inherited sample application under `apps/playground`, and the inherited tools under `tools/data_exporter`, `tools/gpu_trace` and `tools/model_compare` are not part of the product. They can still contain reportable defects, but they are not built into any release.

## Trust boundaries

**Workers.** Each worker loads one local model directory chosen by its caller and speaks a private JSON-lines protocol over its own stdin and stdout. The caller owns authentication, process isolation and protection of request and response data. A worker opens no network listener and provides no sandbox. Model files, tokenizer files and chat templates are inputs whose validation matters: use models from sources you trust. See [worker protocols](docs/WORKER_PROTOCOLS.md).

**Loopback API.** The menu bar application runs an HTTP listener bound to `127.0.0.1` only, authenticated by a random 256-bit bearer key generated at each launch and held in memory. Any process on the same Mac that obtains the key can use the loaded model until the application quits. The key is not written to disk by Stark, but **Copy** in Connection Details places it on the system pasteboard, where clipboard managers and Universal Clipboard can read it. Requests carrying an `Origin` header are rejected, so browser pages cannot use the API. See [local API](docs/LOCAL_API.md).

**Installation routes.** The three ways to install Stark verify different things:

| Route | What is verified |
| --- | --- |
| Download the DMG and drag `Stark.app` to Applications | macOS Gatekeeper checks the Developer ID signature and notarization ticket because the browser download carries a quarantine attribute. |
| `install.sh` | Downloads the versioned archive and its `.sha256` file from the GitHub release and checks that they match. Both files come from the same origin over TLS; the checksum detects corruption, not substitution by someone who can publish release assets. Files written by `curl` carry no quarantine attribute, so Gatekeeper does not assess them, and the script does not check code signatures. |
| **Check for Updates** inside the app | Requires the exact GitHub asset digest, a Developer ID signature from Stark's team, a passing Gatekeeper assessment and a version that matches the release tag. |

Use the DMG or the in-app updater when signature verification matters to you. See [installation](docs/INSTALLATION.md#what-each-install-route-verifies).

**Repository.** A malicious contributor can change build scripts and workflows. Public CI runs on disposable hosted runners with read-only tokens, downloads no models, and exposes no signing credentials to contribution code. See [CI and security checks](docs/CI.md).

## Security invariants

- Worker requests are validated before any native operation: field names and types, lengths, context sizes, token IDs, and non-finite numerical parameters are rejected with fixed error codes.
- A complete request is serialized against one loaded model. The scorer's optional in-memory cache serves only an identical, immediately consecutive token sequence with the same context size; it is cleared on failures and restart and is never persisted. The generator keeps no state between requests.
- Native diagnostics, panic details and capture variables never reach the protocol stream, logs or the UI. Prompts, responses, token vectors, probabilities, credentials and exception dumps are not logged by any component.
- Model and tensor parsing must not cause out-of-bounds access, memory corruption, unintended file access or uncontrolled resource consumption.
- The loopback API accepts loopback hosts only, rejects any `Origin` header, requires the bearer key on every endpoint, and bounds request bodies at 1 MiB and concurrent connections at 16.
- Public CI uses synthetic, model-free tests. It must not download evaluation models, expose repository secrets to contribution code, publish validation artifacts, or activate the disabled inherited cloud, signing or release workflows.
- Dependencies remain locked, external Actions remain pinned to immutable commits, and upstream licenses and attribution remain intact.

## Findings and limitations

Report memory safety flaws, unsafe model parsing, sensitive-data disclosure, protocol or cache isolation failures, unintended process or file access, loopback API authentication or availability defects, installer and updater trust failures, dependency vulnerabilities, and exploitable workflow changes. Establish a reachable input and concrete impact; severity depends on the caller's deployment and privileges.

CI and CodeQL provide automated checks, not proof of runtime safety. CodeQL does not analyze Metal kernels, Swift, or model-dependent numerical results. GPU parity, performance, converted-model compatibility and host application integration require separate validation. No dependency vulnerability is accepted merely because an upstream configuration ignored it; accepted exceptions are time-boxed and listed in `deny.toml`.
