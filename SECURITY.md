# Security policy

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/jshahbazi/stark/security/advisories/new). Include the affected commit or release version, platform, impact, and a minimal reproduction using synthetic inputs. Keep credentials, model weights, real prompts, responses, token vectors, decision probabilities, exception dumps, and private evaluation artifacts out of reports and public issues. Coordinate disclosure through the private report.

## Supported code

Security fixes target the current `main` branch and ship in the next signed release. Signed, notarized releases are published on the [GitHub releases page](https://github.com/jshahbazi/stark/releases); older releases are not patched. The supported product consists of:

- the native workers `stark-score` and `stark-generate` and the `stark-engine` crate they link;
- the `stark_convert` Python package (converter, terminal chat, the menu bar backend and its loopback API);
- the menu bar application in `apps/menu-bar`;
- the installer, build, signing and update scripts under `scripts/` and `install.sh`.

The inherited tools under `tools/data_exporter`, `tools/gpu_trace` and `tools/model_compare` are not part of the product. They can still contain reportable defects, but they are not built into any release.

## Trust boundaries

**Workers.** Each worker loads one local model directory chosen by its caller and speaks a private JSON-lines protocol over its own stdin and stdout. The caller owns authentication, process isolation and protection of request and response data. A worker opens no network listener and provides no sandbox. Model files, tokenizer files and chat templates are inputs whose validation matters: use models from sources you trust. See [worker protocols](docs/WORKER_PROTOCOLS.md).

**Loopback API.** The menu bar application runs an HTTP listener bound to `127.0.0.1` only. Its random 256-bit master bearer key is stored in macOS Keychain and persists across launches until explicitly regenerated. Any process on the same Mac that obtains the key can use the loaded model. **Copy** under **Connections → Connect your own app** places the key on the system pasteboard, where clipboard managers and Universal Clipboard can read it. The managed OpenWork and OpenCode Desktop connections use separate short-lived capabilities; these HTTP bearer tokens are transferable while valid, and explicit Disconnect revokes that client's active grants. Requests carrying an `Origin` header are rejected, so browser pages cannot use the API. See [local API](docs/LOCAL_API.md).

**Installation routes.** The three ways to install Stark verify different things:

| Route | What is verified |
| --- | --- |
| Download the DMG and drag `Stark.app` to Applications | macOS Gatekeeper checks the Developer ID signature and notarization ticket because the browser download carries a quarantine attribute. |
| `install.sh` | Downloads the versioned archive and its `.sha256` file from the GitHub release and checks that they match. Both files come from the same origin over TLS; the checksum detects corruption, not substitution by someone who can publish release assets. It runs `doctor` on the extracted bundle and, for combined installs, the app installer validates app structure and runs `codesign --verify --deep --strict` on both the source and staged app. This does not independently pin Stark's signing identity or perform Gatekeeper assessment of the downloaded archive. |
| **Check for Updates** inside the app | Requires the exact GitHub asset digest, a Developer ID signature from Stark's team, a passing Gatekeeper assessment and a version that matches the release tag. |

Use the DMG or the in-app updater when independently checked publisher identity and Gatekeeper assessment matter to you. See [installation](docs/INSTALLATION.md#what-each-install-route-verifies).

**Repository.** A malicious contributor can change build scripts and workflows. Public CI runs on disposable hosted runners with read-only tokens, downloads no models, and exposes no signing credentials to contribution code. See [CI and security checks](docs/CI.md).

## Security invariants

- Worker requests pass a structural scan before Serde allocates their parsed representation: nesting and string/container nodes are bounded, with a separate byte-derived allowance for large flat token-ID arrays. Before any native operation, invalid field names and types, lengths, context sizes, token IDs and non-finite numerical parameters are rejected with fixed error codes. See [resource admission](docs/WORKER_PROTOCOLS.md#resource-budget).
- A complete request is serialized against one loaded model. The scorer's optional in-memory cache serves only an identical, immediately consecutive token sequence with the same context size; it is cleared on failures and restart and is never persisted. When enabled, the generator may reuse a chunk-aligned shared prefix from the immediately preceding request, including attention KV state and, for hybrid models, a recurrent-state snapshot. Reuse requires the same loaded model instance, context size, host cache scope and state-affecting options such as MTP. Failures, cancellation, image requests and restarts clear the cache; it is never persisted. See [generation prefix reuse](docs/GENERATION.md#prefix-reuse).
- Model-provided Jinja chat templates are compiled and rendered in a fresh native Rust `stark-template` process using MiniJinja, with no Python evaluation fallback. Source remains bounded at 256 KiB, with 16,384 source tokens, 1,000,000 VM fuel, recursion limits, 2 seconds CPU and 5 seconds wall time. The [automatic shared resource budget](docs/WORKER_PROTOCOLS.md#resource-budget) selects logical content/output budget `B` from available memory at startup with a 16 MiB floor and no fixed content ceiling; the host pins the selection for child processes. Encoded requests are bounded at `6B + 2 MiB`, live Rust allocation requests at `128 MiB + 8B` and cumulative requests at `32 MiB + 32B`. `STARK_REQUEST_BUDGET_MIB` remains a developer/operator override accepting positive integers whose derived byte allowances fit the platform's allocation address space. Accounting covers parsing, compilation and intermediate values; it does not bound stacks, allocator overhead or resident memory. Globals, methods and filters are restricted; no filesystem loader or network API is exposed. Child RSS is sampled against the selected live budget; this is not a kernel-enforced memory ceiling. Other POSIX systems additionally enforce that budget as an address-space limit. Unsupported or over-limit templates fail with fixed errors; source, inputs, output and compiler diagnostics are not logged. See [chat template evaluation](docs/GENERATION.md#chat-template-evaluation).
- Native diagnostics, panic details and capture variables never reach the protocol stream, logs or the UI. Prompts, responses, token vectors, probabilities, credentials and exception dumps are not logged by any component.
- Model and tensor parsing must not cause out-of-bounds access, memory corruption, unintended file access or uncontrolled resource consumption.
- The loopback API accepts loopback hosts only, rejects any `Origin` header and requires bearer authentication on every endpoint. The automatic shared budget bounds encoded text bodies at `6B + 2 MiB`, image bodies at that value plus 6 MiB, and aggregate pending bodies at two image-body budgets. Compressed PNG/JPEG file bytes carried in a data URL remain limited to 4 MiB; this is not a decoded-pixel limit. A structural scan bounds JSON at 65,536 nodes and 64 nested containers before parsed-tree allocation. Concurrent connections remain capped at 16. Raw-body reservations and structural admission bound resource exposure without establishing a total application RSS ceiling.
- Function calls are returned as validated API output for the client to handle. Stark does not execute tool functions; clients remain responsible for execution and approvals.
- Public CI uses synthetic, model-free tests. It must not download evaluation models, expose repository secrets to contribution code, publish validation artifacts, or activate the disabled inherited cloud, signing or release workflows.
- Dependencies remain locked, external Actions remain pinned to immutable commits, and upstream licenses and attribution remain intact.

## Findings and limitations

Report memory safety flaws, unsafe model parsing, sensitive-data disclosure, protocol or cache isolation failures, unintended process or file access, loopback API authentication or availability defects, installer and updater trust failures, dependency vulnerabilities, and exploitable workflow changes. Establish a reachable input and concrete impact; severity depends on the caller's deployment and privileges.

CI and CodeQL provide automated checks, not proof of runtime safety. CodeQL does not analyze Metal kernels, Swift, or model-dependent numerical results. GPU parity, performance, converted-model compatibility and host application integration require separate validation. No dependency vulnerability is accepted merely because an upstream configuration ignored it; `deny.toml` ignores no advisories, and any future exception must be time-boxed and enforced by both dependency gates.
