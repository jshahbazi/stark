# Desktop agent backend validation — 2026-10-31

## Scope and source

Branch: `codex/desktop-agent-backend`, based on `origin/main` commit `c3c83fcd`. This record separates protocol tests, a native model task, packaged connection checks, and third-party desktop acceptance. No raw prompts, responses, credentials, token vectors or request/response dumps are retained.

Final product-source manifest SHA-256: `ee9bfe17a85c80843ef382fdfeea3c5b464779c993a149f7c475e0ccabc7fa00`. This hashes sorted `apps/menu-bar/Sources/*.swift` and `tools/converter/stark_convert/*.py` entries as relative-path UTF-8, NUL, and binary SHA-256. The final source also passed the existing package builder and full Swift typecheck; final ad-hoc host executable SHA-256: `07f500b8dce1261f9ea60c9e5980e91f280b1f9f3cbf8aa81e5c854fa26ad8c9`. This final package was built, not desktop-qualified.

The change adds host-owned Keychain credentials, saved listening ports and connection recovery controls, bounded Chat Completions function tools, and template-specific capability reporting. There is no tool executor, Responses API, remote listener, permissive CORS, training, onboarding agent or new release infrastructure.

## Verification results

| Check | Result |
| --- | --- |
| Repository policy | Passed; 41 policy unit tests |
| Converter suite, offline, locked Python 3.13.13 | 5,080 passed, 88 skipped |
| Later connection/auth tests and disconnect regression | 150 desktop/HTTP tests passed; final actual-port regression 2 passed |
| Request/parser focused suite | 87 passed |
| Swift credential runner | 2 passed, including injected failure/race cases and actual disposable Keychain cross-process reuse/regeneration |
| Native Swift controller harness | 105 passed |
| App/installer/distribution/signing/updater script unit suite | 182 passed; builder subset 23 passed |
| Swift source typecheck and ConnectionInfo JSON/curl harness | Passed |
| Ruff check/format, converter lock, shell syntax, diff whitespace | Passed |
| Rust nightly-2026-10-01 formatting | Passed |
| Rust 1.98.1 locked release builds | Generator and optimized scorer passed |
| Rust generator, no default features | 41 passed; existing unused-code warnings |
| Native generator process lifecycle harness | 6 cases passed |

Protocol fixtures cover definitions/choice modes, null assistant text, argument JSON, linked/reordered tool results, bounded nesting and bytes, malformed/truncated/unknown calls, and buffered/streamed equivalence. Desktop tests cover queue release, context overflow, cancellation, replacement fencing, worker failure and retries. Tool deltas are emitted only after validated terminal output, exactly once per index. Usage preserves native prompt/output counts.

Independent review found two defects: OS-level corrupt-entry recovery could not regenerate, and a disconnected client during buffered tool text delivery unloaded the model. Both were corrected and independently rechecked before delivery. A stale empty-tools rejection expectation was updated because neutral `tools: []` is now supported.

## Native model task evidence

Model: `Qwen/Qwen3-0.6B`, source revision `c1899de289a04d12100db370d81485cdf75e47ca`. Artifact: local `qwen3-dense`, unquantized BF16 weights with BF16/F32 tensor inventory, 226 tensors. Template SHA-256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`; parser `qwen3-hermes-v1`.

Fresh native readiness reports a 16,384-token context limit, 4,096-token output limit, `text-delta-v1` streaming and `pipe-v1` cancellation. These are the served worker limits, independent of any larger checkpoint training context.

The production Desktop/worker path, using a temporary library and no production credential, completed an explicitly instructed data-dependent lookup chain. Both initial `auto` and `required` variants completed two separate tool turns, two harmless client-side lookups, and a final answer that used both returned integers. Total: four executed client calls. The native worker remained usable for ordinary follow-up.

Earlier, looser task variants failed: one emitted two calls with `parallel_tool_calls: false` and was safely rejected; another stopped after one lookup. These failures remain part of the aggregate evidence. The successful narrow task does not establish reliable general agent behavior or desktop-client compatibility. Function parameter schemas remain model hints; clients must validate execution arguments and approve effects.

| Artifact | SHA-256 |
| --- | --- |
| Native generator | `6db86e8d3acb0340ee95dd4a3c376b1f5f99dd654feb2876cc080009d897bd99` |
| Native optimized scorer | `eae88ae7e7e62d8b30173689c0155b42d82adab156ee1f141e4b9b93135c135b` |
| Model weights | `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7` |
| Model config | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` |
| Tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

## Packaged app connection checks

Built with the existing `scripts/build_menubar.py` process and Rust 1.98.1 workers. A test copy used bundle identifier `com.starkengine.menubar.validation.desktop20261031`, was re-signed ad hoc after that metadata change, and used its separate derived development Keychain service. The canonical production and development credentials were not accessed. The test's non-secret listening port was 18123.

An in-memory native validation client received the key through Stark's sensitive Copy control into a secure field. It retained the key only in memory and displayed HTTP status categories. Authenticated health returned 200 before and after closing/reopening the package, with no recopy or endpoint change. A port-change conflict at 18124 preserved the active 18123 endpoint and displayed an actionable error. Reopening Stark with saved 18124 while occupied showed an unavailable API and kept the playground usable. Applying 18123 recovered authenticated health with the original client key.

Package version 0.11.0 host executable SHA-256: `64e709a93b54d3ce1be6591d1768812173a1ed5d23fee653da15098c8b6b9479`. A version 0.11.1 package was built for upgrade testing (host SHA-256 `072e29b9224aacfe416d2eac38ad8828696e1845337399d730c6e541017bc09c`), but the Mac locked before its replacement/relaunch could be exercised. Packaged upgrade reuse remains unverified. Developer ID signing/upgrade Keychain access also remains unverified; no release credentials or release infrastructure were used.

Wrong-key/no-model responses, cancellation, model replacement and resource reuse were verified by automated backend fixtures. They were not demonstrated through OpenWork or OpenCode Desktop. No screenshots contain a revealed key.

## Third-party desktop acceptance and remaining blockers

Official releases were inspected on 2026-10-31:

| Client | Exact asset | SHA-256 | Evidence and status |
| --- | --- | --- | --- |
| [OpenWork 0.18.63](https://github.com/different-ai/openwork/releases/tag/v0.18.63) | macOS arm64 zip | `0d4069eb7be1934838b0fc7702d008ba43b1f209fa1b9eb169a22fb34b30d5b3` | Launched; provider UI and bundled adapter/auth implementation inspected. No authenticated model workflow completed. |
| [OpenCode Desktop 2.2.5](https://github.com/anomalyco/opencode/releases/tag/v2.2.5) | macOS arm64 DMG | `aecc7365fe041798d6fdeee6e82c5aad020e2e0282f9658091bd3255db4789d5` | Launched; provider UI inspected independently. No authenticated model workflow completed. |

Asset digests matched official release API digests. OpenWork's bundled `server/dist/engine-v2-preview.js` maps `@ai-sdk/openai-compatible` to the OpenCode compatible adapter and requires an explicit HTTP(S) base URL. Its `managed-opencode-v2.js` defaults model tool capability to true unless configured otherwise. Stark therefore implements actual function definitions/history/results and SSE tool deltas; clients must use the compatible Chat Completions adapter rather than the Responses adapter. Unknown meaningful request fields still fail explicitly.

The exact OpenWork blocker is credential storage: bundled `server/dist/opencode-v2-local-auth.js` reads `{type: api, key: ...}` from OpenCode's ordinary `auth.json`, and `engine-v2-preview.js` injects that key into the v2 provider configuration. No secure in-memory custom-provider credential path was found. Its Connect catalogue had no custom-provider entry. The [official OpenCode provider documentation](https://opencode.ai/docs/providers/#custom-provider) likewise describes configuring custom providers separately from credentials; ordinary [credential storage](https://opencode.ai/docs/providers/#credentials) is `auth.json`. OpenCode Desktop's provider search also had no custom entry in the inspected UI.

Stark's real key was not written to those files, process arguments or environment variables. A secure client credential path is needed before validating either desktop workflow under this task's secret-storage constraint. Disabling Stark authentication, relaxing loopback/origin rules, or storing the key in plaintext was not used as a workaround.

Consequently, **no OpenWork/model or OpenCode Desktop/model combination is qualified end to end**. Streaming chat/follow-up, real desktop tool execution/result return/multiple turns, desktop reconnection, and desktop error/cancellation/replacement scenarios remain blocked at secure client configuration. Native/backend evidence above must not be presented as those acceptance results. Desktop UI work also requires the Mac to be unlocked. Goose was researched only and was not tested.
