# OpenWork and signed Keychain upgrade validation — 2026-10-31

## Scope

This continues [the initial backend validation](DESKTOP_AGENT_BACKEND_2026_10_31.md). Evidence is bound to the packaged builds below. Only aggregate results, versions and hashes are retained. The isolated OpenWork profile held synthetic tasks and was removed after collecting aggregates.

The initial signed packages use the backend at `730a4c03`; package C includes the `store` compatibility change on top of `a395f68b`. Packages A–C serve a 16,384-token context and a 4,096-token backend output limit. Package D includes the authorized model-capacity/default changes through `d4d05d8c`, the unknown-capacity safeguards and `store` compatibility changes. Package E additionally admits the exact upstream Qwen3-4B template and normalizes tool-only assistant content for that template. D and E serve these models' declared 40,960-token context and a 4,096-token output limit.

## Secure upgrade result

The existing Developer ID identity and existing signing helpers signed all 37 embedded runtime images and each host, with hardened runtime and secure timestamps. No release was published or notarized and no release infrastructure was added. All tests use bundle identifier `com.starkengine.menubar.validation.signed20261031`, its separate derived development Keychain service and port 18133. Production and canonical development credentials were not accessed.

| Package | Host SHA-256 | Signed generator SHA-256 |
| --- | --- | --- |
| A, 0.11.3 | `2faa4d2bb35552eadd2c48972ee820537bca66b7f5589c5b2debc4ba80595850` | `1b484b23e6e11ee14dd90437c8db69f545dce6e3cd41e77a7aa3e6e30fe9ec9c` |
| B, 0.11.4 | `fd7b49ed3babcf37a2056bdcdca8b1030c0986e4e8668c54410821d28d8efdf8` | `603e79dda2e1c6d3946c709cfb47ac686b386ca7f750b4ada8efa146346f46da` |
| C, 0.11.5 | `99a7df3f14c3dc45c89cfe8a9aa9d1d5c33602ace40ad1f4ecf8bbf93f1e2c8a` | `f08f5515f6102c7f9e1f78826f8832a7e1927ea4909e8a72c4aa4eab819935d3` |
| D, 0.11.6 | `45be492e8512bffe63bb6d18fb3a48c2677afef2b31d134ba8759109c3be889f` | `72da5b10c6f4902e19d974379c4f24933529b67c176cf130630ced97b381750e` |
| E, 0.11.7 | `941eb51963b40a0e04dee65667656ec299a779fc9542457a53045bdf721e6141` | `584baf70cb9b8100949966ca6fa2778ac8ac7ae3e2fe1688f409a9736c435847` |

Package A created its credential through the production Swift storage implementation. A read-only signed validation process loaded it once and retained it in memory. Authenticated health was 200 and a deliberately wrong key returned 401. After closing A, the original endpoint was unavailable. The existing installer helper atomically replaced A with B at the same location; B automatically read the original Keychain item and served the saved endpoint. The validation process, still using its original in-memory key with one Keychain read, received 200 again and 401 for the wrong key. The same check passed across B to C after the protocol fix.

The same original in-memory probe credential also authenticated after C to D and D to E upgrades, with the probe's Keychain read count still one. D was closed and reopened at the same installed path; OpenWork received a successful fresh assistant reply after the same model was reloaded, without restarting its sidecar or changing its provider, key loader or endpoint.

The packages and reader have matching Developer ID designated requirements. An independent test also verified changed-code and relocated signed readers against an app-created disposable item. Ad-hoc signatures remain specific to a build and do not establish the same upgrade identity; no ACL was weakened to make them equivalent. See Apple's [signing requirements](https://developer.apple.com/documentation/technotes/tn3127-inside-code-signing-requirements) and [Mac Keychains](https://developer.apple.com/documentation/technotes/tn3137-on-mac-keychains).

## Actual client and credential path

Client: official OpenWork 0.18.63 macOS arm64 release, asset SHA-256 `0d4069eb7be1934838b0fc7702d008ba43b1f209fa1b9eb169a22fb34b30d5b3`. Its bundled OpenCode is 2.0.26, source revision `9b4ec5714d481559990db0a816d5dec19541a814`. The desktop was launched with its supported `--blank-slate` flag, isolating HOME/XDG state and client session storage. The official application binary is unchanged.

An external validation-only plugin uses the official v2 `http.request` hook. It accepts only provider `stark` and the exact POST URL `http://127.0.0.1:18133/v1/chat/completions`. It changes only the Authorization header. It does not modify prompts, tools, models, request bodies or responses. The actual key is loaded lazily through a signed Swift child process and cached in the plugin module's memory. The child's stdout is an anonymous pipe/socketpair captured directly in memory. The public configuration contains a deliberately invalid sentinel, never the real credential. Missing or failed credential loading fails closed. Plugin source SHA-256: `075fc9f23f8a0a25cd8121ca2c184fa0ec38ae68eb084d8561ea0846b09340c4`.

The reader is fixed to the disposable service, never creates or rotates a key, checks its own Developer ID identity, pins the official native OpenCode executable's full SHA-256 and live CodeDirectory hash, and verifies the executing parent again after its read. It rejects terminal/file stdout, named/network sockets, unexpected arguments and unapproved parent processes. Three negative gate checks emitted zero credential bytes. It emits exactly 64 credential bytes only through the approved private child channel. The plugin bounds captured output and time, uses fixed errors without child buffers, and caches successful loading for the plugin module's lifetime.

Reader source SHA-256: `91fe1123859cc010a562c0986d576725cb0f3b5276449c7090a3537b575893e2`. Signed reader SHA-256: `a5cdd3d366cb0b539dbda249921b64100dbc1f3cca7d86bd2f684dd2f6efc087`. The helper opens the existing login Keychain using the OS account home and a query-local search list, so OpenWork's isolated HOME does not change the lookup. It changes no global Keychain settings. This is an external validation integration, not OpenWork's built-in Connect UI or a shipped Stark helper.

Public client settings for package D: compatible Chat Completions provider, HTTP transport, model `qwen3-dense`, context 40,960 and output 4,096. Earlier 16,384-token attempts required reduced client limits and disabled compaction, then still exceeded the backend context. Those temporary compaction overrides were removed for D; the client catalog and Connection Details both reported the real limits.

Model: Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`, unquantized BF16/F32 artifact, 226 tensors. Weights SHA-256 `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7`; config `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e`; tokenizer `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`; template `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` with parser `qwen3-hermes-v1`.

Second model: `mlx-community/Qwen3-4B-4bit`, revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, converted with the existing `hf-generic` metadata profile; local public ID `qwen3-4b-4bit`. MLX affine 4-bit, group size 64; 580 tensors, 2,263,417,416 weight bytes. Converted weights SHA-256 `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8`; config `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f`; tokenizer `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`; template `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5`. The independently fetched upstream template matched the fixture and converted artifact exactly. Native loading passed in the signed app; conversion metadata alone was not treated as inference evidence.

## Failures and protocol change

The first real desktop request returned `unsupported_parameter`. A temporary diagnostic collected field names and status categories only; it forwarded authentication unchanged, retained no bodies/headers, and was removed before acceptance. The actual SDK sent `store: false`, confirmed by the [pinned request builder](https://github.com/anomalyco/opencode/blob/9b4ec5714d481559990db0a816d5dec19541a814/packages/ai/src/protocols/openai-chat.ts). Stark now explicitly accepts false/null/default no-storage behavior and rejects true or non-boolean aliases. Full converter result for this change: **5,089 passed, 88 skipped**. The signed package C contains that exact parser source; an independent reviewer checked it byte-for-byte.

An earlier FIFO-only credential experiment was abandoned. OpenWork resolves file references before rendering its native config; a direct reference outside its allowed directory was rejected. Two-stage public indirection avoided serializing the key, but immediate repeated FIFO consumers could receive an empty value, causing 401. The final client integration uses the supported HTTP credential hook and private child pipe instead. No actual key was saved to regular configuration files by these experiments.

Scoped native scans reported zero literal credential matches in the enumerated isolated profile regular files (for example 11,012 files, 193,560,578 bytes). These scans skip nonregular files, limit entries/depth/bytes/time and do not prove absence from encoded forms, prior deleted files, or other locations. Source-path review and sentinel experiments provide separate evidence about credential flow. Final post-workflow scan and cleanup results belong below.

## Desktop workflow result

On macOS 26.7.1, the unchanged official OpenWork desktop release completed the following through the direct Stark endpoint and the external credential plugin:

| Build/model | Observed result |
| --- | --- |
| D / Qwen3-0.6B BF16 | Streaming chat and a follow-up passed. The first main request used 16,917 input tokens, beyond the old cap. Both requested assistant markers were present; two completed assistant messages, 33,364 aggregate input and 48 output tokens. |
| D / Qwen3-0.6B BF16 | A model-selected `read` ran in OpenWork and returned the expected first fixture value and dependent filename. The model stopped prematurely after one call. A stricter retry was cancelled; OpenWork reported `interrupted`/`aborted` and Stark returned to Ready. |
| D / Qwen3-0.6B BF16 | After closing/reopening Stark and reloading the same model, a fresh OpenWork request completed with the expected assistant marker: 16,915 input, 9 output tokens. The client sidecar, credential plugin, key and endpoint were unchanged. |
| E / Qwen3-4B affine 4-bit/group 64 | A bounded task completed two sequential model-selected `read` calls and a final text answer in one task, without intervention: one user message, three streamed assistant messages, two `tool-calls` finishes, one `stop`, and a successful task outcome. Aggregate usage across the three model requests: 50,810 input and 202 output tokens. |
| E / same Qwen3-4B artifact | A follow-up chat completed, then Stark was closed/reopened and the identical model reloaded. Another turn in the same OpenWork session completed with the expected assistant marker, without client reconfiguration or sidecar restart. Across the task and both follow-ups: five assistant requests, 84,780 aggregate input and 211 output tokens. These totals are not a single context window. |

The successful 4B task gave the model only the first fixture's path. Its result supplied a randomized second filename. Both distinct calls completed in OpenWork, with local/client execution rather than provider execution. Their results contained the expected different values; the assistant's final arithmetic used both values and ended with the correct sum. It included an arithmetic expression rather than strictly following the requested final string format. No shell, network or write tool ran. The client's normal read permission policy required no additional approval dialog; Stark executed no tool.

An independent in-memory projection of this actual client session confirmed two unique completed tool calls, matching tool-result IDs, nonempty result content, and both results inside separate `<tool_response>` wrappers in the exact pinned model template. It retained no message bodies. This separates history/protocol evidence from model behavior.

The explicit two-call budget matters: an earlier 4B variant read the first file and then reread the second file repeatedly. It was cancelled after 11 completed reads; a manual continuation subsequently used those existing results to produce the correct total. These failures, and the 0.6B premature stop, prevent any claim of reliable general autonomous-agent behavior. The passing bounded task is a narrow workflow qualification, not a broad agent benchmark or quantization/reference-logit parity result.

The actual client also surfaced the useful no-model instruction after restart; a cached original-key health probe still returned 200 and a wrong-key probe 401. After replacing the running 4B model with 0.6B, a subsequent request for the old model failed with HTTP 404 `provider.invalid-request`; OpenWork displayed Stark's instruction to select the running model ID from Connection Details or `/v1/models`. The attempted in-flight replacement replay completed its short response before replacement, so it is not counted as active-request fencing evidence. That behavior remains covered by backend lifecycle fixtures.

For the saved-port startup replay, a disposable loopback listener held port 18133 before E launched. Connection Details reported the occupied port, offered Apply Port/Retry API and showed no available base URL; no fallback endpoint appeared. Model loading and playground inference still worked, with a verified assistant reply. After the test listener closed, Retry API restored the original endpoint. The original cached key again received 200; the wrong key received 401; the probe still had only one Keychain read.

## Final source checks

Product source manifest SHA-256: `4549616dcff19a630850a59d626870631947825db56fa6de06f14b18d339bda8`. This hashes sorted `apps/menu-bar/Sources/*.swift` and `tools/converter/stark_convert/*.py` paths, NUL separators and binary SHA-256 values. All 34 packaged Python modules in E matched the checkout byte-for-byte.

| Check | Result |
| --- | --- |
| Locked offline converter suite, Python 3.13.13 | 5,122 passed, 88 skipped |
| Independently replayed 4B template/history suite | 58 passed |
| Swift controller harness | 111 passed |
| Credential runner, injected/disposable Keychain storage | 2 passed |
| Builder/updater unit suite | 62 passed |
| Repository policy and policy tests, Python 3.12.13 | Passed; 41 tests |
| Ruff check/format and locked dependency check | Passed |
| Pinned nightly Rust formatting | Passed |
| Rust 1.98.1 generator release suite | 65 passed, 2 private benchmark tests ignored |
| Rust 1.98.1 optimized scorer release suites | 21 main, 21 digest replay and 1 probe check passed |
| Separate generator and optimized scorer all-targets release Clippy | Passed with warnings denied |
| Existing package builder and Developer ID signing verification | Passed; 37 runtime images per package |

Unknown-capacity safeguards have focused API, CLI and Swift evidence: report a null/unknown capacity rather than claiming the unsigned 32-bit protocol ceiling as model capacity; size native text state to actual input plus output; reject unknown-capacity image requests consistently. The finite 40,960-token path remains unchanged. Context reporting does not establish full-length performance or nonzero-state Lily growth parity.

## Independent client check

The embedded native CLI from the official [Goose 1.57.0](https://github.com/aaif-goose/goose/releases/tag/v1.57.0) macOS arm64 desktop asset passed an independent compatible-provider chat and tool check against E and the same Qwen3-4B artifact. This tests Goose's native backend/CLI, not its desktop GUI. The arm64 asset SHA-256 is `8c7e0446fcac698841be1274da3cdf57ae09ac6712175b1715c3b4fa39516d3f`; native binary SHA-256 `dcdcfa2ad6fc8050c74d63cfbc145c972734b8f07307ae9c2be1b8a4e3a4dfec`, CodeDirectory hash `91d8a8ee415c5ff0c8860379a52e9bfc0d3064ac`. Strict signature verification passed. Source revision: `30bba95105e59bc3e7df624cf6b4d433690cfe12`.

The official declarative custom-provider route supports `auth.command`. A separate signed read-only helper variant pins only the official native Goose executable, using the same isolated Keychain item and private output checks. Helper source SHA-256 `40f0e1618938ddf061fda848dc19e683670d458a98ebdd0594db2df8ae30e4a0`; signed helper `2d7c139de794372c7330fd153ae97544f7b11a64c7af20ef0be8846deddabb65`. Source, signature and direct-process/stdout-memory credential handling were independently reviewed. No token was passed in environment, command arguments or configuration. The public config has an empty `api_key_env`, fixed loopback base URL, model context 40,960 and the helper command. It used a separate absolute `GOOSE_PATH_ROOT` under the disposable client profile; no default user profile changed.

Both runs supplied synthetic instructions over stdin and captured stdout/stderr only in memory, with no session/profile prompt additions and logging level off. Chat exited 0 and the assistant marker matched. The harmless Developer `tree` task exited 0 with four messages, exactly one tool request, one tool response, and an assistant-only result matching the expected fixture count. Stderr was empty. No raw prompt, output, tool arguments or usage vectors are retained. Per-request Goose usage totals were not collected.

Goose still created a session database and five temporary LLM request-log files despite `--no-session` and `RUST_LOG=off`. They remained inside the isolated profile, were included in the credential scan, and were removed with that profile. Command auth avoids its static-key Keychain/`secrets.yaml` fallback path. See the pinned [provider registry](https://github.com/aaif-goose/goose/blob/30bba95105e59bc3e7df624cf6b4d433690cfe12/crates/goose/src/config/declarative_providers.rs), [command-auth implementation](https://github.com/aaif-goose/goose/blob/30bba95105e59bc3e7df624cf6b4d433690cfe12/crates/goose/src/providers/command_auth.rs) and [isolated path handling](https://github.com/aaif-goose/goose/blob/30bba95105e59bc3e7df624cf6b4d433690cfe12/crates/goose/src/config/paths.rs).

OpenCode Desktop 2.2.5 remains a UI-only inspection from the initial record. No authenticated OpenCode Desktop or Goose Desktop GUI workflow is claimed. The independent native Goose check establishes an additional protocol/client implementation, without treating OpenWork's bundled OpenCode as an independent desktop validation.

## Privacy cleanup and remaining limits

The final completed native scan covered **11,096 regular files and 208,808,191 bytes**, including both client profiles, Goose's logs/database and OpenWork state. It found **zero literal matches** for the original key. Twelve nonregular entries were skipped. The scan bounds and exclusions described above still apply; this does not prove absence from encoded forms, previously deleted files or other locations.

The original model library's two entries were verified after removing only the imported validation model entry; its ignored converted files remain available for reproduction. Both validation apps and all model/client subprocesses were closed. OpenWork's blank-slate shutdown removed the entire disposable profile, including Goose's child profile and its temporary session/request logs; absence was verified. The synthetic Goose fixture, obsolete FIFO and cached probe process were removed/stopped. The exact disposable signed-test Keychain item was deleted after all checks. Production credentials were never read or modified. No credentials, real request/response dumps or client session bodies are included in this record.

Remaining limits: the stock OpenWork Connect UI is not qualified as a secure custom-provider setup; this workflow used a supported external hook and read-only helper. The bounded 4B task does not establish reliable general agent competence. Production-bundle-identifier upgrades, notarization and public distribution remain unverified. Active-request replacement/worker-failure/context-overflow fencing retains the separately described backend test evidence, with no additional desktop claim.
