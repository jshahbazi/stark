# OpenWork Lily bounded workflow acceptance

Date: 2026-11-01, America/New_York. This records successful runs after the [initial model/tool work](2026-11-01-openwork-model-tools.md) and [schema follow-up](2026-11-01-openwork-lily-followup.md). Those earlier source-bound results and blocked states are preserved.

## First complete workflow combination

| Component | Identity |
| --- | --- |
| Stark | Development version `0.0.0`, build `20261102004738932009`, installed `/Applications/Stark.app` |
| Runtime source | `d50fa5d6b71889f246e9cc097f78d6ebfa928ee6` on `codex/openwork-model-discovery` |
| Signing | Developer ID Team `S5BT2PRN5B`, hardened runtime and secure timestamp; no validation overrides; not notarized |
| Connector | `stark-openwork` `a5534db0526e08c9b9b4df12825c9da7117cae86`, normal bundled connector |
| Client | Unmodified official OpenWork 0.18.63 / bundled OpenCode 2.0.26, Team `F5DJWB4CCV` |
| Model | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` |
| Local public model ID | `qwen38-flash-next-lily-q4-xl-0f94842e` (this registered local import) |
| Precision | Lily q4-xl: Q4 GDN, Q8 attention/shared expert/head/embedding |
| Declared limits | Context 262,144 tokens; maximum output 4,096 tokens |

For that first signed build, host/generation/scoring SHA-256 values were respectively `51bb3b2d75b302358991dd8443c4731c0ee5e791ed0247fd73e5ef2ce9d7b136`, `d9085a9a4a547d4f22216e97a71b3913d82fb18dab635a111606e6fa78fdfb40`, and `01497d0f7e1a773fc25cb24fe7308bf7ca2c3292e8a8917c91b7d787ce8a6716`. Deep strict signature verification passed after the real runs. Native workers were reused through the existing package build/sign/install process; no Rust source or distribution infrastructure changed. The final delivered build and its repeat results are recorded below.

The freshly stream-verified installed model weight aggregate remains `2e34f6d7f80b3f5c9de8e3bec9f2a79cbd6a3518ee7f093edee5014a085ea9c2`; its 55 shards total 106,905,155,192 bytes. The exact config/tokenizer/template hashes and aggregate convention are in the initial record. No second model was loaded for a comparison or benchmark.

## Connection, restart and model identity

The normal Stark Reconnect action migrated the old development broker metadata to the signed production namespace. OpenWork was fully quit/reopened, and Lily appeared in its model picker. The previous record separately establishes Lily-to-JevK5 discovery refresh and template capability reporting.

For the restart check, Stark was quit through its normal application action and reopened. The same Lily model was manually run again; OpenWork remained running. No Connect/Reconnect action, provider configuration edit, port change, key copy or key regeneration was used between the successful runtime workflows. The base URL remained `http://127.0.0.1:8000/v1`; the model ID remained unchanged.

An attributes-only Keychain query after the relaunch returned OSStatus 0, with both creation and modification predating the build cutoff `2026-11-02T00:47:38Z`. It prohibited authentication UI and requested no credential data. Together with the normal signed application/API operation, this verifies production-identifier credential reuse across the local signed package changes and relaunch. It does not qualify a notarized release upgrade.

## Official-client runtime workflow

Automated submission used OpenWork's existing native loopback session/prompt/event API, running its official OpenCode engine and normal Stark connector. The diagnostic read OpenWork's own local server token into RAM only; it supplied no Stark credential and changed no connection/authentication path. Stark returned tool requests; OpenWork executed them.

The strict confirmation run passed all of these checks:

1. Streaming chat produced a visible-text marker and one completed assistant.
2. A follow-up in the same session produced a second, fresh completed assistant using prior history. A previous succeeded outcome alone could not satisfy the check.
3. A separate bounded task read a synthetic starting file, then the second file named in its result. There were exactly two completed `read` parts, two assistant tool turns and two unique call IDs, in the expected path order.
4. The second tool result contained a fresh synthetic marker absent from the task prompt. The final assistant's `type: text` content contained that marker. Reasoning parts could not satisfy this assertion.
5. Each native session's selected provider/model matched Stark/Lily. The loaded local model/build and model hashes were independently correlated.
6. All disposable sessions and the two read fixtures were removed. A final query found zero sessions bearing the validation title.

No permission event occurred in these runs: the client permitted the bounded workspace reads automatically. Human approval-dialog behavior is not claimed. This is model task evidence for one two-read workflow, separately from parser correctness. It does not establish reliable completion of arbitrary agent tasks or qualify other weights sharing the template.

## Timings and accounting

These are observational timings from prompt submission to the first native OpenCode **text delta**, not UI paint time or engine-only measurements. Token counts below are cumulative within each session; the tool task uses its own session.

| Strict confirmation turn | First text | Terminal completion | Cumulative input/output tokens | Text deltas |
| --- | --- | --- | --- | --- |
| Chat | 8.478 s | 8.584 s | 18,956 / 9 | 2 |
| Follow-up | 8.838 s | 8.924 s | 37,949 / 18 | 2 |
| Two-read task, three assistant messages | 30.021 s | 30.304 s | 57,446 / 126 | 4 |

The tool task also observed eight tool events. An earlier valid post-restart run completed chat/follow-up and the two-read task in 8.723 / 9.254 / 31.524 seconds; its final-text assertion was subsequently tightened for the strict confirmation above. No latency improvement is claimed. OpenWork supplied roughly 19,000 input tokens for a short chat, consistent with substantially more input processing than a small Playground turn. This run did not isolate setup, prefill and client overhead as separate causes of the delay. Tool-task first text depends on whether the model emits a preamble; a valid earlier task withheld visible text until its final answer.

## Direct GUI checks

Chat and a follow-up were also submitted using OpenWork's actual composer and Run task button. After explicit selection of Stark/Lily inside that session, two fresh completed assistant messages had matching nested `model.providerID`/`model.id` and matching `type: text` markers. An initial control reply used OpenWork Auto and is excluded from the local-model result. The GUI session was removed afterward. No screenshots or conversation contents were retained.

## Cancellation, authentication and port checks

The cancellation run observed a real text-delta event before calling OpenWork's native interrupt operation. The client became interrupted and idle in 0.015 seconds. The same native worker process identity was preserved before/after interruption and after a fresh recovery turn. Recovery streamed successfully, used the correct session model, returned its fresh marker, and completed in 9.739 seconds (first text 9.507 seconds). Its usage was 18,960 input / 15 output tokens; there were no tool events. All of its own sessions were removed.

A deliberately invalid test credential received HTTP 401 `invalid_api_key` from the installed endpoint. A second LocalAPI bind to its actually occupied port 8000 raised the expected address-in-use error; the original listener still required authentication afterward. The Swift/controller tests cover the actionable port message, stable saved setting and prior-listener preservation. This is separate from manually exercising the occupied-port GUI on the Lily build.

## Final delivered build and lifecycle repeat

The final installed package is development version `0.0.0`, build `20261102015737392158`, runtime source `c4d95213f5671bc862be0e90df380632af29ba83`. It retains the same connector/client/model/precision combination above. The host is Apple M5 Max with 128 GiB unified memory, macOS 26.7.1. Signing identity, hardened runtime, secure timestamp and absence of validation overrides remain unchanged. Installed executable hashes match its signed receipt:

| Executable | SHA-256 |
| --- | --- |
| Host | `abda2f23815c6659dbcbc4fa0fc82325579365d8db5d2a7ac34d50c402d724cf` |
| Generation worker | `f39ff75827573aaae66c74ce896fbf5dcf0158dfa9e85bb48cac0c84bdec1a0d` |
| Scoring worker | `a77b92438f564fa5d11e3b38b05bc28eceffd1145e3e048463473206d51012f8` |

The live occupied-port startup check used a bound, non-listening loopback socket; it never accepted or read requests. Stark showed the exact port-8000 conflict, an unavailable base URL, the unchanged saved port and Apply Port/Retry API controls. Model management still loaded Lily and reported it ready while the API remained unavailable. Releasing the socket and choosing Retry API restored `http://127.0.0.1:8000/v1` with the previous authorization and no credential change. Stopping the model also caused OpenWork to display its unavailable-model/Retry state. This GUI matrix ran on the preceding build; the final change did not alter port or discovery code.

The first real active-model replacement terminated safely but incorrectly surfaced `generation_failed`. The final runtime fixes the classification: changed epoch/worker/model ownership produces `503 model_changed` for both returned and raised old-worker results, including the streaming cancellation path. A genuine same-model worker fault still closes that worker and reports `generation_failed`; explicit cancellation/deadline retain their own categories. A speculative predicate relaxation was rejected during review and is absent from the delivered runtime.

The installed repeat observed a real Lily text delta while the request remained active. The UI disabled Run for other models; the explicit sequence was Stop Lily, then Run JevK5. The old task failed with `model_changed`, zero tool parts/events and no successful completion. Its own session was removed, and Lily was restored. This distinguishes safe model replacement from a backend fault without weakening ownership fencing.

After the deliberate availability failures, OpenWork's frontend retained its unavailable-model state. Its Retry action restored Ready for new tasks; the picker then showed Lily as the current Stark model. This was explicit recovery, not an automatic frontend refresh claim. The final state has Lily running, the normal endpoint restored, the test sessions/fixtures removed, and model-management/connection/playground windows closed.

The complete native workflow was then repeated on the final installed build. All three turns matched Stark/Lily, fresh assistant checks and visible-text markers passed, and the separate task again had exactly two completed reads in two tool turns with two unique IDs, correct paths, the second result marker and final visible result use. All sessions and fixtures were removed.

| Final-build repeat | First text | Terminal completion | Cumulative input/output tokens |
| --- | --- | --- | --- |
| Chat | 20.449 s | 20.590 s | 18,956 / 9 |
| Follow-up | 11.207 s | 11.399 s | 37,949 / 18 |
| Two-read task | 13.147 s | 40.248 s | 57,468 / 147 |

The first turn overlapped remaining local CPU test activity; these are not controlled performance comparisons. The slow response concern remains: across successful runs, short-chat first text took about 8–20 seconds for approximately 19,000 input tokens. No latency fix is claimed.

Deep strict signature verification and installed receipt hash checks passed again. The attributes-only production Keychain query returned OSStatus 0; creation and modification both predated the final build cutoff `2026-11-02T01:57:37Z`, with no credential data requested. One Stark listener on port 8000 and one native model worker remained.

## Verification, exclusions and limits

- Full locked offline converter suite on runtime source `d50fa5d6`: **5,533 passed, 89 skipped**, 188.96 seconds, uv-managed Python 3.13.13.
- Final runtime `c4d95213` plus lifecycle assertion update `6e855023`: **5,548 passed, 89 skipped**, 210.01 seconds. Independent complete Desktop/image suites passed 249 tests with one skip; a separate 55-case lifecycle check passed through resource cleanup/recovery assertions. Nine earlier full-suite failures were old expected status/code assertions and were updated to the deliberate new contract; the genuine backend-failure assertion remains `503 generation_failed`.
- Repository policy passed with its locked Python 3.12.13 script; Ruff, formatting and whitespace checks passed.
- Independent parser review repeated 253 tests and separately verified both XML parsers emit zero executable deltas when a typed first call is followed by an unsupported-type argument.
- Swift/native credential, connection installer, controller, window and build checks remain the passing source-bound results in the earlier records. Credential failure/concurrency/regeneration tests use isolated storage.
- The hosted Build workflow passed at `a723b1d4`, before the final lifecycle runtime/test changes. The final delivery head requires its own hosted Build result; that status is separate from the passing local checks above. No PR was opened.
- An initial 90-second observation lacked the later terminal-outcome handling and is inconclusive. A stale-outcome follow-up observation is excluded; the strict fresh-assistant checks replaced it. A history-polling cancellation probe did not observe an active assistant and is excluded; the successful cancellation run used actual streaming events. These changes affect validation tooling, not the installed runtime.
- Live active-model replacement, occupied-port recovery and unavailable-model GUI checks completed as described above. Worker-fault, overflow, malformed/truncated call and queue-fencing coverage remains the source-bound automated evidence; no live GPU/worker-fault injection is claimed.
- Goose/OpenCode Desktop were not rerun against the XML path. Historical Goose CLI/Qwen3-4B evidence remains separate. Other admitted templates have protocol support and synthetic rendering/parser evidence, not this model-bound task qualification.
- Notarization and notarized release-upgrade behavior remain unverified. The installed package is a signed development build.

Only versions, hashes, counts, booleans, timings and sanitized failure categories are retained. No Stark key was written to plaintext, arguments, environment, screenshots or validation artifacts. No real prompts/responses, token vectors or raw HTTP/SSE dumps are retained in this record.
