# OpenWork Lily validation follow-up

Date: 2026-11-01 (America/New_York). This continues the [initial model/tool record](2026-11-01-openwork-model-tools.md) without changing its source-bound results. Real Lily tool execution remains unqualified: the Mac locked again while resolving a live schema compatibility failure, and the newest installed package has not yet been launched.

## Observed desktop results

Using unmodified official OpenWork 0.18.63 / OpenCode 2.0.26:

- The signed Stark build launched without an unsolicited Connection Details or Playground window. The main modal displayed the worker GPU-memory meter.
- Loading Lily exposed its 262,144-token context, 4,096-token output limit and tool capability in Connection Details.
- The normal **Reconnect** action migrated the existing development broker metadata to the signed application's production broker namespace. No diagnostic credential helper or API-key copying was used. Stark instructed a full OpenWork quit/reopen; afterward Lily appeared under Stark and was selectable.
- Switching the running model from Lily to JevK5 refreshed OpenWork's picker without restarting OpenWork. The old saved Lily selection became unavailable, and the Stark group offered JevK5. Selecting JevK5 worked; its loaded-template tool capability appeared in Stark. This establishes discovery/capability behavior, not task competence.
- A metadata-only Keychain query succeeded with OSStatus 0. The production bearer item's creation and modification both predated the first signed build's 2026-11-01T23:28:30Z cutoff. The query requested attributes only, prohibited authentication UI and performed no writes; no credential data, fingerprints, dates or attribute dictionaries were retained. The existing production credential was reused by that signed app without regeneration.

The first native OpenWork chat request failed before inference with `tool_schema_unsupported`. After adding literal-constrained parameter support, a second request still failed with the sanitized form `untyped_or_composed_parameter`. Both disposable sessions were deleted; both had zero tool calls and no streamed text. The exact client schema was not dumped or retained. Public OpenCode metadata routes returned 404 and did not identify the field.

## Follow-up implementation and review

- `dc3bd10535ca7562c0b650cbbf9387483a0d20df` derives XML decoding types from bounded JSON `enum`/`const` schema literals and provides fixed unsupported-form categories. The [official OpenCode 2.0.26 normalizer](https://github.com/anomalyco/opencode/blob/v2.0.26/packages/ai/src/protocols/utils/tool-schema.ts) documents enum-only schemas and leaves its OpenAI projection unchanged. This source establishes a compatibility requirement; the first follow-up did not resolve the observed request by itself.
- `d50fa5d6b71889f246e9cc097f78d6ebfa928ee6` preserves unused empty/valid annotation-only parameter schemas unchanged. An opaque private sentinel prevents decoding such an argument. Actual use fails the entire completion with `502 tool_argument_type_unsupported`; no generated type is guessed and no earlier call from that response is released. Malformed schemas, invalid explicit `type: null`, unsupported assertions/composition, unresolved references and bounds retain their preflight errors.
- Independent review found no remaining code blocker. Its RAM-only real HTTP/SSE checks for both XML parsers confirmed that a valid first call followed by an unknown-type argument emits zero tool-call deltas and the explicit error. These safety checks do not establish that the newest package resolves the actual OpenWork schema.

## Verification and installed package

| Check | Result |
| --- | --- |
| Full locked offline converter suite on `d50fa5d6` | 5,533 passed, 89 skipped in 188.96 seconds |
| Final parser suite, independently repeated | 253 passed |
| Intermediate parser/request/Desktop suite on `dc3bd105` | 498 passed, 1 skipped |
| Independent atomic HTTP/SSE rejection | Both XML parsers passed, zero executable deltas |
| Repository policy / Ruff / formatting / whitespace | Passed |
| Signed package and installed executable hashes | Passed |

The latest package is installed at `/Applications/Stark.app`: development version `0.0.0`, build `20261102004738932009`, source `d50fa5d6b71889f246e9cc097f78d6ebfa928ee6`, connector `a5534db0526e08c9b9b4df12825c9da7117cae86`. The numeric build timestamp is UTC; this run's local date remains November 1. Developer ID Team `S5BT2PRN5B`, hardened runtime, secure timestamp, 37 signed runtime images, no validation overrides, not notarized. Native workers were reused and signed; no Rust source changed.

| Installed executable | SHA-256 |
| --- | --- |
| Host | `51bb3b2d75b302358991dd8443c4731c0ee5e791ed0247fd73e5ef2ce9d7b136` |
| Generation worker | `d9085a9a4a547d4f22216e97a71b3913d82fb18dab635a111606e6fa78fdfb40` |
| Scoring worker | `01497d0f7e1a773fc25cb24fe7308bf7ca2c3292e8a8917c91b7d787ce8a6716` |

A validation-only invocation of the earlier installed private Python omitted `-B` and added 11 unsealed bytecode files. Only those generated extras were removed. Independent comparison then found identical installed/staged file sets and passing deep strict signatures. Normal Swift backend launches already use `-I -B`. The newest package was installed from its clean signed staging copy; the erroneous smoke invocation was not repeated.

## Remaining acceptance

The pending unlock is required to launch the newest installed app, reload Lily and retry the native OpenWork workflow. Still required: streaming chat/follow-up, two client-side read tool turns, returned results used in a final answer, and reconnect after closing/reopening Stark with the existing endpoint and credential. Actual interruption/model-replacement task checks and an independent desktop client remain unverified for the XML path. Existing automated lifecycle tests and historical Qwen3-4B client evidence remain separate results.

The model identity, quantization and freshly verified installed artifact hashes remain those in the initial record: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, mixed Q4/Q8 Lily q4-xl. No successful Lily agent task is claimed. No Stark credential was copied into a plaintext file, argument, environment variable, screenshot or validation artifact.
