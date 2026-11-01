# OpenWork model discovery and XML tool protocol validation

Date: 2026-11-01. This record distinguishes protocol checks from real desktop/model qualification. The final installed build has not completed the OpenWork workflow: Computer Use reports that the Mac is locked. No Lily agent-competence claim follows from the parser checks.

## Source and installed build

- Branch: `codex/openwork-model-discovery`, based on `6c396202342d9e90dc37b359b3abeb17f7e453b0`.
- Connection migration: `5477449becdb2cd366cc02e9565b9892d004d9e4`.
- XML parsers: `54056292217a545a92eb57cb5055755ce0b6e615`.
- Shared quotation/thinking guard: `d82bf7f429eb184a32b591b3ea4ec7d7609977bd`.
- Bundled connector: `stark-openwork` commit `a5534db0526e08c9b9b4df12825c9da7117cae86`, branch `codex/model-switch-regressions`. Its runtime files are unchanged; this revision adds regression tests.
- Installed into `/Applications/Stark.app` through the existing staged, signature-checked atomic installer. Installation verified the installed host and both worker hashes against the signed receipt. The previous model worker exited.
- Application version `0.0.0`, development build `20261101232830303823`. This does not consume or supersede a stable release version.
- Developer ID Team `S5BT2PRN5B`, hardened runtime and secure timestamp, 37 signed runtime images, no validation overrides. This local package is not notarized.
- Build manifest marks the worktree dirty because an unrelated untracked research document was preserved. The changed runtime source hashes match the committed source above.

| Installed executable | SHA-256 |
| --- | --- |
| Host | `7ec4bc76e31a0a764f198219321dbc5a346d8e4663d18c3d40d867bdea30d442` |
| Generation worker | `00d8ec58b0f58219c2c01da31b86f8c9e7d704d77ff0eecf785b5ba2b504afc5` |
| Scoring worker | `8c9a010fd4962f768d74af4768d961a79b2d17f5a2d67b1607d9c0b33d51a870` |

The native workers were reused from the installed application and signed in the new package. No Rust source changed; no Rust rebuild or new engine-performance qualification is claimed. Python dependencies remained locked, with uv-managed Python 3.13.13; repository policy used its locked Python 3.12.13 script.

## Reproduced connection failure

The existing OpenWork public connector entry pointed to the production broker namespace, while the previously installed ad-hoc development Stark used the development namespace. Connect reported a different-connector conflict. The loaded Lily model had a known 262,144-token context, so unknown-context filtering was not the cause.

The installer now migrates only the exact reserved v1 connector shape between recognized local Stark application locations and current-user Stark broker paths. Foreign, malformed, duplicate and future-version entries remain conflicts. JSONC comments and unrelated settings remain intact. No former application path is executed or read.

A diagnostic using the production installer implementation repaired only public package/socket/base-URL metadata. The existing application could then authorize OpenWork; after fully quitting and reopening OpenWork, its model picker showed the running Lily model. This diagnostic did not read a Stark key. It establishes the original cause and public-metadata repair, not final installed-app migration acceptance.

The next real OpenWork request failed with `tools_unsupported`: Lily uses XML-style function/parameter blocks, while the earlier parser only handled Hermes JSON. The new build implements that XML path and the distinct typed-scalar variant used by the installed JevK5 template.

## Protocol support

All five registered model templates now select a parser: Qwen3-0.6B and Qwen3-4B use `qwen3-hermes-v1`; Lily and the installed pipenetwork mixed-4/8-bit artifact share `qwen38-xml-v1`; JevK5 uses `qwen35-xml-v1`. Exact template hashes gate support. This does not qualify every checkpoint or arbitrary family template for reliable agent tasks.

CPU-only checks rendered complete synthetic history through each actual installed Lily/JevK5 template using `load_template` and `template_messages`, then parsed only the assistant call slice. Both passed 8/8 exact type/value round-trips: integer-looking string, whitespace/newline string, integer, float, boolean, null, object and array. Both preserved tool-result and follow-up markers. No second model was loaded and no GPU inference was used for these protocol checks.

The checks cover effective named/none tool choices, complete history and result ordering, scalar/structured arguments, lossless strings, ambiguous unions, bounded local schema references, unsupported-schema preflight, malformed/truncated/unknown calls, stable per-response call IDs, HTTP buffered/stream equivalence, cancellation, context overflow, worker failure and model-change fencing. Tool execution remains in the client.

Independent review identified and resolved three issues before delivery: exponential reference expansion, unsupported root-schema acceptance, and the old Hermes path accepting calls inside unclosed quotations/thinking. Resolution uses a shared work budget with memoization, explicit early schema errors, and a common prefix guard. The original four unsafe Hermes cases now return `502 invalid_tool_call`.

## Checks

| Check | Result |
| --- | --- |
| Full locked, offline converter suite on final source | 5,467 passed, 89 skipped in 183.97 seconds |
| Focused parser/request/Desktop suite before final common-prefix change | 454 passed, 1 skipped |
| Final parser suite | 190 passed |
| Independent final prefix/stream regressions | 15 passed |
| Connector installer native harness | 30 scenarios passed |
| Menu bar build tests | 37 passed |
| Menu bar controller harness | 116 checks passed |
| Menu bar window and desktop-connection harnesses | Both passed |
| API credential harness | Both tests passed, using isolated storage/disposable service |
| Separate connector Node suite | 13 passed on Node 26.13.0 |
| Repository policy, Ruff, format, whitespace | Passed |
| Installed signatures and signed-receipt hashes | Passed |
| Installed private Python/parser smoke check | All three parser paths passed; packaged module bytes match committed source |

Credential tests cover persistence, concurrent creation, failure/recovery and regeneration without accessing the installed credential. The installer accesses no Keychain item. Production credential reuse after this package's relaunch/upgrade remains unverified until the app can be launched on the unlocked Mac. No credential was rotated or placed in a plaintext file by this work.

## Desktop/model evidence and remaining acceptance

The installed unmodified official client is OpenWork 0.18.63 with bundled OpenCode 2.0.26, macOS Team `F5DJWB4CCV`. The [official release page](https://github.com/different-ai/openwork/releases/tag/v0.18.63) identifies this desktop release. Prior helperless Qwen3-4B evidence remains in the [managed connection record](2026-10-31-openwork-managed-connection.md); it does not qualify the new XML model path.

Target model: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, Lily q4-xl mixed precision: Q4 GDN and Q8 attention/shared expert/head/embedding. Its 55 installed safetensors total 106,905,155,192 bytes. The actual installed shards were stream-hashed, without loading another model or using the GPU.

| Model artifact | SHA-256 |
| --- | --- |
| Named-shard weight aggregate | `2e34f6d7f80b3f5c9de8e3bec9f2a79cbd6a3518ee7f093edee5014a085ea9c2` |
| Converted config | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| Tokenizer | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| Tokenizer configuration | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |
| Raw UTF-8 template | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |

The weight aggregate hashes lexicographically sorted shard basenames, each followed by NUL and the raw 32-byte per-file SHA-256. No weight contents, token vectors, real prompts/responses, credentials or raw HTTP/SSE dumps are retained in this record.

Still required on the final installed build:

1. Launch Stark, load Lily in Chat mode and use its normal Connect action to migrate the existing development broker metadata to the signed application's production broker namespace. Fully quit and reopen OpenWork.
2. Verify streaming chat, a follow-up, two sequential read-only client tool executions, returned tool results and a final assistant response using those results.
3. Close and reopen Stark; verify the existing endpoint/credential reconnect without copying or regenerating a key.
4. Verify actual model switching and useful desktop no-model/cancellation/replacement handling on this build. Automated lifecycle checks passed; they are separate evidence.

Computer Use repeatedly reported the Mac locked during this final stage, and an unlock request remains pending. The signed app is installed but was not launched by the installer. The new XML path has no successful real-client tool task yet. Goose/OpenCode Desktop were not rerun against it; no independent desktop compatibility claim is made for Lily, JevK5 or the mixed Qwen3.8 checkpoint. Production upgrade-access and notarized distribution behavior remain unverified.
