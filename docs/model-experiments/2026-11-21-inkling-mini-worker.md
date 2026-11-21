# 20261121-INKLING-13 · Original Inkling 2 Mini in the native worker

## Identity and scope

| Field | Value |
|---|---|
| Registered and executed | November 21, 2026, Codex |
| Source | `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` |
| Target SHA-256 | `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Draft SHA-256 | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |
| Host | Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.2 |
| Serial-prefill worker | `stark-generate`, Rust 1.98.1 release with locked dependencies, SHA-256 `2f885e270966ff186ddbdb1f5a8e04218dec2679748965351d0368c967ef650f` |
| Grouped-prefill worker | Same build profile, SHA-256 `2579436a9389bd97c5e19287a92bf745aa14105932f0f416770e18f7b4367f37` |
| Evidence scope | Local desktop backend and native worker over the original fixed-three-pass artifact |

The fixed catalog row prepares the original target and draft without requantizing their tensors. Stark verifies the publisher's packed files and metadata, loads the verified weight descriptors into Metal, and runs the original vision tower by default. The desktop uses the source-bound chat template with the native renderer. The command-line preparer also published the pinned vision-language artifact from an owner-private offline stage cloned from the verified snapshot. Raw request and response content stays private.

## Worker checks

The normal desktop backend reached locally tested readiness in 20.23 seconds on one fresh check. This includes weight verification, upload and three visible-response checks, with the original draft and vision tower resident. The worker streamed a chat completion with a separate reasoning field. Its 20-token input produced 83 completion tokens and reached the publisher's end-of-sampling token.

| Control | Result |
|---|---|
| Draft-off and one draft enabled, greedy | Same terminal assistant message and finish reason |
| Draft-off and one draft enabled, seeded temperature 0.7 | Same terminal assistant message and finish reason |
| Image following draft-enabled text | The 256×256 PNG check produced the expected visible color. It expanded the publisher placeholder to 49 image rows and consumed 74 input tokens. The staged file was cleared. |
| Function tool | The OpenAI-compatible API accepted a complete named call, then rendered and answered the matching tool result on the original template. |

An earlier 48-output-token smoke finished in the reasoning frame, leaving an empty visible reply. The subsequent control used a 192-token cap. The installed-image smoke now checks the decoded visible field and supplies the publisher's reasoning-effort option for Inkling. The individual controls establish operation of the named local paths. Image and tool quality, DSpark speed, full-target numerical parity and 4K prefill remain separate gates.

The grouped-prefill worker repeated all four controls with the corrected DSpark fold. It reached readiness in 17.85 seconds on one fresh check. The streamed 20-token text request produced 93 completion tokens, and the image control consumed the same 74 input tokens. On a 1,031-token text request, reusing 1,024 tokens reproduced all 24 output tokens and moved prefill time from 8.28 seconds to 0.15 seconds. Cancelling a request cleared the snapshot. The following request reused zero tokens and reproduced the same output.

## Cache-cold 4K worker screening

The serial and grouped workers were screened in separate sessions on one 4K request each for greedy and sampled generation. The grouped worker clears the decode ceilings in both modes. The prefill rate improves in both modes while the cache-cold target remains open. [Worker benchmark](../benchmarks/2026-11-21-inkling-mini-native-worker.md) · [Paired graph controls](2026-11-21-inkling-mini-full-target.md)
