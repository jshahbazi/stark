# 20261110-TOOL-QUALITY-02: restored Qwen3 0.6B artifact replay

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | 20261110-TOOL-QUALITY-02 / 2026-11-10 / 2026-11-10 / Codex |
| Kind | Model-bound reproduction of a prior tool-output-format failure |
| Attempt state | Completed; workflow failed |
| Result and acceptance criterion | A local API replay of captured OpenWork input ended normally with zero tool calls. The separate tiny-tool workflow failed its initial required-call and exact ordinary-chat checks. |
| Evidence scope | Pinned-source recreation, UI import/load, actual host request capture and local API replay; not proof of metadata identity with the deleted artifact |
| Promotion decision | This artifact is not qualified for OpenWork tool workflows; no parser relaxation or inference-policy change adopted. |

## Inputs and limits

- Repository: `mlx-community/Qwen3-0.6B-4bit`.
- Immutable source revision from the local Hugging Face cache snapshot: `73e3e38d981303bc594367cd910ea6eb48349da8`.
- Snapshot directory: local Hugging Face cache under `models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8`; source files are cache symlinks to content-addressed blobs.
- Final imported output: ignored `runtime/Qwen3-0.6B-4bit`, subsequently deleted after cleanup.
- The conversion itself leaves the registry unchanged. A later temporary UI import was removed after replay. Do not persist raw request or completion content, tokens, probabilities, credentials or exception details.

## Evidence

- Converter command: `stark-convert <local snapshot> --repo mlx-community/Qwen3-0.6B-4bit --revision 73e3e38d981303bc594367cd910ea6eb48349da8 --metadata-profile hf-generic --architecture qwen3 --artifact-mode text --output runtime/recreated-qwen06`, using the locked `tools/converter` environment. The completed output was then renamed to `runtime/Qwen3-0.6B-4bit` for its intended import basename.
- Output artifact: ignored `runtime/Qwen3-0.6B-4bit`; 335,584,424 bytes across 452 tensors; output SHA-256 `61dc73c095219e6d76dc72606c5882f50e7ad3ef9c634ff33a601ed038dc29d4`.
- Source manifest SHA-256 `bf2f93f34b912ce3834f21ee22b1f1aa8db7469fabc245a997f74efce73ea354`; config SHA-256 `acf3fabb2496a186519d5e399e1681822c584a7deb5ff100baba85a6993c79a6`; tokenizer SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
- Converter reported a text-only contract. The artifact loaded successfully in the installed app and in an owned replay worker. This generic metadata-profile artifact is not proven metadata-identical to the deleted UI-converted artifact; its immutable source revision, 4-bit source configuration and output size class match.
- Sealed bounded receipt: ignored `runtime/tool-quality-replay/qwen06-recreated-native.json`; artifact SHA-256 `4fe2fd897e56a503d1d7352baea5fcb2dd0179f00b47c0c5b08a6c0a5e17fb6b`; unchanged bytes confirmed. Worker SHA-256 `3cc9b3072d77bada79a23eb421196a73672aeb17734e0964a9ba282b392f9449`; validator SHA-256 `70489bfd6b03b73a00b963f5e18df1def225478822320e1680f4d59082625b1f`. At greedy/MTP0, context 8,192 and output cap 256, the initial required-call check and exact ordinary-chat check failed. Continuations were not reached after the initial call failed; their false flags are not separate observed continuation failures.

## Actual host request and replay

A temporary Stark-provider observer was verified loaded before capturing an
owned request. Headers were excluded. Bodies remained only in the receiver
process and were discarded when it exited. Initial capture first selected an
auxiliary-looking request with no tools and a 16,000-token output limit. A
256-token replay of that input reached the diagnostic length limit without
tool markers; it is not the main tool-request result and its purpose was not
proven from shape alone. The observer was then restricted to nonempty tool
definitions, preventing that confound.

The captured main request had two messages (system and user), 35 tools,
76,290 canonical tool-definition JSON bytes (SHA-256
`de9c5faf13ea82f1d7d69d0c40ffba24c8ad2bdbe59931b2e10e02462435c98e`),
and 30,353 canonical system JSON bytes (SHA-256
`7ba748f5b316e7952d2cf478caa345a019724308170e518083ad1035b480d212`).
It requested `max_completion_tokens: 10524` and selected the API's greedy/seed-0
defaults, with no stop strings. File creation was not observed in the client run.

After stopping the client task and app worker, the captured input was replayed
through the current Desktop API path using its original output limit. Streaming
was disabled and MTP explicitly disabled; this artifact has no supported draft
head. No returned tool was executed. The worker tokenized 28,252 input tokens.
The total of 38,776 input-plus-maximum-output tokens fits the source's declared
40,960-token context. This replay was admitted, ended at EOS after 132 output
tokens, and returned 504 text bytes with **zero tool calls and zero opening or
closing tool-call markers**. Output SHA-256:
`e60116a925926054e9605e7e42873007f2c75e2b2e72c25de49be7d5a2512d88`.

This demonstrates failure to select a tool under the captured workload, with
normal generation termination. A context-admission error, output truncation,
missing closing delimiter, or rejection of an otherwise valid generated call
does not explain this replay. It supports a model/workload capability limitation.
It does not prove absence of tool training or establish the precise cause of the
earlier malformed post-tool completion. Recognizing a template's call format
does not establish reliable agent behavior, and plain text must not be turned
into an executable call to hide this failure.

## Cleanup and disposition

The temporary entry was removed through Remove from Stark; the original five
entries remained. Generated model files were deleted after retaining aggregate
receipts. Every public configuration injection was restored byte-for-byte and
hash-checked, the capture processes exited, and OpenWork was restarted to unload
the observer. Dense was selected and stopped; OpenWork was idle. Source caches
were preserved. No model math, sampling defaults, parser acceptance, user request
limits, or connector authentication behavior was changed.

## Follow-up history

- Registered on 2026-11-10 after locating the exact source snapshot. Converted from the cached 331 MiB snapshot without downloads. Conversion, temporary UI import/load, actual-input replay, bounded native qualification and cleanup subsequently completed. This artifact remains unqualified for tool workflows.
