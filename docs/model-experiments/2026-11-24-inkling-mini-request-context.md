# Inkling 2 Mini request-sized context allocation

## Identity and state

| Field | Value |
| --- | --- |
| Attempt, date, owner | `20261124-INKCTX-02` · 2026-11-24 · Codex |
| Kind | Memory, request planning and prefix reuse |
| State | Implemented in `9f5ff19f`. Completed local worker and managed API validation. |
| Result | A worker advertised 540,672 tokens. Under concurrent load a second worker advertised 249,856 and kept it fixed across managed OpenCode requests. A 248,399-token request reused 241,664 tokens across a physical resize. |
| Predecessor | [Load-time context and live memory admission](2026-11-24-inkling-mini-memory-context.md), `20261124-INKCTX-01` |
| Source | MLX-Serve policy reference at [`2fe9be834a601c70ef1a597563026340f0e722fe`](https://github.com/ddalcu/mlx-serve/tree/2fe9be834a601c70ef1a597563026340f0e722fe) |

## Implemented policy

After model load, estimate the memory-fit context from Stark's measured state and workspace costs. Use 85% of the memory-derived token count, rounded down to a 1,024-token boundary, then cap it by the verified `model_max_length` of 1,048,576 and production Metal and device limits. Keep advertised context metadata stable for the loaded worker.

For each request, plan against the fully rendered prompt, including expanded image tokens, plus up to 8,192 output tokens. Round physical allocation up in 1,024-token increments, bounded by the logical context limit. Grow live KV storage on demand after a fresh memory check. Report readiness `context_sizing` with raw and selected token counts, applicable limits and estimated byte costs. On completed responses, report `context_allocation` with physical sizes and byte costs.

Allow exact prefix reuse across physical-capacity changes when the logical window, model instance, host cache scope, MTP policy and source mode match. Images continue to invalidate text prefixes.

## Preregistered validation

Use the same two artifact hashes and hardware as `20261124-INKCTX-01`. Run isolated managed OpenCode requests under multiple output budgets. Include 240K and 248K rendered long-input requests with DSpark cold and warm arms, plus image invalidation and the subsequent text recovery. Register a separate live KV growth test to observe stepwise allocation and fresh memory checks independently.

Check stable advertised metadata, request planning bounds, reported physical capacities and byte costs, allocation recovery behavior, exact output IDs on matched serial and DSpark controls, warm-prefix reuse under capacity growth, and zero reuse for the image request and following text request. Keep request content, token vectors and private artifacts local. Do not infer full-length accuracy or standard benchmark performance from these admission and lifecycle checks.

| Evidence | Preregistered identity / result |
| --- | --- |
| Affine8 artifact SHA-256 | `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd` |
| DSpark artifact SHA-256 | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |
| Hardware | Apple M5 Max, 128 GiB unified memory, macOS 26.7.2 (25G332) |
| Worker build | Rust 1.98.1, locked release, deployment target 26.4. Long-run worker SHA-256 `c06890655f1596e86159cc384223bc361933c34ed0c4a84e38720503ef07180f`. Final worker SHA-256 `745b1cb943e7d49785c8abf5e07c359e4f30399226f43b95dffaff16f246e6e7`. |
| Managed API | Isolated managed OpenCode service, with distinct output budgets |
| Regression tests | 176 generator tests and 27 targeted Desktop tests passed. The nightly engine geometry filter passed 29 tests, with one ignored. The installed-artifact engine and decode growth controls also passed. |
| Performance status | Single worker diagnostics only. No standard benchmark result. |

## Observed sizing

| Load | Memory budget | Memory-derived tokens | Checkpoint and engine limits | Advertised tokens |
| --- | ---: | ---: | ---: | ---: |
| Long-run worker | 52,482,539,520 bytes | 636,743 | 1,048,576 | 540,672 |
| Managed OpenCode worker, another worker resident | 24,486,051,840 bytes | 294,989 | 1,048,576 | 249,856 |
| Natural-output growth probe, another worker resident | 28,773,236,736 bytes | 347,323 | 1,048,576 | 294,912 |

| Request | Input tokens | Requested output | Logical context | Initial physical context | Final physical context |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short explicit | 27 | 96 | 262,144 | 1,024 | 1,024 |
| Long serial, cold | 241,902 | 96 | 262,144 | 242,688 | 242,688 |
| Long DSpark, cold | 241,902 | 96 | 262,144 | 242,688 | 242,688 |
| Long DSpark, warm | 248,399 | 96 | 262,144 | 248,832 | 248,832 |
| Long DSpark, cold after image | 248,399 | 96 | 262,144 | 248,832 | 248,832 |
| Managed OpenCode, explicit | 27 | 24, 48 or 96 | 249,856 | 1,024 | 1,024 |
| Managed OpenCode, automatic | 27 | 249,829 | 249,856 | 9,216 | 9,216 |
| Controlled decode growth, serial and DSpark | 4 | 16 | 1,024 | 5 | 1,024 |

Serial and DSpark sampled 96 tokens with the same greedy output IDs and finish reason on 241,902 input tokens. A follow-up reused 241,664 tokens after changing the physical context from 242,688 to 248,832. Its snapshot needed 9,313,276,416 bytes, workspace growth needed 50,331,648 bytes and replacement live caches needed 9,351,025,152 bytes. A synthetic image completed correctly with zero reuse. The subsequent cold text request also had zero reuse and matched the warm output token IDs and finish reason.

The managed OpenCode bridge kept 249,856 in model discovery before and after three explicitly bounded requests and one automatic request. The automatic request could use the full remaining logical context for output. Its physical execution started at 9,216 and finished there. The direct engine growth control matched continuation logits, committed state, features and the active draft proposal against a fresh allocation. The separate decode control grew once and matched the fresh allocation's 16 generated tokens and finish reason in both draft modes.

An earlier concurrent API probe assumed a 262,144-token minimum and stopped at its harness assertion. During investigation, a separate load reported 234,496 in Desktop. Its API harness stopped on a stale variable before testing requests. The corrected probe accepted the capacity selected at load and passed.

The natural-output growth probe ended at EOS after 590 sampled tokens, below its initial reserve. The probe's assertion incorrectly assumed physical storage for the entire 12,500-token allowance, so it could not qualify live growth. The controlled decode and engine cases qualify the growth operation directly.

Peak sampled Metal allocation was 52,319,535,104 bytes across 811 samples of the long run. Warm prefill took 11.066 s and post-image cold prefill took 253.204 s. Each timing is a single worker diagnostic with other local workloads present. Validation exercised input through 248,399 tokens. The values advertised at model load are memory estimates.
