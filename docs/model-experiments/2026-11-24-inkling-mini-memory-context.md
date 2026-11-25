# Inkling 2 Mini load-time context and live memory admission

## Identity and state

| Field | Value |
| --- | --- |
| Attempt, date, owner | `20261124-INKCTX-01` · 2026-11-24 · Codex |
| State | Completed local worker validation |
| Result | The tested artifact selected 131,072 tokens and completed requests with up to 123,400 input tokens on the M5 Max. The managed OpenCode endpoint reported 131,072 for the lifetime of the worker. |
| Disposition | Implemented in `31cb7d8c`. Runtime evidence applies to the checkout workers and artifact below. |
| Related evidence | [Affine8 default](2026-11-23-inkling-mini-affine8-default.md) · [Inkling speed history](speed-inkling.md) |

## Context and memory policy

The earlier worker in `bd0f0701` had a 32,768-token ceiling on the pinned M5 Max path. The candidate selects a stable cap after the model, vision, DSpark and prefill resources are resident. It reserves 256 MiB from each headroom measurement, and uses 85% of the smaller unified-memory or remaining Metal working-set budget for new context state, prefill workspace growth and a full prefix snapshot. The pinned engine geometry allows up to 131,072 tokens within the device buffer limit. Other configurations retain their 4,352-token geometry.

Each generation request rechecks current memory. The estimate covers any replacement live caches, workspace growth and a new snapshot when needed. The automatic planner can use a smaller execution context when memory is tight, and fixed-context requests can receive a recoverable resource-budget error. The advertised cap stays fixed until reload.

## Artifact and build identity

| Field | Value |
| --- | --- |
| Host and toolchain | Apple M5 Max, 128 GiB unified memory, macOS 26.7.2 (25G332). Rust 1.98.1, locked release build, deployment target 26.4. The existing Stark process remained loaded during the runs. |
| Upstream checkpoint | `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` |
| Installed affine8 artifact SHA-256 | `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd` |
| DSpark artifact SHA-256 | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |
| Precision | Original NVFP4 and MXFP8 experts with affine8 dense target weights. Existing target, draft and vision precision policies were used. |
| Interim worker and source diff SHA-256 | `b3fe53ab112ec269a7d86891dc034d0e413104aae3677bcdfc2c551e82068cd5` and `55e00fbd1b971b892c075f85220edbf3e4789783e088daca768bdc4b82bad0cf` over `bd0f0701` |
| Final worker SHA-256 | `db710f14d1ccad96b081f77f3cfd645e4fa72399350fa8ad120e9cc44b7d2964`, code `31cb7d8c` |
| Cargo.lock SHA-256 | `07b6d42497825a651719685a28a107bfc0a32119b8e8b8adeab2d6c01a96695b` |

## Workload and results

The registered target was 131,072 tokens, with 65,536 as a fallback if validation exposed a blocker. One private worker at a time loaded the same affine8 artifact with vision and DSpark available. The text fixture used synthetic natural-language records and a deterministic continuation. Greedy arms used the same rendered token IDs, seed and 96-token output allowance. A tiny synthetic image was used to test cache invalidation. The worker advertised a 2,048-token cache grid and a four-token tail. All content and token vectors stayed private.

| Trial | Result |
| --- | --- |
| Metal geometry | Metal API Validation passed the source packing dispatch shape at both 8 × 65,536 and 8 × 131,072 threadgroups. The targeted engine unit test passed the exact combined snapshot buffer limit and its one-byte-below case. |
| Interim cold serial vs DSpark | On 120,143 input tokens, the two arms sampled 96 tokens each with identical output token IDs and finish reason. DSpark proposed 99 tokens. |
| Interim warm/cold and image lifecycle | With 123,400 input tokens, the warm request reused 118,784. The image answered correctly and reported zero reuse. The next same-scope text request also reported zero reuse, and its output token IDs and finish reason matched the warm request. |
| Final automatic continuation | After a 120,143-token cold DSpark seed, the planner selected 131,072 for the 123,400-token follow-up, leaving 7,672 tokens available for output. Generation reused 118,784 tokens and sampled 96 with 93 proposals. |
| Final managed OpenCode endpoint | A private authenticated OpenCode grant completed streamed requests with an explicit 96-token limit and with automatic output. Both ran at 131,072 and sampled 49 tokens. The automatic plan allowed 131,045 output tokens for 27 input tokens. Model discovery reported 131,072 throughout. |
| Regression tests | 174 generator tests passed with five ignored. The targeted engine limit test and 27 targeted Desktop tests also passed. The tests covered the sizing bound and the host's recoverable resource-budget behavior. |

## Performance and evidence limits

| Diagnostic arm | Prefill | Decode |
| --- | ---: | ---: |
| Interim cold serial, 120,143 input | 80.707 s | 2.976 s |
| Interim cold DSpark, same input | 83.095 s | 2.309 s |
| Interim warm continuation, 123,400 input | 5.452 s | 2.160 s |
| Interim post-image cold continuation, same input | 93.539 s | 2.534 s |
| Final automatic warm continuation, same input | 5.924 s | 2.142 s |

Each timing is a single worker diagnostic. Peak sampled Metal allocation was 37,675,794,432 bytes (35.09 GiB) in both worker trials, over 289 samples for the interim trial and 133 for the final one. This measures current allocations in the worker process. The checks covered one local artifact and device, with actual input up to 123,400 tokens. The OpenCode check exercised the managed API through an isolated Desktop service. Physical memory pressure was not induced.
