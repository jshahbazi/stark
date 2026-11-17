# 20261117-EDGE0-23: larger prefill token batches

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered / executed / owner | `20261117-EDGE0-23` / 2026-11-17 / 2026-11-17 / Claude Code, at the owner's request |
| Kind | Prefill batch size under the token-batched schedule, with a bitwise correctness gate |
| Attempt state | Completed; record written after the measurements, not before them |
| Result and acceptance criterion | Passed: identical F32 bits for every compared value, matching output hashes, higher observed prefill rate |
| Evidence scope | Operator tests, whole-model schedule comparison on the actual artifact, short end-to-end timing observations |
| Promotion decision | Default-off opt-in selector; adoption on the local loading route awaits the owner |
| Disposition | Experiment on the evaluation branch source; no catalogue, product, merge, release or installed-build claim |
| Predecessor / related | [Attempt 22](2026-11-17-edge0-token-rows.md), [attempt 21](2026-11-17-edge0-direct-routes.md), [attempt 20](2026-11-16-edge0-context-loading.md) |

## Hypothesis and change

- Problem: after attempt 22, selected experts are 41 percent of prefill GPU
  time, and a batch of eight tokens gives each selected expert about 1.75
  rows to share a dispatch.
- Change: `STARK_EDGE0_PREFILL_TOKENS` of 16, 24 or 32, fixed once per
  process, sets the ordinary prefill batch. It is honored only with
  `STARK_EDGE0_DIRECT_ROUTES=tokens` or `verify-tokens`. Every other schedule
  keeps eight, the limit of its kernels, and so do wide-output and MTP
  batches.
- Token kernels still take at most eight rows per dispatch. A larger batch
  lets more rows select the same expert, D group and U group, and reduces the
  router round trips of a request.
- Above eight tokens the float trunk matrices use the float token kernels of
  attempt 22, because the original batched kernel stops at eight.
- Unchanged: each token's arithmetic, route slots, per-route U, every F32
  reduction and the causal order of the state recurrence. No shader file
  changes.
- Fixed gate, set before timing: every value the schedules can compare must
  have identical F32 bits on the actual artifact, and the 128-token request
  must reproduce the attempt 19 output hash.
- Under `verify-tokens` the grouped schedule and the original batched kernel
  remain the references and run eight tokens at a time.
- Memory admission charges the selected batch for its host rows. The
  per-segment scratch arena stays at 64 MiB.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model and revision | `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74` | Same |
| Preparation receipt SHA-256 | `8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e` | Same; payloads not rehashed |
| Backend | Native Metal graph, virtual bindings, selected expert records | Same |
| Stark source | The `tokens` schedule at 8 tokens in the candidate binary | Attempt 22 commit `603b957d70b361579f2a023e5dc73fcf34bfafb0` and the nine files below |
| Worker SHA-256, Rust 1.98.1 | Same binary as candidate | `1e10aa8920c8db22cf1d3cfd03a7c8089e466c7f05e7bd975725aaa7535301ce` |
| Weight, activation, accumulation, state precision | Stored Q8_0, Q3_K, IQ4_NL and float formats; F32 elsewhere | Unchanged |
| Batch policy | 8 tokens per ordinary prefill batch | 16, 24 or 32 tokens when selected |
| Hardware and OS | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332) | Same |
| Memory | Not measured | Not measured; prefill scratch allocation per 128-token request is 5,633, 5,912 and 5,613 MB at 8, 16 and 32 tokens |

| Changed source file | SHA-256 |
| --- | --- |
| `decode_graph.rs` | `6ed5abb9f115f7bcb81ef9ba68ac44277c0eeba5f9bc1cc8b94d678d343094f6` |
| `decode_graph_edge0.rs` | `d0a4ec85111f5742af7aadb9e2e67d6112adb611cfed7d7f49d7f902b112c39a` |
| `decode_graph_edge0_tokens.rs` | `5052048baee9a3abc28471bd59fb035f856d80ebc6db7450322505b1e397b60d` |
| `decode_graph_edge0_tokens_tests.rs` | `eaebf708fca6fa267512ae565dcab0edf55ec810dd33ef8c7a9426cf648dcc70` |
| `edge0_metal.rs` | `59db5df6b672f177cea1500fd60efa5bb5a2fad4576e78b2ed15debeaab01801` |
| `edge0_weights_metal.rs` | `39c5c1af91812328a66af505b721c9cfdbea93db220fef4395416c6f581e772e` |
| `qwen.rs` | `f5c993b258e7e4e3f0b1ee678156002171e00b2dc7a2d6dde41ffa2e8d67dd85` |
| `qwen_graph.rs` | `5947db1e2174a363319b1f69da74fe2882b67d8a5a98564533b1c5d764b06f1f` |
| `weight_metal.rs` | `3f008886333f306f6211016d8d39b2a55d48048aa0ba25c682abf751e969abce` |

All nine files are under `crates/stark-engine/src/engine/hybrid/`.

## Workload and controls

- Image capability: not applicable; the pinned source is text-only, as in the
  earlier Edge0 records.
- Prefix reuse: unchanged and unavailable under the existing state-restoration
  blocker. A later snapshot grid must be a multiple of the selected batch.
- MTP: explicit zero drafts on the diagnostic route; unavailable on the local
  loading route. MTP and wide-output batches keep eight tokens and were not
  run with the selector.
- Short workload: attempt 19's 128-token input, context 512, greedy seed 42,
  16-token output limit, one warmup before three measured requests, one worker
  per session, aggregate native profiling enabled, `tokens` schedule.
- Long workload: local loading route, context 8,192, 1,024-token input from
  the same private corpus, 16-token output limit, one warmup before two
  measured requests.
- Order: the final short sessions ran 8, 16, 32, 32, 16 and 8 tokens. The
  local sessions ran 8, 16 and 32.
- Host: no quiet admission, host screen or paging sampler. The runner refuses
  to start while another `stark-generate` process exists and stops only its
  own worker. Every final session started with none.
- Private runner SHA-256:
  `8bd484687a6906428cff6a8b3bfc94b8cb956da4bd1743463246ee0e730e1112`.

## Attempt history and failures

| Follow-up | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| First sessions | 8, 16 and 32 tokens, pre-final binary, browser and other desktop applications active | 37.6, 44.0 and 46.8 prefill tok/s | Preliminary; superseded |
| Final | Same sizes, final binary | 41.49, 48.26 and 54.18 prefill tok/s; decode 14.68, 14.79 and 14.78 | Observation complete. Six measured requests per size |
| Final local route | 1,024-token input | 39.95, 47.86 and 53.60 prefill tok/s; 26.69, 22.44 and 20.14 wall seconds | Two measured requests per size |
| 24 tokens | Accepted by the selector | Not run | Unmeasured |

The 8-token arm's own rate differs between attempt 22 (39.30 prefill tok/s)
and these sessions (41.49), so sizes are compared only within one binary and
one set of sessions.

Not attempted here: token kernels wider than eight rows, batches above 32
tokens, and the ideas attempt 22 lists as changing summation association or
routing authority.

## Correctness and lifecycle

- Whole-model comparison on the final binary: five requests, 475,920 device
  comparisons, 5,402,287,632 F32 values, zero differing bits. One request
  uses 16-token batches and four use 32-token batches; four use the
  diagnostic route with 128-token inputs at corpus offsets 0, 128 and 256,
  and one uses the local route with a 1,024-token input.
- Compared values are those of attempt 22. For a batch above eight tokens,
  each group of eight is compared with the grouped schedule and the original
  batched kernel run on those eight tokens.
- Output hashes match the attempt 19 reference in all 24 short-workload
  requests of the final sessions, warmups included, and the attempt 21 local
  route hash in all nine local requests.
- Model-free tests add bitwise equality of batches of 9, 16, 17 and 32 tokens
  against the original batched kernel applied eight tokens at a time for six
  formats, and of the ordered route reduction for 16 tokens against two
  batches of 8. They reject batches above 32 tokens.
- Tests, pinned nightly, final source: the complete native engine suite passes
  1,117 library tests with 164 ignored, plus 12 tests in its other targets.
  The `engine::hybrid` module accounts for 328 passes and 72 ignored. The new
  test passes. Ignored actual-artifact gates are not new passes.
- Workers exit zero on end of input in every session. Cancellation, reset,
  restart and rejection lifecycle cases were not rerun. The cancellation hook
  runs before each batch, so a larger batch delays a cancellation by up to
  one batch.

## Performance and memory

| Metric, median | 8 tokens | 16 tokens | 32 tokens | Eligibility and conclusion |
| --- | ---: | ---: | ---: | --- |
| Prefill tok/s, 128-token input | 41.49 | 48.26 | 54.18 | Unscreened observation; 1.16x and 1.31x |
| Decode tok/s, first token excluded | 14.68 | 14.79 | 14.78 | Unscreened observation; unchanged |
| End-to-end output tok/s | 3.89 | 4.36 | 4.74 | Unscreened observation |
| Prefill command GPU ms per token | 18.76 | 16.41 | 14.91 | Profile counter |
| Prefill tok/s, 1,024-token input, local route | 39.95 | 47.86 | 53.60 | Two measured requests per size |
| Request wall seconds, same request | 26.69 | 22.44 | 20.14 | Same |
| Observed peak memory | Not measured | Not measured | Not measured | |

Short-workload cells are medians of six measured requests over two sessions
per size.

## Evidence, decision and limits

- Aggregates, the stage table and evidence hashes are in the
  [benchmark report](../benchmarks/2026-11-17-edge0-prefill-batches.md).
  Private profiles, logs, runner and summaries stay under ignored `runtime/`.
- Worked: bit-identical larger batches on both loading routes for the
  recorded requests, with 1.31 times the 8-token prefill rate at 32 tokens.
- Sharing grows slowly: each selected expert receives 1.75, 2.18 and 2.83
  rows at 8, 16 and 32 tokens.
- Not established: strict timing qualification, paging behavior, peak memory,
  lifecycle cases, contexts above 1,024 input tokens, MTP, images, other
  artifacts, the 24-token setting.
- Decision: the selector stays opt-in and defaults to eight tokens. The local
  loading route keeps the grouped schedule until the owner decides otherwise.
- Next evidence for adoption on the local route: the attempt 20 context and
  lifecycle controls with the selectors on, cancellation latency at the
  selected batch, and the paging screens of attempts 12 to 14.
- Remaining prefill cost at 32 tokens, per token with stage timing on:
  selected experts 5.4 ms, state recurrence 2.5 ms, mixes 2.1 ms, U 1.8 ms,
  input projections 1.5 ms, D 0.8 ms, mixer output 0.5 ms, router and shared
  expert 0.3 ms.

## Follow-up history

None.
