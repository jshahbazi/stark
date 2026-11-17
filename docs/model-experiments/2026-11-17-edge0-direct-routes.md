# 20261117-EDGE0-21: stage attribution and direct-route schedule

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered / executed / owner | `20261117-EDGE0-21` / 2026-11-17 / 2026-11-16 and 2026-11-17 / Claude Code, at the owner's request |
| Kind | Performance diagnosis and kernel scheduling, with a bitwise correctness gate |
| Attempt state | Completed; record written after the measurements, not before them |
| Result and acceptance criterion | Passed: identical F32 bits for every compared value, matching output hashes, higher observed prefill and decode rates |
| Evidence scope | Operator and composition tests, whole-model schedule comparison on the actual artifact, short end-to-end timing observations |
| Promotion decision | Default-off opt-in selector; adoption on the local loading route awaits the owner |
| Disposition | Experiment on the evaluation branch source; no catalogue, product, merge, release or installed-build claim |
| Predecessor / related | [Attempt 19](2026-11-13-edge0-singleton-speed.md), [attempt 20](2026-11-16-edge0-context-loading.md), [reference performance ideas](../archive/runs/2026-11-13-edge0-reference-performance-ideas.md) |

## Hypothesis and change

- Problem: observed prefill near 15 tok/s and decode near 12 tok/s on the
  pinned Edge0-145B Q3_K artifact, with no per-stage GPU attribution.
- `21-STAGE` adds diagnostic stage labels to the existing encoder timestamp
  transport and a per-encoder dispatch count. It changes no arithmetic and
  encodes nothing unless `STARK_QWEN38_ENCODER_TIMING=1` and
  `STARK_EDGE0_STAGE_TIMING` are set.
- Finding: the multi-token packed kernel costs more per token row than the
  singleton kernel, grouped prefill runs D for token rows no route uses, and
  1,920 of 7,501 dispatches per decoded token copy latent rows.
- Candidate: `STARK_EDGE0_DIRECT_ROUTES=1` runs singleton kernels on each
  route's own rows, runs D only for selected token rows with its stored
  column-scale quotient in the same dispatch, and encodes D and U rows as
  projection levels. Route slots, per-route U and every F32 reduction are
  unchanged. `edge0_graph.metal` is byte-identical; new kernels live in
  `edge0_direct.metal`.
- Fixed gate, set before timing: every value the two schedules can compare
  must have identical F32 bits on the actual artifact, and the 128-token
  request must reproduce the attempt 19 output hash.
- `STARK_EDGE0_DIRECT_ROUTES=verify` encodes both schedules and rejects any
  differing bit. It is the gate's instrument and a retained diagnostic.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model and revision | `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74` | Same |
| Preparation receipt SHA-256 | `8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e` | Same; payloads not rehashed |
| Backend | Native Metal graph, virtual bindings, selected expert records | Same |
| Stark source | `9280d3182b4e288b00b291ff58b648e50a0d9201`, selector off in the candidate binary | Same commit plus the nine files below |
| Worker SHA-256, Rust 1.98.1 | Same binary as candidate | `a34d4c84815d2682ec473a43c564211823065ccd2736a79d2097d3e063e743c5` |
| Weight, activation, accumulation, state precision | Stored Q8_0, Q3_K, IQ4_NL and float formats; F32 elsewhere | Unchanged |
| Kernel policy | Singleton kernels for one-token jobs, batched kernel otherwise | Singleton kernels for every Q8_0, Q3_K and IQ4_NL token row |
| Hardware and OS | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332) | Same |
| Memory | Not measured | Not measured; prefill scratch allocation falls from 7,268 to 4,745 MB per 128-token request |

| Changed or added source file | SHA-256 |
| --- | --- |
| `decode_graph.rs` | `1022407fbabb3513606ff1e393d7d848b4772168a496456c52bf6e295c20871f` |
| `decode_graph_counters.rs` | `07a870cd3a6832ae81c27a224397c2c31995507b6f5daa7888fb4d88a61b6819` |
| `decode_graph_edge0.rs` | `139fccf81853e991266b20922630586a5013b61adbb1aff58138dbb32a493444` |
| `decode_graph_edge0_direct_tests.rs` (new) | `aa4ff33f366a971a10eaaa0006ef04cc2484ea581acbdfdddf8fca374d978790` |
| `edge0_direct.metal` (new) | `656411e2170487f2486116ecb996445be17a53582d00a2fb068e82a9f44079f8` |
| `edge0_metal.rs` | `b9ad88604eecc003846e15f67d1094e061a314fdb41873bf3e8d609b701fea2a` |
| `edge0_weights_metal.rs` | `4c5ed2fd1af3c632faf0a4ea0e1b7c0ba732ead18d62acd55c3c902bd73fb4a0` |
| `qwen_graph.rs` | `5fbb70cb57daeef5dbbb0d7333b0ccb7d22daaba193a007290ad3825f6ff2297` |
| `weight_metal.rs` | `6393bfcccdcf94d2525b8aa0c1125b98509c7ed294c44bd6938574d6f20bffbf` |

All nine files are under `crates/stark-engine/src/engine/hybrid/`.

## Workload and controls

- Image capability: not applicable; the pinned source is text-only, as in the
  earlier Edge0 records.
- Prefix reuse: unchanged and unavailable under the existing state-restoration
  blocker. The schedule writes the same values, so it adds no restoration
  state.
- MTP: explicit zero drafts on the diagnostic route; unavailable on the local
  loading route. No MTP path uses the new schedule.
- Short workload: attempt 19's 128-token input, context 512, greedy seed 42,
  16-token output limit, one warmup before measured requests, one worker per
  session, aggregate native profiling enabled.
- Long workload: local loading route, context 8,192, 1,024-token input from
  the same private corpus, 16-token output limit.
- Order: final sessions ran off, on, on, off. Earlier follow-ups ran in the
  order the [benchmark report](../benchmarks/2026-11-17-edge0-direct-routes.md)
  lists.
- Host: no quiet admission, host screen or paging sampler. The runner refuses
  to start while another `stark-generate` process exists and stops only its
  own worker.
- Private runner SHA-256:
  `d6c469b45b6ee4b35199202e864107ba755c8de2b3ff1c3d2b2d77aa79f218e0`.

## Attempt history and failures

| Follow-up | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| `21-STAGE` | Stage labels on encoder timestamps, levels 1 and 2 | Decode: U 24%, experts 24%, input projections 13%, mixes 12%, state 10%, D 7%, readout 4% | Diagnostic retained. Timing adds 4% to 16% GPU time |
| `21-STAGE` build | Seeded the build directory from another checkout's cache | Shader precompiled header bound to the other path; build failed | Invalid setup; stale shader cache removed and rebuilt |
| `21-DIRECT` | Direct rows with serial D and U | 15.14 to 19.95 prefill, 11.96 to 13.73 decode tok/s | Superseded by `21-LEVELS1` |
| `21-LEVELS1` | D rows and U rows as projection levels | 19.66 to 20.62 prefill, 13.63 to 13.97 decode tok/s | Retained |
| `21-LEVELS2` | One encoder scope per layer for all expert rows | 20.97 prefill, 14.33 decode tok/s | Not retained: merges per-expert scopes whose paging effect is untested. Selector removed, patch kept privately |
| Final | Selector off and on, same binary | 15.16 to 20.56 prefill, 11.95 to 13.92 decode tok/s | Observation complete |
| Local route | 1,024-token input, off and on | 14.73 to 18.59 prefill, 11.55 to 13.56 decode tok/s | Two measured requests per arm |

Ideas measured in `21-STAGE` and not attempted here because they change
summation association or routing authority: applying U once per projection
group, routing on the device, and cross-token expert batching.

## Correctness and lifecycle

- Whole-model comparison on the final binary: four requests, 98,016 device
  comparisons, 4,405,000,512 F32 values, zero differing bits. Three requests
  use the diagnostic route with 128-token inputs at corpus offsets 0, 128 and
  256; one uses the local route with a 1,024-token input.
- Compared values: split token rows against the batched kernel for every
  trunk, router and shared projection of a token batch, and the latent expert
  rows and expanded rows of every latent stage. Values downstream of those are
  produced by unchanged code from identical inputs.
- Output hashes match the attempt 19 reference in all 32 short-workload
  requests of the final and `21-DIRECT` sessions, warmups included.
- Finite obligations: chained direct rows carry no separate scan. Each kernel
  records sticky failure for every value it writes; a nonfinite input makes
  every row result nonfinite. Model-free tests reject NaN and both infinities
  without a scan, invalid stored scales, bad shapes, aliased rows and
  truncated spans.
- Tests, pinned nightly, final source: the complete native engine suite passes
  1,113 library tests with 163 ignored, plus 12 tests in its other targets.
  The `engine::hybrid` module accounts for 324 passes and 71 ignored. The four
  new tests pass. The ignored encoder-timing test passes when run explicitly.
  An earlier complete run was stopped at 258 tests, without a failure, to keep
  the host quiet for timing. Ignored actual-artifact gates are not new passes.
- Workers exit zero on end of input in every session. Cancellation, reset,
  restart and rejection lifecycle cases were not rerun.

## Performance and memory

| Metric, median | Selector off | Selector on | Repeats | Eligibility and conclusion |
| --- | ---: | ---: | --- | --- |
| Prefill tok/s, 128-token input | 15.16 | 20.56 | Six measured, two sessions per arm | Unscreened observation; 1.356x |
| Decode tok/s, first token excluded | 11.95 | 13.92 | Same | Unscreened observation; 1.164x |
| End-to-end output tok/s | 1.65 | 2.19 | Same | Unscreened observation; 1.327x |
| Decode command GPU ms per token | 63.06 | 52.19 | Same | Profile counter |
| Prefill tok/s, 1,024-token input, local route | 14.73 | 18.59 | Two measured, one session per arm | Unscreened observation |
| Observed peak memory | Not measured | Not measured | | |

## Evidence, decision and limits

- Aggregates, stage tables and evidence hashes are in the
  [benchmark report](../benchmarks/2026-11-17-edge0-direct-routes.md). Private
  profiles, logs, runner and patches stay under ignored `runtime/`.
- Worked: bit-identical direct rows on both loading routes for the recorded
  requests, with higher observed prefill and decode rates.
- Not established: strict timing qualification, paging behavior, lifecycle
  cases, contexts above 1,024 input tokens, MTP, images, other artifacts.
- Decision: the selector stays opt-in and default off. The local loading
  route keeps the grouped schedule until the owner decides otherwise.
- Next evidence for adoption on the local route: the attempt 20 context and
  lifecycle controls with the selector on, and the paging screens of attempts
  12 to 14.
- Remaining cost after this change, per decoded token: experts 13.4 ms, input
  projections 8.1 ms, mixes 7.7 ms, U 7.6 ms, state 6.7 ms, D 3.9 ms, readout
  2.4 ms, plus 49 blocking command completions.

## Follow-up history

- 2026-11-17: [attempt 22](2026-11-17-edge0-token-rows.md) builds on this
  schedule and changes six of the nine files above. The hashes in this record
  bind the attempt 21 source state, committed as
  `6192a4e5e42c3422cfe7e7585f45cca49247aa0a`. The `1` and `verify` selector
  values keep their behavior.
