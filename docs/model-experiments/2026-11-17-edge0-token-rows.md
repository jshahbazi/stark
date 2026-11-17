# 20261117-EDGE0-22: token-batched prefill rows

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered / executed / owner | `20261117-EDGE0-22` / 2026-11-17 / 2026-11-17 / Claude Code, at the owner's request |
| Kind | Kernel scheduling for prefill, with a bitwise correctness gate |
| Attempt state | Completed; record written after the measurements, not before them |
| Result and acceptance criterion | Passed: identical F32 bits for every compared value, matching output hashes, higher observed prefill rate |
| Evidence scope | Operator tests, a model-free kernel timing probe, whole-model schedule comparison on the actual artifact, short end-to-end timing observations |
| Promotion decision | Default-off opt-in selector values; adoption on the local loading route awaits the owner |
| Disposition | Experiment on the evaluation branch source; no catalogue, product, merge, release or installed-build claim |
| Predecessor / related | [Attempt 21](2026-11-17-edge0-direct-routes.md), [attempt 19](2026-11-13-edge0-singleton-speed.md), [attempt 20](2026-11-16-edge0-context-loading.md) |

## Hypothesis and change

- Problem: after attempt 21 a prefill token still costs about as much GPU time
  as a decoded token. Every token row reads and decodes its matrix again, and
  attempt 21's stage timing put about 24 of 44 ms per prefill token on
  matrices that every token of a batch shares.
- The original batched kernel shares one decode across tokens and matches the
  one-token kernels bit for bit, yet takes longer per token row than they do.
  It selects the format and the token count at run time.
- `22-PROBE`: token-batched kernels whose format and token count are
  compile-time constants, timed without the model. Stop rule, fixed before the
  probe: stop unless they are at least 1.5 times faster per token row than
  one-token dispatches on shared matrices.
- `22-ROUTES`, selector value `token-routes`: the trunk projections of a token
  batch, the D rows of a projection group, one expert's gate/up and down rows,
  and up to eight U rows of a group share a dispatch. Rows are addressed by
  offsets in their own buffers, so nothing is gathered or scattered.
- `22-STAGES`, selector value `tokens`: the three projections of each
  hyper-connection mix and the GDN or attention output projection also run
  once per token batch. They replace the generic one-token projection, so
  float matrices get kernels from the same template. The GDN result row is
  persistent state; each token's row is copied before the next token
  overwrites it. One token keeps every one-token dispatch.
- Unchanged: route slots, per-route U, the 256-column stride, the SIMD-group
  sums, the ordered eight-group F32 sum and the causal order of the state
  recurrence. `edge0_graph.metal` and `edge0_direct.metal` are byte-identical;
  the new kernels live in `edge0_tokens.metal`.
- Fixed gate, set before timing: every value the schedules can compare must
  have identical F32 bits on the actual artifact, and the 128-token request
  must reproduce the attempt 19 output hash.
- `STARK_EDGE0_DIRECT_ROUTES=verify-tokens` encodes the token-batched forms
  beside the grouped schedule and beside the one-token mixes and output
  projections, and rejects any differing bit.
- Decode effect: with one token per batch, U rows of the same group share a
  dispatch. Every other decode dispatch is the direct schedule's.

## Artifact, source and environment

| Field | Baselines | Candidate |
| --- | --- | --- |
| Model and revision | `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74` | Same |
| Preparation receipt SHA-256 | `8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e` | Same; payloads not rehashed |
| Backend | Native Metal graph, virtual bindings, selected expert records | Same |
| Stark source | Grouped and direct schedules of the candidate binary | Attempt 21 commit `6192a4e5e42c3422cfe7e7585f45cca49247aa0a` and the nine files below |
| Worker SHA-256, Rust 1.98.1 | Same binary as candidate | `113528545982192fab23d80969d33a40041b5a1f44f59933c4774618ca81a58f` |
| Weight, activation, accumulation, state precision | Stored Q8_0, Q3_K, IQ4_NL and float formats; F32 elsewhere | Unchanged |
| Kernel policy | One-token kernels per token row; generic one-token projection for mixes and mixer outputs | One dispatch per matrix for up to eight token rows |
| Hardware and OS | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332) | Same |
| Memory | Not measured | Not measured; prefill scratch allocation rises from 4,745 to 5,633 MB per 128-token request |

| Changed or added source file | SHA-256 |
| --- | --- |
| `decode_graph.rs` | `ada75ab7d82f3cc66882af2426102367b394209d07400e7d385fe797ddcf0e24` |
| `decode_graph_edge0.rs` | `c8d5a8bed285e61cbcccffbb891d07c3234e111c2c49a93b47c31452e4109c53` |
| `decode_graph_edge0_tokens.rs` (new) | `65af750933e003e1e3ac09d35cba033be756768f15d629bc646588d35e6bf18b` |
| `decode_graph_edge0_tokens_tests.rs` (new) | `e6401fed3215868574405643477b4a3bfe1c58aa55b988b5260d9efe7ab96e98` |
| `edge0_tokens.metal` (new) | `257c1d2d7de461d9ece18da3636d489bd588119c56b6c7bbe28f46ef2113801d` |
| `edge0_metal.rs` | `a2f6d8390c0c2a9841ed0020819bbeaf0a68b4010ae130563088c90bdd603c79` |
| `edge0_weights_metal.rs` | `3e9dde3ef7d9c9d202f756079fe32a9ca52e3617099d296b47e0bfbf2fbb909d` |
| `qwen_graph.rs` | `e5fa479e5e9ac08b33658cc31fa27c123e674a3bae69561bdba66a02016f6e0a` |
| `weight_metal.rs` | `caf5137a0318ddf0d9a8cf3a77b33bca47ca342427789406e73831f47d3e4bd6` |

All nine files are under `crates/stark-engine/src/engine/hybrid/`. Six of them
were also changed by attempt 21, whose record binds their earlier state.
`decode_graph.rs` and `qwen_graph.rs` split the GDN and attention encoders so
a caller can take the row before its output projection; their existing callers
encode the same dispatches in the same order.

## Workload and controls

- Image capability: not applicable; the pinned source is text-only, as in the
  earlier Edge0 records.
- Prefix reuse: unchanged and unavailable under the existing state-restoration
  blocker. The schedule writes the same values, so it adds no restoration
  state.
- MTP: explicit zero drafts on the diagnostic route; unavailable on the local
  loading route. No MTP-enabled request was run with the new selector values.
- Short workload: attempt 19's 128-token input, context 512, greedy seed 42,
  16-token output limit, one warmup before three measured requests, one worker
  per session, aggregate native profiling enabled.
- Long workload: local loading route, context 8,192, 1,024-token input from
  the same private corpus, 16-token output limit, one warmup before two
  measured requests.
- Order: the final short sessions ran grouped, direct, `token-routes`,
  `tokens`, `tokens`, `token-routes`, direct, grouped. The local sessions ran
  direct, `token-routes`, `tokens`.
- Host: no quiet admission, host screen or paging sampler. The runner refuses
  to start while another `stark-generate` process exists and stops only its
  own worker. Every final session started with none.
- Private runner SHA-256:
  `e9cbf4914b60e16a187aeff9fefee6d5b997050e5b98b614a86ff87ffe9471d9`.

## Attempt history and failures

| Follow-up | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| `22-PROBE` | Model-free GPU timing of the token kernels on Edge0's matrix shapes | 1.70 to 1.86 times the one-token rate per token row at two tokens, 2.78 to 3.52 at four, 3.91 to 5.84 at eight | Passed the 1.5x stop rule. Retained as an ignored diagnostic test |
| `22-ROUTES` first sessions | Route and trunk rows, pre-final binary | 29.1 to 30.8 prefill tok/s against 17.6 to 20.2 direct | Preliminary; superseded |
| `22-ROUTES` | Same, one binary, grouped, direct and token-batched arms | 14.54, 18.96 and 29.93 prefill; 11.87, 13.59 and 14.40 decode tok/s | Complete for the short workload. Six measured requests per arm |
| `22-ROUTES` local route | 1,024-token sessions | Runner refused both sessions: the owner's desktop application had started a worker | Not run; repeated in the final sessions |
| `22-STAGES` first sessions | Mixes and mixer output projections added, pre-final binary | 40.5 to 41.1 prefill tok/s against 31.1 to 31.4 for route rows only | Preliminary; superseded |
| Final | All four schedules, final binary | 14.70, 19.43, 30.11 and 39.30 prefill; 11.98, 13.97, 14.84 and 14.90 decode tok/s | Observation complete |
| Final local route | 1,024-token input, three schedules | 19.00, 30.03 and 38.55 prefill; 13.51, 14.25 and 14.27 decode tok/s | Two measured requests per arm |

The direct schedule's own rate differs between attempt 21 (20.56 prefill
tok/s) and these sessions (19.43), so arms are compared only within one binary
and one set of sessions.

Not attempted here because they change summation association, routing
authority or the admitted batch size: applying U once per projection group,
routing on the device, reordered row reductions and token batches above eight.

## Correctness and lifecycle

- Whole-model comparison on the final binary: four requests, 434,976 device
  comparisons, 4,923,110,208 F32 values, zero differing bits. Three requests
  use the diagnostic route with 128-token inputs at corpus offsets 0, 128 and
  256; one uses the local route with a 1,024-token input.
- The `22-ROUTES` binary passed the same four requests with 98,016 comparisons
  and 4,405,000,512 values, the counts of attempt 21.
- Compared values: token-batched rows against the grouped schedule for every
  trunk, router and shared projection of a token batch and for the latent
  expert rows and expanded rows of every latent stage; both outputs of every
  token-batched mix against the one-token mix; every token-batched mixer
  output row against the generic one-token projection of the same row.
- Output hashes match the attempt 19 reference in all 32 short-workload
  requests of the final sessions, warmups included, and the attempt 21 local
  route hash in all nine local requests.
- Finite obligations: each token kernel records sticky failure for every
  value it writes, and a nonfinite input makes every result of its row
  nonfinite. For the mixes and mixer outputs this replaces the generic
  projection's separate input and output scans. Model-free tests reject NaN
  and both infinities without a scan, invalid stored scales, row indices
  beyond their parent, shared output rows, aliased parents, ragged parents and
  truncated spans.
- Tests, pinned nightly, final source: the complete native engine suite passes
  1,116 library tests with 164 ignored, plus 12 tests in its other targets.
  The `engine::hybrid` module accounts for 327 passes and 72 ignored. The
  three new tests pass. The fourth new test is the ignored timing probe, which
  passes when run explicitly. Ignored actual-artifact gates are not new
  passes.
- Workers exit zero on end of input in every session. Cancellation, reset,
  restart and rejection lifecycle cases were not rerun.

## Performance and memory

| Metric, median | Grouped | Direct | `token-routes` | `tokens` | Eligibility and conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| Prefill tok/s, 128-token input | 14.70 | 19.43 | 30.11 | 39.30 | Unscreened observation; 2.02x over direct |
| Decode tok/s, first token excluded | 11.98 | 13.97 | 14.84 | 14.90 | Unscreened observation; 1.07x over direct |
| End-to-end output tok/s | 1.61 | 2.08 | 3.04 | 3.76 | Unscreened observation |
| Prefill command GPU ms per token | 60.69 | 45.03 | 27.62 | 20.47 | Profile counter |
| Prefill tok/s, 1,024-token input, local route | Not run | 19.00 | 30.03 | 38.55 | Two measured requests per arm |
| Request wall seconds, same request | Not run | 55.01 | 35.16 | 27.62 | Same |
| Observed peak memory | Not measured | Not measured | Not measured | Not measured | |

Short-workload cells are medians of six measured requests over two sessions
per arm.

## Evidence, decision and limits

- Aggregates, probe and stage tables and evidence hashes are in the
  [benchmark report](../benchmarks/2026-11-17-edge0-token-rows.md). Private
  profiles, logs, runner and summaries stay under ignored `runtime/`.
- Worked: bit-identical token-batched rows on both loading routes for the
  recorded requests, with about twice the direct schedule's observed prefill
  rate.
- Not established: strict timing qualification, paging behavior, lifecycle
  cases, contexts above 1,024 input tokens, MTP, images, other artifacts.
- Decision: the selector values stay opt-in and default off. The local
  loading route keeps the grouped schedule until the owner decides otherwise.
- Next evidence for adoption on the local route: the attempt 20 context and
  lifecycle controls with the selector on, and the paging screens of attempts
  12 to 14.
- Remaining prefill cost after this change, per token with stage timing on:
  selected experts 8.1 ms, U 2.1 ms, state recurrence 2.6 ms, mixes 2.2 ms,
  input projections 1.6 ms, D 1.1 ms, mixer output 0.6 ms, router and shared
  expert 0.6 ms. Expert rows share least: a batch of eight tokens gives each
  selected expert about 1.75 rows.

## Follow-up history

- 2026-11-17: [attempt 23](2026-11-17-edge0-prefill-batches.md) adds an
  opt-in selector for prefill batches of 16, 24 or 32 tokens under the
  `tokens` schedule and changes eight of the nine files above. The hashes in
  this record bind the source state of commit
  `603b957d70b361579f2a023e5dc73fcf34bfafb0`. At the default of eight tokens
  the selector values of this attempt keep their behavior.
