# HY4STQ-SERVE-01 · Two active text slots with shared Hy4 STQ weights

The [current native STQ defaults](2026-11-27-hy4-stq-catalog-defaults.md)
are the single-slot control. A long OpenCode prefill holds the worker while
other agent requests wait. This experiment admits two text requests and
alternates GPU work at prefill and decode boundaries.

## Artifact and controls

| Field | Value |
| --- | --- |
| Artifact | `beamster/Hy4-Flash-Sushi-STQ-4bpw@61528ba635a623f127c5a7b1e1f4ac2c2b0d4885` |
| Artifact identity | `1ed7e181a27f78ffb1e93f9bef2eb00be3c9c5933cf5d7f50494320b45e9a04f` |
| Host and toolchain | M5 Max, 128 GiB, Metal 4, Rust 1.98.1 release, OpenCode 2.4.2 |
| Precision and draft policy | Original STQ weights, F16 KV, 4,096-row prefill and five greedy DFly2 drafts |
| Control | One native request stream, worker from Stark `90054b7e9` |
| Candidate | One native model and Metal context, two forward graphs and independent private JSON-line channels |
| Disposition | Opt-in candidate through `STARK_HY4_CONCURRENT_SLOTS=2` |

The second slot shares the model weight buffers, compiled Metal library,
Engram source, vision tower and DFly2 scratch. Its forward graph owns its KV,
IndexPool and intermediate buffers. The native phase queue lets a request
waiting for a GPU phase proceed ahead of a later phase. Text requests keep a
logical context within their admitted slot capacity. The load may admit one
slot when both context reservations and the second scratch arena do not fit.

The first slot serves image requests exclusively on the existing image path.
The second slot has no retained-prefix capability. Before reusing a primary
prefix after overlap, the desktop clears any state retained during the overlap.
The single-slot flow retains the existing immediately preceding, exact-prefix
policy. Both slots keep separate cancellation channels and output streams.

## Qualification workload

- Use the fresh 16,229- and 27,229-token synthetic review turns from the
  [tuned OpenCode fixture](2026-11-27-hy4-stq-open-code-performance.md)
  with the same 96-token output budget. Start one, two and four independently
  granted OpenCode CLI sessions per window. Isolate client storage and record
  only aggregate timings, counts and hashes.
- Hold the loaded cap and runtime settings equal where both allocations fit.
  Verify loaded capacity, actual request input counts, expected token hashes,
  draft and prefix behavior and which worker slots executed. Preserve the
  warmed one-agent decode rate and target at least 48 useful tok/s at 27K.
- Compare elapsed time to first nonempty API delta, separation between API
  admission and worker execution, worker prefill and useful decode, and wall
  time until all clients finish. Report native wall time including any wait
  for GPU phases. Record measured concurrency and memory pressure.
- Check flexible windows, a cancelled peer, and image exclusivity separately
  from fan-out timing. An isolated transport or model-free fixture does not
  establish real-weight timing or precision.

## Result

Measurement and promotion are pending. The source implementation is an
opt-in preview.
