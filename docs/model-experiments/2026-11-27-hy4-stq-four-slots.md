# HY4STQ-SERVE-02 · Default shared-weight text slots for Hy4 STQ

This change follows the [two-slot preview](2026-11-27-hy4-stq-dual-slot.md).
The native catalog artifact is
`beamster/Hy4-Flash-Sushi-STQ-4bpw@61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`.
The artifact identity is
`1ed7e181a27f78ffb1e93f9bef2eb00be3c9c5933cf5d7f50494320b45e9a04f`.

The desktop requests four slots by default. The native loader shares one model
and admits the highest slot count the memory budget supports. The API uses a
window that fits the admitted slots, and the primary retains its separately
fitted capacity for Playground and image work. Each text slot keeps its own
context, KV and index state, cancellation pipe, and output stream. A text
request can use a smaller logical window. A load requested with one slot uses
the ordinary automatic fit.

The native phase queue lets the slots share the temporary Metal projection,
routing and attention buffers. Each auxiliary still owns its Engram readers,
draft target, readout and sequence buffers. The fitter reserves its private
buffers and limits common capacity before dropping a slot. Primary prefix
reuse follows the consecutive-request policy. Auxiliary starts and finishes
mark any retained primary state for invalidation before the next primary
request. Limiting auxiliary state to the active request avoids holding
several completed conversations in memory.

## Qualification

- Reuse the 16,229- and 27,229-token OpenCode renders from the two-slot
  preview. Check the same 96-token budgets, input and output hashes, and
  match the expected lane count to actual native admission.
- Record four-agent first-token latency and total group time, retain a solo
  27K control, and keep the existing five-draft and F16 execution path.
- Exercise flexible windows, image handling and consecutive primary prefix
  reuse on the same model instance. Check cancellation and EOF routing as
  model-free controls.

## Result

Stark `b5c971926` built with Rust 1.98.1 and the `hy4` feature. The release
worker SHA-256 is
`4b467832d9088168d483fe976946667f5e26848da0b0d2f81ef2038bc6bbb912`.
The normal app backend admitted four slots in one process on the 128 GiB
M5 Max without slot or context overrides. It reported 68,608 positions on
the primary, 45,056 on each additional slot, and 45,056 for the API. Startup
took 119.87 seconds. Host memory pressure was normal in the samples before
and after startup and at the timed group boundaries.

| OpenCode 2.4.2 workload | Total group time | Median first delta | Median prefill tok/s | Median useful decode tok/s | Peak active slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16,229 tokens × 4 | 96.46 s | 81.52 s | 206.21 | 6.60 | 4 |
| 27,229 tokens × 4 | 139.44 s | 126.97 s | 218.55 | 7.85 | 4 |
| 27,229 tokens × 1 | 41.12 s | 39.16 s | 718.86 | 51.15 | 1 |

Each group is one observation and matched the original input and output
hashes on every request. First delta starts at client launch. Per-agent
rates include the time waiting for the other GPU phases. Active requests
overlapped for 93.64 seconds at 16K and 137.07 seconds at 27K. The first
timed group was 16K × 4. The historical two-slot run used a 49,152-position
cap and a different order. Its four-agent group times were 83.54 and 150.13
seconds, with median first deltas of 54.53 and 104.88 seconds. These runs
support four-way admission and functional comparisons. Timing differences
between them also include context capacity and workload order.

The original continuation reused 12,288 positions with the fresh replay's
output. The primary's 8,192→8,192→12,288 check reused 0→4,096→0 positions.
All four slots produced the same fresh output at 8,192 and 12,288. The
image test used the primary with drafting disabled and cleared the text
prefix. The Python host suites passed 522 tests with one skip. The release
generator suite passed 188 with six ignored. The model-free native budget
test exercised the fallback ladder and explicit smaller caps.

A second run checked shared scratch with four distinct synthetic inputs of
256, 1,250, 4,160 and 4,608 tokens. It generated 32 greedy tokens per
request with DFly2 enabled and zero prefix reuse. The active intervals
overlapped for 13.17 seconds, with a peak of four. Every output matched its
own fresh serial reference. All four output hashes were distinct. Two
inputs crossed the 4,096-token prefill boundary, exercising scratch reuse
between multiple chunks of different sequences.

The signed local app was built from the same source and worker. Its
embedded input and resource hashes and strict signature check passed.
The default enables the slot count that fits. `STARK_HY4_CONCURRENT_SLOTS=1`
selects normal single-slot execution and the automatic single-slot context
fit. Current native scheduling time-slices the GPU at existing phase
boundaries. Shorter prefill slices and batching remain separate performance
experiments.

The private default-run receipt SHA-256 is
`cec97a16ae0384e5fe31cfc5c37ec161d9a03e94b26a72517a1b0da2c0e4db2b`.
The private mixed-input receipt SHA-256 is
`e8e5e8b50f2f397cc0a0697d8631bf77357e3fd63c9c8a6108c3e6b9f128a77d`.
