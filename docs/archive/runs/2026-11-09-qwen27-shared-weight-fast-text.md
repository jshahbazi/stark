# Qwen 27B shared-weight fast text with images loaded

Attempt `20261109-QDIP-07`, registered after initial raw-buffer-sharing source
edits, before constructing or running the new dual graph. Predecessor:
[QDIP-06](../../benchmarks/2026-11-09-qwen-dense-image-text-parity.md), which
found a 94.44% decode regression despite working serial MTP and exact caching.

## Fixed implementation and acceptance

- One loaded image-capable instance retains the admitted reference image graph
  and a canonical text graph matching the current source text-conversion plan.
  The text graph uses its existing optimized kernels and batched MTP; image
  requests keep the reference graph, zero drafts and text-prefix invalidation.
- Only validated raw matrix/embedding/quantization buffers may share immutable
  ownership. Graph-specific normalized controls and kernels remain separate.
  Signed A8 representation changes require private bytes and cannot mutate the
  image graph. Inventory shared storage, controls, copies and loading peak;
  preserve an explicit resource bound and do not duplicate the full decoder.
- Verify canonical configuration equality and target/predictor weight-transform
  equality against the existing full-head text control. Preserve the image
  configuration and its arithmetic. Image and text state identities must reject
  cross-graph use before device execution without poisoning the usable instance.
- Qualify actual fresh/cached output equality on nonperiodic long input near
  context capacity, at MTP0 and MTP3, with seeded same-mode controls, draft-depth
  switches, real image zero-draft/reuse transitions and cancellation recovery.
  Include actual post-prefill position/suffix checks; reported reuse alone is
  insufficient.
- Benchmark candidate text-only/image-loaded arms at matching MTP0 and MTP3,
  greedy seed 42, same source checkpoint and full original predictor. Prefer 64
  output tokens, ABBA loads, one excluded warmup and three measured requests per
  policy/load. Record actual proposals/accepts, first-token/decode/wall timings,
  hashes and paging/quiet-GPU eligibility. The bound is the greater of five
  percent and twice the baseline relative range. Use the sealed predecessor text
  worker or verify unchanged text execution to detect unrelated baseline drift.

GPU/build use is serialized with the other model tasks. Existing model payloads
stay unchanged, no full export/copy is required, and disk retains an eight-GiB
reserve. Publish aggregate counts, identities and hashes only. A failed gate
keeps parity open; tower-off loading is a validation control, never a product
workaround. All other checkpoints and family intakes retain separate gates.

## Results

The bounded Jundot 27B fast-text follow-up passes. This is one checkpoint and
conversion, not a qualification of every dense or MoE export. The predecessor
94.44% text decode regression remains recorded as a failed reference-graph
attempt; serial predictor execution alone did not complete this requirement.

### Exact graph and payload

Source: `Jundot/Qwen3.8-27B-oQ4e-mtp` at
`7dd276b77fcb27c982ebd8d8a434539345713fea`. Image artifact SHA-256:
`6c64740ac1d09365b0099f868114e5c495bac0d1366c27e842c87a9f3a5db647`;
configuration `b4bb4776365308b3e157e21e5fe7777e5650f328879a3f0a4b9981ef978fa2da`.
The unchanged full-head text control is identified in the
[preceding benchmark](../../benchmarks/2026-11-09-qwen-dense-image-text-parity.md).

The typed derived target and predictor configurations exactly equal that
source-conversion text control. All 1,306 target/predictor tensors have equal
dtype, shape and exact payload bytes: 17,077,807,104 bytes compared. The canonical
name/shape/dtype/payload digest is
`d77f8220c2f1b4f2f4eba6bacb635b96346630e829929373108fc4f52f92f2ef`.
Normalization scale offsets and weight specs are preserved. The text graph
changes only the admitted image reference arithmetic/normalization/RoPE policies
back to the ordinary conversion's Legacy text policies. It regains the existing
batched target verification implementation without relaxing M-RoPE or reference
DeltaNet speculation guards.

The image graph, tower and controls stay resident. Only validated raw matrices
and quantization planes share immutable ownership. The measured parameter-buffer
inventory is 18,004,602,336 bytes: 16,743,700,480 shared, 1,260,901,856 unshared,
zero representation-copy bytes. These counters exclude request state, scratch,
allocator slack, kernel storage and process RSS. Startup admission conservatively
bounds a second graph using all packed U8 matrix payloads, additional F32 controls
and a 256-MiB margin; the packed-code bound includes 4/5/6-bit storage and is
deliberately larger than measured extra residency. Optional coarse readout
materializations and unsigned-to-signed A8 copies are counted separately.

Direct checks reject both directions of image/text state mixing before device
execution without fencing the model. A foreign dense session is also rejected.
CPU checks prove shared raw-buffer reuse across numerical profiles, unsigned A8
copy-on-write isolation, and zero unnecessary I8/U4 copies. Normal product
loading keeps the tower; the older tower-omission validator is a reference-graph
control and is not used as this optimized text baseline.

### Paired 64-token speed control

Candidate worker SHA-256:
`34d6d9851c6b6bdf4a69b321eb8df29ba4d3b51a0daa5410a6791e52e4ef8fd4`.
Machine: M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.2. The candidate executes both
unchanged text-only and image-loaded artifacts in ABBA loading order. Each load
runs MTP0 and MTP3, with one excluded warmup and three measured fresh requests
per policy. Workload: identical 16-token input hash, context 2,048, 64 output
tokens, greedy sampling, seed 42, full original predictor readout, no prefix
scope. GPU use is exclusive by coordinator. Every request records paging deltas
and thermal warning status; all 24 measured rows have zero pageout/swapin/swapout
deltas and clear thermal/performance warnings. Background indexing activity was
observed; this remains a quick local control, not a sustained thermal benchmark.

| Policy / metric | Text-only candidate | Image-loaded candidate |
| --- | --- | --- |
| MTP0 useful decode median | 25.156 tok/s | 26.457 tok/s |
| MTP0 decode range | 23.093–26.955 tok/s | 25.094–26.554 tok/s |
| MTP0 first-token median | 0.08123 s | 0.07962 s |
| MTP0 whole-request median | 2.59284 s | 2.46151 s |
| MTP3 useful decode median | 48.309 tok/s | 48.315 tok/s |
| MTP3 decode range | 45.907–50.248 tok/s | 45.674–50.418 tok/s |
| MTP3 first-token median | 0.08206 s | 0.08316 s |
| MTP3 whole-request median | 1.39021 s | 1.39064 s |

All 32 outputs, including warmups, have the same token hash across both arms
and draft depths. Each MTP3 request actually proposes 59 drafts and accepts 43;
MTP0 proposes zero. The image-loaded MTP3 decode median differs by +0.0124%;
its first-token median is 1.34% higher. MTP0 shows no observed regression.
Do not infer an MTP0 speedup from these noisy short samples. Decode, first-token
and whole-request medians pass a stricter five-percent loss bound for both
policies. The registered `max(5%, 2 * baseline relative range)` bounds are much
wider, 30.71% and 17.97%, and do not establish five-percent measurement precision.

One additional load of the sealed predecessor worker
`e41bc8b07681cacb9c6c66d352dd5805cf64baa2285b8287d27edde4a1867d67`
executes the same 64-token text-only workload, one warmup plus three measured
requests per policy. Its MTP0/MTP3 medians are 26.554/45.870 tok/s, with identical
hashes to the candidate. Image-loaded candidate rates are -0.365%/+5.33% against
this prior-worker control. The candidate text-only MTP0 median is 5.26% below
the one-load predecessor median, inside the candidate's observed range; the
paired image-loaded arm directly matches the predecessor. These 64-token rates
must not be compared directly with the predecessor's eight-token 61.091 tok/s
measurement, whose output workload and acceptance history differ.

Image-loaded startup is about 63 seconds versus 1.6 seconds for the text-only
candidate. Full image admission/loading is outside inference timers and remains
a separate cost. No claim is made about startup parity, long-run throughput,
every prompt, other checkpoints or installed application behavior.

### Cache and lifecycle

All 14 controls pass on a nonperiodic 1,792-token input with context 2,048 and
eight generated tokens. Both MTP0 and MTP3 restore 1,024 tokens on immediate
repeats and exactly match fresh greedy output. Draft-depth switches miss; seeded
same-mode cached/fresh/repeated outputs match. Actual engine target position
must equal the full input length after cold or restored-suffix prefill, before
sampling; reported reuse is not the sole correctness check.

A real image request has zero drafts, proposals and reused text tokens. The
next text request starts fresh with MTP3 and the following request restores
1,024 tokens. Scoped cancellation discards its cache/session; fresh text and a
subsequent cached repeat recover on the same loaded worker. Diagnostic cold and
cached whole-request times are about 2.29/0.98 seconds for the initial MTP3 pair;
they are correctness-run observations, not a cached performance qualification.

### Verification and preserved setup failure

Release worker tests: 134 pass, two intentionally ignored. Focused engine
WeightMatrix and ParameterLoader tests: eight pass. Explicitly registered
ignored tests for private canonical configuration and actual loaded graph/state
identity/storage both pass. The pinned-nightly `cpu,metal` test executable is
sealed as `c7b7c00c55b3b30729b1aaac52252ed8c9da639bf54badcbbcb3731fa6b85dd2`.
Independent coordinator review finds no dispatch/storage blocker and recomputes
the paired medians, counts, guard eligibility and strict five-percent loss gates.

The first preliminary benchmark launched before a corrected source seal was
complete: system Python lacked `hashlib.file_digest`. After preserving one
useful pair, its owned third-load worker was deliberately stopped. That partial
receipt retains the resulting `SystemExit`; it is not the qualifying cycle.
The complete cycle above runs after the corrected source/binary/driver seal
using locked, pinned Python. Model payloads stay unchanged. Only this run's
disposable test target was retired after sealing its executable, restoring about
three GiB and leaving 12 GiB free; no model or production target was deleted.

| Private aggregate receipt | SHA-256 |
| --- | --- |
| Runtime source/binary seal | `30ce0fa1ad37e9517f32ada1f74c89c80487787e7902775fa9b115ed0e293a72` |
| Qualifying driver/binary seal | `7af0557a924e95ef36f0b2538bad230657bd7a686e1ed3944a942ace1fbcbe82` |
| Complete target/predictor payload equality | `2c89a4f83d3d602c44417c8feffc004e03f7eea4417e6ec14f47370b325deae7` |
| Parameter storage inventory | `779a300f645ef9365415709c0eb9a458885d894876dce0b59130d460ad8fca42` |
| Preliminary interrupted cycle | `8cef93b48b86de225aab87e46d1a768a52276ad5f11e4c055774ff6964eba17d` |
| Qualifying paired cycle | `3f4f3c81004e02f5f215a854275eb1013a4eaf856519fcecd91fbca101ece4fe` |
| Predecessor worker control | `4fc64e41c3aecd1784392ea020acf5eeba14a21923263f7ac4f10fc7382fa4b7` |
| Nonperiodic near-capacity lifecycle | `32175b13af9b90206e4a754036e7243d9599523d1aa803ecb9dc4975c1ab8fe9` |
| Test executable and check seal | `49d3207aba8d5da4fdf88df3ba7578b728d7d7048375ee7acebf73413654f940` |
