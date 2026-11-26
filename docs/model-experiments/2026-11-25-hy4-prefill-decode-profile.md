# AngelSlim Hy4 Flash 4K prefill and decode attribution

## Artifact and controls

Executed November 25, 2026, on a 128 GiB M5 Max (Mac17,7), macOS 26.7.2
build 25G332, on AC power. These are worker diagnostics for the verified
AngelSlim Q3_K split GGUF at `c50d4dd30be42f2b91138e5731de0e2f32ad56b2`
with its BF16 projector loaded. The [native adoption
record](2026-11-25-hy4-native-q3-k.md) binds the original payload hashes, source
receipt, Tencent tokenizer revision, pinned Strata graph and llama.cpp
comparison. This work starts at Stark `f235488cd0ba2f323493f72fe6b2958bede4f8f9`
and builds with Rust 1.98.1, `--offline --locked --release -p stark-generate
--features hy4`.

The worker receives the existing 4,096-token fixture with an 8,192-token
context, a 64-token cap, greedy sampling, seed seven and two NextN drafts.
All calls disable prefix reuse. Through-first-sample and post-first decode
are separate measurements. The rate uses 63 post-first tokens. The fixed
correctness gate requires the input hash, all 64 reference token IDs, the
projector readiness contract and valid draft counts. The fixture and
historical llama.cpp reference hashes are recorded in the [native adoption
record](2026-11-25-hy4-native-q3-k.md). Only aggregate measurements and
digests leave the private harness.

## 20261125-HY4PROFILE-01 · First request with mapped experts

The existing worker `7704b413d378075500a6573e3f374dd5647bcff61f78a97adcf9ca003020ab17`
loaded mapped experts, the GPU planner and grouped F32 with a 4,096-row
chunk. Readiness took 1.62 s. It produced zero samples before the trial
worker was terminated at about 167 seconds. System swap was 293.94 MiB
before the run and reached 65,460.50 MiB in an observed sample. Swap fell
after termination. This is a resource failure and produces no timing
comparison or output parity result. The owner stopped the benchmark's own
worker and retained the streamed path for subsequent measurements.
Receipt SHA-256 `9e5d9899015a03e9596d5be88679737e2b5d12695183be1f2de9ec3ea77888fb`.

## 20261125-HY4PROFILE-02 · Attribute the streamed prefill

The opt-in profiler uses Metal encoder counter timestamps in the same
command buffer and preserves the host event order. It splits encoders at
stage boundaries, which can perturb execution. Use its output for
attribution and use flags-off requests for throughput. An unavailable or
invalid sample is counted through the number of profiled blocks.

The first profile worker
`343f2cac716a31ab30e90ebef56616dc1ee58a2b4badc52676e0728b3d251f54`
captured all 42 blocks, matched the 64 reference tokens and accepted
41 of 43 drafts. The process's first 4K request took 39.807 s through first
sample, with 16.935 s of reported expert read waits and 84.15 GB of
software expert reads. Software reads can include file-cache traffic.

| Sampled encoder stage | Seconds |
| --- | ---: |
| Attention, dense attention projections and indexer | 5.447 |
| Routed expert preparation, gate/up, down and combine | 4.439 |
| Dense FFN, shared and latent projections | 0.726 |
| MoE route | 0.045 |
| Remaining block work | 0.227 |

The expert wait and encoder spans have different boundaries and can overlap.
Of 52,347 grouped M64 tiles, 11,173 contained at most 32 valid entries
(21.3%). Even halving the nominal matrix work on those tiles would remove
about 10.7% of grouped matrix work. The optimistic estimate against the
total profiled routed expert time is about 0.47 s and excludes any
descriptor or scheduling overhead.

The profiler's 8.036 s decode wall time and exact output are valid. Its
decode GPU-span counter contained an uptime-sized duration, which is
invalid. A subsequent build rejects zero starts and spans exceeding the
host-observed command interval, and reports the excluded-sample count.
Profile receipt SHA-256
`f4ada342228e4fbca8fdbbec75e8885d0d58c5e19d6d0a52c5725927f7f6ae50`.

The candidate worker below repeated the baseline profile after memory
pressure returned to normal. It matched all 64 IDs, sampled 42 blocks and
excluded zero invalid command spans. Through-first-sample time was 13.766 s,
with 13.255 s of command GPU spans and 0.098 s of expert read waits. Stage
times were 6.573 s attention, 5.309 s routed experts, 0.838 s dense FFN,
0.057 s routing and 0.221 s remaining work. The initial 16.935 s wait was
specific to that run after the mapped failure. The remaining attention
projection, index and attention kernel costs need a separate split before
choosing which to change. Receipt SHA-256
`9336322c532ef6cd02b418cbb6d3cc036ad8ba538d99e36267d73ad6cf9a3dd4`.

## 20261125-HY4PROFILE-03 · Smaller F32 grouped expert tiles

The candidate used a 32-row descriptor for a grouped gate/up or down tile
with at most 32 valid entries. It kept the original input and accumulation
precision, group layout, barriers and file read policy. There was no
measured prefill win, and the implementation was removed. The occupancy
counters remain available in the opt-in stage profile.

## 20261125-HY4PROFILE-04 · 36 GB streamed expert arena

The worker reused Strata's anonymous expert arena and live allocation
guards. An explicit 36 GB request must also clear the GPU, RAM and default
process footprint budgets after reserving the target's workspace and vision
scratch. The device reported 35,998,924,800 bytes against 27,996,717,056
for the ordinary arena. No precision, prefix or expert arithmetic changed.

Attempts 03 and 04 used one binary and flags-off stage sampling. Arms ran
serially in the order below. Each first request followed a worker restart
without clearing the OS file cache. The warm request was a second full pass
in the same worker with prefix reuse disabled. The host screen found no
competing heavy process or thermal warning. System swap fell from 6,865 to
6,697 MiB during the bracket.

| Arm | Request | Prefill s | Decode s | Decode tok/s | Target verify s | Expert reads during decode, GB | Target host cut s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 GB baseline | First | 12.253 | 4.348 | 14.49 | 3.717 | 49.90 | 2.130 |
| 28 GB baseline | Warm | 12.765 | 3.843 | 16.39 | 3.371 | 43.89 | 1.838 |
| 32-row tail | First | 14.328 | 4.402 | 14.31 | 3.747 | 50.69 | 2.112 |
| 32-row tail | Warm | 15.328 | 3.753 | 16.79 | 3.291 | 43.49 | 1.828 |
| 36 GB arena | First | 13.170 | 3.663 | 17.20 | 3.020 | 39.37 | 1.633 |
| 36 GB arena | Warm | 14.784 | 2.672 | 23.58 | 2.223 | 27.60 | 1.118 |
| 28 GB baseline | First, end bracket | 15.133 | 4.354 | 14.47 | 3.692 | 50.32 | 2.096 |

All seven requests matched all 64 reference tokens and accepted 41 of 43
drafts in 22 rounds. Each target decode encoded 110 command buffers, 2,816
encoders and 57,898 dispatches. Target CPU encode took 36–42 ms, commit calls
took under 1 ms and Engram waits took 2–3 ms. Host cuts overlap GPU command
spans and can include expert read waits. Target verification accounted for
3.37 of 3.84 warm decode seconds at 28 GB. The 36 GB arm cut software
expert reads 21% on first use and 37% warm against the starting baseline,
with 19% and 44% higher decode throughput. This is evidence of a decode
residency benefit. The prefill baseline moved 24% in the bracket, so the
timings leave the 36 GB arena's effect on prefill unresolved. The larger
arena stays opt-in pending more repetitions, memory-pressure coverage and
natural text checks. It does not establish llama.cpp throughput parity.

Candidate worker SHA-256
`61d1a79a7a5f097229e28126ad7032571f229640eba7a3df73b4ed75531107db`,
private source diff SHA-256
`c4d0fdc040c52c5fc7459b2d76bbac252c88eee2ca825cc890d9a68257733c34`,
and harness SHA-256
`30ecfc895b68322c983f5a33c4320b87b3f269ef920714a05eba252fb98cd7ee`.
Aggregate receipt digests in run order:

| Arm | SHA-256 |
| --- | --- |
| 28 GB, first bracket | `35f538ba0cc067d54e12734a4cdceee27e94573f6a0bb5d93f8c95a45c3b736c` |
| 32-row tail | `f81ad56c57a57267c55176f7eb940f6817a0ec84933cfe3750f957e52ab34389` |
| 36 GB | `5730d43ff634ed1fdb78b40264b000d932035fb039baf47eb1bc097a58ceeba3` |
| 28 GB, last bracket | `d850e144fe9e0074f503a3520636d540b3406d2269c629b0759d51641faa6367` |

## Final worker validation

The release build after removing the tail kernel had SHA-256
`e01906017ceafae57ee9e9c09f178a8dad38c77ca3cb60a3496b32c9ca06bccc`.
With flags unset it loaded the projector, kept the ordinary streamed arena
and 1,024-row chunk, and matched all eight short-fixture reference tokens.
With the 36 GB arena and profiling selected at the 1,024-row geometry, the
image-loaded worker matched all 64 4K reference tokens, accepted 41 of 43
drafts and sampled 168 blocks across four chunks. It excluded zero command
spans and restored 3,072 tokens with an exact continuation under depth two.
This checks the final build's controls and prefix restoration. The table
above remains the uninstrumented throughput evidence from the candidate
build. No application installation or new standard score was produced.

Flags-unset receipt SHA-256
`47e3f16be478046ee291e33a2481dc55ae6fb8e3897be4a4316ab2dcf55ef227`.
Arena, profiler and prefix receipt SHA-256
`916f52e517e518d25668ef10940a44d5fe9e95cfbe48b961295ca03f0065e046`.

## Experiments to follow

- Prefill. Repeat the dense and attention decomposition used on Inkling.
  Separate the dense projection dispatches, IndexPool and MLA kernels
  within the 6.573 s attention stage. Screen the largest projection with
  the existing tensor tile before changing attention or grouped expert
  arithmetic. [Inkling stage decomposition](2026-11-23-inkling-mini-prefill-round4.md#20261122-inkfastpre-06--dense-projection-decomposition)
- Decode. Use the GLM cache-capacity method to repeat the 28 GB and 36 GB
  comparison with fresh and warm requests in both orders. Use resulting
  expert bytes and verify wall time to price reducing evictions within
  a fixed arena. [GLM cache capacity sensitivity](../benchmarks/2026-11-04-glm-mlx-stream-comparison.md#cache-capacity-sensitivity)
