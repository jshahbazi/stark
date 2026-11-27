# 20261126-HY4OC-04 · Retain the prefill tail and parallelize Hy4 routing

## Artifact and controls

Use AngelSlim's original split Q3_K GGUF at
`c50d4dd30be42f2b91138e5731de0e2f32ad56b2` with its BF16 projector, NextN,
and the pinned Tencent metadata. The [resident Stark run](2026-11-26-hy4-resident-opencode.md)
is the workload reference. Its recorded longer continuation restored 12,288
positions from a 16,000-position first prompt. The [Sushi run](https://github.com/jshahbazi/stark/blob/865354abef495cd872488d32d206d46348271eb6/docs/model-experiments/2026-11-26-hy4-sushi-opencode.md)
showed room for both larger reuse and higher prefill throughput at similar
rendered lengths. It used a separate pack and draft head.

| Attempt | Change and criterion | Status |
| --- | --- | --- |
| HY4OC-04A | Compare compensated indexed sparse MLA with the original kernel on one build. Compare synthetic OpenCode digests and measure the selected-cell stage and whole prompts separately. | The compensated stage regressed and changed output hashes. Restored the original default. |
| HY4OC-04B | Retain the last completed IndexPool group of the previous prompt's final prefill chunk, with the existing grid as fallback. The restored continuation must reproduce its fresh replay while reusing more than 12,288 prompt positions when the final tokens match. | Opt-in candidate. |
| HY4OC-04C | Select the router's top 16 out of 512 in parallel. Preserve expert IDs, tie order, selected weights and invalid-row behavior. | Opt-in candidate. Synthetic GPU exactness passed. Full-model timing and parity remain open. |

The prefix candidate takes the end-of-prefill idea from [Sushi's Hy4 live restore points](https://github.com/beamivalice/sushi/blob/v1.8.1/docs/hy4-flash-arch.md#the-state-and-the-prefix-cache).
Stark retains rows in the loaded worker for the immediately preceding scoped
text request. Its cache clears across model unload, cancellation, failures
and images. The router is adapted from [Sushi's Hy4 top-16 selector](https://github.com/beamivalice/sushi/blob/d9e2f9f741799ec367ab00a81d219b5d6b0b37c6/src/hy4flash/hy4_route.zig).
The notice is in [LICENSE.sushi-reference](../../LICENSE.sushi-reference).

## Router component result

On the M5 Max, the incumbent and parallel pipelines produced identical
expert IDs and bitwise equal weights across 7 synthetic groups of 16 rows,
or 1,792 selections and weights. The cases included random scores, ties,
NaNs, all negative infinity, fewer than 16 valid experts, exactly 16 valid
experts and positive infinity ties. The candidate
pipeline admitted 1,024 threads and used 4,608 bytes of static threadgroup
memory. This is a numerical component check.

| Private evidence | SHA-256 |
| --- | --- |
| Shader | `5d4ebd9759c4e515b20cccbce32b95a52e711fbf65a7e178e1e3a1f8db48fab8` |
| Metal check | `53744520d5576abcc4c4690642ba61fb0c5d06046c82a2cc97e1b42b47960b14` |
| Aggregate result | `2e8114ac0871d59b9673b08815fac20035793bc85a3b1064c619d329fa2938b8` |
| Paired timing | `fea8238939103305983a434accd4da3a8fb1f429c7fca82dbef2e6e38d711bf5` |
| 256-thread scratch timing | `eaf4fea245f049f7f818114fe57ee13b161a065917f24fe559042cd3218201ac` |

The same host measured the following median one-dispatch GPU times with
16 interleaved samples per arm. Each width passed an exact output comparison.
The Q3_K download was active during the screen.

| Rows | Original, µs | Parallel, µs |
| ---: | ---: | ---: |
| 1 | 75.625 | 43.021 |
| 16 | 75.313 | 42.563 |
| 4,096 | 243.146 | 268.271 |

The opt-in path applies through 16 rows. A separate 256-thread scratch
variant passed the 4,096-row exactness check and took 600.750 µs against
631.062 µs for the parallel 512-thread arm in that session. The latter
arm was more than twice its prior measured time. The five-percent
within-session difference had insufficient evidence for adoption.
The original selector serves wider batches.

## Planned OpenCode screen

Run the existing fresh 16K, continued 27K and fresh 27K fixtures in OpenCode
2.4.2 on the 128 GiB M5 Max with a 96-token greedy budget, resident experts,
text NextN K2 and 4,096-token prefill chunks. The worker fits context normally
with a 65,536-position upper bound for all arms. Require the same observed
capacity. Record startup, host pressure and swap, prompt and output digests,
reused positions, prefill and decode rates, and client wall time. Reset
the prefix before the independent fresh 27K request.

Run the original and compensated kernels with stage instrumentation in
separate diagnostic sessions. Run each uninstrumented comparison in
control, candidate, candidate, control order. Judge the tail on restored
versus fresh output equality, scope and image invalidation, and continuation
wall time. Judge the router on exact routing and output digests, and price
any whole-request change against the spread of the controls. Two ordered
pairs are a screen. A default change needs the broader quality and timing
gates in the [performance playbook](../PERFORMANCE_PLAYBOOK.md#correctness-gates).

## Build and status

The Hy4-enabled Rust 1.98.1 release build passed 183 generation tests and
all-target Clippy with warnings denied. The pinned nightly engine test
passed the IndexPool boundary cases. The recovered OpenCode harness
completed all three mock client turns. The exact Q3_K download and
checksum preparation gate the real-model screen.

### OpenCode host gate

Review of the live OpenCode path found that the Python host required all
worker reuse counts to land on the compute-chunk grid. An IndexPool tail
count such as 15,996 would fail the continued request. Hy4 now advertises
the optional `subchunk_tokens` alignment when tail reuse is enabled. The
host accepts in-range counts on either advertised grid. A synthetic API
request passed with a finer reuse count, and validation still rejects
misaligned counts and counts that consume the whole prompt.

The rebuilt Rust 1.98.1 release worker used for the queued screen is
bound by SHA-256
`c4c92f5af7d2378ddb205a7a2ec84904ae6565c68a4f1a746bdf9b99780ebaae`.

### Artifact recovery

The first Hub transfer reached 137.5 GB of 156.4 GB before an Xet CDN
request returned HTTP 403. The installer exited while the benchmark runner
was waiting for a verified model. Recovery reused 64.4 GB of completed
files and switched the missing trunk shard to HTTPS byte ranges with
per-range checkpoints. The pinned SHA-256 and native installer remain the
publication gates for the exact model.

All 156,415,403,072 weight bytes passed the native installer on November 27.
The range utility's first handoff used stdout as the installer's progress
descriptor and failed before preparation. Relaying progress through an
inherited private descriptor completed installation using the staged files.

## Selected-cell MLA diagnostics

Two instrumented cases completed on the M5 Max with the 65,536 upper bound
and greedy NextN depth two. The tool-enabled OpenCode requests rendered to
16,015 and 27,015 positions and each returned 96 tokens. The rendered input
and sampling settings matched within each pair. Generated output hashes
differed at both lengths.

| Fresh prompt | Original MLA, s | Compensated MLA, s | Original prefill GPU span, s | Compensated prefill GPU span, s |
| --- | ---: | ---: | ---: | ---: |
| 16,015 | 4.923 | 7.731 | 26.851 | 29.096 |
| 27,015 | 9.440 | 15.148 | 48.919 | 55.373 |

The compensated selected-cell stage took 57% and 60% longer. Treat these
as diagnostic GPU timings. The 16K original arm also spent 11.23 seconds in
Engram versus 2.26 seconds in the compensated arm. Compare uninstrumented
whole-request timing and fixed-position output hashes separately.

| Private diagnostic aggregate | SHA-256 |
| --- | --- |
| Original | `0fd4a49955bd3e6650cd500625cf87757160d587372a55b33ad39b34e490c687` |
| Compensated | `73ff0b4889d90f7524eeaba794f1b6b2c9309facbdd25ff2a38caa90d7dbd594` |

A macOS watchdog restart interrupted the first uninstrumented case during
startup. Restarting changed the filesystem device identifier in the weight
fingerprints. Stark refused the stale receipt, so the original three GGUFs
and pinned sidecars were verified again before the retry. The retry used
a 32,768-position upper bound to reduce resident cache allocation while
keeping both prompt lengths and the 96-token budget in range.

At 32,768 positions, OpenCode performed extra generations and changed the
rendered history on the long turns as it reserved context for compaction.
Discard those timings. The resumed uninstrumented screen uses a 49,152
upper bound and checks the rendered prompt lengths, generation count, draft
depth and sampling mode on every case.

## Uninstrumented MLA comparison

With the 49,152 upper bound, the pairs used identical rendered fresh prompts,
greedy depth two and 96 generated tokens. The original and compensated
kernels returned different output hashes at both lengths in both pairs.
Within each kernel, the fresh output hashes were repeatable and matched
the corresponding instrumented arm. Fresh 16K first-request overhead
varied within each kernel. Compare fresh 27K in pair B, where the preceding
request was fresh 16K in both arms. Pair A's compensated arm had a resumed
request in between and illustrates run order sensitivity.

| Fresh prompt | Prefill tok/s, original | Prefill tok/s, compensated | Decode tok/s, original | Decode tok/s, compensated | OpenCode s, original | OpenCode s, compensated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K, A | 224.31 | 248.53 | 39.97 | 41.85 | 74.565 | 67.440 |
| 16K, B | 386.85 | 292.37 | 37.47 | 41.70 | 44.723 | 57.837 |
| 27K, B | 541.61 | 495.15 | 36.34 | 37.39 | 53.198 | 58.070 |

The host reported memory pressure level 1 and no other inference workers.
Swap ranged from 62 to 283 MiB. Worker startup took 80–91 seconds. The
original 27K arm improved prefill by 9.4% and cut client wall time by 8.4%.
The small decode difference comes from sequences with different token
digests. Keep compensated MLA opt-in. The default change retains resident
experts, the 4,096-token prefill chunk, and memory-fitted context.

| Private timing aggregate | SHA-256 |
| --- | --- |
| Compensated A | `de5c0ff65eb61c8346a05a19b40af5a4e5e07e1f3974fca3bfcd6e9c76e69c24` |
| Original A | `57ca3606e0d381814dc2fb9291d92675798efa2b09aa3f9e3855af66c3e37913` |
| Original B | `145893bf4555723f90e644622dfeccc59ed19601aa5965347fb54ceecbeb73bb` |
| Compensated B | `2066c2575e323ac970b9671ab4732f6841b8e79e58e6ee631d44406e32c38267` |
