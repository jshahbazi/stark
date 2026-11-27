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
| HY4OC-04A | Compare compensated indexed sparse MLA with the original kernel on one build. Compare synthetic OpenCode digests and measure the selected-cell stage and whole prompts separately. Use natural-text quality and image/prefix checks for production qualification. | Exact-artifact evaluation awaiting the Q3_K download. |
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
