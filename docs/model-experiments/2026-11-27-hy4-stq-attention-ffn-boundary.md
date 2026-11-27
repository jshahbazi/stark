# HY4STQ-PERF-05 · Fuse the attention-to-FFN boundary

The starting worker is `8c0f53679`. The previous tuned OpenCode run
measured 34.33 s fresh 27K prefill and 45.67 useful decode tok/s.
It matched the original outputs and draft counts at 16K and 27K.
[Previous tuning and fixture identities](2026-11-27-hy4-stq-decode-prefill-followup.md)

This run uses `beamster/Hy4-Flash-Sushi-STQ-4bpw` at
`61528ba635a623f127c5a7b1e1f4ac2c2b0d4885` on the 128 GiB M5 Max
with the vision tower loaded, F16 KV, 4,096-row chunks, DFly2 depth
five and a fitted 49,152-position ceiling. OpenCode 2.4.2 renders
16,229 fresh, 27,179 continued and 27,229 fresh input tokens with
96 greedy outputs. Prefill counts computed positions. Useful decode
counts 95 output transitions. These are local OpenCode diagnostics.

## Fresh 27K profile

A private stage-instrumented worker from the starting revision measured
294 blocks and 287 routed cuts after subtracting the startup image
check. It reproduced the historical 27,229-position input and output
hashes, zero prefix reuse, 120 proposals, 70 accepted drafts and
25 rounds. The instrumentation accumulates GPU command spans at the
existing profiled cuts.

| GPU stage | Seconds |
| --- | ---: |
| Attention output RMSNorm | 0.048 |
| Four-stream attention residual write-back | 0.260 |
| FFN input mix and norms, including the F32 projection | 0.706 |
| Attention-to-FFN subtotal | 1.014 |
| FFN-to-next-layer norm and residual write-back | 0.297 |
| Attention input mix and norms | 0.715 |
| Routed expert products | 7.519 |
| Routed selection | 0.461 |
| All measured prefill GPU command spans | 33.543 |

The attention-to-FFN subtotal accounted for 3.02% of measured GPU
spans and includes GPU work the candidate retains. The routed cut
callbacks consumed 37 ms of overlapping host time. The instrumented
request took 35.85 s to prefill, with 1.92 s of Engram processing.
The first worker hit a 240 s startup timeout before processing any
fixture, with Engram page caching off. The matched retry enabled that
cache and used a 420 s startup allowance. It reported ready in
99.61 s. Memory pressure stayed normal.

## Boundary and candidate

Sushi fuses the attention output norm, residual expansion and the
FFN mix and norm. Stark already combines the post-projection gates,
stream collapse and FFN weighted norm. Its output RMSNorm, residual
write-back and the four-stream input RMSNorm were three dispatches.
The candidate performs those three operations in one threadgroup
per row, using the incumbent F32 reduction order for both RMSNorms.
It retains the existing F32 8-by-12,288 mix projection and downstream
fused FFN mix/norm. It writes the same residual and normalized
planes without the weighted attention-output scratch plane. The
dump path uses the existing graph.
[Sushi layer implementation](https://github.com/beamivalice/sushi/blob/v1.8.1/src/hy4flash/hy4_model.zig#L2787-L2803)

The DFly2 candidate batches the greedy correction slots on one
command graph. Each slot uses the existing argmax and validates
its selected logit before the next GPU embedding reads the ID.
Confidence requests and one-slot proposals keep their original
dispatch. This reuses the earlier real-weight prototype.
[Earlier DFly2 screen](2026-11-27-hy4-stq-decode-prefill-followup.md#component-experiments)

## Component screens

| Attempt | Result |
| --- | --- |
| HY4STQ-PERF-05a · Fuse the three boundary dispatches | All 115,064,832 F32 words matched at 1, 3, 6, 64, 512 and 4,096 rows. The fusion saved 15–30% over nine paired GPU samples at each width. At 4,096 rows the per-layer median fell from 1.820 to 1.545 ms. |
| HY4STQ-PERF-05b · Batch DFly2 correction with GPU target gather | At the 27,229-position geometry with register MLA and GPU gather in both arms, all 198 proposal IDs matched. The confidence fallback and four invalid or tied argmax cases also matched. Five-draft median wall time went from 11.878 to 11.196 ms at the maximum selected-key width. The one-draft case was flat. At 25 rounds the component delta projects about 17 ms. |
| HY4STQ-PERF-05c · Shared vocabulary head retile | The exact one-split K32 arm matched 1,329,152 score words and 11 winners. It moved a single GPU head invocation from 0.606 to 0.588 ms, worth about 2 ms per request. The faster four-split arm changed score bits. Both stayed out of the worker. |

The fused shader uses the incumbent F32 arithmetic for both
standalone norm shapes, including the small-row reduction order.
GPU prefill accounting now charges the FFN flat norm to the
combined attention boundary stage. Compare wall time and total
GPU spans across runs.

## OpenCode rerun

The release worker enabled `STARK_HY4_STQ_FUSED_ATTN_HC=1` and
`STARK_HY4_STQ_BATCH_DRAFT=1` on top of the preceding tuned STQ
settings. All five OpenCode turns matched the historical rendered
input and output hashes, draft counts and prefix reuse. A prefix
reset preceded each of the three fresh 27K turns. Each generated
96 tokens after computing all 27,229 input positions, with 120
proposed drafts, 70 accepted and 25 rounds.

| Fixture | Prefill s | Computed prefill tok/s | Useful decode tok/s | Reused positions |
| --- | ---: | ---: | ---: | ---: |
| Fresh 16K | 19.45 | 834.27 | 54.45 | 0 |
| Continued 27K | 17.93 | 830.28 | 56.59 | 12,288 |
| Fresh 27K, median of three | 31.87 | 854.48 | 51.88 | 0 |

All three fresh 27K runs cleared 48 tok/s. Their paired prefill
seconds and decode tok/s were 32.67 and 53.35, 31.85 and 51.88,
and 31.87 and 51.09. GPU prefill spans were 31.60–32.39 s.
Draft and verification medians were 0.331 and 1.499 s. The earlier
build's single fresh 27K observation was 34.33 s and 45.67 tok/s.
The separate-process history includes variation outside the
fused boundary and batched corrections.

The continuation reused 12,288 positions. Its fresh replay
computed every position and produced the identical output.
Requests at 8,192, 8,192 and 12,288 positions reused 0, 4,096 and
0 positions and produced identical outputs. The actual image
request disabled DFly2 and invalidated the shared text prefix.
The following text request replayed from zero and matched the
pre-image output.

## Same-binary control and decision

The same worker ran again with only the two new options off.
Both loads reported ready in about 108 s. Memory pressure stayed
normal. The controls matched the five-draft setting and the
candidate's input and output hashes, reuse and draft counts.

| Fixture | Prefill s with both off | Computed prefill tok/s with both off | Useful decode tok/s with both off |
| --- | ---: | ---: | ---: |
| Fresh 16K | 18.51 | 876.74 | 52.65 |
| Continued 27K | 17.02 | 875.12 | 55.97 |
| Fresh 27K, median of two | 32.42 | 841.75 | 51.62 |

The two fresh 27K control turns took 30.86 and 33.98 s to prefill
and decoded at 51.72 and 51.51 tok/s. The candidate's median saved
0.56 s of prefill and added 0.27 decode tok/s against that control.
Both ranges overlap and both modes cleared the 48 tok/s goal.
Control medians were 32.151 s of prefill GPU spans, 0.343 s of
draft and 1.497 s of verification. The shorter candidate fixtures
had slower prefill and faster decode than their single controls.
This bracket establishes the rate on these loads. The component
checks isolate the dispatch and correction savings. The whole-turn
timings do not establish a repeatable gain from enabling both.

Keep the two settings opt-in on the native STQ branch. The next-layer
write-back still uses the existing separate dispatches. The best
prior Sushi v1.8.1 observation was 27.58 s under Sushi's renderer and
KV8. The 31.87 s Stark median is 4.29 s above that observation.
[Previous Sushi comparison](2026-11-27-hy4-stq-dense-attention-prefill.md)

Repository policy, Rust formatting, release Clippy and the native
inspection example passed. Release Rust tests passed 183 with six
ignored. The focused converter and runtime suite passed 189 tests.
Ruff lint and formatting passed.

## Evidence identities

| Artifact or receipt | SHA-256 |
| --- | --- |
| Previous OpenCode baseline aggregate receipt | `4914d9e2bcd064d115d91cfe34a493253dcb879eaafc3744fc3ae550bcb85019` |
| Private boundary profile worker | `23847bebd109bb0e3bdda2ee5c4525bac5131c5052d3e7e1f6ff155c420b2a1f` |
| Private profile instrumentation patch | `cc1daa46d0dd73a49c6ddccbc220565618b79748cdb401008a29c4230febb2b9` |
| Profile startup timeout receipt | `f8593af5671c96647c9f0d8bde9073fb938254c07f935b4b4437313dd9c5f0b7` |
| Matched boundary-profile aggregate receipt | `4062004dedef2d795a0b92bc649ef16b919f7a26baae32dd1cee59699b2bcf99` |
| Matched boundary stage totals | `fee9b052041a7ef6a8e490e777ac447de88592041dd17879273ad91ee52e39b3` |
| Boundary component fixture | `0910c5717a36f800219c0e6b19f7af41c210d213d9990a4931f609e50095d577` |
| DFly2 correction component worker | `e674a605f21b6091ff823e31e776eca45c84799496f2f142f29f300ea33894f2` |
| DFly2 correction aggregate receipt | `54a1984b775a5276e2d90e88091cd7128a6ebabd016f8031ac137ce694bf5c6a` |
| Release worker with current embedded Metal sources | `8ed2135a65a5d870584c9de7b094262d69bdece006d23a24aa4bf105a7b55388` |
| Fusion and batched-correction OpenCode aggregate receipt | `d9c1c650d33debfaec07a17d82eb522204f515ab006e880a6265b01089b37afc` |
| Same-binary flags-off aggregate receipt | `26c46eef195ff114634d80fe952814f00d7f2737be5437aadd4791ff77461a08` |
