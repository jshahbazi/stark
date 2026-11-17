# Edge0 stage attribution and direct-route schedule

Attempt `20261117-EDGE0-21`, 2026-11-16 and 2026-11-17, local non-commercial
evaluation only. An opt-in dispatch schedule raised observed median prefill
from **15.16 to 20.56 tok/s** and decode from **11.95 to 13.92 tok/s** in one
binary, with MTP off and matching outputs. Every F32 value the two schedules
were made to compare, 4,405,000,512 in total, had identical bits. These
sequential observations do not establish full performance qualification. The
selector defaults off on both loading routes.

## Workload and build

The artifact remains
`Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`, with native
preparation receipt SHA-256
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`. Original
stored formats and F32 activation, accumulation, recurrent-state and cache
policies remain. Payloads were not rehashed.

The short workload repeats [attempt 19](2026-11-13-edge0-singleton-speed.md):
the historical diagnostic route with virtual Metal, selected expert-record
bindings, singleton kernels, context 512, a 128-token input, greedy seed 42,
explicit MTP zero and a 16-token output limit. Prefill rate is 128 divided by
native prefill seconds; decode rate is 15 divided by native decode seconds,
excluding the initial token. Input SHA-256
`de3b19a046b7211189a24690cb7ec98de34aebf79a7e142153d29cee92b6155d` and output
SHA-256 `3033ab1a108b72992b627a19aa7fd547c0ae6c72ac2be42c01b77e6022890047`
match the attempt 19 reports in every request of this record.

The host is an Apple M5 Max with 128 GiB, macOS 26.7.2 (25G332). The worker is
built with Rust 1.98.1 from base `9280d3182b4e288b00b291ff58b648e50a0d9201`
plus the changes of this attempt. A self-contained private runner starts one
worker per session and sends one warmup before the measured requests. It adds
no quiet admission, host screen or paging sampler; host activity is
uncontrolled. Aggregate native profiling is enabled in every session. No
prompts, generated text or token vectors are published.

## Stage attribution

`21-STAGE` labels the existing encoder timestamp transport by Edge0 stage.
`STARK_QWEN38_ENCODER_TIMING=1` with `STARK_EDGE0_STAGE_TIMING=1` labels stages
at their existing encoder scopes; level 2 also splits each expert and U job.
Each timed encoder reports its dispatch count. The labels encode nothing unless
both selectors are set.

The untimed grouped schedule spends 62.9 ms of command GPU time per decoded
token and 58.6 ms per prefill token. Level 1 raises those to 65.3 and 64.3 ms,
level 2 to 69.7 and 68.1 ms, so shares below carry a few points of timing
overhead. Each cell is the median of three measured requests in one session.

| Stage, level 1, GPU ms per token | Grouped decode | Direct decode | Grouped prefill | Direct prefill |
| --- | ---: | ---: | ---: | ---: |
| U projection | 15.84 | 7.60 | 13.93 | 6.73 |
| Selected experts | 15.55 | 13.40 | 14.70 | 12.90 |
| GDN and attention input projections | 8.22 | 8.12 | 10.00 | 6.12 |
| Hyper-connection mixes | 7.72 | 7.65 | 6.99 | 6.75 |
| GDN and attention state and output | 6.80 | 6.74 | 6.07 | 5.80 |
| D projection | 4.66 | 3.88 | 8.51 | 2.97 |
| Router and shared expert | 2.39 | 2.38 | 1.73 | 1.23 |
| Vocabulary readout | 2.44 | 2.42 | 0.02 | 0.02 |
| Other labelled stages | 0.84 | 0.87 | 0.43 | 0.42 |
| Gaps between encoders | 0.80 | 0.33 | 1.81 | 1.11 |
| Command GPU total | 65.26 | 53.38 | 64.25 | 44.02 |
| Dispatches per token | 7,501 | 4,294 | 5,085 | 3,640 |

The level 2 pass of the grouped schedule adds these decode observations:

- A U job with two or more routes uses the multi-token kernel and takes 110
  microseconds for 3.4 routes on average. A one-route job on the singleton
  kernel takes 23 microseconds for the same matrix.
- Fused expert gate/up takes 20.6 microseconds per expert and expert down 10.5.
- Copies into and out of expert and U jobs take 3.7 ms per token, about 2
  microseconds per dispatch. Splitting encoders added about 2 microseconds per
  extra encoder.
- The Q8_0 singleton kernel covers the 635,699,200-weight readout in 2.44 ms,
  3.8 picoseconds per weight. Q3_K and IQ4_NL expert jobs run near 6 to 8.
- In prefill, GDN input projections cost 206 microseconds per token and layer
  in eight-token batches, against 175 for one token. D runs for 7.1 token rows
  per token and layer where routes select 4.2. Expert jobs average 1.75 tokens.

## Direct-route schedule

`STARK_EDGE0_DIRECT_ROUTES=1`, fixed once per process, selects the schedule:

- Every packed Q8_0, Q3_K or IQ4_NL token row uses the attempt 19 singleton
  kernel, including trunk and shared projections of a prefill batch.
- Each route's gate/up, down and U rows read and write that route's own slots.
  No latent row is gathered or scattered.
- D runs only for the token rows a route of that group selects, and writes its
  stored column-scale quotient in the same dispatch.
- D rows and U rows are each encoded as one projection level.

Route slots, per-route U, the 256-column stride and the ordered eight-group F32
sum are unchanged, and `edge0_graph.metal` is byte-identical. Chained rows carry
no separate finite scan: each kernel records sticky failure for every value it
writes, and a nonfinite input makes every row result nonfinite. Per-expert
encoder scopes remain.

| Follow-up | Change | Prefill tok/s | Decode tok/s | Disposition |
| --- | --- | ---: | ---: | --- |
| `21-DIRECT` grouped control | Selector off | 15.14 | 11.96 | Reference for next row |
| `21-DIRECT` | Direct rows, serial D and U | 19.95 | 13.73 | Superseded by `21-LEVELS1` |
| `21-LEVELS` control | Direct rows, serial D and U | 19.66 | 13.63 | Reference for next two rows |
| `21-LEVELS1` | D and U rows as projection levels | 20.62 | 13.97 | Retained |
| `21-LEVELS2` | Also one scope per layer for all expert rows | 20.97 | 14.33 | Not retained |

`21-DIRECT` rows are medians of six measured requests over two sessions, in
off, on, on, off order. `21-LEVELS` rows are medians of three measured requests
in one session each, in level 1, level 2, control order, from one experiment
binary whose hash was not recorded. `21-LEVELS2` merges every selected expert
of a layer into one encoder scope. Its effect on the paging screens of
attempts 12 to 14 is untested, so its additional gain was left out and its
selector removed. The patch is retained privately.

## Final observations

The final binary contains `21-LEVELS1` only. Four sessions ran in off, on, on,
off order, each with one warmup and three measured requests.

| Median of six measured requests | Selector off | Selector on | Ratio |
| --- | ---: | ---: | ---: |
| Prefill tok/s (range) | 15.16 (15.14 to 15.18) | 20.56 (20.10 to 20.59) | 1.356 |
| Decode tok/s (range) | 11.95 (11.93 to 12.01) | 13.92 (13.90 to 13.98) | 1.164 |
| End-to-end output tok/s | 1.65 | 2.19 | 1.327 |
| Prefill command GPU ms per token | 58.60 | 42.31 | |
| Decode command GPU ms per token | 63.06 | 52.19 | |
| Prefill blocked wait ms per token | 63.49 | 46.49 | |
| Decode blocked wait ms per token | 77.63 | 66.51 | |
| Prefill scratch allocated, MB per request | 7,268 | 4,745 | |
| Encoder resource boundaries per decoded token | 680 | 528 | |

Both arms still wait on 49 command buffers per decoded token. Categories are
inclusive and must not be summed.

The ordinary local loading route (`STARK_EDGE0_LOCAL_EVALUATION=1`, context
8,192, MTP unavailable) ran a 1,024-token input with a 16-token output limit.
Each arm is one session with one warmup and two measured requests; every
request has input SHA-256
`de279b6197f023b5b44c645007f0c5ea383a13369c4b8e07abf97b5e569f90de` and output
SHA-256 `8c98070cbe13df012fecf2e47995e32ca6508dbbb2aca37bb9b0c51523c90503`.

| Local route, 1,024-token input | Selector off | Selector on |
| --- | ---: | ---: |
| Prefill tok/s | 14.73 | 18.59 |
| Decode tok/s | 11.55 | 13.56 |
| Request wall seconds | 70.85 | 56.20 |

## Correctness and limits

`STARK_EDGE0_DIRECT_ROUTES=verify` encodes both schedules in the same command
and compares on the device. Trunk, router and shared projections of a token
batch compare split rows against the batched kernel. Every latent stage
compares the expert rows and the expanded rows. A differing bit sets the sticky
failure status and rejects the request. An aggregate counter records the values
compared.

| Verification request, final binary | Input / output tokens | Comparisons | F32 values |
| --- | ---: | ---: | ---: |
| Diagnostic route, corpus offset 0 | 128 / 20 | 10,464 | 432,311,376 |
| Diagnostic route, corpus offset 128 | 128 / 3 | 8,832 | 396,799,056 |
| Diagnostic route, corpus offset 256 | 128 / 11 | 9,600 | 413,510,736 |
| Local route, corpus offset 0 | 1,024 / 16 | 69,120 | 3,162,379,344 |
| Total | | 98,016 | 4,405,000,512 |

All four requests succeed. The diagnostic-route requests allow 48 output tokens
and stop at end of sequence. Verification covers the selected schedule against
the grouped schedule of the same binary; it does not re-qualify either against
the frozen reference.

Model-free native tests add bitwise equality of split rows, fused D quotient
and fused gate/up rows against the batched kernels for Q8_0, Q3_K and IQ4_NL at
columns 512 and 1792 and one to eight tokens, a synthetic three-route
composition for one, three and eight tokens, rejection of nonfinite inputs
without a separate scan, invalid stored scales, bad shapes, aliased rows and
truncated spans, and rejection of a one-ulp or signed-zero difference by the
verification kernel. Test counts are in the
[attempt record](../model-experiments/2026-11-17-edge0-direct-routes.md).

| Evidence binding | SHA-256 |
| --- | --- |
| Final Rust 1.98.1 worker, both settings | `a34d4c84815d2682ec473a43c564211823065ccd2736a79d2097d3e063e743c5` |
| `21-DIRECT` worker | `848c651146bd335c4bb7418d56f5a18ab6acce5e45a81f5224c89f2fbb3e7a4f` |
| `21-STAGE` worker | `67ddb9069cc42a1d` (first 16 hexadecimal digits recorded) |
| `edge0_direct.metal` | `656411e2170487f2486116ecb996445be17a53582d00a2fb068e82a9f44079f8` |
| `edge0_graph.metal`, unchanged | `b7b8c86e1abe7c3bf1208415ca490b0a9676926e6a52408fc7d5235d8aa97678` |
| Private runner | `d6c469b45b6ee4b35199202e864107ba755c8de2b3ff1c3d2b2d77aa79f218e0` |
| Matched summary | `35968c79d5a6b26220b5132d7fc70d9f99a84f93114ea29776d085182a42a406` |
| Final off profiles | `ca40f83d9bb615f4f471a0c43eefd9381e34077ae1cf62a5155b19c6ca5926b3`, `4ad9747970ac2a8296cc01603c929777a6b9c83f8714d36ba63aef7e3c4bbb0d` |
| Final on profiles | `55234cbac07c0e0b74fecfa00c82bd6b2b50e65776e30fef9b9d1060a8ad25c2`, `1c20bea306e09f9095ebcbfd3ea7c7a9fa93f0f92a20ef2f7ec9fad935cb492b` |
| Grouped stage profiles: untimed, level 1, level 2 | `dcc024471629d1a7d449e4e041d4bb6621a3c12d088f20b6be174d84d6594900`, `27a0c6c5ca36356b72b48fa2d0a0de2b148fcf95d3082f2addfcf60418facd98`, `98b58ef30f1e2b7d1b0b262ef0fcd8e4cb59db9b921305fbd010d762d1b9039a` |
| Direct level 1 stage profile | `c29470a3e9c65aa47a63e20518c9af3030705ff7ec137576d9066d4c0c89fd13` |
| Diagnostic-route verification profiles | `9a5474c7e4c357bb560bf06abc34f2ac376f4cc0e9e36392e70166fe7180f78f`, `b892d466fd4917819787e4245ffe8e9fbf105de0d8e0b5d108b2a304df55584c`, `1d59290e7c3756be9d7e9acc6781390d1d6308ed062d09f32de5ec2df0af56fb` |
| Local-route verification profile | `5a4d93943766c989a76f2677ee894532cb6216cf39539c437bb215dd05e588f8` |
| Local-route off and on profiles | `4a07df0c38b58ccd4b63966b0697670e9fbb64151ee6f0c86b15453dbfc7d3c8`, `ce0d18855ea96bcefb093c9502eb41a970319b9d35b2c07fabcf87928c6e4e7c` |

The grouped stage profiles of `21-STAGE` predate the direct-route source and
were taken with a separate build. Small samples, one replayed corpus, warm
expert bindings, sequential sessions and unscreened host activity limit causal
and broader-workload conclusions. Paging counters were not sampled. Contexts
above 1,024 input tokens, prefix restoration, MTP speed, installed-build
behavior and production admission remain outside this result.
