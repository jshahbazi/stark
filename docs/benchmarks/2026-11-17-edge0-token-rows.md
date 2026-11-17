# Edge0 token-batched prefill rows

Attempt `20261117-EDGE0-22`, 2026-11-17, local non-commercial evaluation only.
An opt-in schedule lets the token rows of one matrix share a dispatch that
decodes each weight once. In one binary, observed median prefill rises from
**19.43 tok/s** on the direct schedule of attempt 21 (14.70 on the grouped
schedule) to **39.30 tok/s**, and decode from 13.97 to 14.90 tok/s, with MTP
off and matching outputs. Every F32 value the schedules were made to compare,
4,923,110,208 in total, had identical bits. These sequential observations do
not establish full performance qualification. The selector defaults off on
both loading routes.

## Workload and build

The artifact remains
`Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`, with native
preparation receipt SHA-256
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`. Original
stored formats and F32 activation, accumulation, recurrent-state and cache
policies remain. Payloads were not rehashed.

The short workload repeats [attempt 19](2026-11-13-edge0-singleton-speed.md)
and [attempt 21](2026-11-17-edge0-direct-routes.md): the historical diagnostic
route, context 512, a 128-token input, greedy seed 42, explicit MTP zero and a
16-token output limit. Prefill rate is 128 divided by native prefill seconds;
decode rate is 15 divided by native decode seconds, excluding the initial
token. Input SHA-256
`de3b19a046b7211189a24690cb7ec98de34aebf79a7e142153d29cee92b6155d` and output
SHA-256 `3033ab1a108b72992b627a19aa7fd547c0ae6c72ac2be42c01b77e6022890047`
match the attempt 19 reports in every such request of this record.

The host is an Apple M5 Max with 128 GiB, macOS 26.7.2 (25G332). The worker is
built with Rust 1.98.1 from base `9280d3182b4e288b00b291ff58b648e50a0d9201`
plus the changes of attempts 21 and 22. A self-contained private runner starts
one worker per session and sends one warmup before the measured requests. It
adds no quiet admission, host screen or paging sampler, and refuses to start
while another `stark-generate` process exists. Aggregate native profiling is
enabled in every session. No prompts, generated text or token vectors are
published.

## Kernel probe

`22-PROBE` times kernels without the model. Each shape uses several distinct
synthetic matrices in one serial command of 512 token rows; each cell is the
median of seven commands, in microseconds of command GPU time per token row.
The stop rule, fixed beforehand, required at least 1.5 times the one-token
rate on shared matrices.

| Matrix shape | One-token kernel | Original batched kernel, 8 tokens | Token-batched, 2 tokens | 4 tokens | 8 tokens | Gain at 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U, Q8_0, 1792 to 2560 | 15.55 | 23.35 | 9.49 | 5.62 | 3.96 | 3.93 |
| D with stored scale, Q8_0, 2560 to 1792 | 15.30 | 22.77 | 9.44 | 5.59 | 3.91 | 3.91 |
| GDN input projection, Q8_0, 2560 to 10240 | 79.22 | 126.44 | 48.78 | 27.50 | 18.98 | 4.17 |
| Shared expert, Q8_0, 2560 to 640 | 6.39 | 8.85 | 3.76 | 2.19 | 1.54 | 4.14 |
| Fused expert gate/up, Q3_K, 1792 to 640 | 17.56 | 9.33 | 9.48 | 4.99 | 3.01 | 5.84 |
| Expert down, IQ4_NL, 640 to 1792 | 7.33 | 7.06 | 4.22 | 2.54 | 1.81 | 4.06 |

The original batched kernel selects its format and token count at run time.
On Q8_0 matrices it takes longer per token row than one-token dispatches, as
attempt 21 observed in the model. The token-batched kernels fix both at
compile time and keep each token's reduction. The probe is retained as an
ignored test.

## Token-batched schedule

`STARK_EDGE0_DIRECT_ROUTES`, fixed once per process, gains three values:

- `token-routes`: the direct schedule of attempt 21, with shared dispatches
  for the trunk projections of a token batch, the D rows of a projection
  group, one expert's fused gate/up rows and down rows, and up to eight U rows
  of a group. Rows are addressed by offsets in their own buffers.
- `tokens`: also runs the three projections of each hyper-connection mix and
  the GDN or attention output projection once per token batch. These replace
  the generic one-token projection. The GDN result row is persistent state, so
  each token's row is copied before the next token overwrites it.
- `verify-tokens`: encodes `tokens` beside the grouped schedule and beside the
  one-token mixes and output projections, compares on the device and rejects
  the request on any differing bit.

Route slots, per-route U, the 256-column stride, the SIMD-group sums, the
ordered eight-group F32 sum and the causal order of the state recurrence are
unchanged. `edge0_graph.metal` and `edge0_direct.metal` are byte-identical.
Token kernels record sticky failure for every value they write, and a
nonfinite input makes every result of its row nonfinite, so the batched mixes
and output projections carry no separate scans. With one token per batch only
U rows of the same group share a dispatch.

## Final observations

Eight sessions of the final binary ran grouped, direct, `token-routes`,
`tokens`, `tokens`, `token-routes`, direct, grouped, each with one warmup and
three measured requests. No other worker existed at the start of any session.

| Median of six measured requests | Grouped | Direct | `token-routes` | `tokens` |
| --- | ---: | ---: | ---: | ---: |
| Prefill tok/s | 14.70 | 19.43 | 30.11 | 39.30 |
| Prefill tok/s range | 14.03 to 15.28 | 18.48 to 20.56 | 29.63 to 32.02 | 36.13 to 41.91 |
| Decode tok/s | 11.98 | 13.97 | 14.84 | 14.90 |
| Decode tok/s range | 11.94 to 12.13 | 12.81 to 14.06 | 14.51 to 14.96 | 13.75 to 15.11 |
| End-to-end output tok/s | 1.61 | 2.08 | 3.04 | 3.76 |
| Prefill command GPU ms per token | 60.69 | 45.03 | 27.62 | 20.47 |
| Decode command GPU ms per token | 63.08 | 52.71 | 49.77 | 50.62 |
| Prefill blocked wait ms per token | 65.53 | 49.29 | 31.30 | 23.78 |
| Decode blocked wait ms per token | 77.40 | 66.34 | 62.65 | 62.08 |
| Prefill scratch allocated, MB per request | 7,268 | 4,745 | 4,745 | 5,633 |

Against the direct schedule, `tokens` is 2.02 times the prefill rate and 1.07
times the decode rate; against the grouped schedule, 2.67 and 1.24 times. The
direct arm itself ran at 20.56 prefill tok/s in attempt 21 and 19.43 here, so
arms are compared only within these sessions. Categories are inclusive and
must not be summed. All arms still wait on 49 command buffers per decoded
token.

The ordinary local loading route (`STARK_EDGE0_LOCAL_EVALUATION=1`, context
8,192, MTP unavailable) ran a 1,024-token input with a 16-token output limit.
Each arm is one session with one warmup and two measured requests; every
request has input SHA-256
`de279b6197f023b5b44c645007f0c5ea383a13369c4b8e07abf97b5e569f90de` and output
SHA-256 `8c98070cbe13df012fecf2e47995e32ca6508dbbb2aca37bb9b0c51523c90503`,
as in attempt 21.

| Local route, 1,024-token input | Direct | `token-routes` | `tokens` |
| --- | ---: | ---: | ---: |
| Prefill tok/s | 19.00 | 30.03 | 38.55 |
| Decode tok/s | 13.51 | 14.25 | 14.27 |
| Request wall seconds | 55.01 | 35.16 | 27.62 |
| Prefill scratch allocated, MB per request | 38,166 | 38,166 | 45,458 |

Scratch figures sum the allocations of every command of a request; they are
not a peak.

## Stage attribution

One stage-timed session per schedule, final binary, level 1 labels of attempt
21. Each cell is the median of three measured requests. Stage timing adds a
few percent of GPU time.

| Prefill stage, GPU ms per token | Direct | `token-routes` | `tokens` |
| --- | ---: | ---: | ---: |
| Selected experts | 12.89 | 8.10 | 8.09 |
| U projection | 6.74 | 2.06 | 2.06 |
| Hyper-connection mixes | 6.74 | 6.76 | 2.16 |
| GDN and attention input projections | 6.13 | 1.64 | 1.64 |
| GDN and attention state, with output projection | 5.81 | 5.80 | 2.56 |
| Batched mixer output projection | | | 0.55 |
| D projection | 2.97 | 1.06 | 1.06 |
| Router and shared expert | 1.22 | 0.61 | 0.61 |
| Other labelled stages | 0.45 | 0.45 | 0.42 |
| Gaps between encoders | 0.96 | 1.13 | 0.37 |
| Command GPU total | 43.89 | 27.61 | 19.50 |
| Dispatches per token | 3,640 | 2,444 | 1,629 |

Under `tokens` the state row holds the per-token recurrence and row copy only.
Decode changes in one stage: U falls from 7.59 to 4.28 ms per token, and the
command GPU total from 53.45 to 50.19 ms.

Selected experts are now 41 percent of prefill GPU time. Expert rows share
least: 549 expert dispatches per token against 960 means a batch of eight
tokens gives each selected expert about 1.75 rows.

## Correctness and limits

| Verification request, final binary | Input / output tokens | Comparisons | F32 values |
| --- | ---: | ---: | ---: |
| Diagnostic route, corpus offset 0 | 128 / 20 | 40,944 | 479,177,424 |
| Diagnostic route, corpus offset 128 | 128 / 3 | 39,312 | 443,665,104 |
| Diagnostic route, corpus offset 256 | 128 / 11 | 40,080 | 460,376,784 |
| Local route, corpus offset 0 | 1,024 / 16 | 314,640 | 3,539,890,896 |
| Total | | 434,976 | 4,923,110,208 |

All four requests succeed. The diagnostic-route requests allow 48 output
tokens and stop at end of sequence. Route and trunk rows compare against the
grouped schedule as in attempt 21. Both outputs of every token-batched mix
compare against the one-token mix, and every token-batched mixer output row
against the generic one-token projection. Verification covers schedules of the
same binary; it does not re-qualify any of them against the frozen reference.
The intermediate `22-ROUTES` binary passed the same four requests with 98,016
comparisons and 4,405,000,512 values.

Model-free native tests add bitwise equality of token rows against the
one-token kernels for Q8_0, Q3_K and IQ4_NL at columns 512 and 1792 and one to
eleven scattered, permuted rows, with and without the fused D quotient;
against the original batched kernel; against fused gate/up rows; and against
the generic one-token projection for F32, F16, BF16 and Q8_0 matrices,
including a 320-column matrix. They reject nonfinite inputs without a separate
scan, invalid stored scales, row indices beyond their parent, shared output
rows, aliased and ragged parents and truncated spans. Test counts are in the
[attempt record](../model-experiments/2026-11-17-edge0-token-rows.md).

| Evidence binding | SHA-256 |
| --- | --- |
| Final Rust 1.98.1 worker, all settings | `113528545982192fab23d80969d33a40041b5a1f44f59933c4774618ca81a58f` |
| `22-ROUTES` worker | `b7bb0ba3931adf8a9d6f10f7a9ae353b489c6cbe8614a83e704978a4b24d2325` |
| First-session workers, routes then stages | `6041474820d3046134db206dfe94b134ef4622e9418d926f72b27318db5271f4`, `7ca7081ab17e503fc0e6a91416f83fd39fd49022a1b74f3d5ea28b473ac1a878` |
| `edge0_tokens.metal` | `257c1d2d7de461d9ece18da3636d489bd588119c56b6c7bbe28f46ef2113801d` |
| `edge0_graph.metal`, unchanged | `b7b8c86e1abe7c3bf1208415ca490b0a9676926e6a52408fc7d5235d8aa97678` |
| `edge0_direct.metal`, unchanged | `656411e2170487f2486116ecb996445be17a53582d00a2fb068e82a9f44079f8` |
| Private runner, final sessions | `e9cbf4914b60e16a187aeff9fefee6d5b997050e5b98b614a86ff87ffe9471d9` |
| Final summary | `7a14131aa988740b36450a0c694b49da7c5e3b126b6dccc61898dbabc11d9778` |
| Probe log | `717da6e0e23f1b03b56937a695f81ff9e85e4f4773564709075754d21d140264` |
| Grouped profiles | `3e84742011d78a59ec1939962822c01262bea3e094cdddba3a3f56584a80bb4e`, `c4943a802fafa57125bf179c94466d25df030cc77d1baa896350521933d292be` |
| Direct profiles | `7f30ca7c31fd218c803b1947928c0398403e7d2eedc1e9f68668bba4a95f7080`, `9db0449233c0c97672c2d0a8ab95f26302d64f4e5acde8930e995c5028595753` |
| `token-routes` profiles | `7c623d93e45d8a0f484d13c7b81fa221d75785513f710f96c1fabf001e1b37c8`, `28c43b0216c1d704e5346981419b37e9de6c770a46da48eca91ad29bf4498857` |
| `tokens` profiles | `87374909b3f7c761b4ee7ce95f26b09e275b0b8decbdbd569b1852471edf85f0`, `5b1d930c809c40d7647e902aab636ad835b9d0fb3f4cd0b32d4f66bf81f63b0c` |
| Local-route profiles: direct, `token-routes`, `tokens` | `1dd6e457e16750910c158c60304901cc767533b59367c25bb287b84b8507dd0a`, `053d3830731c6517dc60ec2c58d94a4eef4a343c4fea16cdfcdc22fae3e98279`, `b556398f5dcfd65e0b3c2f6ae41aad9ab434384102ad8e25c1263d230f5161a6` |
| Stage profiles: direct, `token-routes`, `tokens` | `5712139b97bfca4a9570bf1475ffae46f49cc7b37445ca885074c0fde687c7dd`, `0f0a47f4e6a7f517e8d730d981f92f9f743883468d9083659db31d743cbc1489`, `d364160f39943f5230113a77a56cda37741fa165bb8a5f80a665e6cbdfe3764b` |
| Diagnostic-route verification profiles | `7efa9d7182f4d07cccc01ee66d9b252580b4033d9dbf2edb7d4d5f474177ce2c`, `14a9fed4aca27894b79bf2f6f53c24e7a9aff436ee5fca1be8e3ce472066571d`, `4429edec117fecf437679b044a62324d2f5b716101989429c93501c2b191b777` |
| Local-route verification profile | `0fa9383b8b391d8b2c5ace2670491f1987e148decba8515e70158a3dbfc7e447` |
| `22-ROUTES` summary | `245d86642c69d20cfe6489a004375a493d6f66b3ddea95d792dae2d6e1e7a9cb` |

The `22-ROUTES` sessions measured 14.54, 18.96 and 29.93 prefill tok/s for the
grouped, direct and token-batched arms of their binary, where the value now
named `token-routes` was called `tokens`. Their local-route sessions did not
run: the runner refused them because the owner's desktop application had
started a worker.

Small samples, one replayed corpus, warm expert bindings, sequential sessions
and unscreened host activity limit causal and broader-workload conclusions.
Paging counters were not sampled. Contexts above 1,024 input tokens, lifecycle
cases, prefix restoration, MTP-enabled requests, installed-build behavior and
production admission remain outside this result.
