# Edge0 larger prefill token batches

Attempt `20261117-EDGE0-23`, 2026-11-17, local non-commercial evaluation only.
Under the token-batched schedule of
[attempt 22](2026-11-17-edge0-token-rows.md), an opt-in selector raises the
ordinary prefill batch from 8 tokens to 16, 24 or 32 so that more rows of a
batch select the same expert. In one binary, observed median prefill rises
from **41.49 tok/s** at 8 tokens to **48.26** at 16 and **54.18** at 32, with
MTP off and matching outputs. Every F32 value the schedules were made to
compare, 5,402,287,632 in total, had identical bits. These sequential
observations do not establish full performance qualification. The selector
defaults to 8 tokens.

## Workload and build

The artifact remains
`Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`, with native
preparation receipt SHA-256
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`. Original
stored formats and F32 activation, accumulation, recurrent-state and cache
policies remain. Payloads were not rehashed.

The workloads, host and private runner are those of attempt 22: the 128-token
input on the historical diagnostic route with a 16-token output limit, and a
1,024-token input on the ordinary local loading route. Input and output
SHA-256 values match attempt 22 in every request:
`de3b19a046b7211189a24690cb7ec98de34aebf79a7e142153d29cee92b6155d` and
`3033ab1a108b72992b627a19aa7fd547c0ae6c72ac2be42c01b77e6022890047` for the
short workload,
`de279b6197f023b5b44c645007f0c5ea383a13369c4b8e07abf97b5e569f90de` and
`8c98070cbe13df012fecf2e47995e32ca6508dbbb2aca37bb9b0c51523c90503` for the
long one. The worker is built with Rust 1.98.1 from commit
`603b957d70b361579f2a023e5dc73fcf34bfafb0` plus the changes of this attempt.
No other `stark-generate` process existed at the start of any final session.
No prompts, generated text or token vectors are published.

## Change

`STARK_EDGE0_PREFILL_TOKENS`, fixed once per process, accepts 16, 24 or 32 and
is honored only with `STARK_EDGE0_DIRECT_ROUTES=tokens` or `verify-tokens`.
Every other setting keeps 8, the limit of the grouped and direct kernels.
Wide-output and MTP batches keep 8 as well.

Token kernels still take at most eight rows per dispatch, so a larger batch
does not decode a trunk matrix less often. It lets more token rows of one
batch select the same expert, D group and U group, and it halves or quarters
the router round trips of a request. Each token's arithmetic and the causal
order of the state recurrence are unchanged. Above eight tokens the float
trunk matrices use the float token kernels, since the original batched kernel
stops at eight. Under `verify-tokens` the grouped schedule remains the
reference and runs eight tokens at a time.

## Final observations

Six sessions of the final binary ran 8, 16, 32, 32, 16 and 8 tokens, each with
one warmup and three measured requests under the `tokens` schedule.

| Median of six measured requests | 8 tokens | 16 tokens | 32 tokens |
| --- | ---: | ---: | ---: |
| Prefill tok/s | 41.49 | 48.26 | 54.18 |
| Prefill tok/s range | 41.09 to 41.72 | 48.00 to 48.41 | 53.89 to 54.38 |
| Decode tok/s | 14.68 | 14.79 | 14.78 |
| End-to-end output tok/s | 3.89 | 4.36 | 4.74 |
| Prefill command GPU ms per token | 18.76 | 16.41 | 14.91 |
| Prefill blocked wait ms per token | 22.35 | 19.31 | 17.27 |
| Prefill command buffers per request | 833 | 441 | 245 |
| Prefill scratch allocated, MB per request | 5,633 | 5,912 | 5,613 |

Against 8 tokens, 16 tokens is 1.16 times the prefill rate and 32 tokens 1.31
times. Decode is unchanged: it encodes one token per batch. The 8-token arm
ran at 39.30 prefill tok/s in attempt 22 and 41.49 here, so sizes are compared
only within these sessions. Scratch figures sum the allocations of every
command of a request; they are not a peak.

| Local route, 1,024-token input | 8 tokens | 16 tokens | 32 tokens |
| --- | ---: | ---: | ---: |
| Prefill tok/s | 39.95 | 47.86 | 53.60 |
| Decode tok/s | 14.22 | 14.40 | 14.53 |
| Request wall seconds | 26.69 | 22.44 | 20.14 |
| Prefill scratch allocated, MB per request | 45,458 | 48,144 | 45,366 |

Each local arm is one session with one warmup and two measured requests.

## Stage attribution

One stage-timed session per size, final binary, level 1 labels. Each cell is
the median of three measured requests.

| Prefill stage, GPU ms per token | 8 tokens | 16 tokens | 32 tokens |
| --- | ---: | ---: | ---: |
| Selected experts | 8.09 | 6.67 | 5.40 |
| GDN and attention state | 2.57 | 2.50 | 2.50 |
| Hyper-connection mixes | 2.16 | 2.10 | 2.09 |
| U projection | 2.06 | 1.90 | 1.83 |
| GDN and attention input projections | 1.64 | 1.50 | 1.51 |
| D projection | 1.06 | 0.90 | 0.82 |
| Router and shared expert | 0.61 | 0.33 | 0.31 |
| Batched mixer output projection | 0.55 | 0.53 | 0.52 |
| Other labelled stages | 0.42 | 0.40 | 0.40 |
| Gaps between encoders | 0.37 | 1.20 | 0.84 |
| Command GPU total | 19.49 | 18.02 | 16.23 |
| Dispatches per token | 1,629 | 1,467 | 1,337 |
| Expert dispatches per token | 549 | 441 | 339 |
| Rows per selected expert | 1.75 | 2.18 | 2.83 |

Sharing grows slowly because routing is diverse: four times the batch gives
each selected expert 1.6 times the rows. Selected experts remain the largest
prefill stage, a third of GPU time at 32 tokens. The router stage falls above
eight tokens because its float matrix moves to the token kernels.

## Correctness and limits

| Verification request, final binary | Batch | Input / output tokens | Comparisons | F32 values |
| --- | ---: | ---: | ---: | ---: |
| Diagnostic route, corpus offset 0 | 16 | 128 / 20 | 40,944 | 479,177,424 |
| Diagnostic route, corpus offset 0 | 32 | 128 / 20 | 40,944 | 479,177,424 |
| Diagnostic route, corpus offset 128 | 32 | 128 / 3 | 39,312 | 443,665,104 |
| Diagnostic route, corpus offset 256 | 32 | 128 / 11 | 40,080 | 460,376,784 |
| Local route, corpus offset 0 | 32 | 1,024 / 16 | 314,640 | 3,539,890,896 |
| Total | | | 475,920 | 5,402,287,632 |

All five requests succeed. Counts equal attempt 22's because the reference
still runs eight tokens at a time. Model-free native tests add bitwise
equality of batches of 9, 16, 17 and 32 tokens against the original batched
kernel applied eight tokens at a time, for Q8_0, Q3_K, IQ4_NL, F32, F16 and
BF16 matrices, and of the ordered route reduction for 16 tokens against two
batches of 8. They reject a batch above 32 tokens. Test counts are in the
[attempt record](../model-experiments/2026-11-17-edge0-prefill-batches.md).

| Evidence binding | SHA-256 |
| --- | --- |
| Final Rust 1.98.1 worker, all settings | `1e10aa8920c8db22cf1d3cfd03a7c8089e466c7f05e7bd975725aaa7535301ce` |
| First-session worker | `78dbe1b487d207d79996fc326a67635d84bf8473b0da3796ce8637b9ef8c69b0` |
| `edge0_tokens.metal`, unchanged | `257c1d2d7de461d9ece18da3636d489bd588119c56b6c7bbe28f46ef2113801d` |
| Private runner | `8bd484687a6906428cff6a8b3bfc94b8cb956da4bd1743463246ee0e730e1112` |
| Final summary | `78c6ec1909ec2c4b8b4278db81fe5f839d14da2253bbc0259f0bb2d55a6829aa` |
| 8-token profiles | `80b1aeff0374bda9fdc5bb009cbc92b7bd2eefb987a5838d7b6999eb1f4ec362`, `638f6541c903de3deeb485a7e663a043206685a49388d2cecdaacc56909f9f8f` |
| 16-token profiles | `4080465d2ec546e16c615c57213dc13fb64400be6c78be6d5a486643591c7752`, `4b4a988b587c7ef3b52069eef0c91015a1db7f586229f952eb92696b7e01b2fd` |
| 32-token profiles | `6e392a64589302672113e07cf748b94c723c4265f214bb42f59d7b04832390e6`, `49330530b21df6aa63eeba75112ea93281395f1d1b4748ad4412e0d97e520184` |
| Local-route profiles: 8, 16, 32 tokens | `5d6b61444451e2aa5b931f6e5153da90a12a9100cf26efe5a0ab68a0033fcb58`, `4b7e30778770bf33dcccad6d84c8c72f17173d9f9b804dd9a7a2c75ce058c785`, `a9517fd9dac20da1885a724a971330a4524bf2f6039bbda1d3115913e2884ae4` |
| Stage profiles: 8, 16, 32 tokens | `7a26880d9c2deee05d698a9a1e4c9d086ca1190c072602bb0512d081e76bcd17`, `b49e5720828e681b151fef27bc6b30de4a12f98ace6c584d5275d21fc42916dd`, `ce838a8f77300e4befe3f88a147ece2da30e91d276e3951c72d581c14f50302a` |
| Verification profiles, in table order | `4a84d9eb4e439bd14104aeffc34a3bcb56f3128398df56f567bf3dca2c183742`, `b0f1b3e02908ebaa6ad8a5220d8eb8b5a0ea1e7af604f6ff39385da8665e8195`, `fd4ef9a5f0bbe71ca45e05d145541f1d80f530a21ebeeae076f876f7535b56c3`, `9e4c4b062ed5df0515079fda033acad5a4628db0aa1511e17ac11263fa9e8cdd`, `c060558cc69cad0baf039f5a38acb91ba31ff393b8f722d0e1f8c560b6df0147` |

A first set of sessions on the pre-final binary, with a browser and other
desktop applications active, measured 37.6, 44.0 and 46.8 prefill tok/s for
8, 16 and 32 tokens. The 24-token setting is accepted but was not run.

Small samples, one replayed corpus, warm expert bindings, sequential sessions
and unscreened host activity limit causal and broader-workload conclusions.
Paging counters and peak memory were not sampled. A cancellation is observed
between batches, so a larger batch delays it by up to one batch. Contexts
above 1,024 input tokens, lifecycle cases, prefix restoration, MTP-enabled
requests, installed-build behavior and production admission remain outside
this result.
