# 20261126-HY4OC-01 · Hy4-flash defaults through OpenCode

## Identity and change

The installed desktop default could reproduce about 172 prefill and 14 decode
tok/s in a 4K greedy worker diagnostic under background CPU load on this Mac.
The published 712 prefill
and 58 decode tok/s came from separate short-window worker configurations.
OpenCode had used roughly 16K–27K positions with a tool schema and conversation
history. This run compares the former default and the automatic tuned load
through the actual OpenCode client at those lengths.

| Field | Value |
| --- | --- |
| Artifact | AngelSlim/Hy4-flash-GGUF `c50d4dd30be42f2b91138e5731de0e2f32ad56b2`, original Q3_K split GGUF, BF16 projector and NextN. The [native record](2026-11-25-hy4-native-q3-k.md#artifact-and-runtime-identities) binds the source digests. |
| Host | Apple M5 Max, 128 GiB unified memory, macOS 26.7.2 |
| Client | OpenCode 2.4.2 standalone CLI with the packaged Stark OpenCode connector, isolated local session and client credential |
| Desktop backend | Signed development bundle documented in the [tool-calling record](2026-11-26-hy4-tool-calling.md), manifest SHA-256 `9d84882cb5a07adfa148cb6661c32dec5fe6adfd2a9568de2e641fef38e1c46b`, used in both arms |
| Baseline worker | Bundled release worker SHA-256 `d95eea727dea941f9560e9e113aa6a29d33ef096a070b874db19b99126d1d27b` |
| Candidate worker | Rust 1.98.1, locked release build with `hy4`, base `508fa7fec99c53cda12c830da822a76846914b67`, Rust diff SHA-256 `305e07799bd988d6a8d97d535fddb82a96d2bbaef9460b688c906e7cdc8dfbf8`, executable SHA-256 `c0c89f4c37c5c47ecb56c205dfdecab7d28e18599aed675e283aa04a3f60b588` |

The implicit Metal 4 selection reuses the qualified streamed mechanisms.
It requests a 40 GB streamed expert arena, 4,096-row prefill, grouped F16
fragments with F32 accumulation, packed attention, tensor MLA, resident
embeddings and a Q5_0 draft head. The automatic policy retries with the
ordinary arena on memory admission failure and can fall back to the legacy
profile. The explicit resident-decode profile remains available with its
separate startup and host-memory tradeoffs. Load capacity still comes from
the available memory and workspace fit. FFN device views keep their existing
8,192-position loaded-capacity guard.

## Workload and controls

OpenCode sent streaming code-review turns with its built-in tool definitions.
The synthetic source was generated locally. The client submitted a fresh
prompt near 16K, appended more code to reach about 27K in the same session,
then used a new session for a fresh 27K prompt. Both workers loaded the image
projector, with no image attached to timing turns. The same 96-token output
ceiling applied through connector discovery. The helper normalized the terminal
SSE length reason to stop after generation so OpenCode would issue exactly one
request per turn. It reset the worker prefix before the fresh 27K turn. The
loaded context was uncapped in both arms. The real desktop worker owned prefix
scope and state. The helper observed dispatch options and aggregate timing, and
hashed rendered inputs and outputs in memory.

Prefill tok/s counts new input tokens after prefix restoration and divides by
worker time to first sample. Decode tok/s counts sampled tokens after the first
sample and divides by worker decode time. OpenCode wall time includes template
rendering, local transport and client processing. Load startup is recorded
separately.

## Attempt history

- The first build attempt used an APFS clone of earlier build outputs. The
  Metal module cache contained a precompiled header tied to the old checkout
  path. Rebuilding that cache at the new path passed. No timed request came
  from the failed build.
- The mock transport run checked OpenCode discovery, tool schema, greedy wire
  sampling and session continuity. It did not generate model performance data.
- In the first baseline run OpenCode treated the 96-token length stop as a
  signal to continue. The first native call processed 16,000 new inputs at
  165.54 tok/s and decoded 96 sampled tokens at 5.05 tok/s with two drafts.
  OpenCode then submitted two additional cached completions and ended with
  an error. The terminal normalization above excluded that behavior from
  the paired timing matrix.
- The first image smoke driver depended on Pillow, which the packaged Python
  does not include. The retry made the synthetic PNG with the standard library
  and reached the native worker.

## Performance and memory

The baseline loaded 452,608 positions with a 27,996,717,056-byte streamed
arena and the candidate loaded 221,184 with a 39,996,981,248-byte arena.
Both started with no context override. Desktop startup took 16.805 s for
the baseline and 16.938 s for the candidate. Warm turns resumed the same
OpenCode session. Fresh turns processed all input positions.

| Turn | Input tokens | Reused, baseline → candidate | Prefill tok/s, baseline → candidate | Decode tok/s, baseline → candidate | First sample seconds, baseline → candidate | OpenCode wall seconds, baseline → candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh 16K | 16,000 | 0 → 0 | 156.73 → 444.62 | 7.50 → 18.79 | 102.08 → 35.99 | 115.52 → 41.79 |
| Continued 27K | 26,950 | 15,360 → 12,288 | 160.81 → 476.24 | 13.55 → 20.79 | 72.07 → 30.79 | 79.86 → 36.00 |
| Fresh 27K | 27,000 | 0 → 0 | 203.18 → 482.51 | 10.44 → 19.12 | 132.89 → 55.96 | 142.65 → 61.57 |

The wider prefix grid ran 3,072 extra rows on the continuation. The faster
prefill still cut first-sample latency by 57%. Both profiles generated 96
tokens with greedy NextN at depth two. Readiness confirmed the expected
profile, prefill geometry, grouped operands and draft-head format in each
arm. System memory pressure stayed at level 1 at the sampling points and
swap stayed at 33.69 MiB.

## Correctness and evidence scope

The fresh turns had identical rendered inputs across profiles. All six
completions returned 96 tokens and each OpenCode invocation completed
successfully. Both arms used tools in the request schema. The synthetic
code-review turns answered with text. The generated token hashes differed
between profiles at both fresh lengths. The resumed comparison used the same
user turns and rendered token count, with a different first assistant reply.
The timing shows performance on matched fresh inputs and a matched continued
workload. The generated-output hashes establish the limits of the comparison.
The earlier [guarded-profile study](2026-11-25-hy4-grouped-fragments.md#20261126-hy4frag-33--bound-ffn-device-views-to-the-qualified-capacity)
supplies the pinned fixture parity evidence.

A separate lifecycle check used the same uncapped candidate executable and
projector. The worker planned request windows of 8,192 and 32,768 on the same
load. A 4,429-token greedy text repeat restored 4,096 tokens and returned the
same eight output IDs as the fresh request. A synthetic 128 × 128 image
completed with zero drafts and zero prefix reuse. The next text request
processed its full prompt and matched the earlier output IDs. The focused
profile and admission-error tests passed. The release generation library
suite with `hy4` passed 181 tests, with six ignored.

There is one observation per arm and turn. The baseline ran first and the
candidate followed on the same machine under the GPU lock. The synthetic
source, decoding policy and controlled output length limit what these rates
predict about other agent workloads. Adopt the automatic tuned streamed load
on this branch. Keep the prepared resident profile explicit. Long-context
quality beyond the pinned fixtures, performance after hours of use and live
fallback on low-memory hosts remain open.

| Private aggregate or harness | SHA-256 |
| --- | --- |
| Timing harness | `563d444edd1fa319e6f7f69fff4441f03301533463747a07bf4fab8d67aca296` |
| First baseline with client retries | `981ce977d973f144cfc63e64651ca2ffb81b48bc7176748027246b49d60fc776` |
| Baseline timing | `b780e4b23a28b5fd01d1ad63f16cc37a45903e3f4a4d2f347af4bb0c2fb2be99` |
| Candidate timing | `f32c236495bb5bd77c7a2fbf5538b3dc376d8a7f5d9f894a41a83b6c79735c8d` |
| Host pressure and swap samples | `25a52772bb7e56d115f91831abbebdc7baa06ea915cfdbb309971fd130401951` |
| Lifecycle harness | `e2ef1d71431a5e0ab9079c7051fa41c9046a4399028ce9ea797d6d8fc14fd4cc` |
| Candidate lifecycle | `4132bfde12b2879a91b204abcbe4c3fba41124f2e489b691ec9387458ff080aa` |
