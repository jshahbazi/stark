# HY4STQ-PERF-03 · Routed experts and long-context attention

This follow-up starts at `1faac530d` with the
[wide dense attention path](2026-11-27-hy4-stq-dense-attention-prefill.md)
enabled. The earlier fresh 27K result was 34.18 s. It had 2.91 s left
to the earlier same-cap Sushi observation and 6.60 s to the faster
repeat. Both Sushi measurements used that runtime's template and KV8.

The artifact is `beamster/Hy4-Flash-Sushi-STQ-4bpw` at
`61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`. The M5 Max 128 GiB
ran Rust 1.98.1 Metal release workers with GPU work serialized.
The OpenCode 2.4.2 sequence is the same 16,229-token fresh request,
27,179-token continuation and 27,229-token fresh request as before.
The model has vision loaded, F16 KV, resident weights, 4,096-row
chunks and a flexible context capped at 49,152. The requests use
12 tools, 96 greedy output tokens and five DFly2 drafts. Prefill rate
counts computed input rows. Useful decode counts 95 output transitions.
These are uninstrumented local OpenCode measurements through the
Stark API. [Pinned pack and earlier controls](2026-11-27-hy4-stq-open-code-performance.md)

## Starting time budget

The instrumented control had wide dense attention enabled. It measured
0.45 s in routing, 10.71 s in routed products, 6.01 s in MLA and
10.97 s in attention projections at fresh 27K. The preceding
uninstrumented request spent 33.85 of 34.18 s inside GPU command spans.
The effort targeted GPU work after routing and in MLA. Profile and
throughput measurements use separate processes.

## Component experiments

| Attempt | Result |
| --- | --- |
| HY4STQ-PERF-03a · PAIR1 K32 routed products | Two adjacent K16 operand slabs share one K32 Metal product with F32 accumulation, the existing F16 product boundary and masked vector writes. Layer-one gate, up and down exercises cover all 512 experts, live jobs of every size from 1 through 32, and 16,128 and 65,280 routed rows. Across those shapes each of three variants matched 375,128,064 active F16 values and preserved padding. The retained K32 variant lowered median projection GPU time 17–21% at the smaller size and 26–28% at the larger size in nine alternating pairs. The explicit K32 funnel and K16 vector store arms tied or lost to it at the larger size. |
| HY4STQ-PERF-03b · Register H128 on routed preparation | Each SIMD group transforms four values per lane. The two cross-group butterflies become register operations with the same sum and difference order. The prepare and prepare-down kernels reuse the existing scratch and F16 casts. Combined output matched all 11,939,328 F32 values across 1,024, 2,653 and 4,096 input rows. Six alternating pairs put the full-chain medians at 6.749 → 6.256 ms, 18.295 → 17.513 ms and 25.905 → 24.272 ms, respectively, with the incumbent K16 products. At 4,096 rows the initial prepare went from 0.984 to 0.605 ms and the down prepare went from 1.720 to 1.107 ms. |
| HY4STQ-PERF-03c · Routed K32 follow-up | A fixed 1,536-wide operand shape, an unclamped full-job load and SIMD reuse of a previous packed word each matched 375,128,064 F16 outputs and all the guard rows. Across six shapes and nine pairs each, the variants ranged from 1.1% faster to 3.1% slower than the retained K32 kernel. Kept the current addressing and tail handling. |
| HY4STQ-PERF-03d · Selected MLA scratch and zero-query fragment | At 23K history, 1,024 query rows and 32 sparse heads, the selector used 511 distinct groups of four keys plus the causal tail. Each arm matched all 16,777,216 F32 outputs. Skipping an all-zero third query fragment with a two-cell score stride pad lowered median GPU time from 6.819 to 6.217 ms over nine alternating pairs. The skip can alter nonfinite propagation. With a finite-key guard it moved 6.828 to 6.534 ms, with 4,210,688 matched output words on a separate 257-row Inf/NaN stress case. The guarded component gain was too small to advance. |
| HY4STQ-PERF-03e · Selected and sliding MLA operand geometry | Score swizzling, unrolled reductions and a K32 join of the two selected probability products finished within 0.5% of the incumbent on the final component geometry. The three sliding tensor extent variants with 1,025 queries, 64 heads and a 512-key window also fell within 0.5%. Joining two key tiles on the M axis of the selected QK product was 15% slower. K64 and K96 QK arms failed Metal compilation because the cooperative input tensors allow K16 and K32. Kept the existing selected and contiguous paths. |

The routed candidates share the existing PAIR1 Metal codecs and the
register butterfly used by the incumbent finish-reduce. They keep route
order, scaling, scratch bounds, precision and router-slot accumulation.
The validation worker exposes both via `STARK_HY4_STQ_WIDE_ROUTED=1`
for prompt chunks of at least 1,024 rows. The narrow and verifier
dispatches stay on their existing pipelines.

## First OpenCode pair and lifecycle

The harness initially omitted environment forwarding for wide routing.
The ready record showed the switch off, making this run a control.
The fresh 27K turn spent 3.77 s in Engram preparation and reads and
returned 41.68 s at prefill and 34.79 tok/s at decode. A broad
filesystem search was active. The harness now checks the selected
path in ready metadata during worker startup.

The first enabled run reproduced the three input and output hashes
and draft counts from the control. The first 16K turn took 47.90 s
with 15.78 s of GPU spans. System swap use grew from 424.75 to
974.44 MiB before that turn completed. The control had reported
17.53 s of 16K GPU spans and 19.52 s of prefill. The first 27K
continuation and fresh turns with the switch on took 18.46 and
32.48 s and decoded at 55.65 and 47.00 tok/s. The fresh turn had
32.18 s of GPU spans and 11 ms in Engram. The outlying 16K and
27K timing samples require a separate clean throughput comparison.

Both prefill arms passed the actual image and flexible-context
checks. The 27K continuation reused 12,288 tokens and agreed with
the fresh replay. The 4,426-token probe used windows of 8,192,
8,192 and 12,288 in order. The generated IDs matched and prefix
reuse was 0, 4,096 and 0. The image request disabled drafting and
cleared the text prefix. The next text turn reproduced the
original output from zero reuse.

## Repeated OpenCode comparison

Both arms used the same binary, with dense attention enabled throughout
and only wide routing switched. The controls left system pressure normal
and swap stable at 966.44 MiB. Input and output hashes, prefix counts
and draft counts matched both earlier arms.

| Fixture | Routed off, prefill / decode tok/s | Routed on, prefill / decode tok/s | Prefill seconds off → on |
| --- | ---: | ---: | ---: |
| Fresh 16K | 813.90 / 48.27 | 831.45 / 48.55 | 19.94 → 19.52 |
| Continued 27K | 760.77 / 52.03 | 793.95 / 46.09 | 19.57 → 18.76 |
| Fresh 27K | 796.07 / 48.43 | 826.77 / 41.02 | 34.20 → 32.93 |

The repeated fresh 27K pair saved 1.27 s of prefill. The first enabled
fresh run at 32.48 s was 1.73 s below the repeated control. Repeated
fresh 27K GPU spans fell from 33.85 to 32.65 s. The enabled runs
measured 41.02–47.00 decode tok/s at fresh 27K and 46.09–55.65 on the
continuation. Both draft and verify timings moved on the slower run.
The new shader selection applies only to prompt chunks of at least
1,024 rows. The output hashes and acceptance counts remained exact.

The top-level requested time gap remains open. The 32.93 s prefill
is about 1.7 s behind the earlier separate Sushi observation and 5.36 s
behind its faster repeat. The input token counts and template differ
between runtimes.

## Long-context decode replay

To check the spread in decode, the same binary ran the fresh 27K
OpenCode turn alone, first with wide routing on and then in a second
process with it off. Each arm then replayed the exact rendered
request five times with the same scope and 24,576 tokens reused.
Every run matched the input and output hashes and 70 accepted
drafts in 25 rounds. System pressure stayed normal with no swap
growth.

| Routed mode | First-use prefill | First-use decode | Warm decode median | Warm decode range |
| --- | ---: | ---: | ---: | ---: |
| Off | 37.00 s | 44.70 tok/s | 48.47 tok/s | 45.77–49.07 tok/s |
| On | 35.04 s | 44.28 tok/s | 45.55 tok/s | 43.37–47.19 tok/s |

The five warm observations are from one process per arm. The
observed decode median was 6.0% lower with wide routing. The
first-use decode difference in this pair was 0.9%. Prefill
selection leaves the decode kernels in place, and the measured
results do not establish decode-rate parity. Keep wide routing
opt-in and reverse the warm replay order in further qualification.
The flexible-context and output checks passed independently of
the speed result.

Repository policy, release Clippy, Rust formatting and Ruff passed.
The locked Rust 1.98.1 generation tests passed 183 checks with six
ignored. The focused converter, tool and API suite passed 189 tests
after selecting this checkout's package on the shared test environment.

## Evidence identities

| Receipt | SHA-256 |
| --- | --- |
| Wide dense attention starting profile worker | `118c448a83c7010e415e56ca0c6865082f8dda1df90993ac5c1258f9c56318be` |
| Wide dense attention starting profile | `0fa938e365aadd4e8644df3b91a227363568c06708d4036689af196b90f68e66` |
| PAIR1 projection screen | `bcbcf868249fb7fdf5ab163457a9f021f2b5397fc73d3cda93dd43617e60d9c4` |
| Register H128 full-chain screen | `5281435ffc3750b7ed0d90882fbb1888b197f70f694f02f2d61086f9b6d74437` |
| PAIR1 K32 follow-up | `2c10848a23df24947a469ffa658bafe793100b6e1ebf39eb4116de43621d6cbd` |
| MLA component harness | `ed5bc6eb219ad99d4acfbf3a029c6dc79f42174619cdbedc1356944eb8d675d0` |
| MLA operating geometry and numerical checks | `9845439d90a51e737bc55dccb840d813332c80851e72cb5de5ae5a4c3d067286` |
| MLA two-key QK geometry screen | `fea2ed30d272c34dcbfa3f6ecd7c00a7147bd0a86b7c0c302b35001cd8ce1629` |
| First routed worker | `2750d07ead7c9e66cac5401df053e828a59e2980ab8140f8a486945de833395d` |
| First control and lifecycle | `0f08410589704a65a159002dce1275861cc52a68735eaafaa62255951157a6b9` |
| First enabled run and lifecycle | `03bb21e3003c458615a3678b98e4498b61cab7eabfe770e4cc7ef4404a37d726` |
| Repeated same-binary control | `8677561758e47f3ba5ad5a25dd5f6a426d4fe524c5aff628ac4e5c3a4f6cc21c` |
| Repeated same-binary routed run | `9149cbfab22dbb3d86f2d99f18227ec2104a0db69eba8aa13224f1b7432af3ff` |
| Wide-routing warm decode replays | `7836138e0e9264013b858da88ec93fe553ff75e14b8b4e62d39df30efa766eb8` |
| Control warm decode replays | `e7bae8af7bde9224efdbe8e3f995822cd7c96ac847da1b066a42a2b3a13f702c` |
