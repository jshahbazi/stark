# HY4STQ-PERF-06 · Price Sushi KV8 and reduce wide dense preparation

The starting worker is `67e4af0a4`. Its fused-boundary and
batched-draft settings measured 31.87 s fresh 27K prefill and
51.88 useful decode tok/s across three OpenCode turns. A previous
Sushi v1.8.1 observation was 27.58 s, a 4.29 s difference across
separate renderers and loads. [Starting measurements](2026-11-27-hy4-stq-attention-ffn-boundary.md)

Both runtimes use `beamster/Hy4-Flash-Sushi-STQ-4bpw` at
`61528ba635a623f127c5a7b1e1f4ac2c2b0d4885` on the 128 GiB M5 Max.
The runs use the OpenCode 2.4.2 synthetic code review, 12 tools,
greedy generation, 96 output tokens, vision loaded and a 49,152
position cap. Each runtime renders its own API input. The measured
prefill and decode phases are local worker and server diagnostics.

## Sushi KV-width control

Sushi's KV8 uses unsigned affine codes over groups of 64 latent
values, with BF16 scale and bias. The rope portion stays BF16.
A latent and rope row occupies 672 bytes at KV8 and 1,152 bytes at
BF16. Wide attention reconstructs BF16 rows once per layer and
chunk before computing attention. That reconstruction is relevant
to whether compressed storage helps wide prefill.
[Pinned Sushi design and measurements](https://github.com/beamivalice/sushi/blob/v1.8.1/docs/hy4-flash-arch.md)

The same v1.8.1 release executable, verified pack, OpenCode
workspace and server settings ran in 8/16/16/8 order. Each length
and width started a separate server to exclude live prefix reuse.
DFly2 remained enabled and disk prefix storage was disabled. The
first attempt relocated the harness without updating its tokenizer
path and stopped before issuing any request. A model-free check
passed after correcting the path.

| Sushi fixture | KV8 prefill s | BF16 prefill s | Seconds saved by KV8 | KV8 decode tok/s | BF16 decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh 16K, 16,179 tokens | 16.81 | 15.96 | −0.86 | 52.77 | 52.91 |
| Fresh 27K, 27,179 tokens | 29.08 | 27.90 | −1.18 | 41.03 | 47.12 |

Each median has two independent process observations. The 16K
prefill pairs were 15.18 and 18.44 s at KV8 and 15.65 and 16.26 s
at BF16. The 27K pairs were 29.05 and 29.11 s at KV8 and 28.47
and 27.32 s at BF16. Both fresh lengths had zero cached tokens
and identical outbound request hashes across the four arms.
All turns used 12 tools, returned 96 tokens and completed under
normal memory pressure with 816.94 MiB swap. Decode quotes 95
output transitions over the server's generation clock.

The KV modes produced identical response hashes at 16K and
distinct, reproducible hashes at 27K. KV8 explained none of the
historical 4.29 s gap as a prefill advantage. It cost 1.18 s at
27K in this bracket. The KV8 reconstruction adds one possible
source of work, and the integrated bracket does not isolate that
stage. The new BF16 Sushi median is 3.97 s below the old Stark
median under separate runtime renderers. Keep Stark's F16 cache
for this throughput effort. KV8 retains its lower storage cost.

## Stark wide dense IO

The exact candidate reuses the register-only PAIR1 H128 transform
for the aligned input and output boundaries of wide dense
projections, and uses it twice in the already fused attention
gate and output preparation. Each SIMD group owns one 128-channel
block with four values per lane. The two cross-group butterflies
stay in registers. The same F16 product bits, scaling, sigmoid,
and BF16 projection option carry through. The existing path runs
on smaller rows or unaligned widths. The option uses the same KV
layout, context admission and decode dispatches.

## Component screens

The register path matched 1,146,175,488 F16 and F32 values across
the 8,192- and 16,384-channel attention shapes, at 1,024,
2,653 and 4,096 rows, with and without BF16 projection rounding.
The screen used release SUH and SVH scales, deterministic finite
values and nine alternating GPU pairs per case. At 4,096 rows
without BF16 projection rounding, input preparation fell about
5–6% and finishing about 3–6% in median paired time. The fused
gate showed about 24% at 8,192 channels and 2% at 16,384.
Some narrow component pairs varied more. The first component
runner included a duplicate Metal source and stopped at shader
compilation. The direct check uses the current embedded source.

An independent PAIR1 K32 trial combined up to 64 routed rows per
job to reuse each decoded weight fragment across four 16-row accumulators. Across 16,128 and 65,280 routed rows from
all 512 layer-one experts, it matched all 375,128,064 F16 values
and preserved the canaries. It took 9–10% longer on the smaller
shape and 31–35% longer on the larger shape. The job count for the larger shape
fell from 2,288 to 1,272 while the sum of the three products rose
from 17.515 to 23.372 ms. The existing 32-row grouping stays in use.

## Stark 16K and 27K OpenCode

The Rust 1.98.1 release worker retained the previous fused boundary,
batched drafting, resident weights, 4,096-row prefill and DFly2 depth
five. The two loads used the same binary. Only
`STARK_HY4_STQ_REGISTER_DENSE_IO=1` differed. The first enabled load
reached ready, but the harness omitted the new readiness field and
ended before sending an OpenCode turn. The retry includes the field
and checks that the worker enabled the option.

| Fixture | Control prefill s | Register prefill s | Register prefill tok/s | Control decode tok/s | Register decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh 16K | 18.41 | 29.08 | 558.04 | 51.54 | 51.02 |
| Continued 27K | 17.21 | 17.26 | 862.62 | 56.57 | 54.79 |
| Fresh 27K, medians | 32.10 | 31.38 | 867.78 | 50.00 | 51.66 |

Fresh 27K prefill saved 0.72 s in wall time and 0.85 s in measured
GPU spans across the two separate worker loads. The two control
turns took 32.08 and 32.12 s. The three enabled turns took 31.38,
31.98 and 31.33 s and decoded at 50.67, 51.82 and 51.66 tok/s.
They all computed every input position. Useful decode counts
95 transitions. Prefill counts computed positions.

The fresh 16K GPU spans moved from 16.44 to 15.83 s while the
wall time rose by 10.67 s. The enabled run reported 2.23 s in
Engram and around 11 s outside either Engram or measured GPU
spans. Swap rose about 35 MiB in that load and pressure stayed
normal. The unaccounted first-use time remains unresolved.

All five OpenCode input and output hashes, draft statistics and
reuse counts matched the earlier Stark run. The continuation reused
12,288 positions and reproduced its output when replayed fresh.
On the same loaded worker, the 8,192/8,192/12,288 requests returned
identical outputs and reused 0/4,096/0 positions. The actual image
request disabled DFly2 and cleared the prefix before the next text turn.

The latest 31.38 s Stark median is 3.48 s above the new Sushi
BF16 median and 3.80 s above the original 27.58 s observation.
Those cross-runtime figures use different rendered inputs and
timing boundaries. Keep the register kernels opt-in on the
native STQ validation branch.

Repository policy, Rust formatting, release Clippy and the native
inspection example passed. Release Rust tests passed 183 with six
ignored. The focused converter and runtime suite passed 189 tests.
Ruff lint and formatting passed.

## Evidence identities

| Artifact or receipt | SHA-256 |
| --- | --- |
| Sushi v1.8.1 release executable | `13feaab689501082fd166f6a77e80683362b83ac5b37b527dee0d88875d3bf57` |
| Sushi 8/16/16/8 option launcher | `148de726c75bf562e7a4c2ffa20e71ef5ac7772b1d154ba70f2401870abc3015` |
| Sushi isolated OpenCode proxy and fixture | `f5a1348a1a57bd26b96978bc3a1c0f6b96b39cf1e6281f86d8f0c3688a42390b` |
| Initial Sushi startup and harness failure | `51de2fd78b14fe0def5a8459fcdb7a2e6ef76713b8986754ad3c1b33881adc4a` |
| Sushi KV width aggregate receipt | `c705e94a0b8bc1e07946532029a4d99de47a92c42d776029d92916e069421651` |
| PAIR1 64-row component receipt | `109ec16eb6c4bcd42e17721e50443825a9742b1acc65acb5a336a3581e28d0ad` |
| Dense IO component fixture | `357d253d26f187d37187ffa6b15883317f7ea13f48a4c02273daf922abf02b85` |
| Dense IO component receipt | `ac5a0aff88bc8916111d9a209f205054c3d2f60a270a3ef68a06d8f5f5897383` |
| Stark release worker with current Metal sources | `c9bfb8b51fbd3d3b85d8554999e57d93d25c43453c13ae9d0a6b2f3244b9d377` |
| Stark readiness reporting failure | `0ab22aa193e5ad17bd938185d8835028c155e53f7c454d5ef22f8ce5dbf5121d` |
| Stark same-binary flags-off control | `122db410116688fc77b5127229dac423f377f2a64d949e80a68fe654ef8317d1` |
| Stark enabled OpenCode and lifecycle aggregate receipt | `a72736b621f7890d8754344afbe41a686912afa44e409bc01d635b68b1ff12d7` |
