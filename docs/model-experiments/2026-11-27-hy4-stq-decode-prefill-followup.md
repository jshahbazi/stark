# HY4STQ-PERF-04 · Wide-routing decode and FFN follow-up

The starting worker is `8038d2066`. The previous matched OpenCode pair
had 32.93 s of fresh 27K prefill with wide routing enabled. Its
separate-process warm decode medians were 48.47 tok/s off and 45.55 tok/s
on. Both target verification and DFly2 draft time moved in that
comparison. Wide routing activates only on prompt chunks of at least
1,024 rows. This study tests whether its decode penalty reproduces
within one loaded process and targets at least 48 useful tok/s while
reducing prefill time. [Prior routed experiment](2026-11-27-hy4-stq-routed-attention-prefill.md)

The artifact is `beamster/Hy4-Flash-Sushi-STQ-4bpw` at
`61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`. The 128 GiB M5 Max
uses the image-loaded Metal worker, FP16 KV, resident weights, 4,096-row
prefill, DFly2 depth five and a fitted 49,152-position cap. The actual
OpenCode 2.4.2 sequence renders 16,229 fresh, 27,179 continued and
27,229 fresh input tokens. The fixtures use 12 tools and 96 greedy
outputs. Prefill counts computed input rows and useful decode counts
95 output transitions. These are local OpenCode and worker diagnostics.
[Artifact, fixture and host controls](2026-11-27-hy4-stq-open-code-performance.md)

## Decode attribution

In the historical warm 27K comparison, wide routing coincided with
about 74 ms more target verification GPU time and 32 ms more draft
wall time per 96-token response. The decode graph and draft depth
were the same. A separate benchmark worker compiles both routed
prefill paths and alternates them for successive requests through a
private selector. The selector changes only the wide prefill dispatch.
The main window and loaded artifact stay fixed. It verifies input and
output hashes, reused positions and DFly2 counts in memory.

The initial benchmark launched from a private worktree. It rendered
a separate 27,229-token input with a different hash. Across
eight warm ABBA/BAAB replays of that input, the routed-on and routed-off
arms matched output hashes, 24,576 reused tokens and 130 proposed,
69 accepted drafts in 26 rounds. Off to on, adjacent-pair medians
moved −8.83% for prefill, +1.79% for useful decode and −1.21% for
target verification GPU time. The two per-arm target verification
medians were 1.58473 and 1.58501 s. Decode ranged from about 39 to
47 tok/s within the same session. A model-free check reproduced the
original input from the original launcher and OpenCode workspace
directories. The separate replay uses that setup and gates the
input hash before native dispatch.

The exact run used one fresh OpenCode prime with wide routing off,
then four replays per arm in ABBA/BAAB order. It matched the original
27,229-token input and 96 output IDs in every request. All warm
replays reused 24,576 positions. Each response proposed 120 drafts,
accepted 70 and took 25 rounds in a 49,152-position window.

| Same-loaded exact 27K, warm | Routed off | Routed on |
| --- | ---: | ---: |
| Median prefill seconds | 3.778 | 3.637 |
| Median useful decode tok/s | 46.58 | 49.15 |
| Decode range, tok/s | 44.87–49.87 | 44.10–49.96 |
| Median verifier GPU seconds | 1.559 | 1.481 |
| Median DFly2 draft wall seconds | 0.389 | 0.376 |

Wide routing won three of four adjacent prefill pairs and two of four
adjacent decode pairs. Three of four wide replays reached 48 tok/s.
The earlier 6% decode penalty did not reproduce inside one worker.
The observed drift affected both draft and target timing. The cause
of the host or device timing changes was not isolated. The only exact
fresh sample in this replay was the routing-off prime, which took
40.89 s to prefill and decoded at 46.34 tok/s. The first attempt
at the exact retry gated the startup image check's text request as
if it were an OpenCode fixture and yielded no timing.

## Component experiments

| Attempt | Result |
| --- | --- |
| HY4STQ-PERF-04a · Wide packed FFN products | Wide K32 applied to eight actual SPEC1 K6 FFN and latent projection banks. Matched all 212,336,640 F16 outputs across 1,024 and 4,096 rows. GPU time fell 25–32% over nine alternating pairs. The 1,025- and 2,653-row tail repeats matched another 152,534,016 F16 outputs, preserved the guard regions and saved 22–30%. The prior sampled 27K FFN bucket was 2.71 s, including preparation and other work. |
| HY4STQ-PERF-04b · Fuse FFN SwiGLU preparation | Keep the ordered gate and up Hadamards, precision boundaries, SwiGLU and down Hadamard in one dispatch. The 1,536- and 13,824-channel release scales matched 238,817,280 F16 outputs across 1,025, 2,653 and 4,096 rows and both activation boundary policies. At 4,096 rows with the F32 boundary, the shared chain went from 0.351 to 0.077 ms and the dense chain from 3.333 to 0.637 ms. Across the full workload the two FFN changes project about 0.48 s of prefill savings from the component results. |
| HY4STQ-PERF-04c · Narrow verifier K32 | Nine real split-K shapes at widths one through six matched all 6,365,184 F32 split partials and 1,279,488 reduced F16 words. The seven attention shapes at width six summed to 0.32725 → 0.30475 ms. Routed latent projections lost 2–3% and keep K16. Threadgroup widths of 32 and 64 showed little benefit. The M8 trial was rejected by Metal's cooperative-register shape constraints. |
| HY4STQ-PERF-04d · Fuse the narrow split-K attention gate preparation | Read the ordered F32 split partials and round to the same F16 value before finishing the gate and preparing the output projection. With release scales at widths one through six, all 1,032,192 prepared F16 outputs matched across the two gate geometries and both precision modes. Paired component time was 0.482–0.569 of the three-dispatch path. This saves about 5–10 µs per gate in isolation. |
| HY4STQ-PERF-04e · DFly2 register MLA | Reuse the existing partitioned register attention on compact DFly2 target and local KV. Both arms matched all 162 proposals across the real draft weights, two selection sizes, three draft widths and nine alternating anchors per case. At 2,051 selected positions and five drafts, proposal median wall time was 12.908 → 12.173 ms. At 512 positions it was 12.065 → 11.971 ms. The initial component setup missed the native shader include path and generated no timing. |
| HY4STQ-PERF-04f · Batch DFly2 correction commands | Run all greedy correction and argmax slots in one command, with per-slot GPU IDs and the incumbent sequential path when the caller requests confidence. Matched all 198 proposals and the confidence fallback results. At 2,051 positions and five drafts, draft wall went 12.171 → 11.438 ms. Measured GPU time stayed near 10.6 ms and correction host overhead fell from 0.862 to 0.186 ms. The expected saving near 25 rounds is about 18 ms, so the prototype stayed out of the worker. |
| HY4STQ-PERF-04g · Gather DFly2 target KV on the GPU | Gather the target's already rotated F16 KV using the selection buffer the GPU produced. Keep the selected order and existing bounds check. With real DFly2 weights, 27,229 target positions and register MLA in both arms, all 162 proposals matched. At 2,051 selected positions and five drafts, proposal wall went 13.284 to 11.935 ms. That is about 34 ms across 25 rounds before full-worker measurement. |
| HY4STQ-PERF-04h · Reuse PAIR1 K32 for narrow indirect verifier jobs | The prior wide product kept the same grouped route ABI. At one, three and six verifier rows, all 2,064,384 F16 values matched across 21 release-bank cases, with guard rows intact. GPU time dropped 10–35%. The six-row mixed-overlap projection sum went 0.4763 to 0.4101 ms. Applying that component delta to 41 routed layers and 25 rounds projects about 68 ms. The first attempt opened the trunk shard and the expert fixture rejected its metadata before measuring. |

The verifier candidate keeps the established K16 split reduction and
F16 boundaries, preserving the route and cache contract. The draft
attention candidate reuses the unchanged FP16 cache and FP32 online
softmax. Its tiled accumulation order differs from the old draft
attention and therefore needs the real output and acceptance checks.
None of the new options changes the flexible-context key. The initial
combined worker was stopped during startup after checking its embedded
Metal source against the patched tree. It never ran a fixture.

## Initial OpenCode combination and lifecycle

The initial validated worker enabled wide FFN, fast verify and DFly2
register MLA in addition to wide dense attention and routing. Input
and output hashes, acceptance counts and reuse matched the original
16K and 27K OpenCode fixtures. It passed the fresh replay of the 27K
continuation, actual image invalidation and the 8,192/8,192/12,288
windows with 0/4,096/0 reused tokens. The original continuation still
reused 12,288 tokens.

| Fixture | Prior settings, prefill seconds / decode tok/s | Combined, prefill seconds / decode tok/s |
| --- | ---: | ---: |
| Fresh 16K | 19.88 / 47.18 | 21.10 / 46.18 |
| Continued 27K | 18.78 / 45.19 | 19.41 / 42.72 |
| Fresh 27K | 38.01 / 41.13 | 41.55 / 38.85 |

The fresh 27K prefill GPU spans totaled 40.96 s with 15 ms in
Engram. In the control the same binary reported 37.33 s of prefill
GPU spans. Both exceeded the earlier 32.65 s GPU observation.
Both draft and verify time rose in the combined run. Pressure stayed
normal and swap held at 904.06 MiB. The two new processes took
155.66 and 230.43 s to report ready. The pair records slower
OpenCode timing in both processes and a further slowdown with the
new settings. It does not determine an effect within one load.

## Same-loaded tuning comparison

The private worker loads both sets of pipelines and selects them at
request start. Wide routing and wide attention are active on every
arm. An original fresh 27K OpenCode request primed the worker under
the prior settings at 33.99 s prefill and 47.73 tok/s decode.
Twelve warm replays matched the original input and output hashes,
24,576 reused positions, 120 proposed drafts, 70 accepted and 25
rounds. The candidate here includes FFN, narrow attention and
DFly2 register MLA with GPU gather. Narrow PAIR1 was added after
this private worker snapshot.

| Warm exact 27K, four replays per arm | Prior | Combined |
| --- | ---: | ---: |
| Median prefill seconds | 3.717 | 3.511 |
| Median useful decode tok/s | 45.86 | 47.81 |
| Median draft seconds | 0.397 | 0.356 |
| Median verify seconds | 1.675 | 1.630 |

The combined mode won four of four adjacent decode pairs and three
of four prefill pairs. It peaked at 49.93 tok/s. The FFN-only
comparison ran two prior and two FFN replays in ACCA order.
Prefill ratios to the adjacent control were 0.958 and 1.061. The
FFN-only end-to-end gain is inconclusive. The first private build
had a copied Metal module cache containing absolute PCH paths and
produced no worker. The successful build regenerated that cache.

## Final OpenCode and flexible context

The release worker includes narrow PAIR1 and GPU DFly2 gather.
It reproduced all three original input and output hashes and
draft counts. The continuation still reused 12,288 positions.

| Fixture | Prefill seconds | Prefill tok/s | Useful decode tok/s |
| --- | ---: | ---: | ---: |
| Fresh 16K | 22.11 | 733.88 | 50.42 |
| Continued 27K | 17.64 | 844.34 | 55.19 |
| Fresh 27K | 34.33 | 793.17 | 45.67 |

The earlier wide-routing repeat measured 19.52, 18.76 and 32.93 s
of prefill on these fixtures. Fresh 16K GPU spans fell from 17.46
to 16.26 s across the two runs. Engram took 4.83 s in the final
fresh 16K request, against 1.74 s before. The continued 27K request
saved 1.12 s across the two runs. Fresh 27K had 34.02 s of GPU
spans and 12 ms of Engram work. It was 1.40 s slower than the
earlier Stark repeat and 6.75 s behind the fastest prior Sushi
observation with that runtime's rendering and KV8.

Five additional same-worker 27K decode replays were warm and
reused 24,576 tokens. They measured 45.84–50.44 tok/s with a
47.71 median, and two reached 48. First-use decode reached 48
on fresh 16K and continued 27K. Fresh 27K is still below the
target. A fresh replay of the continued input reproduced its
output at zero reuse. The 8,192/8,192/12,288-position checks
returned identical outputs with 0/4,096/0 reused tokens. The
actual image request disabled drafting and invalidated the text
prefix before the next text request.

Keep `STARK_HY4_STQ_WIDE_FFN`, `STARK_HY4_STQ_FAST_VERIFY` and
`STARK_HY4_STQ_DRAFT_REGISTER_MLA` opt-in on the native validation
branch. Their basis is component correctness, the controlled
combined warm improvement and this exact-fixture lifecycle.
The isolated FFN result and fresh 27K speed remain open.
Repository policy, release Clippy, the native inspection example
and formatting passed. Release Rust tests passed 183 with six
ignored. The unchanged final Python metadata code passed 189
focused tests and Ruff.

## Remaining differences from Sushi v1.8.1

Stark serves the same STQ pack with grouped trellis experts,
chunk-wide attention gate and output products, register MLA,
fused gate/SwiGLU preparation and DFly2. The tuned Stark dense
path reads packed weights directly. The earlier decoded NAX
prototype lost to direct K32 on the sliding banks it tested.
Sushi selects decoded NAX for several wide projections and uses
BF16 boundaries in its serving graph. [Sushi wide-span path](https://github.com/beamivalice/sushi/blob/v1.8.1/docs/hy4-flash-arch.md#attention-both-kinds-mla) · [Stark NAX screen](2026-11-27-hy4-stq-dense-attention-prefill.md#component-screens)

Sushi fuses the attention output norm, its four-stream residual
write-back and the FFN mix and norm. It also joins normalization
and write-back at the expert output. Stark still issues the
attention output norm, `hc_post` and `hc_pre` as separate
dispatches and submits the prefill graph layer by layer. Profile
the boundary work and the routed-plan cuts on the exact fresh
27K fixture, then screen one F32-order-preserving boundary fusion
under the existing context and reuse checks. The upstream
dispatch-reduction result bundles multiple changes on an M5
Ultra and establishes no Stark speedup by itself.
[Sushi layer graph](https://github.com/beamivalice/sushi/blob/v1.8.1/src/hy4flash/hy4_model.zig#L2787-L2803)

Sushi's KV8 cache reconstructs BF16 rows once per layer and
chunk for wide prefill. Its documented direct low-bit read was
slower in a sparse component test. A matched KV16/KV8 bracket
in Sushi would price the difference before adding a Stark
cache format. [Sushi cache layout and attention measurements](https://github.com/beamivalice/sushi/blob/v1.8.1/docs/hy4-flash-arch.md)

## Evidence identities

| Receipt | SHA-256 |
| --- | --- |
| Same-loaded ABBA on the independently rendered fixture | `e137c9e28d2099c59ba4fafa14440d9e4c851113eb3eb9eb49db72a9bbbc7968` |
| Same-loaded wide-routing ABBA on the original 27K fixture | `e51e2c69dcb684505e6ceb4451462d11738df7711b7149da01acee877a173e99` |
| FFN release banks at 1,024 and 4,096 rows | `0ccd56481827b4b6f312910c93708860369480c34942ba8cbd41780fcbd2fca2` |
| FFN partial tiles | `460d332e7b362de7e88d8c8362a2ca7cc0290b8b8cf75aaf17f62d2604f45f07` |
| Fused FFN prepare | `65986471969efe233a2797f2ae83d01348254b920eee1681aed63a3c714b8885` |
| Narrow K32 screen | `fc6593c07abdb894d7722be492ef0f329095ebeac339f75d2214559f4bf9fcb7` |
| Narrow K32 tracked-source repeat | `81b33d344562516280e2f7f1c6265bdc04b6452d7ddaf90127f61281e184241a` |
| Narrow TG64 screen | `7faa405be4fcd730697f725923ce3e659e3698106cab35ab2c5c0d96ead9b9f8` |
| Narrow gate fusion | `28da8567637fda499744714a0cafcebd56f23180aaa0192b2edf164e1e6e14c7` |
| DFly2 register MLA | `89c41e4dc38ab690e3748fa14f2a202391c53c2a648db9629571ae1bc47ed694` |
| DFly2 correction command batch | `0c29b46b1fd9844bf824daa97b4dac4934e9ecdf6317031d56742a2a7e9fd9ba` |
| DFly2 GPU target gather at 27K | `adc443dba7b6b0fe72301d31b19a0181de2b0d669ad3c9a0e18112b443965a95` |
| PAIR1 K32 narrow indirect verifier component | `89666f7faccf54ae18212f1f85eac6da87e9dae23f8d55b771635539775efe11` |
| Aborted stale-shader worker | `8007fbaa46f34903eb42b4b87de71eab38fdc9a3afc34f3a2017ef45804dd980` |
| Rebuilt combined worker | `b06a8872bdfd640bdf08d4f6576cabacc3a386d5101e837bc7358e62c0322ed2` |
| Combined OpenCode and lifecycle | `16eab8c918774339dfee1b0cfc2d75e8b53910c74ec40ea29e131b85f92daf4a` |
| Same-binary, prior settings | `d7b08a13a6cb78d9daa5faa036208d0697328ce89b411f07f02f404ce8f16e0e` |
| Private same-loaded tuning worker | `ad1b539aa88a5bc7029ca76a08a83ccf510822879cba1d939771c2a6b158f75c` |
| Same-loaded tuning and FFN-only replays | `58411b024788ea3f920ff45b5b80728918fced9ebcdb9947eb29e281efb87ca1` |
| Final release worker | `1ad738d919f0f2cce78958b9a432c3a496e2961dafb5c22617e87a1907786f79` |
| Final OpenCode, warm decode and lifecycle | `4914d9e2bcd064d115d91cfe34a493253dcb879eaafc3744fc3ae550bcb85019` |
