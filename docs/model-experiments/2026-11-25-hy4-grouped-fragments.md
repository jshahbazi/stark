# AngelSlim Hy4-flash Q3_K grouped expert fragments

## 20261125-HY4FRAG-01 · Reuse packed expert weights across prefill rows

The [preceding 4K stage sample](2026-11-25-hy4-multi-query-prefill.md) measured
2.82 s in routed experts. The candidate applies Stark's Inkling gathered
fragment layout to Strata's grouped Q3_K/Q4_K gate and up and Q4_K down
products. Metal 4 tensor products consume F16 operands and accumulate F32.
Gate and up share one product and keep the original clamped activation.
The next packed weight tile is fetched ahead of the product and expanded
into alternate threadgroup storage. The existing route plan, combine,
and decode kernels are shared with the control. The shader header records
the Stark and upstream source lineage.

The [pinned Q3_K artifact and original projector](2026-11-25-hy4-native-q3-k.md#artifact-and-runtime-identities)
ran on the 128 GiB M5 Max with streamed experts, a 40 GB arena, 8,192
positions of capacity, 4,096-token chunks, grouped F16 routed operands,
packed Q8_0 dense projections, multi-query MLA, resident embeddings,
demand retention, and greedy NextN at depth two. The four same-binary
arms ran in the listed order with two requests per load, prefix reuse
and stage sampling disabled. Every request matched 64 reference output
IDs and accepted 41 of 42 proposals in 22 rounds. The host snapshot at
each arm reported AC power, nominal thermal state, and no named competing
processes.

| Arm in run order | First prefill tok/s | Warm prefill tok/s | First decode tok/s | Warm decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Fragments first | 557.94 | 501.59 | 28.35 | 39.11 |
| Control first | 486.89 | 432.37 | 27.33 | 38.44 |
| Control last | 464.23 | 422.47 | 27.38 | 35.74 |
| Fragments last | 493.18 | 490.28 | 27.70 | 39.73 |

Warm prefill improved 13–19% across the ranges. The first-use candidate
range was 493–558 tok/s against 464–487 tok/s. The 743.2 tok/s published
target still allows about 5.51 s for 4,096 tokens. These are local fixture
diagnostics at fixed output length.

Stark base `1da97be52d38d360d2e61d296cf959efc5546263`, full measured
source patch SHA-256
`b8c7f8e7c6983287fc1997b0bc3091d42bb46dfb7f542d625604861a1d552c7a`.
The candidate from the Sol agent before wrapper wiring is
`f5bfe53310288b96bdec93caa84969eee9cbbecb3012e4f414e7d0a75a0aadbd`.
Locked offline Rust 1.98.1 release worker SHA-256
`8e625bf06458e52ba8b1dec14585c89ba1274e471731755afcf3ab840c40c675`.
Harness SHA-256 `f4d3bd627c3ced5bf58a7ad2eba9c75369a8b42bf074197216e5876ab2e13f59`.

| Receipt | SHA-256 |
| --- | --- |
| Fragments first | `91fcb41b14cec1c4c5f81a4fb2e6a9ed5fedf1ea88c969f185e079fb3193cc87` |
| Control first | `054a348e2cc577c2458f222fb097e9f05fd7d91c8d200fd41c34824ab5059f2b` |
| Control last | `0b62fbd60ac82a653ff11c880deb3a3ea9e5250d9c20046582662d9cf6190340` |
| Fragments last | `d67f2ce1f451ad46ba27c63983a83d664144fb0ac3957edcf2429da97068a1c9` |
| Natural-text top-1 | `2dfa1b15fb7dc9f7b54fc48562963d5778e243ed7c5271c715d0adcbb6c6c566` |
| Separate stage sample | `f6b1edfba93791194c7e444bc28570abf911300bd0e517eb2e64197b5c14e1d0` |

The candidate agreed on all 13 teacher-forced natural-text top-1 rows,
with six actual next tokens correct in each arm. The separate profile
recorded all 42 blocks and measured 2.75 s for routed experts and 3.15 s
for attention projections. Other stages shifted from the preceding
sample. The stage-off bracket measures the end-to-end improvement.

## 20261125-HY4FRAG-02 · Verify three and four NextN drafts per round

The depth-two policy spends one target round per 2.86 useful tokens on
the matched 4K fixture. The candidate supports a request ceiling of four
drafts with `STARK_HY4_MTP_MAX_DRAFTS=4`. Its verification window has up
to five rows. Every accepted draft still matches the target greedily.
Prefix identity includes the requested depth. Vision continues to use
ordinary decode. The suggested depth stays two.

One same-worker sweep at the HY4FRAG-01 geometry with fragments enabled
ran zero, two, three, four, four, three, then two drafts. The 4K fixture
produced 64 IDs and the rendered 75-token TypeScript fixture produced
192 IDs. All arms for each input agreed with its serial generated IDs.
The 4K output also matched the llama.cpp reference.

| Input | Drafts per round | Decode tok/s | Proposed / accepted | Target rounds |
| --- | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 15.00 | 0 / 0 | 0 |
| 4,096 | 2 | 32.96–38.72 | 42 / 41 | 22 |
| 4,096 | 3 | 41.19–43.69 | 47 / 47 | 16 |
| 4,096 | 4 | 44.16–45.75 | 51 / 50 | 13 |
| 75 | 0 | 17.60 | 0 / 0 | 0 |
| 75 | 2 | 34.42–34.83 | 149 / 116 | 75 |
| 75 | 3 | 33.97–34.44 | 188 / 127 | 64 |
| 75 | 4 | 32.31–32.39 | 228 / 133 | 58 |

The 4K fixture advanced 4.85 tokens per target round at depth four.
The short coding fixture advanced 3.29. The larger verify windows cost
enough to favor depth two for that coding input. Use matched useful
decode throughput and output agreement when evaluating a depth change.
The first attempt to launch the depth sweep was refused by the serial
GPU lock while the prefill bracket was still active. It measured no work.

| Receipt | SHA-256 |
| --- | --- |
| Depth sweep at 40 GB | `059b2bac228ec32f77f66955337745c6074dc0a03721c2d001c63a9268eba13a` |
| GPU lock refusal | `24f1e11cbd50a0ca54e45742e59e6d34087d0a37689ed41795cf0da50f974d26` |

Depth sweep harness SHA-256
`f941ad1e036781e1409d5b3b58ec496eb1f801500492c84793a2451613a3d918`.

The image-loaded worker reused 4,096 tokens for the same depth-four
continuation and matched a fresh generation exactly. Switching the
request to depth two or changing to an exact-fit context reused zero
tokens and produced the same 16 IDs. Both image answers passed with
zero proposals. The image, invalid-image and cancellation transitions
cleared the prefix. The first smoke used a 4,096-position request with a
4,096-token chunk plus a continuation. It correctly rejected the request
before image checks. The corrected smoke used an 8,192-position request
and asked for four drafts in its text arms.

| Validation receipt | SHA-256 |
| --- | --- |
| Vision smoke, wrong request capacity | `0d32e03428064df5be669f67e044b45363a3e54eb4c221c5cba42ceee2d27842` |
| Vision and depth-four text transitions | `095944980ec9d4b5c1eb365d903f7b5bcdf3c5580c9a2415b2cf3e9704cb9831` |
| Prefix reuse, depth and exact-fit context | `4f52257c4ca2c7d683273e64cf93d3478bf92641667e801d2dd8a626422be61b` |

## 20261125-HY4FRAG-03 · Seed streamed retention from the prompt tail

The candidate protected up to 64 already resident experts per routed
block based on the final 256 prompt rows. It used the existing retention
quota and left room for decode demand. It initiated no new reads.

At 40 GB the first 4K control completed and the candidate timed arm was
refused for competing build activity. At 48 GB the opening control did
not admit the requested 8,192 context. The remaining arms each allocated
47.998 GB and admitted 8,192 positions. Each load ran one depth-two 4K
request, then two depth-four requests. All nine completed outputs matched
the 64 reference IDs.

| Arm | Depth four, first tok/s | Depth four, repeat tok/s | Repeat expert reads GB | Repeat host cuts s |
| --- | ---: | ---: | ---: | ---: |
| Seed first | 48.78 | 50.48 | 16.75 | 0.272 |
| Seed last | 48.40 | 49.78 | 16.80 | 0.286 |
| Closing control | 47.79 | 49.90 | 16.54 | 0.252 |

One seeded request exceeded 50.34 tok/s. The repeat fell below it and
the unseeded closing repeat landed within the seeded range. Expert reads
and host cuts increased slightly in both seeded repeats. Remove the
switch and keep the independent fragments and depth controls.

| Receipt | SHA-256 |
| --- | --- |
| 40 GB opening control | `46efea28a788053aba3b94e4b9eb85493f9d6fbd43a5389cde0a847423bf4405` |
| 40 GB candidate host refusal | `cf51110b86576dd7fa9eae333a7bb90205912df4e46e6c61e52ca0f1ea43c90e` |
| 48 GB opening capacity refusal | `aec55a7a6685905e10bce6900603396dd11064601a6f98a70b946b7c9cac3974` |
| 48 GB seed first | `4bedcc89058f66ed8ca4fe5e44d977d45c14cb6fd810e6757bde1a279f1d7af4` |
| 48 GB seed last | `24833eff5400305651b6236110f15b9eb9251162af7a4f7362418c814992dd5e` |
| 48 GB closing control | `6665bf1742c55c4aa8c44b7e0cd6fc15770068692a54a1b252704b746997b577` |

The 40 GB screen ran on Rust 1.98.1 release binary SHA-256
`44ce4436b4148f3348a4aba0baf2f26efc78f7e6ce73030b405cfdd076de7f2f`.
The 48 GB bracket used the HY4FRAG-01 worker with the depth sweep harness.

## 20261125-HY4FRAG-04 · Share packed attention weights across four query stripes

The gate and output weights account for 73% of the multiply work across
the seven trunk attention projection shapes. The candidate shares one
Q8_0 tile across four query stripes for these two projections. It stages
only weights, preserves the packed path's F16 conversions and K16 F32
accumulation order, and uses a double-buffered 18,432-byte threadgroup
tile with no additional arena allocation.

An image-loaded 40 GB 4K bracket with the routed fragments enabled ran
four arms with all 64 reference IDs and the same draft work. Every host
snapshot included a separate `cargo` job, so this is an exploratory
comparison under contention. The shared path was slower in both
first-use samples and had no clear warm gain. It was removed.

| Arm in run order | First prefill tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| Shared first | 456.18 | 468.19 |
| Control first | 540.32 | 503.37 |
| Control last | 567.73 | 464.17 |
| Shared last | 439.30 | 455.31 |

The separate dense candidate before wrapper wiring is
`678ece2902b85b61f6878040cb33bb9a756ca1a0a81f2d10adf9b6014a39aa2b`.
The full source patch on `1da97be52` is
`19536ead91f769686eb7ab4ad239ba4f87c795bfff428b669cdf8baf0231d4bf`.
Locked offline Rust 1.98.1 release worker SHA-256
`afd79b60d4b45ce50c3a69933c1abed6541fd66ad315214106c3d532f491e605`.

| Receipt | SHA-256 |
| --- | --- |
| Shared first | `93c19d4674dcae6da0f76a80a459e5b65f032b2aa11b0b81fe0f68a3d9b3c689` |
| Control first | `3df081fd507fe46ed1475ce5d7fdf9376b71b8e897cc16f05328a0ad367aa292` |
| Control last | `9ebc9d0c0e27818665e4f68df102a6901ff0069a09a743230c4a3a75bd3d9750` |
| Shared last | `95540c4293684cea42afa43a0544883f636ed3200bc8c80d936170dbbf1fe288` |

## 20261125-HY4FRAG-05 · Skip duplicate route capture in the native verifier

The private candidate skips copying target routes and sorting a second
copy of the read-ahead predictions for diagnostic capture. Demand,
read-ahead, and the GPU plan still consume the same data. All nine 4K
requests at the 48 GB arena matched the reference with the same draft
work in each depth. Background build activity was present at each load.

| Arm | Depth four, first tok/s | Depth four, repeat tok/s | Repeat host cuts s |
| --- | ---: | ---: | ---: |
| Skip first | 42.45 | 46.26 | 0.320 |
| Control | 37.38 | 46.17 | 0.357 |
| Skip last | 45.94 | 41.15 | 0.530 |

The repeat moved in both directions across the bracket. The run did not
establish a throughput gain. Keep this candidate out of the native worker.
The route capture patch SHA-256 is
`d76bd66890453679e8897fba0afb4c8b4539c36d9e2d12f8f13cc105addacef9`.
This ran on the HY4FRAG-04 build. The trial harness SHA-256 is
`23678012ef69de36ea31394772057447f8dd2fbdca4903dde7b6b14420b90e16`.

| Receipt | SHA-256 |
| --- | --- |
| Skip first | `fda0dbb659ed5927a9b1fda081a3b62561898c36e8f92b466243b81ed448e0ab` |
| Control | `be9bf2a3177d58fe3d0ba4007e648b5e6505e2676fc45cb535a421a5dd7d246b` |
| Skip last | `30171161e4e9a41ed837a97f6ffb8bc7b145620f24ae8324d7b7c4638ed944a2` |

## 20261125-HY4FRAG-06 · Reuse the prefill IndexPool scorer in long verification

The target verifier tried Strata's F32 prefill score kernel once it had
2,048 completed pools. It shares pooled key reads and changes the F32
reduction order. The NextN scorer and sparse selector stayed on their
existing paths. The image-loaded HY4FRAG-04 build admitted 34,816
positions with a 46 GB streamed arena. Each completed row used the
same rendered input and generated the requested 192 tokens at depth two.

| Input | Control tok/s | Prefill score kernel tok/s | Exact to control |
| --- | ---: | ---: | --- |
| 16K | 18.57 | 28.21 | Yes |
| 32K | 27.08 | 30.63 | No |

The control started with concurrent build activity. Its 16K host cuts
accounted for 4.66 seconds against 2.26 in the candidate. The timing arms
are exploratory. The 32K output and accepted draft count changed. Remove
this scorer from verification and test a batched score that preserves
the source reduction before drawing a performance conclusion.

The private verification patch SHA-256 is
`29eeca941679f3789276addcdd22e7a31aa18f99199820df1d98a427a80ee34e4`.

| Receipt | SHA-256 |
| --- | --- |
| Long control | `52af56c2257187e954ac3646fa98423e6ef344e9c36b161fd12f4f3f445d148f` |
| Long candidate | `16a847b35aab2543e715cfa494f416d703ae38ed172197a5510b495868ec3850` |

### 32K verifier attribution

A separate nominal-host sample instrumented the original target scorer.
It covered 390 target graph segments in 78 target rounds, matching the
32K control output. The 192-token request spent 7.169 seconds in decode,
including 2.201 seconds in target host cuts and 1.127 seconds in draft
work. The draft timing includes 0.434 seconds of history catch-up.
The encoder samples include only the target path. Sampling splits
encoders and changes the execution schedule.

| Target GPU stage | Seconds |
| --- | ---: |
| Routed expert products and combine | 1.170 |
| Attention projections | 1.040 |
| Dense FFN, shared and latent projections, and vocabulary head | 0.354 |
| MLA | 0.276 |
| IndexPool | 0.232 |
| Attention elementwise work | 0.187 |
| Remaining elementwise work and Engram | 0.144 |
| Router | 0.109 |

IndexPool optimization can address a portion of the gap. The largest
measured time budgets are streamed-expert stalls, routed expert products,
and attention projections. Profile receipt SHA-256
`1c0f172794d61f39f5298d1a807cab3f293704b41fc407c4f4a622513b46d433`.

## 20261125-HY4FRAG-07 · Use direct tensor views for the dense attention gate and output

The candidate expands each of these two packed Q8_0 weights and its
activation once per projection. It reuses Stark's Inkling device-view
matmul with F16 operands and F32 accumulation in K64 steps. The
page-rounded workspace reaches 224 MiB at 4,096 rows and is charged at
admission. The candidate applies to supported shapes from 512 prefill
rows. Decode uses the original projections.

The 40 GB, 8,192-position, image-loaded 4K depth-two bracket used grouped
expert fragments. Every result matched the 64 reference output IDs.
All arm-entry host snapshots were clear of named competing processes.

| Arm in run order | First prefill tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| Device views first | 507.44 | 539.02 |
| Packed control | 480.92 | 503.92 |
| Device views last | 516.24 | 515.91 |

First-use prefill improved 5.5–7.3% and warm prefill 2.4–7.0% against
the intervening control.

The candidate supplied by the Sol agent has patch SHA-256
`984796db355c980bddab208308f98d2ada6e4da369ba1e5bba1e93765af4e46a`.
The full patch on `57c102c86` is
`f7edbe8c492a38e8945693f25416e737fcacd37f5753bbd87ef1cebda7f9a062`.
Locked offline Rust 1.98.1 release binary SHA-256
`c6a9336f052add8a63389bc8242a35a43fef3d4ad865687915dcf2177cf81fc4`.

| Receipt | SHA-256 |
| --- | --- |
| Device views first | `e3d759ac1f3eb77891664d59ebc8709dec570f2d1a5830ccf76d5484178f8c59` |
| Packed control | `b03026c160d6aff2e84eaffe6337802ca99d29bc61b245d4ebfa5c71f23641e8` |
| Device views last | `2ff74b5cdf5e94a8f54918ed491e7962a1b89932ee6f50f85c56c41d44c42141` |

The same build agreed on all 13 teacher-forced rows at 552–840 input
positions, with eight actual next tokens correct in each arm. It passed
both original-projector image requests with zero draft proposals, and
the image, invalid image, and cancellation cache transitions. The
depth-four prefix reused 4,096 tokens and matched the fresh output.
Changing depth or context reused zero tokens and produced the same
continuation. Retain `STARK_HY4_PREFILL_DENSE_DEVICE_VIEWS=1` as an
opt-in alongside packed dense projections.

| Validation receipt | SHA-256 |
| --- | --- |
| Teacher-forced top-1 | `e2e04f9d7f66ef5cf4cb8418c6115ae7c245cbfceffd6fa3a4dd7cf3a3fc7409` |
| Images and cancellation | `dc4d73215fe55117cb714f869badb211ddef761467ff72ca68b8d8d270c5bd32` |
| Depth-four prefix and exact-fit context | `567ddb6f0f296cae8545e7fda4f4d21a14873a6cae8edf9ffc00ab932564fb40` |

## 20261125-HY4FRAG-08 · Screen larger arena admission at a 5,120-position cap

With device views and the private grouped IndexPool candidate disabled,
the HY4FRAG-07 worker refused a 52 GB arena under the live memory guard.
It admitted 49.998 GB at 5,120 positions. A depth-two request at
30.05 tok/s preceded three depth-four requests at 47.42, 49.35 and
49.48 tok/s. Every request matched all 64 reference IDs. The larger
arena did not reproduce the earlier 50.48 tok/s observation.

| Receipt | SHA-256 |
| --- | --- |
| 52 GB capacity refusal | `82fb5f940cd66b288a184c79373522924cbaf3ce52cc5524ac2be008e5071c20` |
| 50 GB depth-four repeats | `7e398fc7d5cd34572e5c0c34ef688e8a6e2a1a9bcc8f11ab9c261a8a830328f5` |

## 20261125-HY4FRAG-09 · Score eight IndexPool keys per threadgroup at source order

The second IndexPool candidate stages each query for eight pools, with
one SIMD group per key. The four lane products, shuffle reductions, and
ascending F32 head accumulation use the original order. It was tested
in both the target and NextN scorer from 2,048 completed pools. The
HY4FRAG-07 worker used a 48 GB arena, 34,816-position cap and K2 with
the dense device views off. Both arms matched all 192 output IDs and
both draft counters for each long input.

| Input | Control tok/s | Grouped tok/s | Host cuts, control / grouped s |
| --- | ---: | ---: | ---: |
| 16K | 31.67 | 28.40 | 1.64 / 2.27 |
| 32K | 29.47 | 30.57 | 1.66 / 1.55 |

Both arms began without a named competing process. A separate compile
job started during the grouped arm. After subtracting host cuts, the
grouped change moved by less than 3% in either direction. The data
support source-order parity on these inputs and give no basis to keep
the kernel. The control also completed the 75-token TypeScript prompt
at 34.76 tok/s over 192 outputs.

On the 48 GB control, subtracting all measured host cut time leaves an
optimistic 43.5 tok/s at 16K and 39.6 at 32K. The targets also require
reducing the cost of each verification round or increasing useful
outputs per round.

Private kernel SHA-256
`a0d8829545f9e050085774bc53617a67f7f97202027f6f53cb0417449a2c96a4`.
Harness SHA-256 `b96a8bea3dc0ff76142248f16ac4b18196c95589c012c9ba309cbfde83a6945a`.

| Receipt | SHA-256 |
| --- | --- |
| 48 GB long control | `6c6456602b170277c69df7f5f45abd237dd1b1a2bd45c2bbe3df39d4c712cb0f` |
| 48 GB grouped scores | `ef54c001fe7657abd050eee3323ad53f3d6f66a8868eddaf74b333c68f67fdca` |

## 20261125-HY4FRAG-10 · Compare Strata's existing streamed I/O schedules

On the 48 GB streamed worker with the original projector and an
8,192-position cap, enabled four inherited scheduling controls one at a
time. The pinned TypeScript fixture used 75 input positions and 192
outputs with depth two. All twelve responses shared one output hash and
accepted 116 of 149 proposals in 75 rounds. Host snapshots before and
after every arm showed no named competing process.

| Arm | First tok/s | Repeat tok/s | Repeat host cuts s | Repeat reads GB |
| --- | ---: | ---: | ---: | ---: |
| Control first | 32.88 | 37.73 | 1.218 | 45.98 |
| Submit read-ahead before waiting | 34.29 | 39.47 | 1.235 | 47.03 |
| Twelve reader threads | 32.34 | 37.21 | 1.302 | 48.90 |
| Two reads per tensor slice | 30.39 | 37.19 | 1.524 | 41.46 |
| Four predictions two ordinals ahead | 33.57 | 38.56 | 1.228 | 47.89 |
| Control last | 33.41 | 39.35 | 1.227 | 45.37 |

Moving read-ahead forward stayed inside the repeat control drift and
slightly increased host cuts. Splitting reads reduced traffic and
increased elapsed cuts. Retain the existing scheduler. The inherited
switches were available only on the private comparator.

Source patch SHA-256
`da081030db74cf165b11e0de0a811dde5c36464e1d3c8f9516c0354090460296`.
Rust 1.98.1 locked release worker SHA-256
`b3004cbad854fd5829320877826950257b34ca77a73ab62ac57214a6d7e1ae1a`.
Harness SHA-256 `d94784d0ee7727dec5abacc33645c502ed5b98d8b3502b5bafe947fd200a8af1`.

| Receipt | SHA-256 |
| --- | --- |
| Control first | `e14d34ae533bf1d2e322ffc40388ddb77ac3769958ab1063f50070d388cd4886` |
| Earlier submission | `157dc62b9b2a4b2ce4457decfb4a4be5933e54c1de7776cfc725291b8b7d7ca2` |
| More reader threads | `ba4b023293379f10c7f7db03305166aaae92d34331fbe9669a6616f4637c24da` |
| Split reads | `2902239a5cbc4a0d896e1f76e9ad4b79b34b5ffcbd4439c9dd4144c58a4f05f4` |
| Two ordinals ahead | `d37d327fa0169a11d75640b4ca6df4b8cba9699d68a01ebca21baf5a2bae15ed` |
| Control last | `50d379b60c650919eaeedfcd239d02472c82c80f5b65b35623695ec6d1f5ef99` |

The inherited contiguous expert sidecar is a separate option. The full
trunk bank would duplicate about 86 GB, above the 60 GiB free on the
measurement volume. No sidecar was produced or timed.

## 20261125-HY4FRAG-11 · Reuse one routed weight tile across 128 rows

The candidate keeps the packed expert weights, K64 staging and K16 F32
accumulation steps. It uses eight SIMD groups to share a weight tile
across up to 128 rows. The tile-list capacity, emission and dispatch
share the geometry across all routed trunk layers. Direct dense tensor
views were on in both arms.

The first bracket was interrupted by an unrelated compile. The opening
two loads completed with exact outputs. Their warm decode moved from
37.80 to 23.18 tok/s as the compile started, and host admission refused
the last two loads. The fresh bracket below had no named competing
process at entry or exit. All eight requests matched all 64 reference
IDs and the same draft work.

| Arm in run order | First prefill tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| 64 rows first | 648.01 | 582.42 |
| 128 rows first | 598.40 | 565.18 |
| 128 rows last | 594.49 | 556.88 |
| 64 rows last | 658.52 | 500.69 |

The wider tile was 8–10% slower on first use. Warm candidate rates stayed
within the control range. Separate stage samples measured 31,776 routed
tiles against 52,390, a 39% reduction. Routed GPU time increased from
2.024 to 2.459 seconds. Keep the 64-row fragments. Under this host window
the existing settings reached 658.52 tok/s first and 582.42 tok/s on a
repeat. The 64-row stage sample measured 1.597 seconds in attention
projections, 0.781 in MLA and 0.399 in attention elementwise work.

Stark base `edb1f3f2a`. The Sol agent's vendor patch SHA-256 is
`c824e59657ed237db8fd680f6e916b4d110b60fef00a23d52e8b8d5cc7d4d0f2`.
Full source patch SHA-256
`a8cd057fac90bb620d59bef8a8230e0b6a9a8aa20b64636545ab71819d455f14`.
Locked offline Rust 1.98.1 release worker SHA-256
`d7d843eb21fe5e4db8a4c4633a8c5d2613e4539fac79dbbef466c3428258364c`.
Clean harness SHA-256 `f18417ff90f4ca9e1f426895a8a80b4636b8447a1370cced2e4d4f9105ac6984`.

| Receipt | SHA-256 |
| --- | --- |
| Interrupted 128 rows, first | `75b52124c180c1dde0b22350da5453568f2fd3cd047e05158e3d5120d028a10f` |
| Interrupted 64 rows, first | `e60e370e04c792c694c16c546c3fbdc99379e3de80d0518424c9cf0a67571eca` |
| Interrupted 64 rows, refusal | `4c1be3f299a3d0eb5f44d52bcd1dc3db0e4f3e0f7263956d5448efc7e720df6d` |
| Interrupted 128 rows, refusal | `5e1e4a7534bcd323c623417ed07987854ee320ebe1d0030872659bbd394a85eb` |
| 64 rows first | `306dc4fb3999d7c9efaad92d11b73af80e51b0d3319a29853ebb06cf547ec813` |
| 128 rows first | `1e1b896a243c61b413e4cb25a6c5fdbd6aaf1aaee1eff72f21e9895a435ba6fe` |
| 128 rows last | `7ad4ce12214dc338bfe5a12eff0dbca36853a2e249adfaf5c695e08e68f33fd1` |
| 64 rows last | `7db36ac470bec5469fdd0c0f6188defb11afe183ff032e6f0a53261711c6ab29` |
| 64-row stage sample | `8fd57eda90228fa028317609def490f345f6e35442da802de548dd4987202454` |
| 128-row stage sample | `72d049880f4630613f4710c020521e85468dec369af133466e23984cbe738504` |

## 20261125-HY4FRAG-12 · Test deeper verification and replay the prompt-lookup gate

The HY4FRAG-11 control path loaded 50.996 GB at a 5,120-position cap.
It ran depth two, four, four, then two on each input with zero prefix
reuse. All generated IDs agreed within each fixture. The two short
coding prompts retained their best throughput at two drafts.

| Input | Depth-two tok/s | Depth-four tok/s | Target rounds at two / four |
| --- | ---: | ---: | ---: |
| Fixed 4K | 29.66–40.35 | 46.88–48.84 | 22 / 13 |
| Rendered 4K coding | 30.54–34.30 | 31.66–31.82 | 82 / 64 |
| Short TypeScript | 39.12–40.79 | 37.90–37.91 | 75 / 58 |

Depth four at the 48 GB, 34,816-position configuration matched the depth-two
output on both long inputs. The depth-four run had fair process thermal
state. This table compares separate loads rather than a single timing bracket.

| Input | Depth-two tok/s | Depth-four tok/s | Target rounds at two / four |
| --- | ---: | ---: | ---: |
| 16K | 31.67 | 26.83 | 79 / 62 |
| 32K | 29.47 | 28.05 | 78 / 60 |

An in-memory replay used Stark's existing 4-gram lookup and backward
agreement rules. The fixed repeat-text input was eligible at 62 of 63
positions. Its 413 overlapping offers all matched. A lookup-only policy
with at most seven proposals per round could cover the continuation in
eight target rounds. This is an offline estimate. The 75-token and 4K
rendered coding fixtures offered zero lookup drafts. Keep those coding
workloads separate from the repeat-text test.

| Receipt | SHA-256 |
| --- | --- |
| 51 GB depth and replay screen | `f64f7506a1994ac3941cdb185b75d91e686168e979f05d77daac6cbd4f86ff4d` |
| 48 GB depth-four long ladder | `7dbfc2453b196b708db1b51f8fbebcb25dbd606241233d30b4f58c620e3718ec` |

Replay helper SHA-256
`04410bdf5613f9ad03ab66b38d21c275f8f5d66afcadcfb4d7d2ee7d75cb62df`.

## 20261125-HY4FRAG-13 · Verify request-local prompt lookup with the native target

Hy4 reuses Stark's request-scoped 4-gram index. Lookup hits skip NextN
proposals and use the same greedy target comparison. The target records
committed hidden rows. A later lookup miss advances the draft cache
from that history. The worker keeps the lookup index in memory for one
request. Depth and lookup mode bind the generation prefix key.

Set `STARK_HY4_MTP_LOOKUP=1` and `STARK_HY4_MTP_MAX_DRAFTS=7` at load
to request greedy `mtp_drafts: 7` with `mtp_lookup: true`. The loaded
worker recommends two drafts. The normal depth-four ceiling applies
when the lookup worker flag is off.

The original-projector worker used a 48 GB arena and an 8,192-position
cap. In the same loaded worker, each input ran K2, K4, K7, twice K7
with lookup, then K4. All output IDs matched across modes for each
input. The 4K fixed fixture also matched all 64 reference IDs. The
code-edit case is a 1,898-position full-source JavaScript edit of
`554a6a249:website/public/app.js`. Its 192-token generated prefix
contained the requested clipboard message in every arm.

| Input | K2 tok/s | K4 tok/s | K7 tok/s | K7 lookup tok/s | Lookup proposals / accepted / rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed repeated-text 4K | 27.84 | 45.58–46.87 | 48.67 | 61.18 / 60.61 | 55 / 55 / 8 |
| Copy-heavy code edit | 32.31 | 40.65–41.45 | 43.26 | 44.96 / 43.75 | 120 / 110 / 18 |

The 4K lookup repeats exceeded 50.34 tok/s. Use the result for this
repeated-text diagnostic. The ordinary coding fixtures had zero lookup
eligibility in HY4FRAG-12. Host snapshots found no named contenders.
Process thermal state moved from fair at entry to nominal at exit.

The first attempt was refused for concurrent build work. An exploratory
restart on the first binary completed ordinary decode and refused the
lookup request at option validation. The corrected binary recognizes
lookup support independently of adaptive depth. An intermediate run
under background build activity completed all arms with exact outputs.
It measured 52.73 and 47.87 tok/s on the 4K lookup repeats and 32.50
and 40.55 on the code edit.

Sol agent patch SHA-256
`bafcd91e8d538d4460cc0771ecb4b2de766e760a48335a1b09998cb48915ffa6`.
Full measured source patch on `554a6a249` SHA-256
`6bfe3561951c0d841ca04a5875e8620c6428b1f0f8a9308d0bf9cecadc0db3a8`.
Rust 1.98.1 locked release worker SHA-256
`3ba43f4498ed49baaeae723f04d5dcc959ce01d9daa8eee712bfa6bea15909cf`.
Lookup harness SHA-256 `b24af044817faa0b3ede7a046ff16b88f0d40450426b18efa3fe9f97e290ae06`.
Code edit fixture helper SHA-256
`1f81c5855634363fa958063fde5f0c8a3805f19f54557220fa507c3a23cb2944`.

| Receipt | SHA-256 |
| --- | --- |
| Host refusal | `1451298298ce41b5535159e30737dcd047086fd588e9cfe4c3bb193f4ed76ad5` |
| Initial lookup validation refusal | `33119c2cb777d27024cf68218fddb2a89bba3582eaa932777e5df5cb945cb39d` |
| Contended exact-output comparison | `a80179866f01bfa91b54f970e2892e62ab9b412976bef88cf7e464a78908ee82` |
| Lookup comparison without named contenders | `f0a2d27ef6b9dd1317ce6b512a8511c6d9299056d8405cdabacda5ae2984d86c` |

The same candidate restored a 4,096-token prefix at depth seven and
matched a fresh continuation exactly. A change of lookup mode, depth,
or context used zero cached tokens and produced the same continuation.
The exact-fit request exercised the context boundary. Both original
projector cases generated the expected outputs with zero proposals.
Image generation, invalid-image rejection, and image cancellation cleared
the text prefix. Retain the request-level lookup flag. The lookup-only
source patch SHA-256 is
`54db6769996940dfa6ff643376648284bc8c186b2cd823e92f0d4e99fde603d7`.

| Validation receipt | SHA-256 |
| --- | --- |
| Images and text prefix transitions | `52b666c983910868021deaabf5effb6a01927fce7ab062fef79f252e0be89277` |
| Same-mode reuse, context boundary and mode changes | `d832faedd3a0bcb8dace33d405ebb1b942d178c2aea67ce63c1556da8b25104e` |

The Rust 1.98.1 locked release worker rebuilt from `f90f3ec63` has
SHA-256 `5cfcde51f02fb14275fd5d391266239b27cad4b94c9b82b2d3198eff2bb69325`.
The original projector was loaded for both admitted runs. Both lookup
repeats accepted 55 of 55 proposals in eight rounds at each setting.
All completed arms matched the 64 reference output IDs.

| Arena and context cap | K2 decode tok/s | K7 lookup decode tok/s | K2 first prefill tok/s | Receipt SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Explicit 48 GB and 8,192 | Admission refused | | | `4e97ec716d3041a9c480626a5c9a28d133d8411e3c9e3abf3e3babb0d0a56d9b` |
| Auto 27.997 GB and 8,192 | 24.80 | 47.17 / 46.31 | 536.32 | `1edda0833395c77b558bc77a8fb7fadbb3b2536bcb285dc07a45a9e13b057454` |
| Explicit 39.997 GB and 5,120 | 28.99 | 65.58 / 63.40 | 640.48 | `58ebb251b181747c2491270aeccb784b65b1844f07aac5ec9553e9ea68fd9f99` |

The auto run reported fair process thermal state throughout. The 40 GB
run reported nominal state throughout. Neither snapshot pair found a
named contender. Context cap and thermal conditions varied between the
two runs, so the rates qualify only the respective measured settings.

## 20261125-HY4FRAG-14 · Fuse the dense attention gate into the device-view epilogue

The candidate keeps the original precise sigmoid and F32 multiply, with
one F16 cast before the output product. Its second ingress packs only
weights and reuses the existing gate scratch. The 4K logical traffic
estimate fell by 38.65 GB across all gate outputs with no extra scratch.

The first bracket was blocked by a concurrent compile. An exploratory
bracket while the unrelated build process remained present had mixed
first-use timing and favored fusion on repeats. A separate bracket with
no named contender began at nominal thermal state and ended at fair.
It matched all 64 reference IDs in all eight requests.

| Arm | First prefill tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| Control first | 649.90 | 461.51 |
| Fused first | 654.68 | 443.35 |
| Fused last | 654.12 | 465.50 |
| Control last | 655.39 | 473.68 |

First use was within control drift. The repeats established no gain.
Remove the fusion. Agent patch SHA-256
`5f3b12c6bc53b29798bf2783349fa6ebabacaf2550b0807adb923386424a9469`.
The initial bracket used binary SHA-256
`affff6b217653194a606068749be2a23b34c04c8cc67ddbcfb302a4da108d90c`
and source patch `d2d58b547560d21310158c9b254c261e183e3df9cb1cbb1e0a675ac81f3b364b`.
The later bracket used the HY4FRAG-13 binary and harness SHA-256
`312f497936818ccde27342ca59ccf2074093e5d3a63bf28838f826d8676ca1e7`.

| Receipt | SHA-256 |
| --- | --- |
| First two host refusals | `529f91a492bbae61a77ff7f168f7aa2f6afe55b4ee64b5c5d3dff4e81b1d0fe5` |
| Last two host refusals | `881fe50398bc9be13b97c905ddcfac9210e56b72ba8002fd1a7022cd1a3cddc9` |
| Exploratory control first | `c6649381e18d20ddf154cb75b18a5b5bc0677db58e2ca058ccb945d43399cc2d` |
| Exploratory fused first | `cbf5dbbafb9e62c87fec5545e88456680b764c66da4066f9795feb9e9b7a2d94` |
| Exploratory fused last | `1ed0f390cff4c28cb2de919a320448a1eec217fe9395a21ba2ef89c1ac16e45c` |
| Exploratory control last | `6405b54c9f49d61ee425b92bc9fe08a28241c3d99143d106b9e76e6adfceba92` |
| Control first | `cfa3283d71866ff49fc92ccc2734842de2fe2d51dc8d73aac461642d6f4b04a9` |
| Fused first | `fe4f1c84d205e32e1f21b64ccc898934b8bbba68fce493e34a4becc16894de3f` |
| Fused last | `8bdb60f6347625afc3038e8e7d4bbcb7003cbd29f673dd1f2ef1147864d92bcb` |
| Control last | `2fef753e73efb57241597a66f7fdb86e4dc41dbe11744f469b85a93eeabd828e` |

## 20261125-HY4FRAG-15 · Screen the AngelSlim DFly companion on the pinned Q3_K target

The [publisher Metal guide](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/blob/main/README_MAC-Metal.md#72-dfly-this-patch)
reports 51.95 tok/s with Q4_K_M on an approximately 81K agent request.
For this screen, the existing patched llama.cpp worker loaded the
AngelSlim Q3_K target and used the pinned 32K rendered coding input,
34,816-position cap, F16 cache, 512 batch and 256 microbatch. Both
arms were text-only and generated 192 tokens. Each used its default
tensor placement. DFly applies a different embedding and output
placement by default. The host started and ended at fair thermal
state with no named contenders in the snapshots. Separate build
activity was observed while the second arm was running.

| Draft mode | Prefill s | Decode tok/s | Proposed / accepted | Target draft rounds |
| --- | ---: | ---: | ---: | ---: |
| Author NextN, K2 | 95.826 | 44.50 | 149 / 115 | 75 |
| DFly Q8_0 companion, K4 | 93.785 | 39.30 | 255 / 127 | 64 |

The two outputs differed. DFly used fewer target rounds and took longer
for the same output count. This exploratory result does not establish a
speedup or parity on the Q3_K workload. Keep it a reference experiment.
Match tensor placement and verify outputs before using the publisher's
Q4_K_M speed to justify a native Q3_K port.

The [companion artifact](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/tree/c50d4dd30be42f2b91138e5731de0e2f32ad56b2)
is 903,685,664 bytes at publisher revision
`c50d4dd30be42f2b91138e5731de0e2f32ad56b2`, SHA-256
`609464e933b332657ca84be9ee8867a8940c388844514f7d0673b2e77a1c7e00`.
Reference worker SHA-256
`b9aad7086abc83bb351abdfa3a41e3f8fde73529fbe8c4a9006d2ee23de0b9bf`.
Harness SHA-256 `cc649d3a46a6c6c9f93580e0c1a19a87522cc0c4acfee86e8e6b26e74b1e2afb`.
Receipt SHA-256 `880d1ba076c73ba8543b62183f1f8a5094ab6a94c29706396fd507bd3caff147`.

The reference source uses the final trunk hidden state after the output
normalization from the previous committed position and five indexed
target caches. The native NextN hidden tap and Hy4 target verifier are
reusable. A later implementation needs the companion's five-layer
graph and selected-cell snapshots at the accepted target position.
Inkling provides matched replay and verification measurements.

## 20261125-HY4FRAG-16 · Reuse tensor attention with the exact sparse selection mask

The existing 16-query Metal tensor kernel now has an indexed prefill
specialization for absolute positions 2,048–4,095. A kernel builds a
bitset from the model's `sel/nsel` output. The tensor product still
checks the original causal upper bound and sink. It uses F16 queries
and probabilities with F32 accumulation. Mask packing borrows up to
1 MiB from existing score scratch. Other positions and the diagnostic
mutation and injection paths use their existing kernels.

The first timing attempts were refused for named competing processes.
Two later runs at explicit 40 and 28 GB arena sizes were refused at
live memory admission. A preceding known worker also refused the
explicit 28 GB request. The automatic arena admitted 27.997 GB and
8,192 positions for both arms. All completed 4K requests matched all
64 reference output IDs.

The initial stage samples ran with an unrelated `cargo` process present.
The mask reduced attributed MLA time from 0.935 to 0.523 seconds. The
mask arm also had 1.119 seconds in prefill encoding on its first run,
against 0.008 for the control. The first-use wall result was slower
despite a shorter GPU span. The subsequent stage-off contended bracket
was inconclusive. Unrelated activity ended during the closing control.

| Stage-off arm under contention | First prefill tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| Control first | 607.73 | 549.22 |
| Mask first | 591.73 | 535.64 |
| Mask last | 554.31 | 540.01 |
| Control last | 569.75 | 462.18 |

Two more pairs reversed launch order with no named competing processes
at entry or exit. The first pair moved from nominal to fair during the
opening candidate. Both arms of the reversal stayed fair.

| Pair and order | First prefill tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| First pair · mask | 615.57 | 540.95 |
| First pair · control | 531.24 | 446.49 |
| Reversal · control | 494.57 | 486.30 |
| Reversal · mask | 545.62 | 516.15 |

The fair-state reversal improved 10.3% on first use and 6.1% warm. The
candidate's strongest request in these pairs used the automatic arena
and reached 615.57 tok/s. The earlier 658.52 tok/s first-use peak in
HY4FRAG-11 used a different memory and host window.

The Sol agent's vendor patch SHA-256 is
`12b448f325a95334d26b601fc2dfe2a03a1b20ecbb9aafab9f8a603ea6a380f5`.
Full measured source patch on `dbba655d8` SHA-256
`bfe5aa15a43c90f0f075b923aa5209a3ae69bfc23f30f49a148617b7147988f2`.
The masked-only patch on `f90f3ec63` is
`db0eb1cdd3c09a3d6889c1e1d8e5e3426bf36c6cb46803d7df277df0a704e638`.
Locked offline Rust 1.98.1 release worker SHA-256
`5bf3a366edfc2861cae8fb0acfe59029782c40258cdf68e9ca0fd89d59d1c0f6`.

| Receipt | SHA-256 |
| --- | --- |
| Opening host refusals | `2a5616118e9358c20dca1e9eae5e60dbd1963eef82cf550d0078ea0587a08a5c` |
| Returning mask host refusal | `294813218f13617428822fe151b4ba0988bf7f46d9a168f03cac3673461e527c` |
| Closing control host refusal | `48854301e1f8f1be084363a564f90e29efd09cad54fa67d4a77af07a1138bb9b` |
| Stage control host refusal | `cc0a5af6c0d14940f5ff0f4a388a0c7be4bf453e0d023665b15487370efbb7ba` |
| Mask stage host refusal | `09ea3f543012883b94b955c97d7814a9190d4e1fc0fc3950274399b1b3269430` |
| Explicit 40 GB control refusal | `e61961fefd8558b6a7ee17e4859ea04bb2915534c98875422a7e3129c325a95c` |
| Explicit 40 GB mask refusal | `d0ea4d78ed5581fd1c8b4f909480ca970f1f937dc4ce5d1be862565fd7f25f6f` |
| Explicit 40 GB stage refusals | `9c7b04e771645a6399ac61cc1899489512e5150b1433fa6d926ab7577bea075c`, `17ce0c06b4bc971bc55c5795a9d674bc8b651399f40b16a17abf03a51a12fedc` |
| Explicit 28 GB control and mask refusals | `dfa6b7fef2f5b07745f66875ad7a85fedcada9a081c31a6d057fb2e77114e661`, `bfba6315edb1726abbe53c2f18877c82c2da2388273726c04f35ad386d601540` |
| Explicit 28 GB stage refusals | `855abf09ac0ea64764a47d46028246ff18c37aa97afd3ab5922d6767c0974f8a`, `e827817d172e4c7a5d390746d344387b1b5db33418d0d55d0a388639cd64b00d` |
| Previous worker's explicit 28 GB refusal | `6d697188622e223ce03c221675a9ebe8c701e310fd52795afe1469e08267ff10` |
| Auto control stage sample | `a6a0a9c73456b108a99fc3b299e832ec6d3d9dd7f71cf00d64d9ced478b2175d` |
| Auto mask stage sample | `4d55f6e3ac3bf4f9d470d539cd47d17f92260ba19cb2c79225a56e19eb92c647` |
| Contended control first | `5c5cf26844a3dfe8e2c7406b387a2f28154b0d7097bb6cf9069aa05deb4a25f8` |
| Contended mask first | `0de044d0b189b69af443c8d9878b6c68f1b13f285bc12fdc846eac277b78ad2b` |
| Contended mask last | `b75f6380763cdd8d0c56450c3bd170f518868ff51363855f22299bd98e6b1d08` |
| Contended control last | `d917b9f05374b7b19a0df68180f866e02ff01eb8c57b4b0c18605b03be41b246` |
| First pair mask | `b67378144485cacd550d847e85b4f79e0c48496a0a5c4dcda9b4562d95fa82b6` |
| First pair control | `ee5195073633b1744b726c9ced4074a25c366b8c325053303e7670eb2a255e19` |
| Reversal control | `ecc1b9f5336ddba04371b9662bff076615d53510bf25ce46831660d424b9bb84` |
| Reversal mask | `2b7ca65183a74422920492b1a0c82fc42aeffbe0954dab67f12552a3d834a3d8` |

The candidate matched 12 of 13 teacher-forced natural-text top-1 results
at 2,120–2,408 input positions. Each arm predicted the actual next token
on six rows. This misses the exact top-1 gate for the masked suffix.
Both images, text prefix hits, and cancellation still passed. A separate
follow-up admitted a 40 GB arena and 5,120 positions, reaching 618.95
first and 532.43 warm with exact 4K reference IDs. Leave the candidate
out of the branch. The source-order sparse suffix remains active.

| Validation or follow-up | SHA-256 |
| --- | --- |
| Teacher-forced top-1 at masked positions | `3547b13269c5fc2a1fb75b8227109d4984e9f9964c1efd5996487003e46a2bc7` |
| Images, prefix hits and cancellation | `a1ae8c7de11815e804f248b78059621dcbcccea9b6272d3875d3fcf17fbfc439` |
| 40 GB follow-up | `e44d9ff80777cc9008a12dee3c13f3e9c8bd522fbd2823c3d569005ea31b5401` |

## 20261125-HY4FRAG-17 · Store the routed SwiGLU intermediate in F16

The packed expert down projection already rounds its input to F16 on
load. This candidate rounds at the gate/up store, following Inkling's
grouped activation layout. It keeps F32 SwiGLU and accumulation, route
scaling and combine order. The F32-sized allocation stays in place for
this screen. The candidate halves the logical intermediate size.

The original-projector worker used a 39,996,981,248-byte arena at 5,120
positions and K2. Both measured sessions reported nominal process
thermal state at entry and exit and zero named contenders. All four
requests matched the 64 reference output IDs with 41 accepted proposals
out of 42 and 22 target rounds.

| Arm and order | First prefill tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| F32 store and F16 load conversion | 620.64 | 576.64 |
| F16 store and direct F16 load | 627.44 | 584.16 |

The gain was 1.1% first and 1.3% warm in one pair. The first-use gain
was 71.6 ms. This is below observed session drift and the remaining
prefill budget. Leave the original path in place.

The screening patch on `2a172b77a` has SHA-256
`20eb3f1db8f26f390f19fd6678bf1217dd5d19d0c004f6836e6dbd5ad26c20c0`.
The Sol agent's half-intermediate patch SHA-256 is
`5f477c9e8ad2bea81e3d6af832ac6da46227b17e90ed8a662866e79c8dfd765d`.
The offline, locked Rust 1.98.1 release worker has SHA-256
`47e56a432be8f8fd54e1dfb349bbb073249004b1688c24d5990304709d56b525`.
The harness and shared request helper have SHA-256
`3a4d7964153e336de8b8bff5160676ed6af5267a0b16bd0b83b828bedd2f7c56`
and `f58ec93d44678da9c6e84f8a2daf608fb157a8916d406455a529ebbcaa81d736`.

| Receipt | SHA-256 |
| --- | --- |
| Control | `e6770923cb10520c9aea5748153ed57ea064d79a8709586c089b2632ba27d4a8` |
| F16 intermediate | `079524266e42ec09d00c08d479ddb745ca879e50f028a297bf775c9fb952a418` |

## 20261125-HY4FRAG-18 · Screen long verifier scheduling and packed projections

This screen used the HY4FRAG-17 worker and harness, with the original
projector, a 43,997,151,232-byte arena, a 34,816-position cap and K2.
Each 32,768-position request came from a fresh worker and allowed 192
output IDs. The variants independently enabled Strata's event listener
after a 50-microsecond spin, disabled Strata's predicted-router
read-ahead, or sent the Q8_0 attention gate and output through the
existing packed tensor kernel at two 32-lane SIMD groups. The last arm
uses F16 operands and F32 accumulation only in widths two through
eight. Single-row forwards keep the existing projection.

| Arm and order | Decode tok/s | Verify s | Target host cuts s | Recorded decode reads GB |
| --- | ---: | ---: | ---: | ---: |
| Control | 27.90 | 5.709 | 1.984 | 83.89 |
| Event listener | 27.50 | 5.786 | 1.941 | 84.61 |
| No read-ahead | 26.42 | 6.062 | 2.446 | 43.71 |
| Packed gate and output | 21.75 | 7.623 | 1.831 | 88.77 |

The event and no-read-ahead arms matched the control's 192 output IDs,
with 113 accepted of 154 proposals in 78 target rounds. The control
stayed nominal, the event run went from nominal to fair, and the
no-read-ahead run stayed fair. The host snapshots reported no named
contenders. The event run showed no throughput gain. Disabling
read-ahead cut the recorded reads by 47.9%, with 0.462 s more host-cut
time in this one-way comparison. Keep the normal scheduling.

The packed verifier changed the output hash, accepted 114 of 152
proposals and used 77 rounds. Its lower speed is directional because its
output changed. It stayed in fair process thermal state with no named
contenders. Reject it on parity alone. The Sol agent's packed verifier
patch SHA-256 is
`00b2a0b1079c3e179d72cf155c688af57647111b93c82bfff759adbeba098549`.

An in-memory seven-token lookup replay on the 32K control offered zero
proposals across 191 output positions. This rules out a lookup speedup
on this fixture at the existing four-token match rule. The separate 16K
replay was refused first by the active benchmark lock and then by the
host eligibility check. The earlier 48 GB
control used this same input and output and was faster at 29.47 tok/s.
Its 6.482 s decode would need to fall to 4.032 s to reach 47.37 tok/s.
Even removing its full observed 1.658 s host-cut body leaves 0.792 s to
recover. Use per-layer cut and demand-read time to choose a bounded
resident bank before attempting full mapped residency again. Price
event boundaries independently where the GPU can build the expert
plan without a host cut. The earlier mapped failure produced no
qualified timing.

| Receipt | SHA-256 |
| --- | --- |
| 32K control | `d04b3218f542a2bc17d41e2913a7c0289f6ff128c46d40e46399c5ca1b8fb89b` |
| 32K event listener | `c5081e5cc30365ebe2710d8fb4284f997cfe62ee5c97ceb86227d83defbf54ba` |
| 32K no read-ahead | `a2087d407a19540a0e83624497a98ea262c8243cd7114f5172c184184e3958ef` |
| 32K packed verifier | `661ef5a86e8f851b71b5eec04a8eed91578247eac75efaefca5e2fe16c6c730c` |
| 16K lock refusal | `acc99af5cf66c67057f3abdb47da077b5c9ca66cd0165162fc1757d4f991d7af` |
| 16K host eligibility refusal | `870791e768607a58eff53897bebb905926ff76ffdf6ad7d0fd71cbd6360bcff2` |

## 20261125-HY4FRAG-19 · Attribute streamed verifier cuts by expert bank

A private scalar profiler attributes the first and second host cuts to
the routed expert ordinal. It uses demand-miss bytes and wait counters
owned by the host execution thread. It records scalar decode stages.

The image-loaded worker used K2, a 44 GB arena and a 34,816-position
capacity. It ran the rendered 32K and then 16K code cases in a single
worker with zero prefix reuse. Each result matched all 192 output IDs
from the earlier 48 GB control. Both fixtures returned zero proposals
in the in-memory replay of four-token lookup at maximum depth seven.
The session went from nominal to fair with zero named contenders in
both snapshots. Treat these instrumented rates as diagnostic.

| Context | Target cuts | Cut body ms | Demand wait ms | Read-ahead wait ms | Demand miss GB | Top three banks, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 6,355 | 2,588.5 | 2,287.4 | 73.7 | 24.41 | 349.2 |
| 16K | 6,437 | 2,256.1 | 1,914.0 | 64.5 | 19.28 | 247.1 |

Ordinal 1 led both workloads with 136.3 ms at 32K and 84.8 ms at 16K.
Most cut time lay across many banks. Use ordinal 1 for a bounded bank
trial only if full residency fails live admission. The top three banks
combined account for 13.5% of the observed 32K cut time before any
capacity tradeoff. Addressing only these banks would leave most of the
gap to the target.

The Sol agent's profiler patch has SHA-256
`43361fc8dd7c7a900164e6ec6244e791e70dc89b88a153e267f659d2585d4b7c`.
The combined patch on `e6672affb` has SHA-256
`af13fcee8bfe924a938fe2cec553dc774ee573187dd6a3b8bd56c4c29c8a19f0`.
The locked Rust 1.98.1 worker and harness have SHA-256
`6c8750cdb8e3e29c7b94e2a43d682aef0f8cc6f69e68122e357e4144c359362a`
and `d5e4a3a505fb4f854328529a7a98956344025b26f06efe8a8849a1cda5924290`.

| Receipt | SHA-256 |
| --- | --- |
| Earlier 16K replay host refusal | `c857a90076b7ed6bfd186fcbacdb84daa3fe860f8c7efc4cd3251fa7df002d81` |
| Initial profile host refusal | `ce1f61f45f1e9c35d2e44dac2e3fdcdef37a71ebf5a94fec2d01aad92f328e64` |
| Both completed profiles and lookup replays | `53c53c10368f7976541a3fd6ad8d8cfc2f3d97f4cf3bb74033487d151967477e` |

## 20261125-HY4FRAG-20 · Prepare the mapped expert banks before generation

The screening worker leaves quantized weights in their clean GGUF
mappings. At load it faults in weight pages in 128 MiB intervals, checks
pressure and swap between intervals, then puts only the expert buffers
in a Metal residency set. Engram rows load lazily. The change uses the
existing mapped-load and context admission limits and host settings.
One owned worker ran at a time and a private
watchdog checked pressure and swap through generation.

The first 4K attempt was refused for background cargo activity. The
retry allowed the sleeping cargo process at entry and found no named
contender at exit. Both process thermal snapshots were nominal. The
original-projector worker loaded in 61.29 s at a 5,120-position cap.
The host pressure sample stayed at level 1 and swap stayed at 4,191
MiB. Both requests matched the 64 reference IDs and accepted 41 of 42
proposals in 22 target rounds.

| Request | Prefill tok/s | K2 decode tok/s | Target host cuts |
| --- | ---: | ---: | ---: |
| First 4K | 611.35 | 50.28 | 0 |
| Warm 4K | 566.14 | 54.41 | 0 |

The 4K result is a provisional observation with the HY4FRAG-19 worker
and harness. Readiness includes prewarming. Prior mapped trials used
different initialization, and one failed with sustained swapping.
Keep the established streamed default while checking long-window
speed, pressure, cache restoration and image lifecycle.

The separate 34,816-position ladder loaded in 69.66 s. All three
192-output requests matched the earlier 48 GB streamed output hashes
and reported zero target host-cut time. The process snapshots both
reported fair thermal state and zero named contenders. Observed
pressure stayed at level 1 and swap at 4,191 MiB. The 48 GB streamed
rows are earlier sessions with different memory plans.

| Case | Earlier streamed K2 tok/s | Prepared mapped K2 tok/s | Mapped verify s | Mapped draft s |
| --- | ---: | ---: | ---: | ---: |
| Short TypeScript | 34.76 | 49.41 | 3.133 | 0.732 |
| 16K rendered code | 31.67 | 42.37 | 3.658 | 0.848 |
| 32K rendered code | 29.47 | 39.72 | 3.748 | 1.059 |

The 32K request spent 0.441 s of its draft time constructing the NextN
state. It needs another 0.777 s of total decode savings to reach 47.37
tok/s. The typed worker selector is `mapped-resident`. Lifecycle checks
and a build receipt for that selector follow separately.

| Receipt | SHA-256 |
| --- | --- |
| Initial host refusal | `6fce1feb6de4c65b8807c57be3c7607be2a16a84fb9785e29fb711bc4a8cb5df` |
| 4K provisional mapped pair | `5867730a311032bbbeaf34dd4f30f7f35b686c2fa42f2b62ffc3a6e895dc19a6` |
| Mapped 34,816-position ladder | `9b348d71260489cf4d1a931c9cd37c5a288d4d2ec9927a5b68820a213ea6ac22` |

## 20261125-HY4FRAG-21 · Keep NextN experts resident

The optional draft mode loads the verified Q8_0 sidecar as one shared
Metal buffer, keeping the three tensor banks and the source bytes. It
charges about 1.85 GB beyond the lazy pool before selecting the window.
The existing GPU planner sorts routed entries by expert and selection
slot. The draft specialization preserves the CPU plan's weight bits.
Draft GEMV and combine use their existing arithmetic and slot order.

The prepared mapped worker advertised 34,816 positions with the
original projector and took 85.45 s to report readiness. It ran K2, K3,
K4 and K2 again on the short TypeScript fixture, then K2 once at each
long length. Every arm matched the earlier 192 output IDs. The 16K and
32K runs matched the earlier draft counts as well. The host snapshots
found zero named contenders and nominal thermal state at exit.
Pressure stayed at level 1 and swap at 4,191 MiB.

| Case | Drafts | Decode tok/s | Verify s | Draft s | Accepted / proposed / rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short TypeScript, first | 2 | 57.05 | 2.906 | 0.441 | 116 / 149 / 75 |
| Short TypeScript | 3 | 60.65 | 2.847 | 0.301 | 127 / 188 / 64 |
| Short TypeScript | 4 | 54.03 | 3.166 | 0.368 | 133 / 228 / 58 |
| Short TypeScript, repeat | 2 | 58.91 | 2.999 | 0.242 | 116 / 149 / 75 |
| 16K rendered code | 2 | 44.10 | 3.785 | 0.544 | 112 / 156 / 79 |
| 32K rendered code | 2 | 39.19 | 4.093 | 0.779 | 113 / 154 / 78 |

Against the separate prepared-target run in HY4FRAG-20, draft time
decreased by 0.303 s at 16K and 0.280 s at 32K. Target verify time
varied across loads. At 32K the end-to-end rate stayed within drift
despite the shorter draft phase. Keep this option for the coding
improvement and constant output. Keep the two-draft recommendation
pending wider, repeated evidence. The published 70 tok/s coding and
long-window targets remain open on these local fixtures.

Sol patch SHA-256 on `e6672affb`
`6b88fa916596b78f9ca837796ae15a416b9fc4ffef6640e419a4649908139909`.
Combined screening patch SHA-256
`b780b221168b445c9936bae8e549e9db2e92acfeca4cf0cc698466aa0aa0fbd9`.
The locked Rust 1.98.1 executable and harness SHA-256 are
`143deca1ec89a66dcf19794ee8fae151094028994ab93a0ffacad7ea90124c9a`
and `1b9befaa98da54968a5fe9ae74a01258ccb5f8ba7b834ef0a679d9ff338da0a9`.
Receipt SHA-256 `e095c533dbdd5601e7fa5d438e1e7523680a71585519f57a79e292f4f622c7a7`.

The typed selectors were then rebuilt without the experimental kernels.
The 8,192-position worker used `STARK_HY4_EXPERT_STORAGE=mapped-resident`
and `STARK_HY4_MTP_RESIDENT_EXPERTS=1` with the original projector.
Readiness took 75.60 s. The two fixed 4K K2 requests measured 54.72
and 54.06 decode tok/s, with 622.25 and 603.47 prefill tok/s. Both
matched all 64 reference IDs. Both host snapshots reported nominal
thermal state and zero named contenders. Pressure stayed at level 1
and swap at 4,175 MiB.

The same worker restored 4,096 text positions at K4 and reproduced the
fresh continuation. Both pinned images produced the expected answers
with zero proposals and zero reuse. Image completion, invalid-image
rejection and image cancellation each cleared the preceding prefix.
Subsequent text decoding reproduced the fresh continuation and restored
4,096 positions after reprime. Adopt both selectors as opt-in modes.

The typed source patch on `e6672affb` has SHA-256
`09445dd45201f737c7125f17450301fcf99ebf38f3d83e795f92dcf8a320b075`.
The locked Rust 1.98.1 executable, lifecycle harness and receipt have
SHA-256 `732909369aa360df0656cb05877e64e45ae6a7a513c15a225ed6e0d893dce165`,
`5a34b4e16030b5ae0b56681ca0226824d98fd2e10eef5995e86ee956015e8717`
and `619c9a4c572afa9b8876927049cce0088698af116a7b0f82a8c9695c8910c7bd`.

## 20261125-HY4FRAG-22 · Isolate sparse MLA value precision and expand direct projections

The value-product candidate leaves the sparse selector, compact gather
order, F32 query-key dot and F32 softmax in place. It uses F16
probability and value operands with F32 accumulation for the value
product at positions 2,048–4,095. The other candidate reuses the
existing packed Q8_0 device-view matrix multiply for the query A,
query B and KV A projections where the shape fits the allocated
workspace.

The HY4FRAG-21 screening executable ran separate image-loaded workers
at a 5,120-position cap and a 39,996,981,248-byte streamed arena. All
six requests matched the 64 reference IDs with K2. The control and
tensor value-product snapshots stayed nominal. The direct projection
arm moved from nominal to fair. The snapshots found zero contenders.

| Arm and order | First prefill tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| Control | 584.33 | 546.41 |
| Tensor value product | 612.58 | 601.37 |
| Direct query and KV views | 517.16 | 526.57 |

The value-product arm improved prefill by 4.8% first and 10.1% warm
in this one-way screen. The direct projections showed no gain and
were removed. Two natural-text invocations requested 8,192 positions
from a 5,120-position worker. The corrected fixture helper submitted
the admitted capacity throughout. Its thirteen test rows exercised
positions 2,120–2,408. The candidate matched 12 of 13 control top-1
results and predicted the true next token on five rows against six for
the control. Keep the existing F32 value-product path. Holding
query-key and softmax arithmetic fixed was insufficient for parity.

The Sol agent's value-product patch SHA-256 is
`3cf07e79cf67a0bf81d88bc55d1bbc0e88d4b5917211769182afac6763fb0a08`.

| Receipt | SHA-256 |
| --- | --- |
| Control | `c23610082092b89b80354b5b45986f21a81894a3227eb0ad11ae1a6a46713544` |
| Tensor value product | `780a069c014035ac32f9ad2916a0806ec05b30c47de54bfe6d41029b8cf31d06` |
| Direct query and KV views | `4138508663fb89373b631af40d5cde25a04733334980eb1cecc5397c84120f82` |
| Natural-text context refusals | `7c8ba13e0d618432731ecb5ec865f286cae2d8f1c4cd52b1c83b064709dd3af8` |
| Correct-capacity natural-text comparison | `b317761940f05cb6a1aca721487f816b7f2ef58cb25eb408ae3ac79159a0b1ac` |

Corrected natural-text harness SHA-256
`4b5ffc08ece1ffbf3659a97ef054b760965580b06c7b0e315d6b23058b12840c`.

## 20261125-HY4FRAG-23 · Consolidate mapped verification work

Three Sol patches were screened together with both resident banks and
the original projector. Per-table Engram inputs let mapped verification
submit one command buffer per target round. Paired Q8 GEMV rows reuse
weight decoding with the original per-row F32 accumulation. Indexed
attention below the selection cutoff uses the same contiguous causal
cells and keeps writing pooled history. This screen applies the
contiguous shortcut only to ordinary decode.

The worker used a 34,816-position cap. All six outputs matched the
previous 192-token hashes and draft counts. The snapshots reported
fair thermal state and zero named contenders. Pressure stayed at
level 1 and swap stayed at 4,175 MiB. Each target round used one
command buffer, down from five in HY4FRAG-21.

| Workload | Earlier resident tok/s | Combined screen tok/s | Target rounds / command buffers |
| --- | ---: | ---: | ---: |
| Short TypeScript, K2 first / repeat | 57.05 / 58.91 | 56.90 / 59.57 | 75 / 75 |
| Short TypeScript, K3 | 60.65 | 64.00 | 64 / 64 |
| Short TypeScript, K4 | 54.03 | 53.68 | 58 / 58 |
| 16K rendered code, K2 | 44.10 | 45.04 | 79 / 79 |
| 32K rendered code, K2 | 39.19 | 42.12 | 78 / 78 |

The combined screen improves the strongest short-code and long-window
results. The one-way comparison leaves attribution and control drift
open. Keep the candidate private pending paired timing and lifecycle
checks. At the measured rates, the remaining gaps for 191 useful
outputs are 0.26 s for code, 0.38 s at 16K, and 0.50 s at 32K.

The screening diff on `24ca8ba51` has SHA-256
`25b3b96a2947eff8b1398087be509df8ebcd86851e37628a6878e3471c6e8e0d`.
The locked Rust 1.98.1 worker, harness and receipt have SHA-256
`f8d4c6ff5967d335e1cfc1b1ea87a3c70a5994f1b2c47d60ce40ed1e94ad66b7`,
`bbc32032c59eee93fd06adf8fc64fe6136d83c31f9d9aad7cec34619376aec6a`
and `fa9d741e7c8b65464713449e1f8fe9677ee46b37b892c8f2715bddbdf33acd66`.

### Reverse the short-code comparison

The same worker binary ran the combined arm first and the flags-off
control second. Each load ran the 192-output TypeScript fixture three
times at K3 with zero prefix reuse, both resident banks and the original
projector. The worker advertised 34,816 positions. All six requests
matched the same output hash and used 64 target rounds. Both loads
reported nominal thermal state, zero named contenders, normal pressure,
and unchanged observed swap.

| Arm | First use tok/s | Warm tok/s | Warm median tok/s | Target commands per round |
| --- | ---: | ---: | ---: | ---: |
| Combined | 59.13 | 64.84 / 60.23 | 62.54 | 1 |
| Flags-off control | 57.18 | 61.64 / 56.70 | 59.17 | 5 |

Warm median decode improved 5.7%. The sample supports the combined
code-path gain across these two loads. Individual contributions and
paired long-window timings remain open.

| Request receipt | SHA-256 |
| --- | --- |
| Combined | `91bdd31f9c59aabf341bbfb38654733c4a5d96563c110a22e9b23f7fe0328a67` |
| Flags-off | `dfe6ea8783427c13c694dd2e148eaaa6c0e91cdfe5feba0c7c2ae8f516338d52` |

### Qualify the prepared worker

The rebuilt worker applies the three changes to mapped-resident mode
with GPU planning. It retained both expert banks and the original
projector at an 8,192-position cap. All 64 IDs matched on two K2 4K
requests at 56.23 and 56.59 tok/s. The two K3 coding requests matched
all 192 IDs at 54.46 and 56.28 tok/s. Every request used one command
buffer per target round. These coding rates came from a separate
lifecycle sequence after the two 4K requests.

The run reproduced 4,096-position text prefix hits before and after
images. Both images matched their expected outputs with zero drafts
and zero reused positions. Image completion, invalid images and
cancellation each invalidated the text prefix and recovered on the
next text request. The static review confirmed that per-table Engram
inputs remain valid through GPU completion and are charged before the
mapped admission recheck.

The host reported fair thermal state initially, nominal at exit and
zero contenders. Readiness took 77.03 s. The first 4K prefill took
16.65 s, including 4.44 s of Engram wait, and the repeat took 6.60 s.
The memory sampling subprocess timed out during the run. The collected
samples showed pressure level 1 and unchanged swap. The interrupted
monitor limits conclusions about peak pressure. The private harness
now handles sampling failures by stopping its owned worker.

Adopt the mapped-resident verifier. Keep the original Q8 arithmetic
for each row and the existing sparse selector above the contiguous
range. The typed patch on `0d640cd95` has SHA-256
`e7153eae0890e964e1ceb120e133c5b4a6592622676a36d90ce1854fd1267e7d`.
The locked Rust 1.98.1 executable, lifecycle harness used for the run,
and receipt have SHA-256
`71ef2cdcca7fe2c371320bc8e3f3b5f961a288c13104a858378a2b7ca2b431ca`,
`99f20cb3f3f38ace74bdaa2d7961efc35d3cd832f8050c485b141b947f37c998`,
and `9b50208f347b6e707c016c54951c7c99ab42d13bbe9377fa7f2995a1835298da`.

## 20261125-HY4FRAG-24 · Screen wider verification and direct routed weight loads

The advanced mapped build combines the earlier command consolidation
with independent gate projections, wider Q3_K and Q4_K expert threadgroups,
contiguous short-context NextN attention, and GPU Q8_0 draft embedding
gathers. On the 34,816-position worker, K3 used both resident banks and
the original projector. Every 192-output request matched its previous
target output IDs. The host went from nominal to fair thermal state
with stable swap and no named contenders.

| Workload | Useful decode tok/s | Target rounds | Verify s | Draft s |
| --- | ---: | ---: | ---: | ---: |
| Short TypeScript, first | 53.56 | 64 | 3.005 | 0.560 |
| Short TypeScript, warm | 64.35 / 59.49 | 64 | 2.677 / 2.915 | 0.291 / 0.295 |
| 16K rendered code | 41.95 | 68 | 3.941 | 0.611 |
| 32K rendered code | 40.75 | 64 | 3.867 | 0.819 |

The additional changes showed no clear gain on short code. K3 reduced
long-context round counts while the observed wall rates stayed below
the earlier K2 screen. The follow-up K2 load reported pressure level 2
and the owned-worker watchdog stopped it before readiness. Peak
observed swap rose by 37.12 MiB. The stopped load produced no rates.

The separate direct-register prefill path keeps the routed K16 product
and its F32 reduction order. It decodes packed weights directly into
SIMD group fragments, removing intermediate weight staging and its K64
barriers. On a 5,120-position streamed worker and 40 GB arena, both
arms matched all 64 reference IDs. The candidate ran first. Both arms
reported nominal thermal state and zero named contenders.

| 4K prefill arm | First use tok/s | Warm tok/s |
| --- | ---: | ---: |
| Direct registers | 582.77 | 582.31 |
| Shared staging | 571.00 | 530.08 |

The direct path improved first use by 2.1% and the repeat by 9.9% in
one pair. In the separate, reversed stage profile, the shared-staging
routed stage took 2.010 s first and 2.169 s warm, against 2.009 and
2.367 s with direct registers. All four requests covered 42 profiled
blocks and matched the same 64 IDs. The stage data show no reduction
in routed work. The direct path also passed all 13 natural-text rows.
Keep the original routed staging because the gain failed to reproduce.

The follow-up retained the original F32 score and softmax and split
the value probabilities into two F16 terms. It matched 12 of 13
control top-1 decisions with six true next tokens correct in both
arms. The discrepancy remains after restoring operand precision.
Keep the original value-product path. The failed gate stopped
throughput scoring of the residual variant.

Source diff on `0d640cd95` SHA-256
`cad75584cb57b68ee10bedfa4dd6c8f25a2de9b33ed20c448102bd2a699191e9`.
Locked Rust 1.98.1 worker SHA-256
`815bbd771b1090bb4f2aeb0bb762f561c47ee57e91c77ad6873effefcbaf4c80`.
Harness SHA-256
`ff4c528de3db600f6ca22b9abd7677c1551799704e22846d0dd4724d261ecb8b`.

| Receipt | SHA-256 |
| --- | --- |
| Advanced K3 ladder | `dcd48c77b7e43cc8b6274831faee556f2dce53f2ecc8a62014c11dfe4186e9c1` |
| Advanced K2 pressure stop | `181f7c2f6683259d378e1d545a7fe01d616197501ba8ad2d8d8e4a7fa958f2b6` |
| Direct registers | `ac418e6de71abcf983e2c340d9f86cef8a9e9c7e80e36694d218e61ea762e698` |
| Shared staging | `540a2e9eccf05a22fb9b80784e4cc9221b79eb31b1aaf0c323137ee7e04dc268` |
| Shared staging stage profile | `79b0dd5ce4d6e5eb77176dde02e078c0c7b4aa4e9ca89d608674bd30be6ae3cb` |
| Direct registers stage profile | `4f0281228a6ad23cc3ae76c02657028b2d29425d96c2a75f5c98f5d20e1533c9` |
| Natural-text register and residual checks | `61bbde1ff7ea88c61fd8a617e9cae281f966f1bc26a43e801a99f32b09da8215` |

Natural-text harness SHA-256
`f9e7565753accaf27233d74f04afd47dff04bb028f3ee68274530cca8e6e61f7`.

## 20261125-HY4FRAG-25 · Isolate prefill encoder boundaries from sampling

The earlier sampled prefill run exceeded its uninstrumented timing.
This screen reused the profiler's stage boundaries with counter
sampling disabled. It kept the same command buffers, stream cuts,
event ordering and concurrent compute dispatch.

Four loads of one Hy4-enabled executable ran the 4K K2 fixture twice
each, in control, candidate, candidate, control order. Every load used
the original projector, a 5,120-position cap and the same 40 GB
streamed arena. All eight outputs matched all 64 reference IDs.
Both snapshots from every load reported nominal thermal state and
zero contenders.

| Order and arm | First use tok/s | Warm tok/s | Receipt SHA-256 |
| --- | ---: | ---: | --- |
| 1. Control | 648.22 | 589.36 | `c5e4eaa43289a57631ba1048cfbeaeb74a3554b4b65b85f17db129e2f754a61d` |
| 2. Encoder boundaries | 611.19 | 520.17 | `1d0c8a5fcbed50aa67be8c4b04fd2cf32ac930baa4431dc7b0b06eee21168a7c` |
| 3. Encoder boundaries | 563.27 | 550.18 | `206f80b65da9101c82c8c3841ab6cde1f0a30df44b0da9ddd8822b71bc240c50` |
| 4. Control | 602.18 | 569.11 | `9a830928d4175e604af14e5f43faa3a4ba55d1068b7136f9825d1ec54bd4e65d` |

The candidate's median was 587.23 tok/s first use and 535.17 warm
against 625.20 and 579.23. Keep the existing encoder boundaries.
Use counters for attribution and uninstrumented timings for scoring.

The initial default-feature executable omitted Hy4 and all four runs
stopped at load. The receipts in attempted order have a concatenated
SHA-256 of `8376d18794eceaa239f9d2e5a9abc7836846f2ba6b1cd3d8ebd95eb5c97d272c`.
The corrected locked Rust 1.98.1 executable explicitly enabled
`stark-generate/hy4` and has SHA-256
`710f2cedff8c570f94be396208336754704a0fecc8fc88276821e9ce3ef38d86`.
The unified source diff on `ec81193fb` and harness have SHA-256
`aa0b8e8cb0161e3feb201b9933f4c6ff35f1d5a1c96bc08ed4b39b43ce28a386`
and `41428073bf0ffeef4ee997860541e5731cd78f9aa0fd10c18ed6c360f0ff5164`.

## 20261125-HY4FRAG-26 · Pack dense prefill K tiles and test wider mapped verification

The dense candidate packs the existing F16 weight and activation values
in contiguous 64-element K slabs. The two large Q8_0 attention
projections keep their F32 accumulation order, output strides, dispatch
geometry and scratch allocation. An offline address check covered
168,042 four-element source coordinates, including partial row tiles.

Four 5,120-position loads used the 40 GB streamed arena, original
projector and 4,096-row chunks. All eight K2 requests matched all 64
reference IDs. Both host snapshots from each load were nominal, with
zero named contenders.

| Order and arm | First use tok/s | Warm tok/s | Receipt SHA-256 |
| --- | ---: | ---: | --- |
| 1. Control | 561.85 | 596.76 | `2945d60bb34c675f35acb1e375c00cbb106cf7faa0df9cfa837cdb8ea3fa354a` |
| 2. K64 layout | 603.18 | 572.61 | `c68311e532a3abca8442ad4a383a5652fe3defd0eebbd638b998f402c1244d60` |
| 3. K64 layout | 675.87 | 616.22 | `f1a389bffc54f34ed5bfddef72f1dc025992b89638f4d67245d4d860ce989f35` |
| 4. Control | 619.45 | 576.47 | `862cf11a3cf5188ce3741065a8884816d1c3752bf56bf2673dfdf483f68bacad` |

First-use wall time fell by 0.50 and 0.55 seconds in the paired orders.
The median was 639.53 tok/s against 590.65. Warm timings moved in
opposite directions. The 675.87 tok/s endpoint remains below 743.2.

In a separate K64-first stage profile with 42 sampled blocks per
request, dense attention projections fell from 1.677 to 1.479 s on
first use and from 1.848 to 1.704 s on repeat. All output IDs matched.
The earlier overlapping profiling launch was refused by the exclusive
benchmark lock and produced no timing. The refusal, K64 profile and
control profile receipts have SHA-256
`44a280667cc4db0a1bd4bbd13620cda8bd5d18d5296b44db4ca8bb2f21c46f09`,
`bea33526566ec86d7317e60cd9440a290b94e151a7f5d7546fdac2e95a940440`,
and `ccd4ea1e80b28eca41b34fa20fc6803f30fb889a5f8fb5979b221535c4cc17c2`.

The natural-text check at 552–840 positions exercised the device-view
path on all thirteen rows. All top-1 decisions matched, with eight
true next tokens predicted in each arm. The harness and receipt have
SHA-256 `468d5ec8e4ddbadedd51fdd1a76ae50934766c99d512d3774fa679bd168dd495`
and `a79bd5030507371beb8619e00398a3b05eb76271445b42e939eef254cfd9d260`.

The independent decode candidate shared each latent-cache load across
more heads, preserving F32 reduction order. It also reused SIMD lanes
for the routed expert planner's compare and exchange stages. Offline
emulation matched 4,000 reference plans across all verifier widths.
Both 33,024-position loads verified target and NextN residency and
retained the projector. Fixture and output hashes matched the earlier
pinned references for every 192-output request. Thermal state was
nominal and pressure stayed at level 1 with stable swap.

| Arm | Short TypeScript K3 tok/s | 16K K2 tok/s | 32K K2 tok/s | Receipt SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Wide attention and SIMD sort | 63.50 / 60.02 | 43.85 | 40.63 | `e1b8238dbf5f30668ffaf492f44bd860350361c877a936706c55d16dcd74db81` |
| Control | 64.22 / 59.19 | 44.62 | 41.97 | `3ed17c63f8bfb066b6c7f54edbff3d761864a9ea19923ab37b4f4a333066376d` |

The long candidate increased target verify time from 3.757 to 3.833 s
at 16K and from 3.804 to 3.946 s at 32K with the same round counts.
The warm coding medians differed by less than 0.1 tok/s. Keep the
existing mapped attention tiles and plan sorter.

The integrated source diff on `ec81193fb`, locked Rust 1.98.1 Hy4
executable and test harness have SHA-256
`447d941802390deb78a66676cc5f74b9fb417eb9f163cf2847049794aec336e0`,
`aa9b589738bb2f7b9ed37dc4366d7f981cafbb174444690d0ac5c8b978450668`,
and `7d36fea9906106dd77e09c7eda0cfc0ce9c6c12e01c73eb1ebfe4c3208432a3d`.
The private in-process memory probe SHA-256 is
`1b3a1d16a2cbcbafc8263a7ca5f6fc0981bd60a079e14e872e636e53468053fa`.

### Qualify the rebuilt dense layout

The isolated build uses the slab layout whenever the existing opt-in
dense device-view path is enabled. The 5,120-position streamed worker
reached 644.05 tok/s on its first 4K prefill and 601.56 on its repeat.
The separate 8,192-position mapped-resident worker reached 658.59
and 613.64 tok/s. Its K2 decode reached 53.37 and 55.04 tok/s.
All requests matched the 64 reference IDs. These are absolute rates
from the rebuilt workers. Both thermal snapshots on each run were
nominal with zero contenders.

The mapped worker retained both banks and the original projector.
It restored 4,096 text positions with exact continuation, matched
both image answers with drafting and text reuse off, and recovered from
invalid images and image cancellation. Readiness took 74.24 s.
Pressure remained at level 1 and swap stayed at 4,222 MiB.

Adopt the K64 layout for dense device views. The source diff on
`2f9db649c`, the locked Rust 1.98.1 isolated release build, and the
private validation harness have SHA-256
`8e75eb302885e3a3aa3250261764f87765820a0a6bc2a0ac335aced32f3e12af`,
`a9dff56bf298a26430be738d368e864f4d4625bb2e265a8f0b35755f17a2feb8`,
and `99500045d487c7d755edf2fbbca21ad9974d566dcf682dc86379ca5339eaa7a3`.

| Receipt | SHA-256 |
| --- | --- |
| Rebuilt 4K streamed worker | `a0a934485d04dd2a1b6964fbdda02bae8d1ff8a2b2baee5d6946a8628cf42fa1` |
| Rebuilt 4K and mapped lifecycle | `588dec1ee968dff06bab944277fed986a9e73384ff305c10de7f402aa40d15d5` |

## 20261125-HY4FRAG-27 · Screen smaller routed staging and group-local dense packing

The routed candidate halves the packed weight staging K dimension
from 64 to 32, while keeping 128 threads, the route assignment, F16
operands and the K16 F32 accumulation loop. Threadgroup storage drops
from 18,432 to 10,240 bytes. The extra stages double the barrier count.
The dense candidate packs each compute group's K slabs next to one
another, keeping the validated device-view tile arithmetic and scratch
size. Offline coordinate checks covered 1,236,992 routed cases and
1,457,808 dense cases.

Four 5,120-position loads used the 40 GB streamed arena and original
projector on the same executable. Each ran two 4K requests with K2
and zero reuse. All outputs matched the 64 reference IDs.

| Order and arm | First use tok/s | Warm tok/s | Thermal state | Receipt SHA-256 |
| --- | ---: | ---: | --- | --- |
| 1. Control | 623.04 | 612.35 | Nominal | `30b906c39780f06c5079efe11e32e622801442a4fbfa4e8aa18ffbb46430b103` |
| 2. Routed K32 staging | 626.12 | 611.28 | Nominal | `dc5d4f7c7cfcf76406e0b5cb9a7d3d7da87f406137d01ae6005ea2014c6ad958` |
| 3. Group-local dense K64 packing | 641.09 | 598.91 | Nominal to fair | `5e7ab7455a5e7ef0771a79f4a58f8c3217655e13574d2b8e6dcd10aa25d95ad5` |
| 4. Control | 631.67 | 622.89 | Fair | `620e6a71c8b2f61a8226453b1dde4b8ea9945fe6504d2e1e7baa545112643267` |

The smaller staging tile landed inside first-use control drift and
slowed the repeat slightly. The dense layout's first-use rate was
1.5–2.9% higher than the controls and its warm rate was lower. Thermal
state changed through that comparison. Keep the current staging and
dense layout. The 743.2 tok/s target remains open.

The private source diff on `4659df684`, isolated locked Rust 1.98.1
Hy4 executable and harness have SHA-256
`b0b5d3345779ddc9c4aaba7c4aa038c07b449e0dcf7a8537e6f4532dfaf1dd0d`,
`86b191fd0c40b42fb9a42b4e143a83ab0789a2a17385f151f1e572c9280611eb`,
and `3dc12a87e73e003feec7d88a0dd75c9c8c7fb49878f6c0ab5ccaa5a455b68e9e`.

## 20261126-HY4FRAG-28 · Combine the two dense tensor products

The candidate replaces two 16 × 32 × 64 products with one
32 × 32 × 64 product on the existing K64 device views. It keeps the
packed F16 operands, F32 accumulation, dispatch grid and scratch
allocation. Offline checks matched 306,140 input coordinates and
covered nonoverlapping output tiles across 19 row counts.

All four 5,120-position workers used the 40 GB streamed arena, original
projector, 4,096-row chunks and K2. All eight complete 4K requests
matched the 64 reference IDs. Thermal snapshots stayed nominal and
no named contenders were present.

| Order and arm | First use tok/s | Warm tok/s | Receipt SHA-256 |
| --- | ---: | ---: | --- |
| 1. Existing products | 444.25 | 473.70 | `dabc136ef00bbd719378f65eeaf389632e434927f95b34d4668d9184bece0b85` |
| 2. Combined product | 573.64 | 497.71 | `bda6b4f147f05d2cffa37547dcbc4fa0fb217f75d19c0838ac78ab8a25fb5476` |
| 3. Combined product | 585.04 | 509.72 | `103e2e6f2e4b09633e7e398b7a1d2ef6e193766c4dfcadf38c122cd339c00af4` |
| 4. Existing products | 573.90 | 499.66 | `f2b57cf5df3c16fdd949a81299cba5b738fec4c7b31ded1b607251cfa72be885` |

The opening control's first GPU span was 8.653 s against 6.887 s in the
closing control. The candidate's first GPU spans, 6.889 and 6.747 s,
were close to the closing control. Separate attribution measured
1.788/1.749 s in dense attention projections for the combined product
against 1.542/1.955 s with the existing products. Each request sampled
42 blocks and matched the reference. The gain failed to reproduce
across timing conditions. Keep the two existing products.

The private source diff on `4fb3b10dc` includes the subsequently
adopted packed cache path. Its SHA-256, the isolated Rust 1.98.1
release executable SHA-256 and harness SHA-256 are
`605d59683bd2c8cfbb3f4bad4eec262de29107a0e2c799c681c79c866752ed0e`,
`c5f82fd7a290d18480b3ce922b4fe42baeef94063831b0ee2b9a0e4ba1196d2c`,
and `846cc698dba05bab7566f33a74cc01d05982961929379573d2ea3e9079558d94`.
The combined and existing product profile receipt hashes are
`c3d0ce3683c39c01a8f5d1e634d670ab23f3693e7c81c4b417a577fbc723444e`
and `513272308dfa6e2537e5010f7a4417f6ed37f641fbee9560066dc7732a51ddc6`.

## 20261126-HY4FRAG-29 · Share packed expert bytes across K16 fragments

The prefill candidate reads shared low-code bytes once for each pair
of Q3_K and Q4_K fragments. Q3_K also shares its high-mask loads.
It keeps the original scales, F16 staging layout and F32 product
order. Offline checks covered 524,288 packed-byte, mask and coordinate
cases and verified that staging writes each coordinate once.

Four fresh 5,120-position workers used the original projector, 40 GB
streamed arena and K2. The engine ran two full 4K requests per worker.
All eight outputs matched the 64 reference IDs. Both host snapshots
from each load reported nominal thermal state.

| Order and arm | First use tok/s | Warm tok/s | Receipt SHA-256 |
| --- | ---: | ---: | --- |
| 1. Existing fragment loads | 620.97 | 608.11 | `81f9745bc3bf28f9db335ed557e7d82d496dc7136ddf1137cf17d0a912e3c13e` |
| 2. Shared packed loads | 580.22 | 542.03 | `a93b5ae98fe87f1e05da425f88ac571c4f645b1a6390e00a2f97b0731291b199` |
| 3. Shared packed loads | 595.19 | 556.67 | `0853ca67253fac88f3794dfaa3eda3dacb1e37072725dac4e74207cd095f9142` |
| 4. Existing fragment loads | 550.70 | 565.63 | `7c77229597d4621c8247626c67ec2b49c4005db02d698742a26089a297675b48` |

First-use results fell inside the control range and warm results were
lower in both orders. Keep the existing fragment loads. Sharing the
logical byte loads did not recover the 4K prefill gap.

The source diff on `4d024cb38`, isolated Rust 1.98.1 Hy4 executable
and harness have SHA-256
`ec3dd25db6b3fd152de488d401b0ef655397530ec5a812c343cdfbcbc234fece`,
`5d95b571c362beb0f09c7b7721363934c86332b985bf5da1482055703d026d4b`,
and `669fc50a21f2c6211ba72194c266309ce97cf353bdf8d235db0e0bfb02ada882`.

## 20261126-HY4FRAG-30 · Use FP32 probabilities in sparse value products

The candidate uses Metal TensorOps to multiply the existing F32
softmax probabilities by gathered F16 values, with F32 accumulation.
It keeps the production selector, QK, softmax and cache layout.
The tensor path covers sparse prefill positions 2,048–4,095 and
changes the accumulation grouping. It leaves target verification,
NextN and image requests on their existing paths. The installed SDK
supports the threadgroup input types. [Apple's mixed-operand example](https://developer.apple.com/videos/play/wwdc2026/330/?time=753)

Four 5,120-position workers used the same 40 GB streamed arena,
original projector, 4,096-row chunks and K2. All eight outputs and
draft counts matched. The thermal state stayed nominal with zero
named contenders.

| Order and arm | First use tok/s | Warm tok/s | Receipt SHA-256 |
| --- | ---: | ---: | --- |
| 1. Existing value product | 646.53 | 616.98 | `e0a02584e5821d79a09b67fc43e320b19f18cfbf22b40c11f6422a0d7a32241c` |
| 2. F32 tensor input | 627.23 | 583.09 | `f39d90cf1956f3c86664315642434ff8e157ab2acd44eba5843179011554f95e` |
| 3. F32 tensor input | 653.15 | 633.73 | `45ab278e5860168c28cd28258e5403b3dfdf0dbee6e34ead2bf80a7e162c4b51` |
| 4. Existing value product | 639.68 | 570.42 | `dad43b0ab95ad0181c07184a3d2134c24d8c222a9b7cb23c27b173c41e3fc40b` |

The speed comparison changed sign with launch order. Separate
stage attribution sampled 42 blocks in every request. The candidate
spent 0.956/0.935 s in MLA against 0.799/1.040 s in the control.
Observed routed durations varied as well.

At 2,120–2,408 positions, 12 of 13 natural-text top-1 decisions
matched. Both arms predicted six true next tokens. Keeping F32
probabilities and the original scores still left a disagreement.
Keep the existing sparse value product.

The source diff on `db01930d5` includes the adopted Q5_0 draft head.
The diff, isolated Rust 1.98.1 executable and timing harness have
SHA-256 `d1b4281dc440c7b336648f876422b47e5230eadeb29208a0833661aa03968ec8`,
`6aa0a7372aaab3ed8f011c91bb7917278a80dabb467bd93c523c18ff828477c4`,
and `2c9ab7f532d08e272cee84676f09dd0319ee0d23b97ea0781990d3f850f4f392`.

| Receipt or harness | SHA-256 |
| --- | --- |
| F32 tensor input profile | `63564a2c35fd8c3ea804646eb3cbb6649d4c25d5bbc014d2c870c3b7d1995969` |
| Existing value product profile | `9932b562080e52e04537e441faa035652f063bd2a84a7abf1b92b39660244288` |
| Natural-text comparison | `924e977d5fe7424580d8ef6c84fc1a1b315c9397728e9089a386eacc95c96967` |
| Natural-text harness | `cf6d2447281ef0253676575bc650687cd557c234a4ee1189c84b6b323efe53dd` |
