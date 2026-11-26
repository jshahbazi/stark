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
