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
