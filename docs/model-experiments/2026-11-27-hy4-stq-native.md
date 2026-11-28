# Hy4-Flash Sushi STQ native bring-up

2026-11-27 · HY4STQ-NATIVE-01 · Native component and trunk diagnostics

## Artifact and build

The source is `beamster/Hy4-Flash-Sushi-STQ-4bpw` at revision
`61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`. The exact 208,397,806,219-byte
pack was verified separately for the Sushi run. The Rust reader checks its
configuration, index, inventory, file identities, all 89 shard headers and
the Engram token map. These open-time checks do not rehash all weight payloads.
The diagnostic borrows the original immutable release files throughout GPU
execution. [Artifact inventory and quantizer identifiers](2026-11-26-hy4-sushi-stq-preflight.md)

The implementation reuses the Hy4 GGUF backend's hyper-connections, MLA,
IndexPool, head, context sizing and rewind. It maps the PAIR1 routed banks
and SPEC1 dense matrices directly and converts the joint SPEC1 K/V
projection into the graph's absorbed BF16 views at load. Engram reads the
four original BF16 tables via positional reads. The separate STQ gate
preserves the order of the independent key and query RMSNorm weights.

The packed codecs and Metal kernels adapt Sushi v1.8.1 at
`b7b7975cc017acba1b4c0877d65543c4c5a4a381`. The MIT and Apple MLX notices
ship alongside the kernels. Tests ran on an M5 Max with 128 GiB, Metal 4
and Rust 1.98.1. GPU runs held the shared Hy4 lock.
[Sushi source and notices](https://github.com/beamivalice/sushi/tree/v1.8.1)

## Component evidence

| Check | Result and scope |
| --- | --- |
| Pinned inventory | 3,179 tensors and 1,256 trunk aliases. Bindings covered 123 PAIR1 K4 expert banks, 420 SPEC1 K6 trunk projections and 44 SPEC1 K4 draft projections. |
| CPU tile codec | Three upstream fixtures, 96 seeded tiles, 256 real tiles and four 128-by-128 blocks matched the independent decoded reference. Component output SHA-256 `b8eb84d26a7e6491d5c3cfb8a0e61deaf71cbeaff0d46bb3851c6d91d06b6b94`. |
| Absorbed KV | The actual 32-head and 64-head projections matched an independent reconstruction on 22,020,096 BF16 output words. The CPU oracle uses the same documented rounding policy. |
| PAIR1 Metal | Three real layer-one projections matched the rounded stage references at relative RMS at most `1.55e-5`. The mapped fused gate/up/down component reached `5.80e-5` relative RMS against the independently rounded CPU chain. All 4,608 weighted combine values matched in router-slot order. |
| SPEC1 Metal | Real K6 trunk and K4 DFly2 query projections at 3,072 by 1,024 passed singleton and 39-row cases. Relative RMS at 39 rows stayed below `2.07e-5` against the matched rounding reference. Full chain warm command medians were 0.433 and 0.427 ms respectively. |
| Packed embedding | 36,864 F32 values for 12 real ID occurrences, with a relative RMS of `2.14e-7` against CPU block reconstruction. Repeated rows matched exactly. Out-of-range inputs were bounded. Warm command medians were 0.034, 0.127 and 1.600 ms at 1, 32 and 512 rows. |
| Engram | All 320 sampled rows matched independent positional reads after closing the original descriptors. The independent gate check covered 12 streams, four exact zero-dot 0.5 gates and a maximum finite-normal absolute difference of `9.32e-9` in the gate. |

The first PAIR1 shader compile used a reserved Metal identifier. The
corrected shader compiled and passed the real-bank checks. An extreme
subnormal Engram gate stress case exceeded the initial absolute
tolerance, with an absolute gate difference of `2.91e-6`. The finite-normal
reference passed and distinguishes independent key/query products from a
swapped weighting.

A private contiguous dense prepare prototype preserved exact output bits
and reduced median 4,096-row GPU command time from 0.476 to 0.201 ms on
one 3,072-wide shape. This is component evidence. The forward integration
initially uses the validated general prepare to keep its materialization
and schedule changes independently testable.

## First full native trunk runs

The 1,024-row graph opened with a fitted 4,096-position context. An
eight-token diagnostic produced finite vocabulary logits and completed
two greedy decode steps. Load, first prefill and decode took 41.730,
16.798 and 0.134 seconds respectively.

The 4,096-row graph opened with an 8,192-position context and completed
a 4,096-token synthetic prefill and eight decode steps. It took 38.308 s
to load, 26.985 s to prefill and 1.208 s to decode, or 151.8 and 6.62
tok/s on this first request. Of the prefill wall time, 6.227 s were
reported inside GPU command spans and 1.343 s in Engram preparation
and reads. Zero software expert-read bytes are expected with mapped
banks and do not measure file-backed page faults. Output SHA-256
`69b44c11ad119a268c13216bbe3a97709ccccb6e534260b6e366b6cbdcb154b3`.
The fused MoE's indirect command matched the direct command bit for bit
on the independent real-layer component fixture.

## Image and draft integration

The released pack stores all 441 vision tensors as BF16. The adapter maps
the matrices in place, widens vectors and position weights as required by
the existing tower, skips the unused CLS position and uses the pack's tanh
GELU projector. The 96 × 64 vision fixture passed all 32 stage comparisons
against a CPU BLAS oracle reading the release weights. Its 24,576 output
values differed by `2.85e-6` relative RMS. Median tower encode time was
14.47 ms across five warm runs. Static vision memory was 1.01 GB, with
1.20 GB reserved for the largest admitted image.

The new DFly2 path binds 92 aliases for the five dense K4 layers and
captures the committed target row and selections from blocks 25, 29, 33,
37 and 41. Its draft cache is bounded to the selected target rows plus the
causal draft window. The real-weight component generated five proposals
with synthetic target state and matched the first proposal at widths one
and five. It exercised both contiguous and selected target rows.

The image-loaded 4K graph at depth two then completed two full prefill
and decode requests. The first prefill took 26.271 s and the warm repeat
took 6.594 s, or 156 and 621 tok/s. The seven decode transitions took
1.128 and 1.262 s, or 6.21 and 5.55 tok/s. Each offered nine drafts and
accepted one. Both had the same generated-ID hash as the earlier native
autoregressive run. The decoded-text hash
`fcc0c5023bd5a8bd7152f233b78a4da9fde7c80ef88f0cfd0a37992b147f9c84`
also matched the reference on this fixture. The reference used the same
4,096 input IDs and 8,192 capacity, with vision loaded, drafting disabled
and its unquantized BF16 cache. Sushi prefill was 892 and 923 tok/s across
its two runs. Sushi reported 37.55 and 44.96 decode tok/s using its
eight-output-token numerator. Stark stores the cache in F16.

The lifecycle continued from a retained 4,096-token prefix and reproduced
every vocabulary logit bit for the suffix when replayed fresh. The same
loaded model processed eight projected rows in an actual image request.
Drafting was disabled on that request, and attempts to retain its prefix
or continue it as text were rejected.

The first integrated vision smoke failed at load because the Rust tensor
map's iteration order did not match the adapter's source-name order.
Sorting the pinned source names corrected it. The first drafting
diagnostic submitted a second anchor after the verifier had committed the
next one. Submitting the anchor only once corrected the diagnostic.

An optional BF16 boundary mode matches the documented Sushi input and
output casts on packed linear projections. Component tests matched the
CPU cast and stage references. At 4K, the mode took 6.243 s warm and
changed the decoded-text hash. The default diagnostic keeps the
matched-output F32 boundary policy.

## OpenCode through Stark

The pinned STQ validation worker `fd5ef340395a6382cd2fe40fdc6042750855115560b5f0ba601e0b215ba35028`
served the three synthetic OpenCode turns through the installed connector
and Stark's local HTTP API. Stark loaded the image tower, passed the
normal image startup checks and advertised the Hy4 XML tool parser.
The 49,152-position fitted ceiling was used with 4,096-row prefill,
greedy generation and DFly2 at depth two. There was one generation per
turn and all three emitted 96 output tokens. Startup including image
checks took 63.76 s.

| Turn | Input | Reused | Prefill tok/s | Decode tok/s | Wall time | DFly2 accepted / offered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh 16K | 16,229 | 0 | 540.68 | 23.55 | 34.85 s | 57 / 76 |
| Continued 27K | 27,179 | 12,288 | 508.11 | 21.74 | 34.35 s | 56 / 77 |
| Fresh 27K | 27,229 | 0 | 492.40 | 23.73 | 60.09 s | 58 / 74 |

Prefill rates use computed rows and decode uses 95 post-prefill tokens.
The 16K run spent 26.90 of 30.02 prefill seconds inside measured GPU
spans. Decode spent 3.69 of 4.03 seconds in target verification and
0.35 seconds in the drafter, despite accepting 75% of the proposed
tokens. Longer continuation input hashes can differ across model
artifacts because the first turn's output is part of the history.

Earlier runs stopped during saved-model admission, template admission
or image accounting. The final run used the pinned STQ header verifier
at desktop admission, the release template hash with the existing
Hy4 XML parser, and the same 72-row image accounting as the Hy4 tower.

### Prepared mapped residency

The follow-up worker `20ca0fde0c24f1f8d069e04dd377594021644f6c6b4981b8a95a75caec57adf4`
reused the GGUF page-preparation and Metal residency code. It faults
the original mapped weight pages under the same pressure and swap
budget and stops the legacy keep-alive before wrapping packed weights.
It holds the admitted weight buffers in a Metal residency set. Image
and DFly2 settings, prefill width and fitted context matched the
earlier run.

| Turn | Prefill tok/s | Decode tok/s | OpenCode wall |
| --- | ---: | ---: | ---: |
| Fresh 16K | 563.81 | 27.92 | 33.03 s |
| Continued 27K | 578.57 | 27.07 | 29.90 s |
| Fresh 27K | 577.81 | 28.59 | 51.14 s |

All three rendered inputs, outputs, reuse counts and DFly2 counts
matched the demand-mapped runs. The observed prefill gains were
4–17% and decode gains were 18–25%. Startup including image checks
increased from 63.76 to 100.49 s. At the two host sampling points,
memory pressure was normal and swap held at 599.56 MiB. This is one
paired matrix across two builds. The residency policy stays opt-in.
The comparison bundles residency and disabling the old keep-alive.

| Private evidence | SHA-256 |
| --- | --- |
| Demand-mapped OpenCode aggregate | `6f1c20ed8280150dea85da726bad95c62169349695aedab6de51e45d8f9b3370` |
| Prepared-resident OpenCode aggregate | `bb19ab585866d169317ef369b59ffe7322c3f368d6b437418a7f05dbc36282d6` |

## Packed kernel screens

The SPEC1 unpack-once screen borrowed the existing direct F16 Metal
device views and kept each 16-wide reduction step. On two real sliding
projections at 1,024 and 4,096 rows, it matched all 99,614,720 intermediate
and result values exactly. It used 96 MiB more scratch and increased
warm GPU time by 37–81%. For the 4,096-row gate and output projections,
GPU times moved from 9.813 to 13.991 ms and 10.444 to 18.915 ms respectively.
The direct packed kernel stays in the graph.

A PAIR1 GEMV screen on real layer-one weights and router picks reduced
one projection from about 0.177 to 0.103 ms at one token. Fully
overlapping six-token routes were roughly twice as slow. Swapping all
three routed projections missed the provisional combined RMS gate
with `1.316e-4` against a `1.2e-4` limit. The down-only screen passed
with `1.26e-5` relative RMS and a median routed-chain change from
0.2625 to 0.2285 ms. The validation worker exposes the down-only
single-token path as an opt-in.

The native worker's down-only arm retained the exact eight generated IDs
from the prior 4K control on both runs. It decoded seven transitions in
7.987 s on first use and 1.258 s warm. This pair repeated the opt-in
arm and has no same-build timing control. The component gain does not
establish a worker throughput gain, so the option stays off by default.

## Qualification boundary

The native worker is enabled for local STQ validation. The catalog
continues to identify Sushi as the runtime. Native quality beyond the
recorded fixtures, default policy, and packaged installation remain
separate gates. Diagnostics emit aggregate results and hashes.

## Local checks

Rust 1.98.1 release builds and Clippy with warnings denied passed for the
Hy4 worker. The focused converter tests passed 515 cases across chat,
images, tool parsing, MTP and STQ preflight. The catalog suite passed
155 cases. The pinned Rust formatter and Ruff checks passed.
