# Retained Lily vision source numerics

Date: 2026-11-07. Diagnostic source: `9064eef6`; attention candidate:
`408381f8`; masked-scratch guard: `9aeea9b1`; integer-barrier candidate:
`771ddfb3`. This record concerns the original-format retained Lily vision
component. Full-worker lifecycle evidence remains in the separate
[image-family record](2026-11-07-qwen-image-families.md).

## Artifact and independent source

The artifact is `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`, with artifact identity
`18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`.
All 333 visual tensors are original BF16 storage: 897,862,112 bytes from the
existing local shard. No weights were fetched, converted or modified.

The CPU producer executes selected unchanged arithmetic bodies from
Transformers `2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397`:
`modeling_qwen4_exp.py` SHA256
`e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df`.
It follows Lily upstream `e22d58fa7cc488493fb1b5d810a198bd29730aeb`'s
`tools/reference/hf_vision_reference.py`, SHA256
`79d37b1a81308db9da677d99506613383b6f3bf5190bf025c34f9ca25822ee92`.
Torch 2.14.1 runs on CPU with four threads; Transformers 5.15.1 supplies only
the pinned scaffold described and hashed in the manifest. Strict parameter
assignment supplies every visual parameter; the rotary inverse-frequency
buffer remains F32.

Two synthetic cases use identical frozen source pixels at 0, 1 and all 27
blocks. The tower gate stays relative RMS at most 0.02 and cosine at least
0.999. No acceptance threshold was relaxed.

## Failure localization

The existing v3 CPU manifest SHA256 is
`02b0e2d74141f5b6a6d64449f9e6cd2e5b046af80ea3e216ef4c4abe5863b0c3`.
On source `c2d7d6cd`, the root-owned sealed native test
SHA256 `473c342216037e20e9cfc8746a70da7e6414ac67af4be101b809f501fdfa0572`
failed the numerical gate. Its six aggregate controls were:

| Case index | Blocks | Relative RMS | Cosine |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0.0006575898 | 0.9999997838 |
| 0 | 1 | 0.0032868664 | 0.9999945984 |
| 0 | 27 | 0.0252454230 | 0.9996813587 |
| 1 | 0 | 0.0001088986 | 0.9999999941 |
| 1 | 1 | 0.0029782713 | 0.9999955650 |
| 1 | 27 | 0.0188444247 | 0.9998283354 |

Only case 0 at full depth exceeded the fixed RMS gate. The aggregate replay
took 36.68 seconds, peaked at 951,255,040 bytes RSS, observed at least
106,239,426,560 bytes available RAM, and did not trip its guard. The original
failure and the aggregate replay receipts are preserved separately.

The optional `--operators` CPU mode records source operator inputs and outputs
at one block, using forward hooks and an observation wrapper around unchanged
eager attention. The v5 manifest SHA256 is
`c4cd9d332849de9d7946fb0e66309a0bb9ee4cf4578aa854561123079a363b66`.
It contains 20,847,793 bytes and peaked at 1,307,852,800 bytes CPU RSS. All
six merged-output digests match v3 exactly. The native ignored unit replays
each operator on its exact source input, so preceding operators cannot
contribute accumulated error to an individual control.

The native operator baseline at integrated source `7cb13f39` reproduced all
six v3 tower metrics. Its sealed test SHA256 is
`757a1de86e4e4890c3af767158bc0129ce0e78f9d743dfe9f859b40cfd2ece19`.
Attention operator index 3 measured relative RMS 0.00423718156 and
0.00408923204 for cases 0 and 1. Every other same-input operator was at or
below 0.000087729 relative RMS, with most near 0.00001 or exact. This isolates
attention arithmetic as the component discrepancy, independently of the
full-depth accumulated result.

A separate CPU policy oracle uses the same source attention Q/K/V input,
F32 dot accumulation, and explicit BF16 boundaries. These are policy
simulations, with native compiler/reduction behavior qualified separately:

| Policy | Case 0 relative RMS | Case 1 relative RMS |
| --- | ---: | ---: |
| Retained online, unnormalized BF16 exponentials | 0.0042373480 | 0.0040892320 |
| BF16 QK/scaled scores, unnormalized BF16 exponentials | 0.0021092730 | 0.0022688445 |
| BF16 QK/scaled scores, globally normalized BF16 probabilities | 0.0001280236 | 0 |

The private oracle receipt SHA256 is
`dadc101afed27fbe7bca8fd9856ac79bb95d58656800053eeb23599cb42ab15a`;
its script SHA256 is
`9c42551f8679b10cc95460280cbd7a24780426fe9ebfe36390be9a1957218e13`.

## Arithmetic change and acceptance

The vision attention kernel now rounds the QK result to BF16 and rounds the
scalar-scaled score to BF16. Its first tiled pass computes global F32
softmax statistics. A second tiled QK pass casts globally normalized
probabilities to BF16 before PV. Both passes retain 32-query/64-key tiles,
with no image-sized square attention allocation. The additional QK pass is
a performance cost that has not been benchmarked here.

The ignored test adds an independent same-input attention gate: relative
RMS at most 0.001 and cosine at least 0.999999. It prints aggregate metrics
before checking either operator or full-tower gates. The original full-tower
gate stays unchanged. No production API, checkpoint format, text arithmetic
or global diagnostic selector is added.

An independent read-only source review checked the two rounding boundaries,
global probability normalization, per-row shuffle grouping, and the barriers
between QK, softmax statistics, probability staging and PV. It found no
blocking arithmetic or synchronization issue. Its observation about masked
key-slot scratch reads led to `9aeea9b1`, which moves those reads inside the
valid-key branch in both passes and removes an unused first-pass V slice.
The inherited single-pass tile-speed claim was also corrected. Valid-key
arithmetic is unchanged; the final source still requires its own replay.

The first native candidate at integrated source `621326da` failed the tight
attention gate. Its seal SHA256 is
`8c2572d0f49a5ced367b51c80ae191be047cbffdf4ea610cbc70ee94e8f1788b`.
Same-input attention relative RMS was 0.00334684894 and 0.00330652702; full
tower relative RMS at 27 blocks was 0.02805142 and 0.02207323. Other operator
controls remained unchanged. Peak RSS was 972,324,864 bytes and the guard
did not trip. This failure is retained and is not qualified evidence.

A follow-up CPU policy diagnostic matches that native attention discrepancy
when QK rounds to BF16 but scaled scores remain F32: 0.00334702816 and
0.003306527016 relative RMS. Explicit native stage observations were then
added in `afb3525c` to compare QK, scaled scores and stored probabilities
with source stages, including an ordinary-versus-observed output comparison.
The v6 source fixture manifest SHA256 is
`4cb0f69a1989f8d3b65c08cc389e8ae8aff0f4d2c036ae684e5872518ea893ad`;
its 27,142,533 bytes were produced at 1,324,548,096 bytes peak CPU RSS.
No acceptance threshold changed following the failed native candidate.

The first trace replay at integrated source `43fc445c`, seal SHA256
`13061d1972d997c1b6421f1b2eba68330978ed6168090048514736811427dd54`,
showed that observing the floating casts changed compiled arithmetic. The
captured four-patch control had exact QK/scaled-score stage agreement but
0.0130567764 probability relative RMS, while observed output differed from
ordinary output at 2,461 BF16 elements (0.0128432321 relative RMS). Ordinary
output retained the earlier failed candidate metrics. The initial wrapper
retained only the last of the two trace records. This is evidence of observer
perturbation, and does not qualify the production stage graph.

Candidate `771ddfb3` replaces intermediate BF16 constructors with explicit
integer round-to-nearest-even barriers after QK, scaled scores and normalized
probabilities. It follows the existing Stark integer BF16 policy used in GDN;
compiler promotion cannot remove an integer bit boundary. A separate CPU
source control checked 326,400 finite F32 bit patterns: all 65,280 finite
BF16 upper patterns with five lower-bit controls each, including every exact
halfway tie. It found zero differences from Torch's BF16 conversion.
Input SHA256:
`69a3440c2c3206af78c73a611946bb817a393cdcff287158d515c1130f5d8728`;
rounded-output SHA256:
`340d7b0bc514884ab6409df032ace615527a43230c246fb2f6cf695af6c377a2`.
The native unit now requires exact observed-versus-ordinary output agreement
in addition to the unchanged same-input attention and full-tower gates.
The independent review also checked the final integer helper's sign/tie
behavior and all score/probability call sites, finding no blocking source or
synchronization issue. Native replay remains the arithmetic qualification.

The integer-barrier replay at integrated source `202269cd`, seal SHA256
`b0e0f0de9444fa56d669eb3760999844826c867ed796aafd51a92793e095536e`,
passed both same-input attention gates. Both cases had exact observed versus
ordinary output agreement. Case 0 QK/scaled-score/probability relative RMS
was 0.000024407 / 0.000025886 / 0.000145343; case 1 was exact at all three
stages. The largest operator relative RMS was 0.000179531 / 0.000087729.
This qualifies the attention correction on these controls, while the full
tower remained unqualified: case 0 at 0/1/27 blocks measured 0.00065759 /
0.00245160 / 0.02625438 relative RMS, and case 1 measured 0.000108899 /
0.00153173 / 0.01731256. Only case 0 at full depth exceeds the frozen gate.
The receipt is retained separately from the earlier failed casts.

Test-only diagnostic `890555dc` adds 17 first-block sequence observations,
including both residual adds and all merger stages, against existing v6
source hooks. Each stage runs the existing production kernel and scratch
layout. A hard bit-equality assertion compares its final merged output with
the ordinary one-pass depth-one graph, checking that readback segmentation
does not change execution. No additional arithmetic policy changed for this
diagnostic.

The first-block sequence replay at integrated source `ec8db81c`, seal SHA256
`d4a8fe4cea44b59dbd0413b65e69e4558d27e6a36dda98cae5422f84159680dc`,
had zero observed-versus-ordinary mismatches for both cases. Case 0 relative
RMS grew from 0.000015031 at patch output to 0.000065840 at norm1,
0.000847711 at attention output, 0.001058058 at norm2, 0.001858651 at
MLP GELU output, 0.002164119 at block output, and 0.002451595 at merged
output. Case 1 first exceeded 0.001 at merger FC1 (0.001250871), ending at
0.001531733. The source-gate failure and all 17 stage aggregates remain
preserved; the observations show accumulated arithmetic differences.

The bounded v7 source capture adds full-depth norm/residual inputs and
outputs without changing source arithmetic. Its manifest SHA256 is
`3824730fa423d8232c48d8c163263ef607fe07fef690c9c5fc6b49a76b3fe436`;
it is 124,272,525 bytes and peaked at 1,312,063,488 bytes CPU RSS. All new
v4-v7 fixture files together remain below 256 MiB. Test-only `a20b5618`
compares 54 full-depth norms per case on exact source inputs.

Test-only `78ff5804` adds the main engine's four-stream patch and 32-stream
dense F32 contraction schedules, followed by the same single bias/BF16
epilogue, as a witness on the actual source input of each linear operator.
The body is source-matched to Torch commit
`5c4886908584029761b579af026dcfb627c84070`, which exactly matches the
producer's `torch_git`. Source lineage is PyTorch `BlasKernel.cpp`,
`ReducedPrecisionFloatGemvFastPathKernel.cpp`, and `functional_base.h`; BSD/MIT
notices remain in `scripts/licenses/distribution`. The witness is compiled
only in tests, with existing M <= 256 and N/K <= 4608 bounds. It records
source bit mismatches for this schedule and the production NAX baseline.
No contraction schedule has been promoted on the basis of source lineage.

Native candidate qualification is pending the root-owned build/replay at the
time of this source checkpoint. Repository policy checks and `git diff
--check` pass. Rust sources were formatted with nightly-2026-10-01. Native
builds and model-bound GPU work are centralized in the integration checkout.

Image requests already clear and bypass generation prefix reuse; this
component change retains that state policy. Image requests have no MTP path.
No full-model answer-quality or throughput claim follows from tower parity.

Private v1-v7 references and all prior failed receipts remain preserved.
An initial operator-producer invocation used the wrong existing pixel
manifest and failed its argument identity check before creating output;
the content-free category was `AssertionError`, phase `arguments`.
