# Upstream and import provenance

- Upstream: https://github.com/trymirai/uzu
- Version: 0.6.1
- Base commit: `8d01581474760bd1fc7f4a547ff1d02f7dada437`
- License: MIT, original Mirai Tech Inc. copyright retained in the root LICENSE.
- Import commit: `40674ab0ef5bd3e664c41da59d3feb9a6df90960`, nine accepted scoring and optimization patches applied in their recorded order.

The import contains the scoring API, topology and state changes, Rust/Metal optimizations, RoPE reuse, direct delta-convolution, and attention preparation optimizations. The standalone scoring crate additionally contains the default-off consecutive distribution cache and full-vocabulary digest probe. Renaming optimization features preserves their implementation and arithmetic.

The pinned upstream ancestry, root license, original upstream README, dependency lock, and JevK5 fixture licenses are preserved. Earlier fork commits were rewritten to keep only standalone engine source and neutral optimization names. Source-bound receipts from the previous history are replaced by fresh build and correctness validation records for the rewritten source.

## Edge0 Phase 0 intake

The format and potential reference-source intake pins
[`Edge0-AI/Edge0`](https://github.com/Edge0-AI/Edge0/tree/b7900cbb322926ccebdcd0c0c5a1280497ecff22)
at `b7900cbb322926ccebdcd0c0c5a1280497ecff22`. Its root framework code is
Apache-2.0 with Copyright 2025 edge0 contributors; the root `NOTICE` records
separately licensed components. The bundled `edge0-strata/` fork is MIT,
Copyright 2026 Niko1221 and the Strata contributors. Its origin record names
fork import `f1933b1e4993fecc10d2e9360d85569619a2c42d`, upstream Strata merge
base `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`, and additional imported
files enumerated by path and SHA-256 in `edge0/IMPORTED_UNCOMMITTED.txt`.
These are publisher-recorded origins; the private fork ancestry was not
independently reconstructed.

The Strata build recipe separately pins llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. The framework's bundled GGML and
tokenizer records instead name llama.cpp
`7ab4ee7baad2d920464cbacfad4f4b07cf111fd2`. These component identities must
remain separate when recording a later port. No Edge0 source, table, kernel
or layout implementation was imported or adapted in Phase 0; no runtime was
built or executed. A later adaptation must preserve the applicable Apache-2.0,
MIT and component notices beside the existing reference licenses, with its
exact source paths, hashes, changes and distribution-notice coverage.

The candidate weights are independently pinned to
[`Edge0/Edge0-145B-q3k`](https://huggingface.co/Edge0/Edge0-145B-q3k/tree/d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74)
at `d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`. Their designated Edge0
contributions carry Edge0 Community Model License 1.0; upstream material
retains Qwen Community License 1.0. Neither is replaced by the source-code
licenses. The owner subsequently selected local non-commercial evaluation only
and authorized continued branch work before merge. No commercial permissions
or product availability are established.
No model payload was downloaded. The
[dated intake record](../docs/archive/runs/2026-11-12-edge0-license-intake.md)
contains immutable primary links, inspected-text hashes, commercial-scope
facts, preservation requirements and unresolved provenance limits. This
record does not establish permission or model qualification.

### Edge0 source preparation

The independent Python admission code in `tools/converter/stark_convert/edge0_source.py`
implements container facts from the pinned Edge0 Strata specifications, without
copying an upstream runtime. `edge0_tokenizer.py` retains the failed comparison
with the pinned original-Qwen tokenizer. The reference code licenses are kept
verbatim in `LICENSE.edge0-reference` (Apache-2.0) and `LICENSE.strata-reference`
(MIT), with the root source notice in `NOTICE.edge0-reference`. The distribution
source inventory and notice collector include all three. Model-weight licenses
remain separate and are retained in the prepared artifact only.

Inspected format authorities at the Edge0 revision above:

| Path | SHA-256 |
| --- | --- |
| `edge0-strata/tools/iq_pack.py` | `6588cba726b74b70ebbe17f8972afecda4cd1e3160f82c15ec15214a8bc55be6` |
| `edge0-strata/tools/mtp_pack.py` | `322502ce2356122c72c5958e0775e4bbbf22ab804a31a8a63c783388e04b3894` |
| `edge0-strata/include/strata/core/latent_dense.hpp` | `31065354f027bfaf388623811ec9efaf0d3bf7b50e65b7e61f2d037a1d591a11` |
| `edge0-strata/src/core/latent_dense.cpp` | `a8cfdbccbb6225bc243c1a28adfa75bf74b012a2da88d67f035a61db0a24fd9d` |

The tokenizer semantic reference is Strata's pinned llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, `src/llama-vocab.cpp`, SHA-256
`82e8996cbbd81c1648b1f4946b5f481138aa897a1343dcd8b1a4abe8cff0b5fc`.
Its existing MIT reference notice is retained in `LICENSE.ggml-hybrid`; this
source revision remains distinct from the older native block-decoder revision
recorded below. A CPU-only tokenizer oracle is an evaluation dependency, not
a production inference process. Its build and any successor tokenizer proof
belong to [attempt `20261112-EDGE0-03`](../docs/model-experiments/2026-11-12-edge0-native-tokenizer.md).

The source-native tokenizer separately derives explicit Unicode classifications
from that llama.cpp revision. `edge0_unicode.json` preserves the full MIT text,
ggml attribution and these source hashes: `src/unicode-data.cpp`
`95170cd1c105a5b41a1b2dce73b0fae8ce8011ef7897600828bb2babe8b26e5d`,
`src/unicode.cpp`
`b1f3a18646197e21f7300c6d3b45c9faf52a123d58a38d7d54f3f225018da45d`,
and `src/unicode.h`
`f1562388d5b9d2dac1152ad8eda56e0bcae25dd407cf15d9dd2e96fe0124f4e7`.
The final explicit-range asset SHA-256 is
`a8d7b7bea4f53942453a535e71b52a91098328eed16dcd466af69ed7a8225820`.
This avoids inheriting the Python regex engine's different Unicode category
tables. The accepted profile is Python preparation only; it establishes no
Rust parser, native generation, stream or model qualification.

### Edge0 reference forward

The diagnostic source adapter, tiled-head/negative-A recurrence and latent MoE
reference follow the same pinned authorities. GGUF transformation facts come
from `conversion/qwen.py`, `conversion/qwen4exp.py` and `src/models/qwen4exp.cpp`
at llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The exact 38 family
metadata fields are recorded in native `edge0_geometry.json` and preparation's
`EXPECTED_GEOMETRY`; runtime equality is required. They do not establish
checkpoint payload integrity independently of the prepared source receipt.

| Conversion/layout authority at the llama.cpp pin | SHA-256 |
| --- | --- |
| `conversion/qwen.py` | `b783f53598e1a65cb0ef7fbe0d0f52da472e2784f63acf13a5aea00f623cfaa6` |
| `conversion/qwen4exp.py` | `12a0a5aea7877fbb8fe35af041a9c34f8b57b05278871b22c24c650b9760dfc3` |
| `src/llama-graph.cpp` | `90f5c6f082790599124630680ba1293d702b369e9ce9088680d210b69905c8d3` |
| `ggml/src/ggml-cpu/ops.cpp` | `3ad6c159f0e9c2e8cef589be2f41f8ca7221a4e465488ad13ae5bff55669158c` |
| `src/llama-memory-hybrid-idx.cpp` | `84a57c465729006c28edbca6bc04d121a2afda1495afb206f956dfc3d17096a1` |

Additional arithmetic authorities at Edge0 revision
`b7900cbb322926ccebdcd0c0c5a1280497ecff22`:

| File | SHA-256 |
| --- | --- |
| `edge0-strata/src/kernels/cuda/router_top10.cu` | `4fa8260d101ba1d5c82e9ee1ee6260ca3f8a7d8020493254682dd0e597c5e94a` |
| `edge0-strata/src/latent/latent_moe.cpp` | `43f929de67fdd421e6d78ec6dacf3d86f4d8b0b3325a5a1bc680bf2a2f4f8239` |
| `edge0-strata/src/kernels/cpu/native_expert.cpp` | `ca4c07774c9f3ea0341bc91bf152f03b885a0a1fbf0f9d7d6d827641515e6b97` |

The independent F64 fixture uses the pinned GGML C quantizers/decoders through
a Stark-authored CPU bridge. New code does not copy the source runtime.
Component, activation-control and actual checkpoint scopes are recorded
separately in [attempt `20261112-EDGE0-05`](../docs/model-experiments/2026-11-12-edge0-reference-forward.md).

### Edge0 Metal graph candidate

The independently authored native graph candidate follows those same pinned
container, quantization, routing, scale and GGUF transformation authorities.
It adds bounded Q8_0/BF16 matrix batching, fused Q3_K gate/up/SwiGLU, IQ4_NL
down products and routed-order U/reduction through Stark's existing Metal
registry. Stored negative-A and tiled V-head conventions remain source-bound.
No upstream runtime was copied, and no new source revision or license replaces
the Apache-2.0, Strata MIT and GGML notices above. The original source bytes,
model licenses and Phase 3 validation remain separate from
[attempt `20261112-EDGE0-06`](../docs/model-experiments/2026-11-12-edge0-metal.md).
That candidate uses the native F32 Qwen Metal graph with mixed stored weights;
it does not establish a Lily-backend port or reduced-precision state validation.

### Edge0 native MTP foundation

The source-specific staged draft reader follows `edge0-strata/tools/mtp_rt.py`
at the same Edge0 revision, SHA-256
`2d128e3a62f8f4810e1837cf4292d0e8473c80c384c2f2e10e2c9bfc4e94a531`.
It retains interleaved gate/up rows and the separate Q2 code/F16-scale planes;
it does not reinterpret the file as contiguous GGML Q2_0 blocks. The exporter
also supplies final gamma values by adding one to F32 norm records. The draft
input, position and cache authority is `edge0-strata/include/strata/core/mtp.hpp`,
SHA-256 `d3d16322e3242a5436ff60cfc1fe630c46706f76b2f1e5151d35aab3245fa2d0`.
Independent foundation work and remaining executable-head gates are recorded in
[attempt `20261113-EDGE0-07`](../docs/model-experiments/2026-11-13-edge0-mtp.md).
Existing Strata MIT/GGML notices remain preserved; no model license or runtime
capability is changed by these checked views.

The private executable F32 head and independent binary64 oracle follow the same
pin and preserve the original draft code/scale planes. Their HC equations are
traced to `edge0-strata/src/kernels/cuda/fused_gr.cu`; dense attention, RoPE,
router and shared-expert authorities are bound individually by Git blob and
SHA-256 in the oracle registration. The candidate uses floating activations
where Strata uses Q8_1 activation blocks. It does not import the CUDA runtime or
inherit its numerical/acceptance results. Source-bound synthetic and actual
component evidence belongs to
[attempt `20261113-EDGE0-08`](../docs/model-experiments/2026-11-13-edge0-mtp-head.md).
The Strata MIT and GGML notices remain separate from the model-weight licenses.

### Edge0 native reader and block components

The independent native Strata readers and the new Q4_0 (GGML 2), IQ4_NL
(20), Q2_0 (42) and FP8 row components are tracked separately in
[attempt `20261112-EDGE0-04`](../docs/model-experiments/2026-11-12-edge0-block-readers.md).
Q4_0/IQ4_NL/Q2_0 scalar layouts and their Metal adaptations follow the exact
llama.cpp revision above: `ggml/src/ggml-quants.c` SHA-256
`07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d`,
`ggml/src/ggml-common.h`
`0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`,
and `ggml/src/ggml-quants.h`
`28ae5fca1f3be636b36cd6c4fa2ca74fd42d229bfbd5352eaf66f3727bb8a6da`.
The existing full GGML MIT notice is byte-identical to that revision's license.

The source-specific FP8 convention follows the pinned Edge0
`edge0-strata/src/kernels/ngram.cpp`, SHA-256
`f782ecbaea1ad7934eb8817fc17e6290788bc991fa03ed74fad6e6833c1cc9b0`.
Its independent CPU qualification compiles an attributed decoder extraction;
the complete checkpoint-driven parity utility was not run. Stored-column-scale
placement follows `edge0-strata/src/kernels/cpu/expert_layout.cpp`
`709d400f3f7eed77182af310cb71dbd44cb186f2144496702312a4bcb8205204`
and active verify/prefill call paths, recorded with their own source hashes in
the attempt. Strata's complete MIT notice remains in `LICENSE.strata-reference`.
These component results do not qualify another artifact or enable a backend.

## Third-party algorithm adaptations

The vision-only BF16 LayerNorm variance implementation adapts the logical
Welford moments cascade and scalar final contraction from PyTorch's
`aten/src/ATen/native/cpu/moments_utils.h`, tag `v2.14.1`, resolved commit
`5c4886908584029761b579af026dcfb627c84070`. The inspected source SHA-256 is
`ddc188f6752d2b7c04f443a476461d507f041ab6a2a141aa459a8cfa10ce3170`. Stark's
implementation is a Rust/Metal adaptation of the logical BF16 NEON schedules
for vision LayerNorm and vision matrix products, including affine sequencing,
32-stream dot accumulation, and four-part FP32 output accumulation; it does not
add a PyTorch runtime dependency or change the text normalization path. The
PyTorch BSD license, the GGML MIT notice embedded in the pinned PyTorch
fast-path source, and the Boost Software License 1.0 notice for the SLEEF
submodule pinned by that PyTorch revision are preserved in
`scripts/licenses/distribution/` and included with packaged distribution
notices. Pinned source paths, hashes, licenses and the adaptation scope are
recorded in the supplemental distribution license manifest. The vision GELU
uses PyTorch's pinned forward formulas and ARM NEON erf polynomial, including
the Arm-derived `exp_u20` routine; its Arm Optimized Routines MIT/Apache notice
and attribution are also preserved. The vision softmax exponential adapts the
pinned SLEEF u10 routine and double-float helpers.

The exact Qwen3.5 vision artifact also selects reference arithmetic for decoder
linears, RMS normalization, SiLU, attention, and DeltaNet. RMS row sums adapt
the pinned PyTorch `SumKernel.cpp` cascade and `ReduceOps.cpp` mean sequence.
The DeltaNet chunk algorithm follows Transformers revision
`2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397`, modeling source SHA-256
`af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c`.
Its cumulative sum uses PyTorch's CPU FP64 accumulator; Metal emulates the
required IEEE additions with integers and rounds each prefix to FP32.
Sequential FP32 fused matrix and triangular-solve schedules are measured
compatibility with Accelerate on the qualified runtime. The source licenses,
Qwen/Hugging Face 2025 attribution, pinned source hashes, and adaptation scope
are included in the distribution notice manifest and copied through its notice
collector. Existing text artifacts retain the Legacy execution profile.

The larger Gemma 4 vision tower's logical-width-72 and hidden-width-1,152 RMS
reductions adapt the same pinned `SumKernel.cpp` cascade, source SHA-256
`9b88aa14e626dd708e50816a5ca89acfc66798226dc82501cf1857caebe96f95`.
The logical 72-channel path adds its two remaining four-channel vectors before
reducing the four independent accumulators. The existing PyTorch BSD notice and
source entry cover this adaptation. E2B/E4B and unified-12B RMS schedules retain
their previous selection. Independent model-free expected values come directly
from Transformers 5.15.1 `Gemma4RMSNorm` on pinned Torch 2.14.1.

The same larger Gemma vision profiles adapt pinned Torch CPU
[`FlashAttentionKernel.cpp`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp)
(`2e77b79c6f340fe01b15e442ea9dd319649087579a31906c08f6afe2c86ec482`)
and its NEON `fexp_u20` approximation for 512-key tiles. QK remains F32;
unnormalized exponential values cross an integer-RNE physical BF16 buffer before
PV, while denominator/output accumulation remains F32. The Rust/Metal adaptation
uses fixed reused planes with linear memory in query count. Source and license
hashes are retained in `pytorch-gemma4-larger-vision-flash`; the existing PyTorch
BSD notice applies. E2B/E4B, unified12B and text attention retain their dispatch.

Larger-vision QK follows `BlasKernel.cpp`'s generic `gemm_transa_` F32-output
route through `cpublas_gemm_no_downcast_impl`, with four independent partials.
Its SHA-256 is `5f55b3a7c17c724e1b5e963f37f248fff1857867faff36d58d3b24da9cdc8363`.
The 32-stream BF16-output `compute_dot` specialization is a different overload;
ordinary BF16-output vision projections keep that schedule. A model-free pinned
Torch Flash logsumexp fixture with one unmasked key exposes the QK F32 result
directly, before probability materialization can hide accumulation differences.

The larger-Flash scalar denominator update explicitly contracts its multiply/add
as witnessed against pinned CPU outputs. The prior PV accumulator is rescaled
separately before addition, retaining the distinct source materialization.

The larger Gemma4 Flash scalar rescale exponential uses an original compensated
range reduction and degree-12 Taylor polynomial, independently checked against
100-digit Decimal exponential and the pinned reference host's scalar libm.
This nearest-F32 numerical policy is distinct from the source libm in rare
one-ULP cases; frozen source attention/features remain the qualification gates.
CPU libm and the source-derived BF16 tile fexp implementation are unchanged.

The subsequent larger-only scalar source-compatibility policy uses a generic
128-entry power table and quadratic residual, independently derived with
`a=ln(2)/128`, `c0=a²/2`, and `c1=a+a³/32` (the leading Chebyshev cubic
correction). Constants are generated from 100-digit Decimal mathematics; no host
instructions, tuned coefficients or table bytes are redistributed. ARM optimized
routines at `20859d3fd9e797dde592c489c8ef7d383f778620`, `math/expf.c`
(`31cb4e66fa51bd28d95025e3a702e9b3c02cb91e8cbe31da73cf8eaf8d71aae7`)
and `math/exp2f_data.c`
(`d58b44276cb8330ffb896c5adff2fc5d85d4b8a8bb809fbd51267a04347602f1`),
are a licensed table/range-reduction reference, distinct from this independently
derived 128/quadratic formula. The selected MIT notice is retained in
`arm-optimized-routines-LICENSE` (the complete upstream dual-license notice).
The original nearest-F32 helper/test remains a separate control. Source-bit
agreement is qualified only for the recorded finite controls and model gates;
a one-ULP broad-control counterexample remains preserved.

The explicit BF16 affine-Q4 source profile adapts MLX v0.32.3 at
[`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`](https://github.com/ml-explore/mlx/tree/64ea011cb65f14d9ce2737e60db9a4ae91ed7441).
Its native QMV arithmetic preserves BF16 four-value bias sums, four-product
F32 subtotals, affine contribution order and the source's eight- or
sixteen-value lane geometry. The source-wide QMV, QMM dispatch and split-K
reduction adaptations retain the original projection widths and typed partial
sum boundaries. Source BF16 D64/D128 MXU attention applies the F32 scale
to accumulated QK scores and uses the source prefill threshold above eight
queries, while preserving stored BF16 queries. Its NAX mass reduction pairs
four lane values, reduces lanes with XOR 1 and 8, and adds each 16-column
fragment directly into the rescaled running mass. Existing profiles retain
their declared arithmetic policies. The source finish preserves the F32
reciprocal and product in Safe math through the final BF16 materialization.

The explicit RMS source policy follows contiguous four-value lane reads,
F32 SIMD reduction and the source reciprocal square root. It materializes
normalized BF16 values before the scale multiplication. The BF16 fallback
attention source policy uses a global row maximum followed by the source F32
exponential reduction and BF16 probabilities before the value product. The
source softmax precision option specifies F32 accumulation; its exponential
remains the explicit source `fast::exp` operation.

The QKV host routing in `encodable_block/mixer/attention/qkv_norm.rs` also
reuses this existing normalization adaptation for contiguous M=1 Q/K rows:
OLMo's 4,096-element full-width vectors and Gemma's 256/512-element head rows,
batched as eight query heads or one key head. The padded SIMD reduction,
configured epsilon, precise reciprocal square root and BF16 normalization/gain
materialization retain the same source expression. The F32-stored gain vector
is shared across the head rows and cast to BF16 before multiplication; the
stored Gemma gains have zero scale offset.

The route is restricted to the explicit BF16 source reduction policy with
`OnlyNormalization`, learned scales, no bias or mean subtraction, and zero
scale offset. Gemma's per-head branch additionally bounds query/KV geometry to
eight query heads and at most one KV head. Weightless V, multirow and other
profiles retain their existing validation and kernels. This host selection
uses the normalization source pin above and adds no upstream kernel source.

Source RMS normalization extends this existing adaptation by retaining each
thread's four rounded values in registers through normalization, within the
validated 4,096-element row limit. This reuse applies to OLMo and Gemma.
Gemma's source PLE residual fusion additionally materializes the residual sum
in BF16 before statistics. A separate native
`tensor_add_scale/materialized_bf16_add_scale` kernel fuses the final PLE
residual addition and layer scale while retaining both BF16 boundaries.
These host selections require BF16, reference PLE and the explicit source RMS
policy; canonical and legacy PLE profiles retain their separate operations.

The scheduling investigation inspected local Qwen optimization revision
`908ae21f4894d841cf2e1ad44451643439543443`, including the thread-local rounded
value in `vendor/lily-core/src/kernels/metal/gdn.metal`. The Gemma changes
independently apply that scheduling pattern. No Lily kernel arithmetic was
copied. The existing MLX source revisions, attribution and MIT notice above
remain the arithmetic provenance for the source RMS path.

Source BF16 vector attention uses a separate Safe, contraction-enabled kernel
in `crates/stark-engine/src/backends/metal/kernel/attention/source_single_pass.metal`,
selected by `single_pass.rs`. This follows the pinned `sdpa_vector.h` arithmetic
and its precompiled `-fno-fast-math` build policy while retaining the inherited
vector geometry and masking. Its scope is a single flat causal query, fewer
than 1,024 keys, head dimension 64, 128 or 256, and no sinks. The existing
source-profile markers select this path; other vector configurations retain
their existing kernels. F32 score, probability and output accumulation precede
the final BF16 materialization.

Source sliding-window M=1 attention also preserves the physical traversal of
MLX-LM's `RotatingKVCache` at revision
`1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7`, path `mlx_lm/models/cache.py`,
SHA-256 `c96cb169fd5e0ec4584a162dd61dc62ab08ccdca03a9850f8a491ce6809145e6`.
Native cache storage still appends a pending suffix before acceptance. A scalar
origin records the source order after bulk acceptance; the source vector kernel
maps that order onto existing storage and substitutes the pending overwrite.
This preserves the source F32 online reduction without copying the KV cache.

The materialized source-attention QK product also preserves the source dense
split-K precision when its host predicate selects two partitions. For the
qualified BF16 D512 geometry with eight query heads and one KV head, the
MXU path retains two 256-element K partitions and their reduction in F32,
then casts each score once to BF16. This applies to source prefill rows 9
through 1,023 after excluding the source wide-vector case and checking the
source split predicate. The adaptation is in `attention/fallback.rs` and
`matmul/mod.rs`; it does not change quantized QMM or other dense products.

The cached D512 QK route separately adapts `gemv.h`'s `GemvWide` schedule in
`attention/source_cached_qk.metal`. On the qualified MXU-capable BF16 source
profile, one query with eight heads collapses to M=8, with one KV head, no ring
or sliding window, and 2 through 1,023 keys. The source uses four query vectors
per group and 32 reduction lanes for at most 64 keys, otherwise 16 lanes.
The native adaptation preserves staged float4 dots, local accumulation and
halving shuffles before its BF16 score store. Other fallback products retain
their existing route.

These are native Rust/Metal paths; MLX remains a private validation dependency.
Precision changes are qualified against matching source-bound references,
with failed and successful runs recorded separately. Operand agreement does
not establish model-level qualification; the latter is recorded separately.

The explicit source RoPE coefficient path is enabled only for the existing BF16
MLX affine-Q4 profile (`materialize_affine_input_sum`) and currently implements
Unscaled, Proportional, and YaRN configurations. It computes coefficients from
absolute scalar positions in the attention preparation kernels; the legacy
precalculated CPU tables remain available for other profiles and configurations.
Plain RoPE uses the source F32 host `log2(base)` coefficient. Custom
Proportional and YaRN frequencies request `metal::precise::pow`, and rotation
uses the source F32 operation order with explicit fast sine and cosine. YaRN
first multiplies the BF16 input by its weak BF16 magnitude scalar and stores
BF16 before rotation. The frequency and prepared-operand controls are source
bound; model-level qualification is recorded separately.
The native adaptation is in `crates/stark-engine/src/encodable_block/mixer/attention/rope.rs`
and `crates/stark-engine/src/backends/metal/kernel/attention/attention_prepare.metal`.
It follows the MLX v0.32.3 RoPE implementation at the revision above and
MLX-LM `models/rope_utils.py` at
[`1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7`](https://github.com/ml-explore/mlx-lm/tree/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7),
SHA-256 `ccba2f0c7b246c008a1188dc3760c162bdd906b15a1c490461dee4be64bc62df`.
The Apple copyright and MIT notice are preserved in `LICENSE.mlx-reference`.

MLX's precompiled array-power kernel calls generic `metal::pow` under
`-fno-fast-math`. The Metal standard library selects that default during
preprocessing, so a function-local Safe pragma in a native build does not
change the earlier function selection. The retained compiler check records
preprocessor SHA-256 `a10bbc2d990a5fc03bc59666de845e5ce4844e529a56e60fa1b044aa9e7ce535`
for the native Metal 4 default and
`d2892c0a851151d93e2e54b9f55ae7da343561ac03fc81c9e6a1d26ca562a43a`
with `-fno-fast-math`, with different fast-math macro states. The source path
therefore requests precise power explicitly. These hashes identify the checked
compiler outputs; source-bound operator and model validation remain separate.

The inspected MLX source files have the following SHA-256 hashes:

| Source path | SHA-256 |
| --- | --- |
| `mlx/backend/metal/kernels/quantized.h` | `8e2ca65d8d8dafe67b97120d1127d54ce57032f621bdba161e54a0bc4a948b8c` |
| `mlx/backend/metal/kernels/gemv.h` | `296ef622ea3c67d7485aa7cdf102bbfcb582e07bd3207793179a9d1e65ee5a65` |
| `mlx/backend/metal/kernels/gemv.metal` | `0bd8bde0c867a17c345a3651f9f0a6c2909e0c74e76ea2a08f373fe4dcafaeda` |
| `mlx/backend/metal/quantized.cpp` | `36b79607ee01bc213511dcdad590b2dc3b96099300551cbc3e7e6c7e13ba5982` |
| `mlx/backend/metal/kernels/bf16_math.h` | `1f374f8380f756eb89acf6a847741cb8fecbe642945e159fb6208d804cc06496` |
| `mlx/backend/metal/kernels/reduction/reduce_col.h` | `800868c1eafbf2b85980731a6303942d2d1061023f6d465f2b996642126cdc62` |
| `mlx/backend/metal/reduce.cpp` | `401674953ae6ea7cfb94cd1d675d4a7321af58143373b6343ea2442516f42f43` |
| `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h` | `e576cbbe13a8fe5fda7b630422f6083faac6114fbf2aad3970323932b1e6cc44` |
| `mlx/backend/metal/kernels/steel/attn/nax.h` | `fb10bcae44095e9fed5c81213d03527dd5fe5ac11be11ec7784bd357c6f317ec` |
| `mlx/backend/metal/kernels/sdpa_vector.h` | `f52c8da21f2c31f3f77168d326a49815fc29bde4b573883896a8eb3298f11508` |
| `mlx/backend/metal/kernels/scaled_dot_product_attention.metal` | `16cc1ff6c99ac606fffec8e35efb3a40101de37aab82db4f2186ae8ac201cc5c` |
| `mlx/backend/metal/scaled_dot_product_attention.cpp` | `4eaca28cbe1d192a3137a970b2f4d7cdecaba67a1f448b0b0b5f0309b6b3961e` |
| `mlx/backend/metal/matmul.cpp` | `21e0172e3b5af6d9d07d7b36d543fefcc5ba3edce2aa67a84be432fe8221e27a` |
| `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk_nax.h` | `9486b177261f657bceae6437a18d412a0e361c3e6b418c78b4c067b463c84597` |
| `mlx/backend/metal/kernels/rope.metal` | `2b35221ebea033ff43100b85eaaddec5c0e34e3cfd52e7c692573764b2b7d0ab` |
| `mlx/backend/metal/rope.cpp` | `9ec2d2f298381af6445b1e2281dd90521988861ed170a602944ec0f1fca17ecc` |
| `mlx/fast_primitives.h` | `8ce8bd4933452246e81702f16798d9d816f162e9f75e905e562acedc7c95f112` |
| `mlx/backend/metal/kernels/softmax.h` | `cb233fe42a7358f5f658c9a1ef52cd2ee9dd3786ddfad01793a976d58f4d4051` |
| `mlx/backend/metal/kernels/softmax.metal` | `d19231c66973edc3944f12529d1cc393029e7f7262b914907c710ef9dbcb39e2` |
| `mlx/backend/metal/softmax.cpp` | `e0f8028a3a41659ff85cf6d2b9f318dff50f85ab61b3f0260bb535f3980deffa` |
| `mlx/backend/metal/kernels/rms_norm.metal` | `b2e04e377fdad1d645581f9beeaf9cbb06d1ad32926161e06cbc15240caf12bf` |
| `mlx/backend/metal/normalization.cpp` | `01bc537389638efe43a9bedeeb2008484864772022ac56e106a3f9911bcaa1aa` |
| `mlx/backend/metal/kernels/defines.h` | `a6c2c824e855078cd092fd9592457616f995755f0e865475c3f006a3f5d6bf1d` |
| `mlx/backend/metal/kernels/binary_ops.h` | `6dbe789faafbf7259f411c7fab7c0fcdd192df47a78f8f0bd408228f7a30e5c7` |
| `mlx/backend/metal/kernels/CMakeLists.txt` | `984c577264fc62a280bad4c41c4d06982a252cb08bea1cdcede32c4c9cc1493e` |
| `mlx/backend/common/metal_kernel.h` | `b64ca4cebcba530d8ec1e0a0f23b57313507a37d05e3615ded3796b563cc83fc` |

The Apple copyright and complete MIT notice are preserved in
[`LICENSE.mlx-reference`](../LICENSE.mlx-reference), copied into standalone
distribution notices and refreshed menu bar app overlays. The distribution
source inventory includes this notice, and packaged image qualification
requires its delivered hash.

Model weights, private fixtures, runtime executables, and host application archives are excluded. The original pinned upstream README is stored verbatim as [UPSTREAM_README.md](UPSTREAM_README.md); its relative links refer to the upstream tree and are not expected to resolve here. The current build and validation receipts identify the complete standalone source and executable hashes.

## Naming and compatibility

Gemma 4 text layout and proportional RoPE conventions were independently
implemented against Google's E2B checkpoint revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7` and MLX-LM revision
`1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7` (MIT, Copyright 2025 Apple Inc.).
The production activation contract follows canonical Transformers 5.15.1
`models/gemma4/modeling_gemma4.py`, SHA-256
`4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3`
(Apache-2.0, Copyright 2026 the HuggingFace Team). Its license is preserved in
`scripts/licenses/distribution/transformers-LICENSE`. Canonical HF BF16/FP32 and
MLX BF16 references are separate source-bound numerical profiles; Python is used
only by their private validators and never by native inference. The canonical
validator uses the unchanged `tools/uv.lock`, while the MLX validation environment
and its independent lock remain ignored under `runtime/`.

Recursive Flint's native Rust forward contract follows the Apache-2.0
`recursive-oss/rec` model module at revision
`0e3150ca4262f572dee25c231d9ac7148414d2af`, path
`engine/src/rec_mlx/model.py`, SHA-256
`56363f7a1caea2b9db317feb192f4bfea32eccb2b6b190dffc3fa93ee09695b1`.
The reference is published by Recursive Superintelligence, Inc. Its complete license is
preserved in `LICENSE.flint-reference` and copied into distribution notices.
The official MLX checkpoint is pinned separately at
`RecursiveSI/flint-int4` revision `3b93f9053b530488405ae9fca51d4900d6ce21f0`.
Byte-view packed matrices retain the original little-endian MLX storage.
The optional source-bound validator imports only the hash-verified reference
model module, with a local cancellation exception standing in for its relative
dependency. Native inference uses no Python model code. See
[Flint support and limits](../docs/FLINT.md) for the execution precision and
private validation procedure.

The native hybrid text path separately implements GLM5-Next and Qwen4-Exp.
GGUF quantization block layouts, IQ codebooks, transformed GLM tensors and
absorbed MLA follow `ggml-org/llama.cpp` revision
`8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`; its complete MIT license is retained
in `LICENSE.ggml-hybrid`. The Rust and Metal implementations preserve that
attribution and do not link to a llama.cpp runtime.

The Qwen4-Exp algorithm adaptation follows PipeNetwork's Apache-2.0 implementation,
repository revision `0352280b927a731fd35276ac622ef2f28da5c936`, and its model
module in `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit` at
`b2c422f3c643e36f04227a64d61796b44a4b1029`. The inspected Python source SHA-256 is
`6fae4ec0decbf77ca4a4571de683bc5580ec75e84325ecb432dfcd2fc81df75e`.
`LICENSE.qwen38-reference` preserves the full license; the engine package declares
the combined MIT and Apache-2.0 licensing, and the distribution notice collector
copies both added notices. These are source adaptations with separate validation
records. They do not change the preserved Uzu source-bound evidence.

The GLM reference validator uses Transformers revision
`95bafdcccfffe17ec383999fc39273867b11385e`, modeling source SHA-256
`7870e7ce6ca477d26de3ec79b31d41553b75a09a726c661f04c2e26e790bdf36`.
The Qwen validator runs the pinned PipeNetwork model module with the repository's
locked MLX comparison environment. These Python runtimes are validation tools;
native loading and execution do not import them. Model weights and synthetic
reference captures remain ignored. The supported checkpoint identities and
precision policy are recorded in [native hybrid models](../docs/NATIVE_HYBRID_MODELS.md).

The dense Qwen3.5-family MTP graph follows `jundot/omlx` revision
`2165e75726ba83d9651dbaced61c97728da77bba`, source
`omlx/patches/mlx_lm_mtp/qwen35_model.py`, SHA-256
`d5343f025f00538dfc4cecd158946282221236aca7ca118b4d78ca294aab385c`.
Stark implements its embedding/target-residual normalization, fusion and
one-layer predictor as a typed Rust/Metal graph; native inference imports no
Python module. Converter tensor naming and the MTP graph are independent
adaptations. The oMLX contributors' Apache-2.0 license is retained in
`LICENSE.omlx-reference` and included in distribution notices. The dense
target uses the existing Qwen3.5 implementation. Its sigmoid attention-output
gate was checked against Transformers v5.8.0 `modeling_qwen3_5.py`, SHA-256
`641da34484f6b9769e28615c680cc4e3202afbf5bdbf1da7d56d5a3d678918c7`.
The same pinned oMLX revision's `prompt_priming.py` and `batch_generator.py`
select its post-normalization draft variant: the target's final normalization
precedes the predictor's hidden-input normalization, and later draft steps
consume the predictor's normalized output. This is a proposal-policy choice;
exact rejection sampling still verifies against the unchanged target route.
The coarse-head-then-exact-shortlist-rescore proposal flow is informed by the
same pinned driver's documented strategy. Stark's optional three/four-bit
coarse matrix is derived with its own integer midpoint and BF16 coefficient
construction in Rust; no external quantization implementation is imported.

The converter's lossless five/six-bit affine unpacking follows the documented
MLX bit layout, inspected in `mlx/backend/metal/kernels/quantized.h` at
`ml-explore/mlx` revision `ba8c005c1b687227ff79a55da5b8aa062d92541f`
(MIT, Apple Inc.). It is an independent scalar implementation with integer
and coefficient-preservation tests; no MLX kernel code is copied into that
converter. The existing MLX MIT notice remains in `vendor/lily-core/LICENSE.mlx`.
The typed `MLXRowPartitionedSpec` metadata and row-copy path are Stark
implementations: segments describe contiguous fused-output rows at source
widths, while row arithmetic retains the original fused output dimension and
W8 planning. Eligible BF16/group-64/output-input matrices keep packed 5/6-bit
bytes; other layouts use lossless widening. The row-reuse GEMV route extends
inherited Stark code. Artifact-specific records are the [original dense 27B
study](../docs/benchmarks/2026-11-06-qwen38-27b-oq4e-mtp.md) and the
[packed-weight follow-up](../docs/benchmarks/2026-11-06-qwen38-27b-packed-followup.md).

The Rust workspace contains only the `stark-engine`, `stark-engine-macros`, `stark-score` and `stark-generate` crates. These upstream identifiers are kept on purpose:

- Uzu Git history, upstream license, revision metadata and original patch hashes.
- The historical `UZU_CAPTURE_` environment names, which both workers clear together with `STARK_CAPTURE_` names.

The menu bar app is built with `swiftc` directly. Inherited self-hosted CI, benchmark deployments, package publishing and cloud artifact destinations are disabled.

## Gemma 3 image execution

The Gemma 3 image adapter, native SigLIP tower and projector independently
implement the original Google/Hugging Face checkpoint layout and equations in
Transformers v4.53.3 `models/gemma3` and `models/siglip` (Apache-2.0; Google and
Hugging Face copyright notices are retained in the source references). The
private canonical replay imports unchanged Transformers 5.15.1 with Torch
2.14.1 and records hashes of both model implementation modules. Model weights
remain subject to Google's Gemma terms and are never committed.

The fixed-resolution bilinear resizer follows the pinned Torchvision CPU
processor's half-pixel antialias convention and adaptive signed-int16
coefficient quantization from PyTorch `UpSampleKernel.cpp` at
`5c4886908584029761b579af026dcfb627c84070`, source SHA-256
`3130e86b56234a8da4c6347d973721d49845d1c7cdbdad2e27dd11f4f869d708`.
The existing PyTorch BSD notice and supplemental distribution manifest retain
its license and adaptation scope. Fixed Pillow 22-bit coefficients are not
numerically identical for nonuniform images. This policy is Gemma3-specific. The
spatial average-pooling CPU/Metal kernels, processor admission, opaque image
ownership, staged transport and artifact contracts are Stark implementations.
Existing source-bound BF16 normalization, activation and matrix primitives keep
their existing PyTorch/SLEEF notices and attribution. See
[Gemma 3 image artifacts](../docs/GEMMA3_IMAGES.md) for exact profile boundaries.

## Idle queue scheduling

The converted text generation worker's private Metal4 shared-event maintenance
and the scoring worker's timed owner-thread input wait are original Stark
adapter changes. An idle tick signals the existing command queue and waits
with a bounded timeout; it adds no model dispatch, shader adaptation or
numerical policy. Native mixed-precision Qwen reuses Lily's existing empty
queue signal under both admitted numerical policies. The
[cross-backend study](../docs/benchmarks/2026-11-08-cross-backend-idle-wake.md)
binds the source patch, worker hashes and artifact-specific controls.

## Removed inherited code

On 2026-10-25 the inherited Uzu code that no product component used was removed, together with the dependencies only it needed:

- The seventeen crates under `crates/legacy` (the `stark` engine facade and its Python, Swift and TypeScript bindings, the HTTP server, `cli`, `cli-tools`, the model registry and download manager, and their support crates), including their example and binding lockfiles.
- `vendor/napi-rs`, the patched napi crates used only by the TypeScript binding, and the matching `[patch.crates-io]` entries.
- The root `Package.swift`, which described the Swift binding, and `apps/playground`, the sample application built on it.
- `.config/platforms.toml`, read only by `cli-tools`.

`stark-engine` used four plain types from the inherited `shoji` crate; they are now defined in the engine with the same fields. The engine unit suite used the inherited engine to download its test model and the inherited `keisoku` crate for a benchmark cooldown; it now reads a converted model from `STARK_TEST_MODEL_PATH` and has no cooldown. `Cargo.lock` went from 813 to 263 packages with no version changes. The removed files remain in Git history at the commit before the removal, and the originals are at the upstream base commit above. `provenance/BUILD_RECEIPT.json` still lists hashes of the removed files because it records the source of the build it describes.
