# Upstream and import provenance

- Upstream: https://github.com/trymirai/uzu
- Version: 0.6.1
- Base commit: `8d01581474760bd1fc7f4a547ff1d02f7dada437`
- License: MIT, original Mirai Tech Inc. copyright retained in the root LICENSE.
- Import commit: `40674ab0ef5bd3e664c41da59d3feb9a6df90960`, nine accepted scoring and optimization patches applied in their recorded order.

The import contains the scoring API, topology and state changes, Rust/Metal optimizations, RoPE reuse, direct delta-convolution, and attention preparation optimizations. The standalone scoring crate additionally contains the default-off consecutive distribution cache and full-vocabulary digest probe. Renaming optimization features preserves their implementation and arithmetic.

The pinned upstream ancestry, root license, original upstream README, dependency lock, and JevK5 fixture licenses are preserved. Earlier fork commits were rewritten to keep only standalone engine source and neutral optimization names. Source-bound receipts from the previous history are replaced by fresh build and correctness validation records for the rewritten source.

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

Source BF16 vector attention uses a separate Safe, contraction-enabled kernel
in `crates/stark-engine/src/backends/metal/kernel/attention/source_single_pass.metal`,
selected by `single_pass.rs`. This follows the pinned `sdpa_vector.h` arithmetic
and its precompiled `-fno-fast-math` build policy while retaining the inherited
vector geometry and masking. Its scope is a single flat causal query, fewer
than 1,024 keys, head dimension 64, 128 or 256, and no sinks. The existing
source-profile markers select this path; other vector configurations retain
their existing kernels. F32 score, probability and output accumulation precede
the final BF16 materialization.

The materialized source-attention QK product also preserves the source dense
split-K precision when its host predicate selects two partitions. For the
qualified BF16 D512 geometry with eight query heads and one KV head, the
MXU path retains two 256-element K partitions and their reduction in F32,
then casts each score once to BF16. This applies to source prefill rows 9
through 1,023 after excluding the source wide-vector case and checking the
source split predicate. The adaptation is in `attention/fallback.rs` and
`matmul/mod.rs`; it does not change quantized QMM or other dense products.

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

The Rust workspace contains only the `stark-engine`, `stark-engine-macros`, `stark-score` and `stark-generate` crates. These upstream identifiers are kept on purpose:

- Uzu Git history, upstream license, revision metadata and original patch hashes.
- The historical `UZU_CAPTURE_` environment names, which both workers clear together with `STARK_CAPTURE_` names.

The menu bar app is built with `swiftc` directly. Inherited self-hosted CI, benchmark deployments, package publishing and cloud artifact destinations are disabled.

## Removed inherited code

On 2026-10-25 the inherited Uzu code that no product component used was removed, together with the dependencies only it needed:

- The seventeen crates under `crates/legacy` (the `stark` engine facade and its Python, Swift and TypeScript bindings, the HTTP server, `cli`, `cli-tools`, the model registry and download manager, and their support crates), including their example and binding lockfiles.
- `vendor/napi-rs`, the patched napi crates used only by the TypeScript binding, and the matching `[patch.crates-io]` entries.
- The root `Package.swift`, which described the Swift binding, and `apps/playground`, the sample application built on it.
- `.config/platforms.toml`, read only by `cli-tools`.

`stark-engine` used four plain types from the inherited `shoji` crate; they are now defined in the engine with the same fields. The engine unit suite used the inherited engine to download its test model and the inherited `keisoku` crate for a benchmark cooldown; it now reads a converted model from `STARK_TEST_MODEL_PATH` and has no cooldown. `Cargo.lock` went from 813 to 263 packages with no version changes. The removed files remain in Git history at the commit before the removal, and the originals are at the upstream base commit above. `provenance/BUILD_RECEIPT.json` still lists hashes of the removed files because it records the source of the build it describes.
