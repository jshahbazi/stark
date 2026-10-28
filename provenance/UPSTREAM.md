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

Model weights, private fixtures, runtime executables, and host application archives are excluded. The original pinned upstream README is stored verbatim as [UPSTREAM_README.md](UPSTREAM_README.md); its relative links refer to the upstream tree and are not expected to resolve here. The current build and validation receipts identify the complete standalone source and executable hashes.

## Naming and compatibility

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
